#!/usr/bin/env python
"""End-to-end analysis driver: pull banks, analyze every cell, assemble results.

Stages
------
pull       download banks + embeddings + question files from the HF dataset
cell       full per-cell analysis (D0, E1-E4 real, spectra, signals, R outcomes)
synthetic  E1 synthetic sweeps (bank-independent)
assemble   cross-cell statistics: E5, R1-R5 winner map, R6/R7, tables
all        pull + cells (parallel) + synthetic + assemble

Every output lands under ``cache/analysis`` and ``cache/figure_data`` so that
figures and tables regenerate from cache only (blueprint Part B rule).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Keep BLAS single-threaded inside worker processes; cells parallelize outside.
for _var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

CACHE = ROOT / "cache"
ANALYSIS = CACHE / "analysis"
FIGURE_DATA = CACHE / "figure_data"
TABLES = CACHE / "tables"

CELLS: List[Tuple[str, str]] = [
    ("qwen2.5-0.5b", "gsm8k"),
    ("qwen2.5-1.5b", "gsm8k"),
    ("llama-3.2-3b", "gsm8k"),
    ("qwen2.5-0.5b", "math"),
    ("qwen2.5-1.5b", "math"),
    ("llama-3.2-3b", "math"),
]
HEADLINE_MODEL = "llama-3.2-3b"
CHEAP_MODEL = "qwen2.5-0.5b"
RULES = ("majority_vote", "pass_at_k", "verifier_best")
HEADLINE_BUDGET = 8
# The anisotropy-corrected embedding kernel; see winner.PRIMARY_EMBEDDING_KERNEL.
EMB = "embedding_qc"


def _write(path: Path, payload: Any) -> None:
    from diversity_reasoning.io import write_json_atomic

    write_json_atomic(path, payload)
    print(f"[write] {path.relative_to(ROOT)}")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _intkeys(cell: Dict[str, Any]) -> Dict[int, Any]:
    return {int(k): v for k, v in cell.items()}


def _normalize_outcomes(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Restore integer budget keys lost in JSON round-trips."""
    for record in records:
        cells = record.get("cells", {})
        for key, value in list(cells.items()):
            if key == "random":
                cells[key] = [_intkeys(seed_cell) for seed_cell in value]
            else:
                cells[key] = _intkeys(value)
    return records


# --------------------------------------------------------------------------
# Stage: pull
# --------------------------------------------------------------------------


def stage_pull(cells: Sequence[Tuple[str, str]]) -> None:
    from diversity_reasoning.hf_bank import pull_bank, pull_embeddings, pull_questions

    for dataset in sorted({dataset for _, dataset in cells}):
        pull_questions(CACHE, dataset)
    for model, dataset in cells:
        try:
            print(f"[pull] {model}/{dataset}:", pull_bank(CACHE, model, dataset))
            print(
                f"[pull] emb {model}/{dataset}:",
                pull_embeddings(CACHE, "bge-large-en-v1.5", model, dataset),
            )
        except FileNotFoundError as error:
            print(f"[pull] {model}/{dataset} not available yet: {error}")


# --------------------------------------------------------------------------
# Stage: cell
# --------------------------------------------------------------------------


def stage_cell(model: str, dataset: str) -> Dict[str, Any]:
    from diversity_reasoning.measurement import (
        anisotropy_diagnostics,
        budget_spectra,
        e1_real_deletion,
        e2_duplication,
        e3_sample_size,
        e4_dimensionality,
    )
    from diversity_reasoning.pools import MathCanonicalizer, load_cell
    from diversity_reasoning.signals import question_signals
    from diversity_reasoning.strata import snell_level_correlation, stratify_cell
    from diversity_reasoning.winner import ALPHA_GRID, CORE_OBJECTIVES, question_outcomes

    started = time.time()
    out = ANALYSIS / model / dataset
    out.mkdir(parents=True, exist_ok=True)
    # Constructed here (not implicitly) so its health is recorded per cell: the
    # oracle decides MATH answer classes and correctness, so silent degradation
    # would corrupt everything downstream.
    canonicalizer = MathCanonicalizer() if dataset == "math" else None
    pools = load_cell(CACHE, model, dataset, canonicalizer=canonicalizer)
    print(f"[cell {model}/{dataset}] {len(pools)} pools loaded in {time.time() - started:.0f}s")
    if canonicalizer is not None:
        stats = canonicalizer.stats()
        _write(out / "oracle_stats.json", stats)
        if stats["failures"] or stats["timeouts"]:
            print(f"[cell {model}/{dataset}] oracle degradation: {stats}")

    strata = stratify_cell(pools)
    strata["snell_vs_level"] = snell_level_correlation(strata)
    _write(out / "strata.json", strata)

    spectra_rows: List[Dict[str, Any]] = []
    for pool in pools:
        spectra_rows.extend(budget_spectra(pool))
    _write(out / "spectra.json", spectra_rows)

    signal_rows = [question_signals(pool) for pool in pools]
    _write(out / "signals.json", signal_rows)

    _write(out / "e1_real.json", e1_real_deletion(pools))
    _write(out / "e2.json", e2_duplication(pools))
    _write(out / "e3.json", e3_sample_size(pools))
    _write(out / "e4.json", e4_dimensionality(pools))
    _write(out / "anisotropy.json", anisotropy_diagnostics(pools))

    # embedding_qc is the headline kernel; the corpus arms are the ablation
    # that justifies it (see anisotropy_diagnostics and P-A4).
    kernels_full = [
        "embedding_qc",  # primary: chains as deviations from their own question
        "embedding:c0",  # ablation: raw encoder space
        "embedding:c1",  # ablation: corpus-level anisotropy correction
        "answer",
    ] + [f"alpha:{a}:qc" for a in ALPHA_GRID]
    # Both selection sweeps are done per pool, then that pool's kernel caches are
    # released. Holding them for every pool costs ~75 MB each and pushes a
    # 192-question cell past RAM into swap.
    pool40: List[Dict[str, Any]] = []
    pool_full: List[Dict[str, Any]] = []
    for pool in pools:
        pool40.append(question_outcomes(pool, pool_size=40, kernels=kernels_full))
        pool_full.append(
            question_outcomes(
                pool,
                pool_size=10**9,
                kernels=["embedding_qc", "answer"],
                objectives=list(CORE_OBJECTIVES),
            )
        )
        pool.release_caches()
    _write(out / "outcomes_pool40.json", pool40)
    _write(out / "outcomes_pool1024.json", pool_full)

    elapsed = time.time() - started
    print(f"[cell {model}/{dataset}] done in {elapsed / 60:.1f} min")
    return {"model": model, "dataset": dataset, "seconds": elapsed}


def _cell_worker(args: Tuple[str, str]) -> Dict[str, Any]:
    model, dataset = args
    try:
        return stage_cell(model, dataset)
    except FileNotFoundError as error:
        print(f"[cell {model}/{dataset}] skipped: {error}")
        return {"model": model, "dataset": dataset, "skipped": str(error)}


def stage_cells(cells: Sequence[Tuple[str, str]], jobs: int) -> None:
    import multiprocessing as mp

    if jobs <= 1:
        for cell in cells:
            _cell_worker(cell)
        return
    context = mp.get_context("spawn")
    with context.Pool(processes=jobs) as pool:
        for result in pool.imap_unordered(_cell_worker, cells):
            print(f"[cells] finished {result.get('model')}/{result.get('dataset')}")


# --------------------------------------------------------------------------
# Stage: synthetic
# --------------------------------------------------------------------------


def stage_synthetic() -> None:
    from diversity_reasoning.measurement import e1_synthetic, effective_number_validation

    out = ANALYSIS / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "e1_synthetic.json", e1_synthetic())
    _write(out / "effective_number.json", effective_number_validation())


# --------------------------------------------------------------------------
# Stage: assemble
# --------------------------------------------------------------------------


def _available_cells() -> List[Tuple[str, str]]:
    """Cells whose outcomes were produced by the current selection protocol.

    Outcomes written before treatment arms were averaged over subsample draws
    are single-draw booleans and are not comparable with averaged rates. Mixing
    the two in one table would silently compare different estimators, so stale
    cells are skipped loudly rather than blended.
    """
    found: List[Tuple[str, str]] = []
    for model, dataset in CELLS:
        path = ANALYSIS / model / dataset / "outcomes_pool40.json"
        if not path.exists():
            continue
        records = _load(path)
        if not records or "subsample_seeds" not in records[0]:
            print(
                f"[assemble] skipping {model}/{dataset}: outcomes predate the "
                "subsample-averaged protocol; re-run `run_analysis.py cell` for it."
            )
            continue
        found.append((model, dataset))
    return found


def _strata_lookup(model: str, dataset: str) -> Dict[str, Dict[str, Any]]:
    strata = _load(ANALYSIS / model / dataset / "strata.json")
    return {row["qid"]: row for row in strata["questions"]}


def _is_degenerate(objective: str, kernel_spec: str) -> bool:
    """True when an objective cannot express a preference on this kernel.

    Greedy VS_0 maximizes richness, which equals the subset size for any set of
    distinct vectors. On a continuous kernel every candidate therefore ties and
    the tie-break decides: measured on real banks it picked the eight
    lowest-indexed chains on 20 of 20 pools. That is index selection, not
    content selection, so it must not be reported as a winning *objective*. It
    stays in the full conditioning table, and remains meaningful on K_ans where
    exact ties make richness informative.
    """
    return objective == "vendi_0" and kernel_spec.startswith("embedding")


def _grouped_delta(
    outcomes: List[Dict[str, Any]],
    cell_key: str,
    budget: int,
    rule: str,
    qid_filter: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    from diversity_reasoning.winner import (
        accuracy_matrix,
        delta_versus_random,
        random_matrix,
    )

    qids_t, treated = accuracy_matrix(outcomes, cell_key, budget, rule)
    qids_r, random_mean, _ = random_matrix(outcomes, budget, rule)
    if not qids_t or qids_t != qids_r:
        common = [q for q in qids_t if q in set(qids_r)]
        index_t = {q: i for i, q in enumerate(qids_t)}
        index_r = {q: i for i, q in enumerate(qids_r)}
        treated = treated[[index_t[q] for q in common]]
        random_mean = random_mean[[index_r[q] for q in common]]
        qids_t = common
    if qid_filter is not None:
        keep = [i for i, q in enumerate(qids_t) if q in qid_filter]
        if len(keep) < 5:
            return None
        treated = treated[keep]
        random_mean = random_mean[keep]
    if treated.size == 0:
        return None
    stats = delta_versus_random(treated, random_mean)
    stats["accuracy"] = float(treated.mean())
    stats["random_accuracy"] = float(random_mean.mean())
    return stats


def _anisotropy_panel() -> Dict[str, Any]:
    """P-A4: kernel diagnostics plus effect sizes at each anisotropy level."""
    panel: Dict[str, Any] = {}
    for model, dataset in _available_cells():
        base = ANALYSIS / model / dataset
        diagnostics_path = base / "anisotropy.json"
        diagnostics = _load(diagnostics_path) if diagnostics_path.exists() else {"rows": []}
        outcomes = _normalize_outcomes(_load(base / "outcomes_pool40.json"))
        effects: List[Dict[str, Any]] = []
        for components, spec in ((0, "embedding:c0"), (1, "embedding:c1"), (-1, "embedding_qc")):
            prefix = f"{spec}|"
            if not any(
                any(key.startswith(prefix) for key in record.get("cells", {}))
                for record in outcomes
            ):
                continue
            for rule in RULES:
                for objective in ["vendi_1", "vendi_inf", "coverage", "facility_location"]:
                    stats = _grouped_delta(outcomes, f"{spec}|{objective}", HEADLINE_BUDGET, rule)
                    if stats is None:
                        continue
                    effects.append(
                        {
                            "components": components,
                            "rule": rule,
                            "objective": objective,
                            "delta": stats["delta"],
                            "ci_low": stats["ci_low"],
                            "ci_high": stats["ci_high"],
                        }
                    )
        panel[f"{model}|{dataset}"] = {
            "diagnostics": diagnostics.get("rows", []),
            "effects": effects,
        }
    return panel


def assemble_winner_map() -> None:
    from diversity_reasoning.winner import holm_family, objective_names, random_matrix

    objectives = objective_names()
    tb3_rows: List[Dict[str, Any]] = []
    p2a: Dict[str, Any] = {}
    r5_rows: List[Dict[str, Any]] = []
    r3_rows: List[Dict[str, Any]] = []
    conditioning_rows: List[Dict[str, Any]] = []

    for model, dataset in _available_cells():
        outcomes = _normalize_outcomes(_load(ANALYSIS / model / dataset / "outcomes_pool40.json"))
        strata = _strata_lookup(model, dataset)
        groups: Dict[str, set] = {"all": set(strata)}
        for row in strata.values():
            groups.setdefault(f"tercile:{row['snell_tercile']}", set()).add(row["qid"])
            groups.setdefault(f"tail:{row['tail']}", set()).add(row["qid"])

        # P-2a curves: accuracy vs budget on the embedding kernel.
        from diversity_reasoning.winner import accuracy_matrix

        budgets = [2, 3, 4, 8, 16, 32]
        for rule in RULES:
            curves: Dict[str, List[float]] = {}
            for objective in ["vendi_1", "vendi_inf", "coverage", "facility_location"]:
                key = f"{EMB}|{objective}"
                curves[objective] = []
                for budget in budgets:
                    _, values = accuracy_matrix(outcomes, key, budget, rule)
                    curves[objective].append(float(values.mean()) if values.size else float("nan"))
            random_curve, random_band = [], []
            for budget in budgets:
                _, mean_r, seed_matrix = random_matrix(outcomes, budget, rule)
                random_curve.append(float(mean_r.mean()) if mean_r.size else float("nan"))
                if seed_matrix.size:
                    seed_acc = seed_matrix.mean(axis=0)
                    low, high = np.quantile(seed_acc, [0.05, 0.95])
                else:
                    low = high = float("nan")
                random_band.append([float(low), float(high)])
            p2a[f"{model}|{dataset}|{rule}"] = {
                "budgets": budgets,
                "curves": curves,
                "random_mean": random_curve,
                "random_band": random_band,
            }

        # TB-3 winner rows + R2/R4 conditioning, with Holm within families.
        for rule in RULES:
            for group_name, qid_set in sorted(groups.items()):
                family: List[Dict[str, Any]] = []
                for objective in objectives:
                    stats = _grouped_delta(
                        outcomes, f"{EMB}|{objective}", HEADLINE_BUDGET, rule, qid_set
                    )
                    if stats is None:
                        continue
                    family.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "rule": rule,
                            "group": group_name,
                            "objective": objective,
                            "degenerate_on_this_kernel": _is_degenerate(objective, EMB),
                            **stats,
                        }
                    )
                if not family:
                    continue
                holm_family(family)
                conditioning_rows.extend(family)
                eligible = [
                    row for row in family if not _is_degenerate(row["objective"], EMB)
                ] or family
                winner = max(eligible, key=lambda row: row["delta"])
                tb3_rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "rule": rule,
                        "group": group_name,
                        "budget": HEADLINE_BUDGET,
                        "winner": winner["objective"],
                        "delta": winner["delta"],
                        "ci": [winner["ci_low"], winner["ci_high"]],
                        "p_holm": winner["p_holm"],
                        "accuracy": winner["accuracy"],
                        "random_accuracy": winner["random_accuracy"],
                        "n": winner["n"],
                        "practically_null": winner["practically_null"],
                    }
                )

        # R5: q inertness on the answer kernel.
        for rule in RULES:
            for budget in budgets:
                accuracies = {}
                for q_name in [o for o in objectives if o.startswith("vendi_")]:
                    _, values = accuracy_matrix(outcomes, f"answer|{q_name}", budget, rule)
                    if values.size:
                        accuracies[q_name] = float(values.mean())
                if accuracies:
                    r5_rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "rule": rule,
                            "budget": budget,
                            "accuracy_by_q": accuracies,
                            "spread": max(accuracies.values()) - min(accuracies.values()),
                        }
                    )

        # R3: alpha threshold per (rule, objective).
        from diversity_reasoning.winner import ALPHA_GRID

        for rule in RULES:
            for objective in ["vendi_1", "vendi_inf", "coverage"]:
                per_alpha = []
                for alpha in (0.0,) + tuple(ALPHA_GRID) + (1.0,):
                    if alpha == 0.0:
                        key = f"{EMB}|{objective}"
                    elif alpha == 1.0:
                        key = f"answer|{objective}"
                    else:
                        key = f"alpha:{alpha}:qc|{objective}"
                    stats = _grouped_delta(outcomes, key, HEADLINE_BUDGET, rule)
                    if stats is not None:
                        per_alpha.append({"alpha": alpha, **stats})
                separating = [
                    row["alpha"] for row in per_alpha if row["ci_low"] > 0 and row["delta"] > 0.01
                ]
                r3_rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "rule": rule,
                        "objective": objective,
                        "curve": per_alpha,
                        "alpha_star": min(separating) if separating else None,
                    }
                )

    _write(TABLES / "tb3_winner_map.json", tb3_rows)
    _write(FIGURE_DATA / "P-A4.json", _anisotropy_panel())
    _write(TABLES / "r_conditioning_full.json", conditioning_rows)
    _write(FIGURE_DATA / "P-2a.json", p2a)
    _write(FIGURE_DATA / "P-2f.json", r5_rows)
    _write(TABLES / "tb4_alpha_star.json", r3_rows)
    _write(FIGURE_DATA / "P-2d.json", r3_rows)

    # P-2c / P-2e: coverage & VS_inf deltas by tail stratum.
    tail_panel: List[Dict[str, Any]] = [
        row
        for row in conditioning_rows
        if row["group"].startswith("tail:") and row["objective"] in {"coverage", "vendi_inf"}
    ]
    _write(FIGURE_DATA / "P-2c.json", tail_panel)

    # P-2b: hypothesis strip.
    hypotheses = _hypothesis_strip(conditioning_rows, r5_rows)
    _write(FIGURE_DATA / "P-2b.json", hypotheses)


def _hypothesis_strip(
    conditioning: List[Dict[str, Any]], r5_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    def pick(rule: str, group: str, objective: str) -> List[Dict[str, Any]]:
        return [
            row
            for row in conditioning
            if row["rule"] == rule and row["group"] == group and row["objective"] == objective
        ]

    def mean_stat(rows: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        if not rows:
            return None
        import numpy as np

        return {
            "delta": float(np.mean([r["delta"] for r in rows])),
            "ci_low": float(np.mean([r["ci_low"] for r in rows])),
            "ci_high": float(np.mean([r["ci_high"] for r in rows])),
        }

    strip: List[Dict[str, Any]] = []
    definitions = [
        ("H0", "Some selection objective beats random (MV, all questions)", None),
        (
            "H1",
            "Coverage hurts majority vote on modal questions",
            ("majority_vote", "tail:modal", "coverage"),
        ),
        (
            "H2",
            "Coverage helps pass@k on minority/tail questions",
            ("pass_at_k", "tail:minority", "coverage"),
        ),
        ("H3", "VS_inf helps pass@k overall", ("pass_at_k", "all", "vendi_inf")),
        (
            "H4",
            "No objective beats random when the answer is absent",
            ("majority_vote", "tail:absent", "coverage"),
        ),
        ("H5", "On K_ans every q behaves identically (Theorem 4.1)", None),
        ("H6", "Answer-aware mixing (alpha>0) is required to beat random on MV", None),
    ]
    for name, text, selector in definitions:
        entry: Dict[str, Any] = {"id": name, "statement": text}
        if name == "H0":
            candidates = [
                row
                for row in conditioning
                if row["rule"] == "majority_vote" and row["group"] == "all"
            ]
            best = max(candidates, key=lambda r: r["delta"], default=None)
            if best:
                entry.update(
                    {
                        "delta": best["delta"],
                        "ci": [best["ci_low"], best["ci_high"]],
                        "verdict": "accept" if best["ci_low"] > 0 else "reject",
                        "detail": (
                            f"best objective {best['objective']} "
                            f"on {best['model']}/{best['dataset']}"
                        ),
                    }
                )
        elif name == "H5":
            spreads = [row["spread"] for row in r5_rows]
            if spreads:
                import numpy as np

                worst = float(np.max(spreads))
                entry.update(
                    {
                        "delta": worst,
                        "ci": [0.0, worst],
                        "verdict": "accept"
                        if worst < 1e-9
                        else ("accept~" if worst < 0.02 else "reject"),
                        "detail": f"max accuracy spread across q on K_ans = {worst:.4f}",
                    }
                )
        elif name == "H6":
            entry["verdict"] = "see TB-4"
        elif selector is not None:
            rows = pick(*selector)
            stat = mean_stat(rows)
            if stat:
                expected_negative = name in {"H1"}
                null_expected = name == "H4"
                if null_expected:
                    verdict = (
                        "accept"
                        if all(
                            r["ci_low"] <= 0 <= r["ci_high"] or abs(r["delta"]) < 0.01 for r in rows
                        )
                        else "reject"
                    )
                elif expected_negative:
                    verdict = "accept" if stat["delta"] < 0 else "reject"
                else:
                    verdict = "accept" if stat["delta"] > 0 else "reject"
                entry.update({**stat, "verdict": verdict, "n_cells": len(rows)})
        strip.append(entry)
    return strip


def assemble_measurement() -> None:
    from diversity_reasoning.measurement import e5_two_log_functionals

    e5_all: Dict[str, Any] = {}
    e1_real: Dict[str, Any] = {}
    e2_all: Dict[str, Any] = {}
    e3_all: Dict[str, Any] = {}
    e4_all: Dict[str, Any] = {}
    strata_summary: Dict[str, Any] = {}
    tau_sensitivity: Dict[str, Any] = {}
    for model, dataset in _available_cells():
        base = ANALYSIS / model / dataset
        cell_id = f"{model}|{dataset}"
        spectra_rows = _load(base / "spectra.json")
        e5_all[cell_id] = e5_two_log_functionals(spectra_rows)
        tau_sensitivity[cell_id] = _tau_sensitivity(spectra_rows)
        e1_real[cell_id] = _load(base / "e1_real.json")
        e2_all[cell_id] = _load(base / "e2.json")
        e3_all[cell_id] = _load(base / "e3.json")
        e4_all[cell_id] = _load(base / "e4.json")
        strata = _load(base / "strata.json")
        strata_summary[cell_id] = {
            "occupancy": strata["occupancy_quintile_x_tail"],
            "underpowered": strata["underpowered_cells"],
            "mean_pass_at_1": strata["mean_pass_at_1"],
            "unparsed_rate": strata["unparsed_rate"],
            "snell_vs_level": strata.get("snell_vs_level", {}),
            "entropy": [row["entropy"] for row in strata["questions"]],
            "tail_counts": {
                label: sum(1 for row in strata["questions"] if row["tail"] == label)
                for label in ("modal", "minority", "tail", "absent")
            },
        }
    effective_path = ANALYSIS / "synthetic" / "effective_number.json"
    if effective_path.exists():
        _write(FIGURE_DATA / "P-0c.json", _load(effective_path))
    synthetic_path = ANALYSIS / "synthetic" / "e1_synthetic.json"
    _write(
        FIGURE_DATA / "P-1a.json",
        {
            "synthetic": _load(synthetic_path) if synthetic_path.exists() else {},
            "real": e1_real,
        },
    )
    _write(FIGURE_DATA / "P-1b.json", e2_all)
    _write(FIGURE_DATA / "P-1c.json", e3_all)
    _write(FIGURE_DATA / "P-1d.json", e4_all)
    _write(FIGURE_DATA / "P-1e.json", e5_all)
    _write(FIGURE_DATA / "P-0.json", strata_summary)
    _write(TABLES / "tb0_strata.json", strata_summary)
    tb1 = {cell: data.get("n_star_tau_0.9") for cell, data in e3_all.items()}
    _write(TABLES / "tb1_sample_size.json", tb1)
    tb2 = {cell: data.get("correlations") for cell, data in e5_all.items()}
    _write(TABLES / "tb2_correlations.json", tb2)
    _write(FIGURE_DATA / "P-A3.json", tau_sensitivity)


def _tau_sensitivity(spectra_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """P-A3: does the nonzero-eigenvalue threshold change coverage or its rank?"""
    from scipy.stats import kendalltau

    result: Dict[str, Any] = {}
    for kernel in sorted({row["kernel"] for row in spectra_rows}):
        rows = [row for row in spectra_rows if row["kernel"] == kernel]
        reference = np.asarray([row["pseudo_logdet"] for row in rows])
        entry: Dict[str, Any] = {"n": len(rows), "arms": {}}
        for arm, label in (("pseudo_logdet_tau8", "1e-8"), ("pseudo_logdet_tau12", "1e-12")):
            if arm not in rows[0]:
                continue
            values = np.asarray([row[arm] for row in rows])
            difference = np.abs(values - reference)
            relative = difference / np.maximum(1e-9, np.abs(reference))
            entry["arms"][label] = {
                "mean_abs_diff": float(np.mean(difference)),
                "median_abs_diff": float(np.median(difference)),
                "max_abs_diff": float(np.max(difference)),
                # The distribution is heavy-tailed: most pools are untouched
                # while a minority shift by hundreds of nats, so report the
                # affected fraction rather than only the mean.
                "fraction_changed_1pct": float(np.mean(relative > 0.01)),
                "kendall_tau_vs_default": float(kendalltau(values, reference).statistic),
            }
        result[kernel] = entry
    return result


def assemble_winnable() -> None:
    """P-2g: what governs how much any selection objective can gain.

    The blueprint's tail-heaviness taxonomy partitions questions into ones a
    selector cannot lose (correct answer is already modal), cannot win (absent
    from the pool entirely), and can actually win — correct answer present but
    not modal. The last group is the only place a selection objective has work
    to do, so its share should predict the achievable gain better than raw
    headroom does. Both are reported so the comparison is visible.
    """
    strata = _load(TABLES / "tb0_strata.json") if (TABLES / "tb0_strata.json").exists() else {}
    conditioning = (
        _load(TABLES / "r_conditioning_full.json")
        if (TABLES / "r_conditioning_full.json").exists()
        else []
    )
    if not strata or not conditioning:
        return
    points: List[Dict[str, Any]] = []
    for cell_id, data in sorted(strata.items()):
        model, dataset = cell_id.split("|")
        counts = data.get("tail_counts", {})
        total = sum(counts.values())
        if not total:
            continue
        winnable = (counts.get("minority", 0) + counts.get("tail", 0)) / total
        for rule in RULES:
            candidates = [
                row
                for row in conditioning
                if row["model"] == model
                and row["dataset"] == dataset
                and row["rule"] == rule
                and row["group"] == "all"
                and not row.get("degenerate_on_this_kernel")
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda row: row["delta"])
            points.append(
                {
                    "cell": cell_id,
                    "model": model,
                    "dataset": dataset,
                    "rule": rule,
                    "winnable_fraction": winnable,
                    "headroom": 1.0 - best["random_accuracy"],
                    "modal_fraction": counts.get("modal", 0) / total,
                    "absent_fraction": counts.get("absent", 0) / total,
                    "best_delta": best["delta"],
                    "best_objective": best["objective"],
                    "ci_low": best["ci_low"],
                    "ci_high": best["ci_high"],
                }
            )
    correlations: Dict[str, Any] = {}
    if len(points) >= 4:
        from scipy.stats import pearsonr, spearmanr

        for rule in RULES:
            subset = [p for p in points if p["rule"] == rule]
            if len(subset) < 4:
                continue
            deltas = np.asarray([p["best_delta"] for p in subset])
            for name in ("winnable_fraction", "headroom"):
                predictor = np.asarray([p[name] for p in subset])
                if np.std(predictor) == 0:
                    continue
                correlations[f"{rule}|{name}"] = {
                    "pearson": float(pearsonr(predictor, deltas).statistic),
                    "spearman": float(spearmanr(predictor, deltas).statistic),
                    "n": len(subset),
                }
    _write(FIGURE_DATA / "P-2g.json", {"points": points, "correlations": correlations})


def _common_budget_spectra(
    spectra: List[Dict[str, Any]], kernel: str = "embedding"
) -> Dict[str, Dict[str, Any]]:
    """One spectrum per question, all at the *same* budget.

    Taking each question's largest available budget silently compares spectra of
    different sizes. On MATH, where unparsed chains shrink pools unevenly, the
    per-question budget ranges 128-1024 and VS_0 becomes a perfect proxy for
    pool size (r = 1.000), manufacturing a -0.99 "anticorrelation" with coverage
    that is pure size confound. Every cross-question comparison therefore uses
    the largest budget present for *all* questions in the cell.
    """
    rows = [row for row in spectra if row["kernel"] == kernel]
    if not rows:
        return {}
    by_question: Dict[str, set] = {}
    for row in rows:
        by_question.setdefault(row["qid"], set()).add(row["budget"])
    if not by_question:
        return {}
    # The largest budget retaining at least this share of questions. Using the
    # strict intersection lets one unusually small pool drag every question down
    # to its budget (a single MATH pool forced 128 on an otherwise 512 cell).
    retention = 0.9
    total = len(by_question)
    candidates = sorted({b for budgets in by_question.values() for b in budgets}, reverse=True)
    for budget in candidates:
        kept = [qid for qid, budgets in by_question.items() if budget in budgets]
        if len(kept) >= retention * total:
            keep = set(kept)
            return {
                row["qid"]: row for row in rows if row["budget"] == budget and row["qid"] in keep
            }
    return {}


def assemble_metric_correlations() -> None:
    """P-4a: correlation between every functional and the other quantities.

    Follows the presentation used for the Vendi score family in the literature
    (Pasarkar & Dieng 2024, Fig. 5): a Pearson-correlation matrix showing how
    each order q relates to the other measures in play. Here the companions are
    coverage, answer entropy, the vote margin, and the downstream outcomes, so
    the panel shows directly which orders carry information the others do not.
    """
    from scipy.stats import pearsonr

    per_cell: Dict[str, Any] = {}
    for model, dataset in _available_cells():
        base = ANALYSIS / model / dataset
        signals = _load(base / "signals.json")
        spectra = _load(base / "spectra.json")
        strata = {row["qid"]: row for row in _load(base / "strata.json")["questions"]}
        # One row per question: the full-pool spectra on the embedding kernel.
        full = _common_budget_spectra(spectra)
        columns: Dict[str, List[float]] = {}
        qids = [s["qid"] for s in signals if s["qid"] in full and s["qid"] in strata]
        if len(qids) < 8:
            continue
        for q in ("0", "0.1", "0.5", "1", "2", "inf"):
            columns[f"VS_{q}"] = [full[qid][f"vs_{q}"] for qid in qids]
        columns["coverage"] = [full[qid]["pseudo_logdet"] for qid in qids]
        by_qid = {s["qid"]: s for s in signals}
        columns["answer entropy"] = [by_qid[qid]["answer_entropy"] for qid in qids]
        columns["vote margin"] = [by_qid[qid]["vote_margin"] for qid in qids]
        columns["mean logprob"] = [by_qid[qid]["mean_logprob"] for qid in qids]
        columns["pass@1"] = [strata[qid]["pass_at_1"] for qid in qids]
        columns["MV correct"] = [float(by_qid[qid]["mv_correct"]) for qid in qids]
        names = list(columns)
        matrix = []
        for a in names:
            row = []
            for b in names:
                x, y = np.asarray(columns[a]), np.asarray(columns[b])
                row.append(
                    float(pearsonr(x, y).statistic)
                    if np.std(x) > 0 and np.std(y) > 0
                    else float("nan")
                )
            matrix.append(row)
        per_cell[f"{model}|{dataset}"] = {
            "names": names,
            "matrix": matrix,
            "n_questions": len(qids),
            "budget": int(next(iter(full.values()))["budget"]),
        }
    _write(FIGURE_DATA / "P-4a.json", per_cell)


def assemble_quality_scatter() -> None:
    """P-4b: each functional against downstream quality, per question.

    The presentation mirrors the Vendi-score-versus-human-evaluation panels in
    the literature (Pasarkar & Dieng 2024, Figs. 4 and 7): one scatter per
    functional against an outcome, annotated with its correlation, so a reader
    can see which orders actually track quality rather than only each other.
    """
    from scipy.stats import pearsonr

    per_cell: Dict[str, Any] = {}
    for model, dataset in _available_cells():
        base = ANALYSIS / model / dataset
        signals = {s["qid"]: s for s in _load(base / "signals.json")}
        strata = {row["qid"]: row for row in _load(base / "strata.json")["questions"]}
        spectra = _load(base / "spectra.json")
        full = _common_budget_spectra(spectra)
        qids = [q for q in full if q in signals and q in strata]
        if len(qids) < 8:
            continue
        series: Dict[str, Any] = {}
        outcome = np.asarray([strata[q]["pass_at_1"] for q in qids])
        for label, values in (
            ("VS_1", [full[q]["vs_1"] for q in qids]),
            ("VS_inf", [full[q]["vs_inf"] for q in qids]),
            ("coverage", [full[q]["pseudo_logdet"] for q in qids]),
            ("answer entropy", [signals[q]["answer_entropy"] for q in qids]),
        ):
            array = np.asarray(values, dtype=float)
            stat = (
                float(pearsonr(array, outcome).statistic)
                if np.std(array) > 0 and np.std(outcome) > 0
                else float("nan")
            )
            series[label] = {
                "x": array.tolist(),
                "y": outcome.tolist(),
                "pearson": stat,
            }
        per_cell[f"{model}|{dataset}"] = {
            "series": series,
            "n_questions": len(qids),
            "budget": int(next(iter(full.values()))["budget"]),
        }
    _write(FIGURE_DATA / "P-4b.json", per_cell)


def assemble_signals() -> None:
    from diversity_reasoning.signals import r6_signal_shootout, r7_escalation

    r6_all: Dict[str, Any] = {}
    signal_rows_by_cell: Dict[str, List[Dict[str, Any]]] = {}
    for model, dataset in _available_cells():
        rows = _load(ANALYSIS / model / dataset / "signals.json")
        signal_rows_by_cell[f"{model}|{dataset}"] = rows
        r6_all[f"{model}|{dataset}"] = r6_signal_shootout(rows)

    # Confound panel: embedding diversity lift, pooled across datasets vs within.
    confound: Dict[str, Any] = {}
    for model in sorted({m for m, _ in _available_cells()}):
        pooled_rows: List[Dict[str, Any]] = []
        per_dataset_lifts: List[float] = []
        for dataset in ("gsm8k", "math"):
            rows = signal_rows_by_cell.get(f"{model}|{dataset}")
            if rows:
                pooled_rows.extend(rows)
                cell_r6 = r6_all[f"{model}|{dataset}"]["signals"].get("embedding_vs1")
                if cell_r6:
                    per_dataset_lifts.append(cell_r6["lift"])
        if pooled_rows and len(per_dataset_lifts) == 2:
            pooled = r6_signal_shootout(pooled_rows)["signals"].get("embedding_vs1")
            confound[model] = {
                "pooled_lift": pooled["lift"] if pooled else None,
                "within_dataset_lifts": per_dataset_lifts,
            }
    _write(FIGURE_DATA / "P-3a.json", r6_all)
    _write(FIGURE_DATA / "P-3b.json", confound)
    tb5 = {
        cell: {
            name: {"auc": s["auc"], "lift": s["lift"], "auc_ci": s["auc_ci"]}
            for name, s in data["signals"].items()
        }
        for cell, data in r6_all.items()
    }
    _write(TABLES / "tb5_signals.json", tb5)

    r7_all: Dict[str, Any] = {}
    for dataset in ("gsm8k", "math"):
        cheap = signal_rows_by_cell.get(f"{CHEAP_MODEL}|{dataset}")
        expensive = signal_rows_by_cell.get(f"{HEADLINE_MODEL}|{dataset}")
        if cheap and expensive:
            r7_all[dataset] = r7_escalation(cheap, expensive)
    _write(FIGURE_DATA / "P-3c.json", r7_all)
    _write(
        TABLES / "tb6_operating_points.json",
        {dataset: data.get("operating_points") for dataset, data in r7_all.items()},
    )


def stage_seed_variance(
    model: str = "qwen2.5-0.5b",
    dataset: str = "gsm8k",
    seeds: Sequence[str] = ("", "-g1", "-g2"),
) -> None:
    """B1 spot check: does the generation seed change what a bank measures?

    Compares the main bank (g = 0) against the -g1/-g2 suffix banks on their
    shared questions: per-question pass@1 spread across seeds, answer-entropy
    spread, and tail-heaviness label agreement. This bounds bank-level
    generation variance, which is otherwise a stated limitation.
    """
    from diversity_reasoning.pools import load_cell
    from diversity_reasoning.spectra import answer_entropy
    from diversity_reasoning.strata import tail_heaviness

    per_seed: Dict[str, Dict[str, Any]] = {}
    for suffix in seeds:
        try:
            pools = load_cell(CACHE, f"{model}{suffix}", dataset, with_embeddings=False)
        except FileNotFoundError:
            print(f"[seedvar] bank {model}{suffix}/{dataset} not present; skipping")
            continue
        per_seed[suffix or "-g0"] = {
            pool.qid: {
                "pass_at_1": pool.pass_at_1,
                "entropy": answer_entropy(list(pool.answer_counts().values()))["entropy"],
                "tail": tail_heaviness(pool),
            }
            for pool in pools
        }
    if len(per_seed) < 2:
        print("[seedvar] fewer than two seed banks available; nothing to compare")
        return
    shared = sorted(set.intersection(*(set(v) for v in per_seed.values())))
    labels = sorted(per_seed)
    pass1 = np.asarray([[per_seed[s][q]["pass_at_1"] for s in labels] for q in shared])
    entropy = np.asarray([[per_seed[s][q]["entropy"] for s in labels] for q in shared])
    tails = [[per_seed[s][q]["tail"] for s in labels] for q in shared]
    agreement = float(np.mean([len(set(row)) == 1 for row in tails]))
    payload = {
        "model": model,
        "dataset": dataset,
        "seed_banks": labels,
        "n_shared_questions": len(shared),
        "pass_at_1_sd_across_seeds_mean": float(pass1.std(axis=1).mean()),
        "pass_at_1_sd_across_seeds_max": float(pass1.std(axis=1).max()),
        "entropy_sd_across_seeds_mean": float(entropy.std(axis=1).mean()),
        "tail_label_agreement": agreement,
        "per_question": [
            {
                "qid": q,
                "pass_at_1": dict(zip(labels, pass1[i].tolist())),
                "tails": dict(zip(labels, tails[i])),
            }
            for i, q in enumerate(shared)
        ],
    }
    _write(TABLES / "seed_variance.json", payload)
    print(
        f"[seedvar] {len(shared)} questions x {len(labels)} seeds: "
        f"pass@1 sd mean={payload['pass_at_1_sd_across_seeds_mean']:.4f}, "
        f"tail agreement={agreement:.2f}"
    )


def stage_encoder_stability(
    model: str,
    dataset: str,
    alternate: str = "mxbai-embed-large-v1",
) -> None:
    """TB-7: rank stability of every functional between two encoders on one cell."""
    from diversity_reasoning.measurement import encoder_stability
    from diversity_reasoning.pools import load_cell

    primary = load_cell(CACHE, model, dataset, encoder_short="bge-large-en-v1.5")
    other = load_cell(CACHE, model, dataset, encoder_short=alternate)
    payload = encoder_stability(primary, other)
    payload.update(
        {"cell": f"{model}|{dataset}", "primary": "bge-large-en-v1.5", "alternate": alternate}
    )
    _write(TABLES / "tb7_encoder_stability.json", payload)


def stage_assemble() -> None:
    FIGURE_DATA.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    assemble_measurement()
    assemble_winner_map()
    assemble_winnable()
    assemble_metric_correlations()
    assemble_quality_scatter()
    assemble_signals()


# --------------------------------------------------------------------------


import numpy as np  # noqa: E402  (used by assemble helpers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["pull", "cell", "cells", "synthetic", "assemble", "encoders", "seedvar", "all"],
    )
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--jobs", type=int, default=6)
    arguments = parser.parse_args()

    from diversity_reasoning.env import load_local_env

    load_local_env()
    if arguments.stage == "pull":
        stage_pull(CELLS)
    elif arguments.stage == "cell":
        if not arguments.model or not arguments.dataset:
            raise SystemExit("cell stage needs --model and --dataset")
        stage_cell(arguments.model, arguments.dataset)
    elif arguments.stage == "cells":
        stage_cells(CELLS, arguments.jobs)
    elif arguments.stage == "synthetic":
        stage_synthetic()
    elif arguments.stage == "assemble":
        stage_assemble()
    elif arguments.stage == "seedvar":
        stage_seed_variance()
    elif arguments.stage == "encoders":
        if not arguments.model or not arguments.dataset:
            raise SystemExit("encoders stage needs --model and --dataset")
        stage_encoder_stability(arguments.model, arguments.dataset)
    else:
        stage_pull(CELLS)
        stage_cells(CELLS, arguments.jobs)
        stage_synthetic()
        stage_assemble()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
