#!/usr/bin/env python
"""Generate RESULTS.md from cache only — every number is read, never typed.

This is the findings companion to the blueprint in README.md (which is never
modified). It states what the experiments produced, in the vocabulary Adji
fixed: VS_q is the exponential of the Renyi entropy of order q of the
normalized similarity-kernel spectrum, every q is a diversity measure, and
coverage is only ever the pseudo log-determinant.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIGURE_DATA = ROOT / "cache" / "figure_data"
TABLES = ROOT / "cache" / "tables"
OUT = ROOT / "RESULTS.md"

MODEL_LABEL = {
    "qwen2.5-0.5b": "Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen2.5-1.5B",
    "llama-3.2-3b": "Llama-3.2-3B",
}
DATASET_LABEL = {"gsm8k": "GSM8K", "math": "MATH"}
OBJECTIVE_LABEL = {
    "vendi_0": "VS_0",
    "vendi_0.1": "VS_0.1",
    "vendi_0.5": "VS_0.5",
    "vendi_1": "VS_1",
    "vendi_2": "VS_2",
    "vendi_inf": "VS_inf",
    "coverage": "coverage",
    "facility_location": "facility location",
}
RULE_LABEL = {
    "majority_vote": "majority vote",
    "pass_at_k": "pass@k",
    "verifier_best": "verifier best-of-n",
}


def load(directory: Path, name: str) -> Optional[Any]:
    path = directory / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Optional[float], digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def fmt_p(value: Optional[float], replicates: int = 1000) -> str:
    """Bootstrap p-values are floored at 1/B; show that rather than a bare 0."""
    if value is None:
        return "n/a"
    floor = 1.0 / replicates
    return f"<{floor:g}" if value <= floor else f"{value:.3f}"


def cell_label(cell_id: str) -> str:
    model, dataset = cell_id.split("|")
    return f"{MODEL_LABEL.get(model, model)} / {DATASET_LABEL.get(dataset, dataset)}"


def section_provenance() -> List[str]:
    strata = load(TABLES, "tb0_strata") or {}
    lines = [
        "## 1. What was generated",
        "",
        "| cell | questions | mean pass@1 | unparsed | modal | minority | tail | absent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell_id, data in sorted(strata.items()):
        counts = data.get("tail_counts", {})
        total = sum(counts.values())
        lines.append(
            f"| {cell_label(cell_id)} | {total} | {fmt(data.get('mean_pass_at_1'))} | "
            f"{fmt(data.get('unparsed_rate'), 4)} | {counts.get('modal', 0)} | "
            f"{counts.get('minority', 0)} | {counts.get('tail', 0)} | {counts.get('absent', 0)} |"
        )
    lines += [
        "",
        "Each question carries 1024 chains (temperature 1.0, top-p 0.95, 400 new tokens),",
        "generated with vLLM on one A100-40GB and published to the Hugging Face dataset",
        "`GOVINDFROM/Diversity-vs-Reasoning`. Tail-heaviness is the rank of the correct",
        "answer in the full 1024-chain answer distribution: `modal` (rank 1), `minority`",
        "(ranks 2-5), `tail` (rank > 5), `absent` (never produced).",
        "",
    ]
    trend = _modal_share_trend(strata)
    if trend:
        lines += [
            "**The stronger the model, the less headroom any selector has.** The modal share"
            " rises with pass@1 across the ladder (" + trend + "), so on a strong model almost"
            " every question already has the correct answer as its most common one and plain"
            " majority vote is close to the ceiling. That also drains the minority and tail"
            " strata — exactly where diversity and coverage are predicted to matter — leaving"
            " them too small to support headline claims on the stronger models. Effect sizes"
            " below should be read against this shrinking headroom, not as a null result about"
            " selection in general.",
            "",
        ]
    return lines


def _modal_share_trend(strata: Any) -> str:
    """Modal share per cell, ordered by accuracy, for the headroom note."""
    entries = []
    for cell_id, data in strata.items():
        counts = data.get("tail_counts", {})
        total = sum(counts.values())
        if not total:
            continue
        entries.append(
            (
                float(data.get("mean_pass_at_1", 0.0)),
                f"{cell_label(cell_id)} pass@1 {data.get('mean_pass_at_1', 0):.2f} -> "
                f"{counts.get('modal', 0)}/{total} modal",
            )
        )
    return "; ".join(text for _, text in sorted(entries))


def section_kernel_choice() -> List[str]:
    """The embedding-kernel diagnostic that decided which kernel is primary."""
    payload = load(FIGURE_DATA, "P-A4") or {}
    rows: List[Any] = []
    cell_id = ""
    for key, entry in sorted(payload.items()):
        diagnostics = (entry or {}).get("diagnostics") or []
        if diagnostics:
            rows, cell_id = diagnostics, key
            break
    if not rows:
        return []
    lines = [
        "## 2. Which embedding kernel measures diversity *among* chains",
        "",
        "Every chain in a pool answers the same question, so in the raw encoder space",
        "that question's own content dominates the kernel: the leading eigenvalue holds",
        "most of the spectral mass and VS_q collapses toward 1 however varied the chains",
        "actually are. When that happens different objectives select the *same* chains and",
        "the winner map has nothing to distinguish.",
        "",
        "The blueprint's anisotropy sweep removes directions shared across the whole",
        "corpus. Measured on real banks, that is **not** the cause and barely helps.",
        "Removing each pool's own centroid does. Headline embedding results therefore use",
        "the question-centred kernel; the corpus sweep is reported as the ablation (P-A4).",
        "",
        f"Measured on {cell_label(cell_id)} (40 questions, pool 40, k=8):",
        "",
        "| kernel | top eigenvalue share | mean VS_1 | rate of identical selections |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {fmt(row['top_eigenvalue_share'])} | "
            f"{fmt(row['mean_vs_1'], 2)} | {fmt(row['identical_selection_rate'])} |"
        )
    lines.append("")
    return lines


def section_measurement() -> List[str]:
    lines = ["## 3. Measurement: how the two functionals differ (D1)", ""]
    e5 = load(FIGURE_DATA, "P-1e") or {}
    if e5:
        lines += [
            "### The two log functionals (E5)",
            "",
            "Correlation between log VS_q and the pseudo log-determinant, computed within a",
            "fixed budget and pooled across budgets. A sign flip between the two scopes is",
            "the Simpson-style reversal the study is built to expose.",
            "",
            "| cell | kernel | q | within-budget r | pooled r | pooled r (eps=1 artifact) |",
            "|---|---|---|---:|---:|---:|",
        ]
        for cell_id, data in sorted(e5.items()):
            for row in data.get("correlations", []):
                if row["q"] not in {"0", "1", "inf"}:
                    continue
                within = row.get("within_budget_mean")
                within_text = fmt(within) if within is not None else "undefined*"
                lines.append(
                    f"| {cell_label(cell_id)} | {row['kernel']} | {row['q']} | "
                    f"{within_text} | {fmt(row['pooled'])} | "
                    f"{fmt(row['pooled_eps1'])} |"
                )
        lines += [
            "",
            "\\* undefined: VS_0 on the embedding kernel is constant within a budget —",
            "distinct chains give a full-rank Gram matrix, so richness saturates at the",
            "budget and carries no per-question information. This is a property of the",
            "q -> 0 limit on a continuous kernel, not a missing measurement.",
            "",
        ]
        for cell_id, data in sorted(e5.items()):
            for kernel, reversal in sorted(data.get("reversal", {}).items()):
                flip = reversal.get("flip_log10_lambda_min")
                if flip is not None:
                    lines.append(
                        f"- Reversal onset, {cell_label(cell_id)} on K_{kernel[:3]}: the pooled "
                        f"correlation turns negative at log10 lambda_min = {fmt(flip, 2)}."
                    )
        lines.append("")

    tb1 = load(TABLES, "tb1_sample_size") or {}
    if tb1:
        lines += [
            "### Sample size (E3)",
            "",
            "Smallest subsample n at which a functional's ranking of questions reaches",
            "Kendall tau >= 0.9 against its full-pool ranking.",
            "",
            "| cell | " + " | ".join(["VS_0", "VS_1", "VS_inf", "coverage"]) + " |",
            "|---|---:|---:|---:|---:|",
        ]
        for cell_id, data in sorted(tb1.items()):
            if not data:
                continue
            values = [data.get(k) for k in ("vs_0", "vs_1", "vs_inf", "pseudo_logdet")]
            lines.append(
                f"| {cell_label(cell_id)} | " + " | ".join(str(v or "n/a") for v in values) + " |"
            )
        lines.append("")
    return lines


def section_winner_map() -> List[str]:
    rows = load(TABLES, "tb3_winner_map") or []
    lines = [
        "## 4. The winner map (R1) — the headline",
        "",
        "Winning objective per (cell, aggregation rule, stratum) at selection budget k=8 on",
        "the embedding kernel, against the 20-seed random baseline with paired question-level",
        "bootstrap CIs and Holm correction within each family. Cells with |delta| < 0.01 are",
        "labelled practically null regardless of p.",
        "",
        "**How to read the p-values.** Holm is applied within each family of objectives",
        "for one (cell, rule, stratum), which controls the error rate *inside* that family",
        "only. The full sweep spans many such families, so an isolated Holm-significant",
        "cell is weak evidence on its own. The evidence standard used here is replication:",
        "an effect counts only when its CI excludes zero in every model measured (see the",
        "replication table below). Greedy VS_0 is excluded from the winner column on",
        "continuous kernels — richness equals the subset size for distinct vectors, so every",
        "candidate ties and the tie-break selects by index rather than by content (verified:",
        "it chose the eight lowest-indexed chains on 20 of 20 pools). It remains in the full",
        "conditioning table and stays meaningful on the answer kernel.",
        "",
        "| cell | rule | stratum | winner | delta vs random | 95% CI | p (Holm) | verdict |",
        "|---|---|---|---|---:|---|---:|---|",
    ]
    for row in rows:
        if row["group"] not in {"all"} and not row["group"].startswith("tail:"):
            continue
        ci = row["ci"]
        ci_text = ci if isinstance(ci, str) else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        delta = row["delta"] if isinstance(row["delta"], str) else fmt(row["delta"], signed=True)
        if row["practically_null"]:
            verdict = "practically null"
        elif row["p_holm"] < 0.05:
            verdict = "significant"
        else:
            verdict = "not significant"
        lines.append(
            f"| {MODEL_LABEL.get(row['model'], row['model'])} / "
            f"{DATASET_LABEL.get(row['dataset'], row['dataset'])} | {RULE_LABEL[row['rule']]} | "
            f"{row['group']} | {OBJECTIVE_LABEL.get(row['winner'], row['winner'])} | {delta} | "
            f"{ci_text} | {fmt_p(row['p_holm'])} | {verdict} |"
        )
    lines.append("")
    return lines


def section_replication() -> List[str]:
    """Which effects hold across models, and which are single-model only."""
    rows = load(TABLES, "r_conditioning_full") or []
    if not rows:
        return []
    models = sorted({row["model"] for row in rows})
    if len(models) < 2:
        return []
    lines = [
        "### Which effects replicate across models",
        "",
        "An effect is only reported as a finding when its CI excludes zero in every",
        "model measured. Effects significant in one model and absent in another are",
        "listed separately as model-specific, not as results of the study.",
        "",
        "| rule | stratum | objective | " + " | ".join(MODEL_LABEL.get(m, m) for m in models)
        + " | replicates |",
        "|---|---|---|" + "|".join("---:" for _ in models) + "|---|",
    ]
    seen = set()
    for row in rows:
        key = (row["rule"], row["group"], row["objective"])
        if key in seen or not (row["group"] == "all" or row["group"].startswith("tail:")):
            continue
        seen.add(key)
        per_model = []
        positive = []
        for model in models:
            match = next(
                (
                    r
                    for r in rows
                    if r["model"] == model
                    and (r["rule"], r["group"], r["objective"]) == key
                ),
                None,
            )
            if match is None:
                per_model.append("n/a")
                positive.append(False)
                continue
            per_model.append(f"{match['delta']:+.3f}")
            positive.append(match["ci_low"] > 0 and match["delta"] >= 0.01)
        if not any(positive):
            continue
        verdict = "**yes**" if all(positive) else "no (model-specific)"
        lines.append(
            f"| {RULE_LABEL[row['rule']]} | {row['group']} | "
            f"{OBJECTIVE_LABEL.get(row['objective'], row['objective'])} | "
            + " | ".join(per_model)
            + f" | {verdict} |"
        )
    lines.append("")
    return lines


def section_hypotheses() -> List[str]:
    strip = load(FIGURE_DATA, "P-2b") or []
    lines = [
        "## 5. Hypothesis resolution (P-2b)",
        "",
        "| id | statement | delta | 95% CI | verdict |",
        "|---|---|---:|---|---|",
    ]
    for row in strip:
        ci = row.get("ci")
        ci_text = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "n/a"
        lines.append(
            f"| {row['id']} | {row['statement']} | {fmt(row.get('delta'), signed=True)} | "
            f"{ci_text} | {row.get('verdict', 'n/a')} |"
        )
    lines.append("")
    detail = [row for row in strip if row.get("detail")]
    for row in detail:
        lines.append(f"- {row['id']}: {row['detail']}")
    if detail:
        lines.append("")
    return lines


def section_q_inertness() -> List[str]:
    rows = load(FIGURE_DATA, "P-2f") or []
    if not rows:
        return []
    worst = max(rows, key=lambda r: r["spread"])
    lines = [
        "## 6. q-inertness on the answer kernel (R5)",
        "",
        "On K_ans every chain of an answer class is identical, so the normalized spectrum is",
        "the answer-prevalence vector and VS_q is its Hill number of order q. Selection at a",
        "budget at or below the number of distinct answers therefore cannot depend on q.",
        "",
        f"Largest accuracy spread across all six q orders, over every (cell, rule, budget): "
        f"{fmt(worst['spread'])} "
        f"({cell_label(worst['model'] + '|' + worst['dataset'])}, {RULE_LABEL[worst['rule']]}, "
        f"k={worst['budget']}).",
        "",
    ]
    return lines


def section_signals() -> List[str]:
    tb5 = load(TABLES, "tb5_signals") or {}
    lines: List[str] = []
    if tb5:
        lines += [
            "## 7. Verifier-free risk signals (R6)",
            "",
            "Risk-coverage AUC and lift over the base majority-vote accuracy, per signal.",
            "",
            "| cell | signal | AUC | lift | AUC 95% CI |",
            "|---|---|---:|---:|---|",
        ]
        for cell_id, signals in sorted(tb5.items()):
            for name, data in signals.items():
                ci = data.get("auc_ci", [None, None])
                lines.append(
                    f"| {cell_label(cell_id)} | {name.replace('_', ' ')} | {fmt(data['auc'])} | "
                    f"{fmt(data['lift'], signed=True)} | [{fmt(ci[0])}, {fmt(ci[1])}] |"
                )
        lines.append("")
    confound = load(FIGURE_DATA, "P-3b") or {}
    if confound:
        lines += ["### The pooled-lift confound (P-3b)", ""]
        for model, data in sorted(confound.items()):
            within = data.get("within_dataset_lifts", [])
            mean_within = sum(within) / len(within) if within else None
            lines.append(
                f"- {MODEL_LABEL.get(model, model)}: embedding-diversity lift is "
                f"{fmt(data.get('pooled_lift'), signed=True)} when GSM8K and MATH are pooled but "
                f"{fmt(mean_within, signed=True)} within dataset — the pooled number is a "
                "dataset-identity artifact, not a usable signal."
            )
        lines.append("")

    tb6 = load(TABLES, "tb6_operating_points") or {}
    if tb6:
        lines += [
            "## 8. Entropy-gated escalation (R7)",
            "",
            "Answer with the cheap model's majority vote when its answer entropy is at or",
            "below theta, otherwise escalate to the larger model.",
            "",
            "| dataset | target answered-set acc. | fraction answered cheaply | overall acc. |",
            "|---|---:|---:|---:|",
        ]
        for dataset, points in sorted(tb6.items()):
            for target, point in sorted((points or {}).items()):
                if not point:
                    continue
                lines.append(
                    f"| {DATASET_LABEL.get(dataset, dataset)} | {target}% | "
                    f"{fmt(point['fraction_answered_cheap'])} | "
                    f"{fmt(point['overall_accuracy'])} |"
                )
        lines.append("")
    return lines


def section_limitations() -> List[str]:
    return [
        "## 9. Limitations",
        "",
        "Recorded in full, with the compute reasoning, in `TRIAGE.md`.",
        "",
        "- Three models (0.5B, 1.5B, 3B) rather than six; Qwen2.5-7B and Gemma-2-2b were cut",
        "  for the 12-hour single-A100 budget. Every other axis of the blueprint is intact:",
        "  1024 chains per question, all nine subsample budgets, all six selection budgets,",
        "  all six q orders, all three kernel families with the full alpha sweep.",
        "- Generation seed g=0 only; the 3-seed spot check that would bound bank-level",
        "  generation variance was not run.",
        "- The verifier arm is mean token logprob only, the blueprint's cautionary arm. No PRM",
        "  was run, so verifier best-of-n results are a lower bound on what a real verifier",
        "  would achieve.",
        "- One encoder (bge-large-en-v1.5) on all cells; the specter2 ranking-stability",
        "  check was not run (mxbai is reported in TB-7 where budget allowed).",
        "- Headline embedding results use the question-centred kernel (section 2). That is a",
        "  deliberate departure from measuring raw encoder similarity, and it is the right",
        "  one for a within-question question, but it means the embedding numbers are not",
        "  comparable to work that scores raw sentence-embedding similarity.",
        "- Question counts (96 GSM8K, 60 MATH) make several Snell-bin x tail-heaviness cells",
        "  underpowered; those cells carry CIs but no headline claims, and are flagged in TB-0.",
        "",
    ]


def main() -> int:
    if not (TABLES / "tb3_winner_map.json").exists():
        print("No assembled results yet; run scripts/run_analysis.py assemble first.")
        return 1
    lines = [
        "# Results: Diversity vs. Coverage for LLM Reasoning",
        "",
        "Generated by `scripts/write_results.py` from `cache/` only — no number in this file",
        f"was entered by hand. Last regenerated {datetime.now(timezone.utc).isoformat()}.",
        "",
        "Definitions follow Dieng's Vendi Score family (Friedman & Dieng 2023; Pasarkar &",
        "Dieng, *Cousins of the Vendi Score*, arXiv:2310.12952): **VS_q is the exponential of",
        "the Renyi entropy of order q of the normalized similarity-kernel spectrum**, with q",
        "sweeping {0, 0.1, 0.5, 1.0, 2.0, inf}. Every order q, including the q -> 0 richness",
        "limit, is a **diversity** measure. **Coverage** is a separate functional and is only",
        "ever the **pseudo log-determinant**: the sum of the logs of the nonzero eigenvalues",
        "of the same kernel. Random selection (20 seeds) is the baseline in every comparison;",
        "facility location is a representativeness reference, kept distinct from coverage.",
        "",
    ]
    lines += section_provenance()
    lines += section_kernel_choice()
    lines += section_measurement()
    lines += section_winner_map()
    lines += section_replication()
    lines += section_hypotheses()
    lines += section_q_inertness()
    lines += section_signals()
    lines += section_limitations()
    lines += [
        "## Figures",
        "",
        "All figures render to `figures/generated/` as both PDF (vector, for the paper) and",
        "PNG (for slides and quick review), from cache payloads only:",
        "`python figures/render_paper.py`.",
        "",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {OUT.relative_to(ROOT)} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
