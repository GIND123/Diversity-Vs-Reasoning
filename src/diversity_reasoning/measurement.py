"""D1 measurement experiments: sensitivity of the two spectral functionals.

E1 rare modes, E2 redundancy, E3 sample size, E4 dimensionality, E5 the two
log functionals. Synthetic constructions use exact block-kernel spectra
(harness identity T1/T4); real constructions consume :class:`~.pools.Pool`.
Every function returns a JSON-serializable payload.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .constants import POOL_BUDGETS, Q_ORDERS
from .kernels import embedding_kernel, l2_normalize
from .pools import Pool
from .spectra import functionals, functionals_from_counts, q_label
from .strata import tail_heaviness

FloatArray = NDArray[np.float64]

FUNCTIONAL_KEYS = [f"vs_{q_label(q)}" for q in Q_ORDERS] + ["pseudo_logdet"]

# Real-pool measurements use the question-centred embedding space. Every chain
# in a pool answers the same question, so in the raw space that question's own
# content carries ~94% of the spectral mass and pins VS_1 near 1.5 whatever the
# chains actually contain. Removing corpus-wide directions barely helps
# (measured: VS_1 1.49 -> 2.15); removing the pool's own centroid does
# (VS_1 -> 9.29). See anisotropy_diagnostics and TRIAGE.md.
ANISOTROPY_COMPONENTS = 1

E1_PREVALENCES = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0)
E2_DUPLICATION_RATES = (0.0, 0.10, 0.25, 0.50, 0.75)
E3_SAMPLE_SIZES = (5, 10, 20, 40, 80, 160, 320, 640, 1024)
E4_DIMENSIONS = (8, 16, 32, 64, 128, 256, 512, 1024)


def _family_spec(family: str) -> str:
    """Map a kernel family name onto the question-centred embedding variant."""
    return "embedding_qc" if family == "embedding" else family


def _summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    low, high = np.quantile(array, [0.025, 0.975])
    return {"mean": float(array.mean()), "low": float(low), "high": float(high)}


# --------------------------------------------------------------------------
# E1: rare modes
# --------------------------------------------------------------------------


def e1_synthetic(
    *,
    n_classes: int = 6,
    n_items: int = 240,
    replicates: int = 50,
    seed: int = 0,
) -> Dict[str, Any]:
    """Six-class pools with one class's prevalence swept toward zero."""
    generator = np.random.default_rng(seed)
    curves: Dict[str, List[Dict[str, float]]] = {key: [] for key in FUNCTIONAL_KEYS}
    for prevalence in E1_PREVALENCES:
        if prevalence == 0.0:
            probabilities = np.full(n_classes - 1, 1.0 / (n_classes - 1))
        else:
            probabilities = np.concatenate(
                [[prevalence], np.full(n_classes - 1, (1 - prevalence) / (n_classes - 1))]
            )
        samples: Dict[str, List[float]] = {key: [] for key in FUNCTIONAL_KEYS}
        for _ in range(replicates):
            counts = generator.multinomial(n_items, probabilities)
            values = functionals_from_counts([int(c) for c in counts if c > 0])
            for key in FUNCTIONAL_KEYS:
                samples[key].append(values[key])
        for key in FUNCTIONAL_KEYS:
            curves[key].append({"x": prevalence, **_summary(samples[key])})
    return {"prevalences": list(E1_PREVALENCES), "curves": curves, "replicates": replicates}


def e1_real_deletion(pools: Sequence[Pool]) -> Dict[str, Any]:
    """Delete the correct chains on minority/tail questions; delta per functional."""
    deltas: Dict[str, List[float]] = {key: [] for key in FUNCTIONAL_KEYS}
    used = 0
    for pool in pools:
        if tail_heaviness(pool) not in {"minority", "tail"}:
            continue
        counts = pool.answer_counts()
        correct_classes = {int(c) for c in np.unique(pool.class_ids[pool.correct])}
        before = functionals_from_counts(list(counts.values()))
        remaining = [count for cid, count in counts.items() if cid not in correct_classes]
        if not remaining:
            continue
        after = functionals_from_counts(remaining)
        for key in FUNCTIONAL_KEYS:
            deltas[key].append(after[key] - before[key])
        used += 1
    return {
        "n_questions": used,
        "delta": {key: _summary(values) for key, values in deltas.items() if values},
    }


# --------------------------------------------------------------------------
# E2: redundancy
# --------------------------------------------------------------------------


def e2_duplication(
    pools: Sequence[Pool],
    *,
    base_size: int = 200,
    n_questions: int = 12,
    seed: int = 0,
) -> Dict[str, Any]:
    """Duplicate real chains at fixed rates, in both duplication regimes.

    Harness test T7 pins two distinct facts, and conflating them misstates what
    coverage does:

    - **uniform** duplication (every chain copied the same number of times)
      leaves the normalized spectrum untouched, so coverage is exactly
      invariant and so is every VS_q;
    - **skewed** duplication (a random subset copied) concentrates mass on the
      copied chains, so VS_q falls — and coverage is *not* invariant either,
      because the normalized spectrum genuinely changes.

    Reporting only the skewed arm would contradict T7; reporting only the
    uniform arm would overstate coverage's robustness. Both are returned.
    """
    generator = np.random.default_rng(seed)
    eligible = [pool for pool in pools if pool.embeddings is not None and pool.size >= base_size]
    chosen = eligible[:n_questions]
    curves: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        "skewed": {key: [] for key in FUNCTIONAL_KEYS},
        "uniform": {key: [] for key in FUNCTIONAL_KEYS},
    }
    for rate in E2_DUPLICATION_RATES:
        samples: Dict[str, Dict[str, List[float]]] = {
            "skewed": {key: [] for key in FUNCTIONAL_KEYS},
            "uniform": {key: [] for key in FUNCTIONAL_KEYS},
        }
        for pool in chosen:
            base = pool.subsample(base_size, seed)
            original = pool.question_centered_embeddings(ANISOTROPY_COMPONENTS)[base]
            for mode in ("skewed", "uniform"):
                embeddings = original
                if rate > 0:
                    if mode == "skewed":
                        extra = int(round(base_size * rate / (1 - rate)))
                        rows = generator.integers(0, base_size, size=extra)
                        embeddings = np.concatenate([original, original[rows]])
                    else:
                        # Uniform: whole-pool copies, so every chain gains the
                        # same multiplicity and the normalized spectrum is fixed.
                        copies = 1 + int(round(rate / max(1e-9, 1 - rate)))
                        embeddings = np.tile(original, (copies, 1))
                kernel, _ = embedding_kernel(embeddings)
                values = functionals(kernel)
                for key in FUNCTIONAL_KEYS:
                    samples[mode][key].append(values[key])
        for mode in ("skewed", "uniform"):
            for key in FUNCTIONAL_KEYS:
                curves[mode][key].append({"x": rate, **_summary(samples[mode][key])})
    near_duplicate_rates = []
    for pool in chosen:
        kernel = pool.kernel("embedding_qc", components=1)
        upper = kernel[np.triu_indices(kernel.shape[0], k=1)]
        near_duplicate_rates.append(float((upper > 0.98).mean()))
    return {
        "rates": list(E2_DUPLICATION_RATES),
        "curves": curves["skewed"],
        "curves_by_mode": curves,
        "n_questions": len(chosen),
        "near_duplicate_pair_rate": _summary(near_duplicate_rates) if near_duplicate_rates else {},
    }


# --------------------------------------------------------------------------
# E3: sample size
# --------------------------------------------------------------------------


def e3_sample_size(
    pools: Sequence[Pool],
    *,
    n_questions: int = 24,
    kernel_family: str = "embedding",
    seed: int = 0,
) -> Dict[str, Any]:
    """Bias, sd, and ranking stability of each functional against pool size."""
    chosen = [pool for pool in pools if pool.embeddings is not None][:n_questions]
    if not chosen:
        return {"skipped": "no embedded pools"}
    full_values: Dict[str, List[float]] = {key: [] for key in FUNCTIONAL_KEYS}
    for pool in chosen:
        values = functionals(pool.kernel(_family_spec(kernel_family), components=1))
        for key in FUNCTIONAL_KEYS:
            full_values[key].append(values[key])

    from scipy.stats import kendalltau

    result_rows: List[Dict[str, Any]] = []
    for n in E3_SAMPLE_SIZES:
        usable = [pool for pool in chosen if pool.size >= n]
        if len(usable) < 5:
            continue
        n_seeds = 100 if n <= 160 else 40
        per_seed_values: Dict[str, List[List[float]]] = {key: [] for key in FUNCTIONAL_KEYS}
        for sample_seed in range(n_seeds):
            row: Dict[str, List[float]] = {key: [] for key in FUNCTIONAL_KEYS}
            for pool in usable:
                indices = pool.subsample(n, seed + sample_seed)
                embeddings = pool.question_centered_embeddings(ANISOTROPY_COMPONENTS)[indices]
                kernel, _ = embedding_kernel(embeddings)
                values = functionals(kernel)
                for key in FUNCTIONAL_KEYS:
                    row[key].append(values[key])
            for key in FUNCTIONAL_KEYS:
                per_seed_values[key].append(row[key])
        for key in FUNCTIONAL_KEYS:
            matrix = np.asarray(per_seed_values[key])  # [seeds, questions]
            reference = np.asarray([full_values[key][chosen.index(pool)] for pool in usable])
            with np.errstate(divide="ignore", invalid="ignore"):
                bias = np.nanmean((matrix.mean(axis=0) - reference) / np.abs(reference))
            sd = float(matrix.std(axis=0).mean())
            taus = [kendalltau(matrix[s], reference).statistic for s in range(matrix.shape[0])]
            result_rows.append(
                {
                    "n": n,
                    "functional": key,
                    "relative_bias": float(bias),
                    "sd": sd,
                    "kendall_tau": float(np.nanmean(taus)),
                    "n_seeds": n_seeds,
                    "n_questions": len(usable),
                }
            )
    n_star: Dict[str, Optional[int]] = {}
    for key in FUNCTIONAL_KEYS:
        qualifying = [
            r["n"] for r in result_rows if r["functional"] == key and r["kendall_tau"] >= 0.9
        ]
        n_star[key] = min(qualifying) if qualifying else None
    return {"rows": result_rows, "n_star_tau_0.9": n_star, "kernel": kernel_family}


# --------------------------------------------------------------------------
# E4: dimensionality
# --------------------------------------------------------------------------


def e4_dimensionality(
    pools: Sequence[Pool],
    *,
    fit_fraction: float = 0.2,
    seed: int = 0,
    max_pairs: int = 2000,
) -> Dict[str, Any]:
    """PCA d sweep: functionals, ranking stability, and the answer cosine gap."""
    embedded = [pool for pool in pools if pool.embeddings is not None]
    if not embedded:
        return {"skipped": "no embedded pools"}
    generator = np.random.default_rng(seed)
    fit_rows = []
    for pool in embedded:
        count = max(1, int(pool.size * fit_fraction))
        indices = generator.choice(pool.size, size=count, replace=False)
        fit_rows.append(pool.question_centered_embeddings(ANISOTROPY_COMPONENTS)[indices])
    fit = np.concatenate(fit_rows)
    mean = fit.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(fit - mean, full_matrices=False)

    from scipy.stats import kendalltau

    full_dimension = fit.shape[1]
    dimensions = [d for d in E4_DIMENSIONS if d <= full_dimension]
    per_dimension: Dict[int, Dict[str, List[float]]] = {}
    gaps: Dict[int, float] = {}
    for d in dimensions:
        basis = components[:d]
        values_by_key: Dict[str, List[float]] = {key: [] for key in FUNCTIONAL_KEYS}
        same_cos: List[float] = []
        diff_cos: List[float] = []
        for pool in embedded:
            projected = (pool.question_centered_embeddings(ANISOTROPY_COMPONENTS) - mean) @ basis.T
            norms = np.linalg.norm(projected, axis=1)
            keep = norms > 1e-12
            if keep.sum() < 2:
                continue
            projected = l2_normalize(projected[keep])
            kernel, _ = embedding_kernel(projected)
            values = functionals(kernel)
            for key in FUNCTIONAL_KEYS:
                values_by_key[key].append(values[key])
            classes = pool.class_ids[keep]
            n = projected.shape[0]
            pairs = generator.integers(0, n, size=(max_pairs, 2))
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            cosines = np.einsum("ij,ij->i", projected[pairs[:, 0]], projected[pairs[:, 1]])
            same = classes[pairs[:, 0]] == classes[pairs[:, 1]]
            if same.any() and (~same).any():
                same_cos.append(float(cosines[same].mean()))
                diff_cos.append(float(cosines[~same].mean()))
        per_dimension[d] = values_by_key
        gaps[d] = float(np.mean(same_cos) - np.mean(diff_cos)) if same_cos else float("nan")

    reference = per_dimension[dimensions[-1]]
    rows: List[Dict[str, Any]] = []
    for d in dimensions:
        for key in FUNCTIONAL_KEYS:
            current = per_dimension[d][key]
            tau = kendalltau(current, reference[key][: len(current)]).statistic
            rows.append(
                {
                    "d": d,
                    "functional": key,
                    "mean": float(np.mean(current)),
                    "kendall_tau_vs_full": float(tau) if tau is not None else float("nan"),
                }
            )
    return {
        "dimensions": dimensions,
        "rows": rows,
        "answer_cosine_gap": {str(d): gaps[d] for d in dimensions},
        "n_questions": len(embedded),
    }


# --------------------------------------------------------------------------
# E5: the two log functionals
# --------------------------------------------------------------------------


def budget_spectra(
    pool: Pool,
    *,
    kernel_families: Sequence[str] = ("embedding", "answer"),
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Functionals per (kernel family, budget) via seeded subsampling."""
    records: List[Dict[str, Any]] = []
    for family in kernel_families:
        if family == "embedding" and pool.embeddings is None:
            continue
        for budget in POOL_BUDGETS:
            if budget > pool.size:
                continue
            indices = pool.subsample(budget, seed) if budget < pool.size else list(range(pool.size))
            if family == "embedding":
                kernel, _ = embedding_kernel(
                    pool.question_centered_embeddings(ANISOTROPY_COMPONENTS)[indices]
                )
                values = functionals(kernel)
            else:
                classes = pool.class_ids[indices]
                _, counts = np.unique(classes, return_counts=True)
                values = functionals_from_counts(counts)
            records.append({"qid": pool.qid, "kernel": family, "budget": budget, **values})
    return records


def anisotropy_diagnostics(
    pools: Sequence[Pool],
    *,
    components_levels: Sequence[int] = (0, 1, 2, 3),
    pool_size: int = 40,
    budget: int = 8,
    n_questions: int = 40,
    seed: int = 0,
) -> Dict[str, Any]:
    """P-A4: what common embedding directions do to the kernel and the selectors.

    Reports, per anisotropy level c, the share of spectral mass held by the top
    eigenvalue, the mean VS_1, and how often two different objectives choose the
    *identical* set. When one direction dominates, every pool looks rank-1 and
    the selectors become indistinguishable, so the winner map has nothing to
    measure. Set agreement is the honest statistic here: outcome-vector counts
    saturate as soon as two selectors differ on any single question.
    """
    from itertools import combinations

    from .winner import selection_orders

    embedded = [pool for pool in pools if pool.embeddings is not None][:n_questions]
    if not embedded:
        return {"skipped": "no embedded pools"}
    objectives = ["vendi_1", "vendi_2", "vendi_inf", "coverage", "facility_location"]
    rows: List[Dict[str, Any]] = []
    # Negative levels label the question-centred arm: -1 removes this question's
    # own centroid and leading direction rather than a corpus-wide one.
    for components in list(components_levels) + [-1]:
        shares: List[float] = []
        vs1: List[float] = []
        agreements: List[float] = []
        for pool in embedded:
            if pool.corpus_directions is None and components > 0:
                continue
            base = pool.subsample(min(pool_size, pool.size), seed)
            if components < 0:
                full = pool.kernel("embedding_qc", components=1)
            else:
                full = pool.kernel("embedding", components=components)
            kernel = full[np.ix_(base, base)]
            eigenvalues = np.clip(np.linalg.eigvalsh(kernel / kernel.shape[0]), 0.0, None)
            shares.append(float(eigenvalues.max() / eigenvalues.sum()))
            vs1.append(functionals(kernel, q_orders=[1.0])["vs_1"])
            orders = selection_orders(kernel, objectives, min(budget, kernel.shape[0]))
            sets = {name: frozenset(order) for name, order in orders.items()}
            pairs = list(combinations(objectives, 2))
            agreements.append(
                float(np.mean([sets[a] == sets[b] for a, b in pairs])) if pairs else 0.0
            )
        if not shares:
            continue
        rows.append(
            {
                "components": components,
                "scope": "question" if components < 0 else "corpus",
                "label": "question-centred" if components < 0 else f"corpus c={components}",
                "top_eigenvalue_share": float(np.mean(shares)),
                "mean_vs_1": float(np.mean(vs1)),
                "identical_selection_rate": float(np.mean(agreements)),
                "n_questions": len(shares),
            }
        )
    return {"rows": rows, "budget": budget, "pool_size": pool_size}


def encoder_stability(
    pools_primary: Sequence[Pool],
    pools_alternate: Sequence[Pool],
) -> Dict[str, Any]:
    """TB-7: do the functionals rank questions the same way under two encoders?

    Both sequences must be the same questions in the same order, embedded with
    different encoders.
    """
    from scipy.stats import kendalltau

    paired = [
        (primary, alternate)
        for primary, alternate in zip(pools_primary, pools_alternate)
        if primary.qid == alternate.qid
        and primary.embeddings is not None
        and alternate.embeddings is not None
    ]
    if len(paired) < 5:
        return {"skipped": "fewer than five paired embedded pools"}
    values: Dict[str, Dict[str, List[float]]] = {
        key: {"primary": [], "alternate": []} for key in FUNCTIONAL_KEYS
    }
    for primary, alternate in paired:
        primary_values = functionals(primary.kernel("embedding_qc", components=1))
        alternate_values = functionals(alternate.kernel("embedding_qc", components=1))
        for key in FUNCTIONAL_KEYS:
            values[key]["primary"].append(primary_values[key])
            values[key]["alternate"].append(alternate_values[key])
    return {
        "n_questions": len(paired),
        "kendall_tau": {
            key: float(kendalltau(data["primary"], data["alternate"]).statistic)
            for key, data in values.items()
        },
    }


def _is_constant(values: FloatArray, relative_tolerance: float = 1e-10) -> bool:
    """Constant up to float noise, which is what makes a correlation undefined."""
    array = np.asarray(values, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(array))))
    return bool(np.ptp(array) <= relative_tolerance * scale)


def e5_two_log_functionals(spectra_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Within-budget and pooled correlations, plus the reversal-onset curve."""
    from scipy.stats import pearsonr, spearmanr

    payload: Dict[str, Any] = {"correlations": [], "reversal": {}, "scatter": {}}
    kernels = sorted({row["kernel"] for row in spectra_rows})
    for kernel in kernels:
        rows = [row for row in spectra_rows if row["kernel"] == kernel]
        budgets = sorted({row["budget"] for row in rows})
        for q in Q_ORDERS:
            key = f"vs_{q_label(q)}"
            within: List[float] = []
            within_spearman: List[float] = []
            degenerate_budgets = 0
            for budget in budgets:
                subset = [row for row in rows if row["budget"] == budget]
                x = np.log([max(row[key], 1e-12) for row in subset])
                y = np.asarray([row["pseudo_logdet"] for row in subset])
                if len(subset) < 5 or _is_constant(y):
                    continue
                if _is_constant(x):
                    # VS_q is identical across every pool at this budget. On
                    # K_emb this is exactly what VS_0 does: distinct chains give
                    # a full-rank Gram matrix, so richness saturates at the
                    # budget and carries no per-question information. The
                    # correlation is undefined, not zero.
                    degenerate_budgets += 1
                    continue
                within.append(float(pearsonr(x, y).statistic))
                within_spearman.append(float(spearmanr(x, y).statistic))
            x_all = np.log([max(row[key], 1e-12) for row in rows])
            y_all = np.asarray([row["pseudo_logdet"] for row in rows])
            eps_all = np.asarray([row["logdet_eps1"] for row in rows])
            pooled = float(pearsonr(x_all, y_all).statistic) if np.std(x_all) > 0 else float("nan")
            pooled_eps = (
                float(pearsonr(x_all, eps_all).statistic) if np.std(x_all) > 0 else float("nan")
            )
            payload["correlations"].append(
                {
                    "kernel": kernel,
                    "q": q_label(q),
                    "within_budget_mean": float(np.mean(within)) if within else None,
                    "within_budget_spearman": (
                        float(np.mean(within_spearman)) if within_spearman else None
                    ),
                    "within_budget_n": len(within),
                    "degenerate_budgets": degenerate_budgets,
                    "note": (
                        "constant within every budget: richness saturates at the budget"
                        if degenerate_budgets and not within
                        else ""
                    ),
                    "pooled": pooled,
                    "pooled_eps1": pooled_eps,
                }
            )
        # Reversal onset on VS_1, the family's central member.
        x_all = np.log([max(row["vs_1"], 1e-12) for row in rows])
        y_all = np.asarray([row["pseudo_logdet"] for row in rows])
        lam = np.log10([max(row["lambda_min"], 1e-300) for row in rows])
        deciles = np.quantile(lam, np.linspace(0, 1, 11))
        onset_rows = []
        for low, high in zip(deciles[:-1], deciles[1:]):
            mask = (lam >= low) & (lam <= high)
            if mask.sum() >= 5 and np.std(x_all[mask]) > 0 and np.std(y_all[mask]) > 0:
                onset_rows.append(
                    {
                        "lambda_min_decile_mid": float((low + high) / 2),
                        "pearson": float(pearsonr(x_all[mask], y_all[mask]).statistic),
                        "n": int(mask.sum()),
                    }
                )
        flip = None
        for row in onset_rows:
            if row["pearson"] < 0:
                flip = row["lambda_min_decile_mid"]
                break
        payload["reversal"][kernel] = {"curve": onset_rows, "flip_log10_lambda_min": flip}
        step = max(1, len(rows) // 400)
        payload["scatter"][kernel] = [
            {
                "log_vs_1": float(np.log(max(row["vs_1"], 1e-12))),
                "pseudo_logdet": float(row["pseudo_logdet"]),
                "log10_lambda_min": float(np.log10(max(row["lambda_min"], 1e-300))),
                "budget": row["budget"],
            }
            for row in rows[::step]
        ]
    return payload
