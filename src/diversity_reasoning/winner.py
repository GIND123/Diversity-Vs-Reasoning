"""R-suite core: the shared selection protocol and aggregation outcomes.

For each question pool this module runs every selection objective on every
kernel variant, applies all three aggregation rules at every output budget, and
returns per-question binary outcomes. The winner map and its conditioned views
(R1-R5) are all groupings of these outcomes; the statistics live in
:func:`delta_versus_random`.

Selection pools follow the blueprint: a seeded 40-chain pool for the full
factorial, and the full 1024 pool for the core selector quartet (VS_1, VS_inf,
coverage, facility location). Greedy prefixes realize every output budget in
{2, 3, 4, 8, 16, 32} from one run per selector.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .constants import Q_ORDERS, SELECTION_BUDGETS
from .metrics import pseudo_logdet
from .pools import Pool
from .selection import facility_location_select, greedy_select
from .spectra import functionals, q_label
from .statistics import holm_adjust

FloatArray = NDArray[np.float64]

CORE_OBJECTIVES = ("vendi_1", "vendi_inf", "coverage", "facility_location")
ALPHA_GRID = (0.1, 0.25, 0.5, 0.75, 0.9)
N_RANDOM_SEEDS = 20
# Subsample draws for the 40-chain selection pool. Both the treatment arms and
# the random baseline are averaged over these, so neither carries draw noise the
# other does not.
SUBSAMPLE_SEEDS = (0, 1, 2, 3, 4)
AGGREGATION_RULES = ("majority_vote", "pass_at_k", "verifier_best")
MAX_BUDGET = max(SELECTION_BUDGETS)
# Every chain in a pool answers the same question, so the raw embedding kernel
# is dominated by that question's own content: mean top-eigenvalue share 0.94 and
# VS_1 near 1.5 on real banks, with different objectives choosing identical sets.
# Removing corpus-wide directions barely helps (that anisotropy is not the
# cause); re-expressing chains as deviations from their own question's centroid
# does. Headline embedding results therefore use the question-centred kernel,
# with the corpus-level sweep kept as the ablation (P-A4).
PRIMARY_EMBEDDING_KERNEL = "embedding_qc"


def objective_names() -> List[str]:
    return [f"vendi_{q_label(q)}" for q in Q_ORDERS] + ["coverage", "facility_location"]


def _vendi_callable(q: Any) -> Callable[[FloatArray], float]:
    def objective(subset: FloatArray) -> float:
        eigenvalues = np.linalg.eigvalsh((subset + subset.T) / (2 * subset.shape[0]))
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        return functionals(subset, q_orders=[q], eigenvalues=eigenvalues)[f"vs_{q_label(q)}"]

    return objective


def _coverage_callable() -> Callable[[FloatArray], float]:
    """Greedy coverage maximizes the raw-spectrum pseudo log-det (blueprint B3).

    On non-singular subsets (typical for K_emb) this is exactly the DPP
    log-volume. On kernels with exact duplicates (K_ans blocks) excluding the
    zero eigenvalues removes the duplication penalty, so greedy coverage is
    duplication-attracted there — an intrinsic property of the functional that
    the winner map measures rather than hides.
    """
    return lambda subset: pseudo_logdet(subset, normalize=False)


def _hill_from_spectra(eigenvalues: FloatArray, q: Any, tau: float = 1e-10) -> FloatArray:
    """Vectorized VS_q over a batch of spectra, one row per candidate.

    Mirrors ``spectra.functionals`` exactly: clip negatives, drop eigenvalues
    below the relative tolerance, renormalize, then take the Hill number of
    order q.
    """
    values = np.clip(eigenvalues, 0.0, None)
    maxima = values.max(axis=1, keepdims=True)
    kept = np.where(values > tau * maxima, values, 0.0)
    totals = kept.sum(axis=1, keepdims=True)
    probabilities = np.divide(kept, totals, out=np.zeros_like(kept), where=totals > 0)
    positive = probabilities > 0
    if isinstance(q, str) or np.isinf(q):
        return np.asarray(1.0 / np.max(probabilities, axis=1), dtype=np.float64)
    order = float(q)
    if order == 0:
        return np.asarray(positive.sum(axis=1), dtype=np.float64)
    if order == 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.where(positive, np.log(probabilities, where=positive), 0.0)
        return np.asarray(np.exp(-(probabilities * logs).sum(axis=1)), dtype=np.float64)
    powered = np.where(positive, probabilities**order, 0.0).sum(axis=1)
    return np.asarray(powered ** (1.0 / (1.0 - order)), dtype=np.float64)


def _batched_greedy(
    kernel: FloatArray,
    budget: int,
    objective_name: str,
    *,
    tau: float = 1e-10,
) -> List[int]:
    """Greedy maximization evaluating all candidates in one batched eigh call.

    Mathematically identical to :func:`selection.greedy_select` — same objective,
    same lowest-index tie-break — but it forms every candidate's bordered
    submatrix at once and hands the stack to ``numpy.linalg.eigvalsh``, so each
    step costs one batched LAPACK call rather than n Python-level ones. The
    blueprint anticipates this speedup for 1024-chain pools and requires it to be
    cross-checked against the naive implementation; see the T11 test.
    """
    size = kernel.shape[0]
    selected: List[int] = []
    is_coverage = objective_name == "coverage"
    q: Any = None
    if not is_coverage:
        suffix = objective_name.split("_", 1)[1]
        q = "inf" if suffix == "inf" else float(suffix)

    for _ in range(budget):
        remaining = np.setdiff1d(np.arange(size), np.asarray(selected, dtype=int))
        if remaining.size == 0:
            break
        depth = len(selected)
        count = remaining.size
        block = np.empty((count, depth + 1, depth + 1), dtype=np.float64)
        if depth:
            chosen = np.asarray(selected, dtype=int)
            block[:, :depth, :depth] = kernel[np.ix_(chosen, chosen)]
            border = kernel[np.ix_(chosen, remaining)].T  # [count, depth]
            block[:, :depth, depth] = border
            block[:, depth, :depth] = border
        block[:, depth, depth] = kernel[remaining, remaining]

        if is_coverage:
            # Raw-spectrum pseudo log-det, matching _coverage_callable.
            spectra = np.clip(np.linalg.eigvalsh(block), 0.0, None)
            maxima = spectra.max(axis=1, keepdims=True)
            kept = spectra > tau * maxima
            with np.errstate(divide="ignore", invalid="ignore"):
                logs = np.where(kept, np.log(spectra, where=kept), 0.0)
            scores = logs.sum(axis=1)
        else:
            spectra = np.linalg.eigvalsh(block / (depth + 1))
            scores = _hill_from_spectra(spectra, q, tau)

        scores = np.where(np.isfinite(scores), scores, -np.inf)
        best = float(scores.max())
        tolerance = 1e-12 * max(1.0, abs(best))
        winners = remaining[np.abs(scores - best) <= tolerance]
        selected.append(int(winners.min()))
    return selected


def selection_orders(
    kernel: FloatArray,
    objectives: Sequence[str],
    budget: int,
    *,
    batched_threshold: int = 128,
) -> Dict[str, List[int]]:
    """Greedy selection order (length ``budget``) per objective on one kernel.

    Pools at or above ``batched_threshold`` candidates use the batched greedy,
    which is exactly equivalent and far faster; smaller pools keep the reference
    implementation. Set ``batched_threshold`` above the pool size to force the
    reference path.
    """
    orders: Dict[str, List[int]] = {}
    size = kernel.shape[0]
    budget = min(budget, size)
    use_batched = size >= batched_threshold
    for name in objectives:
        if name == "facility_location":
            orders[name] = list(facility_location_select(kernel, budget).selected_indices)
            continue
        if use_batched:
            orders[name] = _batched_greedy(kernel, budget, name)
            continue
        if name == "coverage":
            record = greedy_select(kernel, budget, _coverage_callable(), objective_name=name)
        elif name.startswith("vendi_"):
            suffix = name.split("_", 1)[1]
            q: Any = "inf" if suffix == "inf" else float(suffix)
            record = greedy_select(kernel, budget, _vendi_callable(q), objective_name=name)
        else:
            raise ValueError(f"Unknown objective {name!r}")
        orders[name] = list(record.selected_indices)
    return orders


def random_orders(size: int, budget: int, n_seeds: int = N_RANDOM_SEEDS) -> List[List[int]]:
    """Seeded random permutation prefixes; prefix of length k is a uniform k-set."""
    budget = min(budget, size)
    return [list(np.random.default_rng(seed).permutation(size)[:budget]) for seed in range(n_seeds)]


def aggregate_outcomes(
    pool: Pool,
    indices: Sequence[int],
    budgets: Sequence[int] = SELECTION_BUDGETS,
) -> Dict[int, Dict[str, Any]]:
    """MV / pass@k / verifier outcomes for every budget prefix of one order."""
    class_ids = pool.class_ids
    correct_classes = set(int(c) for c in np.unique(class_ids[pool.correct]))
    verifier_scores = pool.mean_logprobs()
    results: Dict[int, Dict[str, Any]] = {}
    for budget in budgets:
        if budget > len(indices):
            continue
        chosen = list(indices[:budget])
        chosen_classes = [int(class_ids[i]) for i in chosen]
        counts: Dict[int, int] = {}
        for value in chosen_classes:
            counts[value] = counts.get(value, 0) + 1
        highest = max(counts.values())
        tied = sorted(c for c, n in counts.items() if n == highest)
        tie = len(tied) > 1
        if tie:
            means = {
                c: float(np.mean([verifier_scores[i] for i in chosen if int(class_ids[i]) == c]))
                for c in tied
            }
            best = max(means.values())
            prediction = min(c for c in tied if np.isclose(means[c], best))
        else:
            prediction = tied[0]
        best_chain = chosen[int(np.argmax([verifier_scores[i] for i in chosen]))]
        results[budget] = {
            "majority_vote": bool(prediction in correct_classes),
            "pass_at_k": bool(any(c in correct_classes for c in chosen_classes)),
            "verifier_best": bool(pool.correct[best_chain]),
            "tie": tie,
        }
    return results


def question_outcomes(
    pool: Pool,
    *,
    pool_size: int,
    kernels: Sequence[str],
    objectives: Optional[Sequence[str]] = None,
    subsample_seeds: Sequence[int] = SUBSAMPLE_SEEDS,
    n_random_seeds: int = N_RANDOM_SEEDS,
) -> Dict[str, Any]:
    """All (kernel, objective, budget, rule) outcomes for one question.

    ``kernels`` entries are ``answer``, ``embedding_qc``, ``embedding[:cN]``, or
    ``alpha:A[:qc|:cN]``. ``pool_size`` below the pool size draws a seeded
    subsample; anything at or above it uses the full parsed pool.

    Every arm is averaged over ``subsample_seeds``. Averaging only the random
    baseline (as an earlier version did) leaves each treatment arm carrying the
    noise of one arbitrary draw, which on small strata is large enough to flip
    the sign of the measured effect. Both arms therefore see the same draws and
    each cell reports a mean success rate in [0, 1] rather than a single bool.
    """
    if objectives is None:
        objectives = objective_names()
    if pool.size < 2:
        return {"qid": pool.qid, "skipped": "pool too small", "cells": {}}

    seeds = list(subsample_seeds) if pool_size < pool.size else [subsample_seeds[0]]
    accumulated: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
    random_draws: List[Dict[int, Dict[str, Any]]] = []
    sizes: List[int] = []

    for seed in seeds:
        base = pool.subsample(pool_size, seed) if pool_size < pool.size else list(range(pool.size))
        view_classes = pool.class_ids[base]
        n = len(base)
        sizes.append(n)

        def kernel_for(spec: str, base: List[int] = base, view: Any = view_classes) -> FloatArray:
            if spec == "answer":
                values = np.asarray([str(c) for c in view], dtype=object)
                return np.equal.outer(values, values).astype(np.float64)
            parts = spec.split(":")
            components = 0
            if parts[-1].startswith("c") and parts[-1][1:].isdigit():
                components = int(parts[-1][1:])
                parts = parts[:-1]
            if parts[0] == "embedding_qc":
                full = pool.kernel("embedding_qc", components=max(1, components))
            elif parts[0] == "embedding":
                full = pool.kernel("embedding", components=components)
            elif parts[0] == "alpha":
                family = parts[2] if len(parts) > 2 else "embedding"
                if family == "qc":
                    alpha = float(parts[1])
                    full = alpha * pool.kernel("answer") + (1 - alpha) * pool.kernel(
                        "embedding_qc", components=1
                    )
                else:
                    full = pool.kernel("mixed", float(parts[1]), components=components)
            else:
                raise ValueError(f"Unknown kernel spec {spec!r}")
            return full[np.ix_(base, base)]

        max_budget = min(MAX_BUDGET, n)
        for spec in kernels:
            kernel = kernel_for(spec)
            for name, order in selection_orders(kernel, objectives, max_budget).items():
                key = f"{spec}|{name}"
                outcomes = aggregate_outcomes(pool, [base[i] for i in order])
                bucket = accumulated.setdefault(key, {})
                for budget, values in outcomes.items():
                    slot = bucket.setdefault(budget, {rule: [] for rule in AGGREGATION_RULES})
                    for rule in AGGREGATION_RULES:
                        slot[rule].append(1.0 if values[rule] else 0.0)
        for prefix in random_orders(n, max_budget, n_random_seeds):
            random_draws.append(aggregate_outcomes(pool, [base[i] for i in prefix]))

    cells: Dict[str, Any] = {
        key: {
            budget: {rule: float(np.mean(values)) for rule, values in slot.items()}
            for budget, slot in bucket.items()
        }
        for key, bucket in accumulated.items()
    }
    cells["random"] = random_draws
    return {
        "qid": pool.qid,
        "n_pool": int(np.mean(sizes)),
        "subsample_seeds": list(seeds),
        "cells": cells,
    }


def delta_versus_random(
    treatment: Sequence[float],
    random_mean: Sequence[float],
    *,
    replicates: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired question-level bootstrap of accuracy delta vs the random baseline."""
    treated = np.asarray(list(treatment), dtype=np.float64)
    baseline = np.asarray(list(random_mean), dtype=np.float64)
    if treated.shape != baseline.shape or treated.size == 0:
        raise ValueError("Misaligned paired outcome vectors")
    delta = treated - baseline
    generator = np.random.default_rng(seed)
    draws = delta[generator.integers(0, delta.size, size=(replicates, delta.size))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    below = float((draws <= 0).mean())
    above = float((draws >= 0).mean())
    # A bootstrap p cannot resolve below 1/replicates: if no resampled mean
    # crosses zero the honest statement is "p < 1/B", not "p = 0". Report the
    # floor so nothing downstream prints an exact zero it cannot support.
    p_value = max(1.0 / replicates, min(1.0, 2.0 * min(below, above)))
    return {
        "delta": float(delta.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "p": p_value,
        "n": int(delta.size),
        "practically_null": bool(abs(delta.mean()) < 0.01),
    }


def holm_family(entries: List[Dict[str, Any]], key: str = "p") -> None:
    """In-place Holm correction over one hypothesis family."""
    adjusted = holm_adjust([entry[key] for entry in entries])
    for entry, value in zip(entries, adjusted):
        entry["p_holm"] = float(value)


def accuracy_matrix(
    outcomes: Sequence[Dict[str, Any]],
    cell_key: str,
    budget: int,
    rule: str,
) -> Tuple[List[str], FloatArray]:
    """Per-question binary outcomes for one (kernel|objective, budget, rule)."""
    qids: List[str] = []
    values: List[float] = []
    for record in outcomes:
        cell = record.get("cells", {}).get(cell_key)
        if cell is None or budget not in cell:
            continue
        qids.append(record["qid"])
        values.append(float(cell[budget][rule]))
    return qids, np.asarray(values, dtype=np.float64)


def random_matrix(
    outcomes: Sequence[Dict[str, Any]],
    budget: int,
    rule: str,
) -> Tuple[List[str], FloatArray, FloatArray]:
    """Random baseline: per-question mean over seeds plus per-seed accuracies."""
    qids: List[str] = []
    means: List[float] = []
    per_seed: List[List[float]] = []
    for record in outcomes:
        seeds = record.get("cells", {}).get("random")
        if not seeds or budget not in seeds[0]:
            continue
        qids.append(record["qid"])
        values = [float(s[budget][rule]) for s in seeds if budget in s]
        means.append(float(np.mean(values)))
        per_seed.append(values)
    seed_matrix = np.asarray(per_seed, dtype=np.float64)
    return qids, np.asarray(means, dtype=np.float64), seed_matrix
