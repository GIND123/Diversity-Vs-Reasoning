"""D3 payoff: verifier-free risk signals (R6) and entropy-gated escalation (R7).

Signals are computed on the full 1024-chain bank of each question; the decision
they score is whether the full-pool majority vote is correct. Risk-coverage
curves answer the most-confident fraction f of questions and measure accuracy
on that answered set.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .measurement import ANISOTROPY_COMPONENTS
from .pools import Pool
from .spectra import answer_entropy, functionals

SIGNAL_NAMES = ("answer_entropy", "vote_margin", "mean_logprob", "embedding_vs1")


def majority_class(pool: Pool) -> int:
    counts = pool.answer_counts()
    highest = max(counts.values())
    return min(cid for cid, count in counts.items() if count == highest)


def question_signals(pool: Pool) -> Dict[str, Any]:
    """Confidence signals for one question, higher = more confident."""
    counts = pool.answer_counts()
    ordered = sorted(counts.values(), reverse=True)
    margin = (ordered[0] - (ordered[1] if len(ordered) > 1 else 0)) / max(1, pool.size)
    entropy = answer_entropy(list(counts.values()))["entropy"]
    verifier = float(pool.mean_logprobs().max())
    embedding_vs1: Optional[float] = None
    if pool.embeddings is not None:
        embedding_vs1 = functionals(
            pool.kernel("embedding_qc", components=ANISOTROPY_COMPONENTS), q_orders=[1.0]
        )["vs_1"]
    mv_class = majority_class(pool)
    correct_classes = {int(c) for c in np.unique(pool.class_ids[pool.correct])}
    return {
        "qid": pool.qid,
        "answer_entropy": entropy,
        "vote_margin": float(margin),
        "mean_logprob": verifier,
        "embedding_vs1": embedding_vs1,
        "mv_correct": bool(mv_class in correct_classes),
        "mean_chain_tokens": float(pool.token_counts.mean()) if pool.size else 0.0,
    }


def risk_coverage(
    confidences: Sequence[float],
    outcomes: Sequence[bool],
    grid: int = 50,
) -> Dict[str, Any]:
    """Accuracy on the answered set as coverage grows from most confident."""
    order = np.argsort(-np.asarray(confidences, dtype=np.float64), kind="stable")
    correct = np.asarray(outcomes, dtype=np.float64)[order]
    n = correct.size
    curve: List[Dict[str, float]] = []
    for fraction in np.linspace(1 / n, 1.0, min(grid, n)):
        kept = max(1, int(round(fraction * n)))
        curve.append({"coverage": kept / n, "accuracy": float(correct[:kept].mean())})
    coverages = np.asarray([point["coverage"] for point in curve])
    accuracies = np.asarray([point["accuracy"] for point in curve])
    area = float(np.sum((accuracies[1:] + accuracies[:-1]) / 2 * np.diff(coverages)))
    auc = area / float(coverages[-1] - coverages[0] + 1e-12)
    base = float(correct.mean())
    return {"curve": curve, "auc": auc, "base_accuracy": base, "lift": auc - base}


def r6_signal_shootout(signal_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Risk-coverage AUC and lift per signal; bootstrap CI over questions."""
    generator = np.random.default_rng(0)
    outcomes = [bool(row["mv_correct"]) for row in signal_rows]
    results: Dict[str, Any] = {"signals": {}, "n_questions": len(signal_rows)}
    for name in SIGNAL_NAMES:
        raw = [row.get(name) for row in signal_rows]
        if any(value is None for value in raw):
            continue
        values = [float(value) for value in raw if value is not None]
        # Entropy is an uncertainty: negate so higher = more confident.
        confidences = [-v if name in {"answer_entropy", "embedding_vs1"} else v for v in values]
        primary = risk_coverage(confidences, outcomes)
        aucs = []
        for _ in range(200):
            draw = generator.integers(0, len(outcomes), size=len(outcomes))
            aucs.append(
                risk_coverage([confidences[i] for i in draw], [outcomes[i] for i in draw], grid=25)[
                    "auc"
                ]
            )
        low, high = np.quantile(aucs, [0.025, 0.975])
        results["signals"][name] = {
            **primary,
            "auc_ci": [float(low), float(high)],
        }
    return results


def r7_escalation(
    cheap_rows: Sequence[Dict[str, Any]],
    expensive_rows: Sequence[Dict[str, Any]],
    *,
    chains_per_question: int = 1024,
) -> Dict[str, Any]:
    """Entropy-gated escalation from the cheap model to the expensive one."""
    cheap = {row["qid"]: row for row in cheap_rows}
    expensive = {row["qid"]: row for row in expensive_rows}
    shared = sorted(set(cheap) & set(expensive))
    if not shared:
        return {"skipped": "no shared questions"}
    entropies = np.asarray([cheap[qid]["answer_entropy"] for qid in shared])
    cheap_correct = np.asarray([cheap[qid]["mv_correct"] for qid in shared], dtype=bool)
    expensive_correct = np.asarray([expensive[qid]["mv_correct"] for qid in shared], dtype=bool)
    cheap_cost = np.asarray([cheap[qid]["mean_chain_tokens"] for qid in shared])
    expensive_cost = np.asarray([expensive[qid]["mean_chain_tokens"] for qid in shared])

    thresholds = np.quantile(entropies, np.linspace(0, 1, 41))
    curve: List[Dict[str, float]] = []
    for theta in thresholds:
        answered = entropies <= theta
        escalated = ~answered
        overall = np.where(answered, cheap_correct, expensive_correct)
        tokens = (cheap_cost.sum() + expensive_cost[escalated].sum()) * chains_per_question
        curve.append(
            {
                "theta": float(theta),
                "fraction_answered_cheap": float(answered.mean()),
                "answered_set_accuracy": (
                    float(cheap_correct[answered].mean()) if answered.any() else 1.0
                ),
                "overall_accuracy": float(overall.mean()),
                "total_generated_tokens": float(tokens),
            }
        )
    operating_points = {}
    for target in (0.90, 0.95, 0.99):
        qualifying = [
            point
            for point in curve
            if point["answered_set_accuracy"] >= target and point["fraction_answered_cheap"] > 0
        ]
        best = max(qualifying, key=lambda p: p["fraction_answered_cheap"]) if qualifying else None
        operating_points[f"{int(target * 100)}"] = best
    return {
        "n_questions": len(shared),
        "cheap_accuracy": float(cheap_correct.mean()),
        "expensive_accuracy": float(expensive_correct.mean()),
        "curve": curve,
        "operating_points": operating_points,
    }
