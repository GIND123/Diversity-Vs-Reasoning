"""Aggregation rules applied symmetrically to selected reasoning chains."""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional, Sequence

from .schemas import AggregationResult


def majority_vote(
    answers: Sequence[str],
    *,
    target: Optional[str] = None,
    logprob_sums: Optional[Sequence[float]] = None,
) -> AggregationResult:
    if not answers:
        raise ValueError("answers cannot be empty")
    counts = Counter(answers)
    highest = max(counts.values())
    tied = sorted(answer for answer, count in counts.items() if count == highest)
    tie = len(tied) > 1
    if tie and logprob_sums is not None:
        if len(logprob_sums) != len(answers):
            raise ValueError("logprob_sums must align with answers")
        means = {
            answer: sum(
                score for candidate, score in zip(answers, logprob_sums) if candidate == answer
            )
            / counts[answer]
            for answer in tied
        }
        best = max(means.values())
        prediction = min(answer for answer in tied if math.isclose(means[answer], best))
    else:
        prediction = tied[0]
    return AggregationResult(
        rule="majority_vote",
        prediction=prediction,
        correct=None if target is None else prediction == target,
        tie=tie,
        metadata={"vote_count": counts[prediction]},
    )


def pass_at_k(answers: Sequence[str], target: str) -> AggregationResult:
    passed = target in answers
    return AggregationResult(
        rule="pass_at_k",
        prediction=target if passed else None,
        correct=passed,
        metadata={"k": len(answers)},
    )


def unbiased_pass_at_k(pool_size: int, correct_count: int, k: int) -> float:
    """The standard unbiased pass@k pool estimator."""
    if not 0 <= correct_count <= pool_size or not 1 <= k <= pool_size:
        raise ValueError("Invalid pass@k arguments")
    if pool_size - correct_count < k:
        return 1.0
    return 1.0 - math.comb(pool_size - correct_count, k) / math.comb(pool_size, k)


def verifier_best(
    answers: Sequence[str],
    scores: Sequence[float],
    *,
    target: Optional[str] = None,
    verifier_name: str,
) -> AggregationResult:
    if not answers or len(answers) != len(scores):
        raise ValueError("answers and scores must be nonempty and aligned")
    best_score = max(scores)
    index = min(index for index, score in enumerate(scores) if score == best_score)
    prediction = answers[index]
    return AggregationResult(
        rule="verifier_best",
        prediction=prediction,
        correct=None if target is None else prediction == target,
        metadata={"verifier": verifier_name, "score": best_score},
    )
