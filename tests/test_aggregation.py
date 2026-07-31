from __future__ import annotations

import pytest

from diversity_reasoning.aggregation import majority_vote, unbiased_pass_at_k, verifier_best


def test_majority_vote_logprob_tiebreak() -> None:
    result = majority_vote(
        ["a", "b"],
        target="b",
        logprob_sums=[-10.0, -2.0],
    )
    assert result.tie
    assert result.prediction == "b"
    assert result.correct


def test_unbiased_pass_at_k_edges() -> None:
    assert unbiased_pass_at_k(10, 0, 3) == 0
    assert unbiased_pass_at_k(10, 10, 3) == 1
    assert unbiased_pass_at_k(10, 1, 1) == pytest.approx(0.1)


def test_verifier_best() -> None:
    result = verifier_best(["3", "4"], [-2.0, -1.0], target="4", verifier_name="test")
    assert result.correct
    assert result.metadata["verifier"] == "test"
