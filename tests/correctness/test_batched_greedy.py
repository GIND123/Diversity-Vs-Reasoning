"""T11: the batched greedy must equal the reference greedy exactly.

The blueprint permits a faster greedy over 1024-chain pools provided it is
cross-checked against the naive implementation. These are the cross-checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from diversity_reasoning.metrics import block_kernel
from diversity_reasoning.winner import objective_names, selection_orders

pytestmark = pytest.mark.correctness

REFERENCE = 10**9  # threshold above any pool size, forcing the reference path


def _random_kernel(n: int, d: int, seed: int) -> np.ndarray:
    rows = np.random.default_rng(seed).normal(size=(n, d))
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    kernel = rows @ rows.T
    kernel = (1.0 + kernel) / 2.0
    np.fill_diagonal(kernel, 1.0)
    return (kernel + kernel.T) / 2.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_t11_batched_matches_reference_on_embedding_kernels(seed: int) -> None:
    kernel = _random_kernel(48, 16, seed)
    budget = 8
    reference = selection_orders(kernel, objective_names(), budget, batched_threshold=REFERENCE)
    batched = selection_orders(kernel, objective_names(), budget, batched_threshold=1)
    assert reference == batched


@pytest.mark.parametrize("multiplicities", [(6, 4, 2), (10, 1, 1, 1), (3, 3, 3, 3)])
def test_t11_batched_matches_reference_on_block_kernels(multiplicities) -> None:
    """Block kernels have exact ties, so the tie-break must match too."""
    kernel = block_kernel(multiplicities)
    budget = min(4, kernel.shape[0])
    reference = selection_orders(kernel, objective_names(), budget, batched_threshold=REFERENCE)
    batched = selection_orders(kernel, objective_names(), budget, batched_threshold=1)
    assert reference == batched


def test_t11_batched_matches_reference_at_larger_budget() -> None:
    kernel = _random_kernel(64, 24, seed=7)
    reference = selection_orders(
        kernel, ["vendi_1", "vendi_inf", "coverage"], 16, batched_threshold=REFERENCE
    )
    batched = selection_orders(
        kernel, ["vendi_1", "vendi_inf", "coverage"], 16, batched_threshold=1
    )
    assert reference == batched


def test_batched_path_actually_engages_on_large_pools() -> None:
    """Guards the threshold wiring: a 1024-pool must take the fast path."""
    from diversity_reasoning import winner

    kernel = _random_kernel(200, 16, seed=11)
    calls = {"n": 0}
    original = winner._batched_greedy

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    winner._batched_greedy = counting  # type: ignore[assignment]
    try:
        selection_orders(kernel, ["vendi_1", "coverage"], 4)
    finally:
        winner._batched_greedy = original  # type: ignore[assignment]
    assert calls["n"] == 2
