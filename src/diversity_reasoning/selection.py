"""Deterministic selectors and brute-force validation helpers."""

from __future__ import annotations

import itertools
from typing import Callable, List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .metrics import pseudo_logdet, validate_kernel, vendi_score
from .schemas import SelectionRecord

Objective = Callable[[NDArray[np.float64]], float]


def _principal(kernel: NDArray[np.float64], indices: Sequence[int]) -> NDArray[np.float64]:
    return kernel[np.ix_(indices, indices)]


def vendi_objective(q: Union[float, str]) -> Objective:
    return lambda subset: vendi_score(subset, q=q)


def coverage_objective(tau: float = 1e-10) -> Objective:
    return lambda subset: pseudo_logdet(subset, tau=tau)


def greedy_select(
    kernel: NDArray[np.float64],
    budget: int,
    objective: Objective,
    *,
    objective_name: str = "custom",
) -> SelectionRecord:
    """Greedily maximize an objective, breaking numerical ties by pool index."""
    matrix = validate_kernel(kernel)
    size = matrix.shape[0]
    if not 1 <= budget <= size:
        raise ValueError(f"budget must be in [1, {size}]")
    selected: List[int] = []
    trace: List[float] = []
    previous = 0.0
    for _ in range(budget):
        candidates: List[Tuple[float, int]] = []
        for index in range(size):
            if index in selected:
                continue
            value = objective(_principal(matrix, [*selected, index]))
            if not np.isfinite(value):
                value = float("-inf")
            candidates.append((float(value), index))
        best_value = max(value for value, _ in candidates)
        tolerance = 1e-12 * max(1.0, abs(best_value))
        best_index = min(
            index for value, index in candidates if abs(value - best_value) <= tolerance
        )
        selected.append(best_index)
        trace.append(best_value - previous)
        previous = best_value
    return SelectionRecord(
        objective=objective_name,
        selected_indices=selected,
        gain_trace=trace,
        pool_size=size,
        output_budget=budget,
    )


def brute_force_select(
    kernel: NDArray[np.float64],
    budget: int,
    objective: Objective,
) -> Tuple[int, ...]:
    """Return the lexicographically first globally optimal subset."""
    matrix = validate_kernel(kernel)
    if not 1 <= budget <= matrix.shape[0]:
        raise ValueError("Invalid budget")
    scored = [
        (float(objective(_principal(matrix, indices))), indices)
        for indices in itertools.combinations(range(matrix.shape[0]), budget)
    ]
    maximum = max(value for value, _ in scored)
    tolerance = 1e-12 * max(1.0, abs(maximum))
    return min(indices for value, indices in scored if abs(value - maximum) <= tolerance)


def facility_location_select(
    kernel: NDArray[np.float64],
    budget: int,
) -> SelectionRecord:
    """Greedy facility-location representativeness reference."""
    matrix = validate_kernel(kernel)
    size = matrix.shape[0]
    if not 1 <= budget <= size:
        raise ValueError("Invalid budget")
    selected: List[int] = []
    gains: List[float] = []
    covered = np.zeros(size, dtype=np.float64)
    for _ in range(budget):
        candidate_gains = [
            (float(np.maximum(covered, matrix[:, index]).sum() - covered.sum()), index)
            for index in range(size)
            if index not in selected
        ]
        best_gain = max(gain for gain, _ in candidate_gains)
        best = min(index for gain, index in candidate_gains if np.isclose(gain, best_gain))
        selected.append(best)
        gains.append(best_gain)
        covered = np.maximum(covered, matrix[:, best])
    return SelectionRecord("facility_location", selected, gains, size, budget)


def random_select(pool_size: int, budget: int, seed: int) -> SelectionRecord:
    if not 1 <= budget <= pool_size:
        raise ValueError("Invalid budget")
    generator = np.random.default_rng(seed)
    selected = generator.choice(pool_size, size=budget, replace=False).tolist()
    return SelectionRecord("random", selected, [], pool_size, budget, seed)
