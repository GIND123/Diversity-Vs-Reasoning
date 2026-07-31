"""Blocking selector checks T6 and T8."""

from __future__ import annotations

import numpy as np
import pytest

from diversity_reasoning.artifacts import subsample_indices
from diversity_reasoning.selection import (
    brute_force_select,
    coverage_objective,
    greedy_select,
    random_select,
    vendi_objective,
)

pytestmark = pytest.mark.correctness


@pytest.mark.parametrize(
    ("name", "objective"),
    [
        ("vendi_1", vendi_objective(1.0)),
        ("vendi_inf", vendi_objective(float("inf"))),
        ("pseudo_logdet", coverage_objective()),
    ],
)
def test_t6_greedy_matches_brute_force_on_planted_pool(name: str, objective: object) -> None:
    kernel = np.eye(6)
    greedy = greedy_select(kernel, 4, objective, objective_name=name)  # type: ignore[arg-type]
    optimum = brute_force_select(kernel, 4, objective)  # type: ignore[arg-type]
    assert tuple(sorted(greedy.selected_indices)) == optimum


def test_t8_subsampling_and_selection_are_deterministic() -> None:
    first = subsample_indices(1024, 64, seed=932)
    second = subsample_indices(1024, 64, seed=932)
    assert first == second
    first_random = random_select(100, 16, seed=12)
    second_random = random_select(100, 16, seed=12)
    assert first_random == second_random
