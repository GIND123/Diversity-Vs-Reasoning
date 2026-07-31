from __future__ import annotations

import numpy as np

from diversity_reasoning.statistics import holm_adjust, paired_bootstrap_delta


def test_paired_bootstrap_delta_is_seeded() -> None:
    treatment = [1, 1, 0, 1]
    baseline = [0, 1, 0, 0]
    first = paired_bootstrap_delta(treatment, baseline, replicates=200, seed=3)
    second = paired_bootstrap_delta(treatment, baseline, replicates=200, seed=3)
    assert first == second
    assert first[0] == 0.5


def test_holm_adjustment() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
    assert np.all(adjusted >= np.asarray([0.01, 0.04, 0.03]))
