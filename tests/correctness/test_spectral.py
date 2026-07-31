"""Blocking spectral checks T1-T5, T7, and legacy revalidation T10."""

from __future__ import annotations

import numpy as np
import pytest

from diversity_reasoning.metrics import (
    block_kernel,
    hill_number,
    kernel_eigenvalues,
    pseudo_logdet,
    vendi_score,
)

pytestmark = pytest.mark.correctness

Q_ORDERS = (0.0, 0.1, 0.5, 1.0, 2.0, float("inf"))


def test_t1_theorem_4_1_block_identity() -> None:
    multiplicities = (5, 3, 1)
    total = sum(multiplicities)
    kernel = block_kernel(multiplicities)
    eigenvalues = kernel_eigenvalues(kernel, normalize=True)
    positive = eigenvalues[eigenvalues > 1e-10]
    expected = np.sort(np.asarray(multiplicities, dtype=float) / total)
    assert positive.size == len(multiplicities)
    np.testing.assert_allclose(positive, expected, atol=1e-8, rtol=0)
    probabilities = np.asarray(multiplicities, dtype=float) / total
    for q in Q_ORDERS:
        assert vendi_score(kernel, q) == pytest.approx(hill_number(probabilities, q), abs=1e-8)


def test_t2_vendi_monotonicity() -> None:
    generator = np.random.default_rng(17)
    features = generator.normal(size=(20, 6))
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    kernel = features @ features.T
    scores = [vendi_score(kernel, q) for q in Q_ORDERS]
    for left, right in zip(scores, scores[1:]):
        assert left + 1e-6 * max(1.0, left) >= right


def test_t3_order_2_bounds() -> None:
    kernel = block_kernel((9, 4, 2, 1))
    score_2 = vendi_score(kernel, 2.0)
    score_inf = vendi_score(kernel, float("inf"))
    tolerance = 1e-6 * score_2
    assert np.sqrt(score_2) <= score_inf + tolerance
    assert score_inf <= score_2 + tolerance


def test_t4_pseudo_logdet_block_value() -> None:
    multiplicities = np.asarray((7, 2, 1), dtype=float)
    kernel = block_kernel(multiplicities.astype(int).tolist())
    expected = float(np.log(multiplicities / multiplicities.sum()).sum())
    assert pseudo_logdet(kernel) == pytest.approx(expected, abs=1e-8)
    epsilon_variant = float(np.linalg.slogdet(kernel / kernel.shape[0] + np.eye(len(kernel)))[1])
    assert epsilon_variant != pytest.approx(expected)


def test_t5_q_invariance_for_distinct_answers() -> None:
    kernel = np.eye(11)
    for q in Q_ORDERS:
        assert vendi_score(kernel, q) == pytest.approx(11.0, abs=1e-8)


def test_t7_duplication_invariance_and_vendi_sensitivity() -> None:
    # Uniform duplication keeps normalized nonzero eigenvalues unchanged.
    base = block_kernel((3, 1))
    uniform_duplicate = np.tile(base, (2, 2))
    assert pseudo_logdet(uniform_duplicate) == pytest.approx(pseudo_logdet(base), abs=1e-8)
    # Concentrating additional copies in the modal block reduces VS_inf.
    skewed_duplicate = block_kernel((4, 1))
    assert vendi_score(skewed_duplicate, float("inf")) < vendi_score(base, float("inf"))


@pytest.mark.parametrize("multiplicities", [(40, 20, 5, 1), (8, 8), (5, 3, 2)])
def test_t10_legacy_pool_revalidation(multiplicities: tuple[int, ...]) -> None:
    kernel = block_kernel(multiplicities)
    scores = [vendi_score(kernel, q) for q in Q_ORDERS]
    assert all(left + 1e-6 * left >= right for left, right in zip(scores, scores[1:]))
    assert np.sqrt(scores[-2]) <= scores[-1] + 1e-6 * scores[-2]
    assert scores[-1] <= scores[-2] + 1e-6 * scores[-2]
