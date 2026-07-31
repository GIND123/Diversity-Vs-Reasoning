"""Alignment with the published definitions of the Vendi Score family.

Each test re-implements an equation directly from the source paper and checks
our code against it, so a refactor cannot silently drift from the literature.

Pasarkar & Dieng, "Cousins of the Vendi Score", AISTATS 2024 (arXiv:2310.12952):

    Eq (1)  Hill number:  H_q(p) = 1/(1-q) log sum_{i in supp(p)} p_i^q
                          D_q(p) = exp(H_q(p))
    Eq (6)  Vendi score:  VS_q(x,k) = exp( 1/(1-q) log sum_{i in supp(lambda)}
                          lambda_i^q ), lambda = eigenvalues of the *normalized*
                          similarity matrix, supp = the nonzero ones.
    Eq (7)  VS_inf <= ... <= VS_1 <= VS_0            (monotone decreasing in q)
    Eq (8)  sqrt(VS_2) <= VS_inf <= VS_2
    Thm 4.1 For block kernels with multiplicities M_i and C = sum M_i, the
            normalized kernel K/C has exactly N nonzero eigenvalues, and
            lambda_i = M_i / C.

Friedman & Dieng, "The Vendi Score" (arXiv:2210.02410): the original score is
the q = 1 member, the exponential of the Shannon (von Neumann) entropy of the
eigenvalues. The q-parameterized family is Renyi, not von Neumann.
"""

from __future__ import annotations

import numpy as np
import pytest

from diversity_reasoning.constants import Q_ORDERS
from diversity_reasoning.metrics import block_kernel, vendi_score
from diversity_reasoning.spectra import functionals, functionals_from_counts, q_label

pytestmark = pytest.mark.correctness


def paper_hill_number(probabilities: np.ndarray, q: float) -> float:
    """Eq (1), transcribed literally, with the documented q -> 1, inf limits."""
    support = probabilities[probabilities > 0]
    if np.isinf(q):
        return float(1.0 / support.max())
    if q == 1.0:
        return float(np.exp(-np.sum(support * np.log(support))))
    return float(np.exp(np.log(np.sum(support**q)) / (1.0 - q)))


def paper_vendi_score(kernel: np.ndarray, q: float) -> float:
    """Eq (6): Hill number of the nonzero eigenvalues of the normalized kernel."""
    eigenvalues = np.linalg.eigvalsh(kernel / kernel.shape[0])
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    nonzero = eigenvalues[eigenvalues > 1e-10 * eigenvalues.max()]
    return paper_hill_number(nonzero, q)


def random_kernel(n: int, d: int, seed: int) -> np.ndarray:
    rows = np.random.default_rng(seed).normal(size=(n, d))
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    kernel = (1.0 + rows @ rows.T) / 2.0
    np.fill_diagonal(kernel, 1.0)
    return (kernel + kernel.T) / 2.0


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 1.0, 2.0, float("inf")])
def test_our_vs_q_matches_equation_6(seed: int, q: float) -> None:
    kernel = random_kernel(24, 8, seed)
    ours = functionals(kernel, q_orders=[q])[f"vs_{q_label(q)}"]
    assert ours == pytest.approx(paper_vendi_score(kernel, q), rel=1e-9)


@pytest.mark.parametrize("multiplicities", [(3, 2, 1), (5, 5), (10, 1, 1, 1), (7,)])
def test_theorem_4_1_eigenvalues_equal_prevalence(multiplicities) -> None:
    """Thm 4.1: N nonzero eigenvalues, each exactly M_i / C."""
    counts = np.asarray(multiplicities, dtype=float)
    total = counts.sum()
    kernel = block_kernel(multiplicities)
    eigenvalues = np.linalg.eigvalsh(kernel / kernel.shape[0])
    nonzero = np.sort(eigenvalues[eigenvalues > 1e-10])[::-1]
    assert nonzero.size == len(multiplicities)
    assert np.allclose(nonzero, np.sort(counts / total)[::-1], atol=1e-9)


@pytest.mark.parametrize("multiplicities", [(3, 2, 1), (5, 5), (10, 1, 1, 1)])
def test_block_kernel_vs_q_equals_hill_number_of_prevalence(multiplicities) -> None:
    """Thm 4.1 corollary: on block kernels VS_q recovers D_q(M/C) exactly."""
    counts = np.asarray(multiplicities, dtype=float)
    prevalence = counts / counts.sum()
    kernel = block_kernel(multiplicities)
    for q in Q_ORDERS:
        expected = paper_hill_number(prevalence, float(q))
        assert vendi_score(kernel, q) == pytest.approx(expected, rel=1e-8)
        assert functionals_from_counts(list(multiplicities))[f"vs_{q_label(q)}"] == pytest.approx(
            expected, rel=1e-8
        )


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_equation_7_monotone_decreasing_in_q(seed: int) -> None:
    values = functionals(random_kernel(30, 10, seed))
    ordered = [values[f"vs_{q_label(q)}"] for q in Q_ORDERS]
    for left, right in zip(ordered, ordered[1:]):
        assert left >= right - 1e-9 * max(1.0, abs(left))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_equation_8_order_two_bounds_order_infinity(seed: int) -> None:
    values = functionals(random_kernel(30, 10, seed))
    vs2, vs_inf = values["vs_2"], values["vs_inf"]
    assert np.sqrt(vs2) <= vs_inf + 1e-9
    assert vs_inf <= vs2 + 1e-9


def test_effective_number_axiom_identical_items_score_one() -> None:
    """Axiom 1: identical items must score 1; N dissimilar items must score N."""
    identical = np.ones((12, 12))
    for q in Q_ORDERS:
        assert functionals(identical, q_orders=[q])[f"vs_{q_label(q)}"] == pytest.approx(1.0)
    dissimilar = np.eye(7)
    for q in Q_ORDERS:
        assert functionals(dissimilar, q_orders=[q])[f"vs_{q_label(q)}"] == pytest.approx(7.0)


def test_q1_is_the_original_vendi_score() -> None:
    """The q=1 member is the exponential of the Shannon entropy of the spectrum."""
    kernel = random_kernel(20, 6, seed=5)
    eigenvalues = np.clip(np.linalg.eigvalsh(kernel / kernel.shape[0]), 0.0, None)
    support = eigenvalues[eigenvalues > 1e-10]
    shannon = float(np.exp(-np.sum(support * np.log(support))))
    assert functionals(kernel, q_orders=[1.0])["vs_1"] == pytest.approx(shannon, rel=1e-9)


def test_normalized_spectrum_sums_to_one_for_unit_diagonal_kernels() -> None:
    """Eq (6) reads lambda straight off the normalized kernel, with no rescaling.

    Our implementation renormalizes the retained eigenvalues after thresholding;
    that is only legitimate because a unit-diagonal kernel has trace n, so the
    normalized spectrum already sums to 1 and the renormalization is a no-op.
    """
    for seed in range(3):
        kernel = random_kernel(25, 9, seed)
        eigenvalues = np.linalg.eigvalsh(kernel / kernel.shape[0])
        assert eigenvalues.sum() == pytest.approx(1.0, abs=1e-9)
