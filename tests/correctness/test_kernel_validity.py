"""T12: every kernel handed to VS_q must actually be a similarity kernel.

The Vendi score is a functional of a similarity matrix, and its interpretation
as an effective number depends on that matrix being a valid kernel: symmetric,
positive semidefinite, and with unit self-similarity. If any of those fails the
eigenvalues are not a prevalence distribution and the score is meaningless
however correctly the arithmetic is done.

These checks run over every kernel construction the study uses, including the
question-centred variant introduced here, so a future change to kernel
construction cannot quietly invalidate every downstream number.
"""

from __future__ import annotations

import numpy as np
import pytest

from diversity_reasoning.kernels import answer_kernel, embedding_kernel, mixed_kernel
from diversity_reasoning.metrics import block_kernel
from diversity_reasoning.pools import build_pool, fit_corpus_directions

pytestmark = pytest.mark.correctness

TOLERANCE = 1e-8


def assert_valid_kernel(kernel: np.ndarray, name: str) -> None:
    assert kernel.ndim == 2 and kernel.shape[0] == kernel.shape[1], f"{name}: not square"
    assert np.allclose(kernel, kernel.T, atol=TOLERANCE), f"{name}: not symmetric"
    assert np.allclose(np.diag(kernel), 1.0, atol=1e-6), f"{name}: unit diagonal violated"
    eigenvalues = np.linalg.eigvalsh(kernel)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    assert eigenvalues.min() > -TOLERANCE * scale, f"{name}: not PSD ({eigenvalues.min():.2e})"


def make_pool(n=40, d=16, seed=0, shift=6.0):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n, d)) + shift
    rows = [
        {
            "chain_id": i,
            "text": f"c{i}",
            "answer": str(i % 5),
            "parsed": True,
            "logprob_sum": -float(i + 1),
            "token_count": 10,
            "generation_seed": 0,
        }
        for i in range(n)
    ]
    return build_pool("q", "gsm8k", "m", rows, "0", embeddings=embeddings)


def test_t12_answer_kernel_is_valid() -> None:
    assert_valid_kernel(answer_kernel([str(i % 4) for i in range(20)]), "K_ans")


def test_t12_block_kernel_is_valid() -> None:
    assert_valid_kernel(block_kernel([5, 3, 2]), "block")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_t12_embedding_kernel_is_valid(seed: int) -> None:
    rng = np.random.default_rng(seed)
    kernel, _ = embedding_kernel(rng.normal(size=(30, 12)))
    assert_valid_kernel(kernel, "K_emb")


@pytest.mark.parametrize("components", [0, 1, 2])
def test_t12_corpus_adjusted_kernel_is_valid(components: int) -> None:
    pools = [make_pool(seed=s) for s in range(3)]
    fit_corpus_directions(pools, components=3, per_question=40)
    assert_valid_kernel(
        pools[0].kernel("embedding", components=components), f"K_emb c={components}"
    )


def test_t12_question_centred_kernel_is_valid() -> None:
    """The variant this study introduces must satisfy the same requirements."""
    pool = make_pool()
    assert_valid_kernel(pool.kernel("embedding_qc", components=1), "K_emb question-centred")


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_t12_mixed_kernel_is_valid(alpha: float) -> None:
    """A convex combination of two valid kernels must remain valid."""
    pool = make_pool()
    kernel = mixed_kernel(
        pool.kernel("answer"), pool.kernel("embedding_qc", components=1), alpha
    )
    assert_valid_kernel(kernel, f"K_alpha={alpha}")


def test_t12_eigenvalues_of_normalized_kernel_form_a_distribution() -> None:
    """Eq. 6 reads lambda off the normalized kernel as if it were a prevalence
    vector; that is only legitimate if the eigenvalues are nonnegative and sum
    to one, which unit diagonal plus PSD guarantees."""
    pool = make_pool()
    for spec in ("answer", "embedding_qc"):
        kernel = pool.kernel(spec, components=1)
        eigenvalues = np.linalg.eigvalsh(kernel / kernel.shape[0])
        assert eigenvalues.min() > -TOLERANCE
        assert eigenvalues.sum() == pytest.approx(1.0, abs=1e-9)
