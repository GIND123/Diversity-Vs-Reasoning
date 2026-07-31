"""Blocking extraction/encoder-pipeline check T9."""

from __future__ import annotations

import numpy as np
import pytest

from diversity_reasoning.answers import extract_gsm8k, extract_math, normalize_numeric
from diversity_reasoning.kernels import answer_kernel, embedding_kernel, mixed_kernel

pytestmark = pytest.mark.correctness


def test_t9_answer_extractors_and_answer_kernel() -> None:
    assert extract_gsm8k("Reasoning with 12 intermediate steps. #### 1,250.00") == "1250"
    assert extract_math(r"Work here. \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert normalize_numeric("-0.000") == "0"
    kernel = answer_kernel(["2", "2", "3"])
    np.testing.assert_array_equal(
        kernel,
        np.asarray([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float),
    )


def test_t9_embedding_and_mixed_kernel_contract() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]])
    embedding, clipped = embedding_kernel(embeddings)
    answer = answer_kernel(["a", "a", "b"])
    combined = mixed_kernel(answer, embedding, alpha=0.25)
    assert clipped
    assert combined.shape == (3, 3)
    np.testing.assert_allclose(np.diag(combined), 1.0)
    np.testing.assert_allclose(combined, combined.T)
