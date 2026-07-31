"""Answer, embedding, mixed, and anisotropy-adjusted kernels."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def l2_normalize(rows: FloatArray) -> FloatArray:
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Expected a two-dimensional feature matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero embedding")
    return np.asarray(values / norms, dtype=np.float64)


def embedding_kernel(
    embeddings: FloatArray,
    *,
    clip_if_negative: bool = True,
) -> Tuple[FloatArray, bool]:
    """Build cosine K_emb and report whether affine clipping fired."""
    normalized = l2_normalize(embeddings)
    kernel = normalized @ normalized.T
    clipped = bool(np.any(kernel < 0))
    if clipped and clip_if_negative:
        kernel = (1.0 + kernel) / 2.0
    np.fill_diagonal(kernel, 1.0)
    return (kernel + kernel.T) / 2.0, clipped


def answer_kernel(answers: Sequence[str]) -> FloatArray:
    """Build exact-equivalence K_ans for already normalized parsed answers."""
    if not answers:
        raise ValueError("At least one parsed answer is required")
    if any(answer is None for answer in answers):
        raise ValueError("Unparsed answers must be excluded before kernel construction")
    values = np.asarray(list(answers), dtype=object)
    return np.equal.outer(values, values).astype(np.float64)


def mixed_kernel(
    answer: FloatArray,
    embedding: FloatArray,
    alpha: float,
) -> FloatArray:
    """Build K_alpha = alpha K_ans + (1-alpha) K_emb."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    if answer.shape != embedding.shape:
        raise ValueError("Answer and embedding kernels must have equal shapes")
    return alpha * np.asarray(answer) + (1 - alpha) * np.asarray(embedding)


def remove_top_components(
    embeddings: FloatArray,
    components: int,
    *,
    fit_embeddings: Optional[FloatArray] = None,
) -> FloatArray:
    """Remove the leading common PCA directions and L2-normalize again."""
    values = np.asarray(embeddings, dtype=np.float64)
    if components < 0 or components >= min(values.shape):
        raise ValueError("components must be nonnegative and below matrix rank bounds")
    if components == 0:
        return l2_normalize(values)
    fit = values if fit_embeddings is None else np.asarray(fit_embeddings, dtype=np.float64)
    centered_fit = fit - fit.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered_fit, full_matrices=False)
    directions = right[:components]
    centered = values - fit.mean(axis=0, keepdims=True)
    adjusted = centered - (centered @ directions.T) @ directions
    return l2_normalize(adjusted)
