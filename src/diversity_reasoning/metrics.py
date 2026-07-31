"""Spectral diversity and coverage functionals.

Definitions follow Dieng's Vendi Score family (Friedman & Dieng, 2023) and its
order-q extension (Pasarkar & Dieng, "Cousins of the Vendi Score",
arXiv:2310.12952):

- ``VS_q`` is the exponential of the **Rényi entropy** of order q of the
  normalized similarity-kernel spectrum. q tunes sensitivity to prevalence;
  q = 1 recovers the original Vendi Score (Shannon/von Neumann case). Every
  order q — including the q -> 0 richness limit — is a *diversity* measure.
- **Coverage** is a separate functional, never a member of the family: the
  pseudo log-determinant, i.e. the sum of the logs of the nonzero eigenvalues
  of the same kernel.

The q sweep used across the study is {0, 0.1, 0.5, 1.0, 2.0, inf}
(``constants.Q_ORDERS``), computed via ``score_K`` from the pinned
vertaix/Vendi-Score commit recorded in ``ENVIRONMENT.md``.
"""

from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from vendi_score.vendi import score_K  # type: ignore[import-untyped]

from .schemas import SpectrumRecord

QOrder = Union[float, str]
FloatArray = NDArray[np.float64]


def validate_kernel(kernel: FloatArray, atol: float = 1e-8) -> FloatArray:
    """Return a float64 symmetric kernel or raise a useful error."""
    matrix = np.asarray(kernel, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Kernel must be a square matrix")
    if matrix.shape[0] == 0:
        raise ValueError("Kernel must contain at least one item")
    if not np.isfinite(matrix).all():
        raise ValueError("Kernel contains NaN or infinite values")
    if not np.allclose(matrix, matrix.T, atol=atol, rtol=0):
        raise ValueError("Kernel must be symmetric")
    return (matrix + matrix.T) / 2


def kernel_eigenvalues(
    kernel: FloatArray,
    *,
    normalize: bool,
    negative_tolerance: float = 1e-8,
) -> FloatArray:
    """Compute sorted eigenvalues while rejecting materially non-PSD kernels."""
    matrix = validate_kernel(kernel)
    if normalize:
        matrix = matrix / matrix.shape[0]
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues.min()) < -negative_tolerance * scale:
        raise ValueError(
            f"Kernel is not positive semidefinite: minimum eigenvalue={eigenvalues.min():.3e}"
        )
    return np.clip(eigenvalues, 0.0, None)


def _upstream_q(q: QOrder) -> QOrder:
    if isinstance(q, str):
        if q.lower() in {"inf", "infinity"}:
            return "inf"
        return float(q)
    if np.isinf(q):
        return "inf"
    return float(q)


def vendi_score(kernel: FloatArray, q: QOrder = 1.0) -> float:
    """Compute VS_q with the blueprint-pinned upstream ``score_K``.

    Eigenvalues below the configured relative tolerance are removed before the
    upstream entropy calculation. This prevents numerical null eigenvalues from
    being counted as real modes at q=0.
    """
    matrix = validate_kernel(kernel)
    eigenvalues = kernel_eigenvalues(matrix, normalize=True)
    maximum = float(eigenvalues.max())
    kept = eigenvalues[eigenvalues > 1e-10 * maximum]
    probabilities = kept / kept.sum()
    # score_K always divides by matrix size. Scaling the diagonal by its size
    # presents the cleaned normalized spectrum to the pinned implementation.
    spectral_kernel = np.diag(probabilities * probabilities.size)
    return float(score_K(spectral_kernel, q=_upstream_q(q)))


def hill_number(probabilities: Sequence[float], q: QOrder = 1.0) -> float:
    """Independent Hill-number reference used by correctness test T1."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or np.any(values < 0):
        raise ValueError("Probabilities must be a nonnegative vector")
    total = float(values.sum())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Probabilities must sum to one, got {total}")
    positive = values[values > 0]
    order = _upstream_q(q)
    if order == "inf":
        return float(1.0 / positive.max())
    numeric_order = float(order)
    if numeric_order == 1:
        return float(np.exp(-np.sum(positive * np.log(positive))))
    if numeric_order == 0:
        return float(positive.size)
    return float(np.sum(positive**numeric_order) ** (1.0 / (1.0 - numeric_order)))


def pseudo_logdet(
    kernel: FloatArray,
    *,
    tau: float = 1e-10,
    normalize: bool = True,
) -> float:
    """Sum log nonzero eigenvalues using a relative spectral threshold.

    The default ``K / n`` convention matches correctness test T4 and makes
    uniform duplication of a pool invariant, as required by T7.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    eigenvalues = kernel_eigenvalues(kernel, normalize=normalize)
    maximum = float(eigenvalues.max())
    if maximum == 0:
        return float("-inf")
    kept = eigenvalues[eigenvalues > tau * maximum]
    return float(np.log(kept).sum())


def spectrum_record(
    kernel: FloatArray,
    q_orders: Sequence[QOrder],
    *,
    tau: float = 1e-10,
) -> SpectrumRecord:
    """Build the exact cached spectral artifact described in the blueprint."""
    raw = kernel_eigenvalues(kernel, normalize=False)
    normalized = kernel_eigenvalues(kernel, normalize=True)
    maximum = float(normalized.max())
    kept = normalized[normalized > tau * maximum]
    scores: Dict[str, float] = {}
    for q in q_orders:
        key = "inf" if _upstream_q(q) == "inf" else f"{float(q):g}"
        scores[key] = vendi_score(kernel, q)
    return SpectrumRecord(
        eigenvalues_raw=raw.tolist(),
        eigenvalues_normalized=normalized.tolist(),
        vendi_scores=scores,
        pseudo_logdet=float(np.log(kept).sum()) if kept.size else float("-inf"),
        n_nonzero=int(kept.size),
        lambda_min=float(kept.min()) if kept.size else 0.0,
        lambda_max=maximum,
        threshold=tau,
    )


def block_kernel(multiplicities: Sequence[int]) -> FloatArray:
    """Construct an exact-answer block kernel for synthetic checks."""
    if not multiplicities or any(value <= 0 for value in multiplicities):
        raise ValueError("Multiplicities must be positive")
    labels = np.concatenate(
        [np.full(count, index, dtype=np.int64) for index, count in enumerate(multiplicities)]
    )
    return np.equal.outer(labels, labels).astype(np.float64)
