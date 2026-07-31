"""One-eigendecomposition computation of every spectral functional.

``vendi_score`` in :mod:`.metrics` runs its own eigendecomposition per q, which
is exact but wasteful inside the analysis sweeps (nine budgets x kernels x
thousands of pools). ``functionals`` computes the eigenvalues once and derives
every VS_q as the Hill number of the normalized nonzero spectrum — the identity
the correctness harness pins down (T1) — plus the coverage pseudo log-det and
the epsilon=1 artifact arm kept for E5.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from .constants import Q_ORDERS
from .metrics import hill_number, kernel_eigenvalues

FloatArray = NDArray[np.float64]
QOrder = Union[float, str]

RELATIVE_TOLERANCE = 1e-10


def q_label(q: QOrder) -> str:
    if isinstance(q, str) or np.isinf(q):
        return "inf"
    return f"{float(q):g}"


def functionals(
    kernel: FloatArray,
    *,
    q_orders: Sequence[QOrder] = Q_ORDERS,
    tau: float = RELATIVE_TOLERANCE,
    eigenvalues: Optional[FloatArray] = None,
) -> Dict[str, float]:
    """All diversity orders plus coverage from a single eigendecomposition."""
    if eigenvalues is None:
        eigenvalues = kernel_eigenvalues(kernel, normalize=True)
    maximum = float(eigenvalues.max())
    if maximum <= 0:
        raise ValueError("Kernel spectrum is identically zero")
    kept = eigenvalues[eigenvalues > tau * maximum]
    probabilities = kept / kept.sum()
    result: Dict[str, float] = {f"vs_{q_label(q)}": hill_number(probabilities, q) for q in q_orders}
    # Normalized-spectrum convention, pinned by harness test T4; the raw-spectrum
    # variant (blueprint B3's caching convention) differs by exactly r * log(n).
    result["pseudo_logdet"] = float(np.log(kept).sum())
    result["pseudo_logdet_raw"] = float(np.log(kept).sum() + kept.size * np.log(eigenvalues.size))
    result["logdet_eps1"] = float(np.log1p(eigenvalues).sum())
    # Threshold sensitivity arms for appendix plot P-A3.
    for name, sensitivity_tau in (("tau8", 1e-8), ("tau12", 1e-12)):
        kept_alt = eigenvalues[eigenvalues > sensitivity_tau * maximum]
        result[f"pseudo_logdet_{name}"] = float(np.log(kept_alt).sum())
    result["n_nonzero"] = float(kept.size)
    result["lambda_min"] = float(kept.min())
    result["lambda_max"] = maximum
    return result


def functionals_from_counts(
    counts: Sequence[int],
    *,
    q_orders: Sequence[QOrder] = Q_ORDERS,
) -> Dict[str, float]:
    """Exact block-kernel functionals from answer-class multiplicities (T1/T4)."""
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(values <= 0):
        raise ValueError("counts must be positive")
    probabilities = values / values.sum()
    result = {f"vs_{q_label(q)}": hill_number(probabilities, q) for q in q_orders}
    result["pseudo_logdet"] = float(np.log(probabilities).sum())
    # Zero eigenvalues contribute log1p(0) = 0, so the epsilon arm needs only
    # the nonzero block spectrum.
    result["logdet_eps1"] = float(np.log1p(probabilities).sum())
    result["n_nonzero"] = float(values.size)
    result["lambda_min"] = float(probabilities.min())
    result["lambda_max"] = float(probabilities.max())
    return result


def answer_entropy(counts: Sequence[int]) -> Dict[str, float]:
    """Shannon entropy of the answer distribution and its normalized form."""
    values = np.asarray(counts, dtype=np.float64)
    probabilities = values / values.sum()
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    distinct = int(values.size)
    normalized = entropy / np.log(distinct) if distinct > 1 else 0.0
    return {"entropy": entropy, "normalized_entropy": float(normalized), "n_distinct": distinct}
