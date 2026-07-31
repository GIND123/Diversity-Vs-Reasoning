"""Question-level inference utilities used by all result tables."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from numpy.typing import NDArray


def paired_bootstrap_delta(
    treatment: Iterable[float],
    baseline: Iterable[float],
    *,
    replicates: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    treated = np.asarray(list(treatment), dtype=np.float64)
    reference = np.asarray(list(baseline), dtype=np.float64)
    if treated.shape != reference.shape or treated.ndim != 1 or treated.size == 0:
        raise ValueError("Paired samples must be aligned nonempty vectors")
    if replicates <= 0 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    delta = treated - reference
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, delta.size, size=(replicates, delta.size))
    samples = delta[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    return float(delta.mean()), float(low), float(high)


def holm_adjust(p_values: Iterable[float]) -> NDArray[np.float64]:
    """Holm family-wise p-value adjustment in original input order."""
    values = np.asarray(list(p_values), dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, (values.size - np.arange(values.size)) * values[order])
    )
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return adjusted
