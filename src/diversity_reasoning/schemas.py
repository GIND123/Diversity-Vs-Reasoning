"""JSON-serializable artifact schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ChainRecord:
    chain_id: int
    text: str
    answer: Optional[str]
    parsed: bool
    logprob_sum: Optional[float] = None
    token_count: Optional[int] = None
    generation_seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpectrumRecord:
    eigenvalues_raw: List[float]
    eigenvalues_normalized: List[float]
    vendi_scores: Dict[str, float]
    pseudo_logdet: float
    n_nonzero: int
    lambda_min: float
    lambda_max: float
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionRecord:
    objective: str
    selected_indices: List[int]
    gain_trace: List[float]
    pool_size: int
    output_budget: int
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AggregationResult:
    rule: str
    prediction: Optional[str]
    correct: Optional[bool]
    tie: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
