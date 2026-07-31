"""Pure generation-side logic shared by the local CLI and the Modal workers.

Nothing here imports torch, vLLM, or the Hub. The GPU worker calls these
functions so that seeding, chain numbering, answer parsing, and manifest
contents are identical whether a bank is produced on Modal or locally.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .answers import extract_answer
from .prompts import PROMPT_TEMPLATE_VERSION, dataset_key
from .schemas import ChainRecord

_SEED_MODULUS = 2**31 - 1


@dataclass(frozen=True)
class GenerationSettings:
    """Decoding settings from ``configs/base.yaml``; recorded in every manifest."""

    chains_per_question: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 400
    micro_batch_chains: int = 128
    generation_seed: int = 0

    def __post_init__(self) -> None:
        if self.chains_per_question <= 0 or self.micro_batch_chains <= 0:
            raise ValueError("Chain counts must be positive")
        if self.chains_per_question % self.micro_batch_chains:
            raise ValueError("chains_per_question must be a multiple of micro_batch_chains")
        if not 0 < self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("Decoding parameters are out of range")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

    @property
    def micro_batches(self) -> int:
        return self.chains_per_question // self.micro_batch_chains

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["prompt_template_version"] = PROMPT_TEMPLATE_VERSION
        return payload


def request_seed(generation_seed: int, qid: str, micro_batch_index: int) -> int:
    """Derive a per-request sampling seed that is stable across machines.

    Uses SHA-256 rather than ``hash`` so a bank regenerated on another host or
    Python build reproduces byte-identical chains.
    """
    if micro_batch_index < 0:
        raise ValueError("micro_batch_index must be nonnegative")
    digest = hashlib.sha256(f"{generation_seed}:{qid}:{micro_batch_index}".encode())
    return int.from_bytes(digest.digest()[:8], "big") % _SEED_MODULUS


def request_seeds(settings: GenerationSettings, qid: str) -> List[int]:
    return [
        request_seed(settings.generation_seed, qid, index)
        for index in range(settings.micro_batches)
    ]


def build_chain_records(
    dataset: str,
    texts: Sequence[str],
    *,
    logprob_sums: Optional[Sequence[Optional[float]]] = None,
    token_counts: Optional[Sequence[Optional[int]]] = None,
    generation_seed: int = 0,
    start_chain_id: int = 0,
) -> List[ChainRecord]:
    """Parse raw completions into cached chain records, flagging failures."""
    dataset_key(dataset)
    if logprob_sums is not None and len(logprob_sums) != len(texts):
        raise ValueError("logprob_sums must align with texts")
    if token_counts is not None and len(token_counts) != len(texts):
        raise ValueError("token_counts must align with texts")
    records: List[ChainRecord] = []
    for offset, text in enumerate(texts):
        answer = extract_answer(dataset, text)
        records.append(
            ChainRecord(
                chain_id=start_chain_id + offset,
                text=text,
                answer=answer,
                parsed=answer is not None,
                logprob_sum=None if logprob_sums is None else logprob_sums[offset],
                token_count=None if token_counts is None else token_counts[offset],
                generation_seed=generation_seed,
            )
        )
    return records


def unparsed_rate(records: Sequence[ChainRecord]) -> float:
    """Fraction of chains with no extractable answer (blueprint threshold 5%)."""
    if not records:
        return 0.0
    return sum(1 for record in records if not record.parsed) / len(records)


def shard_plan(qids: Sequence[str], shard_size: int) -> List[Tuple[int, List[str]]]:
    """Split an ordered question list into numbered, resumable Hub shards."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    ordered = sorted(qids)
    if len(set(ordered)) != len(ordered):
        raise ValueError("Duplicate qids in shard plan")
    return [
        (index, ordered[start : start + shard_size])
        for index, start in enumerate(range(0, len(ordered), shard_size))
    ]


def bank_manifest(
    *,
    model_id: str,
    model_short: str,
    dataset: str,
    settings: GenerationSettings,
    question_count: int,
    chain_count: int,
    unparsed: float,
    shards: Sequence[int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Manifest written next to every bank; the provenance record for the paper."""
    payload: Dict[str, Any] = {
        "model_id": model_id,
        "model_short": model_short,
        "dataset": dataset_key(dataset),
        "settings": settings.to_dict(),
        "question_count": question_count,
        "chain_count": chain_count,
        "unparsed_rate": round(float(unparsed), 6),
        "unparsed_rate_exceeds_5pct": bool(unparsed > 0.05),
        "shards": sorted(int(index) for index in shards),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    return payload
