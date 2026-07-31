"""Question-bank construction for GSM8K and MATH.

Row-to-``Question`` conversion is pure and unit tested. Only ``load_questions``
touches the Hugging Face Hub, and it imports ``datasets`` lazily so the analysis
environment stays free of the generation dependency stack.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .answers import extract_math, normalize_numeric
from .prompts import dataset_key

MATH_CONFIGS: Tuple[str, ...] = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

_LEVEL = re.compile(r"level\s*([1-5])", re.IGNORECASE)


@dataclass(frozen=True)
class Question:
    """One evaluation question with its model-independent strata metadata."""

    qid: str
    dataset: str
    question: str
    reference_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Question:
        return cls(
            qid=str(payload["qid"]),
            dataset=str(payload["dataset"]),
            question=str(payload["question"]),
            reference_answer=str(payload["reference_answer"]),
            metadata=dict(payload.get("metadata", {})),
        )


def math_level(value: str) -> Optional[int]:
    """Parse ``"Level 3"`` into ``3``; unlabelled or ``Level ?`` rows give None."""
    match = _LEVEL.search(value or "")
    return int(match.group(1)) if match else None


def gsm8k_reference(answer_field: str) -> str:
    """Take the canonical GSM8K gold answer after the ``####`` marker."""
    if "####" not in answer_field:
        raise ValueError("GSM8K gold answer must contain the #### marker")
    normalized = normalize_numeric(answer_field.rsplit("####", 1)[-1].strip())
    if normalized is None:
        raise ValueError("GSM8K gold answer is not numeric after normalization")
    return normalized


def math_reference(solution_field: str) -> Optional[str]:
    """Take the boxed gold answer from a Hendrycks MATH solution."""
    return extract_math(solution_field)


def gsm8k_question(index: int, row: Mapping[str, Any]) -> Question:
    return Question(
        qid=f"gsm8k-test-{index:05d}",
        dataset="gsm8k",
        question=str(row["question"]).strip(),
        reference_answer=gsm8k_reference(str(row["answer"])),
        metadata={"source_index": index, "split": "test"},
    )


def math_question(config: str, index: int, row: Mapping[str, Any]) -> Optional[Question]:
    """Build a MATH question, or None when the gold answer is not boxed."""
    reference = math_reference(str(row["solution"]))
    if reference is None:
        return None
    return Question(
        qid=f"math-{config}-test-{index:04d}",
        dataset="math",
        question=str(row["problem"]).strip(),
        reference_answer=reference,
        metadata={
            "source_index": index,
            "split": "test",
            "subject": config,
            "type": str(row.get("type", "")),
            "level": math_level(str(row.get("level", ""))),
        },
    )


def _stable_rank(qid: str, seed: int) -> str:
    """Seeded, reproducible ordering key that does not depend on Python hashing."""
    return hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()


def select_questions(
    questions: Sequence[Question],
    *,
    limit: Optional[int] = None,
    seed: int = 0,
    stratify_by: Optional[str] = None,
) -> List[Question]:
    """Deterministically subsample a question bank, optionally balancing strata.

    ``stratify_by`` names a metadata key (``level`` for MATH). Questions are
    drawn round-robin across strata so a reduced bank keeps every stratum
    populated, which is the triage rule the blueprint mandates.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    ordered = sorted(questions, key=lambda item: _stable_rank(item.qid, seed))
    if limit is None or limit >= len(ordered):
        return sorted(ordered, key=lambda item: item.qid)

    if stratify_by is None:
        chosen = ordered[:limit]
    else:
        buckets: Dict[str, List[Question]] = {}
        for item in ordered:
            buckets.setdefault(str(item.metadata.get(stratify_by)), []).append(item)
        chosen = []
        position = 0
        keys = sorted(buckets)
        while len(chosen) < limit:
            drained = True
            for key in keys:
                bucket = buckets[key]
                if position < len(bucket):
                    drained = False
                    chosen.append(bucket[position])
                    if len(chosen) == limit:
                        break
            if drained:
                break
            position += 1
    return sorted(chosen, key=lambda item: item.qid)


def questions_from_rows(
    dataset: str,
    rows: Iterable[Mapping[str, Any]],
    config: str = "",
) -> List[Question]:
    """Convert raw Hub rows into questions (pure; used by tests and the loader)."""
    key = dataset_key(dataset)
    built: List[Question] = []
    for index, row in enumerate(rows):
        if key == "gsm8k":
            built.append(gsm8k_question(index, row))
        else:
            if not config:
                raise ValueError("MATH rows require the subject config name")
            question = math_question(config, index, row)
            if question is not None:
                built.append(question)
    return built


def load_questions(
    dataset: str,
    *,
    limit: Optional[int] = None,
    seed: int = 0,
    token: Optional[str] = None,
    gsm8k_repo: str = "openai/gsm8k",
    math_repo: str = "EleutherAI/hendrycks_math",
    math_configs: Sequence[str] = MATH_CONFIGS,
) -> List[Question]:
    """Load a question bank from the Hub. Requires the ``generation`` extra."""
    from datasets import load_dataset

    key = dataset_key(dataset)
    if key == "gsm8k":
        rows = load_dataset(gsm8k_repo, "main", split="test", token=token)
        questions = questions_from_rows("gsm8k", rows)
        return select_questions(questions, limit=limit, seed=seed)

    questions = []
    for config in math_configs:
        rows = load_dataset(math_repo, config, split="test", token=token)
        questions.extend(questions_from_rows("math", rows, config=config))
    return select_questions(questions, limit=limit, seed=seed, stratify_by="level")
