"""One fixed prompt template per dataset, versioned so banks stay comparable.

The blueprint fixes a single prompt template per dataset for the entire
generation matrix. Changing any string here changes every chain bank, so the
version tag below must be bumped and recorded in the bank manifest.
"""

from __future__ import annotations

from typing import Dict, List

PROMPT_TEMPLATE_VERSION = "v1"

_SYSTEM = "You are a careful mathematical reasoner. Think step by step, then give the final answer."

_GSM8K_USER = (
    "Solve the problem below. Show your reasoning step by step, then state the final "
    "answer on its own final line in exactly this format:\n"
    "#### <number>\n\n"
    "Problem: {question}"
)

_MATH_USER = (
    "Solve the problem below. Show your reasoning step by step, then put the final "
    "answer inside \\boxed{{}} on the last line.\n\n"
    "Problem: {question}"
)

_USER_TEMPLATES: Dict[str, str] = {
    "gsm8k": _GSM8K_USER,
    "math": _MATH_USER,
}


def dataset_key(dataset: str) -> str:
    """Map any dataset spelling onto the canonical ``gsm8k``/``math`` key."""
    normalized = dataset.strip().lower()
    if "gsm8k" in normalized:
        return "gsm8k"
    if "math" in normalized:
        return "math"
    raise ValueError(f"Unknown dataset {dataset!r}; expected a GSM8K or MATH variant")


def system_prompt(dataset: str) -> str:
    dataset_key(dataset)
    return _SYSTEM


def user_prompt(dataset: str, question: str) -> str:
    if not question.strip():
        raise ValueError("Question text cannot be empty")
    return _USER_TEMPLATES[dataset_key(dataset)].format(question=question.strip())


def chat_messages(dataset: str, question: str) -> List[Dict[str, str]]:
    """Build the chat turns handed to each model's own chat template."""
    return [
        {"role": "system", "content": system_prompt(dataset)},
        {"role": "user", "content": user_prompt(dataset, question)},
    ]
