"""Dataset-specific answer extraction and equivalence."""

from __future__ import annotations

import multiprocessing as mp
import re
from decimal import Decimal, InvalidOperation
from queue import Empty
from typing import Optional, Tuple

_NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def normalize_numeric(value: str) -> Optional[str]:
    """Normalize a numeric answer through ``Decimal`` without float drift."""
    candidate = value.strip().replace(",", "")
    match = _NUMBER.fullmatch(candidate)
    if not match:
        return None
    try:
        number = Decimal(candidate)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"-0", "+0", ""} else normalized


def extract_gsm8k(text: str) -> Optional[str]:
    """Extract the final GSM8K number, preferring the canonical #### marker."""
    tail = text.rsplit("####", 1)[-1]
    matches = _NUMBER.findall(tail)
    if not matches and tail != text:
        matches = _NUMBER.findall(text)
    return normalize_numeric(matches[-1]) if matches else None


def _last_boxed(text: str) -> Optional[str]:
    starts = [match.end() for match in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(starts):
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index].strip()
    return None


def normalize_math_text(value: str) -> str:
    """Apply conservative LaTeX/string normalization before equivalence."""
    normalized = value.strip()
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace(r"\,", "").replace(" ", "")
    normalized = normalized.rstrip(".")
    numeric = normalize_numeric(normalized)
    return numeric if numeric is not None else normalized


def extract_math(text: str) -> Optional[str]:
    boxed = _last_boxed(text)
    return normalize_math_text(boxed) if boxed is not None else None


def extract_answer(dataset: str, text: str) -> Optional[str]:
    normalized_name = dataset.lower()
    if "gsm8k" in normalized_name:
        return extract_gsm8k(text)
    if "math" in normalized_name:
        return extract_math(text)
    raise ValueError(f"Unknown evaluator for dataset {dataset!r}")


def _sympy_worker(left: str, right: str, queue: mp.Queue[bool]) -> None:
    import sympy  # type: ignore[import-untyped]

    prepared_left = left.replace("^", "**")
    prepared_right = right.replace("^", "**")
    try:
        result = sympy.simplify(sympy.sympify(prepared_left) - sympy.sympify(prepared_right)) == 0
    except (sympy.SympifyError, TypeError, ValueError):
        result = False
    queue.put(bool(result))


def math_equivalent(left: str, right: str, timeout_seconds: float = 5.0) -> Tuple[bool, bool]:
    """Return ``(equivalent, timed_out)`` with normalized fallback semantics."""
    normalized_left = normalize_math_text(left)
    normalized_right = normalize_math_text(right)
    if normalized_left == normalized_right:
        return True, False
    context = mp.get_context("spawn")
    queue: mp.Queue[bool] = context.Queue(maxsize=1)
    process = context.Process(target=_sympy_worker, args=(normalized_left, normalized_right, queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return normalized_left == normalized_right, True
    try:
        return queue.get_nowait(), False
    except Empty:
        return False, False
