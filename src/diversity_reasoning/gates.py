"""Correctness-gate fingerprinting and enforcement."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

from .constants import DEFAULT_GATE_STAMP, ROOT


def _gate_inputs(root: Path = ROOT) -> Iterable[Path]:
    for name in ("src", "tests", "experiments", "figures", "scripts"):
        directory = root / name
        if directory.exists():
            yield from sorted(directory.rglob("*.py"))
    for name in ("pyproject.toml",):
        path = root / name
        if path.exists():
            yield path


def gate_fingerprint(root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    for path in _gate_inputs(root):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_gate_stamp(
    stamp: Path = DEFAULT_GATE_STAMP,
    *,
    pytest_version: str,
) -> Dict[str, str]:
    payload = {
        "fingerprint": gate_fingerprint(),
        "passed_at": datetime.now(timezone.utc).isoformat(),
        "pytest_version": pytest_version,
    }
    stamp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def assert_correctness_gate(stamp: Path = DEFAULT_GATE_STAMP) -> None:
    if not stamp.exists():
        raise RuntimeError("Correctness gate has not passed. Run `make gate` first.")
    payload = json.loads(stamp.read_text(encoding="utf-8"))
    expected = gate_fingerprint()
    if payload.get("fingerprint") != expected:
        raise RuntimeError(
            "Correctness inputs changed after the last green gate. Run `make gate` again."
        )
