"""Minimal `.env` loader that never logs secret values."""

from __future__ import annotations

import os
from pathlib import Path

from .constants import ROOT


def load_local_env(path: Path = ROOT / ".env") -> bool:
    """Load missing environment variables from a local key-value file."""
    if not path.exists():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"Invalid environment variable name in {path}")
        os.environ.setdefault(key, value.strip().strip("'\""))
    return True
