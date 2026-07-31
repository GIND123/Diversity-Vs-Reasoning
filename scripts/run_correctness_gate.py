#!/usr/bin/env python3
"""Run the blocking correctness tests and stamp the exact tested source tree."""

from __future__ import annotations

import subprocess
import sys

import pytest

from diversity_reasoning.gates import write_gate_stamp


def main() -> int:
    command = [sys.executable, "-m", "pytest", "-m", "correctness", "tests"]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    payload = write_gate_stamp(pytest_version=pytest.__version__)
    print(f"Correctness gate passed: {payload['fingerprint'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
