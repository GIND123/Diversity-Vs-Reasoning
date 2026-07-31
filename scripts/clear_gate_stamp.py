#!/usr/bin/env python3
"""Remove only the correctness stamp, if present."""

from diversity_reasoning.constants import DEFAULT_GATE_STAMP

DEFAULT_GATE_STAMP.unlink(missing_ok=True)
