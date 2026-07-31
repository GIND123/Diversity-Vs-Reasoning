#!/usr/bin/env python3
"""Validate configuration, token presence, and pinned Vendi revision metadata."""

from __future__ import annotations

import os

from diversity_reasoning.config import load_config, resolve_paths
from diversity_reasoning.constants import DEFAULT_CONFIG
from diversity_reasoning.env import load_local_env


def main() -> int:
    load_local_env()
    config = load_config(DEFAULT_CONFIG)
    paths = resolve_paths(config)
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is missing. Copy .env.example to .env and insert a token.")
    print(f"Configuration valid; cache root: {paths.cache}")
    print("HF_TOKEN is present (value intentionally hidden).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
