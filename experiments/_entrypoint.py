"""Small adapter used by each independently executable experiment module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from diversity_reasoning.constants import DEFAULT_CONFIG
from diversity_reasoning.env import load_local_env
from diversity_reasoning.experiment import prepare_experiment
from diversity_reasoning.registry import EXPERIMENTS


def run(experiment_id: str, argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=EXPERIMENTS[experiment_id].title)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    load_local_env()
    result = prepare_experiment(
        EXPERIMENTS[experiment_id],
        config_path=arguments.config,
        dry_run=arguments.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0
