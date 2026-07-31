"""Shared experiment metadata and preparation logic."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

from .config import load_config, resolve_paths
from .gates import assert_correctness_gate
from .io import write_json_atomic


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    title: str
    purpose: str
    plot_ids: Sequence[str]
    table_ids: Sequence[str] = ()
    requires_cache_groups: Sequence[str] = ()
    optional: bool = False


def describe(spec: ExperimentSpec) -> str:
    return json.dumps(asdict(spec), indent=2)


def prepare_experiment(
    spec: ExperimentSpec,
    *,
    config_path: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    """Validate an experiment launch and, unless dry, persist its run manifest."""
    config = load_config(config_path)
    paths = resolve_paths(config)
    available = {group: (paths.cache / group).exists() for group in spec.requires_cache_groups}
    payload: Dict[str, Any] = {
        "experiment": asdict(spec),
        "config": str(config_path),
        "cache_groups": available,
        "dry_run": dry_run,
    }
    if dry_run:
        return payload
    assert_correctness_gate(paths.gate_stamp)
    missing = [group for group, present in available.items() if not present]
    if missing:
        raise RuntimeError(
            f"{spec.experiment_id} is missing cache groups: {', '.join(missing)}. "
            "Populate upstream artifacts before launch."
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = paths.outputs / spec.experiment_id.lower() / run_id / "manifest.json"
    payload["prepared_at"] = datetime.now(timezone.utc).isoformat()
    payload["status"] = "prepared"
    write_json_atomic(destination, payload)
    payload["manifest"] = str(destination)
    return payload
