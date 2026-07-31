"""Configuration loading with predictable project-root-relative paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .constants import DEFAULT_CONFIG, ROOT


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved filesystem locations used by the pipeline."""

    root: Path
    cache: Path
    data: Path
    outputs: Path
    gate_stamp: Path


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load one YAML mapping and reject ambiguous top-level documents."""
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def load_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load the primary experiment configuration."""
    return load_yaml(path)


def resolve_paths(config: Mapping[str, Any], root: Path = ROOT) -> ProjectPaths:
    """Resolve configured project paths without depending on the process CWD."""
    project = config.get("project", {})
    if not isinstance(project, Mapping):
        raise ValueError("config.project must be a mapping")

    def resolve(key: str, default: str) -> Path:
        candidate = Path(str(project.get(key, default)))
        return candidate if candidate.is_absolute() else root / candidate

    return ProjectPaths(
        root=root,
        cache=resolve("cache_dir", "cache"),
        data=resolve("data_dir", "data"),
        outputs=resolve("output_dir", "outputs"),
        gate_stamp=resolve("gate_stamp", ".correctness-gate"),
    )
