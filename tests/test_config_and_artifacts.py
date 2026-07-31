from __future__ import annotations

from pathlib import Path

from diversity_reasoning.artifacts import generation_path, safe_component
from diversity_reasoning.config import load_config, resolve_paths
from diversity_reasoning.constants import DEFAULT_CONFIG, ROOT
from diversity_reasoning.registry import EXPERIMENTS


def test_base_config_resolves_from_repository_root() -> None:
    paths = resolve_paths(load_config(DEFAULT_CONFIG))
    assert paths.root == ROOT
    assert paths.cache == ROOT / "cache"


def test_artifact_path_sanitizes_remote_ids(tmp_path: Path) -> None:
    path = generation_path(tmp_path, "Qwen/Qwen2.5", "openai/gsm8k", "question 1")
    expected = tmp_path / "gen" / "Qwen-Qwen2.5" / "openai-gsm8k" / "question-1" / "chains.jsonl"
    assert path == expected
    assert safe_component("x/y") == "x-y"


def test_all_blueprint_experiments_registered() -> None:
    assert set(EXPERIMENTS) == {
        "D0",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
    }
