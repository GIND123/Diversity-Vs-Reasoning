"""Canonical cache paths and deterministic subsampling."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np


def safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    if not component or component in {".", ".."}:
        raise ValueError(f"Unsafe empty path component derived from {value!r}")
    return component


def generation_path(cache: Path, model: str, dataset: str, qid: str) -> Path:
    return (
        cache
        / "gen"
        / safe_component(model)
        / safe_component(dataset)
        / safe_component(qid)
        / "chains.jsonl"
    )


def embedding_path(
    cache: Path,
    encoder: str,
    model: str,
    dataset: str,
    qid: str,
) -> Path:
    return (
        cache
        / "emb"
        / safe_component(encoder)
        / safe_component(model)
        / safe_component(dataset)
        / f"{safe_component(qid)}.npy"
    )


def spectrum_path(
    cache: Path,
    kernel: str,
    encoder: str,
    model: str,
    dataset: str,
    qid: str,
    budget: int,
    seed: int,
) -> Path:
    return (
        cache
        / "spec"
        / safe_component(kernel)
        / safe_component(encoder)
        / safe_component(model)
        / safe_component(dataset)
        / safe_component(qid)
        / str(budget)
        / f"{seed}.npz"
    )


def selection_path(
    cache: Path,
    objective: str,
    kernel: str,
    model: str,
    dataset: str,
    qid: str,
    output_budget: int,
) -> Path:
    return (
        cache
        / "sel"
        / safe_component(objective)
        / safe_component(kernel)
        / safe_component(model)
        / safe_component(dataset)
        / safe_component(qid)
        / f"{output_budget}.json"
    )


def subsample_indices(pool_size: int, budget: int, seed: int) -> List[int]:
    """Produce sorted deterministic indices for a seeded nested-free subsample."""
    if not 1 <= budget <= pool_size:
        raise ValueError("budget must lie within the pool")
    generator = np.random.default_rng(seed)
    selected = np.sort(generator.choice(pool_size, size=budget, replace=False))
    return [int(value) for value in selected]
