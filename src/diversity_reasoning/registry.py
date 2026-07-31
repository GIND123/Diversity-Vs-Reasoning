"""Single source of truth for experiment deliverables and prerequisites."""

from __future__ import annotations

from typing import Dict

from .experiment import ExperimentSpec

_SPECS = [
    ExperimentSpec(
        "D0",
        "Stratification",
        "Build Snell, MATH-level, tail-heaviness, and entropy strata.",
        ("P-0a", "P-0b"),
        ("TB-0",),
        ("gen",),
    ),
    ExperimentSpec(
        "E1",
        "Rare modes",
        "Measure functional sensitivity as rare modes disappear.",
        ("P-1a",),
        requires_cache_groups=("gen", "spec"),
    ),
    ExperimentSpec(
        "E2",
        "Redundancy",
        "Measure response to exact and near-duplicate reasoning chains.",
        ("P-1b",),
        requires_cache_groups=("gen", "emb", "spec"),
    ),
    ExperimentSpec(
        "E3",
        "Sample size",
        "Estimate bias, variance, and ranking stability against pool size.",
        ("P-1c",),
        ("TB-1",),
        ("spec",),
    ),
    ExperimentSpec(
        "E4",
        "Dimensionality",
        "Measure metric and selection stability across PCA dimensions.",
        ("P-1d",),
        requires_cache_groups=("emb", "spec", "sel"),
    ),
    ExperimentSpec(
        "E5",
        "The two log functionals",
        "Quantify within-budget correlation, pooled reversal, and reversal onset.",
        ("P-1e", "P-1f"),
        ("TB-2",),
        ("spec",),
    ),
    ExperimentSpec(
        "E6",
        "Adaptive q",
        "Optional isolated implementation slot for the adaptive-q rule.",
        (),
        requires_cache_groups=("spec",),
        optional=True,
    ),
    ExperimentSpec(
        "R1",
        "Winner map",
        "Run the headline factorial selector comparison.",
        ("P-2a", "P-2b"),
        ("TB-3",),
        ("sel", "agg"),
    ),
    ExperimentSpec(
        "R2",
        "When objectives hurt",
        "Condition diversity and coverage effects on answer tail-heaviness.",
        ("P-2c",),
        requires_cache_groups=("sel", "agg"),
    ),
    ExperimentSpec(
        "R3",
        "Alpha threshold",
        "Find the minimum answer-awareness mixture that separates from random.",
        ("P-2d",),
        ("TB-4",),
        ("sel", "agg"),
    ),
    ExperimentSpec(
        "R4",
        "Tail conditioning",
        "Re-report R1 effects in modal, minority, tail, and absent strata.",
        ("P-2e",),
        requires_cache_groups=("agg",),
    ),
    ExperimentSpec(
        "R5",
        "q inertness",
        "Make the answer-kernel q-invariance theorem visible.",
        ("P-2f",),
        requires_cache_groups=("sel", "agg"),
    ),
    ExperimentSpec(
        "R6",
        "Signals shootout",
        "Compare entropy, margin, verifier, and embedding risk signals.",
        ("P-3a", "P-3b"),
        ("TB-5",),
        ("agg",),
    ),
    ExperimentSpec(
        "R7",
        "Entropy-gated escalation",
        "Evaluate accuracy/coverage/compute operating points.",
        ("P-3c",),
        ("TB-6",),
        ("agg",),
    ),
]

EXPERIMENTS: Dict[str, ExperimentSpec] = {spec.experiment_id: spec for spec in _SPECS}
