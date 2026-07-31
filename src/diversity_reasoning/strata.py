"""D0: stratification of every question from its full 1024-chain bank.

Produces, per (model, dataset, question): pass@1, Snell difficulty bins
(terciles for headline claims per the 2026-07-30 triage entry, quintiles for
TB-0), the tail-heaviness label of the correct answer, and answer entropy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from .pools import Pool
from .spectra import answer_entropy

TAIL_LABELS = ("modal", "minority", "tail", "absent")


def tail_heaviness(pool: Pool) -> str:
    """Rank of the correct answer class in the full bank's answer distribution."""
    counts = pool.answer_counts()
    correct_classes = {int(c) for c in np.unique(pool.class_ids[pool.correct])}
    if not correct_classes:
        return "absent"
    ranked = sorted(counts, key=lambda class_id: (-counts[class_id], class_id))
    best_rank = min(ranked.index(class_id) + 1 for class_id in correct_classes)
    if best_rank == 1:
        return "modal"
    if best_rank <= 5:
        return "minority"
    return "tail"


def quantile_bins(values: Sequence[float], n_bins: int) -> List[int]:
    """Assign 1-based quantile bins (1 = hardest / lowest value)."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return []
    edges = np.quantile(array, np.linspace(0, 1, n_bins + 1)[1:-1])
    return [int(b) + 1 for b in np.searchsorted(edges, array, side="right")]


def stratify_cell(pools: Sequence[Pool]) -> Dict[str, Any]:
    """Full D0 record for one (model, dataset) cell."""
    pass1 = [pool.pass_at_1 for pool in pools]
    terciles = quantile_bins(pass1, 3)
    quintiles = quantile_bins(pass1, 5)
    rows: List[Dict[str, Any]] = []
    for pool, tercile, quintile in zip(pools, terciles, quintiles):
        counts = list(pool.answer_counts().values())
        entropy = answer_entropy(counts)
        rows.append(
            {
                "qid": pool.qid,
                "pass_at_1": pool.pass_at_1,
                "snell_tercile": tercile,
                "snell_quintile": quintile,
                "tail": tail_heaviness(pool),
                "math_level": pool.metadata.get("level"),
                "subject": pool.metadata.get("subject"),
                "n_parsed": pool.size,
                "n_unparsed": pool.n_unparsed,
                **entropy,
            }
        )
    occupancy: Dict[str, Dict[str, int]] = {}
    for row in rows:
        bin_key = f"q{row['snell_quintile']}"
        occupancy.setdefault(bin_key, {label: 0 for label in TAIL_LABELS})
        occupancy[bin_key][row["tail"]] += 1
    underpowered = [
        f"{bin_key}/{label}"
        for bin_key, cells in occupancy.items()
        for label, count in cells.items()
        if 0 < count < 30
    ]
    return {
        "model": pools[0].model if pools else "",
        "dataset": pools[0].dataset if pools else "",
        "questions": rows,
        "occupancy_quintile_x_tail": occupancy,
        "underpowered_cells": underpowered,
        "mean_pass_at_1": float(np.mean(pass1)) if pass1 else 0.0,
        "unparsed_rate": float(
            sum(r["n_unparsed"] for r in rows)
            / max(1, sum(r["n_parsed"] + r["n_unparsed"] for r in rows))
        ),
    }


def snell_level_correlation(strata: Dict[str, Any]) -> Dict[str, Any]:
    """P-0a support: Snell-bin vs MATH-level occupancy and Spearman rho."""
    from scipy.stats import spearmanr

    rows = [row for row in strata["questions"] if row.get("math_level")]
    if len(rows) < 3:
        return {"heatmap": [], "spearman_rho": None, "p_value": None}
    levels = sorted({int(row["math_level"]) for row in rows})
    bins = sorted({int(row["snell_quintile"]) for row in rows})
    heatmap = [
        [
            sum(
                1
                for row in rows
                if int(row["snell_quintile"]) == b and int(row["math_level"]) == level
            )
            for level in levels
        ]
        for b in bins
    ]
    rho, p_value = spearmanr(
        [int(row["snell_quintile"]) for row in rows],
        [int(row["math_level"]) for row in rows],
    )
    return {
        "heatmap": heatmap,
        "bins": bins,
        "levels": levels,
        "spearman_rho": float(rho),
        "p_value": float(p_value),
    }
