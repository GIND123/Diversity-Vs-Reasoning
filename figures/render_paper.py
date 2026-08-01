"""Render every paper figure (PDF + PNG) and table from cache payloads only.

Reads ``cache/figure_data`` and ``cache/tables`` (written by
``scripts/run_analysis.py``); writes ``figures/generated/{ID}.pdf|.png`` and
``figures/generated/tables/{ID}.md|.tex``. No number is ever hand-entered.

Color system (validated reference palette): the Vendi q-family rides one blue
ordinal ramp with per-q markers (one family, ordered by q); coverage is orange;
facility location is aqua; the random baseline is always the neutral gray band.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DATA = ROOT / "cache" / "figure_data"
TABLES_DATA = ROOT / "cache" / "tables"
OUT = ROOT / "figures" / "generated"
TABLES_OUT = OUT / "tables"

plt.style.use(ROOT / "figures" / "style.mplstyle")

# Ordinal blue ramp (reference palette, ordinal-legal steps) for the q family.
Q_ORDER = ("0", "0.1", "0.5", "1", "2", "inf")
Q_COLORS = dict(zip(Q_ORDER, ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]))
Q_MARKERS = dict(zip(Q_ORDER, ["o", "s", "D", "^", "v", "*"]))
COVERAGE = "#eb6834"
FACILITY = "#1baf7a"
RANDOM_LINE = "#52514e"
RANDOM_BAND = "#d8d7d3"
EPSILON_ARM = "#e87ba4"
ACCEPT = "#008300"
REJECT = "#e34948"
NEUTRAL = "#52514e"
SIGNAL_COLORS = {
    "answer_entropy": "#2a78d6",
    "vote_margin": "#eb6834",
    "mean_logprob": "#1baf7a",
    "embedding_vs1": "#eda100",
}
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "ref-blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

MODEL_LABEL = {
    "qwen2.5-0.5b": "Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen2.5-1.5B",
    "llama-3.2-3b": "Llama-3.2-3B",
}
DATASET_LABEL = {"gsm8k": "GSM8K", "math": "MATH"}
RULE_LABEL = {
    "majority_vote": "majority vote",
    "pass_at_k": "pass@k",
    "verifier_best": "verifier best-of-n",
}
HEADLINE_MODEL = "llama-3.2-3b"


def functional_style(key: str) -> Dict[str, Any]:
    if key.startswith("vs_"):
        q = key[3:]
        return {"color": Q_COLORS[q], "marker": Q_MARKERS[q], "label": f"VS$_{{{q}}}$"}
    if key == "pseudo_logdet":
        return {"color": COVERAGE, "marker": "P", "label": "coverage (pseudo log-det)"}
    return {"color": NEUTRAL, "marker": ".", "label": key}


def objective_style(name: str) -> Dict[str, Any]:
    if name.startswith("vendi_"):
        q = name.split("_", 1)[1]
        return {"color": Q_COLORS[q], "marker": Q_MARKERS[q], "label": f"greedy VS$_{{{q}}}$"}
    if name == "coverage":
        return {"color": COVERAGE, "marker": "P", "label": "greedy coverage"}
    if name == "facility_location":
        return {"color": FACILITY, "marker": "X", "label": "facility location"}
    return {"color": NEUTRAL, "marker": ".", "label": name}


def load(name: str) -> Optional[Any]:
    path = FIGURE_DATA / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_table(name: str) -> Optional[Any]:
    path = TABLES_DATA / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save(figure: plt.Figure, handle: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        figure.savefig(OUT / f"{handle}.{extension}")
    plt.close(figure)
    print(f"[figure] {handle}")


def cell_label(cell_id: str) -> str:
    model, dataset = cell_id.split("|")
    return f"{MODEL_LABEL.get(model, model)} / {DATASET_LABEL.get(dataset, dataset)}"


# --------------------------------------------------------------------------
# P-0: stratification context
# --------------------------------------------------------------------------


def render_p0() -> None:
    payload = load("P-0")
    if not payload:
        return
    math_cells = [
        c for c in payload if c.endswith("|math") and payload[c]["snell_vs_level"].get("heatmap")
    ]
    if math_cells:
        figure, axes = plt.subplots(1, len(math_cells), figsize=(3.0 * len(math_cells), 2.6))
        axes = np.atleast_1d(axes)
        for ax, cell in zip(axes, math_cells):
            data = payload[cell]["snell_vs_level"]
            matrix = np.asarray(data["heatmap"], dtype=float)
            image = ax.imshow(matrix, aspect="auto", cmap=SEQ_CMAP)
            ax.set_xticks(range(len(data["levels"])), [f"L{level}" for level in data["levels"]])
            ax.set_yticks(range(len(data["bins"])), [f"Q{b}" for b in data["bins"]])
            ax.set_xlabel("MATH level")
            ax.set_ylabel("Snell quintile (1 = hardest)")
            rho = data.get("spearman_rho")
            ax.set_title(f"{cell_label(cell)}\nSpearman $\\rho$ = {rho:.2f}")
            ax.grid(False)
            figure.colorbar(image, ax=ax, shrink=0.85, label="questions")
        save(figure, "P-0a")

    figure, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
    for ax, dataset in zip(axes, ("gsm8k", "math")):
        cells = [c for c in payload if c.endswith(f"|{dataset}")]
        for index, cell in enumerate(cells):
            entropies = payload[cell]["entropy"]
            ax.hist(
                entropies,
                bins=14,
                histtype="step",
                linewidth=1.6,
                color=list(MODEL_COLORS.values())[index % 3],
                label=MODEL_LABEL.get(cell.split("|")[0], cell),
            )
        ax.set_xlabel("answer entropy $H(p)$ (nats)")
        ax.set_title(DATASET_LABEL[dataset])
    axes[0].set_ylabel("questions")
    axes[0].legend()
    save(figure, "P-0b")


MODEL_COLORS = {
    "qwen2.5-0.5b": "#2a78d6",
    "qwen2.5-1.5b": "#eb6834",
    "llama-3.2-3b": "#1baf7a",
}


# --------------------------------------------------------------------------
# P-1a / P-1b / P-1c / P-1d: measurement
# --------------------------------------------------------------------------


def _plot_functional_curves(
    ax: plt.Axes,
    curves: Dict[str, List[Dict[str, float]]],
    keys: Sequence[str],
    x_label: str,
    log_x: bool = True,
) -> None:
    for key in keys:
        rows = curves.get(key, [])
        if not rows:
            continue
        style = functional_style(key)
        x = [row["x"] for row in rows]
        mean = [row["mean"] for row in rows]
        ax.plot(
            x,
            mean,
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            markersize=3.5,
        )
        ax.fill_between(
            x,
            [row["low"] for row in rows],
            [row["high"] for row in rows],
            color=style["color"],
            alpha=0.15,
            linewidth=0,
        )
    if log_x:
        ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel(x_label)


def render_p0c() -> None:
    """P-0c: the effective-number axiom, on synthetic pools with known truth.

    Left: N balanced dissimilar classes must score exactly N for every order q
    — the property that makes a Vendi score interpretable as an effective
    number, and the one the raw embedding kernel violated on real pools.
    Right: as one class takes over, the orders separate exactly as intended,
    low q holding near the class count and high q tracking the dominant mass.
    """
    payload = load("P-0c")
    if not payload:
        return
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    balanced = payload["balanced"]
    counts = [row["n_classes"] for row in balanced]
    axes[0].plot(
        counts, counts, linestyle="--", color=NEUTRAL, linewidth=1.0, label="ground truth (= N)"
    )
    for q in Q_ORDER:
        style = functional_style(f"vs_{q}")
        axes[0].plot(
            counts,
            [row[f"vs_{q}"] for row in balanced],
            marker=style["marker"],
            markersize=3.5,
            color=style["color"],
            label=style["label"],
            alpha=0.9,
        )
    axes[0].set_xlabel("number of balanced dissimilar classes $N$")
    axes[0].set_ylabel("VS$_q$")
    axes[0].set_title("Every order recovers $N$ exactly", fontsize=8.5)
    axes[0].legend(fontsize=6, ncol=2)

    imbalanced = payload["imbalanced"]
    dominance = [row["dominance"] for row in imbalanced]
    n_classes = payload["n_classes_imbalanced"]
    axes[1].axhline(
        n_classes, linestyle="--", color=NEUTRAL, linewidth=1.0, label=f"class count ({n_classes})"
    )
    for q in Q_ORDER:
        style = functional_style(f"vs_{q}")
        means = [row[f"vs_{q}"]["mean"] for row in imbalanced]
        axes[1].plot(
            dominance,
            means,
            marker=style["marker"],
            markersize=3.5,
            color=style["color"],
            label=style["label"],
        )
        axes[1].fill_between(
            dominance,
            [row[f"vs_{q}"]["low"] for row in imbalanced],
            [row[f"vs_{q}"]["high"] for row in imbalanced],
            color=style["color"],
            alpha=0.13,
            linewidth=0,
        )
    axes[1].set_xlabel("share of mass held by the dominant class")
    axes[1].set_ylabel("VS$_q$")
    axes[1].set_title("Orders separate under imbalance", fontsize=8.5)
    axes[1].legend(fontsize=6, ncol=2)
    figure.suptitle(
        "Effective-number axiom on synthetic pools with known ground truth", y=1.03, fontsize=9
    )
    save(figure, "P-0c")


def render_p1a() -> None:
    payload = load("P-1a")
    if not payload or not payload.get("synthetic"):
        return
    synthetic = payload["synthetic"]
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 2.7))
    vs_keys = [f"vs_{q}" for q in Q_ORDER]
    _plot_functional_curves(axes[0], synthetic["curves"], vs_keys, "rare-class prevalence")
    axes[0].set_ylabel("VS$_q$ (effective classes)")
    axes[0].set_title("Synthetic: diversity family")
    axes[0].legend(ncol=2, fontsize=6.5)
    _plot_functional_curves(
        axes[1], synthetic["curves"], ["pseudo_logdet"], "rare-class prevalence"
    )
    axes[1].set_ylabel("pseudo log-det")
    axes[1].set_title("Synthetic: coverage")
    axes[1].legend(fontsize=6.5)

    real = payload.get("real", {})
    keys = vs_keys + ["pseudo_logdet"]
    datasets = sorted({cell.split("|")[1] for cell in real})
    width = 0.8 / max(1, len(datasets))
    for d_index, dataset in enumerate(datasets):
        cells = [c for c in real if c.endswith(f"|{dataset}") and real[c].get("delta")]
        means = []
        for key in keys:
            values = [real[c]["delta"][key]["mean"] for c in cells if key in real[c]["delta"]]
            means.append(float(np.mean(values)) if values else 0.0)
        positions = np.arange(len(keys)) + (d_index - (len(datasets) - 1) / 2) * width
        colors = [functional_style(key)["color"] for key in keys]
        axes[2].bar(
            positions,
            means,
            width=width * 0.9,
            color=colors,
            alpha=1.0 if d_index == 0 else 0.55,
            label=DATASET_LABEL.get(dataset, dataset),
        )
    axes[2].axhline(0, color=NEUTRAL, linewidth=0.8)
    axes[2].set_xticks(
        range(len(keys)), [functional_style(k)["label"] for k in keys], rotation=45, ha="right"
    )
    axes[2].set_ylabel("$\\Delta$ on deleting correct chains")
    axes[2].set_title("Real: minority/tail deletion")
    axes[2].legend(fontsize=6.5)
    save(figure, "P-1a")


def render_p1b() -> None:
    """P-1b: what duplication does, split by duplication regime.

    Uniform duplication leaves the normalized spectrum untouched, so every
    functional is flat (harness T7). Skewed duplication concentrates mass, so
    VS_q falls — and coverage moves as well, which is the honest version of
    "coverage is duplication-invariant".
    """
    payload = load("P-1b")
    if not payload:
        return
    cell = next((c for c in payload if c.startswith(HEADLINE_MODEL)), None) or next(
        iter(payload), None
    )
    if cell is None:
        return
    modes = payload[cell].get("curves_by_mode")
    if not modes:
        modes = {"skewed": payload[cell].get("curves", {})}
    vs_keys = [f"vs_{q}" for q in Q_ORDER]
    labels = {"skewed": "skewed (random subset copied)", "uniform": "uniform (whole pool copied)"}
    order = [m for m in ("skewed", "uniform") if m in modes]
    figure, axes = plt.subplots(2, len(order), figsize=(3.6 * len(order), 5.0), squeeze=False)
    for column, mode in enumerate(order):
        curves = modes[mode]
        _plot_functional_curves(axes[0][column], curves, vs_keys, "duplicate rate", log_x=False)
        axes[0][column].set_ylabel("VS$_q$")
        axes[0][column].set_title(labels[mode], fontsize=8)
        _plot_functional_curves(
            axes[1][column], curves, ["pseudo_logdet"], "duplicate rate", log_x=False
        )
        axes[1][column].set_ylabel("pseudo log-det (coverage)")
    axes[0][0].legend(ncol=2, fontsize=6)
    figure.suptitle(f"Redundancy response — {cell_label(cell)}", y=1.02, fontsize=9)
    save(figure, "P-1b")


def render_p1c() -> None:
    payload = load("P-1c")
    if not payload:
        return
    cell = next((c for c in payload if c.startswith(HEADLINE_MODEL)), None) or next(
        iter(payload), None
    )
    if cell is None or not payload[cell].get("rows"):
        return
    rows = payload[cell]["rows"]
    keys = [f"vs_{q}" for q in Q_ORDER] + ["pseudo_logdet"]
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 2.6))
    for panel, (field, label) in enumerate(
        [
            ("relative_bias", "relative bias"),
            ("sd", "sd across seeds"),
            ("kendall_tau", "ranking $\\tau$ vs full pool"),
        ]
    ):
        for key in keys:
            series = sorted([(row["n"], row[field]) for row in rows if row["functional"] == key])
            if not series:
                continue
            style = functional_style(key)
            axes[panel].plot(
                [s[0] for s in series],
                [s[1] for s in series],
                color=style["color"],
                marker=style["marker"],
                markersize=3.5,
                label=style["label"],
            )
        axes[panel].set_xscale("log", base=2)
        axes[panel].set_xlabel("subsample size $n$")
        axes[panel].set_ylabel(label)
    axes[2].axhline(0.9, color=NEUTRAL, linewidth=0.8, linestyle="--")
    axes[0].legend(ncol=2, fontsize=6)
    figure.suptitle(cell_label(cell), y=1.04, fontsize=8)
    save(figure, "P-1c")


def render_p1d() -> None:
    payload = load("P-1d")
    if not payload:
        return
    cell = next((c for c in payload if c.startswith(HEADLINE_MODEL)), None) or next(
        iter(payload), None
    )
    if cell is None or not payload[cell].get("rows"):
        return
    data = payload[cell]
    rows = data["rows"]
    keys = [f"vs_{q}" for q in Q_ORDER]
    figure, axes = plt.subplots(1, 4, figsize=(12.4, 2.6))
    for key in keys:
        series = sorted([(r["d"], r["mean"]) for r in rows if r["functional"] == key])
        style = functional_style(key)
        axes[0].plot(
            [s[0] for s in series],
            [s[1] for s in series],
            color=style["color"],
            marker=style["marker"],
            markersize=3.5,
            label=style["label"],
        )
    axes[0].set_ylabel("VS$_q$")
    axes[0].legend(ncol=2, fontsize=6)
    series = sorted([(r["d"], r["mean"]) for r in rows if r["functional"] == "pseudo_logdet"])
    axes[1].plot(
        [s[0] for s in series], [s[1] for s in series], color=COVERAGE, marker="P", markersize=3.5
    )
    axes[1].set_ylabel("pseudo log-det")
    for key in keys + ["pseudo_logdet"]:
        series = sorted(
            [(r["d"], r["kendall_tau_vs_full"]) for r in rows if r["functional"] == key]
        )
        style = functional_style(key)
        axes[2].plot(
            [s[0] for s in series],
            [s[1] for s in series],
            color=style["color"],
            marker=style["marker"],
            markersize=3.5,
        )
    axes[2].set_ylabel("ranking $\\tau$ vs full $d$")
    axes[2].axhline(0.9, color=NEUTRAL, linewidth=0.8, linestyle="--")
    gap = data.get("answer_cosine_gap", {})
    dims = sorted(int(d) for d in gap)
    axes[3].plot(dims, [gap[str(d)] for d in dims], color="#4a3aa7", marker="o", markersize=3.5)
    axes[3].set_ylabel("same $-$ diff answer cosine gap")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("PCA dimension $d$")
    figure.suptitle(cell_label(cell), y=1.04, fontsize=8)
    save(figure, "P-1d")


# --------------------------------------------------------------------------
# P-1e / P-1f: the two log functionals
# --------------------------------------------------------------------------


def render_p1e() -> None:
    payload = load("P-1e")
    if not payload:
        return
    cell = next((c for c in payload if c.startswith(HEADLINE_MODEL)), None) or next(
        iter(payload), None
    )
    if cell is None:
        return
    data = payload[cell]
    scatter = data.get("scatter", {}).get("embedding") or data.get("scatter", {}).get("answer")
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    if scatter:
        x = [row["log_vs_1"] for row in scatter]
        y = [row["pseudo_logdet"] for row in scatter]
        color = [row["log10_lambda_min"] for row in scatter]
        points = axes[0].scatter(x, y, c=color, cmap=SEQ_CMAP, s=9, alpha=0.75, linewidths=0)
        figure.colorbar(points, ax=axes[0], label="$\\log_{10}\\lambda_{\\min}$")
        axes[0].set_xlabel("$\\log$ VS$_1$")
        axes[0].set_ylabel("pseudo log-det (coverage)")
        axes[0].set_title("Pooled across budgets")
    correlations = [
        c for c in data.get("correlations", []) if c["kernel"] == "embedding"
    ] or data.get("correlations", [])
    q_positions = list(range(len(Q_ORDER)))
    for field, label, color, marker in (
        ("within_budget_mean", "within budget", "#2a78d6", "o"),
        ("pooled", "pooled across budgets", COVERAGE, "s"),
        ("pooled_eps1", "pooled, $\\epsilon=1$ artifact", EPSILON_ARM, "D"),
    ):
        values = []
        for q in Q_ORDER:
            row = next((c for c in correlations if c["q"] == q), None)
            value = row.get(field) if row else None
            values.append(np.nan if value is None else value)
        axes[1].plot(q_positions, values, color=color, marker=marker, markersize=4, label=label)
    axes[1].axhline(0, color=NEUTRAL, linewidth=0.8)
    axes[1].set_xticks(q_positions, [f"q={q}" for q in Q_ORDER])
    axes[1].set_ylabel("corr($\\log$ VS$_q$, coverage)")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].legend(fontsize=6.5)
    axes[1].set_title("Simpson-style reversal")
    figure.suptitle(cell_label(cell), y=1.04, fontsize=8)
    save(figure, "P-1e")


def render_p1f() -> None:
    payload = load("P-1e")
    if not payload:
        return
    cell = next((c for c in payload if c.startswith(HEADLINE_MODEL)), None) or next(
        iter(payload), None
    )
    if cell is None:
        return
    reversal = payload[cell].get("reversal", {})
    figure, ax = plt.subplots(figsize=(3.6, 2.6))
    for index, (kernel, data) in enumerate(sorted(reversal.items())):
        curve = data.get("curve", [])
        if not curve:
            continue
        color = ["#2a78d6", COVERAGE][index % 2]
        ax.plot(
            [row["lambda_min_decile_mid"] for row in curve],
            [row["pearson"] for row in curve],
            marker="o",
            markersize=3.5,
            color=color,
            label=f"$K_{{{kernel[:3]}}}$",
        )
        flip = data.get("flip_log10_lambda_min")
        if flip is not None:
            ax.axvline(flip, color=color, linewidth=0.8, linestyle="--")
    ax.axhline(0, color=NEUTRAL, linewidth=0.8)
    ax.set_xlabel("$\\log_{10}\\lambda_{\\min}$ decile midpoint")
    ax.set_ylabel("pooled corr($\\log$ VS$_1$, coverage)")
    ax.legend()
    ax.set_title(f"Reversal onset — {cell_label(cell)}", fontsize=8)
    save(figure, "P-1f")


# --------------------------------------------------------------------------
# P-2a: winner curves (lead figure)
# --------------------------------------------------------------------------


def render_p2a(model: str = HEADLINE_MODEL, handle: str = "P-2a") -> None:
    payload = load("P-2a")
    if not payload:
        return
    rules = [r for r in ("majority_vote", "pass_at_k", "verifier_best")]
    datasets = [d for d in ("gsm8k", "math") if f"{model}|{d}|majority_vote" in payload]
    if not datasets:
        return
    figure, axes = plt.subplots(
        len(datasets),
        len(rules),
        figsize=(3.2 * len(rules), 2.5 * len(datasets)),
        sharex=True,
    )
    axes = np.atleast_2d(axes)
    for row_index, dataset in enumerate(datasets):
        for col_index, rule in enumerate(rules):
            ax = axes[row_index][col_index]
            data = payload.get(f"{model}|{dataset}|{rule}")
            if not data:
                continue
            budgets = data["budgets"]
            band = np.asarray(data["random_band"], dtype=float)
            ax.fill_between(
                budgets,
                band[:, 0],
                band[:, 1],
                color=RANDOM_BAND,
                alpha=0.6,
                linewidth=0,
                label="random (5-95%)",
            )
            ax.plot(
                budgets,
                data["random_mean"],
                color=RANDOM_LINE,
                linewidth=1.2,
                linestyle="--",
                label="random mean",
            )
            for objective in ("vendi_1", "vendi_inf", "coverage", "facility_location"):
                style = objective_style(objective)
                ax.plot(
                    budgets,
                    data["curves"][objective],
                    color=style["color"],
                    marker=style["marker"],
                    markersize=3.5,
                    label=style["label"],
                )
            ax.set_xscale("log", base=2)
            ax.set_xticks(budgets, [str(b) for b in budgets])
            if row_index == 0:
                ax.set_title(RULE_LABEL[rule])
            if col_index == 0:
                ax.set_ylabel(f"{DATASET_LABEL[dataset]}\naccuracy")
            if row_index == len(datasets) - 1:
                ax.set_xlabel("selection budget $k$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=6, fontsize=6.5, bbox_to_anchor=(0.5, 1.08)
    )
    figure.suptitle(MODEL_LABEL.get(model, model), y=1.14, fontsize=9)
    save(figure, handle)


# --------------------------------------------------------------------------
# P-2b: hypothesis strip
# --------------------------------------------------------------------------


def render_p2b() -> None:
    payload = load("P-2b")
    if not payload:
        return
    rows = [row for row in payload if row.get("delta") is not None]
    figure, ax = plt.subplots(figsize=(4.6, 0.5 + 0.42 * len(payload)))
    y_all = np.arange(len(payload))[::-1]
    for y, row in zip(y_all, payload):
        verdict = str(row.get("verdict", ""))
        color = (
            ACCEPT if verdict.startswith("accept") else (REJECT if verdict == "reject" else NEUTRAL)
        )
        if row.get("delta") is not None and row.get("ci"):
            ax.plot(row["ci"], [y, y], color=color, linewidth=1.6)
            ax.plot([row["delta"]], [y], marker="o", color=color, markersize=5)
        label = f"{row['id']}: {row['statement']}"
        ax.text(
            -0.015,
            y,
            label,
            ha="right",
            va="center",
            fontsize=6.8,
            transform=ax.get_yaxis_transform(),
        )
        ax.text(
            1.01,
            y,
            verdict or "n/a",
            ha="left",
            va="center",
            fontsize=6.8,
            color=color,
            transform=ax.get_yaxis_transform(),
        )
    ax.axvline(0, color=NEUTRAL, linewidth=0.8)
    ax.set_yticks([])
    ax.set_xlabel("$\\Delta$ accuracy vs random (95% CI)")
    ax.set_ylim(-0.6, len(payload) - 0.4)
    save(figure, "P-2b")
    del rows


# --------------------------------------------------------------------------
# P-2c / P-2e: tail conditioning
# --------------------------------------------------------------------------

TAIL_ORDER = ("modal", "minority", "tail", "absent")


def render_p2c() -> None:
    payload = load("P-2c")
    if not payload:
        return
    rules = ("majority_vote", "pass_at_k")
    datasets = sorted({row["dataset"] for row in payload})
    figure, axes = plt.subplots(
        len(datasets), len(rules), figsize=(3.4 * len(rules), 2.5 * len(datasets)), squeeze=False
    )
    for r_index, dataset in enumerate(datasets):
        for c_index, rule in enumerate(rules):
            ax = axes[r_index][c_index]
            rows = [
                row
                for row in payload
                if row["dataset"] == dataset
                and row["rule"] == rule
                and row["model"] == HEADLINE_MODEL
            ] or [row for row in payload if row["dataset"] == dataset and row["rule"] == rule]
            positions = np.arange(len(TAIL_ORDER))
            for o_index, objective in enumerate(("coverage", "vendi_inf")):
                style = objective_style(objective)
                deltas, err_low, err_high = [], [], []
                for label in TAIL_ORDER:
                    row = next(
                        (
                            r
                            for r in rows
                            if r["group"] == f"tail:{label}" and r["objective"] == objective
                        ),
                        None,
                    )
                    deltas.append(row["delta"] if row else np.nan)
                    err_low.append(row["delta"] - row["ci_low"] if row else 0)
                    err_high.append(row["ci_high"] - row["delta"] if row else 0)
                ax.bar(
                    positions + (o_index - 0.5) * 0.38,
                    deltas,
                    width=0.35,
                    color=style["color"],
                    label=style["label"],
                    yerr=[err_low, err_high],
                    error_kw={"linewidth": 0.8, "ecolor": NEUTRAL},
                )
            ax.axhline(0, color=NEUTRAL, linewidth=0.8)
            ax.set_xticks(positions, TAIL_ORDER)
            ax.set_title(f"{DATASET_LABEL[dataset]} — {RULE_LABEL[rule]}", fontsize=8)
            if c_index == 0:
                ax.set_ylabel("$\\Delta$ vs random")
    axes[0][0].legend(fontsize=6.5)
    save(figure, "P-2c")


def render_p2e() -> None:
    payload = load_table("r_conditioning_full")
    if not payload:
        return
    rows = [
        row
        for row in payload
        if row["model"] == HEADLINE_MODEL
        and row["rule"] == "majority_vote"
        and row["group"].startswith("tail:")
    ]
    if not rows:
        return
    objectives = [f"vendi_{q}" for q in Q_ORDER] + ["coverage", "facility_location"]
    figure, axes = plt.subplots(1, 4, figsize=(12.0, 2.5), sharey=True)
    for ax, label in zip(axes, TAIL_ORDER):
        group_rows = {row["objective"]: row for row in rows if row["group"] == f"tail:{label}"}
        deltas = [group_rows[o]["delta"] if o in group_rows else np.nan for o in objectives]
        colors = [objective_style(o)["color"] for o in objectives]
        errors = [
            [
                group_rows[o]["delta"] - group_rows[o]["ci_low"] if o in group_rows else 0
                for o in objectives
            ],
            [
                group_rows[o]["ci_high"] - group_rows[o]["delta"] if o in group_rows else 0
                for o in objectives
            ],
        ]
        ax.bar(
            range(len(objectives)),
            deltas,
            color=colors,
            yerr=errors,
            error_kw={"linewidth": 0.8, "ecolor": NEUTRAL},
        )
        ax.axhline(0, color=NEUTRAL, linewidth=0.8)
        ax.set_xticks(
            range(len(objectives)),
            [objective_style(o)["label"].replace("greedy ", "") for o in objectives],
            rotation=45,
            ha="right",
            fontsize=6,
        )
        ax.set_title(f"{label}", fontsize=8)
    axes[0].set_ylabel("$\\Delta$ vs random (MV)")
    figure.suptitle(
        f"{MODEL_LABEL[HEADLINE_MODEL]} — majority vote by tail heaviness", y=1.05, fontsize=8.5
    )
    save(figure, "P-2e")


# --------------------------------------------------------------------------
# P-2d: alpha threshold; P-2f: q inertness
# --------------------------------------------------------------------------


def render_p2d() -> None:
    payload = load("P-2d")
    if not payload:
        return
    rows = [row for row in payload if row["model"] == HEADLINE_MODEL] or payload
    datasets = sorted({row["dataset"] for row in rows})
    rules = ("majority_vote", "pass_at_k", "verifier_best")
    figure, axes = plt.subplots(
        len(datasets), len(rules), figsize=(3.2 * len(rules), 2.4 * len(datasets)), squeeze=False
    )
    for r_index, dataset in enumerate(datasets):
        for c_index, rule in enumerate(rules):
            ax = axes[r_index][c_index]
            for objective in ("vendi_1", "vendi_inf", "coverage"):
                row = next(
                    (
                        r
                        for r in rows
                        if r["dataset"] == dataset
                        and r["rule"] == rule
                        and r["objective"] == objective
                    ),
                    None,
                )
                if not row or not row.get("curve"):
                    continue
                style = objective_style(objective)
                alphas = [p["alpha"] for p in row["curve"]]
                ax.plot(
                    alphas,
                    [p["accuracy"] for p in row["curve"]],
                    color=style["color"],
                    marker=style["marker"],
                    markersize=3.5,
                    label=style["label"],
                )
                ax.plot(
                    alphas,
                    [p["random_accuracy"] for p in row["curve"]],
                    color=RANDOM_LINE,
                    linewidth=1.0,
                    linestyle="--",
                )
                if row.get("alpha_star") is not None:
                    ax.axvline(
                        row["alpha_star"], color=style["color"], linewidth=0.8, linestyle=":"
                    )
            if r_index == 0:
                ax.set_title(RULE_LABEL[rule])
            if c_index == 0:
                ax.set_ylabel(f"{DATASET_LABEL[dataset]}\naccuracy @ k={8}")
            ax.set_xlabel("$\\alpha$ (answer-awareness)")
    axes[0][0].legend(fontsize=6.5)
    figure.suptitle(MODEL_LABEL.get(HEADLINE_MODEL, ""), y=1.05, fontsize=9)
    save(figure, "P-2d")


def render_p2f() -> None:
    payload = load("P-2f")
    if not payload:
        return
    rows = [row for row in payload if row["model"] == HEADLINE_MODEL] or payload
    datasets = sorted({row["dataset"] for row in rows})
    rules = ("majority_vote", "pass_at_k")
    budgets = sorted({row["budget"] for row in rows})
    budget_colors = dict(
        zip(budgets, ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"])
    )
    figure, axes = plt.subplots(
        len(datasets), len(rules), figsize=(3.3 * len(rules), 2.4 * len(datasets)), squeeze=False
    )
    for r_index, dataset in enumerate(datasets):
        for c_index, rule in enumerate(rules):
            ax = axes[r_index][c_index]
            for budget in budgets:
                row = next(
                    (
                        r
                        for r in rows
                        if r["dataset"] == dataset and r["rule"] == rule and r["budget"] == budget
                    ),
                    None,
                )
                if not row:
                    continue
                values = [row["accuracy_by_q"].get(f"vendi_{q}", np.nan) for q in Q_ORDER]
                ax.plot(
                    range(len(Q_ORDER)),
                    values,
                    color=budget_colors[budget],
                    marker="o",
                    markersize=3,
                    label=f"k={budget}",
                )
            ax.set_xticks(range(len(Q_ORDER)), [f"{q}" for q in Q_ORDER])
            ax.set_xlabel("Vendi order $q$")
            ax.set_ylim(-0.02, 1.02)
            if r_index == 0:
                ax.set_title(RULE_LABEL[rule])
            if c_index == 0:
                ax.set_ylabel(f"{DATASET_LABEL[dataset]}\naccuracy on $K_{{ans}}$")
    axes[0][0].legend(fontsize=6, ncol=2)
    figure.suptitle(
        f"{MODEL_LABEL.get(HEADLINE_MODEL, '')} — q-inertness on the answer kernel (Theorem 4.1)",
        y=1.05,
        fontsize=8.5,
    )
    save(figure, "P-2f")


# --------------------------------------------------------------------------
# P-3: signals and escalation
# --------------------------------------------------------------------------


def render_p3a() -> None:
    payload = load("P-3a")
    if not payload:
        return
    cells = sorted(payload)
    figure, axes = plt.subplots(1, len(cells), figsize=(2.9 * len(cells), 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cell in zip(axes, cells):
        signals = payload[cell]["signals"]
        for name in ("answer_entropy", "vote_margin", "mean_logprob", "embedding_vs1"):
            data = signals.get(name)
            if not data:
                continue
            curve = data["curve"]
            ax.plot(
                [p["coverage"] for p in curve],
                [p["accuracy"] for p in curve],
                color=SIGNAL_COLORS[name],
                label=name.replace("_", " "),
                linewidth=1.4,
            )
        base = next(iter(signals.values()))["base_accuracy"] if signals else None
        if base is not None:
            ax.axhline(base, color=NEUTRAL, linewidth=0.8, linestyle="--")
        ax.set_xlabel("fraction answered")
        ax.set_title(cell_label(cell), fontsize=7.5)
    axes[0].set_ylabel("accuracy on answered set")
    axes[0].legend(fontsize=6)
    save(figure, "P-3a")


def render_p3b() -> None:
    payload = load("P-3b")
    if not payload:
        return
    models = sorted(payload)
    figure, ax = plt.subplots(figsize=(3.8, 2.6))
    positions = np.arange(len(models))
    pooled = [payload[m]["pooled_lift"] for m in models]
    within = [float(np.mean(payload[m]["within_dataset_lifts"])) for m in models]
    ax.bar(positions - 0.19, pooled, width=0.36, color="#eb6834", label="pooled across datasets")
    ax.bar(positions + 0.19, within, width=0.36, color="#2a78d6", label="within dataset (mean)")
    ax.axhline(0, color=NEUTRAL, linewidth=0.8)
    ax.set_xticks(positions, [MODEL_LABEL.get(m, m) for m in models], fontsize=7)
    ax.set_ylabel("risk-coverage lift of VS$_1$($K_{emb}$)")
    ax.set_title("Dataset-identity confound in the pooled lift", fontsize=8)
    ax.legend(fontsize=6.5)
    save(figure, "P-3b")


def render_p3c() -> None:
    payload = load("P-3c")
    if not payload:
        return
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    dataset_colors = {"gsm8k": "#2a78d6", "math": "#eb6834"}
    for dataset, data in sorted(payload.items()):
        if "curve" not in data:
            continue
        curve = data["curve"]
        color = dataset_colors.get(dataset, NEUTRAL)
        axes[0].plot(
            [p["fraction_answered_cheap"] for p in curve],
            [p["answered_set_accuracy"] for p in curve],
            color=color,
            label=DATASET_LABEL.get(dataset, dataset),
            linewidth=1.5,
        )
        axes[1].plot(
            [p["total_generated_tokens"] / 1e6 for p in curve],
            [p["overall_accuracy"] for p in curve],
            color=color,
            label=DATASET_LABEL.get(dataset, dataset),
            linewidth=1.5,
        )
        axes[1].axhline(data["expensive_accuracy"], color=color, linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("fraction answered by cheap model")
    axes[0].set_ylabel("answered-set accuracy")
    axes[0].legend(fontsize=6.5)
    axes[1].set_xlabel("total generated tokens (millions)")
    axes[1].set_ylabel("overall accuracy")
    axes[0].set_title("Entropy-gated escalation", fontsize=8)
    axes[1].set_title("Accuracy vs compute (dashed: escalate always)", fontsize=8)
    save(figure, "P-3c")


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def render_p2g() -> None:
    """P-2g: the share of winnable questions predicts the achievable gain.

    A selector cannot lose a question whose correct answer is already modal and
    cannot win one where it is absent. Only the present-but-not-modal group is
    winnable, so its share should bound what any objective can deliver.
    """
    payload = load("P-2g")
    if not payload or not payload.get("points"):
        return
    points = payload["points"]
    correlations = payload.get("correlations", {})
    rules = [r for r in ("majority_vote", "pass_at_k") if any(p["rule"] == r for p in points)]
    if not rules:
        return
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for ax, predictor, xlabel in zip(
        axes,
        ("winnable_fraction", "headroom"),
        ("winnable share (answer present, not modal)", "headroom (1 - random accuracy)"),
    ):
        for rule, marker in zip(rules, ("o", "s")):
            subset = [p for p in points if p["rule"] == rule]
            colour = Q_COLORS["1"] if rule == "majority_vote" else COVERAGE
            ax.errorbar(
                [p[predictor] for p in subset],
                [p["best_delta"] for p in subset],
                yerr=[
                    [p["best_delta"] - p["ci_low"] for p in subset],
                    [p["ci_high"] - p["best_delta"] for p in subset],
                ],
                fmt=marker,
                color=colour,
                markersize=5,
                linewidth=0,
                elinewidth=0.7,
                ecolor=NEUTRAL,
                label=RULE_LABEL[rule],
            )
            stats = correlations.get(f"{rule}|{predictor}")
            if stats:
                ax.plot([], [], " ", label=f"   r = {stats['pearson']:+.2f} (n={stats['n']})")
        ax.axhline(0, color=NEUTRAL, linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=0)
    axes[0].set_ylabel("best $\\Delta$ vs random (k=8)")
    axes[0].set_title("Winnable share", fontsize=8.5)
    axes[1].set_title("Headroom", fontsize=8.5)
    axes[0].legend(fontsize=6, loc="upper left")
    axes[1].legend(fontsize=6, loc="upper left")
    figure.suptitle(
        "What bounds the gain from selection: winnable share, not headroom",
        y=1.03,
        fontsize=9,
    )
    save(figure, "P-2g")


DIVERGING = LinearSegmentedColormap.from_list(
    "ref-diverging",
    ["#0d366b", "#2a78d6", "#86b6ef", "#f0efec", "#f2a682", "#eb6834", "#8f3410"],
)


def render_p4a() -> None:
    """P-4a: correlation matrix over the functionals and companion measures.

    Presentation follows the convention used for the Vendi family in the
    literature (Pasarkar & Dieng 2024, Fig. 5). Diverging palette on a neutral
    midpoint so sign reads at a glance; blue is positive, orange negative.
    """
    payload = load("P-4a")
    if not payload:
        return
    cells = [c for c in payload if payload[c].get("matrix")]
    if not cells:
        return
    show = cells[:2]
    figure, axes = plt.subplots(1, len(show), figsize=(5.4 * len(show), 4.6), squeeze=False)
    for ax, cell in zip(axes[0], show):
        data = payload[cell]
        matrix = np.asarray(data["matrix"], dtype=float)
        names = data["names"]
        image = ax.imshow(matrix, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(names)), names, fontsize=6)
        ax.grid(False)
        for i in range(len(names)):
            for j in range(len(names)):
                if np.isfinite(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=4.6,
                        color="white" if abs(matrix[i, j]) > 0.55 else "#0b0b0b",
                    )
        ax.set_title(f"{cell_label(cell)}  (n={data['n_questions']})", fontsize=8)
        figure.colorbar(image, ax=ax, shrink=0.8, label="Pearson r")
    figure.suptitle(
        "Correlations among the diversity orders, coverage, and outcome measures",
        y=1.02,
        fontsize=9,
    )
    save(figure, "P-4a")


def render_p4b() -> None:
    """P-4b: each functional against downstream question difficulty.

    Mirrors the Vendi-score-versus-external-evaluation panels in the literature
    (Pasarkar & Dieng 2024, Figs. 4 and 7): one scatter per functional with its
    correlation annotated.
    """
    payload = load("P-4b")
    if not payload:
        return
    cell = next((c for c in payload if c.startswith(HEADLINE_MODEL)), None) or next(
        iter(payload), None
    )
    if cell is None:
        return
    series = payload[cell]["series"]
    labels = [k for k in ("VS_1", "VS_inf", "coverage", "answer entropy") if k in series]
    figure, axes = plt.subplots(1, len(labels), figsize=(2.7 * len(labels), 2.7))
    axes = np.atleast_1d(axes)
    colours = {
        "VS_1": Q_COLORS["1"],
        "VS_inf": Q_COLORS["inf"],
        "coverage": COVERAGE,
        "answer entropy": "#4a3aa7",
    }
    for ax, label in zip(axes, labels):
        data = series[label]
        ax.scatter(
            data["x"], data["y"], s=11, alpha=0.65, linewidths=0, color=colours.get(label, NEUTRAL)
        )
        ax.set_xlabel(label)
        stat = data.get("pearson")
        title = f"r = {stat:+.2f}" if stat is not None and np.isfinite(stat) else "r = n/a"
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("pass@1 (full 1024-chain pool)")
    figure.suptitle(
        f"Do the functionals track question difficulty? — {cell_label(cell)} "
        f"(n={payload[cell]['n_questions']})",
        y=1.04,
        fontsize=8.5,
    )
    save(figure, "P-4b")


def render_pa4() -> None:
    """P-A4: the embedding kernel's concentration is question-specific.

    Left: spectral concentration and effective modes per correction arm.
    Right: how often two objectives pick the identical set — the direct measure
    of whether the winner map can distinguish selectors at all.
    """
    payload = load("P-A4")
    if not payload:
        return
    cell = next((c for c in payload if c.startswith(HEADLINE_MODEL)), None) or next(
        iter(payload), None
    )
    entry = payload.get(cell) or {}
    diagnostics = entry.get("diagnostics") or []
    if not diagnostics:
        return
    labels = [row["label"] for row in diagnostics]
    positions = np.arange(len(labels))
    colors = [COVERAGE if row["scope"] == "question" else Q_COLORS["1"] for row in diagnostics]

    figure, axes = plt.subplots(1, 3, figsize=(10.2, 2.7))
    axes[0].bar(positions, [row["top_eigenvalue_share"] for row in diagnostics], color=colors)
    axes[0].set_ylabel("top eigenvalue share of spectrum")
    axes[0].set_ylim(0, 1)
    axes[1].bar(positions, [row["mean_vs_1"] for row in diagnostics], color=colors)
    axes[1].set_ylabel("mean VS$_1$ (effective modes)")
    axes[2].bar(positions, [row["identical_selection_rate"] for row in diagnostics], color=colors)
    axes[2].set_ylabel("rate of identical selections")
    for ax in axes:
        ax.set_xticks(positions, labels, rotation=30, ha="right", fontsize=6.5)
    figure.suptitle(
        f"Embedding-kernel concentration is question-specific — {cell_label(cell)}",
        y=1.05,
        fontsize=8.5,
    )
    save(figure, "P-A4")
    _render_pa4_effects(entry, cell)


def _render_pa4_effects(entry: Dict[str, Any], cell: str) -> None:
    """P-A4b: effect sizes vs random under each kernel-correction arm."""
    rows = entry.get("effects") or []
    if not rows:
        return
    level_labels = {0: "raw", 1: "corpus c=1", -1: "question-centred"}
    levels = [c for c in (0, 1, -1) if any(row["components"] == c for row in rows)]
    objectives = ["vendi_1", "vendi_inf", "coverage", "facility_location"]
    rules = [r for r in ("majority_vote", "pass_at_k") if any(row["rule"] == r for row in rows)]
    figure, axes = plt.subplots(1, len(rules), figsize=(3.6 * len(rules), 2.7), squeeze=False)
    width = 0.8 / len(objectives)
    positions = np.arange(len(levels))
    for ax, rule in zip(axes[0], rules):
        for index, objective in enumerate(objectives):
            deltas, lows, highs = [], [], []
            for level in levels:
                row = next(
                    (
                        r
                        for r in rows
                        if r["components"] == level
                        and r["rule"] == rule
                        and r["objective"] == objective
                    ),
                    None,
                )
                deltas.append(row["delta"] if row else np.nan)
                lows.append(row["delta"] - row["ci_low"] if row else 0.0)
                highs.append(row["ci_high"] - row["delta"] if row else 0.0)
            style = objective_style(objective)
            ax.bar(
                positions + (index - (len(objectives) - 1) / 2) * width,
                deltas,
                width=width * 0.9,
                color=style["color"],
                label=style["label"],
                yerr=[lows, highs],
                error_kw={"linewidth": 0.7, "ecolor": NEUTRAL},
            )
        ax.axhline(0, color=NEUTRAL, linewidth=0.8)
        ax.set_xticks(positions, [level_labels[c] for c in levels], fontsize=7)
        ax.set_title(RULE_LABEL[rule], fontsize=8)
        ax.set_ylabel("$\\Delta$ vs random (k=8)")
    axes[0][0].legend(fontsize=6)
    figure.suptitle(f"Effect sizes by kernel correction — {cell_label(cell)}", y=1.05, fontsize=8.5)
    save(figure, "P-A4b")


def render_pa3() -> None:
    payload = load("P-A3")
    if not payload:
        return
    cells = sorted(payload)
    kernels = sorted({k for cell in cells for k in payload[cell]})
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    arm_colors = {"1e-8": "#eb6834", "1e-12": "#2a78d6"}
    positions = np.arange(len(cells))
    for ax, kernel in zip(axes, kernels[:2]):
        for offset, (arm, color) in enumerate(arm_colors.items()):
            values = [
                payload[cell]
                .get(kernel, {})
                .get("arms", {})
                .get(arm, {})
                .get("mean_abs_diff", np.nan)
                for cell in cells
            ]
            ax.bar(
                positions + (offset - 0.5) * 0.38,
                values,
                width=0.35,
                color=color,
                label=f"$\\tau$ = {arm}",
            )
        ax.set_xticks(
            positions, [cell_label(c) for c in cells], rotation=30, ha="right", fontsize=6
        )
        ax.set_ylabel("mean |$\\Delta$ pseudo log-det| vs $\\tau$=1e-10")
        ax.set_title(f"$K_{{{kernel[:3]}}}$", fontsize=8)
    axes[0].legend(fontsize=6.5)
    save(figure, "P-A3")


def _markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |\n"
    header += "|" + "|".join("---" for _ in columns) + "|\n"
    body = ""
    for row in rows:
        rendered = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            rendered.append(str(value))
        body += "| " + " | ".join(rendered) + " |\n"
    return header + body


def write_tables() -> None:
    TABLES_OUT.mkdir(parents=True, exist_ok=True)
    tb3 = load_table("tb3_winner_map")
    if tb3:
        for row in tb3:
            row["ci"] = f"[{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}]"
            row["delta"] = f"{row['delta']:+.3f}"
            row["winner_label"] = objective_style(row["winner"])["label"].replace("greedy ", "")
        columns = [
            "model",
            "dataset",
            "rule",
            "group",
            "winner_label",
            "delta",
            "ci",
            "p_holm",
            "accuracy",
            "random_accuracy",
            "n",
            "practically_null",
        ]
        (TABLES_OUT / "TB-3.md").write_text(
            "# TB-3: Winner map (budget k=8, embedding kernel)\n\n" + _markdown_table(tb3, columns),
            encoding="utf-8",
        )
        print("[table] TB-3")
    for name, handle in (
        ("tb0_strata", "TB-0"),
        ("tb1_sample_size", "TB-1"),
        ("tb2_correlations", "TB-2"),
        ("tb4_alpha_star", "TB-4"),
        ("tb5_signals", "TB-5"),
        ("tb6_operating_points", "TB-6"),
        ("tb7_encoder_stability", "TB-7"),
        ("seed_variance", "TB-8"),
    ):
        data = load_table(name)
        if data is None:
            continue
        (TABLES_OUT / f"{handle}.md").write_text(
            f"# {handle}\n\n```json\n" + json.dumps(data, indent=2, sort_keys=True) + "\n```\n",
            encoding="utf-8",
        )
        print(f"[table] {handle}")


def main() -> int:
    render_p0()
    render_p0c()
    render_p1a()
    render_p1b()
    render_p1c()
    render_p1d()
    render_p1e()
    render_p1f()
    render_p2a()
    for model in ("qwen2.5-0.5b", "qwen2.5-1.5b"):
        render_p2a(model, f"P-2a_{model}")
    render_p2b()
    render_p2c()
    render_p2e()
    render_p2d()
    render_p2f()
    render_p2g()
    render_p3a()
    render_p3b()
    render_p3c()
    render_p4a()
    render_p4b()
    render_pa3()
    render_pa4()
    write_tables()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
