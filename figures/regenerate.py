"""Regenerate every figure/table from cache-only rendering payloads.

Upstream analysis writes ``cache/figure_data/<HANDLE>.json``. This renderer
supports multipanel line, scatter, bar, and heatmap payloads and never embeds a
paper number in source code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from diversity_reasoning.constants import ROOT

if __package__:
    from .catalog import FIGURE_IDS, TABLE_IDS
else:
    from catalog import FIGURE_IDS, TABLE_IDS


def _load_payload(cache_dir: Path, handle: str) -> Dict[str, Any]:
    path = cache_dir / "figure_data" / f"{handle}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing cache rendering payload: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("handle") != handle:
        raise ValueError(f"{path} must be an object with handle={handle!r}")
    return payload


def _series(ax: Any, panel: Mapping[str, Any]) -> None:
    kind = panel.get("kind")
    if kind == "heatmap":
        image = ax.imshow(panel["values"], aspect="auto", cmap=panel.get("cmap", "viridis"))
        ax.figure.colorbar(image, ax=ax)
        ax.set_xticks(range(len(panel.get("x_labels", []))), panel.get("x_labels", []))
        ax.set_yticks(range(len(panel.get("y_labels", []))), panel.get("y_labels", []))
    else:
        for row in panel.get("series", []):
            label = row.get("label")
            color = "#777777" if label and label.lower() == "random" else None
            if kind == "line":
                ax.plot(row["x"], row["y"], label=label, color=color, marker=row.get("marker"))
                if "low" in row and "high" in row:
                    ax.fill_between(
                        row["x"],
                        row["low"],
                        row["high"],
                        color=color,
                        alpha=0.18,
                    )
            elif kind == "scatter":
                ax.scatter(row["x"], row["y"], label=label, color=color, alpha=0.7)
            elif kind == "bar":
                ax.bar(row["x"], row["y"], label=label, color=color)
            else:
                raise ValueError(f"Unsupported panel kind: {kind!r}")
    ax.set_title(panel.get("title", ""))
    ax.set_xlabel(panel.get("x_label", ""))
    ax.set_ylabel(panel.get("y_label", ""))
    if "y_lim" in panel:
        ax.set_ylim(*panel["y_lim"])
    if panel.get("legend", True) and panel.get("series"):
        ax.legend()


def render_figure(payload: Mapping[str, Any], destination: Path) -> None:
    import matplotlib.pyplot as plt

    style = ROOT / "figures" / "style.mplstyle"
    plt.style.use(style)
    panels = payload.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError("Figure payload requires a nonempty panels list")
    columns = int(payload.get("columns", min(2, len(panels))))
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        squeeze=False,
        figsize=payload.get("figsize", [5 * columns, 3.4 * rows]),
    )
    flat = axes.ravel()
    for axis, panel in zip(flat, panels):
        _series(axis, panel)
    for axis in flat[len(panels) :]:
        axis.set_visible(False)
    if payload.get("title"):
        figure.suptitle(payload["title"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination)
    plt.close(figure)


def render_table(payload: Mapping[str, Any], destination: Path) -> None:
    columns = payload.get("columns")
    rows = payload.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError("Table payload requires columns and rows lists")
    lines = [
        "| " + " | ".join(str(value) for value in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        if len(row) != len(columns):
            raise ValueError("Table row width does not match columns")
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selected(requested: Optional[Sequence[str]]) -> Iterable[str]:
    all_handles = (*FIGURE_IDS, *TABLE_IDS)
    if not requested:
        return all_handles
    unknown = sorted(set(requested) - set(all_handles))
    if unknown:
        raise ValueError(f"Unknown handles: {', '.join(unknown)}")
    return requested


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handle", action="append", dest="handles")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "generated")
    parser.add_argument("--list", action="store_true")
    arguments = parser.parse_args(argv)
    handles: List[str] = list(_selected(arguments.handles))
    if arguments.list:
        print("\n".join(handles))
        return 0
    for handle in handles:
        payload = _load_payload(arguments.cache_dir, handle)
        if handle.startswith("P-"):
            render_figure(payload, arguments.output_dir / f"{handle}.pdf")
        else:
            render_table(payload, arguments.output_dir / f"{handle}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
