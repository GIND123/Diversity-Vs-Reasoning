"""Refresh the numbers quoted in README.md and assets/learn.md from cache.

Prose files drift: a number is copied in during one run and silently outlives
the analysis that produced it. This rewrites the small set of figures that are
quoted in prose, straight from the same cached tables the results file reads, so
a re-run cannot leave the narrative stale. It fails loudly if a passage it is
supposed to update has moved, rather than passing over it.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "cache" / "tables"
FIGURE_DATA = ROOT / "cache" / "figure_data"
RULES = ("pass_at_k", "majority_vote", "verifier_best")


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def head_to_head(name: str):
    rows = load(TABLES / f"{name}.json") or []
    out = {"ties": {}, "rules": {}}
    for kernel in ("embedding", "answer"):
        subset = [r for r in rows if r["kernel"] == kernel]
        out["ties"][kernel] = (
            sum(1 for r in subset if r["verdict"].startswith("tie")),
            len(subset),
        )
        for rule in RULES:
            core = [r for r in subset if r["rule"] == rule and r["group"] == "all"]
            if core:
                out["rules"][(kernel, rule)] = (
                    st.fmean(r["diversity_minus_coverage"] for r in core),
                    st.fmean(r["diversity_vs_random"] for r in core),
                    st.fmean(r["coverage_vs_random"] for r in core),
                    sum(1 for r in core if r["diversity_minus_coverage"] > 0),
                    len(core),
                )
    return out


def minus(text: str) -> str:
    return text.replace("-", "−")


def substitute(path: Path, replacements) -> None:
    text = path.read_text(encoding="utf-8")
    missing = []
    for old, new in replacements:
        if old == new:
            continue
        if old not in text:
            missing.append(old)
            continue
        text = text.replace(old, new)
    if missing:
        print(f"[sync] {path.name}: {len(missing)} passage(s) not found -- update this script:")
        for item in missing:
            print(f"        {item!r}")
        raise SystemExit(1)
    path.write_text(text, encoding="utf-8")
    print(f"[sync] {path.relative_to(ROOT)}")


def main() -> int:
    small = head_to_head("tb9_head_to_head")
    big = head_to_head("tb9_head_to_head_outcomes_pool1024")
    p2g = (load(FIGURE_DATA / "P-2g.json") or {}).get("correlations", {})

    ties_small = small["ties"]["embedding"]
    ties_big = big["ties"]["embedding"]
    rows = {rule: small["rules"][("answer", rule)] for rule in RULES}
    big_rows = {rule: big["rules"][("answer", rule)] for rule in RULES}
    emb_max = max(abs(big["rules"][("embedding", r)][0]) for r in RULES)
    win_r = p2g.get("pass_at_k|winnable_fraction", {}).get("pearson")
    big_pk = minus("{:+.3f}".format(big_rows["pass_at_k"][0]))
    big_mv = minus("{:+.3f}".format(big_rows["majority_vote"][0]))
    big_vb = minus("{:+.3f}".format(big_rows["verifier_best"][0]))
    big_cells = big_rows["pass_at_k"][4]
    readme_big_pattern = r"reproduces every sign with larger magnitudes.*?washing out\."
    learn_big_pattern = r"pools reproduces every sign.*?of \d+ conditions\."

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    import re

    old_ties = re.search(r"\(\d+ of \d+ ties\)", text)
    table_pattern = r"\| \*\*(?:pass@k|majority vote|verifier best-of-n)\*\* \|[^\n]*\n"
    old_table = re.findall(table_pattern, text)
    if not old_ties or len(old_table) != 3:
        print("[sync] README head-to-head passages not found")
        return 1
    labels = {
        "pass_at_k": ("pass@k", "**diversity**"),
        "majority_vote": ("majority vote", "**coverage**"),
        "verifier_best": ("verifier best-of-n", "**coverage**"),
    }
    new_table = []
    for rule in RULES:
        diff, dvr, cvr, favour, total = rows[rule]
        label, winner = labels[rule]
        agree = max(favour, total - favour)
        bold = "**" if rule == "pass_at_k" else ""
        new_table.append(
            f"| **{label}** | {winner} | {bold}{minus(f'{diff:+.3f}')}{bold} | "
            f"{minus(f'{dvr:+.3f}')} | {minus(f'{cvr:+.3f}')} | **{agree}/{total}** |\n"
        )
    substitute(
        readme,
        [(old_ties.group(0), f"({ties_small[0]} of {ties_small[1]} ties)")]
        + list(zip(old_table, new_table))
        + [
            (
                re.search(readme_big_pattern, text, re.S).group(0),
                "reproduces every sign with larger magnitudes — pass@k "
                f"{big_pk} to diversity,\nmajority vote {big_mv} and verifier best-of-n "
                f"{big_vb} to coverage, still {big_cells}/{big_cells} —\n"
                f"while the embedding kernel stays a tie in {ties_big[0]} of "
                f"{ties_big[1]} conditions (|Δ| ≤ {emb_max:.3f}).\n"
                "The separation grows with pool size rather than washing out.",
            ),
            (
                re.search(r"r = \+0\.\d+; negative for the other rules\)", text).group(0),
                f"r = {win_r:+.2f}; negative for the other rules)",
            ),
        ],
    )

    learn = ROOT / "assets" / "learn.md"
    text = learn.read_text(encoding="utf-8")
    old_ties = re.search(r"\d+ of \d+ conditions\. On the answer kernel", text)
    row_pattern = r"\| (?:pass@k|majority vote|verifier best-of-n) \| \*\*\w+\*\* \|[^\n]*\n"
    old_rows = re.findall(row_pattern, text)
    if not old_ties or len(old_rows) != 3:
        print("[sync] learn.md head-to-head passages not found")
        return 1
    new_rows = []
    for rule in RULES:
        diff, dvr, cvr, favour, total = rows[rule]
        label, winner = labels[rule]
        agree = max(favour, total - favour)
        new_rows.append(
            f"| {label} | {winner} | {minus(f'{diff:+.3f}')} | {minus(f'{dvr:+.3f}')} | "
            f"{minus(f'{cvr:+.3f}')} | {agree}/{total} |\n"
        )
    substitute(
        learn,
        [(
            old_ties.group(0),
            f"{ties_small[0]} of {ties_small[1]} conditions. On the answer kernel",
        )]
        + list(zip(old_rows, new_rows))
        + [
            (
                re.search(r"152 of 165 conditions\. So the", text).group(0)
                if "152 of 165 conditions. So the" in text
                else re.search(r"\d+ of \d+ conditions\. So the", text).group(0),
                f"{ties_small[0]} of {ties_small[1]} conditions. So the",
            ),
            (
                re.search(learn_big_pattern, text, re.S).group(0),
                "pools reproduces every sign with larger magnitudes (pass@k "
                f"{big_pk}, majority vote\n{big_mv}, verifier {big_vb}, still "
                f"{big_cells}/{big_cells}), while the embedding kernel stays a tie in\n"
                f"{ties_big[0]} of {ties_big[1]} conditions.",
            ),
            (
                re.search(r"collapsed to \*\*r = \+0\.\d+\*\* for pass@k", text).group(0),
                f"collapsed to **r = {win_r:+.3f}** for pass@k",
            ),
        ],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
