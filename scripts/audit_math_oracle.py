"""T9b: audit the MATH equivalence oracle on REAL answer strings, not synthetic ones.

The unit tests (T9) check the canonicalizer on 21 hand-written pairs whose
answer is known. That proves the code does what it was written to do; it does
not measure how often it is *right* on the answers models actually produce.

This audits the partition on real chains. For a stratified sample of distinct
answer strings drawn from real MATH pools, every within-class and a matched set
of across-class pairs is re-adjudicated by an INDEPENDENT route: high-precision
numeric evaluation of both sides via sympy's parser, which shares no code with
the canonicalizer's string-normalisation path. Disagreements are printed in
full so they can be read rather than summarised.

The dangerous error is a FALSE MERGE (two genuinely different answers put in
one class), because it inflates the majority-vote mode and deflates answer-space
diversity. A false SPLIT is conservative in the opposite direction.
"""

from __future__ import annotations

import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diversity_reasoning.pools import MathCanonicalizer  # noqa: E402

SAMPLE_QUESTIONS = 40
MAX_PAIRS_PER_KIND = 400
SEED = 0


def independent_value(expr: str) -> Optional[complex]:
    """Numeric value of an answer string, via a route the canonicalizer does not use."""
    import sympy
    from sympy.parsing.latex import parse_latex

    text = expr.strip().strip("$").replace("\\!", "").replace("\\,", "").replace(" ", "")
    if not text:
        return None
    for parse in (lambda s: parse_latex(s), lambda s: sympy.sympify(s)):
        try:
            parsed = parse(text)
            value = complex(sympy.N(parsed, 30))
            if value == value:  # not NaN
                return value
        except Exception:
            continue
    return None


def close(a: complex, b: complex) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def main() -> int:
    rng = random.Random(SEED)
    canon = MathCanonicalizer()

    files = sorted(glob.glob("cache/gen/*/math/*/chains.jsonl"))
    if not files:
        print("no MATH chain banks found under cache/gen/*/math/")
        return 1
    rng.shuffle(files)

    classes: Dict[str, List[str]] = defaultdict(list)
    seen: set = set()
    for path in files[:SAMPLE_QUESTIONS]:
        with open(path) as handle:
            for line in handle:
                answer = json.loads(line).get("answer")
                if not answer or answer in seen:
                    continue
                seen.add(answer)
                classes[canon.canonical(answer)].append(answer)

    within: List[Tuple[str, str]] = []
    for members in classes.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                within.append((members[i], members[j]))
    rng.shuffle(within)
    within = within[:MAX_PAIRS_PER_KIND]

    keys = [k for k, v in classes.items() if v]
    across: List[Tuple[str, str]] = []
    for _ in range(MAX_PAIRS_PER_KIND * 4):
        if len(keys) < 2:
            break
        a, b = rng.sample(keys, 2)
        across.append((rng.choice(classes[a]), rng.choice(classes[b])))
    across = across[:MAX_PAIRS_PER_KIND]

    print(f"distinct real answer strings sampled : {len(seen)}")
    print(f"canonical classes formed             : {len(classes)}")
    print(f"pairs audited: {len(within)} within-class, {len(across)} across-class")
    print()

    report = {}
    for kind, pairs, oracle_says_equal in (
        ("within-class (oracle says EQUAL)", within, True),
        ("across-class (oracle says DIFFERENT)", across, False),
    ):
        adjudicated = agree = 0
        disagreements: List[Tuple[str, str]] = []
        for left, right in pairs:
            lv, rv = independent_value(left), independent_value(right)
            if lv is None or rv is None:
                continue
            adjudicated += 1
            if close(lv, rv) == oracle_says_equal:
                agree += 1
            else:
                disagreements.append((left, right))
        rate = agree / adjudicated if adjudicated else float("nan")
        label = "false SPLIT" if oracle_says_equal else "false MERGE"
        print(f"{kind}")
        print(f"  adjudicable pairs (both sides numerically parsable): {adjudicated}/{len(pairs)}")
        print(
            f"  agreement with the independent adjudicator        : "
            f"{agree}/{adjudicated} = {rate:.4f}"
        )
        print(f"  disagreements ({label}): {len(disagreements)}")
        for left, right in disagreements[:15]:
            print(f"    {left!r:<40} vs {right!r}")
        print()
        report[kind] = {
            "pairs": len(pairs),
            "adjudicable": adjudicated,
            "agreement": rate,
            "disagreements": len(disagreements),
            "examples": [list(p) for p in disagreements[:25]],
        }

    out = Path("cache/tables/t9b_oracle_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"distinct_answers": len(seen), "classes": len(classes), "results": report}, indent=2
    ))
    print(f"[write] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
