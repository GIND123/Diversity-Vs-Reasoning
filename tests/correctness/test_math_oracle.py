"""T9: the MATH answer oracle, on hand-checked pairs.

The oracle decides both which chains share an answer class and which classes are
correct, so a silent failure here corrupts every MATH result at once. An earlier
subprocess-based implementation degraded to string comparison whenever the
worker could not start, which is exactly the failure these tests exist to catch.
"""

from __future__ import annotations

import pytest

from diversity_reasoning.pools import MathCanonicalizer, OracleUnavailable

pytestmark = pytest.mark.correctness


@pytest.fixture(scope="module")
def oracle() -> MathCanonicalizer:
    return MathCanonicalizer()


EQUIVALENT = [
    (r"\frac{1}{2}", "0.5"),
    (r"\frac{2}{4}", r"\frac{1}{2}"),
    ("2^3", "8"),
    (r"\sqrt{4}", "2"),
    ("x+x", "2x"),
    (r"\frac{-1}{2}", "-0.5"),
    ("2\\pi", "6.283185307179586"),
    ("1/4", "0.25"),
]

DISTINCT = [
    ("3", "4"),
    (r"\frac{1}{3}", "0.333"),
    ("x", "y"),
    ("2", "-2"),
    (r"\sqrt{2}", "1.41"),
]


@pytest.mark.parametrize(("left", "right"), EQUIVALENT)
def test_t9_equivalent_answers_are_recognized(oracle, left: str, right: str) -> None:
    assert oracle.equivalent(left, right) is True


@pytest.mark.parametrize(("left", "right"), DISTINCT)
def test_t9_distinct_answers_stay_distinct(oracle, left: str, right: str) -> None:
    assert oracle.equivalent(left, right) is False


@pytest.mark.parametrize(
    "group",
    [
        [r"\frac{1}{2}", "0.5", r"\frac{2}{4}"],
        ["2^3", "8", "8.0"],
        ["x+x", "2x"],
        [r"\sqrt{9}", "3"],
    ],
)
def test_t9_equivalent_answers_share_one_class_key(oracle, group) -> None:
    """Class keys drive the answer kernel; equivalent answers must not split."""
    assert len({oracle.canonical(value) for value in group}) == 1


def test_t9_distinct_answers_get_distinct_class_keys(oracle) -> None:
    assert oracle.canonical("3") != oracle.canonical("4")
    assert oracle.canonical("x") != oracle.canonical("y")


def test_t9_self_check_rejects_a_broken_oracle(monkeypatch) -> None:
    """A dead oracle must raise, never fall through to string comparison."""

    def broken(self, operation, payload):
        return None

    monkeypatch.setattr(MathCanonicalizer, "_evaluate", broken)
    with pytest.raises(OracleUnavailable):
        MathCanonicalizer()


def test_t9_unparseable_answers_are_counted_not_hidden(oracle) -> None:
    before = oracle.failures
    assert oracle.equivalent(r"\text{blue}", "$$$") is False
    assert oracle.failures > before


def test_t9_identical_strings_short_circuit(oracle) -> None:
    assert oracle.equivalent("anything at all", "anything at all") is True
