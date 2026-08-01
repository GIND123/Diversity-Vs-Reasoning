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


class TestT9bFalseMergeRegression:
    """Real answers that the earlier LaTeX normalizer silently merged into "0".

    The old normalizer matched ``\\frac`` with a regex that could not see nested
    braces, then stripped every remaining brace and backslash. The residue of
    ``\\frac{1}{2\\sqrt{10}}`` parsed as a call to sympy's own ``frac``
    (fractional part), which evaluates to 0, so the answer joined the class of
    the literal answer "0". Audited on real MATH banks this affected 1.33% of
    parsed chains overall and up to 43% of a single question's chains.

    A false merge is the dangerous direction: it inflates the modal answer class
    and deflates answer-space diversity, which is exactly what the head-to-head
    result is measured on.
    """

    @pytest.fixture(scope="class")
    def canonicalizer(self):
        return MathCanonicalizer()

    @pytest.mark.parametrize(
        "answer",
        [
            r"\frac{1}{2\sqrt{10}}",
            r"\frac{3}{\sqrt{10}}",
            r"\frac{1}{\sqrt{11}}",
            r"\frac{-4}{\log_{10}5}",
            r"\frac{1}{2(1+\sqrt{5})}",
            r"0\frac{2}{5}",
            r"24\frac{0}{1}",
            r"\text{0}",
            r"-\infty",
            r"\begin{pmatrix}0&-1\\1&0\end{pmatrix}",
        ],
    )
    def test_nonzero_answers_never_land_in_the_zero_class(self, canonicalizer, answer):
        assert canonicalizer.canonical(answer) != canonicalizer.canonical("0")

    def test_nested_brace_fractions_evaluate_correctly(self, canonicalizer):
        """Refusing to parse would also pass the test above; it must be right."""
        assert canonicalizer.canonical(r"\frac{1}{2\sqrt{10}}") == canonicalizer.canonical(
            r"\frac{\sqrt{10}}{20}"
        )
        assert canonicalizer.canonical(r"\frac{1}{2(1+\sqrt{5})}") == canonicalizer.canonical(
            r"\frac{\sqrt{5}-1}{8}"
        )

    def test_unhandled_latex_falls_back_to_the_string_itself(self, canonicalizer):
        """The conservative direction: split, never merge."""
        for answer in (r"-\infty", r"\text{None}", r"\log_{10}5"):
            assert canonicalizer.canonical(answer) == answer

    def test_ambiguous_mixed_numbers_are_refused_not_guessed(self, canonicalizer):
        """0\\frac{2}{5} is 2/5 to one writer and 0 to another; guessing gave 0."""
        assert canonicalizer.canonical(r"0\frac{2}{5}") == r"0\frac{2}{5}"
        assert canonicalizer.canonical(r"0\frac{2}{5}") != canonicalizer.canonical("0")

    def test_ordinary_equivalences_still_resolve(self, canonicalizer):
        """The fix must not buy safety by refusing everything."""
        assert canonicalizer.canonical(r"\frac{2}{4}") == canonicalizer.canonical("0.5")
        assert canonicalizer.canonical(r"\dfrac{1}{3}") == canonicalizer.canonical(
            r"\frac{2}{6}"
        )
        assert canonicalizer.canonical(r"\sqrt{4}") == canonicalizer.canonical("2")
