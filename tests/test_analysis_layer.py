"""Tests for pools, strata, spectra, winner protocol, and signals."""

from __future__ import annotations

import numpy as np
import pytest

from diversity_reasoning.metrics import block_kernel, pseudo_logdet, vendi_score
from diversity_reasoning.pools import Pool, build_pool
from diversity_reasoning.signals import question_signals, risk_coverage
from diversity_reasoning.spectra import answer_entropy, functionals, functionals_from_counts
from diversity_reasoning.strata import quantile_bins, stratify_cell, tail_heaviness
from diversity_reasoning.winner import (
    aggregate_outcomes,
    delta_versus_random,
    question_outcomes,
    random_orders,
    selection_orders,
)


def make_pool(
    answers,
    correct_answer="0",
    embeddings=None,
    logprobs=None,
):
    rows = [
        {
            "chain_id": index,
            "text": f"chain {index}",
            "answer": answer,
            "parsed": answer is not None,
            "logprob_sum": (logprobs[index] if logprobs else -float(index + 1)),
            "token_count": 10,
            "generation_seed": 0,
        }
        for index, answer in enumerate(answers)
    ]
    return build_pool(
        "q0",
        "gsm8k",
        "test-model",
        rows,
        correct_answer,
        embeddings=embeddings,
    )


class TestSpectraIdentities:
    def test_functionals_match_pinned_vendi_score(self) -> None:
        kernel = block_kernel([3, 2, 1])
        fast = functionals(kernel)
        for q in (0.0, 0.5, 1.0, 2.0, float("inf")):
            key = f"vs_{q:g}" if not np.isinf(q) else "vs_inf"
            assert fast[key] == pytest.approx(vendi_score(kernel, q), rel=1e-8)
        assert fast["pseudo_logdet"] == pytest.approx(pseudo_logdet(kernel), rel=1e-10)

    def test_counts_path_matches_kernel_path(self) -> None:
        counts = [5, 3, 2]
        from_counts = functionals_from_counts(counts)
        from_kernel = functionals(block_kernel(counts))
        for key, value in from_counts.items():
            assert value == pytest.approx(from_kernel[key], abs=1e-8), key

    def test_entropy_normalization(self) -> None:
        flat = answer_entropy([5, 5, 5, 5])
        assert flat["normalized_entropy"] == pytest.approx(1.0)
        assert answer_entropy([10])["entropy"] == 0.0


class TestE5Degeneracy:
    def _rows(self, vs_values, logdets, budget=8, kernel="embedding"):
        return [
            {
                "qid": f"q{i}",
                "kernel": kernel,
                "budget": budget,
                "vs_0": vs,
                "vs_0.1": vs,
                "vs_0.5": vs,
                "vs_1": vs,
                "vs_2": vs,
                "vs_inf": vs,
                "pseudo_logdet": logdet,
                "logdet_eps1": logdet,
                "lambda_min": 1e-6,
                "lambda_max": 1.0,
            }
            for i, (vs, logdet) in enumerate(zip(vs_values, logdets))
        ]

    def test_constant_functional_reports_undefined_not_nan(self) -> None:
        """VS_0 on K_emb saturates at the budget; the correlation is undefined."""
        from diversity_reasoning.measurement import e5_two_log_functionals

        rows = self._rows([8.0] * 8, [float(-i) for i in range(8)])
        result = e5_two_log_functionals(rows)
        entry = next(c for c in result["correlations"] if c["q"] == "1")
        assert entry["within_budget_mean"] is None
        assert entry["degenerate_budgets"] == 1
        assert "saturates" in entry["note"]

    def test_float_noise_still_counts_as_constant(self) -> None:
        from diversity_reasoning.measurement import e5_two_log_functionals

        noisy = [8.0 + i * 1e-14 for i in range(8)]
        rows = self._rows(noisy, [float(-i) for i in range(8)])
        entry = next(c for c in e5_two_log_functionals(rows)["correlations"] if c["q"] == "1")
        assert entry["within_budget_mean"] is None

    def test_varying_functional_gives_a_correlation(self) -> None:
        """The statistic correlates log VS_q against coverage, so an exactly
        log-linear fixture must return r = 1."""
        from diversity_reasoning.measurement import e5_two_log_functionals

        rows = self._rows([float(np.exp(i)) for i in range(8)], [float(i) for i in range(8)])
        entry = next(c for c in e5_two_log_functionals(rows)["correlations"] if c["q"] == "1")
        assert entry["within_budget_mean"] == pytest.approx(1.0, abs=1e-6)
        assert entry["degenerate_budgets"] == 0
        assert entry["within_budget_n"] == 1


class TestPools:
    def test_unparsed_chains_are_excluded_symmetrically(self) -> None:
        pool = make_pool(["1", None, "2", "1"], correct_answer="1")
        assert pool.size == 3
        assert pool.n_unparsed == 1
        assert pool.pass_at_1 == pytest.approx(2 / 3)
        assert pool.answer_counts() == {0: 2, 1: 1}

    def test_answer_kernel_blocks(self) -> None:
        pool = make_pool(["1", "2", "1"], correct_answer="2")
        kernel = pool.kernel("answer")
        assert kernel[0, 2] == 1.0 and kernel[0, 1] == 0.0

    def test_mixed_kernel_interpolates(self) -> None:
        embeddings = np.eye(3)
        pool = make_pool(["1", "2", "1"], embeddings=embeddings)
        mixed = pool.kernel("mixed", 0.5)
        emb = pool.kernel("embedding")
        ans = pool.kernel("answer")
        assert np.allclose(mixed, 0.5 * ans + 0.5 * emb)


class TestAnisotropy:
    def _anisotropic_pools(self, n=24, d=8, shift=6.0):
        """Pools whose embeddings share one dominant common direction."""
        rng = np.random.default_rng(0)
        common = np.zeros(d)
        common[0] = 1.0
        pools = []
        for _ in range(3):
            content = rng.normal(size=(n, d))
            content[:, 0] = 0.0
            embeddings = content + shift * common
            pools.append(
                make_pool([str(i % 4) for i in range(n)], correct_answer="0", embeddings=embeddings)
            )
        from diversity_reasoning.pools import fit_corpus_directions

        fit_corpus_directions(pools, components=3, per_question=n)
        return pools

    def test_removing_common_direction_restores_diversity(self) -> None:
        """The failure this guards against: one shared direction dominates the
        spectrum, pinning VS_q near 1 however varied the chains actually are."""
        pool = self._anisotropic_pools()
        raw_kernel = pool[0].kernel("embedding", components=0)
        corrected_kernel = pool[0].kernel("embedding", components=1)

        def top_share(kernel):
            eigenvalues = np.clip(np.linalg.eigvalsh(kernel / kernel.shape[0]), 0, None)
            return float(eigenvalues.max() / eigenvalues.sum())

        raw_share = top_share(raw_kernel)
        corrected_share = top_share(corrected_kernel)
        assert raw_share > 0.6, "fixture must actually be anisotropic"
        assert raw_share - corrected_share > 0.25
        assert functionals(corrected_kernel)["vs_1"] > 2 * functionals(raw_kernel)["vs_1"]

    def test_corpus_directions_are_shared_across_pools(self) -> None:
        pools = self._anisotropic_pools()
        first = pools[0].corpus_directions
        assert first is not None
        for pool in pools[1:]:
            assert pool.corpus_directions is first

    def test_kernel_cache_separates_component_levels(self) -> None:
        pool = self._anisotropic_pools()[0]
        k0 = pool.kernel("embedding", components=0)
        k1 = pool.kernel("embedding", components=1)
        assert not np.allclose(k0, k1)
        assert pool.kernel("embedding", components=0) is k0

    def test_question_centring_beats_corpus_correction_on_shared_content(self) -> None:
        """Every chain in a pool answers the same question, so the pool's own
        centroid — not a corpus-wide direction — is what flattens its spectrum.

        Many questions with distinct topic directions, so no single topic sits
        among the leading corpus directions. That is the real setting: removing
        a few corpus directions cannot remove 96 different question topics, but
        question-centring removes each pool's own.
        """
        rng = np.random.default_rng(1)
        n_questions, n, d = 12, 32, 24
        topics = rng.normal(size=(n_questions, d))
        topics /= np.linalg.norm(topics, axis=1, keepdims=True)
        pools = [
            make_pool(
                [str(i % 4) for i in range(n)],
                correct_answer="0",
                embeddings=rng.normal(size=(n, d)) + 9.0 * topics[q],
            )
            for q in range(n_questions)
        ]
        from diversity_reasoning.pools import fit_corpus_directions

        fit_corpus_directions(pools, components=3, per_question=n)
        pool = pools[0]
        raw = functionals(pool.kernel("embedding", components=0))["vs_1"]
        corpus = functionals(pool.kernel("embedding", components=1))["vs_1"]
        centred = functionals(pool.kernel("embedding_qc", components=1))["vs_1"]
        assert centred > 2 * corpus, "question-centring must dominate the corpus fix"
        assert corpus < 2 * raw, "corpus correction alone barely helps here"

    def test_question_centring_needs_no_corpus_fit(self) -> None:
        rng = np.random.default_rng(2)
        pool = make_pool([str(i % 3) for i in range(12)], embeddings=rng.normal(size=(12, 6)) + 5.0)
        assert pool.corpus_directions is None
        assert pool.kernel("embedding_qc", components=1).shape == (12, 12)

    def test_missing_directions_raise(self) -> None:
        pool = make_pool(["1", "2"], embeddings=np.eye(2))
        with pytest.raises(ValueError, match="corpus directions"):
            pool.kernel("embedding", components=1)


class TestCommonBudgetSelection:
    """Cross-question comparisons must fix the budget.

    Taking each question's own largest budget makes VS_0 a proxy for pool size
    (r = 1.000 measured on MATH) and manufactures a -0.99 "anticorrelation"
    between diversity and coverage that is pure size confound.
    """

    def _rows(self, spec):
        return [
            {"qid": qid, "kernel": "embedding", "budget": b, "vs_1": 1.0}
            for qid, budgets in spec.items()
            for b in budgets
        ]

    def _select(self, spec):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "scripts" / "run_analysis.py"
        loader = importlib.util.spec_from_file_location("run_analysis_mod", path)
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        return module._common_budget_spectra(self._rows(spec))

    def test_all_questions_share_one_budget(self) -> None:
        chosen = self._select({"a": [128, 512, 1024], "b": [128, 512], "c": [128, 512]})
        assert len({row["budget"] for row in chosen.values()}) == 1

    def test_one_small_pool_does_not_drag_everyone_down(self) -> None:
        """A single 128-chain outlier must not force the whole cell to 128."""
        spec = {f"q{i}": [128, 512, 1024] for i in range(20)}
        spec["tiny"] = [128]
        chosen = self._select(spec)
        budget = next(iter(chosen.values()))["budget"]
        assert budget == 1024
        assert "tiny" not in chosen
        assert len(chosen) == 20

    def test_returns_empty_when_nothing_shared(self) -> None:
        assert self._select({}) == {}


class TestCacheRelease:
    """Per-pool caches must be droppable; retaining them across a large cell
    exceeded RAM and made a 192-question run 30x slower than a 96-question one."""

    def test_release_clears_kernel_and_embedding_caches(self) -> None:
        rng = np.random.default_rng(9)
        pool = make_pool([str(i % 3) for i in range(12)], embeddings=rng.normal(size=(12, 6)))
        first = pool.kernel("embedding_qc", components=1)
        pool.question_centered_embeddings(1)
        assert pool._kernel_cache and pool._embedding_cache
        pool.release_caches()
        assert not pool._kernel_cache and not pool._embedding_cache
        rebuilt = pool.kernel("embedding_qc", components=1)
        assert rebuilt is not first
        assert np.allclose(rebuilt, first), "released caches must rebuild identically"


class TestDuplicationRegimes:
    """T7 has two halves; E2 must report both rather than blur them."""

    def _pool(self, n=60, d=12):
        rng = np.random.default_rng(3)
        return make_pool(
            [str(i % 5) for i in range(n)],
            correct_answer="0",
            embeddings=rng.normal(size=(n, d)) + 4.0,
        )

    def test_uniform_duplication_leaves_every_functional_fixed(self) -> None:
        from diversity_reasoning.kernels import embedding_kernel

        embeddings = self._pool().question_centered_embeddings(1)
        base = functionals(embedding_kernel(embeddings)[0])
        tripled = functionals(embedding_kernel(np.tile(embeddings, (3, 1)))[0])
        for key in ("vs_1", "vs_inf", "pseudo_logdet"):
            assert tripled[key] == pytest.approx(base[key], abs=1e-6), key

    def test_skewed_duplication_moves_coverage_too(self) -> None:
        """Coverage is invariant only under uniform duplication, not in general."""
        from diversity_reasoning.kernels import embedding_kernel

        embeddings = self._pool().question_centered_embeddings(1)
        rng = np.random.default_rng(4)
        rows = rng.integers(0, embeddings.shape[0], size=30)
        skewed = np.concatenate([embeddings, embeddings[rows]])
        base = functionals(embedding_kernel(embeddings)[0])
        after = functionals(embedding_kernel(skewed)[0])
        assert after["vs_1"] < base["vs_1"]
        assert abs(after["pseudo_logdet"] - base["pseudo_logdet"]) > 1e-6

    def test_e2_returns_both_regimes(self) -> None:
        from diversity_reasoning.measurement import e2_duplication

        result = e2_duplication([self._pool()], base_size=40, n_questions=1)
        assert set(result["curves_by_mode"]) == {"skewed", "uniform"}
        uniform = result["curves_by_mode"]["uniform"]["pseudo_logdet"]
        assert uniform[0]["mean"] == pytest.approx(uniform[-1]["mean"], abs=1e-6)


class TestStrata:
    def test_tail_labels(self) -> None:
        assert tail_heaviness(make_pool(["7"] * 6 + ["1"], correct_answer="7")) == "modal"
        assert tail_heaviness(make_pool(["1"] * 5 + ["7"] * 2, correct_answer="7")) == "minority"
        assert tail_heaviness(make_pool(["1", "2"], correct_answer="9")) == "absent"
        spread = [str(i) for i in range(6)] * 3 + ["9"]
        assert tail_heaviness(make_pool(spread, correct_answer="9")) == "tail"

    def test_quantile_bins_are_balanced(self) -> None:
        bins = quantile_bins(list(range(30)), 3)
        assert sorted(set(bins)) == [1, 2, 3]
        assert bins[0] == 1 and bins[-1] == 3

    def test_stratify_cell_flags_underpowered(self) -> None:
        pools = [make_pool(["1", "2", "1"], correct_answer="1") for _ in range(6)]
        record = stratify_cell(pools)
        assert record["questions"][0]["tail"] == "modal"
        assert record["underpowered_cells"]  # 6 questions can't reach 30 anywhere


class TestWinnerProtocol:
    def test_vendi_selects_distinct_classes_coverage_concentrates(self) -> None:
        """The mechanistic contrast behind P-2c: on a block kernel the Vendi
        selectors spread across answer classes while greedy pseudo-logdet is
        duplication-attracted (zero eigenvalues are excluded, so duplicates
        cost nothing)."""
        kernel = block_kernel([4, 2, 2])
        orders = selection_orders(kernel, ["vendi_1", "coverage"], 3)
        vendi_classes = {0 if i < 4 else (1 if i < 6 else 2) for i in orders["vendi_1"]}
        assert vendi_classes == {0, 1, 2}
        coverage_classes = {0 if i < 4 else (1 if i < 6 else 2) for i in orders["coverage"]}
        assert coverage_classes == {0}

    def test_q_inertness_on_answer_kernel_at_class_budget(self) -> None:
        """Budget <= #classes: every q order selects one chain per class (T5 twin)."""
        kernel = block_kernel([5, 3, 2, 1])
        orders = selection_orders(kernel, [f"vendi_{q}" for q in ("0", "0.5", "1", "2", "inf")], 4)
        reference = sorted(next(iter(orders.values())))
        for order in orders.values():
            assert sorted(order) == reference

    def test_aggregate_outcomes_rules(self) -> None:
        pool = make_pool(["1", "1", "2", "3"], correct_answer="2", logprobs=[-1, -2, -3, -4])
        outcomes = aggregate_outcomes(pool, [0, 1, 2], budgets=[2, 3])
        assert outcomes[3]["majority_vote"] is False  # class "1" wins the vote
        assert outcomes[3]["pass_at_k"] is True  # "2" is present
        assert outcomes[3]["verifier_best"] is False  # chain 0 has best logprob
        assert outcomes[2]["tie"] is False  # {"1","1"} unanimous

    def test_tie_breaks_by_mean_logprob(self) -> None:
        pool = make_pool(["1", "2"], correct_answer="2", logprobs=[-5.0, -1.0])
        outcomes = aggregate_outcomes(pool, [0, 1], budgets=[2])
        assert outcomes[2]["tie"] is True
        assert outcomes[2]["majority_vote"] is True  # "2" has higher mean logprob

    def test_random_orders_are_seeded_and_uniform_prefix(self) -> None:
        first = random_orders(10, 4, n_seeds=3)
        second = random_orders(10, 4, n_seeds=3)
        assert first == second
        assert all(len(set(order)) == 4 for order in first)

    def test_outcomes_average_over_subsample_seeds(self) -> None:
        """Treatment and baseline must see the same draws.

        Averaging only the random arm leaves each treatment arm carrying the
        noise of one arbitrary subsample, which on a small stratum was large
        enough to flip the sign of a 'significant' effect.
        """
        embeddings = np.random.default_rng(5).normal(size=(60, 8))
        pool = make_pool([str(i % 6) for i in range(60)], correct_answer="0", embeddings=embeddings)
        record = question_outcomes(
            pool,
            pool_size=20,
            kernels=["answer"],
            objectives=["vendi_1"],
            subsample_seeds=(0, 1, 2, 3),
        )
        assert record["subsample_seeds"] == [0, 1, 2, 3]
        cell = record["cells"]["answer|vendi_1"]
        value = cell[8]["pass_at_k"]
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0
        # Four subsample draws x 20 random seeds all land in the baseline.
        assert len(record["cells"]["random"]) == 80

    def test_full_pool_ignores_extra_subsample_seeds(self) -> None:
        """With no subsampling there is nothing to average over."""
        pool = make_pool([str(i % 3) for i in range(12)], correct_answer="0")
        record = question_outcomes(
            pool,
            pool_size=10**9,
            kernels=["answer"],
            objectives=["vendi_1"],
            subsample_seeds=(0, 1, 2),
        )
        assert record["subsample_seeds"] == [0]
        assert len(record["cells"]["random"]) == 20

    def test_question_outcomes_shape(self) -> None:
        embeddings = np.random.default_rng(0).normal(size=(12, 6))
        pool = make_pool([str(i % 3) for i in range(12)], correct_answer="0", embeddings=embeddings)
        record = question_outcomes(pool, pool_size=40, kernels=["answer", "embedding"])
        assert "answer|vendi_1" in record["cells"]
        assert "embedding|coverage" in record["cells"]
        assert len(record["cells"]["random"]) == 20
        budgets = set(record["cells"]["answer|vendi_1"])
        assert budgets == {2, 3, 4, 8}  # pool of 12 caps the budget ladder

    def test_delta_versus_random_detects_signal(self) -> None:
        treated = [1.0] * 40
        baseline = [0.5] * 40
        stats = delta_versus_random(treated, baseline)
        assert stats["delta"] == pytest.approx(0.5)
        assert stats["ci_low"] == pytest.approx(0.5)
        assert stats["p"] < 0.01
        null = delta_versus_random([0.5] * 40, [0.5] * 40)
        assert null["practically_null"] is True

    def test_bootstrap_p_never_reports_exact_zero(self) -> None:
        """1000 replicates cannot resolve below p = 1/1000; reporting 0 overclaims."""
        stats = delta_versus_random([1.0] * 60, [0.0] * 60, replicates=1000)
        assert stats["p"] == pytest.approx(1e-3)
        coarse = delta_versus_random([1.0] * 60, [0.0] * 60, replicates=200)
        assert coarse["p"] == pytest.approx(5e-3)


class TestSignals:
    def test_risk_coverage_perfect_signal(self) -> None:
        confidences = [3.0, 2.0, 1.0, 0.0]
        outcomes = [True, True, False, False]
        result = risk_coverage(confidences, outcomes, grid=4)
        assert result["curve"][0]["accuracy"] == 1.0
        assert result["base_accuracy"] == 0.5
        assert result["lift"] > 0

    def test_question_signals_fields(self) -> None:
        pool = make_pool(["1", "1", "2"], correct_answer="1")
        signals = question_signals(pool)
        assert signals["mv_correct"] is True
        assert signals["vote_margin"] == pytest.approx(1 / 3)
        assert signals["answer_entropy"] > 0
        assert signals["embedding_vs1"] is None


class TestPoolBudgetSubsample:
    def test_subsample_is_deterministic(self) -> None:
        pool = make_pool([str(i) for i in range(50)])
        assert pool.subsample(8, 0) == pool.subsample(8, 0)
        assert pool.subsample(8, 0) != pool.subsample(8, 1)


def test_pool_dataclass_is_reusable() -> None:
    pool = make_pool(["1", "2"])
    assert isinstance(pool, Pool)
    first = pool.kernel("answer")
    assert pool.kernel("answer") is first  # cached
