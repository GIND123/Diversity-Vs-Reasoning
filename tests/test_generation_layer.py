"""Tests for the generation, prompt, question-bank, and Hub-layout layers."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from diversity_reasoning.datasets import (
    Question,
    gsm8k_question,
    gsm8k_reference,
    math_level,
    math_question,
    math_reference,
    questions_from_rows,
    select_questions,
)
from diversity_reasoning.generation import (
    GenerationSettings,
    bank_manifest,
    build_chain_records,
    request_seed,
    request_seeds,
    shard_plan,
    unparsed_rate,
)
from diversity_reasoning.hf_bank import (
    CHAIN_FIELDS,
    decode_shard,
    encode_shard,
    manifest_remote_path,
    materialize_shard,
    questions_remote_path,
    read_cached_bank,
    shard_remote_path,
    shard_rows,
)
from diversity_reasoning.prompts import (
    PROMPT_TEMPLATE_VERSION,
    chat_messages,
    dataset_key,
    user_prompt,
)

GSM8K_ROW = {
    "question": "Janet has 3 apples and buys 4 more. How many does she have?",
    "answer": "3 + 4 = <<3+4=7>>7\n#### 7",
}
MATH_ROW = {
    "problem": "What is $2+2$?",
    "level": "Level 3",
    "type": "Algebra",
    "solution": "Adding gives $\\boxed{4}$.",
}


class TestPrompts:
    def test_dataset_key_normalizes_variants(self) -> None:
        assert dataset_key("GSM8K") == "gsm8k"
        assert dataset_key("openai/gsm8k") == "gsm8k"
        assert dataset_key("EleutherAI/hendrycks_math") == "math"
        with pytest.raises(ValueError):
            dataset_key("humaneval")

    def test_templates_carry_the_required_answer_marker(self) -> None:
        assert "####" in user_prompt("gsm8k", "Q?")
        assert "\\boxed{}" in user_prompt("math", "Q?")

    def test_chat_messages_are_system_then_user(self) -> None:
        messages = chat_messages("gsm8k", "Q?")
        assert [message["role"] for message in messages] == ["system", "user"]
        assert "Q?" in messages[1]["content"]

    def test_empty_question_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            user_prompt("gsm8k", "   ")

    def test_version_tag_is_recorded(self) -> None:
        assert GenerationSettings().to_dict()["prompt_template_version"] == PROMPT_TEMPLATE_VERSION


class TestQuestionBank:
    def test_gsm8k_reference_uses_the_gold_marker(self) -> None:
        assert gsm8k_reference("blah\n#### 1,024") == "1024"
        with pytest.raises(ValueError):
            gsm8k_reference("no marker here")

    def test_math_reference_and_level(self) -> None:
        assert math_reference(str(MATH_ROW["solution"])) == "4"
        assert math_level("Level 3") == 3
        assert math_level("Level ?") is None

    def test_row_conversion_populates_strata_metadata(self) -> None:
        gsm = gsm8k_question(7, GSM8K_ROW)
        assert gsm.qid == "gsm8k-test-00007"
        assert gsm.reference_answer == "7"

        math = math_question("algebra", 2, MATH_ROW)
        assert math is not None
        assert math.qid == "math-algebra-test-0002"
        assert math.metadata["level"] == 3
        assert math.metadata["subject"] == "algebra"

    def test_unboxed_math_rows_are_dropped(self) -> None:
        assert math_question("algebra", 0, {**MATH_ROW, "solution": "The answer is 4."}) is None

    def test_questions_from_rows_requires_a_math_config(self) -> None:
        with pytest.raises(ValueError):
            questions_from_rows("math", [MATH_ROW])

    def test_question_roundtrips_through_dict(self) -> None:
        original = gsm8k_question(1, GSM8K_ROW)
        assert Question.from_dict(original.to_dict()) == original


class TestSelectQuestions:
    def _bank(self, count: int = 40) -> list:
        return [
            Question(
                qid=f"math-algebra-test-{index:04d}",
                dataset="math",
                question=f"Q{index}",
                reference_answer=str(index),
                metadata={"level": (index % 5) + 1},
            )
            for index in range(count)
        ]

    def test_selection_is_deterministic_and_seed_sensitive(self) -> None:
        bank = self._bank()
        first = [item.qid for item in select_questions(bank, limit=10, seed=0)]
        assert first == [item.qid for item in select_questions(bank, limit=10, seed=0)]
        assert first != [item.qid for item in select_questions(bank, limit=10, seed=1)]
        assert first == sorted(first)

    def test_stratification_keeps_every_level_populated(self) -> None:
        chosen = select_questions(self._bank(), limit=10, seed=0, stratify_by="level")
        levels = {item.metadata["level"] for item in chosen}
        assert len(chosen) == 10
        assert levels == {1, 2, 3, 4, 5}

    def test_limit_above_bank_size_returns_everything(self) -> None:
        bank = self._bank(5)
        assert len(select_questions(bank, limit=99)) == 5
        assert len(select_questions(bank)) == 5


class TestGenerationSettings:
    def test_micro_batches_divide_the_chain_budget(self) -> None:
        settings = GenerationSettings(chains_per_question=1024, micro_batch_chains=128)
        assert settings.micro_batches == 8

    def test_indivisible_micro_batch_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            GenerationSettings(chains_per_question=100, micro_batch_chains=32)

    def test_out_of_range_decoding_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            GenerationSettings(top_p=1.5)

    def test_seeds_are_stable_distinct_and_content_addressed(self) -> None:
        settings = GenerationSettings(chains_per_question=8, micro_batch_chains=2)
        seeds = request_seeds(settings, "gsm8k-test-00001")
        assert len(set(seeds)) == 4
        assert seeds == request_seeds(settings, "gsm8k-test-00001")
        assert seeds[0] != request_seed(0, "gsm8k-test-00002", 0)
        assert request_seed(0, "q", 0) != request_seed(1, "q", 0)
        # Pinned value: a change here silently invalidates every existing bank.
        assert request_seed(0, "gsm8k-test-00000", 0) == 1802739740


class TestChainRecords:
    def test_records_parse_answers_and_flag_failures(self) -> None:
        records = build_chain_records(
            "gsm8k",
            ["reasoning\n#### 12", "no answer at all in words"],
            logprob_sums=[-3.5, -9.0],
            token_counts=[40, 60],
            generation_seed=0,
            start_chain_id=128,
        )
        assert [record.chain_id for record in records] == [128, 129]
        assert records[0].answer == "12" and records[0].parsed
        assert records[1].answer is None and not records[1].parsed
        assert unparsed_rate(records) == 0.5

    def test_misaligned_side_channels_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_chain_records("gsm8k", ["a", "b"], logprob_sums=[-1.0])

    def test_math_records_use_the_boxed_extractor(self) -> None:
        records = build_chain_records("math", ["so $\\boxed{\\frac{1}{2}}$"])
        assert records[0].answer == "\\frac{1}{2}"


class TestShardingAndManifest:
    def test_shard_plan_is_ordered_and_covers_every_question(self) -> None:
        plan = shard_plan([f"q{index:02d}" for index in range(5)], 2)
        assert plan == [(0, ["q00", "q01"]), (1, ["q02", "q03"]), (2, ["q04"])]

    def test_duplicate_qids_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            shard_plan(["a", "a"], 2)

    def test_manifest_flags_a_high_unparsed_rate(self) -> None:
        manifest = bank_manifest(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            model_short="qwen2.5-0.5b",
            dataset="gsm8k",
            settings=GenerationSettings(),
            question_count=2,
            chain_count=2048,
            unparsed=0.07,
            shards=[1, 0],
        )
        assert manifest["shards"] == [0, 1]
        assert manifest["unparsed_rate_exceeds_5pct"] is True
        assert manifest["settings"]["prompt_template_version"] == PROMPT_TEMPLATE_VERSION


class TestHubLayout:
    def test_remote_paths_are_stable(self) -> None:
        assert (
            shard_remote_path("qwen2.5-0.5b", "gsm8k", 3)
            == "gen/qwen2.5-0.5b/gsm8k/shard-0003.jsonl.gz"
        )
        assert manifest_remote_path("gemma2-2b", "math") == "manifests/gemma2-2b/math.json"
        assert questions_remote_path("openai/gsm8k") == "questions/gsm8k.jsonl"
        with pytest.raises(ValueError):
            shard_remote_path("m", "gsm8k", -1)

    def test_shard_roundtrip_preserves_every_field(self) -> None:
        records = build_chain_records("gsm8k", ["#### 5"], logprob_sums=[-1.5], token_counts=[9])
        rows = shard_rows("gsm8k-test-00000", records)
        decoded = decode_shard(encode_shard(rows))
        assert decoded == [{field: rows[0].get(field) for field in CHAIN_FIELDS}]

    def test_encoding_is_byte_reproducible(self) -> None:
        rows = shard_rows("q", build_chain_records("gsm8k", ["#### 5"]))
        assert encode_shard(rows) == encode_shard(rows)

    def test_materialize_writes_the_blueprint_cache_layout(self, tmp_path: Path) -> None:
        rows = shard_rows("gsm8k-test-00000", build_chain_records("gsm8k", ["#### 5", "#### 6"]))
        rows += shard_rows("gsm8k-test-00001", build_chain_records("gsm8k", ["#### 7"]))
        written = materialize_shard(tmp_path, "qwen2.5-0.5b", "gsm8k", rows)

        expected = tmp_path / "gen" / "qwen2.5-0.5b" / "gsm8k" / "gsm8k-test-00000" / "chains.jsonl"
        assert expected in written
        first = [json.loads(line) for line in expected.read_text().splitlines()]
        assert [row["chain_id"] for row in first] == [0, 1]
        assert "qid" not in first[0]

        bank = read_cached_bank(tmp_path, "qwen2.5-0.5b", "gsm8k")
        assert sorted(bank) == ["gsm8k-test-00000", "gsm8k-test-00001"]
        assert len(bank["gsm8k-test-00000"]) == 2

    def test_shards_are_valid_gzip_jsonl(self) -> None:
        blob = encode_shard(shard_rows("q", build_chain_records("gsm8k", ["#### 5"])))
        line = gzip.decompress(blob).decode("utf-8").strip()
        assert json.loads(line)["qid"] == "q"

    def test_missing_bank_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_cached_bank(tmp_path, "nope", "gsm8k")
