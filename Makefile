.PHONY: bootstrap test gate lint dry-run clean-stamp smoke generate bank-status

bootstrap:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest

gate:
	.venv/bin/python scripts/run_correctness_gate.py

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
	.venv/bin/python -m mypy src/diversity_reasoning

dry-run:
	.venv/bin/dvr list
	.venv/bin/dvr prepare R1 --dry-run

clean-stamp:
	.venv/bin/python scripts/clear_gate_stamp.py

# Cheap end-to-end check of the Modal + Hugging Face path (2 questions, 8 chains).
smoke:
	modal run modal_app/generate_chains.py --model Qwen/Qwen2.5-0.5B-Instruct \
		--dataset gsm8k --questions 2 --chains 8 --micro-batch 4 --shard-questions 2

# Full bank for one (model, dataset). Override MODEL/DATASET/QUESTIONS on the CLI.
MODEL ?= Qwen/Qwen2.5-1.5B-Instruct
DATASET ?= gsm8k
QUESTIONS ?= 500
generate:
	modal run modal_app/generate_chains.py --model $(MODEL) --dataset $(DATASET) \
		--questions $(QUESTIONS) --chains 1024

bank-status:
	.venv/bin/dvr bank status
