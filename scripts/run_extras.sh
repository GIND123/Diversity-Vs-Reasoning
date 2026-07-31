#!/bin/zsh
# Post-campaign extras, in reviewer-value order (TRIAGE.md, 2026-07-31):
#
#  1. B1 generation-seed spot check: g in {1, 2} on a fixed 24-question subset
#     of the Qwen2.5-0.5B GSM8K bank (suffix banks -g1/-g2; the question file
#     pushes 24 rows here and is restored by the 192-question run below, which
#     executes last for exactly that reason).
#  2. GSM8K 0.5B extension 96 -> 192 questions: doubles power in the
#     minority/tail strata where the diversity effect actually lives.
#     --force because adding questions remixes the sorted shard layout.
#  3. TB-7 encoder stability: mxbai-embed-large-v1 on the Llama GSM8K bank.
set -uo pipefail

MODAL="${MODAL_BIN:-$HOME/.venvs/modal-cli/bin/modal}"
RUN="$MODAL run modal_app/generate_chains.py::main"
LOG_DIR="outputs/campaign"
mkdir -p "$LOG_DIR"

step() {
  local name="$1"; shift
  echo "[extras] $(date -u +%FT%TZ) start $name"
  "$@" > "$LOG_DIR/$name.log" 2>&1
  exit_code=$?
  echo "[extras] $(date -u +%FT%TZ) done $name exit=$exit_code"
}

step seedcheck_g1 ${=RUN} --model Qwen/Qwen2.5-0.5B-Instruct --dataset gsm8k \
  --questions 24 --chains 1024 --micro-batch 128 --shard-questions 16 \
  --seed 1 --question-seed 0 --bank-suffix "-g1" --force

step seedcheck_g2 ${=RUN} --model Qwen/Qwen2.5-0.5B-Instruct --dataset gsm8k \
  --questions 24 --chains 1024 --micro-batch 128 --shard-questions 16 \
  --seed 2 --question-seed 0 --bank-suffix "-g2" --force

step gsm8k_0.5b_192q ${=RUN} --model Qwen/Qwen2.5-0.5B-Instruct --dataset gsm8k \
  --questions 192 --chains 1024 --micro-batch 128 --shard-questions 16 --force

step encoder_mxbai "$MODAL" run modal_app/generate_chains.py::embed_only \
  --model meta-llama/Llama-3.2-3B-Instruct --dataset gsm8k \
  --encoder mixedbread-ai/mxbai-embed-large-v1

echo "[extras] $(date -u +%FT%TZ) extras complete"
