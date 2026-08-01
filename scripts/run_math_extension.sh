#!/bin/zsh
# Scale MATH questions on the two weak models.
#
# Rationale from the completed run: the achievable gain tracks the *winnable*
# share (correct answer present but not modal), and MATH on the weak models has
# by far the highest winnable share (0.45 vs 0.06 for Llama/GSM8K). Doubling
# GSM8K questions on Qwen2.5-0.5B already turned a non-significant effect into a
# Holm-significant one and grew the tail stratum from 12 to 27. Tripling MATH
# questions on the two weak models targets the same power where it is scarcest.
#
# Llama-3.2-3B/MATH is deliberately left at 60: its winnable share is low and
# every objective is null there, so extra questions would buy nothing.
#
# --force because growing the question set redistributes the sorted shard
# layout; measured rate ~1.4-1.5 min/question at the 1024-token MATH budget.
set -uo pipefail

MODAL="${MODAL_BIN:-$HOME/.venvs/modal-cli/bin/modal}"
RUN="$MODAL run --detach modal_app/generate_chains.py::main"
LOG_DIR="outputs/campaign"
mkdir -p "$LOG_DIR"

cell() {
  local model="$1" questions="$2"
  local name exit_code
  name="$(echo "$model" | tr '/' '_')_math_${questions}q"
  echo "[mathext] $(date -u +%FT%TZ) start $name"
  ${=RUN} --model "$model" --dataset math --questions "$questions" \
    --chains 1024 --micro-batch 128 --shard-questions 16 \
    --max-new-tokens 1024 --force \
    > "$LOG_DIR/$name.log" 2>&1
  exit_code=$?      # capture before any substitution; `status` is read-only in zsh
  echo "[mathext] $(date -u +%FT%TZ) done $name exit=$exit_code"
}

cell Qwen/Qwen2.5-0.5B-Instruct 180
cell Qwen/Qwen2.5-1.5B-Instruct 120

echo "[mathext] $(date -u +%FT%TZ) MATH extension complete"
