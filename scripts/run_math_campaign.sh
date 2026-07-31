#!/bin/zsh
# MATH generation with an adequate token budget.
#
# The blueprint's 400-token cap truncates MATH solutions: the first 0.5B MATH
# bank came back 67.4% unparsed, 98.4% of those sitting exactly at the cap.
# 1024 tokens fits the distribution (parsed median at 400 was 303). --force
# overwrites the truncated shards already on the volume and the Hub.
# See TRIAGE.md, 2026-07-31.
set -uo pipefail

MODAL="${MODAL_BIN:-$HOME/.venvs/modal-cli/bin/modal}"
RUN="$MODAL run modal_app/generate_chains.py::main"
LOG_DIR="outputs/campaign"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
mkdir -p "$LOG_DIR"

cell() {
  local model="$1" questions="$2"
  local name
  name="$(echo "$model" | tr '/' '_')_math"
  echo "[math] $(date -u +%FT%TZ) start $name tokens=$MAX_NEW_TOKENS"
  ${=RUN} --model "$model" --dataset math --questions "$questions" \
    --chains 1024 --micro-batch 128 --shard-questions 16 \
    --max-new-tokens "$MAX_NEW_TOKENS" --force \
    > "$LOG_DIR/$name.log" 2>&1
  echo "[math] $(date -u +%FT%TZ) done $name exit=$?"
}

cell Qwen/Qwen2.5-0.5B-Instruct 60
cell Qwen/Qwen2.5-1.5B-Instruct 60
cell meta-llama/Llama-3.2-3B-Instruct 60

echo "[math] $(date -u +%FT%TZ) MATH campaign complete"
