#!/bin/zsh
# Full generation campaign on the approved single-A100 profile.
# Scale decisions and their justification are logged in TRIAGE.md (2026-07-30).
# Each cell generates 1024 chains/question, pushes shards to the HF dataset,
# then embeds every chain with bge-large. Shards already on the volume are
# skipped, so re-running this script resumes instead of regenerating.
set -uo pipefail

# Absolute path: `modal` is only an alias in interactive shells.
MODAL="${MODAL_BIN:-$HOME/.venvs/modal-cli/bin/modal}"
RUN="$MODAL run modal_app/generate_chains.py::main"
LOG_DIR="outputs/campaign"
mkdir -p "$LOG_DIR"

cell() {
  local model="$1" dataset="$2" questions="$3"; shift 3
  local name
  name="$(echo "$model" | tr '/' '_')_${dataset}"
  echo "[campaign] $(date -u +%FT%TZ) start $name questions=$questions $*"
  ${=RUN} --model "$model" --dataset "$dataset" --questions "$questions" \
    --chains 1024 --micro-batch 128 --shard-questions 16 "$@" \
    > "$LOG_DIR/$name.log" 2>&1
  exit_code=$?  # capture first: $(date) below would reset $?, and
                # `status` is read-only in zsh
  echo "[campaign] $(date -u +%FT%TZ) done $name exit=$exit_code"
}

# GSM8K first: cheaper per question, unblocks the analysis pipeline earliest.
# --force on the 0.5b cell overwrites the 8-question calibration shard layout.
cell Qwen/Qwen2.5-0.5B-Instruct gsm8k 96 --force
cell Qwen/Qwen2.5-1.5B-Instruct gsm8k 96
cell meta-llama/Llama-3.2-3B-Instruct gsm8k 96

cell Qwen/Qwen2.5-0.5B-Instruct math 60
cell Qwen/Qwen2.5-1.5B-Instruct math 60
cell meta-llama/Llama-3.2-3B-Instruct math 60

echo "[campaign] $(date -u +%FT%TZ) campaign complete"
