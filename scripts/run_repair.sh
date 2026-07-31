#!/bin/zsh
# Fill in shards lost when a Modal app was stopped mid-run.
#
# Two failure modes this fixes, both hit on 2026-07-31:
#
#  1. `modal run` tears the app down if the local client disconnects ("Stopping
#     app - local client disconnected"). The Qwen2.5-1.5B MATH cell lost shards
#     2-3 that way. `--detach` keeps the app alive server-side.
#  2. `echo "... exit=$?"` reports the exit code of the $(date) substitution,
#     not of the command, so every cell logged exit=0 even when it failed. The
#     exit code is captured into a variable *before* any substitution runs.
#
# No --force: shards already on the volume are skipped, so this only generates
# what is missing.
set -uo pipefail

MODAL="${MODAL_BIN:-$HOME/.venvs/modal-cli/bin/modal}"
RUN="$MODAL run --detach modal_app/generate_chains.py::main"
LOG_DIR="outputs/campaign"
mkdir -p "$LOG_DIR"

repair() {
  local model="$1" dataset="$2" questions="$3" tokens="$4"
  local name exit_code
  name="$(echo "$model" | tr '/' '_')_${dataset}_repair"
  echo "[repair] $(date -u +%FT%TZ) start $name"
  ${=RUN} --model "$model" --dataset "$dataset" --questions "$questions" \
    --chains 1024 --micro-batch 128 --shard-questions 16 \
    --max-new-tokens "$tokens" \
    > "$LOG_DIR/$name.log" 2>&1
  exit_code=$?       # captured before any command substitution resets $?
                     # (`status` is read-only in zsh - do not use it)
  echo "[repair] $(date -u +%FT%TZ) done $name exit=$exit_code"
  return $exit_code
}

repair Qwen/Qwen2.5-1.5B-Instruct math 60 1024

echo "[repair] $(date -u +%FT%TZ) repair complete"
