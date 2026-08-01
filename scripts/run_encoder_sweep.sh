#!/bin/zsh
# Multi-encoder sweep over already-generated banks.
#
# The Vendi score is a functional of a similarity kernel, so every claim we make
# about it is conditional on the encoder that induces that kernel. One encoder
# cannot distinguish "a property of reasoning-chain pools" from "a property of
# bge-large". This sweep embeds the same chains with encoders from four
# different families so the measurement findings — the near-rank-1 raw kernel,
# the effect of question-centring, and the rank stability of each order q — can
# be shown to hold across encoders rather than within one.
#
# Embedding is cheap (~5 min per cell); no chains are regenerated.
set -uo pipefail

MODAL="${MODAL_BIN:-$HOME/.venvs/modal-cli/bin/modal}"
RUN="$MODAL run --detach modal_app/generate_chains.py::embed_only"
LOG_DIR="outputs/campaign"
mkdir -p "$LOG_DIR"

# Four encoder families: BGE (already run), MXBAI (already run), E5, GTE, MPNet.
ENCODERS=(
  "intfloat/e5-large-v2"
  "thenlper/gte-large"
  "sentence-transformers/all-mpnet-base-v2"
)
CELLS=(
  "Qwen/Qwen2.5-0.5B-Instruct gsm8k"
  "meta-llama/Llama-3.2-3B-Instruct gsm8k"
  "Qwen/Qwen2.5-0.5B-Instruct math"
)

for encoder in "${ENCODERS[@]}"; do
  for cell in "${CELLS[@]}"; do
    model="${cell%% *}"; dataset="${cell##* }"
    name="enc_$(echo "${encoder}_${model}_${dataset}" | tr '/.' '__')"
    exit_code=0
    echo "[encsweep] $(date -u +%FT%TZ) start $encoder on $model/$dataset"
    ${=RUN} --model "$model" --dataset "$dataset" --encoder "$encoder" \
      > "$LOG_DIR/$name.log" 2>&1
    exit_code=$?   # capture first; `status` is read-only in zsh
    echo "[encsweep] $(date -u +%FT%TZ) done $encoder on $model/$dataset exit=$exit_code"
  done
done

echo "[encsweep] $(date -u +%FT%TZ) encoder sweep complete"
