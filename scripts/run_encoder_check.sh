#!/bin/zsh
# TB-7 encoder-stability check: embed one already-generated cell with a second
# encoder, using GPU budget left over after the six generation cells.
#
# Kept separate from run_campaign.sh on purpose: zsh reads a script by byte
# offset while it executes, so editing a running campaign script can corrupt
# the commands it has not reached yet. Never append to a script that is live.
set -uo pipefail

MODAL="${MODAL_BIN:-$HOME/.venvs/modal-cli/bin/modal}"
MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
DATASET="${DATASET:-gsm8k}"
ENCODER="${ENCODER:-mixedbread-ai/mxbai-embed-large-v1}"
LOG_DIR="outputs/campaign"
mkdir -p "$LOG_DIR"

echo "[encoders] $(date -u +%FT%TZ) start $MODEL/$DATASET with $ENCODER"
$MODAL run modal_app/generate_chains.py::embed_only \
  --model "$MODEL" --dataset "$DATASET" --encoder "$ENCODER" \
  > "$LOG_DIR/encoder_stability.log" 2>&1
exit_code=$?
echo "[encoders] $(date -u +%FT%TZ) done exit=$exit_code"
