# Reproducibility Environment

This file records implementation pins required by the technical blueprint.

- Vendi-Score repository: `https://github.com/vertaix/Vendi-Score`
- Vendi-Score commit: `ff1dfdbe6356b98a6087540f215b9a9db6db7c11`
- Vendi-Score distribution version at that commit: `0.0.3`
- Supported Python: `>=3.9`
- Primary environment definition: `pyproject.toml`
- Local secrets: `.env` (ignored by Git)

Create the environment with `make bootstrap`. For archival experiment runs, save
`python -m pip freeze` and accelerator details next to the run manifest in
`outputs/<run-id>/environment.txt`.

## Generation stack (Modal)

Chain generation runs on Modal GPUs via `modal_app/generate_chains.py`. The
container images pin:

- vLLM: `0.11.0` (Python 3.12 image, `VLLM_USE_V1=1`)
- huggingface_hub: `0.35.3` (with `hf_transfer`)
- datasets: `4.0.0`
- Vendi-Score: the commit pinned above, installed in both images so the worker
  imports the same `diversity_reasoning` package as the analysis environment
- Modal client: `>=1.0,<2` (validated on `1.2.6`)

GPU tiers: L40S for models below 5B parameters, H100 for 7B-class models
(`configs/base.yaml: modal.gpu_small / modal.gpu_large`). Decoding follows
Part B of the blueprint: temperature 1.0, top-p 0.95, 400 new tokens, each
model's own chat template, one fixed prompt template per dataset
(`diversity_reasoning.prompts`, version tag `v1`).

Sampling seeds are content-addressed: `request_seed = SHA256(generation_seed,
qid, micro_batch_index) mod (2^31 - 1)`. The same bank therefore reproduces on
any host, and `tests/test_generation_layer.py` pins one seed value so a silent
change to the scheme fails the suite.

Modal state:

- Volume `dvr-chain-bank` — shards and manifests, so an interrupted run resumes
- Volume `dvr-hf-cache` — Hugging Face model cache shared across runs
- Secret — `HF_TOKEN` read from the local `.env` at launch and forwarded to the
  containers; it unlocks the gated models and authenticates the dataset push

## Data sources

- GSM8K: `openai/gsm8k`, config `main`, split `test`
- MATH: `EleutherAI/hendrycks_math`, all seven subject configs, split `test`,
  levels 1-5 (replaces `lighteval/MATH`, removed from the Hub; see `TRIAGE.md`)

## Chain bank of record

Hugging Face dataset repo: `GOVINDFROM/Diversity-vs-Reasoning` (private).
Every chain produced by any run is pushed there; `configs/base.yaml: hf` holds
the repo id. Restore the local analysis cache with
`dvr bank pull --model <short> --dataset <gsm8k|math>`, which materializes the
blueprint's `cache/gen/{model}/{dataset}/{qid}/chains.jsonl` layout.
