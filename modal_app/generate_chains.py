"""Modal GPU chain generation + embedding for the diversity-vs-coverage study.

Runs vLLM on a single A100-40GB (the project's approved compute profile),
generates the chain bank per question, embeds every chain with the primary
encoder, writes everything to a Modal volume, and pushes it to the Hugging Face
dataset repo. The Hugging Face token is read from the project's local ``.env``
and forwarded as a Modal secret; it unlocks the gated models and the dataset
push.

Usage::

    # one (model, dataset) cell: generate, then embed
    modal run modal_app/generate_chains.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --dataset gsm8k --questions 128 --chains 1024

    # embedding only (bank already generated)
    modal run modal_app/generate_chains.py::embed_only \
        --model Qwen/Qwen2.5-0.5B-Instruct --dataset gsm8k

All parsing, seeding, and manifest logic lives in ``diversity_reasoning`` so a
Modal-produced bank is bit-comparable with a locally produced one.

``from __future__ import annotations`` is deliberately absent: Modal resolves
``modal.parameter()`` annotations at class-definition time and rejects the
stringified forms that the future import would produce.
"""

import io
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import modal

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "diversity-vs-reasoning-generation"
DEFAULT_REPO = "GOVINDFROM/Diversity-vs-Reasoning"

VLLM_VERSION = "0.11.0"
HF_HUB_VERSION = "0.35.3"
DATASETS_VERSION = "4.0.0"
SENTENCE_TRANSFORMERS_VERSION = "5.1.2"
# Same commit as ENVIRONMENT.md; imported through diversity_reasoning.metrics.
VENDI_SCORE_PIN = (
    "vendi-score @ git+https://github.com/vertaix/Vendi-Score.git"
    "@ff1dfdbe6356b98a6087540f215b9a9db6db7c11"
)

# The approved compute profile: one A100-40GB, everything sequential through it.
GPU_KIND = "A100-40GB"
DEFAULT_ENCODER = "BAAI/bge-large-en-v1.5"

BANK_ROOT = "/bank"
HF_CACHE = "/hf-cache"
SOURCE_ROOT = "/opt/dvr/src"

app = modal.App(APP_NAME)


def _hf_token() -> str:
    """Read HF_TOKEN from the project's .env without needing python-dotenv.

    ``modal.Secret.from_dotenv`` would pull in an extra local dependency, and the
    launcher is expected to run from the standalone Modal CLI environment.
    """
    import os

    env_file = ROOT / ".env"
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    return value
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN missing: copy .env.example to .env and add your token.")
    return token


hf_secret = modal.Secret.from_dict({"HF_TOKEN": _hf_token()})
bank_volume = modal.Volume.from_name("dvr-chain-bank", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("dvr-hf-cache", create_if_missing=True)

_shared_env = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_HOME": HF_CACHE,
    "PYTHONPATH": SOURCE_ROOT,
    "TOKENIZERS_PARALLELISM": "false",
}

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"huggingface_hub[hf_transfer]=={HF_HUB_VERSION}",
        f"datasets=={DATASETS_VERSION}",
        "numpy>=1.26,<3",
        "scipy>=1.10,<2",
        "scikit-learn>=1.1,<2",
        "sympy>=1.12,<2",
        "PyYAML>=6.0,<7",
        VENDI_SCORE_PIN,
    )
    .env(_shared_env)
    .add_local_dir(ROOT / "src", SOURCE_ROOT)
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        f"huggingface_hub[hf_transfer]=={HF_HUB_VERSION}",
        f"sentence-transformers=={SENTENCE_TRANSFORMERS_VERSION}",
        "sympy>=1.12,<2",
        "scikit-learn>=1.1,<2",
        VENDI_SCORE_PIN,
    )
    .env({**_shared_env, "VLLM_USE_V1": "1"})
    .add_local_dir(ROOT / "src", SOURCE_ROOT)
)


def model_short_name(model_id: str) -> str:
    """``Qwen/Qwen2.5-1.5B-Instruct`` -> ``qwen2.5-1.5b``."""
    tail = model_id.split("/")[-1].lower()
    tail = re.sub(r"-(instruct|it|chat)$", "", tail)
    return tail


def encoder_short_name(encoder_id: str) -> str:
    """``BAAI/bge-large-en-v1.5`` -> ``bge-large-en-v1.5``."""
    return encoder_id.split("/")[-1].lower()


# --------------------------------------------------------------------------
# Question preparation (CPU)
# --------------------------------------------------------------------------


@app.function(
    image=cpu_image,
    secrets=[hf_secret],
    volumes={BANK_ROOT: bank_volume},
    timeout=60 * 60,
)
def prepare_questions(
    dataset: str,
    questions: Optional[int] = None,
    seed: int = 0,
    repo_id: str = DEFAULT_REPO,
    push: bool = True,
) -> List[Dict[str, Any]]:
    """Load, subsample, cache, and publish the question bank for one dataset."""
    import os

    from diversity_reasoning.datasets import load_questions
    from diversity_reasoning.hf_bank import ensure_repo, push_questions
    from diversity_reasoning.prompts import dataset_key

    token = os.environ["HF_TOKEN"]
    key = dataset_key(dataset)
    bank = load_questions(key, limit=questions, seed=seed, token=token)
    rows = [item.to_dict() for item in bank]

    # Merge, never replace. The published question file is shared by every bank,
    # so a run with a smaller --questions limit (a seed-variance spot check, say)
    # must not shrink it and orphan the questions of larger banks. Union by qid.
    published: Dict[str, Dict[str, Any]] = {}
    destination = Path(BANK_ROOT) / "questions" / f"{key}.jsonl"
    if destination.exists():
        for line in destination.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing = json.loads(line)
                published[str(existing["qid"])] = existing
    dropped = len(published)
    for row in rows:
        published[str(row["qid"])] = row
    merged = [published[qid] for qid in sorted(published)]

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in merged), encoding="utf-8"
    )
    bank_volume.commit()

    if push:
        ensure_repo(repo_id, token=token)
        push_questions(merged, key, repo_id=repo_id, token=token)
    print(
        f"[questions] {key}: {len(rows)} requested, {len(merged)} published "
        f"(was {dropped}; merged, never truncated)"
    )
    return rows


# --------------------------------------------------------------------------
# Generation (GPU)
# --------------------------------------------------------------------------


def _load_engine(
    model_id: str,
    max_model_len: int,
    gpu_memory_utilization_pct: int,
) -> Tuple[Any, Any]:
    from vllm import LLM

    engine = LLM(
        model=model_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization_pct / 100.0,
        dtype="bfloat16",
        seed=0,
        disable_log_stats=True,
    )
    return engine, engine.get_tokenizer()


def _render_prompt(tokenizer: Any, dataset: str, question: str) -> str:
    """Apply the model's own chat template, folding system turns when unsupported."""
    from diversity_reasoning.prompts import chat_messages

    messages = chat_messages(dataset, question)
    try:
        return str(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    except Exception:  # noqa: BLE001 - Gemma templates reject the system role
        merged = [
            {
                "role": "user",
                "content": f"{messages[0]['content']}\n\n{messages[1]['content']}",
            }
        ]
        return str(
            tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
        )


def _generate_shard(
    engine: Any,
    tokenizer: Any,
    *,
    model_id: str,
    dataset: str,
    shard_index: int,
    bank_suffix: str = "",
    questions: Sequence[Dict[str, Any]],
    settings_payload: Dict[str, Any],
    repo_id: str,
    push: bool,
    force: bool,
) -> Dict[str, Any]:
    """Generate one shard, write it to the volume, and push it to the Hub."""
    import os

    from vllm import SamplingParams

    from diversity_reasoning.generation import (
        GenerationSettings,
        build_chain_records,
        request_seed,
        unparsed_rate,
    )
    from diversity_reasoning.hf_bank import encode_shard, shard_remote_path, upload_bytes
    from diversity_reasoning.prompts import dataset_key
    from diversity_reasoning.schemas import ChainRecord

    key = dataset_key(dataset)
    short = model_short_name(model_id) + bank_suffix
    settings = GenerationSettings(**settings_payload)
    local_path = Path(BANK_ROOT) / shard_remote_path(short, key, shard_index)

    if local_path.exists() and not force:
        print(f"[skip] shard {shard_index:04d} already present at {local_path}")
        return {"shard": shard_index, "status": "skipped", "path": str(local_path)}

    started = time.time()
    prompts = [_render_prompt(tokenizer, key, item["question"]) for item in questions]
    per_question: Dict[str, List[ChainRecord]] = {item["qid"]: [] for item in questions}

    for micro_index in range(settings.micro_batches):
        params = [
            SamplingParams(
                n=settings.micro_batch_chains,
                temperature=settings.temperature,
                top_p=settings.top_p,
                max_tokens=settings.max_new_tokens,
                logprobs=0,
                seed=request_seed(settings.generation_seed, item["qid"], micro_index),
            )
            for item in questions
        ]
        outputs = engine.generate(prompts, params)
        for item, output in zip(questions, outputs):
            completions = list(output.outputs)
            per_question[item["qid"]].extend(
                build_chain_records(
                    key,
                    [completion.text for completion in completions],
                    logprob_sums=[
                        None
                        if completion.cumulative_logprob is None
                        else float(completion.cumulative_logprob)
                        for completion in completions
                    ],
                    token_counts=[len(completion.token_ids) for completion in completions],
                    generation_seed=settings.generation_seed,
                    start_chain_id=micro_index * settings.micro_batch_chains,
                )
            )

    rows: List[Dict[str, Any]] = []
    all_records: List[ChainRecord] = []
    for item in questions:
        records = sorted(per_question[item["qid"]], key=lambda record: record.chain_id)
        all_records.extend(records)
        rows.extend({"qid": item["qid"], **record.to_dict()} for record in records)

    blob = encode_shard(rows)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(blob)
    bank_volume.commit()

    remote = None
    if push:
        remote = upload_bytes(
            blob,
            shard_remote_path(short, key, shard_index),
            repo_id=repo_id,
            token=os.environ["HF_TOKEN"],
            commit_message=f"Add {short}/{key} shard {shard_index:04d}",
        )

    generated_tokens = sum(record.token_count or 0 for record in all_records)
    elapsed = time.time() - started
    summary = {
        "shard": shard_index,
        "status": "generated",
        "questions": len(questions),
        "chains": len(all_records),
        "unparsed_rate": unparsed_rate(all_records),
        "seconds": round(elapsed, 1),
        "generated_tokens": generated_tokens,
        "tokens_per_second": round(generated_tokens / elapsed, 1) if elapsed > 0 else 0.0,
        "mean_tokens_per_chain": round(generated_tokens / max(1, len(all_records)), 1),
        "bytes": len(blob),
        "remote": remote,
    }
    print(f"[shard {shard_index:04d}] {json.dumps(summary)}")
    return summary


@app.cls(
    gpu=GPU_KIND,
    image=gpu_image,
    secrets=[hf_secret],
    volumes={BANK_ROOT: bank_volume, HF_CACHE: hf_cache_volume},
    timeout=6 * 60 * 60,
    scaledown_window=240,
    max_containers=1,
)
class ChainGenerator:
    """Generation worker; one warm vLLM engine on the single approved A100."""

    model_id: str = modal.parameter()
    max_model_len: int = modal.parameter(default=2048)
    gpu_memory_utilization_pct: int = modal.parameter(default=90)

    @modal.enter()
    def start(self) -> None:
        self.engine, self.tokenizer = _load_engine(
            self.model_id, self.max_model_len, self.gpu_memory_utilization_pct
        )

    @modal.method()
    def generate_shard(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return _generate_shard(self.engine, self.tokenizer, model_id=self.model_id, **task)


# --------------------------------------------------------------------------
# Embeddings (GPU)
# --------------------------------------------------------------------------


@app.function(
    gpu=GPU_KIND,
    image=gpu_image,
    secrets=[hf_secret],
    volumes={BANK_ROOT: bank_volume, HF_CACHE: hf_cache_volume},
    timeout=2 * 60 * 60,
    max_containers=1,
)
def embed_bank(
    model_short: str,
    dataset: str,
    encoder_id: str = DEFAULT_ENCODER,
    batch_size: int = 256,
    repo_id: str = DEFAULT_REPO,
    push: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """Embed every chain of a bank, shard by shard, and push npz files to the Hub.

    Each npz holds one array per qid (float16, [n_chains, d]) in chain order,
    matching the corresponding generation shard exactly.
    """
    import os

    import numpy as np
    from sentence_transformers import SentenceTransformer

    from diversity_reasoning.hf_bank import (
        decode_shard,
        embedding_shard_remote_path,
        upload_bytes,
    )
    from diversity_reasoning.prompts import dataset_key

    key = dataset_key(dataset)
    encoder_short = encoder_short_name(encoder_id)
    bank_dir = Path(BANK_ROOT) / "gen" / model_short / key
    shard_files = sorted(bank_dir.glob("shard-*.jsonl.gz"))
    if not shard_files:
        raise FileNotFoundError(f"No generation shards under {bank_dir}")

    encoder = SentenceTransformer(
        encoder_id, device="cuda", model_kwargs={"torch_dtype": "float16"}
    )
    started = time.time()
    done, skipped, chains_total = 0, 0, 0
    for shard_file in shard_files:
        shard_index = int(shard_file.stem.split("-")[1].split(".")[0])
        remote = embedding_shard_remote_path(encoder_short, model_short, key, shard_index)
        local = Path(BANK_ROOT) / remote
        if local.exists() and not force:
            skipped += 1
            continue
        rows = decode_shard(shard_file.read_bytes())
        texts = [str(row["text"]) for row in rows]
        vectors = encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float16)
        arrays: Dict[str, List[Any]] = {}
        for row, vector in zip(rows, vectors):
            arrays.setdefault(str(row["qid"]), []).append(vector)
        payload = {qid: np.stack(vecs) for qid, vecs in arrays.items()}
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **payload)
        blob = buffer.getvalue()
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(blob)
        bank_volume.commit()
        if push:
            upload_bytes(
                blob,
                remote,
                repo_id=repo_id,
                token=os.environ["HF_TOKEN"],
                commit_message=f"Add {encoder_short} emb {model_short}/{key} {shard_index:04d}",
            )
        done += 1
        chains_total += len(texts)

    summary = {
        "encoder": encoder_id,
        "model": model_short,
        "dataset": key,
        "shards_embedded": done,
        "shards_skipped": skipped,
        "chains": chains_total,
        "seconds": round(time.time() - started, 1),
    }
    print(f"[embed] {json.dumps(summary)}")
    return summary


# --------------------------------------------------------------------------
# Finalization (CPU)
# --------------------------------------------------------------------------


@app.function(
    image=cpu_image,
    secrets=[hf_secret],
    volumes={BANK_ROOT: bank_volume},
    timeout=30 * 60,
)
def finalize_bank(
    model_id: str,
    dataset: str,
    settings_payload: Dict[str, Any],
    summaries: List[Dict[str, Any]],
    question_count: int,
    repo_id: str = DEFAULT_REPO,
    push: bool = True,
    bank_suffix: str = "",
) -> Dict[str, Any]:
    """Write and publish the bank manifest once every shard has landed."""
    import os

    from diversity_reasoning.generation import GenerationSettings, bank_manifest
    from diversity_reasoning.hf_bank import manifest_remote_path, upload_bytes
    from diversity_reasoning.prompts import dataset_key

    key = dataset_key(dataset)
    short = model_short_name(model_id) + bank_suffix
    generated = [row for row in summaries if row.get("status") == "generated"]
    chains = sum(int(row.get("chains", 0)) for row in generated)
    weighted = sum(
        float(row.get("unparsed_rate", 0.0)) * int(row.get("chains", 0)) for row in generated
    )
    manifest = bank_manifest(
        model_id=model_id,
        model_short=short,
        dataset=key,
        settings=GenerationSettings(**settings_payload),
        question_count=question_count,
        chain_count=chains,
        unparsed=(weighted / chains) if chains else 0.0,
        shards=[int(row["shard"]) for row in summaries],
        extra={
            "vllm_version": VLLM_VERSION,
            "runner": "modal",
            "gpu": GPU_KIND,
            "shard_summaries": summaries,
        },
    )

    destination = Path(BANK_ROOT) / "manifests" / short / f"{key}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bank_volume.commit()

    if push:
        upload_bytes(
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            manifest_remote_path(short, key),
            repo_id=repo_id,
            token=os.environ["HF_TOKEN"],
            commit_message=f"Update {short}/{key} manifest",
        )
    if manifest["unparsed_rate_exceeds_5pct"]:
        print(
            f"[warn] unparsed rate {manifest['unparsed_rate']:.3f} exceeds 5%: "
            "inspect 20 random failures before using this bank (blueprint B2)."
        )
    print(f"[manifest] {json.dumps({k: v for k, v in manifest.items() if k != 'shard_summaries'})}")
    return manifest


# --------------------------------------------------------------------------
# Entrypoints
# --------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dataset: str = "gsm8k",
    questions: int = 128,
    chains: int = 1024,
    micro_batch: int = 128,
    # `seed` is the generation seed g; `question_seed` picks the question
    # subsample. The B1 variance spot check varies g on a fixed question set,
    # so the two must be independent.
    seed: int = 0,
    question_seed: int = 0,
    shard_questions: int = 16,
    # Blueprint default is 400. MATH needs more: at 400 tokens 98% of unparsed
    # chains were truncations, leaving only short solutions in the pool.
    max_new_tokens: int = 400,
    max_model_len: int = 2048,
    gpu_memory_utilization_pct: int = 90,
    repo: str = DEFAULT_REPO,
    push: bool = True,
    force: bool = False,
    embed: bool = True,
    encoder: str = DEFAULT_ENCODER,
    # Distinguishes variance-check banks (e.g. "-g1") from the main bank in
    # every storage path; the main bank uses the empty suffix.
    bank_suffix: str = "",
) -> None:
    """Generate one (model, dataset) chain bank, publish it, then embed it."""
    settings_payload = {
        "chains_per_question": chains,
        "micro_batch_chains": min(micro_batch, chains),
        "generation_seed": seed,
        "max_new_tokens": max_new_tokens,
    }

    question_rows = prepare_questions.remote(dataset, questions, question_seed, repo, push)
    by_qid = {row["qid"]: row for row in question_rows}
    ordered = sorted(by_qid)
    plan = [
        (index, ordered[start : start + shard_questions])
        for index, start in enumerate(range(0, len(ordered), shard_questions))
    ]

    worker = ChainGenerator(
        model_id=model,
        max_model_len=max_model_len,
        gpu_memory_utilization_pct=gpu_memory_utilization_pct,
    )
    tasks = [
        {
            "dataset": dataset,
            "shard_index": shard_index,
            "questions": [by_qid[qid] for qid in qids],
            "settings_payload": settings_payload,
            "repo_id": repo,
            "push": push,
            "force": force,
            "bank_suffix": bank_suffix,
        }
        for shard_index, qids in plan
    ]
    print(
        f"[launch] {model} x {dataset}: {len(ordered)} questions, {chains} chains each, "
        f"{len(tasks)} shards on one {GPU_KIND}"
    )

    summaries = list(worker.generate_shard.map(tasks))
    finalize_bank.remote(
        model, dataset, settings_payload, summaries, len(ordered), repo, push, bank_suffix
    )
    if embed:
        embed_bank.remote(
            model_short_name(model) + bank_suffix, dataset, encoder, 256, repo, push, force
        )


@app.local_entrypoint()
def embed_only(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dataset: str = "gsm8k",
    encoder: str = DEFAULT_ENCODER,
    repo: str = DEFAULT_REPO,
    push: bool = True,
    force: bool = False,
) -> None:
    """Embed an existing bank without touching generation."""
    embed_bank.remote(model_short_name(model), dataset, encoder, 256, repo, push, force)
