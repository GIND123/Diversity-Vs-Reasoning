"""Hugging Face dataset repository as the durable home of every chain bank.

Layout inside the dataset repo::

    questions/{dataset}.jsonl                     one row per evaluation question
    gen/{model_short}/{dataset}/shard-NNNN.jsonl.gz   chains, sharded by question
    manifests/{model_short}/{dataset}.json        decoding settings and provenance
    README.md                                     generated dataset card

The Hub stores question-sharded files because a repo cannot carry one small file
per question. ``materialize_shard`` expands a shard back into the per-question
``cache/gen/{model}/{dataset}/{qid}/chains.jsonl`` layout the blueprint's
downstream stages read, so nothing after generation changes.
"""

from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .artifacts import generation_path, safe_component
from .io import read_jsonl, write_json_atomic, write_jsonl_atomic
from .prompts import dataset_key
from .schemas import ChainRecord

DEFAULT_DATASET_REPO = "GOVINDFROM/Diversity-vs-Reasoning"
TOKEN_ENV = "HF_TOKEN"
CHAIN_FIELDS = (
    "qid",
    "chain_id",
    "text",
    "answer",
    "parsed",
    "logprob_sum",
    "token_count",
    "generation_seed",
)


def resolve_token(token: Optional[str] = None, env_var: str = TOKEN_ENV) -> str:
    """Return the Hub token, loading ``.env`` first and never logging its value."""
    if token:
        return token
    from .env import load_local_env

    load_local_env()
    value = os.environ.get(env_var, "")
    if not value:
        raise RuntimeError(
            f"{env_var} is not set. Copy .env.example to .env and add a Hugging Face "
            "token with write access to the dataset repo."
        )
    return value


def questions_remote_path(dataset: str) -> str:
    return f"questions/{dataset_key(dataset)}.jsonl"


def shard_remote_path(model_short: str, dataset: str, shard_index: int) -> str:
    if shard_index < 0:
        raise ValueError("shard_index must be nonnegative")
    return (
        f"gen/{safe_component(model_short)}/{dataset_key(dataset)}/shard-{shard_index:04d}.jsonl.gz"
    )


def manifest_remote_path(model_short: str, dataset: str) -> str:
    return f"manifests/{safe_component(model_short)}/{dataset_key(dataset)}.json"


def embedding_shard_remote_path(
    encoder_short: str,
    model_short: str,
    dataset: str,
    shard_index: int,
) -> str:
    if shard_index < 0:
        raise ValueError("shard_index must be nonnegative")
    return (
        f"emb/{safe_component(encoder_short)}/{safe_component(model_short)}/"
        f"{dataset_key(dataset)}/shard-{shard_index:04d}.npz"
    )


def encode_shard(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """Serialize shard rows as gzipped JSONL with a stable byte layout."""
    payload = "".join(
        json.dumps({field: row.get(field) for field in CHAIN_FIELDS}, sort_keys=True) + "\n"
        for row in rows
    )
    return gzip.compress(payload.encode("utf-8"), mtime=0)


def decode_shard(blob: bytes) -> List[Dict[str, Any]]:
    text = gzip.decompress(blob).decode("utf-8")
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or "qid" not in value:
            raise ValueError(f"Shard line {line_number} is not a chain row")
        rows.append(value)
    return rows


def shard_rows(qid: str, records: Sequence[ChainRecord]) -> List[Dict[str, Any]]:
    """Attach the question id to chain records for shard storage."""
    return [{"qid": qid, **record.to_dict()} for record in records]


def materialize_shard(
    cache: Path,
    model_short: str,
    dataset: str,
    rows: Sequence[Mapping[str, Any]],
) -> List[Path]:
    """Expand shard rows into per-question ``chains.jsonl`` cache files."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        payload = {field: row.get(field) for field in CHAIN_FIELDS if field != "qid"}
        grouped[str(row["qid"])].append(payload)
    written: List[Path] = []
    for qid, chains in grouped.items():
        chains.sort(key=lambda item: int(item["chain_id"]))
        path = generation_path(cache, model_short, dataset_key(dataset), qid)
        write_jsonl_atomic(path, chains)
        written.append(path)
    return written


def read_cached_bank(
    cache: Path,
    model_short: str,
    dataset: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Read a locally cached bank back into ``{qid: chain rows}``."""
    root = cache / "gen" / safe_component(model_short) / dataset_key(dataset)
    if not root.exists():
        raise FileNotFoundError(f"No cached bank at {root}")
    bank: Dict[str, List[Dict[str, Any]]] = {}
    for chains_file in sorted(root.glob("*/chains.jsonl")):
        bank[chains_file.parent.name] = list(read_jsonl(chains_file))
    if not bank:
        raise FileNotFoundError(f"No chains.jsonl files under {root}")
    return bank


def dataset_card(repo_id: str) -> str:
    """Generate the Hub dataset card. The project README is never touched."""
    return f"""---
license: mit
task_categories:
  - text-generation
tags:
  - reasoning
  - diversity
  - vendi-score
  - math
configs:
  - config_name: default
    data_files:
      - split: train
        path: gen/*/*/shard-*.jsonl.gz
---

# {repo_id}

Reasoning-chain banks for the study *Diversity vs. Coverage: Which Is Best for LLM
Reasoning?* Each bank is 1024 sampled chains per question (temperature 1.0,
top-p 0.95, 400 new tokens), generated with vLLM on Modal GPUs.

## Layout

| Path | Contents |
|---|---|
| `questions/{{dataset}}.jsonl` | questions, gold answers, strata metadata |
| `gen/{{model}}/{{dataset}}/shard-NNNN.jsonl.gz` | the chain rows (fields below) |
| `manifests/{{model}}/{{dataset}}.json` | decoding settings and provenance |

Chain row fields: `qid`, `chain_id`, `text`, `answer`, `parsed`, `logprob_sum`,
`token_count`, `generation_seed`. Manifest fields: decoding settings, prompt
template version, unparsed rate, shard list, vLLM version, GPU.

Datasets: GSM8K test split (`openai/gsm8k`) and Hendrycks MATH levels 1-5
(`EleutherAI/hendrycks_math`, test split).

Chains marked `parsed: false` have no extractable final answer. They are kept
for transparency and excluded symmetrically from every selector and aggregation.

Restore the analysis cache layout with `dvr bank pull --model <short> --dataset <key>`.
"""


def _api(token: Optional[str] = None) -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=resolve_token(token))


def ensure_repo(
    repo_id: str = DEFAULT_DATASET_REPO,
    *,
    private: bool = True,
    token: Optional[str] = None,
    write_card: bool = True,
) -> str:
    """Create the dataset repo if absent and keep its card current."""
    api = _api(token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    if write_card:
        api.upload_file(
            path_or_fileobj=dataset_card(repo_id).encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update dataset card",
        )
    return repo_id


def upload_bytes(
    blob: bytes,
    path_in_repo: str,
    *,
    repo_id: str = DEFAULT_DATASET_REPO,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> str:
    api = _api(token)
    api.upload_file(
        path_or_fileobj=blob,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message or f"Add {path_in_repo}",
    )
    return path_in_repo


def list_bank_files(
    *,
    repo_id: str = DEFAULT_DATASET_REPO,
    prefix: str = "gen/",
    token: Optional[str] = None,
) -> List[str]:
    api = _api(token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    return sorted(name for name in files if name.startswith(prefix))


def push_questions(
    questions: Sequence[Mapping[str, Any]],
    dataset: str,
    *,
    repo_id: str = DEFAULT_DATASET_REPO,
    token: Optional[str] = None,
) -> str:
    payload = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in questions)
    return upload_bytes(
        payload.encode("utf-8"),
        questions_remote_path(dataset),
        repo_id=repo_id,
        token=token,
        commit_message=f"Add {dataset_key(dataset)} question bank ({len(questions)} questions)",
    )


def push_cached_bank(
    cache: Path,
    model_short: str,
    dataset: str,
    *,
    repo_id: str = DEFAULT_DATASET_REPO,
    shard_questions: int = 32,
    token: Optional[str] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Upload a locally generated bank, sharding it the same way Modal does."""
    from .generation import shard_plan

    bank = read_cached_bank(cache, model_short, dataset)
    uploaded: List[str] = []
    plan = shard_plan(sorted(bank), shard_questions)
    for shard_index, qids in plan:
        rows: List[Dict[str, Any]] = []
        for qid in qids:
            rows.extend({"qid": qid, **chain} for chain in bank[qid])
        uploaded.append(
            upload_bytes(
                encode_shard(rows),
                shard_remote_path(model_short, dataset, shard_index),
                repo_id=repo_id,
                token=token,
                commit_message=f"Add {model_short}/{dataset_key(dataset)} shard {shard_index:04d}",
            )
        )
    if manifest is not None:
        uploaded.append(
            upload_bytes(
                (json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n").encode("utf-8"),
                manifest_remote_path(model_short, dataset),
                repo_id=repo_id,
                token=token,
                commit_message=f"Update {model_short}/{dataset_key(dataset)} manifest",
            )
        )
    return uploaded


def pull_bank(
    cache: Path,
    model_short: str,
    dataset: str,
    *,
    repo_id: str = DEFAULT_DATASET_REPO,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Download a bank from the Hub into the local per-question cache layout."""
    from huggingface_hub import hf_hub_download

    resolved = resolve_token(token)
    prefix = f"gen/{safe_component(model_short)}/{dataset_key(dataset)}/"
    shards = list_bank_files(repo_id=repo_id, prefix=prefix, token=resolved)
    if not shards:
        raise FileNotFoundError(f"No shards under {prefix} in {repo_id}")
    questions = 0
    for remote in shards:
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=remote,
            token=resolved,
        )
        rows = decode_shard(Path(local).read_bytes())
        questions += len(materialize_shard(cache, model_short, dataset, rows))
    bank_root = cache / "gen" / safe_component(model_short) / dataset_key(dataset)
    manifest_path = bank_root / "manifest.json"
    try:
        manifest_local = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=manifest_remote_path(model_short, dataset),
            token=resolved,
        )
        write_json_atomic(
            manifest_path,
            json.loads(Path(manifest_local).read_text(encoding="utf-8")),
        )
    except Exception:  # noqa: BLE001 - a bank without a manifest is still usable
        manifest_path = None  # type: ignore[assignment]
    return {
        "repo_id": repo_id,
        "model": model_short,
        "dataset": dataset_key(dataset),
        "shards": len(shards),
        "questions": questions,
        "manifest": str(manifest_path) if manifest_path else None,
    }


def pull_questions(
    cache: Path,
    dataset: str,
    *,
    repo_id: str = DEFAULT_DATASET_REPO,
    token: Optional[str] = None,
) -> Path:
    """Download the question bank into ``cache/questions/{dataset}.jsonl``."""
    from huggingface_hub import hf_hub_download

    resolved = resolve_token(token)
    local = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=questions_remote_path(dataset),
        token=resolved,
    )
    destination = cache / "questions" / f"{dataset_key(dataset)}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(Path(local).read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def pull_embeddings(
    cache: Path,
    encoder_short: str,
    model_short: str,
    dataset: str,
    *,
    repo_id: str = DEFAULT_DATASET_REPO,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Download embedding shards into the blueprint's per-question ``.npy`` layout."""
    import numpy as np
    from huggingface_hub import hf_hub_download

    from .artifacts import embedding_path

    resolved = resolve_token(token)
    prefix = (
        f"emb/{safe_component(encoder_short)}/{safe_component(model_short)}/{dataset_key(dataset)}/"
    )
    shards = list_bank_files(repo_id=repo_id, prefix=prefix, token=resolved)
    if not shards:
        raise FileNotFoundError(f"No embedding shards under {prefix} in {repo_id}")
    questions = 0
    for remote in shards:
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=remote,
            token=resolved,
        )
        with np.load(local) as payload:
            for qid in payload.files:
                destination = embedding_path(
                    cache, encoder_short, model_short, dataset_key(dataset), qid
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.save(destination, payload[qid].astype(np.float32))
                questions += 1
    return {
        "repo_id": repo_id,
        "encoder": encoder_short,
        "model": model_short,
        "dataset": dataset_key(dataset),
        "shards": len(shards),
        "questions": questions,
    }
