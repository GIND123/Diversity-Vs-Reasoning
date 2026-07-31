"""Command-line interface for gate checks and experiment preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .constants import DEFAULT_CONFIG
from .env import load_local_env
from .experiment import prepare_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dvr", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    list_command = subcommands.add_parser("list", help="list registered experiments")
    list_command.add_argument("--json", action="store_true", dest="as_json")

    prepare = subcommands.add_parser("prepare", help="validate and prepare an experiment run")
    prepare.add_argument("experiment", help="experiment ID such as E1 or R1")
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--dry-run", action="store_true")

    status = subcommands.add_parser("status", help="show cache/gate readiness")
    status.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    bank = subcommands.add_parser("bank", help="manage chain banks on the Hugging Face Hub")
    bank_actions = bank.add_subparsers(dest="bank_command", required=True)

    bank_status = bank_actions.add_parser("status", help="list published bank shards")
    bank_status.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    bank_pull = bank_actions.add_parser("pull", help="download a bank into cache/gen")
    bank_pull.add_argument("--model", required=True, help="short model name, e.g. qwen2.5-3b")
    bank_pull.add_argument("--dataset", required=True, help="gsm8k or math")
    bank_pull.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    bank_push = bank_actions.add_parser("push", help="upload a locally generated bank")
    bank_push.add_argument("--model", required=True, help="short model name, e.g. qwen2.5-3b")
    bank_push.add_argument("--dataset", required=True, help="gsm8k or math")
    bank_push.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    generate = subcommands.add_parser("generate", help="print the Modal generation command")
    generate.add_argument("--model", required=True, help="full Hub model id")
    generate.add_argument("--dataset", required=True, help="gsm8k or math")
    generate.add_argument("--questions", type=int, default=50)
    generate.add_argument("--chains", type=int, default=1024)
    generate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def _hub_settings(config_path: Path) -> Dict[str, Any]:
    from .config import load_config, resolve_paths

    config = load_config(config_path)
    hub = dict(config.get("hf", {}))
    return {"paths": resolve_paths(config), "hub": hub, "config": config}


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_local_env()
    arguments = _parser().parse_args(argv)
    from .registry import EXPERIMENTS

    if arguments.command == "list":
        rows = [
            {
                "id": spec.experiment_id,
                "title": spec.title,
                "optional": spec.optional,
            }
            for spec in EXPERIMENTS.values()
        ]
        if arguments.as_json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                suffix = " (optional)" if row["optional"] else ""
                print(f"{row['id']:>2}  {row['title']}{suffix}")
        return 0
    if arguments.command == "prepare":
        key = arguments.experiment.upper()
        if key not in EXPERIMENTS:
            raise SystemExit(f"Unknown experiment {key}. Run `dvr list`.")
        result = prepare_experiment(
            EXPERIMENTS[key],
            config_path=arguments.config,
            dry_run=arguments.dry_run,
        )
        print(json.dumps(result, indent=2))
        return 0
    if arguments.command == "status":
        from .config import load_config, resolve_paths

        config = load_config(arguments.config)
        paths = resolve_paths(config)
        result = {
            "gate_stamp": paths.gate_stamp.exists(),
            "cache": {
                group: (paths.cache / group).exists()
                for group in ("gen", "emb", "spec", "sel", "agg")
            },
        }
        print(json.dumps(result, indent=2))
        return 0
    if arguments.command == "bank":
        from .hf_bank import list_bank_files, pull_bank, push_cached_bank

        settings = _hub_settings(arguments.config)
        repo_id = str(settings["hub"].get("dataset_repo"))
        if arguments.bank_command == "status":
            files = list_bank_files(repo_id=repo_id)
            print(json.dumps({"repo_id": repo_id, "files": files}, indent=2))
            return 0
        if arguments.bank_command == "pull":
            result = pull_bank(
                settings["paths"].cache,
                arguments.model,
                arguments.dataset,
                repo_id=repo_id,
            )
            print(json.dumps(result, indent=2))
            return 0
        if arguments.bank_command == "push":
            uploaded = push_cached_bank(
                settings["paths"].cache,
                arguments.model,
                arguments.dataset,
                repo_id=repo_id,
                shard_questions=int(settings["hub"].get("shard_questions", 16)),
            )
            print(json.dumps({"repo_id": repo_id, "uploaded": uploaded}, indent=2))
            return 0
        return 2
    if arguments.command == "generate":
        settings = _hub_settings(arguments.config)
        command = [
            "modal",
            "run",
            "modal_app/generate_chains.py",
            f"--model={arguments.model}",
            f"--dataset={arguments.dataset}",
            f"--questions={arguments.questions}",
            f"--chains={arguments.chains}",
            f"--repo={settings['hub'].get('dataset_repo')}",
        ]
        print(" ".join(command))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
