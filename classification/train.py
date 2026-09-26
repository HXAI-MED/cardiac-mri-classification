"""Classification training."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from training_code.classification.dataset import processed_dir
from training_code.classification.protocol import TaskSpec
from training_code.runtime import cinema_git_commit, save_json


def foundation_checkpoint(args: argparse.Namespace) -> Path:
    if args.foundation_checkpoint is not None:
        # Keep the visible filename: CineMA selects the loader from .pt versus
        # .safetensors.  resolve() may dereference a Hugging Face cache symlink
        # to an extensionless blob and make the checkpoint format undetectable.
        path = Path(os.path.abspath(args.foundation_checkpoint.expanduser()))
        if not path.is_file():
            raise FileNotFoundError(path)
    else:
        path = Path(
            hf_hub_download(
                repo_id="mathpluscode/CineMA",
                filename="pretrained/cinema.safetensors",
                cache_dir=str(args.output_dir / "hf_cache"),
            )
        )
    if path.suffix not in (".pt", ".safetensors"):
        raise RuntimeError(f"CineMA checkpoint filename must retain a .pt or .safetensors suffix, got: {path}")
    return path


def hydra_override(name: str, value: Any) -> str:
    if value is None:
        rendered = "null"
    elif isinstance(value, bool):
        rendered = "true" if value else "false"
    else:
        rendered = str(value)
    return f"{name}={rendered}"


def find_single_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted((run_dir / "ckpt").glob("ckpt_*.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one retained checkpoint in {run_dir / 'ckpt'}, found {checkpoints}")
    return checkpoints[0]


def run_official_training(args: argparse.Namespace, spec: TaskSpec, mode: str, seed: int) -> None:
    data_root = processed_dir(args, spec.dataset)
    output_group = "official_smoke" if args.smoke else "official_training"
    run_dir = args.output_dir / output_group / spec.key / mode / f"seed_{seed}"
    done = run_dir / "DONE.json"
    if args.resume and done.exists():
        print(f"[resume] {spec.key}/{mode}/seed_{seed}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    module = f"cinema.classification.{spec.dataset}.train"
    command = [
        sys.executable,
        "-m",
        module,
        hydra_override("data.dir", data_root),
        hydra_override("model.views", spec.view),
        hydra_override("seed", seed),
        hydra_override("logging.dir", run_dir),
    ]
    if mode == "cinema_finetune":
        command += [
            hydra_override("model.name", "convvit"),
            hydra_override("model.ckpt_path", foundation_checkpoint(args)),
            hydra_override("model.freeze_pretrained", False),
        ]
    elif mode == "cinema_frozen":
        command += [
            hydra_override("model.name", "convvit"),
            hydra_override("model.ckpt_path", foundation_checkpoint(args)),
            hydra_override("model.freeze_pretrained", True),
        ]
    elif mode == "cinema_randinit":
        command += [hydra_override("model.name", "convvit"), hydra_override("model.ckpt_path", None)]
    elif mode == "resnet50_randinit":
        command += [
            hydra_override("model.name", "resnet"),
            hydra_override("model.resnet.depth", 50),
            hydra_override("model.ckpt_path", None),
        ]
    else:
        raise ValueError(mode)

    if args.smoke:
        command += [
            hydra_override("data.max_n_samples", 24),
            hydra_override("train.n_workers", 0),
            hydra_override("train.n_epochs", 1),
            hydra_override("train.n_warmup_epochs", 0),
            hydra_override("train.eval_interval", 1),
            hydra_override("train.batch_size", 4),
            hydra_override("train.batch_size_per_device", 1),
        ]

    (run_dir / "command.txt").write_text(" ".join(map(str, command)) + "\n", encoding="utf-8")
    print("[official]", " ".join(map(str, command)))
    subprocess.check_call(command)
    checkpoint = find_single_checkpoint(run_dir)
    eval_module = f"cinema.classification.{spec.dataset}.eval"
    eval_command = [
        sys.executable,
        "-m",
        eval_module,
        "--data_dir",
        str(data_root),
        "--ckpt_path",
        str(checkpoint),
        "--split",
        "test",
    ]
    subprocess.check_call(eval_command)
    metrics_path = checkpoint.parent / f"{spec.dataset}_eval_{checkpoint.stem}" / "test" / "classification_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Official evaluator did not create {metrics_path}")
    metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
    save_json(
        done,
        {
            "task": spec.key,
            "mode": mode,
            "seed": seed,
            "checkpoint": checkpoint,
            "cinema_commit": cinema_git_commit(),
            "smoke": args.smoke,
            "test": {key: float(value) for key, value in metrics.items()},
        },
    )
