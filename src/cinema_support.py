"""Integration with the separately installed CineMA source checkout."""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from cinema.convvit import get_model as get_convvit_model
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from .protocol import TASKS, TaskSpec, protocols
from .utils import (
    apply_architecture_overrides,
    apply_overrides,
    checkpoint_root,
    cinema_git_commit,
    output_file,
    prepare_run_directory,
    save_json,
)


def processed_dir(args: argparse.Namespace, dataset: str) -> Path:
    explicit = getattr(
        args, f"{dataset}_processed", getattr(args, "processed_dir", None)
    )
    if explicit is not None:
        return explicit.expanduser().resolve()
    return (args.output_dir / "processed" / dataset).resolve()


def raw_dir(args: argparse.Namespace, dataset: str) -> Path | None:
    value = getattr(args, f"{dataset}_raw", getattr(args, "raw_dir", None))
    return None if value is None else value.expanduser().resolve()


def load_task_config(
    dataset: str, data_root: Path, seed: int = 0, view: str = "sax"
) -> DictConfig:
    """Read CineMA defaults directly; change only the selected data, view and seed."""
    package = importlib.import_module(f"cinema.classification.{dataset}")
    path = Path(inspect.getfile(package)).resolve().parent / "config.yaml"
    config = OmegaConf.load(path)
    config.data.dir = str(data_root)
    config.model.views = view
    config.seed = seed
    expected = TASKS[f"{dataset}_sax"].classes
    installed = tuple(config.data[config.data.class_column])
    if installed != expected:
        raise RuntimeError(
            f"Installed CineMA {dataset} classes changed: {installed} != {expected}"
        )
    return config


def local_config(spec: TaskSpec, data_root: Path, seed: int = 0) -> DictConfig:
    return load_task_config(spec.dataset, data_root, seed, spec.view)


def validate_raw_layout(dataset: str, root: Path) -> None:
    if dataset == "acdc":
        required = [
            root / "training",
            root / "testing",
            root / "training" / "patient001" / "Info.cfg",
        ]
    else:
        required = [
            root / "dataset_information.csv",
            root / "dataset",
            root / "dataset" / "001" / "001_LA_ED.nii.gz",
            root / "dataset" / "001" / "001_SA_ED.nii.gz",
        ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{dataset} raw-data layout is not CineMA-compatible. Missing: {missing}"
        )


def preprocess_dataset(args: argparse.Namespace, dataset: str) -> None:
    source = raw_dir(args, dataset)
    if source is None:
        setting = f"{dataset}_raw" if hasattr(args, f"{dataset}_raw") else "raw_dir"
        raise ValueError(f"data.{setting} is required for preprocessing {dataset}")
    validate_raw_layout(dataset, source)
    target = processed_dir(args, dataset)

    expected_files = [target / "train_metadata.csv", target / "test_metadata.csv"]
    if dataset == "mnms2":
        expected_files.append(target / "val_metadata.csv")
    if all(p.exists() for p in expected_files) and not args.force_preprocess:
        print(f"[preprocess] Reusing existing official {dataset} output: {target}")
        return

    target.mkdir(parents=True, exist_ok=True)
    module = f"cinema.data.{dataset}.preprocess"
    command = [
        sys.executable,
        "-m",
        module,
        "--data_dir",
        str(source),
        "--out_dir",
        str(target),
    ]
    print("[preprocess]", shlex.join(command))
    environment = os.environ.copy()
    # Some M&Ms2 NIfTI files contain a valid but non-orthogonal sform.  ITK's
    # permissive mode preserves the official CineMA preprocessing path while
    # allowing those files to be read (ITK may still emit informational warnings).
    environment.setdefault("ITK_NIFTI_SFORM_PERMISSIVE", "1")
    subprocess.check_call(command, env=environment)


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
        raise RuntimeError(
            f"CineMA checkpoint filename must retain a .pt or .safetensors suffix, got: {path}"
        )
    return path


def find_single_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted((run_dir / "ckpt").glob("ckpt_*.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected one retained checkpoint in {run_dir / 'ckpt'}, found {checkpoints}"
        )
    return checkpoints[0]


def load_teacher(
    checkpoint: Path, config_path: Path, device: torch.device
) -> torch.nn.Module:
    config = OmegaConf.load(config_path)
    configured_classes = tuple(config.data[config.data.class_column])
    if configured_classes != protocols["acdc"]["classes"]:
        raise RuntimeError(
            f"Teacher classes {configured_classes} do not match {protocols['acdc']['classes']}"
        )
    if config.model.views != protocols["shared"]["view"]:
        raise RuntimeError(
            f"Teacher view {config.model.views!r} is not {protocols['shared']['view']!r}"
        )
    teacher = get_convvit_model(config)
    teacher.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    teacher.set_grad_ckpt(False)
    teacher.requires_grad_(False).eval().to(device)
    return teacher


def run_official_training(
    args: argparse.Namespace, spec: TaskSpec, mode: str, seed: int
) -> None:
    """Launch the original CineMA trainer with a saved, effective task config."""
    data_root = processed_dir(args, spec.dataset)
    group = "official_smoke" if args.smoke else "official_training"
    run_dir = checkpoint_root(args) / group / spec.key / mode / f"seed_{seed}"
    config = local_config(spec, data_root, seed)
    if args.experiment == "architecture_bank":
        apply_architecture_overrides(args, config, spec.dimensionality)
    else:
        apply_overrides(args, config)
    config.logging.dir = str(run_dir)
    if mode in ("cinema_finetune", "cinema_frozen"):
        config.model.name = "convvit"
        config.model.ckpt_path = str(foundation_checkpoint(args))
        config.model.freeze_pretrained = mode == "cinema_frozen"
    elif mode == "cinema_randinit":
        config.model.name = "convvit"
        config.model.ckpt_path = None
    elif mode == "resnet50_randinit":
        config.model.name = "resnet"
        config.model.resnet.depth = 50
        config.model.ckpt_path = None
    else:
        raise ValueError(mode)
    if args.smoke:
        config.data.max_n_samples = 24
    previous = prepare_run_directory(
        args, run_dir, config, completion_file="DONE.json", mode=mode, seed=seed
    )
    if previous is not None:
        return
    OmegaConf.save(config, run_dir / "cinema_config.yaml")
    command = [
        sys.executable,
        "-m",
        f"cinema.classification.{spec.dataset}.train",
        "--config-path",
        str(run_dir),
        "--config-name",
        "cinema_config",
    ]
    output_file(run_dir / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    print("[official]", shlex.join(command))
    subprocess.check_call(command)
    checkpoint = find_single_checkpoint(run_dir)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            f"cinema.classification.{spec.dataset}.eval",
            "--data_dir",
            str(data_root),
            "--ckpt_path",
            str(checkpoint),
            "--split",
            "test",
        ]
    )
    metrics_path = (
        checkpoint.parent
        / f"{spec.dataset}_eval_{checkpoint.stem}"
        / "test/classification_metrics.csv"
    )
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Official evaluator did not create {metrics_path}")
    metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
    save_json(
        run_dir / "DONE.json",
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
