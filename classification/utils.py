"""Training settings and path helpers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from training_code.classification.protocol import protocols


def apply_overrides(args: argparse.Namespace, config: DictConfig) -> None:
    overrides = {
        "n_epochs": args.epochs,
        "n_warmup_epochs": args.warmup_epochs,
        "eval_interval": args.eval_interval,
        "batch_size": args.effective_batch_size,
        "batch_size_per_device": args.batch_size_per_device,
        "n_workers": args.workers,
        "lr": args.lr,
    }
    for key, value in overrides.items():
        if value is not None:
            config.train[key] = value
    if args.patience is not None:
        config.train.early_stopping.patience = args.patience
    if args.smoke:
        config.train.n_epochs = 1
        config.train.n_warmup_epochs = 0
        config.train.eval_interval = 1
        config.train.batch_size = 4
        config.train.batch_size_per_device = 1
        config.train.n_workers = 0


apply_training_overrides = apply_overrides


def prepare_run(config: DictConfig, experiment: str) -> argparse.Namespace:
    """Validate YAML settings and adapt paths for the existing training routines."""
    values = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    arguments = {"output_dir": Path(values["logging"]["dir"]).expanduser()}
    renamed = {
        "n_epochs": "epochs",
        "n_warmup_epochs": "warmup_epochs",
        "batch_size": "effective_batch_size",
        "n_workers": "workers",
    }
    for section in ("data", "model", "train", "run"):
        for key, value in values[section].items():
            name = renamed.get(key, key)
            if name in arguments:
                raise ValueError(f"Duplicate setting: {name}")
            arguments[name] = value

    for name, choices in protocols["options"][experiment].items():
        selected = arguments[name]
        if not isinstance(selected, list) or not selected:
            raise ValueError(f"{name} must be a non-empty YAML list")
        invalid = [value for value in selected if value not in choices]
        if invalid:
            raise ValueError(f"Invalid {name}: {invalid}; choose from {list(choices)}")

    seeds = arguments["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("run.seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("run.seeds must be unique")
    for name in (
        "smoke",
        "resume",
        "amp",
        "deterministic",
        "force_preprocess",
        "retry_failures",
        "continue_on_error",
        "allow_calibration_failure",
    ):
        if name in arguments and type(arguments[name]) is not bool:
            raise ValueError(f"{name} must be true or false")
    for name in (
        "epochs",
        "warmup_epochs",
        "eval_interval",
        "patience",
        "effective_batch_size",
        "batch_size_per_device",
        "batch_size_per_device_2d",
        "batch_size_per_device_3d",
        "workers",
        "teacher_seed",
    ):
        value = arguments.get(name)
        if value is not None and type(value) is not int:
            raise ValueError(f"{name} must be an integer or null")
    for name in ("lr", "weight_decay", "temperature", "supervised_weight", "calibration_tolerance"):
        value = arguments.get(name)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f"{name} must be a finite number or null")
    for name, value in arguments.items():
        if value is not None and (name.endswith(("_dir", "_raw", "_processed")) or name == "foundation_checkpoint"):
            arguments[name] = Path(value).expanduser()
    return argparse.Namespace(**arguments)
