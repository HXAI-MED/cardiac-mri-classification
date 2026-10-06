"""Configuration, reproducibility, and experiment path helpers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import inspect
import json
import math
import random
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .protocol import protocols

if TYPE_CHECKING:
    import torch

# Shared


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def save_json_arrays(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
    )


def cinema_git_commit() -> str | None:
    try:
        cinema_module = importlib.import_module("cinema")
        repository = Path(inspect.getfile(cinema_module)).resolve().parent.parent
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def seed_cudnn(seed: int) -> None:
    """Use the central-slice experiments' deterministic cuDNN policy."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cleanup_cuda() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def amp_dtype_and_device() -> tuple[torch.dtype, torch.device]:
    import torch

    if not torch.cuda.is_available():
        return torch.float32, torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return dtype, torch.device("cuda")


def grad_scaler(enabled: bool):
    import torch

    use_scaler = (
        enabled and torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    )
    try:
        return torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_scaler)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), indent=2, allow_nan=False), encoding="utf-8"
    )


def checkpoint_root(args: argparse.Namespace, experiment: str | None = None) -> Path:
    """Keep each experiment's checkpoints and result readers in one namespace."""
    return args.output_dir / "checkpoints" / (experiment or args.experiment)


def prepare_run_directory(
    args: argparse.Namespace,
    directory: Path,
    config: DictConfig,
    *,
    completion_file: str = "summary.json",
    **details: Any,
) -> dict[str, Any] | None:
    """Reuse matching completed runs; refuse to overwrite a different experiment."""
    settings = jsonable(
        {
            "experiment": args.experiment,
            "cinema_commit": cinema_git_commit(),
            "config": OmegaConf.to_container(config, resolve=True),
            # ponytail: hash split metadata only; hash volumes if they can change in place.
            "data_metadata": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(config.data.dir).glob("*_metadata.csv"))
            },
            "amp": args.amp,
            "deterministic": getattr(args, "deterministic", None),
            **details,
        }
    )
    settings_path = directory / "run_settings.json"
    completed = directory / completion_file
    if settings_path.is_file():
        previous = json.loads(settings_path.read_text(encoding="utf-8"))
        if previous != settings:
            raise FileExistsError(
                f"Run settings differ from {settings_path}. "
                "Choose a new logging.dir to keep both experiments."
            )
    elif directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"Existing run has no verified settings: {directory}. "
            "Choose a new logging.dir; existing results were left in place."
        )
    if args.resume and completed.is_file():
        result = json.loads(completed.read_text(encoding="utf-8"))
        checkpoint = result.get("checkpoint")
        if checkpoint is not None and not Path(checkpoint).is_file():
            raise FileNotFoundError(
                f"Completed run checkpoint is missing: {checkpoint}"
            )
        print(f"[resume] {directory}")
        return result
    directory.mkdir(parents=True, exist_ok=True)
    # A forced rerun must not leave a previous completion marker after a failure.
    completed.unlink(missing_ok=True)
    save_json(settings_path, settings)
    return None


def seed_deterministic(seed: int, deterministic: bool) -> None:
    """Optionally enable deterministic algorithms for full-volume experiments."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Shared


def apply_overrides(args: argparse.Namespace, config: DictConfig) -> None:
    overrides = {
        "n_epochs": getattr(args, "epochs", None),
        "n_warmup_epochs": getattr(args, "warmup_epochs", None),
        "eval_interval": getattr(args, "eval_interval", None),
        "batch_size": getattr(args, "effective_batch_size", None),
        "batch_size_per_device": getattr(args, "batch_size_per_device", None),
        "n_workers": getattr(args, "workers", None),
        "lr": getattr(args, "lr", None),
        "weight_decay": getattr(args, "weight_decay", None),
    }
    for key, value in overrides.items():
        if value is not None:
            config.train[key] = value
    if getattr(args, "patience", None) is not None:
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
    # Hydra owns its resolver-dependent settings; direct YAML callers need only
    # the experiment configuration.
    experiment_config = OmegaConf.masked_copy(
        config, [key for key in config if key != "hydra"]
    )
    values = OmegaConf.to_container(
        experiment_config, resolve=True, throw_on_missing=True
    )
    arguments = {
        "experiment": experiment,
        "output_dir": Path(values["logging"]["dir"]).expanduser(),
    }
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
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int for seed in seeds)
    ):
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
    for name in (
        "lr",
        "weight_decay",
        "temperature",
        "supervised_weight",
        "calibration_tolerance",
    ):
        value = arguments.get(name)
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number or null")
    for name, value in arguments.items():
        if value is not None and (
            name.endswith(("_dir", "_raw", "_processed"))
            or name == "foundation_checkpoint"
        ):
            arguments[name] = Path(value).expanduser()
    return argparse.Namespace(**arguments)


# Acdc 3D

ACDC_3D_INITIALIZATION = protocols["shared"]["initialization"]


ACDC_3D_TASK_KEY = protocols["acdc_3d"]["task_key"]


def acdc_3d_task_root(args: argparse.Namespace, smoke: bool | None = None) -> Path:
    use_smoke = args.smoke if smoke is None else smoke
    group = "architecture_smoke" if use_smoke else "architecture_bank"
    return checkpoint_root(args, "acdc_3d") / group / "3d" / ACDC_3D_TASK_KEY


def acdc_3d_run_dir_for(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
) -> Path:
    return (
        acdc_3d_task_root(args)
        / backbone
        / formulation
        / ACDC_3D_INITIALIZATION
        / f"seed_{seed}"
    )


# Acdc Distillation

ACDC_DISTILLATION_INITIALIZATION = protocols["acdc_distillation"]["initialization"]


ACDC_DISTILLATION_TASK_KEY = protocols["acdc_3d"]["task_key"]


def acdc_distillation_task_root(
    args: argparse.Namespace, smoke: bool | None = None
) -> Path:
    use_smoke = args.smoke if smoke is None else smoke
    group = (
        "architecture_smoke_distilled" if use_smoke else "architecture_bank_distilled"
    )
    return (
        checkpoint_root(args, "acdc_distillation")
        / group
        / "3d"
        / ACDC_DISTILLATION_TASK_KEY
    )


def acdc_distillation_run_dir_for(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
) -> Path:
    return (
        acdc_distillation_task_root(args)
        / backbone
        / formulation
        / ACDC_DISTILLATION_INITIALIZATION
        / f"seed_{seed}"
    )


def teacher_seed_for(args: argparse.Namespace, student_seed: int) -> int:
    return student_seed if args.teacher_seed is None else args.teacher_seed


def teacher_paths(
    args: argparse.Namespace, student_seed: int
) -> tuple[Path, Path, int]:
    if args.teacher_dir is None:
        raise ValueError("model.teacher_dir is required for distillation training")
    directory = args.teacher_dir.expanduser().resolve()
    teacher_seed = teacher_seed_for(args, student_seed)
    checkpoint = directory / f"acdc_sax_{teacher_seed}.safetensors"
    config = directory / "config.yaml"
    missing = [str(path) for path in (checkpoint, config) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing fine-tuned ACDC ConvViT teacher files: {missing}"
        )
    return checkpoint, config, teacher_seed


# Architecture Bank


def apply_architecture_overrides(
    args: argparse.Namespace,
    config: DictConfig,
    dimensionality: int,
) -> None:
    """Apply explicit screening/full-run overrides without changing CineMA defaults silently."""
    overrides = {
        "n_epochs": args.epochs,
        "n_warmup_epochs": args.warmup_epochs,
        "eval_interval": args.eval_interval,
        "batch_size": args.effective_batch_size,
        "n_workers": args.workers,
        "lr": args.lr,
    }
    for key, value in overrides.items():
        if value is not None:
            config.train[key] = value
    if args.patience is not None:
        config.train.early_stopping.patience = args.patience
    batch_per_device = (
        args.batch_size_per_device_2d
        if dimensionality == 2
        else args.batch_size_per_device_3d
    )
    if batch_per_device is not None:
        config.train.batch_size_per_device = batch_per_device

    if args.smoke:
        config.train.n_epochs = 1
        config.train.n_warmup_epochs = 0
        config.train.eval_interval = 1
        config.train.batch_size = 4
        config.train.batch_size_per_device = 1
        config.train.n_workers = 0


# Mnms2 3D

MNMS2_3D_INITIALIZATION = protocols["shared"]["initialization"]


MNMS2_3D_TASK_KEY = protocols["mnms2_3d"]["task_key"]


def mnms2_3d_task_root(args: argparse.Namespace, smoke: bool | None = None) -> Path:
    use_smoke = args.smoke if smoke is None else smoke
    group = "architecture_smoke" if use_smoke else "architecture_bank"
    return checkpoint_root(args, "mnms2_3d") / group / "3d" / MNMS2_3D_TASK_KEY


def mnms2_3d_run_dir_for(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
) -> Path:
    return (
        mnms2_3d_task_root(args)
        / backbone
        / formulation
        / MNMS2_3D_INITIALIZATION
        / f"seed_{seed}"
    )


def artifact_path(path: Path) -> Path:
    """Map a run artifact to its category, keeping its experiment/seed hierarchy.

    Checkpoints and resume metadata stay together. Paths outside a checkpoints
    tree retain the original layout, which also supports existing result readers.
    """
    if path.name in {"history.csv", "command.txt"}:
        category = "logs"
    elif path.name == "predictions.csv":
        category = "predictions"
    elif path.suffix == ".csv" or path.name == "metrics.json":
        category = "metrics"
    else:
        return path
    for parent in path.parents:
        if parent.name == "checkpoints":
            return parent.parent / category / path.relative_to(parent)
    return path


def output_file(path: Path) -> Path:
    """Create the parent directory for a categorized output file."""
    path = artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
