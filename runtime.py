"""Reproducibility, device selection, provenance, and JSON output."""

from __future__ import annotations

import gc
import importlib
import inspect
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def amp_dtype_and_device() -> tuple[torch.dtype, torch.device]:
    if not torch.cuda.is_available():
        return torch.float32, torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return dtype, torch.device("cuda")


def grad_scaler(enabled: bool):
    use_scaler = enabled and torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
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
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False), encoding="utf-8")


def seed_deterministic(seed: int, deterministic: bool) -> None:
    """Optionally enable deterministic algorithms for full-volume experiments."""
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
