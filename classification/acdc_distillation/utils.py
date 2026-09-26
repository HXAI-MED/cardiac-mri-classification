"""Training settings and path helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from training_code.classification.protocol import protocols

INITIALIZATION = protocols["acdc_distillation"]["initialization"]
TASK_KEY = protocols["acdc_3d"]["task_key"]


def task_root(args: argparse.Namespace, smoke: bool | None = None) -> Path:
    use_smoke = args.smoke if smoke is None else smoke
    group = "architecture_smoke_distilled" if use_smoke else "architecture_bank_distilled"
    return args.output_dir / group / "3d" / TASK_KEY


def run_dir_for(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
) -> Path:
    return task_root(args) / backbone / formulation / INITIALIZATION / f"seed_{seed}"


def teacher_seed_for(args: argparse.Namespace, student_seed: int) -> int:
    return student_seed if args.teacher_seed is None else args.teacher_seed


def teacher_paths(args: argparse.Namespace, student_seed: int) -> tuple[Path, Path, int]:
    if args.teacher_dir is None:
        raise ValueError("model.teacher_dir is required for distillation training")
    directory = args.teacher_dir.expanduser().resolve()
    teacher_seed = teacher_seed_for(args, student_seed)
    checkpoint = directory / f"acdc_sax_{teacher_seed}.safetensors"
    config = directory / "config.yaml"
    missing = [str(path) for path in (checkpoint, config) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing fine-tuned ACDC ConvViT teacher files: {missing}")
    return checkpoint, config, teacher_seed
