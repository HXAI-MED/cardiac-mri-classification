"""Training settings and path helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from training_code.classification.protocol import protocols

INITIALIZATION = protocols["shared"]["initialization"]
TASK_KEY = protocols["mnms2_3d"]["task_key"]


def task_root(args: argparse.Namespace, smoke: bool | None = None) -> Path:
    use_smoke = args.smoke if smoke is None else smoke
    group = "architecture_smoke" if use_smoke else "architecture_bank"
    return args.output_dir / group / "3d" / TASK_KEY


def run_dir_for(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
) -> Path:
    return task_root(args) / backbone / formulation / INITIALIZATION / f"seed_{seed}"
