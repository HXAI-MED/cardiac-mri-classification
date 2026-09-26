"""Classification models and feature extraction."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from cinema.convvit import get_model as get_convvit_model
from training_code.classification.protocol import protocols

CLASSES = protocols["acdc"]["classes"]
VIEW = protocols["shared"]["view"]


def load_teacher(checkpoint: Path, config_path: Path, device: torch.device) -> torch.nn.Module:
    config = OmegaConf.load(config_path)
    configured_classes = tuple(config.data[config.data.class_column])
    if configured_classes != CLASSES:
        raise RuntimeError(f"Teacher classes {configured_classes} do not match {CLASSES}")
    if config.model.views != VIEW:
        raise RuntimeError(f"Teacher view {config.model.views!r} is not {VIEW!r}")
    teacher = get_convvit_model(config)
    teacher.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    teacher.set_grad_ckpt(False)
    teacher.requires_grad_(False).eval().to(device)
    return teacher
