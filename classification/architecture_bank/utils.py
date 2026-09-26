"""Training settings and path helpers."""

from __future__ import annotations

import argparse

from omegaconf import DictConfig


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
    batch_per_device = args.batch_size_per_device_2d if dimensionality == 2 else args.batch_size_per_device_3d
    if batch_per_device is not None:
        config.train.batch_size_per_device = batch_per_device

    if args.smoke:
        config.train.n_epochs = 1
        config.train.n_warmup_epochs = 0
        config.train.eval_interval = 1
        config.train.batch_size = 4
        config.train.batch_size_per_device = 1
        config.train.n_workers = 0
