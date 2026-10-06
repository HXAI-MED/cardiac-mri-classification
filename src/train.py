"""Shared CNN training mechanics and experiment-specific model setup."""

from __future__ import annotations

import argparse
import math
import time
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from cinema.optim import adjust_learning_rate, get_n_accum_steps
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from .cinema_support import load_task_config, local_config, processed_dir
from .dataset import (
    acdc_2d_make_loaders,
    acdc_3d_local_config,
    acdc_3d_make_loaders,
    acdc_3d_processed_dir,
    acdc_3d_split_metadata,
    make_loaders,
    mnms2_3d_local_config,
    mnms2_3d_make_loaders,
    mnms2_3d_processed_dir,
    mnms2_3d_split_metadata,
    mnms2_sax_2d_make_loaders,
    split_metadata,
)
from .evaluate import (
    acdc_2d_evaluate,
    acdc_2d_save_predictions,
    acdc_3d_evaluate,
    acdc_3d_save_predictions,
    architecture_bank_evaluate_cnn,
    mnms2_3d_evaluate,
    mnms2_3d_save_predictions,
    mnms2_sax_2d_evaluate,
    mnms2_sax_2d_save_predictions,
    reproduction_evaluate_cnn,
    save_predictions,
)
from .models import (
    ACDC3DClassifier,
    ArchitectureBank2DClassifier,
    ArchitectureBank3DClassifier,
    MnMs2SAX3DClassifier,
    Reproduction2DClassifier,
    acdc_3d_feature_layer_candidates,
    architecture_bank_feature_layer_candidates,
    feature_layer_candidates,
    mnms2_3d_feature_layer_candidates,
)
from .models_2d import SAX2DClassifier
from .protocol import TASKS, TaskSpec, protocols
from .utils import (
    acdc_3d_run_dir_for,
    amp_dtype_and_device,
    apply_architecture_overrides,
    apply_overrides,
    apply_training_overrides,
    checkpoint_root,
    cinema_git_commit,
    cleanup_cuda,
    grad_scaler,
    mnms2_3d_run_dir_for,
    output_file,
    prepare_run_directory,
    save_json,
    save_json_arrays,
)

ACDC_3D_CINEMA_RANDINIT_WEIGHT_DECAY = protocols["acdc_3d"][
    "cinema_randinit_weight_decay"
]


ACDC_3D_CLASSES = protocols["acdc"]["classes"]


ACDC_3D_DATASET = protocols["acdc_3d"]["dataset"]


ACDC_3D_INITIALIZATION = protocols["shared"]["initialization"]


ACDC_3D_METRIC_NAMES = protocols["shared"]["metric_names"]


ACDC_3D_PROTOCOL_VERSION = protocols["acdc_3d"]["protocol_version"]


ACDC_3D_TASK_KEY = protocols["acdc_3d"]["task_key"]


ACDC_3D_VIEW = protocols["shared"]["view"]


MODELS_2D = protocols["architecture_bank"]["models_2d"]


MODELS_3D = protocols["architecture_bank"]["models_3d"]


MNMS2_3D_CLASSES = protocols["mnms2_3d"]["classes"]


MNMS2_3D_DATASET = protocols["mnms2_3d"]["dataset"]


MNMS2_3D_INITIALIZATION = protocols["shared"]["initialization"]


MNMS2_3D_METRIC_NAMES = protocols["shared"]["metric_names"]


MNMS2_3D_TASK_KEY = protocols["mnms2_3d"]["task_key"]


MNMS2_3D_VIEW = protocols["shared"]["view"]


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    config: DictConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp: bool,
    epoch: int,
    accumulation_steps: int,
    *,
    view: str = "sax",
) -> dict[str, float]:
    """Use CineMA's schedule and loss for any CNN; flush the final gradient group."""
    n_steps = len(loader)
    if n_steps == 0:
        raise ValueError(
            "Training loader is empty; reduce train.batch_size_per_device."
        )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = total_correct = total_seen = 0
    for step, batch in enumerate(tqdm(loader, desc=f"train/e{epoch + 1}", leave=False)):
        learning_rate = adjust_learning_rate(
            optimizer=optimizer,
            step=epoch + step / n_steps,
            warmup_steps=int(config.train.n_warmup_epochs),
            max_n_steps=int(config.train.n_epochs),
            lr=float(config.train.lr),
            min_lr=float(config.train.min_lr),
        )
        image = batch[f"{view}_image"].to(device, non_blocking=True)
        target = batch["label"].long().to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(image)
            raw_loss = F.cross_entropy(
                logits, target, label_smoothing=float(config.train.label_smoothing)
            )
            group_start = (step // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, n_steps - group_start)
            loss = raw_loss / group_size
        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0 or step + 1 == n_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.train.clip_grad)
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(raw_loss.item()) * target.size(0)
        total_correct += int((logits.argmax(1) == target).sum().item())
        total_seen += target.size(0)
    return {
        "train_loss": total_loss / total_seen,
        "train_accuracy": total_correct / total_seen,
        "lr": learning_rate,
    }


def fit_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: DictConfig,
    run_dir: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp: bool,
    checkpoint_metadata: dict[str, Any],
    evaluate: Any,
    export_predictions: Any,
    *,
    view: str = "sax",
    weight_decay: float | None = None,
) -> tuple[float, int]:
    """Train a CNN using CineMA defaults and reload its best validation checkpoint."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.train.lr),
        betas=tuple(config.train.betas),
        weight_decay=float(config.train.weight_decay)
        if weight_decay is None
        else weight_decay,
    )
    scaler = grad_scaler(amp)
    accumulation_steps = get_n_accum_steps(
        batch_size=int(config.train.batch_size),
        batch_size_per_device=int(config.train.batch_size_per_device),
        world_size=1,
    )
    best_mcc = -math.inf
    best_epoch = -1
    bad_evaluations = 0
    history = []
    checkpoint = run_dir / "best_val_mcc.pt"
    for epoch in range(int(config.train.n_epochs)):
        stats = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            config,
            device,
            amp_dtype,
            amp,
            epoch,
            accumulation_steps,
            view=view,
        )
        final_epoch = epoch + 1 == int(config.train.n_epochs)
        if (epoch + 1) % int(config.train.eval_interval) != 0 and not final_epoch:
            continue
        pids, targets, probabilities, metrics, *timing = evaluate(model, val_loader)
        row = {"epoch": epoch + 1, **stats}
        row.update(
            {
                f"val_{name}": metrics[name]
                for name in protocols["shared"]["metric_names"]
            }
        )
        if timing:
            row["validation_seconds"] = timing[0]
        history.append(row)
        pd.DataFrame(history).to_csv(output_file(run_dir / "history.csv"), index=False)
        print(
            f"[eval] {run_dir.name} epoch={epoch + 1} MCC={metrics['mcc']:.4f} AUROC={metrics['roc_auc']:.4f}"
        )
        if metrics["mcc"] > best_mcc + float(config.train.early_stopping.min_delta):
            best_mcc = float(metrics["mcc"])
            best_epoch = epoch + 1
            bad_evaluations = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    **checkpoint_metadata,
                    "epoch": best_epoch,
                },
                checkpoint,
            )
            export_predictions(
                run_dir / "best_validation",
                pids,
                targets,
                probabilities,
                metrics=metrics,
            )
        else:
            bad_evaluations += 1
        if bad_evaluations >= int(config.train.early_stopping.patience):
            break
    if best_epoch < 0:
        raise RuntimeError(f"No validation checkpoint was written for {run_dir}")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return best_mcc, best_epoch


def train_mid_sax_2d(
    args: argparse.Namespace,
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    backbone: str,
    formulation: str,
    initialization: str,
    seed: int,
    *,
    dataset: str,
) -> dict[str, Any]:
    classes = TASKS[f"{dataset}_sax"].classes
    task_key = protocols["acdc_2d" if dataset == "acdc" else "mnms2_sax_2d"]["task_key"]
    make_loaders = (
        acdc_2d_make_loaders if dataset == "acdc" else mnms2_sax_2d_make_loaders
    )
    evaluate = acdc_2d_evaluate if dataset == "acdc" else mnms2_sax_2d_evaluate
    export = (
        acdc_2d_save_predictions if dataset == "acdc" else mnms2_sax_2d_save_predictions
    )
    family = (
        "acdc_2d_architecture_bank"
        if dataset == "acdc"
        else "mnms2_sax_mid_2d_architecture_bank"
    )
    output_group = "architecture_smoke" if args.smoke else "architecture_bank"
    run_dir = (
        checkpoint_root(args)
        / output_group
        / "2d"
        / task_key
        / backbone
        / formulation
        / initialization
        / f"seed_{seed}"
    )
    summary_path = run_dir / "summary.json"

    config = load_task_config(dataset, data_root, seed)
    apply_overrides(args, config)
    previous = prepare_run_directory(
        args,
        run_dir,
        config,
        backbone=backbone,
        formulation=formulation,
        seed=seed,
        initialization=initialization,
    )
    if previous is not None:
        return previous
    train_loader, val_loader, test_loader = make_loaders(
        data_root, splits, config, seed
    )
    amp_dtype, device = amp_dtype_and_device()
    model = SAX2DClassifier(backbone, formulation, initialization, len(classes)).to(
        device
    )
    feature_layers = feature_layer_candidates(model, backbone)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    architecture = {
        "family": family,
        "task": task_key,
        "dataset": dataset,
        "view": "sax_mid",
        "dimensionality": 2,
        "input_policy": "patient_mid_sax_ed_es",
        "backbone": backbone,
        "formulation": formulation,
        "initialization": initialization,
        "classes": list(classes),
        "feature_dim": model.feature_dim,
        "feature_layer_candidates": feature_layers,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "cinema_commit": cinema_git_commit(),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    save_json_arrays(run_dir / "architecture.json", architecture)
    print(
        f"[architecture] {dataset}/2d/{backbone}/{formulation}/{initialization}/seed{seed} "
        f"parameters={parameter_count:,} feature_layers={feature_layers}"
    )

    checkpoint = run_dir / "best_val_mcc.pt"

    best_mcc, best_epoch = fit_classifier(
        model,
        train_loader,
        val_loader,
        config,
        run_dir,
        device,
        amp_dtype,
        args.amp,
        {
            **{
                key: architecture[key]
                for key in (
                    "family",
                    "task",
                    "classes",
                    "dimensionality",
                    "input_policy",
                    "backbone",
                    "formulation",
                    "initialization",
                    "feature_dim",
                    "feature_layer_candidates",
                    "cinema_commit",
                )
            },
            "seed": seed,
        },
        partial(evaluate, device=device, amp_dtype=amp_dtype, amp=args.amp),
        export,
        view="sax",
    )

    if not checkpoint.is_file():
        raise RuntimeError(f"No checkpoint was created for {run_dir}")
    test_pids, test_y, test_probs, test_metrics = evaluate(
        model, test_loader, device, amp_dtype, amp=args.amp
    )
    export(run_dir / "test", test_pids, test_y, test_probs, test_metrics)
    summary = {
        "family": family,
        "task": task_key,
        "dataset": dataset,
        "view": "sax_mid",
        "dimensionality": 2,
        "input_policy": "patient_mid_sax_ed_es",
        "backbone": backbone,
        "formulation": formulation,
        "initialization": initialization,
        "seed": seed,
        "best_val_mcc": best_mcc,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "feature_layer_candidates": feature_layers,
        "checkpoint": checkpoint,
        "test": {
            key: test_metrics[key]
            for key in (
                "accuracy",
                "f1",
                "macro_f1",
                "balanced_accuracy",
                "mcc",
                "roc_auc",
            )
        },
    }
    save_json_arrays(summary_path, summary)
    del model
    cleanup_cuda()
    return summary


def acdc_3d_train_one(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
) -> dict[str, Any]:
    run_dir = acdc_3d_run_dir_for(args, backbone, formulation, seed)
    summary_path = run_dir / "summary.json"
    failure_path = run_dir / "failure.json"

    started = time.perf_counter()
    data_root = acdc_3d_processed_dir(args)
    splits = acdc_3d_split_metadata(data_root)
    config = acdc_3d_local_config(data_root, seed)
    apply_training_overrides(args, config)
    previous = prepare_run_directory(
        args,
        run_dir,
        config,
        backbone=backbone,
        formulation=formulation,
        seed=seed,
        protocol_version=ACDC_3D_PROTOCOL_VERSION,
        optimizer_weight_decay=ACDC_3D_CINEMA_RANDINIT_WEIGHT_DECAY
        if args.weight_decay is None
        else args.weight_decay,
    )
    if previous is not None:
        return previous
    if args.resume and failure_path.is_file() and not args.retry_failures:
        print(f"[resume] skipping recorded failure: {failure_path}")
        return {}
    patch_size = tuple(int(value) for value in config.data.sax.patch_size)
    train_loader, val_loader, test_loader = acdc_3d_make_loaders(
        config,
        data_root,
        splits,
        seed=seed,
        deterministic=args.deterministic,
    )
    amp_dtype, device = amp_dtype_and_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = ACDC3DClassifier(backbone, formulation, len(ACDC_3D_CLASSES)).to(device)
    feature_layers = acdc_3d_feature_layer_candidates(model, backbone)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    architecture = {
        "protocol_version": ACDC_3D_PROTOCOL_VERSION,
        "family": "acdc_3d_architecture_bank",
        "dataset": ACDC_3D_DATASET,
        "task": ACDC_3D_TASK_KEY,
        "view": ACDC_3D_VIEW,
        "dimensionality": 3,
        "input_policy": "full_sax_ed_es_volume_per_patient",
        "backbone": backbone,
        "formulation": formulation,
        "initialization": ACDC_3D_INITIALIZATION,
        "classes": list(ACDC_3D_CLASSES),
        "feature_dim": model.feature_dim,
        "feature_layer_candidates": feature_layers,
        "anisotropic_adaptation": (
            "cinema_native_resnet3d"
            if backbone.startswith("resnet")
            else "final_dense_transition_pool_2x2x1"
            if backbone.startswith("densenet")
            else "preserve_slice_axis_stride_1"
            if backbone.startswith("efficientnet")
            else "convnext_native_anisotropic"
            if backbone.startswith("convnext3d")
            else None
        ),
        "volumetric_architecture_source": (
            "torchvision_mobilenetv3_configuration_inflated_to_3d"
            if backbone.startswith("mobilenet_v3")
            else "cinema_native_resnet3d"
            if backbone.startswith("resnet")
            else "torchvision_densenet161_configuration_in_monai_densenet3d"
            if backbone == "densenet161_3d"
            else "monai_native_3d"
            if not backbone.startswith("convnext3d")
            else "custom_convnext_tiny_3d"
        ),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "optimizer_weight_decay": (
            ACDC_3D_CINEMA_RANDINIT_WEIGHT_DECAY
            if args.weight_decay is None
            else args.weight_decay
        ),
        "cinema_commit": cinema_git_commit(),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    save_json(run_dir / "architecture.json", architecture)
    print(
        f"[train] {ACDC_3D_TASK_KEY}/{backbone}/{formulation}/{ACDC_3D_INITIALIZATION}/seed_{seed} "
        f"parameters={parameter_count:,} layers={feature_layers}"
    )

    optimizer_weight_decay = (
        ACDC_3D_CINEMA_RANDINIT_WEIGHT_DECAY
        if args.weight_decay is None
        else args.weight_decay
    )
    checkpoint = run_dir / "best_val_mcc.pt"

    best_mcc, best_epoch = fit_classifier(
        model,
        train_loader,
        val_loader,
        config,
        run_dir,
        device,
        amp_dtype,
        args.amp,
        {
            "protocol_version": ACDC_3D_PROTOCOL_VERSION,
            "family": "acdc_3d_architecture_bank",
            "dataset": ACDC_3D_DATASET,
            "task": ACDC_3D_TASK_KEY,
            "classes": list(ACDC_3D_CLASSES),
            "view": ACDC_3D_VIEW,
            "dimensionality": 3,
            "input_policy": "full_sax_ed_es_volume_per_patient",
            "backbone": backbone,
            "formulation": formulation,
            "initialization": ACDC_3D_INITIALIZATION,
            "feature_dim": model.feature_dim,
            "feature_layer_candidates": feature_layers,
            "seed": seed,
            "cinema_commit": cinema_git_commit(),
        },
        partial(
            acdc_3d_evaluate,
            patch_size=patch_size,
            device=device,
            amp_dtype=amp_dtype,
            amp=args.amp,
        ),
        acdc_3d_save_predictions,
        view="sax",
        weight_decay=optimizer_weight_decay,
    )

    if not checkpoint.is_file():
        raise RuntimeError(f"No validation checkpoint was written: {checkpoint}")
    test_pids, test_y, test_probabilities, test_metrics, test_seconds = (
        acdc_3d_evaluate(model, test_loader, patch_size, device, amp_dtype, args.amp)
    )
    acdc_3d_save_predictions(
        run_dir / "test", test_pids, test_y, test_probabilities, test_metrics
    )

    training_seconds = time.perf_counter() - started
    peak_gpu_memory_gb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if device.type == "cuda"
        else 0.0
    )
    summary = {
        "protocol_version": ACDC_3D_PROTOCOL_VERSION,
        "family": "acdc_3d_architecture_bank",
        "dataset": ACDC_3D_DATASET,
        "task": ACDC_3D_TASK_KEY,
        "view": ACDC_3D_VIEW,
        "dimensionality": 3,
        "input_policy": "full_sax_ed_es_volume_per_patient",
        "backbone": backbone,
        "formulation": formulation,
        "initialization": ACDC_3D_INITIALIZATION,
        "seed": seed,
        "best_val_mcc": best_mcc,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "optimizer_weight_decay": optimizer_weight_decay,
        "feature_layer_candidates": feature_layers,
        "checkpoint": checkpoint,
        "training_seconds": training_seconds,
        "test_inference_seconds": test_seconds,
        "test_inference_ms_per_patient": 1000.0 * test_seconds / max(len(test_y), 1),
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "test": {name: test_metrics[name] for name in ACDC_3D_METRIC_NAMES},
    }
    save_json(summary_path, summary)
    if failure_path.exists():
        failure_path.unlink()
    print(
        f"[test] {backbone}/{formulation}/seed_{seed}: "
        f"MCC={test_metrics['mcc']:.4f} AUROC={test_metrics['roc_auc']:.4f} "
        f"macro-F1={test_metrics['macro_f1']:.4f}"
    )
    del model
    cleanup_cuda()
    return summary


def train_architecture(
    args: argparse.Namespace,
    spec: TaskSpec,
    backbone: str,
    formulation: str,
    initialization: str,
    seed: int,
) -> dict[str, Any]:
    dimensionality = spec.dimensionality
    if dimensionality == 2 and backbone not in MODELS_2D:
        raise ValueError(f"{backbone} is not a registered 2-D architecture")
    if dimensionality == 3 and backbone not in MODELS_3D:
        raise ValueError(f"{backbone} is not a registered 3-D architecture")
    if dimensionality == 3 and initialization != "randinit":
        raise ValueError("The native 3-D bank currently supports randinit only")

    output_group = "architecture_smoke" if args.smoke else "architecture_bank"
    run_dir = (
        checkpoint_root(args)
        / output_group
        / f"{dimensionality}d"
        / spec.key
        / backbone
        / formulation
        / initialization
        / f"seed_{seed}"
    )
    summary_path = run_dir / "summary.json"

    data_root = processed_dir(args, spec.dataset)
    splits = split_metadata(spec, data_root)
    config = local_config(spec, data_root, seed)
    apply_architecture_overrides(args, config, dimensionality)
    previous = prepare_run_directory(
        args,
        run_dir,
        config,
        backbone=backbone,
        formulation=formulation,
        seed=seed,
        initialization=initialization,
    )
    if previous is not None:
        return previous
    train_loader, val_loader, test_loader = make_loaders(
        spec, config, data_root, splits, seed
    )
    amp_dtype, device = amp_dtype_and_device()
    if dimensionality == 2:
        model = ArchitectureBank2DClassifier(
            backbone, formulation, initialization, len(spec.classes)
        ).to(device)
    else:
        model = ArchitectureBank3DClassifier(
            backbone, formulation, len(spec.classes)
        ).to(device)
    feature_layers = architecture_bank_feature_layer_candidates(
        model, backbone, dimensionality
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    save_json(
        run_dir / "architecture.json",
        {
            "task": spec.key,
            "dataset": spec.dataset,
            "view": spec.view,
            "dimensionality": dimensionality,
            "backbone": backbone,
            "formulation": formulation,
            "initialization": initialization,
            "classes": spec.classes,
            "feature_dim": model.feature_dim,
            "feature_layer_candidates": feature_layers,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "cinema_commit": cinema_git_commit(),
            "config": OmegaConf.to_container(config, resolve=True),
        },
    )
    print(
        f"[architecture] {dimensionality}d/{spec.key}/{backbone}/{formulation}/{initialization}/seed{seed} "
        f"parameters={parameter_count:,} feature_layers={feature_layers}"
    )
    checkpoint = run_dir / "best_val_mcc.pt"

    best_mcc, best_epoch = fit_classifier(
        model,
        train_loader,
        val_loader,
        config,
        run_dir,
        device,
        amp_dtype,
        args.amp,
        {
            "family": "architecture_bank",
            "task": spec.key,
            "classes": list(spec.classes),
            "dimensionality": dimensionality,
            "backbone": backbone,
            "formulation": formulation,
            "initialization": initialization,
            "feature_dim": model.feature_dim,
            "feature_layer_candidates": feature_layers,
            "seed": seed,
            "cinema_commit": cinema_git_commit(),
        },
        partial(
            architecture_bank_evaluate_cnn,
            spec=spec,
            device=device,
            amp_dtype=amp_dtype,
            amp=args.amp,
        ),
        partial(save_predictions, classes=spec.classes),
        view=spec.view,
    )

    pids, y, probs, test_metrics = architecture_bank_evaluate_cnn(
        model, test_loader, spec, device, amp_dtype, amp=args.amp
    )
    save_predictions(run_dir / "test", pids, y, probs, spec.classes, test_metrics)
    summary = {
        "family": "architecture_bank",
        "task": spec.key,
        "dimensionality": dimensionality,
        "backbone": backbone,
        "formulation": formulation,
        "initialization": initialization,
        "seed": seed,
        "best_val_mcc": best_mcc,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "feature_layer_candidates": feature_layers,
        "checkpoint": checkpoint,
        "test": {
            k: test_metrics[k]
            for k in (
                "accuracy",
                "f1",
                "macro_f1",
                "balanced_accuracy",
                "mcc",
                "roc_auc",
            )
        },
    }
    save_json(summary_path, summary)
    del model
    cleanup_cuda()
    return summary


def mnms2_3d_train_one(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
) -> dict[str, Any]:
    run_dir = mnms2_3d_run_dir_for(args, backbone, formulation, seed)
    summary_path = run_dir / "summary.json"
    failure_path = run_dir / "failure.json"

    started = time.perf_counter()
    data_root = mnms2_3d_processed_dir(args)
    splits = mnms2_3d_split_metadata(data_root)
    config = mnms2_3d_local_config(data_root, seed)
    apply_training_overrides(args, config)
    previous = prepare_run_directory(
        args,
        run_dir,
        config,
        backbone=backbone,
        formulation=formulation,
        seed=seed,
        optimizer_weight_decay=config.train.weight_decay,
    )
    if previous is not None:
        return previous
    if args.resume and failure_path.is_file() and not args.retry_failures:
        print(f"[resume] skipping recorded failure: {failure_path}")
        return {}
    patch_size = tuple(int(value) for value in config.data.sax.patch_size)
    train_loader, val_loader, test_loader = mnms2_3d_make_loaders(
        config,
        data_root,
        splits,
        seed=seed,
        deterministic=args.deterministic,
    )
    amp_dtype, device = amp_dtype_and_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = MnMs2SAX3DClassifier(backbone, formulation, len(MNMS2_3D_CLASSES)).to(
        device
    )
    feature_layers = mnms2_3d_feature_layer_candidates(model, backbone)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    architecture = {
        "family": "mnms2_3d_architecture_bank",
        "dataset": MNMS2_3D_DATASET,
        "task": MNMS2_3D_TASK_KEY,
        "view": MNMS2_3D_VIEW,
        "dimensionality": 3,
        "input_policy": "full_sax_ed_es_volume_per_patient",
        "backbone": backbone,
        "formulation": formulation,
        "initialization": MNMS2_3D_INITIALIZATION,
        "classes": list(MNMS2_3D_CLASSES),
        "feature_dim": model.feature_dim,
        "feature_layer_candidates": feature_layers,
        "anisotropic_adaptation": (
            "final_dense_transition_pool_2x2x1"
            if backbone.startswith("densenet")
            else "last_axis_replicate_padding_to_32"
            if backbone.startswith("efficientnet")
            else "convnext_native_anisotropic"
            if backbone.startswith("convnext3d")
            else None
        ),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "cinema_commit": cinema_git_commit(),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    save_json(run_dir / "architecture.json", architecture)
    print(
        f"[train] {MNMS2_3D_TASK_KEY}/{backbone}/{formulation}/{MNMS2_3D_INITIALIZATION}/seed_{seed} "
        f"parameters={parameter_count:,} layers={feature_layers}"
    )

    checkpoint = run_dir / "best_val_mcc.pt"

    best_mcc, best_epoch = fit_classifier(
        model,
        train_loader,
        val_loader,
        config,
        run_dir,
        device,
        amp_dtype,
        args.amp,
        {
            "family": "mnms2_3d_architecture_bank",
            "dataset": MNMS2_3D_DATASET,
            "task": MNMS2_3D_TASK_KEY,
            "classes": list(MNMS2_3D_CLASSES),
            "view": MNMS2_3D_VIEW,
            "dimensionality": 3,
            "input_policy": "full_sax_ed_es_volume_per_patient",
            "backbone": backbone,
            "formulation": formulation,
            "initialization": MNMS2_3D_INITIALIZATION,
            "feature_dim": model.feature_dim,
            "feature_layer_candidates": feature_layers,
            "seed": seed,
            "cinema_commit": cinema_git_commit(),
        },
        partial(
            mnms2_3d_evaluate,
            patch_size=patch_size,
            device=device,
            amp_dtype=amp_dtype,
            amp=args.amp,
        ),
        mnms2_3d_save_predictions,
        view="sax",
    )

    if not checkpoint.is_file():
        raise RuntimeError(f"No validation checkpoint was written: {checkpoint}")
    test_pids, test_y, test_probabilities, test_metrics, test_seconds = (
        mnms2_3d_evaluate(model, test_loader, patch_size, device, amp_dtype, args.amp)
    )
    mnms2_3d_save_predictions(
        run_dir / "test", test_pids, test_y, test_probabilities, test_metrics
    )

    training_seconds = time.perf_counter() - started
    peak_gpu_memory_gb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if device.type == "cuda"
        else 0.0
    )
    summary = {
        "family": "mnms2_3d_architecture_bank",
        "dataset": MNMS2_3D_DATASET,
        "task": MNMS2_3D_TASK_KEY,
        "view": MNMS2_3D_VIEW,
        "dimensionality": 3,
        "input_policy": "full_sax_ed_es_volume_per_patient",
        "backbone": backbone,
        "formulation": formulation,
        "initialization": MNMS2_3D_INITIALIZATION,
        "seed": seed,
        "best_val_mcc": best_mcc,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "feature_layer_candidates": feature_layers,
        "checkpoint": checkpoint,
        "training_seconds": training_seconds,
        "test_inference_seconds": test_seconds,
        "test_inference_ms_per_patient": 1000.0 * test_seconds / max(len(test_y), 1),
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "test": {name: test_metrics[name] for name in MNMS2_3D_METRIC_NAMES},
    }
    save_json(summary_path, summary)
    if failure_path.exists():
        failure_path.unlink()
    print(
        f"[test] {backbone}/{formulation}/seed_{seed}: "
        f"MCC={test_metrics['mcc']:.4f} AUROC={test_metrics['roc_auc']:.4f} "
        f"macro-F1={test_metrics['macro_f1']:.4f}"
    )
    del model
    cleanup_cuda()
    return summary


def train_cnn(
    args: argparse.Namespace,
    spec: TaskSpec,
    backbone: str,
    formulation: str,
    initialization: str,
    seed: int,
) -> dict[str, Any]:
    if spec.dimensionality != 2:
        raise ValueError(
            f"{spec.key} is 3-D. Use CineMA's official ResNet baseline, not a 2-D torchvision model."
        )
    output_group = "cnn_smoke" if args.smoke else "cnn_extensions"
    run_dir = (
        checkpoint_root(args)
        / output_group
        / spec.key
        / backbone
        / formulation
        / initialization
        / f"seed_{seed}"
    )
    summary_path = run_dir / "summary.json"

    data_root = processed_dir(args, spec.dataset)
    splits = split_metadata(spec, data_root)
    config = local_config(spec, data_root, seed)
    apply_overrides(args, config)
    previous = prepare_run_directory(
        args,
        run_dir,
        config,
        backbone=backbone,
        formulation=formulation,
        seed=seed,
        initialization=initialization,
    )
    if previous is not None:
        return previous
    train_loader, val_loader, test_loader = make_loaders(
        spec, config, data_root, splits, seed
    )
    amp_dtype, device = amp_dtype_and_device()
    model = Reproduction2DClassifier(
        backbone, formulation, initialization, len(spec.classes)
    ).to(device)
    checkpoint = run_dir / "best_val_mcc.pt"

    best_mcc, best_epoch = fit_classifier(
        model,
        train_loader,
        val_loader,
        config,
        run_dir,
        device,
        amp_dtype,
        args.amp,
        {
            "task": spec.key,
            "classes": list(spec.classes),
            "backbone": backbone,
            "formulation": formulation,
            "initialization": initialization,
            "seed": seed,
            "cinema_commit": cinema_git_commit(),
        },
        partial(
            reproduction_evaluate_cnn,
            spec=spec,
            device=device,
            amp_dtype=amp_dtype,
            amp=args.amp,
        ),
        partial(save_predictions, classes=spec.classes),
        view=spec.view,
    )

    pids, y, probs, test_metrics = reproduction_evaluate_cnn(
        model, test_loader, spec, device, amp_dtype, amp=args.amp
    )
    save_predictions(run_dir / "test", pids, y, probs, spec.classes, test_metrics)
    summary = {
        "family": "cnn_extension",
        "task": spec.key,
        "backbone": backbone,
        "formulation": formulation,
        "initialization": initialization,
        "seed": seed,
        "best_val_mcc": best_mcc,
        "best_epoch": best_epoch,
        "checkpoint": checkpoint,
        "test": {
            k: test_metrics[k]
            for k in (
                "accuracy",
                "f1",
                "macro_f1",
                "balanced_accuracy",
                "mcc",
                "roc_auc",
            )
        },
    }
    save_json(summary_path, summary)
    del model
    cleanup_cuda()
    return summary


def acdc_2d_train_one_run(
    args, data_root, splits, backbone, formulation, initialization, seed
):
    return train_mid_sax_2d(
        args,
        data_root,
        splits,
        backbone,
        formulation,
        initialization,
        seed,
        dataset="acdc",
    )


def mnms2_sax_2d_train_one_run(
    args, data_root, splits, backbone, formulation, initialization, seed
):
    return train_mid_sax_2d(
        args,
        data_root,
        splits,
        backbone,
        formulation,
        initialization,
        seed,
        dataset="mnms2",
    )
