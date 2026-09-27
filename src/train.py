"""Training loops and experiment dispatch."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from cinema.classification.train import classification_forward
from cinema.optim import adjust_learning_rate, get_n_accum_steps
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from .dataset import (
    acdc_2d_load_official_splits,
    acdc_2d_make_loaders,
    acdc_2d_resolve_processed_dir,
    acdc_2d_save_split_audit,
    acdc_2d_validate_inputs,
    acdc_3d_local_config,
    acdc_3d_make_dataset,
    acdc_3d_make_loaders,
    acdc_3d_preprocess,
    acdc_3d_processed_dir,
    acdc_3d_split_metadata,
    acdc_3d_validate_inputs,
    load_acdc_config,
    load_mnms2_config,
    local_config,
    make_loaders,
    mnms2_3d_local_config,
    mnms2_3d_make_loaders,
    mnms2_3d_preprocess,
    mnms2_3d_processed_dir,
    mnms2_3d_split_metadata,
    mnms2_3d_validate_inputs,
    mnms2_sax_2d_load_official_splits,
    mnms2_sax_2d_make_loaders,
    mnms2_sax_2d_resolve_processed_dir,
    mnms2_sax_2d_save_split_audit,
    mnms2_sax_2d_validate_inputs,
    preprocess_acdc,
    preprocess_dataset,
    preprocess_mnms2,
    processed_dir,
    save_split_audit,
    split_metadata,
)
from .evaluate import (
    acdc_2d_aggregate_results,
    acdc_2d_evaluate,
    acdc_2d_save_predictions,
    acdc_3d_aggregate_results,
    acdc_3d_evaluate,
    acdc_3d_save_predictions,
    acdc_distillation_aggregate_results,
    aggregate_architecture_results,
    aggregate_cnn_results,
    aggregate_official_results,
    architecture_bank_calibrate_task,
    architecture_bank_evaluate_cnn,
    calibration_is_valid,
    mnms2_3d_aggregate_results,
    mnms2_3d_evaluate,
    mnms2_3d_save_predictions,
    mnms2_sax_2d_aggregate_results,
    mnms2_sax_2d_evaluate,
    mnms2_sax_2d_save_predictions,
    reproduction_calibrate_task,
    reproduction_evaluate_cnn,
    save_predictions,
)
from .models import (
    ACDC2DClassifier,
    ACDC3DClassifier,
    ArchitectureBank2DClassifier,
    ArchitectureBank3DClassifier,
    MnMs2SAX3DClassifier,
    MnMs2SAXMid2DClassifier,
    Reproduction2DClassifier,
    acdc_3d_feature_layer_candidates,
    architecture_bank_feature_layer_candidates,
    feature_layer_candidates,
    load_teacher,
    mnms2_3d_feature_layer_candidates,
)
from .protocol import TASKS, TaskSpec, protocols, resolve_tasks
from .utils import (
    acdc_3d_run_dir_for,
    acdc_3d_task_root,
    acdc_distillation_run_dir_for,
    acdc_distillation_task_root,
    amp_dtype_and_device,
    apply_architecture_overrides,
    apply_overrides,
    apply_training_overrides,
    cinema_git_commit,
    cleanup_cuda,
    grad_scaler,
    mnms2_3d_run_dir_for,
    mnms2_3d_task_root,
    output_file,
    prepare_run,
    save_json,
    save_json_arrays,
    teacher_paths,
)

# Acdc 2D

ACDC_2D_CLASSES = protocols["acdc"]["classes"]


ACDC_2D_TASK_KEY = protocols["acdc_2d"]["task_key"]


def acdc_2d_train_one_run(
    args: argparse.Namespace,
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    backbone: str,
    formulation: str,
    initialization: str,
    seed: int,
) -> dict[str, Any]:
    output_group = "architecture_smoke" if args.smoke else "architecture_bank"
    run_dir = (
        args.output_dir
        / "checkpoints"
        / output_group
        / "2d"
        / ACDC_2D_TASK_KEY
        / backbone
        / formulation
        / initialization
        / f"seed_{seed}"
    )
    summary_path = run_dir / "summary.json"
    if args.resume and summary_path.is_file():
        print(f"[resume] {backbone}/{formulation}/{initialization}/seed_{seed}")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)

    config = load_acdc_config(data_root, seed)
    apply_overrides(args, config)
    train_loader, val_loader, test_loader = acdc_2d_make_loaders(
        data_root, splits, config, seed
    )
    amp_dtype, device = amp_dtype_and_device()
    model = ACDC2DClassifier(backbone, formulation, initialization).to(device)
    feature_layers = feature_layer_candidates(model, backbone)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    architecture = {
        "family": "acdc_2d_architecture_bank",
        "task": ACDC_2D_TASK_KEY,
        "dataset": "acdc",
        "view": "sax_mid",
        "dimensionality": 2,
        "input_policy": "patient_mid_sax_ed_es",
        "backbone": backbone,
        "formulation": formulation,
        "initialization": initialization,
        "classes": list(ACDC_2D_CLASSES),
        "feature_dim": model.feature_dim,
        "feature_layer_candidates": feature_layers,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "cinema_commit": cinema_git_commit(),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    save_json_arrays(run_dir / "architecture.json", architecture)
    print(
        f"[architecture] acdc/2d/{backbone}/{formulation}/{initialization}/seed{seed} "
        f"parameters={parameter_count:,} feature_layers={feature_layers}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.train.lr),
        betas=tuple(config.train.betas),
        weight_decay=float(config.train.weight_decay),
    )
    scaler = grad_scaler(args.amp)
    accumulation_steps = get_n_accum_steps(
        batch_size=int(config.train.batch_size),
        batch_size_per_device=int(config.train.batch_size_per_device),
        world_size=1,
    )
    best_mcc = -math.inf
    best_epoch = -1
    bad_evaluations = 0
    checkpoint = run_dir / "best_val_mcc.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(int(config.train.n_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        n_steps = len(train_loader)
        progress = tqdm(train_loader, desc=f"acdc/{backbone}/e{epoch + 1}", leave=False)
        for step, batch in enumerate(progress):
            learning_rate = adjust_learning_rate(
                optimizer=optimizer,
                step=step / max(len(train_loader), 1) + epoch,
                warmup_steps=int(config.train.n_warmup_epochs),
                max_n_steps=int(config.train.n_epochs),
                lr=float(config.train.lr),
                min_lr=float(config.train.min_lr),
            )
            image = batch["sax_image"].to(device, non_blocking=True)
            target = batch["label"].long().to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=args.amp and device.type == "cuda",
            ):
                logits = model(image)
                raw_loss = F.cross_entropy(
                    logits,
                    target,
                    label_smoothing=float(config.train.label_smoothing),
                )
                group_start = (step // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, n_steps - group_start)
                loss = raw_loss / group_size
            scaler.scale(loss).backward()
            update = (step + 1) % accumulation_steps == 0 or (step + 1) == n_steps
            if update:
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

        final_epoch = epoch + 1 == int(config.train.n_epochs)
        if (epoch + 1) % int(config.train.eval_interval) != 0 and not final_epoch:
            continue
        val_pids, val_y, val_probs, val_metrics = acdc_2d_evaluate(
            model, val_loader, device, amp_dtype
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(total_seen, 1),
                "train_accuracy": total_correct / max(total_seen, 1),
                "lr": learning_rate,
                **{
                    f"val_{key}": val_metrics[key]
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
        )
        pd.DataFrame(history).to_csv(output_file(run_dir / "history.csv"), index=False)
        print(
            f"[architecture] acdc/2d/{backbone}/{formulation}/{initialization}/seed{seed} "
            f"epoch={epoch + 1} val_MCC={val_metrics['mcc']:.4f} "
            f"val_AUROC={val_metrics['roc_auc']:.4f} val_F1={val_metrics['f1']:.4f}"
        )
        if val_metrics["mcc"] > best_mcc + float(config.train.early_stopping.min_delta):
            best_mcc = float(val_metrics["mcc"])
            best_epoch = epoch + 1
            bad_evaluations = 0
            torch.save(
                {
                    "model": model.state_dict(),
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
                    "epoch": best_epoch,
                },
                checkpoint,
            )
            acdc_2d_save_predictions(
                run_dir / "best_validation", val_pids, val_y, val_probs, val_metrics
            )
        else:
            bad_evaluations += 1
        if bad_evaluations >= int(config.train.early_stopping.patience):
            break

    if not checkpoint.is_file():
        raise RuntimeError(f"No checkpoint was created for {run_dir}")
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    test_pids, test_y, test_probs, test_metrics = acdc_2d_evaluate(
        model, test_loader, device, amp_dtype
    )
    acdc_2d_save_predictions(
        run_dir / "test", test_pids, test_y, test_probs, test_metrics
    )
    summary = {
        "family": "acdc_2d_architecture_bank",
        "task": ACDC_2D_TASK_KEY,
        "dataset": "acdc",
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
    del model, optimizer, scaler
    cleanup_cuda()
    return summary


def acdc_2d_run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages)

    print("=" * 100)
    print("ACDC patient-level 2-D architecture-bank benchmark")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CineMA commit: {cinema_git_commit()}")
    print(f"Task: {ACDC_2D_TASK_KEY}")
    print(f"Classes: {list(ACDC_2D_CLASSES)}")
    print(f"Models: {list(args.models_2d)}")
    print(f"Formulations: {list(args.formulations_2d)}")
    print(f"Initializations: {list(args.initializations_2d)}")
    print(f"Seeds: {list(args.seeds)}")
    print(f"Stages: {list(args.stages)}")
    print("=" * 100)

    if "preprocess" in stages:
        preprocess_acdc(args)
    data_root = acdc_2d_resolve_processed_dir(args)
    splits = acdc_2d_load_official_splits(data_root)
    acdc_2d_save_split_audit(args.output_dir, splits)
    config = load_acdc_config(data_root, seed=0)
    if stages & {"validate", "train"}:
        acdc_2d_validate_inputs(args.output_dir, data_root, splits, config)

    if "train" in stages:
        for backbone in args.models_2d:
            for formulation in args.formulations_2d:
                for initialization in args.initializations_2d:
                    for seed in args.seeds:
                        acdc_2d_train_one_run(
                            args,
                            data_root,
                            splits,
                            backbone,
                            formulation,
                            initialization,
                            seed,
                        )
        if not args.smoke:
            acdc_2d_aggregate_results(args)
    print(f"Done. Results: {args.output_dir}")


# Acdc 3D

ACDC_3D_CINEMA_RANDINIT_WEIGHT_DECAY = protocols["acdc_3d"][
    "cinema_randinit_weight_decay"
]


ACDC_3D_CLASSES = protocols["acdc"]["classes"]


ACDC_3D_DATASET = protocols["acdc_3d"]["dataset"]


ACDC_3D_INITIALIZATION = protocols["shared"]["initialization"]


ACDC_3D_METRIC_NAMES = protocols["shared"]["metric_names"]


ACDC_3D_PROTOCOL_VERSION = protocols["acdc_3d"]["protocol_version"]


ACDC_3D_SHARED_ABLATION_MODELS = protocols["shared"]["shared_ablation_models"]


ACDC_3D_TASK_KEY = protocols["acdc_3d"]["task_key"]


ACDC_3D_VIEW = protocols["shared"]["view"]


def acdc_3d_train_one(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
) -> dict[str, Any]:
    run_dir = acdc_3d_run_dir_for(args, backbone, formulation, seed)
    summary_path = run_dir / "summary.json"
    failure_path = run_dir / "failure.json"
    if args.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("protocol_version") == ACDC_3D_PROTOCOL_VERSION:
            print(
                f"[resume] {ACDC_3D_TASK_KEY}/{backbone}/{formulation}/{ACDC_3D_INITIALIZATION}/seed_{seed}"
            )
            return previous
        print(f"[resume] retraining stale protocol: {summary_path}")
    if args.resume and failure_path.is_file() and not args.retry_failures:
        print(f"[resume] skipping recorded failure: {failure_path}")
        return {}
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    data_root = acdc_3d_processed_dir(args)
    splits = acdc_3d_split_metadata(data_root)
    config = acdc_3d_local_config(data_root, seed)
    apply_training_overrides(args, config)
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.train.lr),
        betas=tuple(config.train.betas),
        weight_decay=optimizer_weight_decay,
    )
    scaler = grad_scaler(args.amp)
    accumulation_steps = get_n_accum_steps(
        batch_size=int(config.train.batch_size),
        batch_size_per_device=int(config.train.batch_size_per_device),
        world_size=1,
    )
    best_mcc = -math.inf
    best_epoch = -1
    bad_evaluations = 0
    checkpoint = run_dir / "best_val_mcc.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(int(config.train.n_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        number_of_steps = len(train_loader)
        current_lr = float(config.train.lr)
        for step, batch in enumerate(
            tqdm(
                train_loader,
                desc=f"{ACDC_3D_TASK_KEY}/{backbone}/e{epoch + 1}",
                leave=False,
            )
        ):
            current_lr = adjust_learning_rate(
                optimizer=optimizer,
                step=step / max(len(train_loader), 1) + epoch,
                warmup_steps=int(config.train.n_warmup_epochs),
                max_n_steps=int(config.train.n_epochs),
                lr=float(config.train.lr),
                min_lr=float(config.train.min_lr),
            )
            image = batch["sax_image"].to(device, non_blocking=True)
            target = batch["label"].long().to(device, non_blocking=True)
            enabled = args.amp and device.type == "cuda"
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=enabled
            ):
                logits = model(image)
                raw_loss = F.cross_entropy(
                    logits,
                    target,
                    label_smoothing=float(config.train.label_smoothing),
                )
                group_start = (step // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, number_of_steps - group_start)
                loss = raw_loss / max(group_size, 1)
            scaler.scale(loss).backward()
            update = (step + 1) % accumulation_steps == 0 or (
                step + 1
            ) == number_of_steps
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config.train.clip_grad)
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(raw_loss.item()) * target.size(0)
            total_correct += int((logits.argmax(dim=1) == target).sum().item())
            total_seen += int(target.size(0))

        final_epoch = epoch + 1 == int(config.train.n_epochs)
        if (epoch + 1) % int(config.train.eval_interval) != 0 and not final_epoch:
            continue

        val_pids, val_y, val_probabilities, val_metrics, val_seconds = acdc_3d_evaluate(
            model, val_loader, patch_size, device, amp_dtype, args.amp
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(total_seen, 1),
            "train_accuracy": total_correct / max(total_seen, 1),
            "lr": current_lr,
            "validation_seconds": val_seconds,
            **{f"val_{name}": val_metrics[name] for name in ACDC_3D_METRIC_NAMES},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_file(run_dir / "history.csv"), index=False)
        print(
            f"[eval] {ACDC_3D_TASK_KEY}/{backbone}/{formulation}/seed_{seed} epoch={epoch + 1} "
            f"MCC={val_metrics['mcc']:.4f} AUROC={val_metrics['roc_auc']:.4f} "
            f"macro-F1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["mcc"] > best_mcc + float(config.train.early_stopping.min_delta):
            best_mcc = float(val_metrics["mcc"])
            best_epoch = epoch + 1
            bad_evaluations = 0
            torch.save(
                {
                    "model": model.state_dict(),
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
                    "epoch": best_epoch,
                    "cinema_commit": cinema_git_commit(),
                },
                checkpoint,
            )
            acdc_3d_save_predictions(
                run_dir / "best_validation",
                val_pids,
                val_y,
                val_probabilities,
                val_metrics,
            )
        else:
            bad_evaluations += 1
        if bad_evaluations >= int(config.train.early_stopping.patience):
            print(
                f"[early-stop] {backbone}/{formulation}/seed_{seed} at epoch {epoch + 1}"
            )
            break

    if not checkpoint.is_file():
        raise RuntimeError(f"No validation checkpoint was written: {checkpoint}")
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
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
    del model, optimizer, scaler
    cleanup_cuda()
    return summary


def acdc_3d_save_run_manifest(args: argparse.Namespace) -> None:
    save_json(
        acdc_3d_task_root(args) / "run_manifest.json",
        {
            "protocol_version": ACDC_3D_PROTOCOL_VERSION,
            "dataset": ACDC_3D_DATASET,
            "task": ACDC_3D_TASK_KEY,
            "view": ACDC_3D_VIEW,
            "dimensionality": 3,
            "input_policy": "full_sax_ed_es_volume_per_patient",
            "models": list(args.models),
            "formulations": list(args.formulations),
            "shared_ablation_models": list(ACDC_3D_SHARED_ABLATION_MODELS),
            "initialization": ACDC_3D_INITIALIZATION,
            "optimizer_weight_decay": (
                ACDC_3D_CINEMA_RANDINIT_WEIGHT_DECAY
                if args.weight_decay is None
                else args.weight_decay
            ),
            "seeds": list(args.seeds),
            "stages": list(args.stages),
            "smoke": args.smoke,
            "resume": args.resume,
            "amp": args.amp,
            "deterministic": args.deterministic,
            "python": sys.version,
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cinema_commit": cinema_git_commit(),
        },
    )


def acdc_3d_record_failure(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
    error: BaseException,
) -> None:
    directory = acdc_3d_run_dir_for(args, backbone, formulation, seed)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(
        directory / "failure.json",
        {
            "protocol_version": ACDC_3D_PROTOCOL_VERSION,
            "dataset": ACDC_3D_DATASET,
            "task": ACDC_3D_TASK_KEY,
            "backbone": backbone,
            "formulation": formulation,
            "initialization": ACDC_3D_INITIALIZATION,
            "seed": seed,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )


def acdc_3d_run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Seeds must be unique: {args.seeds}")

    print("=" * 100)
    print("ACDC FULL-VOLUME 3-D CNN ARCHITECTURE BANK")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CineMA commit: {cinema_git_commit()}")
    print(f"Processed data: {acdc_3d_processed_dir(args)}")
    print(f"Models: {list(args.models)}")
    print(f"Formulations: {list(args.formulations)}")
    print(f"Seeds: {list(args.seeds)}")
    print("=" * 100)

    stages = set(args.stages)
    if "preprocess" in stages:
        acdc_3d_preprocess(args)
    if stages & {"validate", "train"}:
        acdc_3d_validate_inputs(args)
    if "train" in stages:
        acdc_3d_save_run_manifest(args)
        for backbone in args.models:
            for formulation in args.formulations:
                for seed in args.seeds:
                    try:
                        acdc_3d_train_one(args, backbone, formulation, seed)
                    except Exception as error:
                        acdc_3d_record_failure(args, backbone, formulation, seed, error)
                        cleanup_cuda()
                        print(
                            f"[failure] {backbone}/{formulation}/seed_{seed}: {type(error).__name__}: {error}",
                            file=sys.stderr,
                        )
                        if not args.continue_on_error:
                            raise
    if "aggregate" in stages or ("train" in stages and not args.smoke):
        acdc_3d_aggregate_results(args)
    print(f"Done. Results: {acdc_3d_task_root(args)}")


# Acdc Distillation

ACDC_DISTILLATION_CINEMA_RANDINIT_WEIGHT_DECAY = protocols["acdc_3d"][
    "cinema_randinit_weight_decay"
]


ACDC_DISTILLATION_CLASSES = protocols["acdc"]["classes"]


ACDC_DISTILLATION_DATASET = protocols["acdc_3d"]["dataset"]


ACDC_DISTILLATION_INITIALIZATION = protocols["acdc_distillation"]["initialization"]


ACDC_DISTILLATION_METRIC_NAMES = protocols["shared"]["metric_names"]


ACDC_DISTILLATION_PROTOCOL_VERSION = protocols["acdc_distillation"]["protocol_version"]


ACDC_DISTILLATION_TASK_KEY = protocols["acdc_3d"]["task_key"]


ACDC_DISTILLATION_VIEW = protocols["shared"]["view"]


def distillation_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    supervised_weight: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    supervised = F.cross_entropy(
        student_logits,
        labels,
        label_smoothing=label_smoothing,
    )
    distilled = (
        F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        )
        * temperature**2
    )
    total = supervised_weight * supervised + (1.0 - supervised_weight) * distilled
    return total, supervised, distilled


def teacher_cache_metadata(
    checkpoint: Path,
    config: Path,
    pids: list[str],
) -> dict[str, Any]:
    checkpoint_stat = checkpoint.stat()
    config_stat = config.stat()
    return {
        "protocol_version": ACDC_DISTILLATION_PROTOCOL_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "config": str(config),
        "config_size": config_stat.st_size,
        "config_mtime_ns": config_stat.st_mtime_ns,
        "pids": pids,
        "classes": list(ACDC_DISTILLATION_CLASSES),
    }


def get_teacher_targets(
    args: argparse.Namespace,
    config: Any,
    data_root: Path,
    train_frame: pd.DataFrame,
    student_seed: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], int, Path]:
    checkpoint, teacher_config, teacher_seed = teacher_paths(args, student_seed)
    pids = train_frame["pid"].astype(str).tolist()
    expected_metadata = teacher_cache_metadata(checkpoint, teacher_config, pids)
    cache_dir = (
        args.output_dir
        / "teacher_targets"
        / ACDC_DISTILLATION_TASK_KEY
        / f"seed_{teacher_seed}"
    )
    metadata_path = cache_dir / "metadata.json"
    logits_path = cache_dir / "logits.npz"

    if metadata_path.is_file() and logits_path.is_file():
        observed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if observed_metadata == expected_metadata:
            with np.load(logits_path) as cached:
                logits = cached["logits"]
            if logits.shape == (len(pids), len(ACDC_DISTILLATION_CLASSES)):
                print(f"[teacher-cache] reusing {logits_path}")
                return (
                    {
                        pid: torch.from_numpy(row.copy())
                        for pid, row in zip(pids, logits, strict=True)
                    },
                    teacher_seed,
                    checkpoint,
                )

    print(f"[teacher-cache] computing seed {teacher_seed} targets from {checkpoint}")
    teacher = load_teacher(checkpoint, teacher_config, device)
    dataset = acdc_3d_make_dataset(config, data_root, train_frame, "train", False)
    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=1,
        drop_last=False,
        pin_memory=device.type == "cuda",
        num_workers=int(config.train.n_workers),
    )
    patch_size = tuple(int(value) for value in config.data.sax.patch_size)
    rows: list[np.ndarray] = []
    for batch in tqdm(loader, desc=f"teacher seed {teacher_seed}", leave=False):
        image = batch["sax_image"].to(device, non_blocking=True)
        logits = classification_forward(
            teacher,
            {ACDC_DISTILLATION_VIEW: image},
            {ACDC_DISTILLATION_VIEW: patch_size},
            amp_dtype,
        )
        rows.append(logits.float().cpu().numpy()[0])
    logits = np.asarray(rows, dtype=np.float32)
    if (
        logits.shape != (len(pids), len(ACDC_DISTILLATION_CLASSES))
        or not np.isfinite(logits).all()
    ):
        raise RuntimeError(
            f"Invalid cached teacher logits shape/values: {logits.shape}"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_dir / "logits.tmp.npz"
    np.savez_compressed(temporary_path, logits=logits)
    temporary_path.replace(logits_path)
    save_json(metadata_path, expected_metadata)
    del teacher, loader, dataset
    cleanup_cuda()
    return (
        {
            pid: torch.from_numpy(row.copy())
            for pid, row in zip(pids, logits, strict=True)
        },
        teacher_seed,
        checkpoint,
    )


def acdc_distillation_train_one(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
) -> dict[str, Any]:
    run_dir = acdc_distillation_run_dir_for(args, backbone, formulation, seed)
    summary_path = run_dir / "summary.json"
    if args.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("protocol_version") == ACDC_DISTILLATION_PROTOCOL_VERSION:
            print(
                f"[resume] {backbone}/{formulation}/{ACDC_DISTILLATION_INITIALIZATION}/seed_{seed}"
            )
            return previous
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    data_root = acdc_3d_processed_dir(args)
    splits = acdc_3d_split_metadata(data_root)
    config = acdc_3d_local_config(data_root, seed)
    apply_training_overrides(args, config)
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
    teacher_targets, teacher_seed, teacher_checkpoint = get_teacher_targets(
        args,
        config,
        data_root,
        splits["train"],
        seed,
        device,
        amp_dtype,
    )

    student = ACDC3DClassifier(
        backbone, formulation, len(ACDC_DISTILLATION_CLASSES)
    ).to(device)
    optimizer_weight_decay = (
        ACDC_DISTILLATION_CINEMA_RANDINIT_WEIGHT_DECAY
        if args.weight_decay is None
        else args.weight_decay
    )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(config.train.lr),
        betas=tuple(config.train.betas),
        weight_decay=optimizer_weight_decay,
    )
    scaler = grad_scaler(args.amp)
    accumulation_steps = get_n_accum_steps(
        batch_size=int(config.train.batch_size),
        batch_size_per_device=int(config.train.batch_size_per_device),
        world_size=1,
    )
    patch_size = tuple(int(value) for value in config.data.sax.patch_size)
    checkpoint = run_dir / "best_val_mcc.pt"
    best_mcc = -math.inf
    best_epoch = -1
    bad_evaluations = 0
    history: list[dict[str, Any]] = []

    save_json(
        run_dir / "architecture.json",
        {
            "protocol_version": ACDC_DISTILLATION_PROTOCOL_VERSION,
            "family": "acdc_3d_architecture_bank_distillation",
            "dataset": ACDC_DISTILLATION_DATASET,
            "task": ACDC_DISTILLATION_TASK_KEY,
            "backbone": backbone,
            "formulation": formulation,
            "initialization": ACDC_DISTILLATION_INITIALIZATION,
            "teacher_seed": teacher_seed,
            "teacher_checkpoint": teacher_checkpoint,
            "temperature": args.temperature,
            "supervised_weight": args.supervised_weight,
            "optimizer_weight_decay": optimizer_weight_decay,
            "parameter_count": sum(
                parameter.numel() for parameter in student.parameters()
            ),
            "config": OmegaConf.to_container(config, resolve=True),
        },
    )

    for epoch in range(int(config.train.n_epochs)):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {
            "loss": 0.0,
            "supervised": 0.0,
            "distilled": 0.0,
            "correct": 0,
            "seen": 0,
        }
        number_of_steps = len(train_loader)
        current_lr = float(config.train.lr)
        for step, batch in enumerate(
            tqdm(train_loader, desc=f"KD {backbone}/e{epoch + 1}", leave=False)
        ):
            current_lr = adjust_learning_rate(
                optimizer=optimizer,
                step=step / max(number_of_steps, 1) + epoch,
                warmup_steps=int(config.train.n_warmup_epochs),
                max_n_steps=int(config.train.n_epochs),
                lr=float(config.train.lr),
                min_lr=float(config.train.min_lr),
            )
            image = batch["sax_image"].to(device, non_blocking=True)
            labels = batch["label"].long().to(device, non_blocking=True)
            soft_targets = torch.stack(
                [teacher_targets[str(pid)] for pid in batch["pid"]]
            ).to(device, non_blocking=True)
            enabled = args.amp and device.type == "cuda"
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=enabled
            ):
                logits = student(image)
                raw_loss, supervised_loss, distilled_loss = distillation_loss(
                    logits,
                    labels,
                    soft_targets,
                    args.temperature,
                    args.supervised_weight,
                    float(config.train.label_smoothing),
                )
                group_start = (step // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, number_of_steps - group_start)
                loss = raw_loss / max(group_size, 1)
            scaler.scale(loss).backward()
            update = (step + 1) % accumulation_steps == 0 or step + 1 == number_of_steps
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    student.parameters(), float(config.train.clip_grad)
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            batch_size = labels.size(0)
            totals["loss"] += float(raw_loss.item()) * batch_size
            totals["supervised"] += float(supervised_loss.item()) * batch_size
            totals["distilled"] += float(distilled_loss.item()) * batch_size
            totals["correct"] += int((logits.argmax(dim=1) == labels).sum().item())
            totals["seen"] += batch_size

        final_epoch = epoch + 1 == int(config.train.n_epochs)
        if (epoch + 1) % int(config.train.eval_interval) != 0 and not final_epoch:
            continue
        val_pids, val_y, val_probabilities, val_metrics, val_seconds = acdc_3d_evaluate(
            student, val_loader, patch_size, device, amp_dtype, args.amp
        )
        seen = max(int(totals["seen"]), 1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": totals["loss"] / seen,
                "train_supervised_loss": totals["supervised"] / seen,
                "train_distillation_loss": totals["distilled"] / seen,
                "train_accuracy": totals["correct"] / seen,
                "lr": current_lr,
                "validation_seconds": val_seconds,
                **{
                    f"val_{name}": val_metrics[name]
                    for name in ACDC_DISTILLATION_METRIC_NAMES
                },
            }
        )
        pd.DataFrame(history).to_csv(output_file(run_dir / "history.csv"), index=False)
        print(
            f"[eval] {backbone}/{formulation}/seed_{seed} epoch={epoch + 1} "
            f"MCC={val_metrics['mcc']:.4f} AUROC={val_metrics['roc_auc']:.4f}"
        )
        if val_metrics["mcc"] > best_mcc + float(config.train.early_stopping.min_delta):
            best_mcc = float(val_metrics["mcc"])
            best_epoch = epoch + 1
            bad_evaluations = 0
            torch.save(
                {
                    "model": student.state_dict(),
                    "protocol_version": ACDC_DISTILLATION_PROTOCOL_VERSION,
                    "backbone": backbone,
                    "formulation": formulation,
                    "teacher_seed": teacher_seed,
                    "teacher_checkpoint": str(teacher_checkpoint),
                    "temperature": args.temperature,
                    "supervised_weight": args.supervised_weight,
                    "seed": seed,
                    "epoch": best_epoch,
                },
                checkpoint,
            )
            acdc_3d_save_predictions(
                run_dir / "best_validation",
                val_pids,
                val_y,
                val_probabilities,
                val_metrics,
            )
        else:
            bad_evaluations += 1
        if bad_evaluations >= int(config.train.early_stopping.patience):
            break

    if not checkpoint.is_file():
        raise RuntimeError(f"No validation checkpoint was written: {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    student.load_state_dict(state["model"])
    test_pids, test_y, test_probabilities, test_metrics, test_seconds = (
        acdc_3d_evaluate(student, test_loader, patch_size, device, amp_dtype, args.amp)
    )
    acdc_3d_save_predictions(
        run_dir / "test", test_pids, test_y, test_probabilities, test_metrics
    )

    training_seconds = time.perf_counter() - started
    summary = {
        "protocol_version": ACDC_DISTILLATION_PROTOCOL_VERSION,
        "family": "acdc_3d_architecture_bank_distillation",
        "dataset": ACDC_DISTILLATION_DATASET,
        "task": ACDC_DISTILLATION_TASK_KEY,
        "backbone": backbone,
        "formulation": formulation,
        "initialization": ACDC_DISTILLATION_INITIALIZATION,
        "seed": seed,
        "teacher_seed": teacher_seed,
        "teacher_checkpoint": teacher_checkpoint,
        "temperature": args.temperature,
        "supervised_weight": args.supervised_weight,
        "best_val_mcc": best_mcc,
        "best_epoch": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in student.parameters()),
        "checkpoint": checkpoint,
        "training_seconds": training_seconds,
        "test_inference_seconds": test_seconds,
        "peak_gpu_memory_gb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "test": {name: test_metrics[name] for name in ACDC_DISTILLATION_METRIC_NAMES},
    }
    save_json(summary_path, summary)
    print(
        f"[test] {backbone}/{formulation}/seed_{seed}: "
        f"MCC={test_metrics['mcc']:.4f} AUROC={test_metrics['roc_auc']:.4f}"
    )
    del student, optimizer, scaler, teacher_targets
    cleanup_cuda()
    return summary


def acdc_distillation_run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.temperature <= 0:
        raise ValueError("train.temperature must be positive")
    if not 0.0 <= args.supervised_weight <= 1.0:
        raise ValueError("train.supervised_weight must be between 0 and 1")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Seeds must be unique: {args.seeds}")

    stages = set(args.stages)
    if "preprocess" in stages:
        acdc_3d_preprocess(args)
    if stages & {"validate", "train"}:
        acdc_3d_validate_inputs(args)
    if "train" in stages:
        for backbone in args.models:
            for formulation in args.formulations:
                for seed in args.seeds:
                    try:
                        acdc_distillation_train_one(args, backbone, formulation, seed)
                    except Exception as error:
                        failure = (
                            acdc_distillation_run_dir_for(
                                args, backbone, formulation, seed
                            )
                            / "failure.json"
                        )
                        save_json(
                            failure,
                            {
                                "protocol_version": ACDC_DISTILLATION_PROTOCOL_VERSION,
                                "backbone": backbone,
                                "formulation": formulation,
                                "seed": seed,
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        cleanup_cuda()
                        print(
                            f"[failure] {backbone}/{formulation}/seed_{seed}: {error}"
                        )
                        if not args.continue_on_error:
                            raise
    if "aggregate" in stages or ("train" in stages and not args.smoke):
        acdc_distillation_aggregate_results(args)
    gc.collect()
    print(f"Done. Distilled results: {acdc_distillation_task_root(args)}")


# Shared


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
        raise RuntimeError(
            f"Expected one retained checkpoint in {run_dir / 'ckpt'}, found {checkpoints}"
        )
    return checkpoints[0]


def run_official_training(
    args: argparse.Namespace, spec: TaskSpec, mode: str, seed: int
) -> None:
    data_root = processed_dir(args, spec.dataset)
    output_group = "official_smoke" if args.smoke else "official_training"
    run_dir = (
        args.output_dir
        / "checkpoints"
        / output_group
        / spec.key
        / mode
        / f"seed_{seed}"
    )
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
        command += [
            hydra_override("model.name", "convvit"),
            hydra_override("model.ckpt_path", None),
        ]
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

    (output_file(run_dir / "command.txt")).write_text(
        " ".join(map(str, command)) + "\n", encoding="utf-8"
    )
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
    # The external CineMA evaluator writes to its own native checkpoint layout.
    metrics_path = (
        checkpoint.parent
        / f"{spec.dataset}_eval_{checkpoint.stem}"
        / "test"
        / "classification_metrics.csv"
    )
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


# Architecture Bank

MODELS_2D = protocols["architecture_bank"]["models_2d"]


MODELS_3D = protocols["architecture_bank"]["models_3d"]


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
        args.output_dir
        / "checkpoints"
        / output_group
        / f"{dimensionality}d"
        / spec.key
        / backbone
        / formulation
        / initialization
        / f"seed_{seed}"
    )
    summary_path = run_dir / "summary.json"
    if args.resume and summary_path.exists():
        print(
            f"[resume] {dimensionality}d/{spec.key}/{backbone}/{formulation}/{initialization}/seed_{seed}"
        )
        return json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)

    data_root = processed_dir(args, spec.dataset)
    splits = split_metadata(spec, data_root)
    config = local_config(spec, data_root, seed)
    apply_architecture_overrides(args, config, dimensionality)
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.train.lr),
        betas=tuple(config.train.betas),
        weight_decay=float(config.train.weight_decay),
    )
    scaler = grad_scaler(args.amp)
    n_accum = get_n_accum_steps(
        batch_size=int(config.train.batch_size),
        batch_size_per_device=int(config.train.batch_size_per_device),
        world_size=1,
    )
    best_mcc = -math.inf
    best_epoch = -1
    bad_evaluations = 0
    checkpoint = run_dir / "best_val_mcc.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(int(config.train.n_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        n_steps = len(train_loader)
        for step, batch in enumerate(
            tqdm(train_loader, desc=f"{spec.key}/{backbone}/e{epoch + 1}", leave=False)
        ):
            lr = adjust_learning_rate(
                optimizer=optimizer,
                step=step / len(train_loader) + epoch,
                warmup_steps=int(config.train.n_warmup_epochs),
                max_n_steps=int(config.train.n_epochs),
                lr=float(config.train.lr),
                min_lr=float(config.train.min_lr),
            )
            image = batch[f"{spec.view}_image"].to(device, non_blocking=True)
            target = batch["label"].long().to(device, non_blocking=True)
            enabled = args.amp and device.type == "cuda"
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=enabled
            ):
                logits = model(image)
                raw_loss = F.cross_entropy(
                    logits, target, label_smoothing=float(config.train.label_smoothing)
                )
                group_start = (step // n_accum) * n_accum
                accumulation_group_size = min(n_accum, n_steps - group_start)
                loss = raw_loss / accumulation_group_size
            scaler.scale(loss).backward()
            update = (step + 1) % n_accum == 0 or (step + 1) == n_steps
            if update:
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

        if (epoch + 1) % int(config.train.eval_interval) != 0 and (epoch + 1) != int(
            config.train.n_epochs
        ):
            continue
        vpids, vy, vprobs, val_metrics = architecture_bank_evaluate_cnn(
            model, val_loader, spec, device, amp_dtype
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(total_seen, 1),
            "train_accuracy": total_correct / max(total_seen, 1),
            "lr": lr,
            **{
                f"val_{k}": val_metrics[k]
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
        history.append(row)
        pd.DataFrame(history).to_csv(output_file(run_dir / "history.csv"), index=False)
        print(
            f"[architecture] {dimensionality}d/{spec.key}/{backbone}/{formulation}/{initialization}/seed{seed} "
            f"epoch={epoch + 1} val_MCC={val_metrics['mcc']:.4f} val_AUROC={val_metrics['roc_auc']:.4f} "
            f"val_F1={val_metrics['f1']:.4f}"
        )
        if val_metrics["mcc"] > best_mcc + float(config.train.early_stopping.min_delta):
            best_mcc = float(val_metrics["mcc"])
            best_epoch = epoch + 1
            bad_evaluations = 0
            torch.save(
                {
                    "model": model.state_dict(),
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
                    "epoch": best_epoch,
                    "cinema_commit": cinema_git_commit(),
                },
                checkpoint,
            )
            save_predictions(
                run_dir / "best_validation",
                vpids,
                vy,
                vprobs,
                spec.classes,
                val_metrics,
            )
        else:
            bad_evaluations += 1
        if bad_evaluations >= int(config.train.early_stopping.patience):
            break

    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    pids, y, probs, test_metrics = architecture_bank_evaluate_cnn(
        model, test_loader, spec, device, amp_dtype
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
    del model, optimizer, scaler
    cleanup_cuda()
    return summary


def architecture_bank_run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages)
    run_2d = bool(stages & {"cnn", "cnn2d"})
    run_3d = "cnn3d" in stages
    selected = resolve_tasks(list(args.tasks))
    validation_specs = (
        list(selected) if stages & {"preprocess", "calibrate", "official"} else []
    )
    if run_2d:
        for key in args.cnn2d_tasks:
            if TASKS[key] not in validation_specs:
                validation_specs.append(TASKS[key])
    if run_3d:
        for key in args.cnn3d_tasks:
            if TASKS[key] not in validation_specs:
                validation_specs.append(TASKS[key])
    datasets = sorted({spec.dataset for spec in validation_specs})

    print("=" * 100)
    print("CineMA validation and 2-D/3-D architecture-bank benchmark")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CineMA commit: {cinema_git_commit()}")
    print(f"Tasks: {[spec.key for spec in selected]}")
    if run_2d:
        print(f"2-D tasks/models: {list(args.cnn2d_tasks)} / {list(args.models_2d)}")
    if run_3d:
        print(f"3-D tasks/models: {list(args.cnn3d_tasks)} / {list(args.models_3d)}")
    print(f"Stages: {list(args.stages)}")
    print("=" * 100)

    if "preprocess" in args.stages:
        for dataset in datasets:
            preprocess_dataset(args, dataset)

    # Validate and persist splits before any evaluation or training.
    for spec in validation_specs:
        splits = split_metadata(spec, processed_dir(args, spec.dataset))
        save_split_audit(args, spec, splits)

    calibration_summaries = []
    if "calibrate" in args.stages:
        for spec in selected:
            calibration_summaries.append(architecture_bank_calibrate_task(args, spec))
        passed = all(item["passed"] for item in calibration_summaries)
        save_json(
            args.output_dir / "checkpoints" / "calibration" / "gate.json",
            {
                "passed": passed,
                "tasks": {x["task"]: x["passed"] for x in calibration_summaries},
            },
        )
        if not passed and not args.allow_calibration_failure:
            raise RuntimeError(
                "Released-checkpoint calibration failed. CNN/official training is intentionally blocked."
            )

    training_requested = "official" in stages or run_2d or run_3d
    if training_requested and not args.smoke and not args.allow_calibration_failure:
        gate_tasks: list[TaskSpec] = []
        if "official" in stages:
            gate_tasks.extend(selected)
        if run_2d:
            gate_tasks.extend(TASKS[key] for key in args.cnn2d_tasks)
        if run_3d:
            gate_tasks.extend(TASKS[key] for key in args.cnn3d_tasks)
        gate_tasks = list(dict.fromkeys(gate_tasks))
        if not calibration_is_valid(args, gate_tasks):
            raise RuntimeError(
                "Run run.stages=[preprocess,calibrate] first. Full training is blocked until released checkpoints reproduce."
            )

    if "official" in stages:
        for spec in selected:
            for mode in args.official_modes:
                for seed in args.seeds:
                    run_official_training(args, spec, mode, seed)
        aggregate_official_results(args)

    if run_2d:
        for task_key in args.cnn2d_tasks:
            spec = TASKS[task_key]
            for backbone in args.models_2d:
                for formulation in args.formulations_2d:
                    for initialization in args.initializations_2d:
                        for seed in args.seeds:
                            train_architecture(
                                args, spec, backbone, formulation, initialization, seed
                            )

    if run_3d:
        for task_key in args.cnn3d_tasks:
            spec = TASKS[task_key]
            for backbone in args.models_3d:
                for formulation in args.formulations_3d:
                    for seed in args.seeds:
                        train_architecture(
                            args, spec, backbone, formulation, "randinit", seed
                        )

    if (run_2d or run_3d) and not args.smoke:
        aggregate_architecture_results(args)

    print(f"Done. Results: {args.output_dir}")


# Mnms2 3D

MNMS2_3D_CLASSES = protocols["mnms2_3d"]["classes"]


MNMS2_3D_DATASET = protocols["mnms2_3d"]["dataset"]


MNMS2_3D_INITIALIZATION = protocols["shared"]["initialization"]


MNMS2_3D_METRIC_NAMES = protocols["shared"]["metric_names"]


MNMS2_3D_SHARED_ABLATION_MODELS = protocols["shared"]["shared_ablation_models"]


MNMS2_3D_TASK_KEY = protocols["mnms2_3d"]["task_key"]


MNMS2_3D_VIEW = protocols["shared"]["view"]


def mnms2_3d_train_one(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
) -> dict[str, Any]:
    run_dir = mnms2_3d_run_dir_for(args, backbone, formulation, seed)
    summary_path = run_dir / "summary.json"
    failure_path = run_dir / "failure.json"
    if args.resume and summary_path.is_file():
        print(
            f"[resume] {MNMS2_3D_TASK_KEY}/{backbone}/{formulation}/{MNMS2_3D_INITIALIZATION}/seed_{seed}"
        )
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if args.resume and failure_path.is_file() and not args.retry_failures:
        print(f"[resume] skipping recorded failure: {failure_path}")
        return {}
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    data_root = mnms2_3d_processed_dir(args)
    splits = mnms2_3d_split_metadata(data_root)
    config = mnms2_3d_local_config(data_root, seed)
    apply_training_overrides(args, config)
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.train.lr),
        betas=tuple(config.train.betas),
        weight_decay=float(config.train.weight_decay),
    )
    scaler = grad_scaler(args.amp)
    accumulation_steps = get_n_accum_steps(
        batch_size=int(config.train.batch_size),
        batch_size_per_device=int(config.train.batch_size_per_device),
        world_size=1,
    )
    best_mcc = -math.inf
    best_epoch = -1
    bad_evaluations = 0
    checkpoint = run_dir / "best_val_mcc.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(int(config.train.n_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        number_of_steps = len(train_loader)
        current_lr = float(config.train.lr)
        for step, batch in enumerate(
            tqdm(
                train_loader,
                desc=f"{MNMS2_3D_TASK_KEY}/{backbone}/e{epoch + 1}",
                leave=False,
            )
        ):
            current_lr = adjust_learning_rate(
                optimizer=optimizer,
                step=step / max(len(train_loader), 1) + epoch,
                warmup_steps=int(config.train.n_warmup_epochs),
                max_n_steps=int(config.train.n_epochs),
                lr=float(config.train.lr),
                min_lr=float(config.train.min_lr),
            )
            image = batch["sax_image"].to(device, non_blocking=True)
            target = batch["label"].long().to(device, non_blocking=True)
            enabled = args.amp and device.type == "cuda"
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=enabled
            ):
                logits = model(image)
                raw_loss = F.cross_entropy(
                    logits,
                    target,
                    label_smoothing=float(config.train.label_smoothing),
                )
                group_start = (step // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, number_of_steps - group_start)
                loss = raw_loss / max(group_size, 1)
            scaler.scale(loss).backward()
            update = (step + 1) % accumulation_steps == 0 or (
                step + 1
            ) == number_of_steps
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config.train.clip_grad)
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(raw_loss.item()) * target.size(0)
            total_correct += int((logits.argmax(dim=1) == target).sum().item())
            total_seen += int(target.size(0))

        final_epoch = epoch + 1 == int(config.train.n_epochs)
        if (epoch + 1) % int(config.train.eval_interval) != 0 and not final_epoch:
            continue

        val_pids, val_y, val_probabilities, val_metrics, val_seconds = (
            mnms2_3d_evaluate(
                model, val_loader, patch_size, device, amp_dtype, args.amp
            )
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(total_seen, 1),
            "train_accuracy": total_correct / max(total_seen, 1),
            "lr": current_lr,
            "validation_seconds": val_seconds,
            **{f"val_{name}": val_metrics[name] for name in MNMS2_3D_METRIC_NAMES},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_file(run_dir / "history.csv"), index=False)
        print(
            f"[eval] {MNMS2_3D_TASK_KEY}/{backbone}/{formulation}/seed_{seed} epoch={epoch + 1} "
            f"MCC={val_metrics['mcc']:.4f} AUROC={val_metrics['roc_auc']:.4f} "
            f"macro-F1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["mcc"] > best_mcc + float(config.train.early_stopping.min_delta):
            best_mcc = float(val_metrics["mcc"])
            best_epoch = epoch + 1
            bad_evaluations = 0
            torch.save(
                {
                    "model": model.state_dict(),
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
                    "epoch": best_epoch,
                    "cinema_commit": cinema_git_commit(),
                },
                checkpoint,
            )
            mnms2_3d_save_predictions(
                run_dir / "best_validation",
                val_pids,
                val_y,
                val_probabilities,
                val_metrics,
            )
        else:
            bad_evaluations += 1
        if bad_evaluations >= int(config.train.early_stopping.patience):
            print(
                f"[early-stop] {backbone}/{formulation}/seed_{seed} at epoch {epoch + 1}"
            )
            break

    if not checkpoint.is_file():
        raise RuntimeError(f"No validation checkpoint was written: {checkpoint}")
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
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
    del model, optimizer, scaler
    cleanup_cuda()
    return summary


def mnms2_3d_save_run_manifest(args: argparse.Namespace) -> None:
    save_json(
        mnms2_3d_task_root(args) / "run_manifest.json",
        {
            "dataset": MNMS2_3D_DATASET,
            "task": MNMS2_3D_TASK_KEY,
            "view": MNMS2_3D_VIEW,
            "dimensionality": 3,
            "input_policy": "full_sax_ed_es_volume_per_patient",
            "models": list(args.models),
            "formulations": list(args.formulations),
            "shared_ablation_models": list(MNMS2_3D_SHARED_ABLATION_MODELS),
            "initialization": MNMS2_3D_INITIALIZATION,
            "seeds": list(args.seeds),
            "stages": list(args.stages),
            "smoke": args.smoke,
            "resume": args.resume,
            "amp": args.amp,
            "deterministic": args.deterministic,
            "python": sys.version,
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cinema_commit": cinema_git_commit(),
        },
    )


def mnms2_3d_record_failure(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
    error: BaseException,
) -> None:
    directory = mnms2_3d_run_dir_for(args, backbone, formulation, seed)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(
        directory / "failure.json",
        {
            "dataset": MNMS2_3D_DATASET,
            "task": MNMS2_3D_TASK_KEY,
            "backbone": backbone,
            "formulation": formulation,
            "initialization": MNMS2_3D_INITIALIZATION,
            "seed": seed,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )


def mnms2_3d_run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Seeds must be unique: {args.seeds}")

    print("=" * 100)
    print("M&Ms2 FULL-VOLUME 3-D CNN ARCHITECTURE BANK")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CineMA commit: {cinema_git_commit()}")
    print(f"Processed data: {mnms2_3d_processed_dir(args)}")
    print(f"Models: {list(args.models)}")
    print(f"Formulations: {list(args.formulations)}")
    print(f"Seeds: {list(args.seeds)}")
    print("=" * 100)

    stages = set(args.stages)
    if "preprocess" in stages:
        mnms2_3d_preprocess(args)
    if stages & {"validate", "train"}:
        mnms2_3d_validate_inputs(args)
    if "train" in stages:
        mnms2_3d_save_run_manifest(args)
        for backbone in args.models:
            for formulation in args.formulations:
                for seed in args.seeds:
                    try:
                        mnms2_3d_train_one(args, backbone, formulation, seed)
                    except Exception as error:
                        mnms2_3d_record_failure(
                            args, backbone, formulation, seed, error
                        )
                        cleanup_cuda()
                        print(
                            f"[failure] {backbone}/{formulation}/seed_{seed}: {type(error).__name__}: {error}",
                            file=sys.stderr,
                        )
                        if not args.continue_on_error:
                            raise
    if "aggregate" in stages or ("train" in stages and not args.smoke):
        mnms2_3d_aggregate_results(args)
    print(f"Done. Results: {mnms2_3d_task_root(args)}")


# Mnms2 Sax 2D

MNMS2_SAX_2D_CLASSES = protocols["mnms2_sax_2d"]["classes"]


MNMS2_SAX_2D_TASK_KEY = protocols["mnms2_sax_2d"]["task_key"]


def mnms2_sax_2d_train_one_run(
    args: argparse.Namespace,
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    backbone: str,
    formulation: str,
    initialization: str,
    seed: int,
) -> dict[str, Any]:
    output_group = "architecture_smoke" if args.smoke else "architecture_bank"
    run_dir = (
        args.output_dir
        / "checkpoints"
        / output_group
        / "2d"
        / MNMS2_SAX_2D_TASK_KEY
        / backbone
        / formulation
        / initialization
        / f"seed_{seed}"
    )
    summary_path = run_dir / "summary.json"
    if args.resume and summary_path.is_file():
        print(f"[resume] {backbone}/{formulation}/{initialization}/seed_{seed}")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)

    config = load_mnms2_config(data_root, seed)
    apply_overrides(args, config)
    train_loader, val_loader, test_loader = mnms2_sax_2d_make_loaders(
        data_root, splits, config, seed
    )
    amp_dtype, device = amp_dtype_and_device()
    model = MnMs2SAXMid2DClassifier(backbone, formulation, initialization).to(device)
    feature_layers = feature_layer_candidates(model, backbone)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    architecture = {
        "family": "mnms2_sax_mid_2d_architecture_bank",
        "task": MNMS2_SAX_2D_TASK_KEY,
        "dataset": "mnms2",
        "view": "sax_mid",
        "dimensionality": 2,
        "input_policy": "patient_mid_sax_ed_es",
        "backbone": backbone,
        "formulation": formulation,
        "initialization": initialization,
        "classes": list(MNMS2_SAX_2D_CLASSES),
        "feature_dim": model.feature_dim,
        "feature_layer_candidates": feature_layers,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "cinema_commit": cinema_git_commit(),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    save_json_arrays(run_dir / "architecture.json", architecture)
    print(
        f"[architecture] mnms2/2d/{backbone}/{formulation}/{initialization}/seed{seed} "
        f"parameters={parameter_count:,} feature_layers={feature_layers}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.train.lr),
        betas=tuple(config.train.betas),
        weight_decay=float(config.train.weight_decay),
    )
    scaler = grad_scaler(args.amp)
    accumulation_steps = get_n_accum_steps(
        batch_size=int(config.train.batch_size),
        batch_size_per_device=int(config.train.batch_size_per_device),
        world_size=1,
    )
    best_mcc = -math.inf
    best_epoch = -1
    bad_evaluations = 0
    checkpoint = run_dir / "best_val_mcc.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(int(config.train.n_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        n_steps = len(train_loader)
        progress = tqdm(
            train_loader, desc=f"mnms2/{backbone}/e{epoch + 1}", leave=False
        )
        for step, batch in enumerate(progress):
            learning_rate = adjust_learning_rate(
                optimizer=optimizer,
                step=step / max(len(train_loader), 1) + epoch,
                warmup_steps=int(config.train.n_warmup_epochs),
                max_n_steps=int(config.train.n_epochs),
                lr=float(config.train.lr),
                min_lr=float(config.train.min_lr),
            )
            image = batch["sax_image"].to(device, non_blocking=True)
            target = batch["label"].long().to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=args.amp and device.type == "cuda",
            ):
                logits = model(image)
                raw_loss = F.cross_entropy(
                    logits,
                    target,
                    label_smoothing=float(config.train.label_smoothing),
                )
                group_start = (step // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, n_steps - group_start)
                loss = raw_loss / group_size
            scaler.scale(loss).backward()
            update = (step + 1) % accumulation_steps == 0 or (step + 1) == n_steps
            if update:
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

        final_epoch = epoch + 1 == int(config.train.n_epochs)
        if (epoch + 1) % int(config.train.eval_interval) != 0 and not final_epoch:
            continue
        val_pids, val_y, val_probs, val_metrics = mnms2_sax_2d_evaluate(
            model, val_loader, device, amp_dtype
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(total_seen, 1),
                "train_accuracy": total_correct / max(total_seen, 1),
                "lr": learning_rate,
                **{
                    f"val_{key}": val_metrics[key]
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
        )
        pd.DataFrame(history).to_csv(output_file(run_dir / "history.csv"), index=False)
        print(
            f"[architecture] mnms2/2d/{backbone}/{formulation}/{initialization}/seed{seed} "
            f"epoch={epoch + 1} val_MCC={val_metrics['mcc']:.4f} "
            f"val_AUROC={val_metrics['roc_auc']:.4f} val_F1={val_metrics['f1']:.4f}"
        )
        if val_metrics["mcc"] > best_mcc + float(config.train.early_stopping.min_delta):
            best_mcc = float(val_metrics["mcc"])
            best_epoch = epoch + 1
            bad_evaluations = 0
            torch.save(
                {
                    "model": model.state_dict(),
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
                    "epoch": best_epoch,
                },
                checkpoint,
            )
            mnms2_sax_2d_save_predictions(
                run_dir / "best_validation", val_pids, val_y, val_probs, val_metrics
            )
        else:
            bad_evaluations += 1
        if bad_evaluations >= int(config.train.early_stopping.patience):
            break

    if not checkpoint.is_file():
        raise RuntimeError(f"No checkpoint was created for {run_dir}")
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    test_pids, test_y, test_probs, test_metrics = mnms2_sax_2d_evaluate(
        model, test_loader, device, amp_dtype
    )
    mnms2_sax_2d_save_predictions(
        run_dir / "test", test_pids, test_y, test_probs, test_metrics
    )
    summary = {
        "family": "mnms2_sax_mid_2d_architecture_bank",
        "task": MNMS2_SAX_2D_TASK_KEY,
        "dataset": "mnms2",
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
    del model, optimizer, scaler
    cleanup_cuda()
    return summary


def mnms2_sax_2d_run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages)

    print("=" * 100)
    print("M&Ms2 central-SAX patient-level 2-D architecture-bank benchmark")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CineMA commit: {cinema_git_commit()}")
    print(f"Task: {MNMS2_SAX_2D_TASK_KEY}")
    print(f"Classes: {list(MNMS2_SAX_2D_CLASSES)}")
    print(f"Models: {list(args.models_2d)}")
    print(f"Formulations: {list(args.formulations_2d)}")
    print(f"Initializations: {list(args.initializations_2d)}")
    print(f"Seeds: {list(args.seeds)}")
    print(f"Stages: {list(args.stages)}")
    print("=" * 100)

    if "preprocess" in stages:
        preprocess_mnms2(args)
    data_root = mnms2_sax_2d_resolve_processed_dir(args)
    splits = mnms2_sax_2d_load_official_splits(data_root)
    mnms2_sax_2d_save_split_audit(args.output_dir, splits)
    config = load_mnms2_config(data_root, seed=0)
    if stages & {"validate", "train"}:
        mnms2_sax_2d_validate_inputs(args.output_dir, data_root, splits, config)

    if "train" in stages:
        for backbone in args.models_2d:
            for formulation in args.formulations_2d:
                for initialization in args.initializations_2d:
                    for seed in args.seeds:
                        mnms2_sax_2d_train_one_run(
                            args,
                            data_root,
                            splits,
                            backbone,
                            formulation,
                            initialization,
                            seed,
                        )
        if not args.smoke:
            mnms2_sax_2d_aggregate_results(args)
    print(f"Done. Results: {args.output_dir}")


# Reproduction


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
        args.output_dir
        / "checkpoints"
        / output_group
        / spec.key
        / backbone
        / formulation
        / initialization
        / f"seed_{seed}"
    )
    summary_path = run_dir / "summary.json"
    if args.resume and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)

    data_root = processed_dir(args, spec.dataset)
    splits = split_metadata(spec, data_root)
    config = local_config(spec, data_root, seed)
    if args.smoke:
        config.train.n_epochs = 1
        config.train.n_warmup_epochs = 0
        config.train.eval_interval = 1
        config.train.batch_size = 4
        config.train.batch_size_per_device = 1
        config.train.n_workers = 0
    train_loader, val_loader, test_loader = make_loaders(
        spec, config, data_root, splits, seed
    )
    amp_dtype, device = amp_dtype_and_device()
    model = Reproduction2DClassifier(
        backbone, formulation, initialization, len(spec.classes)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.train.lr),
        betas=tuple(config.train.betas),
        weight_decay=float(config.train.weight_decay),
    )
    scaler = grad_scaler(args.amp)
    n_accum = get_n_accum_steps(
        batch_size=int(config.train.batch_size),
        batch_size_per_device=int(config.train.batch_size_per_device),
        world_size=1,
    )
    best_mcc = -math.inf
    best_epoch = -1
    bad_evaluations = 0
    checkpoint = run_dir / "best_val_mcc.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(int(config.train.n_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for step, batch in enumerate(
            tqdm(train_loader, desc=f"{spec.key}/{backbone}/e{epoch + 1}", leave=False)
        ):
            lr = adjust_learning_rate(
                optimizer=optimizer,
                step=step / len(train_loader) + epoch,
                warmup_steps=int(config.train.n_warmup_epochs),
                max_n_steps=int(config.train.n_epochs),
                lr=float(config.train.lr),
                min_lr=float(config.train.min_lr),
            )
            image = batch[f"{spec.view}_image"].to(device, non_blocking=True)
            target = batch["label"].long().to(device, non_blocking=True)
            enabled = args.amp and device.type == "cuda"
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=enabled
            ):
                logits = model(image)
                raw_loss = F.cross_entropy(
                    logits, target, label_smoothing=float(config.train.label_smoothing)
                )
                loss = raw_loss / n_accum
            scaler.scale(loss).backward()
            update = (step + 1) % n_accum == 0
            if update:
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

        if (epoch + 1) % int(config.train.eval_interval) != 0:
            continue
        vpids, vy, vprobs, val_metrics = reproduction_evaluate_cnn(
            model, val_loader, spec, device, amp_dtype
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(total_seen, 1),
            "train_accuracy": total_correct / max(total_seen, 1),
            "lr": lr,
            **{
                f"val_{k}": val_metrics[k]
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
        history.append(row)
        pd.DataFrame(history).to_csv(output_file(run_dir / "history.csv"), index=False)
        print(
            f"[cnn] {spec.key}/{backbone}/{formulation}/{initialization}/seed{seed} "
            f"epoch={epoch + 1} val_MCC={val_metrics['mcc']:.4f} val_AUROC={val_metrics['roc_auc']:.4f} "
            f"val_F1={val_metrics['f1']:.4f}"
        )
        if val_metrics["mcc"] > best_mcc + float(config.train.early_stopping.min_delta):
            best_mcc = float(val_metrics["mcc"])
            best_epoch = epoch + 1
            bad_evaluations = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "task": spec.key,
                    "classes": list(spec.classes),
                    "backbone": backbone,
                    "formulation": formulation,
                    "initialization": initialization,
                    "seed": seed,
                    "epoch": best_epoch,
                    "cinema_commit": cinema_git_commit(),
                },
                checkpoint,
            )
            save_predictions(
                run_dir / "best_validation",
                vpids,
                vy,
                vprobs,
                spec.classes,
                val_metrics,
            )
        else:
            bad_evaluations += 1
        if bad_evaluations >= int(config.train.early_stopping.patience):
            break

    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    pids, y, probs, test_metrics = reproduction_evaluate_cnn(
        model, test_loader, spec, device, amp_dtype
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
    del model, optimizer, scaler
    cleanup_cuda()
    return summary


def reproduction_run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = resolve_tasks(list(args.tasks))
    validation_specs = list(selected)
    if "cnn" in args.stages:
        for key in args.cnn_tasks:
            if TASKS[key] not in validation_specs:
                validation_specs.append(TASKS[key])
    datasets = sorted({spec.dataset for spec in validation_specs})

    print("=" * 100)
    print("CineMA exact reproduction and CNN-extension benchmark")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CineMA commit: {cinema_git_commit()}")
    print(f"Tasks: {[spec.key for spec in selected]}")
    print(f"Stages: {list(args.stages)}")
    print("=" * 100)

    if "preprocess" in args.stages:
        for dataset in datasets:
            preprocess_dataset(args, dataset)

    # Validate and persist splits before any evaluation or training.
    for spec in validation_specs:
        splits = split_metadata(spec, processed_dir(args, spec.dataset))
        save_split_audit(args, spec, splits)

    calibration_summaries = []
    if "calibrate" in args.stages:
        for spec in selected:
            calibration_summaries.append(reproduction_calibrate_task(args, spec))
        passed = all(item["passed"] for item in calibration_summaries)
        save_json(
            args.output_dir / "checkpoints" / "calibration" / "gate.json",
            {
                "passed": passed,
                "tasks": {x["task"]: x["passed"] for x in calibration_summaries},
            },
        )
        if not passed and not args.allow_calibration_failure:
            raise RuntimeError(
                "Released-checkpoint calibration failed. CNN/official training is intentionally blocked."
            )

    training_requested = "official" in args.stages or "cnn" in args.stages
    if training_requested and not args.smoke and not args.allow_calibration_failure:
        gate_tasks = (
            selected
            if "official" in args.stages
            else [TASKS[k] for k in args.cnn_tasks]
        )
        if not calibration_is_valid(args, gate_tasks):
            raise RuntimeError(
                "Run run.stages=[preprocess,calibrate] first. Full training is blocked until released checkpoints reproduce."
            )

    if "official" in args.stages:
        for spec in selected:
            for mode in args.official_modes:
                for seed in args.seeds:
                    run_official_training(args, spec, mode, seed)
        aggregate_official_results(args)

    if "cnn" in args.stages:
        for task_key in args.cnn_tasks:
            spec = TASKS[task_key]
            for backbone in args.cnn_models:
                for formulation in args.formulations:
                    for initialization in args.initializations:
                        for seed in args.seeds:
                            train_cnn(
                                args, spec, backbone, formulation, initialization, seed
                            )
        aggregate_cnn_results(args)

    print(f"Done. Results: {args.output_dir}")


EXPERIMENTS = {
    "acdc_2d": acdc_2d_run,
    "acdc_3d": acdc_3d_run,
    "acdc_distillation": acdc_distillation_run,
    "mnms2_sax_2d": mnms2_sax_2d_run,
    "mnms2_3d": mnms2_3d_run,
    "architecture_bank": architecture_bank_run,
    "reproduction": reproduction_run,
}


def run(config: DictConfig) -> None:
    """Validate the selected experiment and execute its requested stages."""
    experiment = str(config.experiment)
    if experiment not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment {experiment!r}; choose from {list(EXPERIMENTS)}"
        )
    args = prepare_run(config, experiment)
    for category in ("checkpoints", "metrics", "predictions", "logs"):
        (args.output_dir / category).mkdir(parents=True, exist_ok=True)
    EXPERIMENTS[experiment](args)
