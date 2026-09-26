"""Classification training."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import traceback
from typing import Any

import hydra
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from cinema.optim import adjust_learning_rate, get_n_accum_steps
from training_code.classification.acdc_3d.dataset import (
    local_config,
    make_loaders,
    preprocess,
    processed_dir,
    split_metadata,
    validate_inputs,
)
from training_code.classification.acdc_3d.eval import aggregate_results, evaluate, save_predictions
from training_code.classification.acdc_3d.model import EDES3DClassifier, feature_layer_candidates
from training_code.classification.acdc_3d.utils import run_dir_for, task_root
from training_code.classification.protocol import protocols
from training_code.classification.utils import apply_training_overrides, prepare_run
from training_code.runtime import amp_dtype_and_device, cinema_git_commit, cleanup_cuda, grad_scaler, save_json

CINEMA_RANDINIT_WEIGHT_DECAY = protocols["acdc_3d"]["cinema_randinit_weight_decay"]
CLASSES = protocols["acdc"]["classes"]
DATASET = protocols["acdc_3d"]["dataset"]
INITIALIZATION = protocols["shared"]["initialization"]
METRIC_NAMES = protocols["shared"]["metric_names"]
PROTOCOL_VERSION = protocols["acdc_3d"]["protocol_version"]
SHARED_ABLATION_MODELS = protocols["shared"]["shared_ablation_models"]
TASK_KEY = protocols["acdc_3d"]["task_key"]
VIEW = protocols["shared"]["view"]


def train_one(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
) -> dict[str, Any]:
    run_dir = run_dir_for(args, backbone, formulation, seed)
    summary_path = run_dir / "summary.json"
    failure_path = run_dir / "failure.json"
    if args.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("protocol_version") == PROTOCOL_VERSION:
            print(f"[resume] {TASK_KEY}/{backbone}/{formulation}/{INITIALIZATION}/seed_{seed}")
            return previous
        print(f"[resume] retraining stale protocol: {summary_path}")
    if args.resume and failure_path.is_file() and not args.retry_failures:
        print(f"[resume] skipping recorded failure: {failure_path}")
        return {}
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    data_root = processed_dir(args)
    splits = split_metadata(data_root)
    config = local_config(data_root, seed)
    apply_training_overrides(args, config)
    patch_size = tuple(int(value) for value in config.data.sax.patch_size)
    train_loader, val_loader, test_loader = make_loaders(
        config,
        data_root,
        splits,
        seed=seed,
        deterministic=args.deterministic,
    )
    amp_dtype, device = amp_dtype_and_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = EDES3DClassifier(backbone, formulation, len(CLASSES)).to(device)
    feature_layers = feature_layer_candidates(model, backbone)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    architecture = {
        "protocol_version": PROTOCOL_VERSION,
        "family": "acdc_3d_architecture_bank",
        "dataset": DATASET,
        "task": TASK_KEY,
        "view": VIEW,
        "dimensionality": 3,
        "input_policy": "full_sax_ed_es_volume_per_patient",
        "backbone": backbone,
        "formulation": formulation,
        "initialization": INITIALIZATION,
        "classes": list(CLASSES),
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
        "optimizer_weight_decay": (CINEMA_RANDINIT_WEIGHT_DECAY if args.weight_decay is None else args.weight_decay),
        "cinema_commit": cinema_git_commit(),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    save_json(run_dir / "architecture.json", architecture)
    print(
        f"[train] {TASK_KEY}/{backbone}/{formulation}/{INITIALIZATION}/seed_{seed} "
        f"parameters={parameter_count:,} layers={feature_layers}"
    )

    optimizer_weight_decay = CINEMA_RANDINIT_WEIGHT_DECAY if args.weight_decay is None else args.weight_decay
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
        for step, batch in enumerate(tqdm(train_loader, desc=f"{TASK_KEY}/{backbone}/e{epoch + 1}", leave=False)):
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
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enabled):
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
            update = (step + 1) % accumulation_steps == 0 or (step + 1) == number_of_steps
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.train.clip_grad))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(raw_loss.item()) * target.size(0)
            total_correct += int((logits.argmax(dim=1) == target).sum().item())
            total_seen += int(target.size(0))

        final_epoch = epoch + 1 == int(config.train.n_epochs)
        if (epoch + 1) % int(config.train.eval_interval) != 0 and not final_epoch:
            continue

        val_pids, val_y, val_probabilities, val_metrics, val_seconds = evaluate(
            model, val_loader, patch_size, device, amp_dtype, args.amp
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(total_seen, 1),
            "train_accuracy": total_correct / max(total_seen, 1),
            "lr": current_lr,
            "validation_seconds": val_seconds,
            **{f"val_{name}": val_metrics[name] for name in METRIC_NAMES},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(
            f"[eval] {TASK_KEY}/{backbone}/{formulation}/seed_{seed} epoch={epoch + 1} "
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
                    "protocol_version": PROTOCOL_VERSION,
                    "family": "acdc_3d_architecture_bank",
                    "dataset": DATASET,
                    "task": TASK_KEY,
                    "classes": list(CLASSES),
                    "view": VIEW,
                    "dimensionality": 3,
                    "input_policy": "full_sax_ed_es_volume_per_patient",
                    "backbone": backbone,
                    "formulation": formulation,
                    "initialization": INITIALIZATION,
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
                val_pids,
                val_y,
                val_probabilities,
                val_metrics,
            )
        else:
            bad_evaluations += 1
        if bad_evaluations >= int(config.train.early_stopping.patience):
            print(f"[early-stop] {backbone}/{formulation}/seed_{seed} at epoch {epoch + 1}")
            break

    if not checkpoint.is_file():
        raise RuntimeError(f"No validation checkpoint was written: {checkpoint}")
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    test_pids, test_y, test_probabilities, test_metrics, test_seconds = evaluate(
        model, test_loader, patch_size, device, amp_dtype, args.amp
    )
    save_predictions(run_dir / "test", test_pids, test_y, test_probabilities, test_metrics)

    training_seconds = time.perf_counter() - started
    peak_gpu_memory_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "family": "acdc_3d_architecture_bank",
        "dataset": DATASET,
        "task": TASK_KEY,
        "view": VIEW,
        "dimensionality": 3,
        "input_policy": "full_sax_ed_es_volume_per_patient",
        "backbone": backbone,
        "formulation": formulation,
        "initialization": INITIALIZATION,
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
        "test": {name: test_metrics[name] for name in METRIC_NAMES},
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


def save_run_manifest(args: argparse.Namespace) -> None:
    save_json(
        task_root(args) / "run_manifest.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "dataset": DATASET,
            "task": TASK_KEY,
            "view": VIEW,
            "dimensionality": 3,
            "input_policy": "full_sax_ed_es_volume_per_patient",
            "models": list(args.models),
            "formulations": list(args.formulations),
            "shared_ablation_models": list(SHARED_ABLATION_MODELS),
            "initialization": INITIALIZATION,
            "optimizer_weight_decay": (
                CINEMA_RANDINIT_WEIGHT_DECAY if args.weight_decay is None else args.weight_decay
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


def record_failure(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
    error: BaseException,
) -> None:
    directory = run_dir_for(args, backbone, formulation, seed)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(
        directory / "failure.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "dataset": DATASET,
            "task": TASK_KEY,
            "backbone": backbone,
            "formulation": formulation,
            "initialization": INITIALIZATION,
            "seed": seed,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )


def run(args: argparse.Namespace) -> None:
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
    print(f"Processed data: {processed_dir(args)}")
    print(f"Models: {list(args.models)}")
    print(f"Formulations: {list(args.formulations)}")
    print(f"Seeds: {list(args.seeds)}")
    print("=" * 100)

    stages = set(args.stages)
    if "preprocess" in stages:
        preprocess(args)
    if stages & {"validate", "train"}:
        validate_inputs(args)
    if "train" in stages:
        save_run_manifest(args)
        for backbone in args.models:
            for formulation in args.formulations:
                for seed in args.seeds:
                    try:
                        train_one(args, backbone, formulation, seed)
                    except Exception as error:
                        record_failure(args, backbone, formulation, seed, error)
                        cleanup_cuda()
                        print(
                            f"[failure] {backbone}/{formulation}/seed_{seed}: {type(error).__name__}: {error}",
                            file=sys.stderr,
                        )
                        if not args.continue_on_error:
                            raise
    if "aggregate" in stages or ("train" in stages and not args.smoke):
        aggregate_results(args)
    print(f"Done. Results: {task_root(args)}")


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    """Run the experiment using the adjacent YAML configuration."""
    run(prepare_run(config, "acdc_3d"))


if __name__ == "__main__":
    main()
