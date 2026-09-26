"""Classification training."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from cinema.classification.train import classification_forward
from cinema.optim import adjust_learning_rate, get_n_accum_steps
from training_code.classification.acdc_3d.dataset import (
    local_config,
    make_dataset,
    make_loaders,
    preprocess,
    processed_dir,
    split_metadata,
    validate_inputs,
)
from training_code.classification.acdc_3d.eval import evaluate, save_predictions
from training_code.classification.acdc_3d.model import EDES3DClassifier
from training_code.classification.acdc_distillation.eval import aggregate_results
from training_code.classification.acdc_distillation.model import load_teacher
from training_code.classification.acdc_distillation.utils import run_dir_for, task_root, teacher_paths
from training_code.classification.protocol import protocols
from training_code.classification.utils import apply_training_overrides, prepare_run
from training_code.runtime import amp_dtype_and_device, cleanup_cuda, grad_scaler, save_json

CINEMA_RANDINIT_WEIGHT_DECAY = protocols["acdc_3d"]["cinema_randinit_weight_decay"]
CLASSES = protocols["acdc"]["classes"]
DATASET = protocols["acdc_3d"]["dataset"]
INITIALIZATION = protocols["acdc_distillation"]["initialization"]
METRIC_NAMES = protocols["shared"]["metric_names"]
PROTOCOL_VERSION = protocols["acdc_distillation"]["protocol_version"]
TASK_KEY = protocols["acdc_3d"]["task_key"]
VIEW = protocols["shared"]["view"]


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
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "config": str(config),
        "config_size": config_stat.st_size,
        "config_mtime_ns": config_stat.st_mtime_ns,
        "pids": pids,
        "classes": list(CLASSES),
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
    cache_dir = args.output_dir / "teacher_targets" / TASK_KEY / f"seed_{teacher_seed}"
    metadata_path = cache_dir / "metadata.json"
    logits_path = cache_dir / "logits.npz"

    if metadata_path.is_file() and logits_path.is_file():
        observed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if observed_metadata == expected_metadata:
            with np.load(logits_path) as cached:
                logits = cached["logits"]
            if logits.shape == (len(pids), len(CLASSES)):
                print(f"[teacher-cache] reusing {logits_path}")
                return (
                    {pid: torch.from_numpy(row.copy()) for pid, row in zip(pids, logits, strict=True)},
                    teacher_seed,
                    checkpoint,
                )

    print(f"[teacher-cache] computing seed {teacher_seed} targets from {checkpoint}")
    teacher = load_teacher(checkpoint, teacher_config, device)
    dataset = make_dataset(config, data_root, train_frame, "train", False)
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
            {VIEW: image},
            {VIEW: patch_size},
            amp_dtype,
        )
        rows.append(logits.float().cpu().numpy()[0])
    logits = np.asarray(rows, dtype=np.float32)
    if logits.shape != (len(pids), len(CLASSES)) or not np.isfinite(logits).all():
        raise RuntimeError(f"Invalid cached teacher logits shape/values: {logits.shape}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_dir / "logits.tmp.npz"
    np.savez_compressed(temporary_path, logits=logits)
    temporary_path.replace(logits_path)
    save_json(metadata_path, expected_metadata)
    del teacher, loader, dataset
    cleanup_cuda()
    return {pid: torch.from_numpy(row.copy()) for pid, row in zip(pids, logits, strict=True)}, teacher_seed, checkpoint


def train_one(
    args: argparse.Namespace,
    backbone: str,
    formulation: str,
    seed: int,
) -> dict[str, Any]:
    run_dir = run_dir_for(args, backbone, formulation, seed)
    summary_path = run_dir / "summary.json"
    if args.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("protocol_version") == PROTOCOL_VERSION:
            print(f"[resume] {backbone}/{formulation}/{INITIALIZATION}/seed_{seed}")
            return previous
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    data_root = processed_dir(args)
    splits = split_metadata(data_root)
    config = local_config(data_root, seed)
    apply_training_overrides(args, config)
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
    teacher_targets, teacher_seed, teacher_checkpoint = get_teacher_targets(
        args,
        config,
        data_root,
        splits["train"],
        seed,
        device,
        amp_dtype,
    )

    student = EDES3DClassifier(backbone, formulation, len(CLASSES)).to(device)
    optimizer_weight_decay = CINEMA_RANDINIT_WEIGHT_DECAY if args.weight_decay is None else args.weight_decay
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
            "protocol_version": PROTOCOL_VERSION,
            "family": "acdc_3d_architecture_bank_distillation",
            "dataset": DATASET,
            "task": TASK_KEY,
            "backbone": backbone,
            "formulation": formulation,
            "initialization": INITIALIZATION,
            "teacher_seed": teacher_seed,
            "teacher_checkpoint": teacher_checkpoint,
            "temperature": args.temperature,
            "supervised_weight": args.supervised_weight,
            "optimizer_weight_decay": optimizer_weight_decay,
            "parameter_count": sum(parameter.numel() for parameter in student.parameters()),
            "config": OmegaConf.to_container(config, resolve=True),
        },
    )

    for epoch in range(int(config.train.n_epochs)):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "supervised": 0.0, "distilled": 0.0, "correct": 0, "seen": 0}
        number_of_steps = len(train_loader)
        current_lr = float(config.train.lr)
        for step, batch in enumerate(tqdm(train_loader, desc=f"KD {backbone}/e{epoch + 1}", leave=False)):
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
            soft_targets = torch.stack([teacher_targets[str(pid)] for pid in batch["pid"]]).to(
                device, non_blocking=True
            )
            enabled = args.amp and device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enabled):
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
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(config.train.clip_grad))
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
        val_pids, val_y, val_probabilities, val_metrics, val_seconds = evaluate(
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
                **{f"val_{name}": val_metrics[name] for name in METRIC_NAMES},
            }
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
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
                    "protocol_version": PROTOCOL_VERSION,
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
            break

    if not checkpoint.is_file():
        raise RuntimeError(f"No validation checkpoint was written: {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    student.load_state_dict(state["model"])
    test_pids, test_y, test_probabilities, test_metrics, test_seconds = evaluate(
        student, test_loader, patch_size, device, amp_dtype, args.amp
    )
    save_predictions(run_dir / "test", test_pids, test_y, test_probabilities, test_metrics)

    training_seconds = time.perf_counter() - started
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "family": "acdc_3d_architecture_bank_distillation",
        "dataset": DATASET,
        "task": TASK_KEY,
        "backbone": backbone,
        "formulation": formulation,
        "initialization": INITIALIZATION,
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
            float(torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0
        ),
        "test": {name: test_metrics[name] for name in METRIC_NAMES},
    }
    save_json(summary_path, summary)
    print(
        f"[test] {backbone}/{formulation}/seed_{seed}: "
        f"MCC={test_metrics['mcc']:.4f} AUROC={test_metrics['roc_auc']:.4f}"
    )
    del student, optimizer, scaler, teacher_targets
    cleanup_cuda()
    return summary


def run(args: argparse.Namespace) -> None:
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
        preprocess(args)
    if stages & {"validate", "train"}:
        validate_inputs(args)
    if "train" in stages:
        for backbone in args.models:
            for formulation in args.formulations:
                for seed in args.seeds:
                    try:
                        train_one(args, backbone, formulation, seed)
                    except Exception as error:
                        failure = run_dir_for(args, backbone, formulation, seed) / "failure.json"
                        save_json(
                            failure,
                            {
                                "protocol_version": PROTOCOL_VERSION,
                                "backbone": backbone,
                                "formulation": formulation,
                                "seed": seed,
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        cleanup_cuda()
                        print(f"[failure] {backbone}/{formulation}/seed_{seed}: {error}")
                        if not args.continue_on_error:
                            raise
    if "aggregate" in stages or ("train" in stages and not args.smoke):
        aggregate_results(args)
    gc.collect()
    print(f"Done. Distilled results: {task_root(args)}")


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    """Run the experiment using the adjacent YAML configuration."""
    run(prepare_run(config, "acdc_distillation"))


if __name__ == "__main__":
    main()
