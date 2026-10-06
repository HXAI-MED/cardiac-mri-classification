"""Teacher targets, student distillation and baseline comparisons."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from cinema.classification.train import classification_forward
from cinema.optim import adjust_learning_rate, get_n_accum_steps
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from .cinema_support import load_teacher
from .dataset import (
    acdc_3d_local_config,
    acdc_3d_make_dataset,
    acdc_3d_make_loaders,
    acdc_3d_processed_dir,
    acdc_3d_split_metadata,
)
from .evaluate import acdc_3d_evaluate, acdc_3d_save_predictions
from .metrics import metrics_from_probabilities
from .models import ACDC3DClassifier
from .protocol import protocols
from .utils import (
    acdc_3d_task_root,
    acdc_distillation_run_dir_for,
    acdc_distillation_task_root,
    amp_dtype_and_device,
    apply_training_overrides,
    artifact_path,
    cleanup_cuda,
    grad_scaler,
    output_file,
    prepare_run_directory,
    save_json,
    teacher_paths,
)

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


BASELINE_PROTOCOL_VERSION = protocols["acdc_3d"]["protocol_version"]


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
        protocol_version=ACDC_DISTILLATION_PROTOCOL_VERSION,
        temperature=args.temperature,
        supervised_weight=args.supervised_weight,
        teacher=teacher_cache_metadata(
            *teacher_paths(args, seed)[:2], splits["train"]["pid"].astype(str).tolist()
        ),
        optimizer_weight_decay=ACDC_DISTILLATION_CINEMA_RANDINIT_WEIGHT_DECAY
        if args.weight_decay is None
        else args.weight_decay,
    )
    if previous is not None:
        return previous
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


def acdc_distillation_aggregate_results(args: argparse.Namespace) -> None:
    root = acdc_distillation_task_root(args, smoke=False)
    rows: list[dict[str, Any]] = []
    for path in root.rglob("summary.json") if root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("protocol_version") != ACDC_DISTILLATION_PROTOCOL_VERSION:
            continue
        row = {key: value for key, value in item.items() if key != "test"}
        row.update({f"test_{key}": value for key, value in item["test"].items()})
        rows.append(row)
    if not rows:
        print(f"[aggregate] no completed distilled runs under {root}")
        return
    frame = pd.DataFrame(rows).sort_values("test_mcc", ascending=False)
    frame.to_csv(output_file(root / "all_seed_results.csv"), index=False)

    group_columns = [
        "backbone",
        "formulation",
        "initialization",
        "temperature",
        "supervised_weight",
    ]
    metric_columns = [f"test_{name}" for name in ACDC_DISTILLATION_METRIC_NAMES]
    summary_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys, strict=True))
        row["n_seeds"] = len(group)
        row["complete_seed_set"] = set(group["seed"].astype(int)) == set(args.seeds)
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = (
                float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
            )
        summary_rows.append(row)
    distilled_summary = pd.DataFrame(summary_rows).sort_values(
        "test_mcc_mean", ascending=False
    )
    distilled_summary.to_csv(output_file(root / "mean_std_summary.csv"), index=False)

    baseline_rows: list[dict[str, Any]] = []
    baseline_root = acdc_3d_task_root(args, smoke=False)
    for path in baseline_root.rglob("summary.json") if baseline_root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("protocol_version") != BASELINE_PROTOCOL_VERSION:
            continue
        baseline_rows.append(
            {
                "backbone": item["backbone"],
                "formulation": item["formulation"],
                **{
                    f"test_{name}": item["test"][name]
                    for name in ACDC_DISTILLATION_METRIC_NAMES
                },
            }
        )
    if baseline_rows:
        baseline = (
            pd.DataFrame(baseline_rows)
            .groupby(["backbone", "formulation"], as_index=False)[metric_columns]
            .mean()
        )
        distilled = distilled_summary[
            [
                "backbone",
                "formulation",
                *(f"{column}_mean" for column in metric_columns),
            ]
        ].rename(columns={f"{column}_mean": column for column in metric_columns})
        comparison = distilled.merge(
            baseline,
            on=["backbone", "formulation"],
            suffixes=("_distilled", "_randinit"),
        )
        for column in metric_columns:
            comparison[f"{column}_delta"] = (
                comparison[f"{column}_distilled"] - comparison[f"{column}_randinit"]
            )
        comparison.to_csv(output_file(root / "comparison_vs_randinit.csv"), index=False)

    ensemble_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        predictions = [
            pd.read_csv(
                artifact_path(Path(checkpoint).parent / "test" / "predictions.csv"),
                dtype={"pid": str},
            )
            for checkpoint in group["checkpoint"]
        ]
        reference = predictions[0]
        probability_columns = [f"prob_{name}" for name in ACDC_DISTILLATION_CLASSES]
        for prediction in predictions[1:]:
            if prediction["pid"].tolist() != reference["pid"].tolist():
                raise RuntimeError(f"Patient order differs while ensembling {keys}")
        probabilities = np.mean(
            np.stack([item[probability_columns].to_numpy() for item in predictions]),
            axis=0,
        )
        targets = reference["target"].to_numpy(dtype=np.int64)
        metrics = metrics_from_probabilities(targets, probabilities)
        backbone, formulation, initialization, temperature, supervised_weight = keys
        output = (
            root / str(backbone) / str(formulation) / str(initialization) / "ensemble"
        )
        acdc_3d_save_predictions(
            output, reference["pid"].tolist(), targets, probabilities, metrics
        )
        ensemble_rows.append(
            {
                "backbone": backbone,
                "formulation": formulation,
                "initialization": initialization,
                "temperature": temperature,
                "supervised_weight": supervised_weight,
                "n_seeds": len(group),
                **{name: metrics[name] for name in ACDC_DISTILLATION_METRIC_NAMES},
            }
        )
    pd.DataFrame(ensemble_rows).sort_values("mcc", ascending=False).to_csv(
        output_file(root / "probability_ensemble_results.csv"), index=False
    )
    print(f"[aggregate] wrote {len(frame)} distilled runs to {root}")
