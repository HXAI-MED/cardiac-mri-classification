"""Inference, metrics, and result aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from cinema import ConvViT
from cinema.classification.train import classification_forward
from training_code.classification.dataset import make_loaders, processed_dir, save_split_audit, split_metadata
from training_code.classification.eval import released_config, save_predictions
from training_code.classification.protocol import TASKS, TaskSpec, protocols
from training_code.runtime import amp_dtype_and_device, cleanup_cuda, save_json

PUBLISHED = protocols["shared"]["published"]


def metrics_from_probabilities(
    y_true: np.ndarray, probabilities: np.ndarray, classes: tuple[str, ...]
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prediction = probabilities.argmax(axis=1)
    labels = list(range(len(classes)))
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction, average="micro", labels=labels)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", labels=labels, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "roc_auc": float(roc_auc_score(y_true, probabilities, average="macro", multi_class="ovo", labels=labels)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=labels),
        "classification_report": classification_report(
            y_true,
            prediction,
            labels=labels,
            target_names=list(classes),
            output_dict=True,
            zero_division=0,
        ),
    }
    return result


@torch.no_grad()
def evaluate_cinema(
    model: nn.Module,
    loader: DataLoader,
    spec: TaskSpec,
    config: DictConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    model.eval()
    model.to(device)
    patch_size = config.data.sax.patch_size if spec.view == "sax" else config.data.lax.patch_size
    patch_size_dict = {spec.view: tuple(patch_size)}
    pids: list[str] = []
    targets: list[int] = []
    probabilities: list[np.ndarray] = []
    for batch in tqdm(loader, desc=f"evaluate {spec.key}", leave=False):
        image_dict = {spec.view: batch[f"{spec.view}_image"].to(device, non_blocking=True)}
        logits = classification_forward(model, image_dict, patch_size_dict, amp_dtype)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        pids.extend(str(x) for x in batch["pid"])
        targets.extend(batch["label"].numpy().astype(int).tolist())
        probabilities.extend(probs)
    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    metrics = metrics_from_probabilities(target_array, probability_array, spec.classes)
    return pids, target_array, probability_array, metrics


def calibrate_task(args: argparse.Namespace, spec: TaskSpec) -> dict[str, Any]:
    data_root = processed_dir(args, spec.dataset)
    splits = split_metadata(spec, data_root)
    save_split_audit(args, spec, splits)
    cache_dir = args.output_dir / "hf_cache"
    config = released_config(spec, cache_dir, data_root)
    config.model.views = spec.view
    _, _, test_loader = make_loaders(spec, config, data_root, splits, seed=0)
    amp_dtype, device = amp_dtype_and_device()
    rows: list[dict[str, Any]] = []
    seed_predictions: list[np.ndarray] = []
    seed_targets: np.ndarray | None = None
    seed_pids: list[str] | None = None

    for seed in args.seeds:
        run_dir = args.output_dir / "calibration" / spec.key / f"seed_{seed}"
        model = ConvViT.from_finetuned(
            repo_id="mathpluscode/CineMA",
            model_filename=(f"finetuned/classification_cvd/{spec.hf_name}/{spec.hf_name}_{seed}.safetensors"),
            config_filename=f"finetuned/classification_cvd/{spec.hf_name}/config.yaml",
            cache_dir=str(cache_dir),
        )
        pids, targets, probs, metrics = evaluate_cinema(model, test_loader, spec, config, device, amp_dtype)
        save_predictions(run_dir, pids, targets, probs, spec.classes, metrics)
        rows.append(
            {
                "task": spec.key,
                "seed": seed,
                **{k: metrics[k] for k in ("accuracy", "f1", "macro_f1", "balanced_accuracy", "mcc", "roc_auc")},
            }
        )
        if seed_targets is not None and not np.array_equal(seed_targets, targets):
            raise RuntimeError(f"Target mismatch across released seeds for {spec.key}")
        if seed_pids is not None and seed_pids != pids:
            raise RuntimeError(f"Patient-order mismatch across released seeds for {spec.key}")
        seed_targets, seed_pids = targets, pids
        seed_predictions.append(probs)
        del model
        cleanup_cuda()

    frame = pd.DataFrame(rows)
    summary = {
        "task": spec.key,
        "n_seeds": len(rows),
        "mean": {column: float(frame[column].mean()) for column in frame.columns if column not in ("task", "seed")},
        "std": {
            column: float(frame[column].std(ddof=1)) if len(frame) > 1 else 0.0
            for column in frame.columns
            if column not in ("task", "seed")
        },
        "published": PUBLISHED[spec.key]["cinema_finetune"],
    }
    summary["absolute_error"] = {
        "roc_auc": abs(summary["mean"]["roc_auc"] - summary["published"]["roc_auc"]),
        "f1": abs(summary["mean"]["f1"] - summary["published"]["f1"]),
    }
    summary["passed"] = all(error <= args.calibration_tolerance for error in summary["absolute_error"].values())

    ensemble_probs = np.mean(np.stack(seed_predictions, axis=0), axis=0)
    ensemble_metrics = metrics_from_probabilities(seed_targets, ensemble_probs, spec.classes)
    save_predictions(
        args.output_dir / "calibration" / spec.key / "probability_ensemble",
        seed_pids,
        seed_targets,
        ensemble_probs,
        spec.classes,
        ensemble_metrics,
    )
    summary["ensemble"] = {
        k: ensemble_metrics[k] for k in ("accuracy", "f1", "macro_f1", "balanced_accuracy", "mcc", "roc_auc")
    }
    frame.to_csv(args.output_dir / "calibration" / spec.key / "seed_metrics.csv", index=False)
    save_json(args.output_dir / "calibration" / spec.key / "summary.json", summary)
    print(
        f"[calibration] {spec.key}: mean AUROC={summary['mean']['roc_auc']:.4f} "
        f"(published {summary['published']['roc_auc']:.4f}), mean F1/accuracy={summary['mean']['f1']:.4f} "
        f"(published {summary['published']['f1']:.4f}) -> {'PASS' if summary['passed'] else 'FAIL'}"
    )
    return summary


@torch.no_grad()
def evaluate_cnn(
    model: nn.Module,
    loader: DataLoader,
    spec: TaskSpec,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    model.eval()
    pids: list[str] = []
    targets: list[int] = []
    probabilities: list[np.ndarray] = []
    for batch in loader:
        image = batch[f"{spec.view}_image"].to(device, non_blocking=True)
        enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enabled):
            logits = model(image)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        pids.extend(str(x) for x in batch["pid"])
        targets.extend(batch["label"].numpy().astype(int).tolist())
        probabilities.extend(probs)
    y = np.asarray(targets, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    return pids, y, p, metrics_from_probabilities(y, p, spec.classes)


def aggregate_architecture_results(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    root = args.output_dir / "architecture_bank"
    for path in root.rglob("summary.json") if root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        row = {k: v for k, v in item.items() if k != "test"}
        row.update({f"test_{k}": v for k, v in item["test"].items()})
        rows.append(row)
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "all_seed_results.csv", index=False)
    groups = ["dimensionality", "task", "backbone", "formulation", "initialization"]
    metrics = ["test_accuracy", "test_f1", "test_macro_f1", "test_balanced_accuracy", "test_mcc", "test_roc_auc"]
    summary_rows = []
    for keys, group in frame.groupby(groups, dropna=False):
        row = dict(zip(groups, keys))
        row["n_seeds"] = len(group)
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("test_roc_auc_mean", ascending=False)
    summary.to_csv(root / "mean_std_summary.csv", index=False)

    ensemble_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(groups, dropna=False):
        dimensionality, task, backbone, formulation, initialization = keys
        spec = TASKS[str(task)]
        prediction_frames: list[pd.DataFrame] = []
        for checkpoint in group["checkpoint"]:
            prediction_path = Path(checkpoint).parent / "test" / "predictions.csv"
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            prediction_frames.append(pd.read_csv(prediction_path, dtype={"pid": str}))

        reference = prediction_frames[0]
        probability_columns = [f"prob_{name}" for name in spec.classes]
        seed_probabilities = []
        for prediction_frame in prediction_frames:
            if prediction_frame["pid"].tolist() != reference["pid"].tolist():
                raise RuntimeError(f"Patient-order mismatch while ensembling {keys}")
            if prediction_frame["target"].tolist() != reference["target"].tolist():
                raise RuntimeError(f"Target mismatch while ensembling {keys}")
            seed_probabilities.append(prediction_frame[probability_columns].to_numpy(dtype=np.float64))

        ensemble_probabilities = np.mean(np.stack(seed_probabilities, axis=0), axis=0)
        targets = reference["target"].to_numpy(dtype=np.int64)
        ensemble_metrics = metrics_from_probabilities(targets, ensemble_probabilities, spec.classes)
        ensemble_dir = (
            root
            / f"{int(dimensionality)}d"
            / str(task)
            / str(backbone)
            / str(formulation)
            / str(initialization)
            / "probability_ensemble"
        )
        save_predictions(
            ensemble_dir,
            reference["pid"].tolist(),
            targets,
            ensemble_probabilities,
            spec.classes,
            ensemble_metrics,
        )
        ensemble_rows.append(
            {
                "dimensionality": int(dimensionality),
                "task": task,
                "backbone": backbone,
                "formulation": formulation,
                "initialization": initialization,
                "n_seeds": len(prediction_frames),
                **{
                    key: ensemble_metrics[key]
                    for key in ("accuracy", "f1", "macro_f1", "balanced_accuracy", "mcc", "roc_auc")
                },
            }
        )
    pd.DataFrame(ensemble_rows).sort_values("roc_auc", ascending=False).to_csv(
        root / "probability_ensemble_results.csv", index=False
    )
