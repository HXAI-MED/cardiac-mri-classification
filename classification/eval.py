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
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from cinema.transform import get_patch_grid, patch_grid_sample
from training_code.classification.dataset import local_config
from training_code.classification.protocol import TaskSpec, protocols
from training_code.runtime import save_json

CLASSES = protocols["acdc"]["classes"]
PUBLISHED = protocols["shared"]["published"]


def metrics_from_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prediction = probabilities.argmax(axis=1)
    labels = list(range(len(CLASSES)))
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction, average="micro", labels=labels)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", labels=labels, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
                average="macro",
                multi_class="ovo",
                labels=labels,
            )
        ),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=labels),
        "classification_report": classification_report(
            y_true,
            prediction,
            labels=labels,
            target_names=list(CLASSES),
            output_dict=True,
            zero_division=0,
        ),
    }


@torch.no_grad()
def predict_probability(
    model: nn.Module,
    image: torch.Tensor,
    patch_size: tuple[int, int, int],
    device: torch.device,
    amp_dtype: torch.dtype,
    amp: bool,
) -> torch.Tensor:
    if image.shape[0] != 1:
        raise ValueError("Patch-based evaluation requires batch size 1")
    spatial_shape = tuple(int(value) for value in image.shape[2:])
    if any(size < patch for size, patch in zip(spatial_shape, patch_size, strict=True)):
        raise ValueError(f"Image {spatial_shape} is smaller than patch {patch_size}")

    enabled = amp and device.type == "cuda"
    if spatial_shape == patch_size:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enabled):
            return torch.softmax(model(image).float(), dim=1)

    overlap = tuple(size // 2 for size in patch_size)
    starts = get_patch_grid(spatial_shape, patch_size, overlap)
    patches = patch_grid_sample(image[0], starts, patch_size)
    probabilities: list[torch.Tensor] = []
    for patch in patches:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enabled):
            probabilities.append(torch.softmax(model(patch.unsqueeze(0)).float(), dim=1))
    return torch.cat(probabilities, dim=0).mean(dim=0, keepdim=True)


def save_predictions(
    directory: Path,
    pids: list[str],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[str, ...],
    metrics: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    prediction = probabilities.argmax(axis=1)
    table = pd.DataFrame(
        {
            "pid": pids,
            "target": y_true,
            "target_name": [classes[i] for i in y_true],
            "prediction": prediction,
            "prediction_name": [classes[i] for i in prediction],
        }
    )
    for index, name in enumerate(classes):
        table[f"prob_{name}"] = probabilities[:, index]
    table.to_csv(directory / "predictions.csv", index=False)
    pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes).to_csv(directory / "confusion_matrix.csv")
    save_json(directory / "metrics.json", metrics)


def released_config(spec: TaskSpec, cache_dir: Path, data_root: Path) -> DictConfig:
    path = hf_hub_download(
        repo_id="mathpluscode/CineMA",
        filename=f"finetuned/classification_cvd/{spec.hf_name}/config.yaml",
        cache_dir=str(cache_dir),
    )
    # The released checkpoint config contains only the fields needed to rebuild
    # the model.  Dataset transforms, loader settings, and training defaults live
    # in the full config shipped with the CineMA package.  Merge the checkpoint
    # values over that full task config instead of treating the small released
    # YAML as a standalone data config.
    checkpoint_config = OmegaConf.load(path)
    config = OmegaConf.merge(local_config(spec, data_root, seed=0), checkpoint_config)
    config.data.dir = str(data_root)
    config.model.views = spec.view
    if tuple(config.data[config.data.class_column]) != spec.classes:
        raise RuntimeError(f"Released checkpoint classes changed for {spec.key}")
    return config


def calibration_is_valid(args: argparse.Namespace, tasks: list[TaskSpec]) -> bool:
    failures = []
    for spec in tasks:
        path = args.output_dir / "calibration" / spec.key / "summary.json"
        if not path.exists():
            failures.append(f"{spec.key}: missing calibration")
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not summary.get("passed", False):
            failures.append(f"{spec.key}: calibration failed")
    if failures:
        print("[gate] " + "; ".join(failures))
        return False
    return True


def aggregate_official_results(args: argparse.Namespace) -> None:
    root = args.output_dir / "official_training"
    rows: list[dict[str, Any]] = []
    for path in root.rglob("DONE.json") if root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        row = {key: value for key, value in item.items() if key != "test"}
        row.update({f"test_{key}": value for key, value in item["test"].items()})
        rows.append(row)
    if not rows:
        return

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "all_seed_results.csv", index=False)
    groups = ["task", "mode", "smoke"]
    metric_columns = [column for column in frame.columns if column.startswith("test_")]
    summary_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(groups, dropna=False):
        row = dict(zip(groups, keys))
        row["n_seeds"] = len(group)
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        reference = PUBLISHED.get(str(row["task"]), {}).get(str(row["mode"]))
        if reference and not bool(row["smoke"]):
            row["published_roc_auc"] = reference["roc_auc"]
            row["published_f1"] = reference["f1"]
            row["roc_auc_absolute_error"] = abs(row["test_roc_auc_mean"] - reference["roc_auc"])
            row["f1_absolute_error"] = abs(row["test_f1_mean"] - reference["f1"])
        summary_rows.append(row)
    pd.DataFrame(summary_rows).sort_values(["task", "mode"]).to_csv(root / "mean_std_vs_published.csv", index=False)
