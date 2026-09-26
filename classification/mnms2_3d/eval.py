"""Inference, metrics, and result aggregation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

from training_code.classification.eval import predict_probability
from training_code.classification.mnms2_3d.utils import task_root
from training_code.classification.protocol import protocols
from training_code.runtime import save_json

CLASSES = protocols["mnms2_3d"]["classes"]
DATASET = protocols["mnms2_3d"]["dataset"]
METRIC_NAMES = protocols["shared"]["metric_names"]
TASK_KEY = protocols["mnms2_3d"]["task_key"]


def metrics_from_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
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


def save_predictions(
    directory: Path,
    pids: list[str],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    prediction = probabilities.argmax(axis=1)
    table = pd.DataFrame(
        {
            "pid": pids,
            "target": y_true,
            "target_name": [CLASSES[index] for index in y_true],
            "prediction": prediction,
            "prediction_name": [CLASSES[index] for index in prediction],
        }
    )
    for index, name in enumerate(CLASSES):
        table[f"prob_{name}"] = probabilities[:, index]
    table.to_csv(directory / "predictions.csv", index=False)
    pd.DataFrame(metrics["confusion_matrix"], index=CLASSES, columns=CLASSES).to_csv(directory / "confusion_matrix.csv")
    save_json(directory / "metrics.json", metrics)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    patch_size: tuple[int, int, int],
    device: torch.device,
    amp_dtype: torch.dtype,
    amp: bool,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any], float]:
    model.eval()
    pids: list[str] = []
    targets: list[int] = []
    probabilities: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for batch in loader:
        image = batch["sax_image"].to(device, non_blocking=True)
        probability = predict_probability(model, image, patch_size, device, amp_dtype, amp)
        pids.extend(str(pid) for pid in batch["pid"])
        targets.extend(batch["label"].numpy().astype(int).tolist())
        probabilities.extend(probability.cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    metrics = metrics_from_probabilities(target_array, probability_array)
    return pids, target_array, probability_array, metrics, elapsed


def aggregate_results(args: argparse.Namespace) -> None:
    root = task_root(args, smoke=False)
    rows: list[dict[str, Any]] = []
    for path in root.rglob("summary.json") if root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("task") != TASK_KEY:
            continue
        row = {key: value for key, value in item.items() if key != "test"}
        row.update({f"test_{key}": value for key, value in item["test"].items()})
        rows.append(row)

    failures: list[dict[str, Any]] = []
    for path in root.rglob("failure.json") if root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        failures.append({key: value for key, value in item.items() if key != "traceback"})
    if failures:
        pd.DataFrame(failures).to_csv(root / "failures.csv", index=False)

    if not rows:
        print(f"[aggregate] No completed runs under {root}")
        return
    frame = pd.DataFrame(rows).sort_values(["test_mcc", "test_roc_auc"], ascending=False)
    frame.to_csv(root / "all_seed_results.csv", index=False)

    group_columns = ["backbone", "formulation", "initialization"]
    aggregate_columns = [
        *(f"test_{name}" for name in METRIC_NAMES),
        "training_seconds",
        "test_inference_ms_per_patient",
        "peak_gpu_memory_gb",
    ]
    summary_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys, strict=True))
        row.update({"dataset": DATASET, "task": TASK_KEY, "n_seeds": len(group)})
        row["complete_seed_set"] = set(group["seed"].astype(int)) == set(args.seeds)
        row["parameter_count"] = int(group["parameter_count"].iloc[0])
        for column in aggregate_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("test_mcc_mean", ascending=False)
    summary.to_csv(root / "mean_std_summary.csv", index=False)

    ensemble_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if group["seed"].duplicated().any():
            raise RuntimeError(f"Duplicate seed while ensembling {keys}")
        prediction_frames: list[pd.DataFrame] = []
        for checkpoint in group["checkpoint"]:
            prediction_path = Path(checkpoint).parent / "test" / "predictions.csv"
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            prediction_frames.append(pd.read_csv(prediction_path, dtype={"pid": str}))
        reference = prediction_frames[0]
        probability_columns = [f"prob_{name}" for name in CLASSES]
        seed_probabilities: list[np.ndarray] = []
        for prediction_frame in prediction_frames:
            if prediction_frame["pid"].tolist() != reference["pid"].tolist():
                raise RuntimeError(f"Patient-order mismatch while ensembling {keys}")
            if prediction_frame["target"].tolist() != reference["target"].tolist():
                raise RuntimeError(f"Target mismatch while ensembling {keys}")
            seed_probabilities.append(prediction_frame[probability_columns].to_numpy(dtype=np.float64))
        probabilities = np.mean(np.stack(seed_probabilities, axis=0), axis=0)
        targets = reference["target"].to_numpy(dtype=np.int64)
        metrics = metrics_from_probabilities(targets, probabilities)
        backbone, formulation, initialization = keys
        ensemble_directory = root / str(backbone) / str(formulation) / str(initialization) / "probability_ensemble"
        save_predictions(
            ensemble_directory,
            reference["pid"].tolist(),
            targets,
            probabilities,
            metrics,
        )
        ensemble_rows.append(
            {
                "dataset": DATASET,
                "task": TASK_KEY,
                "backbone": backbone,
                "formulation": formulation,
                "initialization": initialization,
                "n_seeds": len(group),
                "complete_seed_set": set(group["seed"].astype(int)) == set(args.seeds),
                **{name: metrics[name] for name in METRIC_NAMES},
            }
        )
    pd.DataFrame(ensemble_rows).sort_values("mcc", ascending=False).to_csv(
        root / "probability_ensemble_results.csv", index=False
    )
    print(f"[aggregate] Wrote results for {len(frame)} completed runs to {root}")
