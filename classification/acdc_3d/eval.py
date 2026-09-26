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
from torch.utils.data import DataLoader

from training_code.classification.acdc_3d.utils import run_dir_for, task_root
from training_code.classification.eval import metrics_from_probabilities, predict_probability
from training_code.classification.protocol import protocols
from training_code.runtime import save_json

CLASSES = protocols["acdc"]["classes"]
DATASET = protocols["acdc_3d"]["dataset"]
INITIALIZATION = protocols["shared"]["initialization"]
MAIN_MODELS = protocols["acdc_3d"]["main_models"]
METRIC_NAMES = protocols["shared"]["metric_names"]
PROTOCOL_VERSION = protocols["acdc_3d"]["protocol_version"]
TASK_KEY = protocols["acdc_3d"]["task_key"]


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
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in root.rglob("summary.json") if root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("task") != TASK_KEY or item.get("protocol_version") != PROTOCOL_VERSION:
            continue
        row = {key: value for key, value in item.items() if key != "test"}
        row.update({f"test_{key}": value for key, value in item["test"].items()})
        row["main_bank"] = item.get("backbone") in MAIN_MODELS
        rows.append(row)

    failures: list[dict[str, Any]] = []
    for path in root.rglob("failure.json") if root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("protocol_version") != PROTOCOL_VERSION:
            continue
        failures.append({key: value for key, value in item.items() if key != "traceback"})
    if failures:
        pd.DataFrame(failures).to_csv(root / "failures.csv", index=False)

    status_rows: list[dict[str, Any]] = []
    for backbone in args.models:
        for formulation in args.formulations:
            for seed in args.seeds:
                directory = run_dir_for(args, backbone, formulation, seed)
                summary_path = directory / "summary.json"
                failure_path = directory / "failure.json"
                summary_exists = (
                    summary_path.is_file()
                    and json.loads(summary_path.read_text(encoding="utf-8")).get("protocol_version") == PROTOCOL_VERSION
                )
                failure_exists = (
                    failure_path.is_file()
                    and json.loads(failure_path.read_text(encoding="utf-8")).get("protocol_version") == PROTOCOL_VERSION
                )
                status_rows.append(
                    {
                        "backbone": backbone,
                        "formulation": formulation,
                        "initialization": INITIALIZATION,
                        "seed": seed,
                        "status": (
                            "completed" if summary_exists else "failed" if failure_exists else "missing_or_running"
                        ),
                    }
                )
    status = pd.DataFrame(status_rows)
    status.to_csv(root / "run_status.csv", index=False)
    status_counts = status["status"].value_counts().to_dict()
    print(
        "[aggregate] Requested-run status: "
        + ", ".join(f"{name}={status_counts.get(name, 0)}" for name in ("completed", "failed", "missing_or_running"))
    )

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
        row["main_bank"] = bool(group["main_bank"].iloc[0])
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
