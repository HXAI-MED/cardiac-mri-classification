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

from training_code.classification.protocol import protocols
from training_code.runtime import save_json_arrays as save_json

CLASSES = protocols["mnms2_sax_2d"]["classes"]
TASK_KEY = protocols["mnms2_sax_2d"]["task_key"]


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


def save_predictions(
    output_dir: Path,
    pids: list[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = probabilities.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "pid": pids,
            "target": targets,
            "target_name": [CLASSES[index] for index in targets],
            "prediction": predictions,
            "prediction_name": [CLASSES[index] for index in predictions],
        }
    )
    for index, class_name in enumerate(CLASSES):
        frame[f"prob_{class_name}"] = probabilities[:, index]
    frame.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(metrics["confusion_matrix"], index=CLASSES, columns=CLASSES).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    serializable = {
        key: value for key, value in metrics.items() if key not in ("confusion_matrix", "classification_report")
    }
    serializable["classification_report"] = metrics["classification_report"]
    save_json(output_dir / "metrics.json", serializable)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    model.eval()
    pids: list[str] = []
    targets: list[int] = []
    probabilities: list[np.ndarray] = []
    for batch in loader:
        image = batch["sax_image"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            logits = model(image)
        probabilities.extend(torch.softmax(logits.float(), dim=1).cpu().numpy())
        pids.extend(str(pid) for pid in batch["pid"])
        targets.extend(batch["label"].numpy().astype(int).tolist())
    y_true = np.asarray(targets, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    return pids, y_true, probs, metrics_from_probabilities(y_true, probs)


def aggregate_results(args: argparse.Namespace) -> None:
    root = args.output_dir / "architecture_bank" / "2d" / TASK_KEY
    rows = []
    for summary_path in root.rglob("summary.json") if root.exists() else []:
        item = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {key: value for key, value in item.items() if key != "test"}
        row.update({f"test_{key}": value for key, value in item["test"].items()})
        rows.append(row)
    if not rows:
        print("[aggregate] No completed full runs found.")
        return
    frame = pd.DataFrame(rows).sort_values(["backbone", "formulation", "initialization", "seed"])
    frame.to_csv(root / "all_seed_results.csv", index=False)

    groups = ["backbone", "formulation", "initialization"]
    metrics = [
        "test_accuracy",
        "test_f1",
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_mcc",
        "test_roc_auc",
    ]
    summary_rows = []
    ensemble_rows = []
    for keys, group in frame.groupby(groups, dropna=False):
        output = dict(zip(groups, keys))
        output["task"] = TASK_KEY
        output["n_seeds"] = int(group["seed"].nunique())
        for metric in metrics:
            output[f"{metric}_mean"] = float(group[metric].mean())
            output[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        summary_rows.append(output)

        prediction_frames = []
        for checkpoint in group["checkpoint"]:
            prediction_path = Path(checkpoint).parent / "test" / "predictions.csv"
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            prediction_frames.append(pd.read_csv(prediction_path, dtype={"pid": str}))
        reference = prediction_frames[0]
        probability_columns = [f"prob_{class_name}" for class_name in CLASSES]
        probability_stack = []
        for prediction_frame in prediction_frames:
            if prediction_frame["pid"].tolist() != reference["pid"].tolist():
                raise RuntimeError(f"Patient-order mismatch for {keys}")
            if prediction_frame["target"].tolist() != reference["target"].tolist():
                raise RuntimeError(f"Target mismatch for {keys}")
            probability_stack.append(prediction_frame[probability_columns].to_numpy(float))
        ensemble_probabilities = np.mean(np.stack(probability_stack), axis=0)
        targets = reference["target"].to_numpy(np.int64)
        ensemble_metrics = metrics_from_probabilities(targets, ensemble_probabilities)
        ensemble_dir = root / keys[0] / keys[1] / keys[2] / "probability_ensemble"
        save_predictions(
            ensemble_dir,
            reference["pid"].tolist(),
            targets,
            ensemble_probabilities,
            ensemble_metrics,
        )
        ensemble_rows.append(
            {
                "task": TASK_KEY,
                **dict(zip(groups, keys)),
                "n_seeds": len(prediction_frames),
                **{
                    key: ensemble_metrics[key]
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

    pd.DataFrame(summary_rows).sort_values("test_mcc_mean", ascending=False).to_csv(
        root / "mean_std_summary.csv", index=False
    )
    pd.DataFrame(ensemble_rows).sort_values("mcc", ascending=False).to_csv(
        root / "probability_ensemble_results.csv", index=False
    )
    print(f"[aggregate] Results written to {root}")
