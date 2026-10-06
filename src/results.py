"""Aggregate completed seeds and export comparison/ensemble tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import (
    acdc_2d_save_predictions,
    acdc_3d_save_predictions,
    mnms2_3d_save_predictions,
    mnms2_sax_2d_save_predictions,
    save_predictions,
)
from .metrics import (
    architecture_bank_metrics_from_probabilities,
    metrics_from_probabilities,
    mnms2_3d_metrics_from_probabilities,
    mnms2_sax_2d_metrics_from_probabilities,
    reproduction_metrics_from_probabilities,
)
from .protocol import TASKS, protocols
from .utils import (
    acdc_3d_run_dir_for,
    acdc_3d_task_root,
    artifact_path,
    checkpoint_root,
    mnms2_3d_task_root,
    output_file,
)

PUBLISHED = protocols["shared"]["published"]


ACDC_2D_CLASSES = protocols["acdc"]["classes"]


ACDC_2D_TASK_KEY = protocols["acdc_2d"]["task_key"]


ACDC_3D_CLASSES = protocols["acdc"]["classes"]


ACDC_3D_DATASET = protocols["acdc_3d"]["dataset"]


INITIALIZATION = protocols["shared"]["initialization"]


MAIN_MODELS = protocols["acdc_3d"]["main_models"]


ACDC_3D_METRIC_NAMES = protocols["shared"]["metric_names"]


ACDC_3D_PROTOCOL_VERSION = protocols["acdc_3d"]["protocol_version"]


ACDC_3D_TASK_KEY = protocols["acdc_3d"]["task_key"]


MNMS2_3D_CLASSES = protocols["mnms2_3d"]["classes"]


MNMS2_3D_DATASET = protocols["mnms2_3d"]["dataset"]


MNMS2_3D_METRIC_NAMES = protocols["shared"]["metric_names"]


MNMS2_3D_TASK_KEY = protocols["mnms2_3d"]["task_key"]


MNMS2_SAX_2D_CLASSES = protocols["mnms2_sax_2d"]["classes"]


MNMS2_SAX_2D_TASK_KEY = protocols["mnms2_sax_2d"]["task_key"]


def aggregate_official_results(args: argparse.Namespace) -> None:
    root = checkpoint_root(args) / "official_training"
    rows: list[dict[str, Any]] = []
    for path in root.rglob("DONE.json") if root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        row = {key: value for key, value in item.items() if key != "test"}
        row.update({f"test_{key}": value for key, value in item["test"].items()})
        rows.append(row)
    if not rows:
        return

    frame = pd.DataFrame(rows)
    frame.to_csv(output_file(root / "all_seed_results.csv"), index=False)
    groups = ["task", "mode", "smoke"]
    metric_columns = [column for column in frame.columns if column.startswith("test_")]
    summary_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(groups, dropna=False):
        row = dict(zip(groups, keys))
        row["n_seeds"] = len(group)
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            )
        reference = PUBLISHED.get(str(row["task"]), {}).get(str(row["mode"]))
        if reference and not bool(row["smoke"]):
            row["published_roc_auc"] = reference["roc_auc"]
            row["published_f1"] = reference["f1"]
            row["roc_auc_absolute_error"] = abs(
                row["test_roc_auc_mean"] - reference["roc_auc"]
            )
            row["f1_absolute_error"] = abs(row["test_f1_mean"] - reference["f1"])
        summary_rows.append(row)
    pd.DataFrame(summary_rows).sort_values(["task", "mode"]).to_csv(
        output_file(root / "mean_std_vs_published.csv"), index=False
    )


def acdc_2d_aggregate_results(args: argparse.Namespace) -> None:
    root = checkpoint_root(args) / "architecture_bank" / "2d" / ACDC_2D_TASK_KEY
    rows = []
    for summary_path in root.rglob("summary.json") if root.exists() else []:
        item = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {key: value for key, value in item.items() if key != "test"}
        row.update({f"test_{key}": value for key, value in item["test"].items()})
        rows.append(row)
    if not rows:
        print("[aggregate] No completed full runs found.")
        return
    frame = pd.DataFrame(rows).sort_values(
        ["backbone", "formulation", "initialization", "seed"]
    )
    frame.to_csv(output_file(root / "all_seed_results.csv"), index=False)

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
        output["task"] = ACDC_2D_TASK_KEY
        output["n_seeds"] = int(group["seed"].nunique())
        for metric in metrics:
            output[f"{metric}_mean"] = float(group[metric].mean())
            output[f"{metric}_std"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            )
        summary_rows.append(output)

        prediction_frames = []
        for checkpoint in group["checkpoint"]:
            prediction_path = artifact_path(
                Path(checkpoint).parent / "test" / "predictions.csv"
            )
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            prediction_frames.append(pd.read_csv(prediction_path, dtype={"pid": str}))
        reference = prediction_frames[0]
        probability_columns = [f"prob_{class_name}" for class_name in ACDC_2D_CLASSES]
        probability_stack = []
        for prediction_frame in prediction_frames:
            if prediction_frame["pid"].tolist() != reference["pid"].tolist():
                raise RuntimeError(f"Patient-order mismatch for {keys}")
            if prediction_frame["target"].tolist() != reference["target"].tolist():
                raise RuntimeError(f"Target mismatch for {keys}")
            probability_stack.append(
                prediction_frame[probability_columns].to_numpy(float)
            )
        ensemble_probabilities = np.mean(np.stack(probability_stack), axis=0)
        targets = reference["target"].to_numpy(np.int64)
        ensemble_metrics = metrics_from_probabilities(targets, ensemble_probabilities)
        ensemble_dir = root / keys[0] / keys[1] / keys[2] / "probability_ensemble"
        acdc_2d_save_predictions(
            ensemble_dir,
            reference["pid"].tolist(),
            targets,
            ensemble_probabilities,
            ensemble_metrics,
        )
        ensemble_rows.append(
            {
                "task": ACDC_2D_TASK_KEY,
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
        output_file(root / "mean_std_summary.csv"), index=False
    )
    pd.DataFrame(ensemble_rows).sort_values("mcc", ascending=False).to_csv(
        output_file(root / "probability_ensemble_results.csv"), index=False
    )
    print(f"[aggregate] Results written to {root}")


def acdc_3d_aggregate_results(args: argparse.Namespace) -> None:
    root = acdc_3d_task_root(args, smoke=False)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in root.rglob("summary.json") if root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        if (
            item.get("task") != ACDC_3D_TASK_KEY
            or item.get("protocol_version") != ACDC_3D_PROTOCOL_VERSION
        ):
            continue
        row = {key: value for key, value in item.items() if key != "test"}
        row.update({f"test_{key}": value for key, value in item["test"].items()})
        row["main_bank"] = item.get("backbone") in MAIN_MODELS
        rows.append(row)

    failures: list[dict[str, Any]] = []
    for path in root.rglob("failure.json") if root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("protocol_version") != ACDC_3D_PROTOCOL_VERSION:
            continue
        failures.append(
            {key: value for key, value in item.items() if key != "traceback"}
        )
    if failures:
        pd.DataFrame(failures).to_csv(output_file(root / "failures.csv"), index=False)

    status_rows: list[dict[str, Any]] = []
    for backbone in args.models:
        for formulation in args.formulations:
            for seed in args.seeds:
                directory = acdc_3d_run_dir_for(args, backbone, formulation, seed)
                summary_path = directory / "summary.json"
                failure_path = directory / "failure.json"
                summary_exists = (
                    summary_path.is_file()
                    and json.loads(summary_path.read_text(encoding="utf-8")).get(
                        "protocol_version"
                    )
                    == ACDC_3D_PROTOCOL_VERSION
                )
                failure_exists = (
                    failure_path.is_file()
                    and json.loads(failure_path.read_text(encoding="utf-8")).get(
                        "protocol_version"
                    )
                    == ACDC_3D_PROTOCOL_VERSION
                )
                status_rows.append(
                    {
                        "backbone": backbone,
                        "formulation": formulation,
                        "initialization": INITIALIZATION,
                        "seed": seed,
                        "status": (
                            "completed"
                            if summary_exists
                            else "failed"
                            if failure_exists
                            else "missing_or_running"
                        ),
                    }
                )
    status = pd.DataFrame(status_rows)
    status.to_csv(output_file(root / "run_status.csv"), index=False)
    status_counts = status["status"].value_counts().to_dict()
    print(
        "[aggregate] Requested-run status: "
        + ", ".join(
            f"{name}={status_counts.get(name, 0)}"
            for name in ("completed", "failed", "missing_or_running")
        )
    )

    if not rows:
        print(f"[aggregate] No completed runs under {root}")
        return
    frame = pd.DataFrame(rows).sort_values(
        ["test_mcc", "test_roc_auc"], ascending=False
    )
    frame.to_csv(output_file(root / "all_seed_results.csv"), index=False)

    group_columns = ["backbone", "formulation", "initialization"]
    aggregate_columns = [
        *(f"test_{name}" for name in ACDC_3D_METRIC_NAMES),
        "training_seconds",
        "test_inference_ms_per_patient",
        "peak_gpu_memory_gb",
    ]
    summary_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "dataset": ACDC_3D_DATASET,
                "task": ACDC_3D_TASK_KEY,
                "n_seeds": len(group),
            }
        )
        row["main_bank"] = bool(group["main_bank"].iloc[0])
        row["complete_seed_set"] = set(group["seed"].astype(int)) == set(args.seeds)
        row["parameter_count"] = int(group["parameter_count"].iloc[0])
        for column in aggregate_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = (
                float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("test_mcc_mean", ascending=False)
    summary.to_csv(output_file(root / "mean_std_summary.csv"), index=False)

    ensemble_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if group["seed"].duplicated().any():
            raise RuntimeError(f"Duplicate seed while ensembling {keys}")
        prediction_frames: list[pd.DataFrame] = []
        for checkpoint in group["checkpoint"]:
            prediction_path = artifact_path(
                Path(checkpoint).parent / "test" / "predictions.csv"
            )
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            prediction_frames.append(pd.read_csv(prediction_path, dtype={"pid": str}))
        reference = prediction_frames[0]
        probability_columns = [f"prob_{name}" for name in ACDC_3D_CLASSES]
        seed_probabilities: list[np.ndarray] = []
        for prediction_frame in prediction_frames:
            if prediction_frame["pid"].tolist() != reference["pid"].tolist():
                raise RuntimeError(f"Patient-order mismatch while ensembling {keys}")
            if prediction_frame["target"].tolist() != reference["target"].tolist():
                raise RuntimeError(f"Target mismatch while ensembling {keys}")
            seed_probabilities.append(
                prediction_frame[probability_columns].to_numpy(dtype=np.float64)
            )
        probabilities = np.mean(np.stack(seed_probabilities, axis=0), axis=0)
        targets = reference["target"].to_numpy(dtype=np.int64)
        metrics = metrics_from_probabilities(targets, probabilities)
        backbone, formulation, initialization = keys
        ensemble_directory = (
            root
            / str(backbone)
            / str(formulation)
            / str(initialization)
            / "probability_ensemble"
        )
        acdc_3d_save_predictions(
            ensemble_directory,
            reference["pid"].tolist(),
            targets,
            probabilities,
            metrics,
        )
        ensemble_rows.append(
            {
                "dataset": ACDC_3D_DATASET,
                "task": ACDC_3D_TASK_KEY,
                "backbone": backbone,
                "formulation": formulation,
                "initialization": initialization,
                "n_seeds": len(group),
                "complete_seed_set": set(group["seed"].astype(int)) == set(args.seeds),
                **{name: metrics[name] for name in ACDC_3D_METRIC_NAMES},
            }
        )
    pd.DataFrame(ensemble_rows).sort_values("mcc", ascending=False).to_csv(
        output_file(root / "probability_ensemble_results.csv"), index=False
    )
    print(f"[aggregate] Wrote results for {len(frame)} completed runs to {root}")


def aggregate_architecture_results(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    root = checkpoint_root(args) / "architecture_bank"
    for path in root.rglob("summary.json") if root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        row = {k: v for k, v in item.items() if k != "test"}
        row.update({f"test_{k}": v for k, v in item["test"].items()})
        rows.append(row)
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(output_file(root / "all_seed_results.csv"), index=False)
    groups = ["dimensionality", "task", "backbone", "formulation", "initialization"]
    metrics = [
        "test_accuracy",
        "test_f1",
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_mcc",
        "test_roc_auc",
    ]
    summary_rows = []
    for keys, group in frame.groupby(groups, dropna=False):
        row = dict(zip(groups, keys))
        row["n_seeds"] = len(group)
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        "test_roc_auc_mean", ascending=False
    )
    summary.to_csv(output_file(root / "mean_std_summary.csv"), index=False)

    ensemble_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(groups, dropna=False):
        dimensionality, task, backbone, formulation, initialization = keys
        spec = TASKS[str(task)]
        prediction_frames: list[pd.DataFrame] = []
        for checkpoint in group["checkpoint"]:
            prediction_path = artifact_path(
                Path(checkpoint).parent / "test" / "predictions.csv"
            )
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
            seed_probabilities.append(
                prediction_frame[probability_columns].to_numpy(dtype=np.float64)
            )

        ensemble_probabilities = np.mean(np.stack(seed_probabilities, axis=0), axis=0)
        targets = reference["target"].to_numpy(dtype=np.int64)
        ensemble_metrics = architecture_bank_metrics_from_probabilities(
            targets, ensemble_probabilities, spec.classes
        )
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
    pd.DataFrame(ensemble_rows).sort_values("roc_auc", ascending=False).to_csv(
        output_file(root / "probability_ensemble_results.csv"), index=False
    )


def mnms2_3d_aggregate_results(args: argparse.Namespace) -> None:
    root = mnms2_3d_task_root(args, smoke=False)
    rows: list[dict[str, Any]] = []
    for path in root.rglob("summary.json") if root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("task") != MNMS2_3D_TASK_KEY:
            continue
        row = {key: value for key, value in item.items() if key != "test"}
        row.update({f"test_{key}": value for key, value in item["test"].items()})
        rows.append(row)

    failures: list[dict[str, Any]] = []
    for path in root.rglob("failure.json") if root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        failures.append(
            {key: value for key, value in item.items() if key != "traceback"}
        )
    if failures:
        pd.DataFrame(failures).to_csv(output_file(root / "failures.csv"), index=False)

    if not rows:
        print(f"[aggregate] No completed runs under {root}")
        return
    frame = pd.DataFrame(rows).sort_values(
        ["test_mcc", "test_roc_auc"], ascending=False
    )
    frame.to_csv(output_file(root / "all_seed_results.csv"), index=False)

    group_columns = ["backbone", "formulation", "initialization"]
    aggregate_columns = [
        *(f"test_{name}" for name in MNMS2_3D_METRIC_NAMES),
        "training_seconds",
        "test_inference_ms_per_patient",
        "peak_gpu_memory_gb",
    ]
    summary_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "dataset": MNMS2_3D_DATASET,
                "task": MNMS2_3D_TASK_KEY,
                "n_seeds": len(group),
            }
        )
        row["complete_seed_set"] = set(group["seed"].astype(int)) == set(args.seeds)
        row["parameter_count"] = int(group["parameter_count"].iloc[0])
        for column in aggregate_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = (
                float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("test_mcc_mean", ascending=False)
    summary.to_csv(output_file(root / "mean_std_summary.csv"), index=False)

    ensemble_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if group["seed"].duplicated().any():
            raise RuntimeError(f"Duplicate seed while ensembling {keys}")
        prediction_frames: list[pd.DataFrame] = []
        for checkpoint in group["checkpoint"]:
            prediction_path = artifact_path(
                Path(checkpoint).parent / "test" / "predictions.csv"
            )
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            prediction_frames.append(pd.read_csv(prediction_path, dtype={"pid": str}))
        reference = prediction_frames[0]
        probability_columns = [f"prob_{name}" for name in MNMS2_3D_CLASSES]
        seed_probabilities: list[np.ndarray] = []
        for prediction_frame in prediction_frames:
            if prediction_frame["pid"].tolist() != reference["pid"].tolist():
                raise RuntimeError(f"Patient-order mismatch while ensembling {keys}")
            if prediction_frame["target"].tolist() != reference["target"].tolist():
                raise RuntimeError(f"Target mismatch while ensembling {keys}")
            seed_probabilities.append(
                prediction_frame[probability_columns].to_numpy(dtype=np.float64)
            )
        probabilities = np.mean(np.stack(seed_probabilities, axis=0), axis=0)
        targets = reference["target"].to_numpy(dtype=np.int64)
        metrics = mnms2_3d_metrics_from_probabilities(targets, probabilities)
        backbone, formulation, initialization = keys
        ensemble_directory = (
            root
            / str(backbone)
            / str(formulation)
            / str(initialization)
            / "probability_ensemble"
        )
        mnms2_3d_save_predictions(
            ensemble_directory,
            reference["pid"].tolist(),
            targets,
            probabilities,
            metrics,
        )
        ensemble_rows.append(
            {
                "dataset": MNMS2_3D_DATASET,
                "task": MNMS2_3D_TASK_KEY,
                "backbone": backbone,
                "formulation": formulation,
                "initialization": initialization,
                "n_seeds": len(group),
                "complete_seed_set": set(group["seed"].astype(int)) == set(args.seeds),
                **{name: metrics[name] for name in MNMS2_3D_METRIC_NAMES},
            }
        )
    pd.DataFrame(ensemble_rows).sort_values("mcc", ascending=False).to_csv(
        output_file(root / "probability_ensemble_results.csv"), index=False
    )
    print(f"[aggregate] Wrote results for {len(frame)} completed runs to {root}")


def mnms2_sax_2d_aggregate_results(args: argparse.Namespace) -> None:
    root = checkpoint_root(args) / "architecture_bank" / "2d" / MNMS2_SAX_2D_TASK_KEY
    rows = []
    for summary_path in root.rglob("summary.json") if root.exists() else []:
        item = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {key: value for key, value in item.items() if key != "test"}
        row.update({f"test_{key}": value for key, value in item["test"].items()})
        rows.append(row)
    if not rows:
        print("[aggregate] No completed full runs found.")
        return
    frame = pd.DataFrame(rows).sort_values(
        ["backbone", "formulation", "initialization", "seed"]
    )
    frame.to_csv(output_file(root / "all_seed_results.csv"), index=False)

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
        output["task"] = MNMS2_SAX_2D_TASK_KEY
        output["n_seeds"] = int(group["seed"].nunique())
        for metric in metrics:
            output[f"{metric}_mean"] = float(group[metric].mean())
            output[f"{metric}_std"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            )
        summary_rows.append(output)

        prediction_frames = []
        for checkpoint in group["checkpoint"]:
            prediction_path = artifact_path(
                Path(checkpoint).parent / "test" / "predictions.csv"
            )
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            prediction_frames.append(pd.read_csv(prediction_path, dtype={"pid": str}))
        reference = prediction_frames[0]
        probability_columns = [
            f"prob_{class_name}" for class_name in MNMS2_SAX_2D_CLASSES
        ]
        probability_stack = []
        for prediction_frame in prediction_frames:
            if prediction_frame["pid"].tolist() != reference["pid"].tolist():
                raise RuntimeError(f"Patient-order mismatch for {keys}")
            if prediction_frame["target"].tolist() != reference["target"].tolist():
                raise RuntimeError(f"Target mismatch for {keys}")
            probability_stack.append(
                prediction_frame[probability_columns].to_numpy(float)
            )
        ensemble_probabilities = np.mean(np.stack(probability_stack), axis=0)
        targets = reference["target"].to_numpy(np.int64)
        ensemble_metrics = mnms2_sax_2d_metrics_from_probabilities(
            targets, ensemble_probabilities
        )
        ensemble_dir = root / keys[0] / keys[1] / keys[2] / "probability_ensemble"
        mnms2_sax_2d_save_predictions(
            ensemble_dir,
            reference["pid"].tolist(),
            targets,
            ensemble_probabilities,
            ensemble_metrics,
        )
        ensemble_rows.append(
            {
                "task": MNMS2_SAX_2D_TASK_KEY,
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
        output_file(root / "mean_std_summary.csv"), index=False
    )
    pd.DataFrame(ensemble_rows).sort_values("mcc", ascending=False).to_csv(
        output_file(root / "probability_ensemble_results.csv"), index=False
    )
    print(f"[aggregate] Results written to {root}")


def aggregate_cnn_results(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    root = checkpoint_root(args) / "cnn_extensions"
    for path in root.rglob("summary.json") if root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        row = {k: v for k, v in item.items() if k != "test"}
        row.update({f"test_{k}": v for k, v in item["test"].items()})
        rows.append(row)
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(output_file(root / "all_seed_results.csv"), index=False)
    groups = ["task", "backbone", "formulation", "initialization"]
    metrics = [
        "test_accuracy",
        "test_f1",
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_mcc",
        "test_roc_auc",
    ]
    summary_rows = []
    for keys, group in frame.groupby(groups, dropna=False):
        row = dict(zip(groups, keys))
        row["n_seeds"] = len(group)
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        "test_roc_auc_mean", ascending=False
    )
    summary.to_csv(output_file(root / "mean_std_summary.csv"), index=False)

    ensemble_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(groups, dropna=False):
        task, backbone, formulation, initialization = keys
        spec = TASKS[str(task)]
        prediction_frames: list[pd.DataFrame] = []
        for checkpoint in group["checkpoint"]:
            prediction_path = artifact_path(
                Path(checkpoint).parent / "test" / "predictions.csv"
            )
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
            seed_probabilities.append(
                prediction_frame[probability_columns].to_numpy(dtype=np.float64)
            )

        ensemble_probabilities = np.mean(np.stack(seed_probabilities, axis=0), axis=0)
        targets = reference["target"].to_numpy(dtype=np.int64)
        ensemble_metrics = reproduction_metrics_from_probabilities(
            targets, ensemble_probabilities, spec.classes
        )
        ensemble_dir = (
            root
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
                "task": task,
                "backbone": backbone,
                "formulation": formulation,
                "initialization": initialization,
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
    pd.DataFrame(ensemble_rows).sort_values("roc_auc", ascending=False).to_csv(
        output_file(root / "probability_ensemble_results.csv"), index=False
    )
