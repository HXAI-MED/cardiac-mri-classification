"""Inference, prediction export, and result aggregation."""

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
from cinema import ConvViT
from cinema.classification.train import classification_forward
from cinema.transform import get_patch_grid, patch_grid_sample
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    local_config,
    make_loaders,
    processed_dir,
    save_split_audit,
    split_metadata,
)
from .metrics import (
    architecture_bank_metrics_from_probabilities,
    metrics_from_probabilities,
    mnms2_3d_metrics_from_probabilities,
    mnms2_sax_2d_metrics_from_probabilities,
    reproduction_metrics_from_probabilities,
)
from .protocol import TASKS, TaskSpec, protocols
from .utils import (
    acdc_3d_run_dir_for,
    acdc_3d_task_root,
    acdc_distillation_task_root,
    amp_dtype_and_device,
    artifact_path,
    cleanup_cuda,
    mnms2_3d_task_root,
    output_file,
    save_json,
    save_json_arrays,
)

# Shared

CLASSES = protocols["acdc"]["classes"]


PUBLISHED = protocols["shared"]["published"]


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
            probabilities.append(
                torch.softmax(model(patch.unsqueeze(0)).float(), dim=1)
            )
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
    table.to_csv(output_file(directory / "predictions.csv"), index=False)
    pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes).to_csv(
        output_file(directory / "confusion_matrix.csv")
    )
    save_json(output_file(directory / "metrics.json"), metrics)


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
        path = (
            args.output_dir / "checkpoints" / "calibration" / spec.key / "summary.json"
        )
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
    root = args.output_dir / "checkpoints" / "official_training"
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


# Acdc 2D

ACDC_2D_CLASSES = protocols["acdc"]["classes"]


ACDC_2D_TASK_KEY = protocols["acdc_2d"]["task_key"]


def acdc_2d_save_predictions(
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
            "target_name": [ACDC_2D_CLASSES[index] for index in targets],
            "prediction": predictions,
            "prediction_name": [ACDC_2D_CLASSES[index] for index in predictions],
        }
    )
    for index, class_name in enumerate(ACDC_2D_CLASSES):
        frame[f"prob_{class_name}"] = probabilities[:, index]
    frame.to_csv(output_file(output_dir / "predictions.csv"), index=False)
    pd.DataFrame(
        metrics["confusion_matrix"], index=ACDC_2D_CLASSES, columns=ACDC_2D_CLASSES
    ).to_csv(output_file(output_dir / "confusion_matrix.csv"))
    serializable = {
        key: value
        for key, value in metrics.items()
        if key not in ("confusion_matrix", "classification_report")
    }
    serializable["classification_report"] = metrics["classification_report"]
    save_json_arrays(output_file(output_dir / "metrics.json"), serializable)


@torch.no_grad()
def acdc_2d_evaluate(
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


def acdc_2d_aggregate_results(args: argparse.Namespace) -> None:
    root = (
        args.output_dir / "checkpoints" / "architecture_bank" / "2d" / ACDC_2D_TASK_KEY
    )
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


# Acdc 3D

ACDC_3D_CLASSES = protocols["acdc"]["classes"]


ACDC_3D_DATASET = protocols["acdc_3d"]["dataset"]


INITIALIZATION = protocols["shared"]["initialization"]


MAIN_MODELS = protocols["acdc_3d"]["main_models"]


ACDC_3D_METRIC_NAMES = protocols["shared"]["metric_names"]


ACDC_3D_PROTOCOL_VERSION = protocols["acdc_3d"]["protocol_version"]


ACDC_3D_TASK_KEY = protocols["acdc_3d"]["task_key"]


def acdc_3d_save_predictions(
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
            "target_name": [ACDC_3D_CLASSES[index] for index in y_true],
            "prediction": prediction,
            "prediction_name": [ACDC_3D_CLASSES[index] for index in prediction],
        }
    )
    for index, name in enumerate(ACDC_3D_CLASSES):
        table[f"prob_{name}"] = probabilities[:, index]
    table.to_csv(output_file(directory / "predictions.csv"), index=False)
    pd.DataFrame(
        metrics["confusion_matrix"], index=ACDC_3D_CLASSES, columns=ACDC_3D_CLASSES
    ).to_csv(output_file(directory / "confusion_matrix.csv"))
    save_json(output_file(directory / "metrics.json"), metrics)


@torch.no_grad()
def acdc_3d_evaluate(
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
        probability = predict_probability(
            model, image, patch_size, device, amp_dtype, amp
        )
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


# Acdc Distillation

BASELINE_PROTOCOL_VERSION = protocols["acdc_3d"]["protocol_version"]


ACDC_DISTILLATION_CLASSES = protocols["acdc"]["classes"]


ACDC_DISTILLATION_METRIC_NAMES = protocols["shared"]["metric_names"]


ACDC_DISTILLATION_PROTOCOL_VERSION = protocols["acdc_distillation"]["protocol_version"]


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


# Architecture Bank

ARCHITECTURE_BANK_PUBLISHED = protocols["shared"]["published"]


@torch.no_grad()
def architecture_bank_evaluate_cinema(
    model: nn.Module,
    loader: DataLoader,
    spec: TaskSpec,
    config: DictConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    model.eval()
    model.to(device)
    patch_size = (
        config.data.sax.patch_size if spec.view == "sax" else config.data.lax.patch_size
    )
    patch_size_dict = {spec.view: tuple(patch_size)}
    pids: list[str] = []
    targets: list[int] = []
    probabilities: list[np.ndarray] = []
    for batch in tqdm(loader, desc=f"evaluate {spec.key}", leave=False):
        image_dict = {
            spec.view: batch[f"{spec.view}_image"].to(device, non_blocking=True)
        }
        logits = classification_forward(model, image_dict, patch_size_dict, amp_dtype)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        pids.extend(str(x) for x in batch["pid"])
        targets.extend(batch["label"].numpy().astype(int).tolist())
        probabilities.extend(probs)
    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    metrics = architecture_bank_metrics_from_probabilities(
        target_array, probability_array, spec.classes
    )
    return pids, target_array, probability_array, metrics


def architecture_bank_calibrate_task(
    args: argparse.Namespace, spec: TaskSpec
) -> dict[str, Any]:
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
        run_dir = (
            args.output_dir / "checkpoints" / "calibration" / spec.key / f"seed_{seed}"
        )
        model = ConvViT.from_finetuned(
            repo_id="mathpluscode/CineMA",
            model_filename=(
                f"finetuned/classification_cvd/{spec.hf_name}/{spec.hf_name}_{seed}.safetensors"
            ),
            config_filename=f"finetuned/classification_cvd/{spec.hf_name}/config.yaml",
            cache_dir=str(cache_dir),
        )
        pids, targets, probs, metrics = architecture_bank_evaluate_cinema(
            model, test_loader, spec, config, device, amp_dtype
        )
        save_predictions(run_dir, pids, targets, probs, spec.classes, metrics)
        rows.append(
            {
                "task": spec.key,
                "seed": seed,
                **{
                    k: metrics[k]
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
        )
        if seed_targets is not None and not np.array_equal(seed_targets, targets):
            raise RuntimeError(f"Target mismatch across released seeds for {spec.key}")
        if seed_pids is not None and seed_pids != pids:
            raise RuntimeError(
                f"Patient-order mismatch across released seeds for {spec.key}"
            )
        seed_targets, seed_pids = targets, pids
        seed_predictions.append(probs)
        del model
        cleanup_cuda()

    frame = pd.DataFrame(rows)
    summary = {
        "task": spec.key,
        "n_seeds": len(rows),
        "mean": {
            column: float(frame[column].mean())
            for column in frame.columns
            if column not in ("task", "seed")
        },
        "std": {
            column: float(frame[column].std(ddof=1)) if len(frame) > 1 else 0.0
            for column in frame.columns
            if column not in ("task", "seed")
        },
        "published": ARCHITECTURE_BANK_PUBLISHED[spec.key]["cinema_finetune"],
    }
    summary["absolute_error"] = {
        "roc_auc": abs(summary["mean"]["roc_auc"] - summary["published"]["roc_auc"]),
        "f1": abs(summary["mean"]["f1"] - summary["published"]["f1"]),
    }
    summary["passed"] = all(
        error <= args.calibration_tolerance
        for error in summary["absolute_error"].values()
    )

    ensemble_probs = np.mean(np.stack(seed_predictions, axis=0), axis=0)
    ensemble_metrics = architecture_bank_metrics_from_probabilities(
        seed_targets, ensemble_probs, spec.classes
    )
    save_predictions(
        args.output_dir
        / "checkpoints"
        / "calibration"
        / spec.key
        / "probability_ensemble",
        seed_pids,
        seed_targets,
        ensemble_probs,
        spec.classes,
        ensemble_metrics,
    )
    summary["ensemble"] = {
        k: ensemble_metrics[k]
        for k in ("accuracy", "f1", "macro_f1", "balanced_accuracy", "mcc", "roc_auc")
    }
    frame.to_csv(
        output_file(
            args.output_dir
            / "checkpoints"
            / "calibration"
            / spec.key
            / "seed_metrics.csv"
        ),
        index=False,
    )
    save_json(
        args.output_dir / "checkpoints" / "calibration" / spec.key / "summary.json",
        summary,
    )
    print(
        f"[calibration] {spec.key}: mean AUROC={summary['mean']['roc_auc']:.4f} "
        f"(published {summary['published']['roc_auc']:.4f}), mean F1/accuracy={summary['mean']['f1']:.4f} "
        f"(published {summary['published']['f1']:.4f}) -> {'PASS' if summary['passed'] else 'FAIL'}"
    )
    return summary


@torch.no_grad()
def architecture_bank_evaluate_cnn(
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
    return pids, y, p, architecture_bank_metrics_from_probabilities(y, p, spec.classes)


def aggregate_architecture_results(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    root = args.output_dir / "checkpoints" / "architecture_bank"
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


# Mnms2 3D

MNMS2_3D_CLASSES = protocols["mnms2_3d"]["classes"]


MNMS2_3D_DATASET = protocols["mnms2_3d"]["dataset"]


MNMS2_3D_METRIC_NAMES = protocols["shared"]["metric_names"]


MNMS2_3D_TASK_KEY = protocols["mnms2_3d"]["task_key"]


def mnms2_3d_save_predictions(
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
            "target_name": [MNMS2_3D_CLASSES[index] for index in y_true],
            "prediction": prediction,
            "prediction_name": [MNMS2_3D_CLASSES[index] for index in prediction],
        }
    )
    for index, name in enumerate(MNMS2_3D_CLASSES):
        table[f"prob_{name}"] = probabilities[:, index]
    table.to_csv(output_file(directory / "predictions.csv"), index=False)
    pd.DataFrame(
        metrics["confusion_matrix"], index=MNMS2_3D_CLASSES, columns=MNMS2_3D_CLASSES
    ).to_csv(output_file(directory / "confusion_matrix.csv"))
    save_json(output_file(directory / "metrics.json"), metrics)


@torch.no_grad()
def mnms2_3d_evaluate(
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
        probability = predict_probability(
            model, image, patch_size, device, amp_dtype, amp
        )
        pids.extend(str(pid) for pid in batch["pid"])
        targets.extend(batch["label"].numpy().astype(int).tolist())
        probabilities.extend(probability.cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    metrics = mnms2_3d_metrics_from_probabilities(target_array, probability_array)
    return pids, target_array, probability_array, metrics, elapsed


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


# Mnms2 Sax 2D

MNMS2_SAX_2D_CLASSES = protocols["mnms2_sax_2d"]["classes"]


MNMS2_SAX_2D_TASK_KEY = protocols["mnms2_sax_2d"]["task_key"]


def mnms2_sax_2d_save_predictions(
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
            "target_name": [MNMS2_SAX_2D_CLASSES[index] for index in targets],
            "prediction": predictions,
            "prediction_name": [MNMS2_SAX_2D_CLASSES[index] for index in predictions],
        }
    )
    for index, class_name in enumerate(MNMS2_SAX_2D_CLASSES):
        frame[f"prob_{class_name}"] = probabilities[:, index]
    frame.to_csv(output_file(output_dir / "predictions.csv"), index=False)
    pd.DataFrame(
        metrics["confusion_matrix"],
        index=MNMS2_SAX_2D_CLASSES,
        columns=MNMS2_SAX_2D_CLASSES,
    ).to_csv(output_file(output_dir / "confusion_matrix.csv"))
    serializable = {
        key: value
        for key, value in metrics.items()
        if key not in ("confusion_matrix", "classification_report")
    }
    serializable["classification_report"] = metrics["classification_report"]
    save_json_arrays(output_file(output_dir / "metrics.json"), serializable)


@torch.no_grad()
def mnms2_sax_2d_evaluate(
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
    return pids, y_true, probs, mnms2_sax_2d_metrics_from_probabilities(y_true, probs)


def mnms2_sax_2d_aggregate_results(args: argparse.Namespace) -> None:
    root = (
        args.output_dir
        / "checkpoints"
        / "architecture_bank"
        / "2d"
        / MNMS2_SAX_2D_TASK_KEY
    )
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


# Reproduction

REPRODUCTION_PUBLISHED = protocols["shared"]["published"]


@torch.no_grad()
def reproduction_evaluate_cinema(
    model: nn.Module,
    loader: DataLoader,
    spec: TaskSpec,
    config: DictConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    model.eval()
    model.to(device)
    patch_size = (
        config.data.sax.patch_size if spec.view == "sax" else config.data.lax.patch_size
    )
    patch_size_dict = {spec.view: tuple(patch_size)}
    pids: list[str] = []
    targets: list[int] = []
    probabilities: list[np.ndarray] = []
    for batch in tqdm(loader, desc=f"evaluate {spec.key}", leave=False):
        image_dict = {
            spec.view: batch[f"{spec.view}_image"].to(device, non_blocking=True)
        }
        logits = classification_forward(model, image_dict, patch_size_dict, amp_dtype)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        pids.extend(str(x) for x in batch["pid"])
        targets.extend(batch["label"].numpy().astype(int).tolist())
        probabilities.extend(probs)
    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    metrics = reproduction_metrics_from_probabilities(
        target_array, probability_array, spec.classes
    )
    return pids, target_array, probability_array, metrics


def reproduction_calibrate_task(
    args: argparse.Namespace, spec: TaskSpec
) -> dict[str, Any]:
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
        run_dir = (
            args.output_dir / "checkpoints" / "calibration" / spec.key / f"seed_{seed}"
        )
        model = ConvViT.from_finetuned(
            repo_id="mathpluscode/CineMA",
            model_filename=(
                f"finetuned/classification_cvd/{spec.hf_name}/{spec.hf_name}_{seed}.safetensors"
            ),
            config_filename=f"finetuned/classification_cvd/{spec.hf_name}/config.yaml",
            cache_dir=str(cache_dir),
        )
        pids, targets, probs, metrics = reproduction_evaluate_cinema(
            model, test_loader, spec, config, device, amp_dtype
        )
        save_predictions(run_dir, pids, targets, probs, spec.classes, metrics)
        rows.append(
            {
                "task": spec.key,
                "seed": seed,
                **{
                    k: metrics[k]
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
        )
        if seed_targets is not None and not np.array_equal(seed_targets, targets):
            raise RuntimeError(f"Target mismatch across released seeds for {spec.key}")
        if seed_pids is not None and seed_pids != pids:
            raise RuntimeError(
                f"Patient-order mismatch across released seeds for {spec.key}"
            )
        seed_targets, seed_pids = targets, pids
        seed_predictions.append(probs)
        del model
        cleanup_cuda()

    frame = pd.DataFrame(rows)
    summary = {
        "task": spec.key,
        "n_seeds": len(rows),
        "mean": {
            column: float(frame[column].mean())
            for column in frame.columns
            if column not in ("task", "seed")
        },
        "std": {
            column: float(frame[column].std(ddof=1)) if len(frame) > 1 else 0.0
            for column in frame.columns
            if column not in ("task", "seed")
        },
        "published": REPRODUCTION_PUBLISHED[spec.key]["cinema_finetune"],
    }
    summary["absolute_error"] = {
        "roc_auc": abs(summary["mean"]["roc_auc"] - summary["published"]["roc_auc"]),
        "f1": abs(summary["mean"]["f1"] - summary["published"]["f1"]),
    }
    summary["passed"] = all(
        error <= args.calibration_tolerance
        for error in summary["absolute_error"].values()
    )

    ensemble_probs = np.mean(np.stack(seed_predictions, axis=0), axis=0)
    ensemble_metrics = reproduction_metrics_from_probabilities(
        seed_targets, ensemble_probs, spec.classes
    )
    save_predictions(
        args.output_dir
        / "checkpoints"
        / "calibration"
        / spec.key
        / "probability_ensemble",
        seed_pids,
        seed_targets,
        ensemble_probs,
        spec.classes,
        ensemble_metrics,
    )
    summary["ensemble"] = {
        k: ensemble_metrics[k]
        for k in ("accuracy", "f1", "macro_f1", "balanced_accuracy", "mcc", "roc_auc")
    }
    frame.to_csv(
        output_file(
            args.output_dir
            / "checkpoints"
            / "calibration"
            / spec.key
            / "seed_metrics.csv"
        ),
        index=False,
    )
    save_json(
        args.output_dir / "checkpoints" / "calibration" / spec.key / "summary.json",
        summary,
    )
    print(
        f"[calibration] {spec.key}: mean AUROC={summary['mean']['roc_auc']:.4f} "
        f"(published {summary['published']['roc_auc']:.4f}), mean F1/accuracy={summary['mean']['f1']:.4f} "
        f"(published {summary['published']['f1']:.4f}) -> {'PASS' if summary['passed'] else 'FAIL'}"
    )
    return summary


@torch.no_grad()
def reproduction_evaluate_cnn(
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
    return pids, y, p, reproduction_metrics_from_probabilities(y, p, spec.classes)


def aggregate_cnn_results(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    root = args.output_dir / "checkpoints" / "cnn_extensions"
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
