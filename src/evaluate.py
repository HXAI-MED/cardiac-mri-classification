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

from .cinema_support import local_config, processed_dir
from .dataset import make_loaders, save_split_audit, split_metadata
from .metrics import (
    architecture_bank_metrics_from_probabilities,
    metrics_from_probabilities,
    mnms2_3d_metrics_from_probabilities,
    mnms2_sax_2d_metrics_from_probabilities,
)
from .protocol import TaskSpec, protocols
from .utils import (
    amp_dtype_and_device,
    checkpoint_root,
    cleanup_cuda,
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
        path = checkpoint_root(args) / "calibration" / spec.key / "summary.json"
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
    amp: bool = True,
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
            enabled=amp and device.type == "cuda",
        ):
            logits = model(image)
        probabilities.extend(torch.softmax(logits.float(), dim=1).cpu().numpy())
        pids.extend(str(pid) for pid in batch["pid"])
        targets.extend(batch["label"].numpy().astype(int).tolist())
    y_true = np.asarray(targets, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    return pids, y_true, probs, metrics_from_probabilities(y_true, probs)


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


# Acdc Distillation

BASELINE_PROTOCOL_VERSION = protocols["acdc_3d"]["protocol_version"]


ACDC_DISTILLATION_CLASSES = protocols["acdc"]["classes"]


ACDC_DISTILLATION_METRIC_NAMES = protocols["shared"]["metric_names"]


ACDC_DISTILLATION_PROTOCOL_VERSION = protocols["acdc_distillation"]["protocol_version"]


# Architecture Bank

ARCHITECTURE_BANK_PUBLISHED = protocols["shared"]["published"]


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
        run_dir = checkpoint_root(args) / "calibration" / spec.key / f"seed_{seed}"
        model = ConvViT.from_finetuned(
            repo_id="mathpluscode/CineMA",
            model_filename=(
                f"finetuned/classification_cvd/{spec.hf_name}/{spec.hf_name}_{seed}.safetensors"
            ),
            config_filename=f"finetuned/classification_cvd/{spec.hf_name}/config.yaml",
            cache_dir=str(cache_dir),
        )
        pids, targets, probs, metrics = evaluate_cinema(
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
        checkpoint_root(args) / "calibration" / spec.key / "probability_ensemble",
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
            checkpoint_root(args) / "calibration" / spec.key / "seed_metrics.csv"
        ),
        index=False,
    )
    save_json(
        checkpoint_root(args) / "calibration" / spec.key / "summary.json",
        summary,
    )
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
    amp: bool = True,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    model.eval()
    pids: list[str] = []
    targets: list[int] = []
    probabilities: list[np.ndarray] = []
    for batch in loader:
        image = batch[f"{spec.view}_image"].to(device, non_blocking=True)
        enabled = amp and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enabled):
            logits = model(image)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        pids.extend(str(x) for x in batch["pid"])
        targets.extend(batch["label"].numpy().astype(int).tolist())
        probabilities.extend(probs)
    y = np.asarray(targets, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    return pids, y, p, architecture_bank_metrics_from_probabilities(y, p, spec.classes)


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
    amp: bool = True,
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
            enabled=amp and device.type == "cuda",
        ):
            logits = model(image)
        probabilities.extend(torch.softmax(logits.float(), dim=1).cpu().numpy())
        pids.extend(str(pid) for pid in batch["pid"])
        targets.extend(batch["label"].numpy().astype(int).tolist())
    y_true = np.asarray(targets, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    return pids, y_true, probs, mnms2_sax_2d_metrics_from_probabilities(y_true, probs)


# Existing callers use these names for the same CineMA evaluation protocol.
architecture_bank_evaluate_cinema = evaluate_cinema
reproduction_evaluate_cinema = evaluate_cinema
architecture_bank_calibrate_task = calibrate_task
reproduction_calibrate_task = calibrate_task
architecture_bank_evaluate_cnn = evaluate_cnn
reproduction_evaluate_cnn = evaluate_cnn
