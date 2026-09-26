"""Classification training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

import hydra
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from cinema.optim import adjust_learning_rate, get_n_accum_steps
from training_code.classification.architecture_bank.eval import (
    aggregate_architecture_results,
    calibrate_task,
    evaluate_cnn,
)
from training_code.classification.architecture_bank.model import (
    EDES2DClassifier,
    EDES3DClassifier,
    feature_layer_candidates,
)
from training_code.classification.architecture_bank.utils import apply_architecture_overrides
from training_code.classification.dataset import (
    local_config,
    make_loaders,
    preprocess_dataset,
    processed_dir,
    save_split_audit,
    split_metadata,
)
from training_code.classification.eval import aggregate_official_results, calibration_is_valid, save_predictions
from training_code.classification.protocol import TASKS, TaskSpec, protocols, resolve_tasks
from training_code.classification.train import run_official_training
from training_code.classification.utils import prepare_run
from training_code.runtime import amp_dtype_and_device, cinema_git_commit, cleanup_cuda, grad_scaler, save_json

MODELS_2D = protocols["architecture_bank"]["models_2d"]
MODELS_3D = protocols["architecture_bank"]["models_3d"]


def train_architecture(
    args: argparse.Namespace,
    spec: TaskSpec,
    backbone: str,
    formulation: str,
    initialization: str,
    seed: int,
) -> dict[str, Any]:
    dimensionality = spec.dimensionality
    if dimensionality == 2 and backbone not in MODELS_2D:
        raise ValueError(f"{backbone} is not a registered 2-D architecture")
    if dimensionality == 3 and backbone not in MODELS_3D:
        raise ValueError(f"{backbone} is not a registered 3-D architecture")
    if dimensionality == 3 and initialization != "randinit":
        raise ValueError("The native 3-D bank currently supports randinit only")

    output_group = "architecture_smoke" if args.smoke else "architecture_bank"
    run_dir = (
        args.output_dir
        / output_group
        / f"{dimensionality}d"
        / spec.key
        / backbone
        / formulation
        / initialization
        / f"seed_{seed}"
    )
    summary_path = run_dir / "summary.json"
    if args.resume and summary_path.exists():
        print(f"[resume] {dimensionality}d/{spec.key}/{backbone}/{formulation}/{initialization}/seed_{seed}")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)

    data_root = processed_dir(args, spec.dataset)
    splits = split_metadata(spec, data_root)
    config = local_config(spec, data_root, seed)
    apply_architecture_overrides(args, config, dimensionality)
    train_loader, val_loader, test_loader = make_loaders(spec, config, data_root, splits, seed)
    amp_dtype, device = amp_dtype_and_device()
    if dimensionality == 2:
        model = EDES2DClassifier(backbone, formulation, initialization, len(spec.classes)).to(device)
    else:
        model = EDES3DClassifier(backbone, formulation, len(spec.classes)).to(device)
    feature_layers = feature_layer_candidates(model, backbone, dimensionality)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    save_json(
        run_dir / "architecture.json",
        {
            "task": spec.key,
            "dataset": spec.dataset,
            "view": spec.view,
            "dimensionality": dimensionality,
            "backbone": backbone,
            "formulation": formulation,
            "initialization": initialization,
            "classes": spec.classes,
            "feature_dim": model.feature_dim,
            "feature_layer_candidates": feature_layers,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "cinema_commit": cinema_git_commit(),
            "config": OmegaConf.to_container(config, resolve=True),
        },
    )
    print(
        f"[architecture] {dimensionality}d/{spec.key}/{backbone}/{formulation}/{initialization}/seed{seed} "
        f"parameters={parameter_count:,} feature_layers={feature_layers}"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.train.lr),
        betas=tuple(config.train.betas),
        weight_decay=float(config.train.weight_decay),
    )
    scaler = grad_scaler(args.amp)
    n_accum = get_n_accum_steps(
        batch_size=int(config.train.batch_size),
        batch_size_per_device=int(config.train.batch_size_per_device),
        world_size=1,
    )
    best_mcc = -math.inf
    best_epoch = -1
    bad_evaluations = 0
    checkpoint = run_dir / "best_val_mcc.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(int(config.train.n_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        n_steps = len(train_loader)
        for step, batch in enumerate(tqdm(train_loader, desc=f"{spec.key}/{backbone}/e{epoch + 1}", leave=False)):
            lr = adjust_learning_rate(
                optimizer=optimizer,
                step=step / len(train_loader) + epoch,
                warmup_steps=int(config.train.n_warmup_epochs),
                max_n_steps=int(config.train.n_epochs),
                lr=float(config.train.lr),
                min_lr=float(config.train.min_lr),
            )
            image = batch[f"{spec.view}_image"].to(device, non_blocking=True)
            target = batch["label"].long().to(device, non_blocking=True)
            enabled = args.amp and device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enabled):
                logits = model(image)
                raw_loss = F.cross_entropy(logits, target, label_smoothing=float(config.train.label_smoothing))
                group_start = (step // n_accum) * n_accum
                accumulation_group_size = min(n_accum, n_steps - group_start)
                loss = raw_loss / accumulation_group_size
            scaler.scale(loss).backward()
            update = (step + 1) % n_accum == 0 or (step + 1) == n_steps
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.train.clip_grad))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(raw_loss.item()) * target.size(0)
            total_correct += int((logits.argmax(1) == target).sum().item())
            total_seen += target.size(0)

        if (epoch + 1) % int(config.train.eval_interval) != 0 and (epoch + 1) != int(config.train.n_epochs):
            continue
        vpids, vy, vprobs, val_metrics = evaluate_cnn(model, val_loader, spec, device, amp_dtype)
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(total_seen, 1),
            "train_accuracy": total_correct / max(total_seen, 1),
            "lr": lr,
            **{
                f"val_{k}": val_metrics[k]
                for k in ("accuracy", "f1", "macro_f1", "balanced_accuracy", "mcc", "roc_auc")
            },
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(
            f"[architecture] {dimensionality}d/{spec.key}/{backbone}/{formulation}/{initialization}/seed{seed} "
            f"epoch={epoch + 1} val_MCC={val_metrics['mcc']:.4f} val_AUROC={val_metrics['roc_auc']:.4f} "
            f"val_F1={val_metrics['f1']:.4f}"
        )
        if val_metrics["mcc"] > best_mcc + float(config.train.early_stopping.min_delta):
            best_mcc = float(val_metrics["mcc"])
            best_epoch = epoch + 1
            bad_evaluations = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "family": "architecture_bank",
                    "task": spec.key,
                    "classes": list(spec.classes),
                    "dimensionality": dimensionality,
                    "backbone": backbone,
                    "formulation": formulation,
                    "initialization": initialization,
                    "feature_dim": model.feature_dim,
                    "feature_layer_candidates": feature_layers,
                    "seed": seed,
                    "epoch": best_epoch,
                    "cinema_commit": cinema_git_commit(),
                },
                checkpoint,
            )
            save_predictions(run_dir / "best_validation", vpids, vy, vprobs, spec.classes, val_metrics)
        else:
            bad_evaluations += 1
        if bad_evaluations >= int(config.train.early_stopping.patience):
            break

    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    pids, y, probs, test_metrics = evaluate_cnn(model, test_loader, spec, device, amp_dtype)
    save_predictions(run_dir / "test", pids, y, probs, spec.classes, test_metrics)
    summary = {
        "family": "architecture_bank",
        "task": spec.key,
        "dimensionality": dimensionality,
        "backbone": backbone,
        "formulation": formulation,
        "initialization": initialization,
        "seed": seed,
        "best_val_mcc": best_mcc,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "feature_layer_candidates": feature_layers,
        "checkpoint": checkpoint,
        "test": {k: test_metrics[k] for k in ("accuracy", "f1", "macro_f1", "balanced_accuracy", "mcc", "roc_auc")},
    }
    save_json(summary_path, summary)
    del model, optimizer, scaler
    cleanup_cuda()
    return summary


def run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages)
    run_2d = bool(stages & {"cnn", "cnn2d"})
    run_3d = "cnn3d" in stages
    selected = resolve_tasks(list(args.tasks))
    validation_specs = list(selected) if stages & {"preprocess", "calibrate", "official"} else []
    if run_2d:
        for key in args.cnn2d_tasks:
            if TASKS[key] not in validation_specs:
                validation_specs.append(TASKS[key])
    if run_3d:
        for key in args.cnn3d_tasks:
            if TASKS[key] not in validation_specs:
                validation_specs.append(TASKS[key])
    datasets = sorted({spec.dataset for spec in validation_specs})

    print("=" * 100)
    print("CineMA validation and 2-D/3-D architecture-bank benchmark")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CineMA commit: {cinema_git_commit()}")
    print(f"Tasks: {[spec.key for spec in selected]}")
    if run_2d:
        print(f"2-D tasks/models: {list(args.cnn2d_tasks)} / {list(args.models_2d)}")
    if run_3d:
        print(f"3-D tasks/models: {list(args.cnn3d_tasks)} / {list(args.models_3d)}")
    print(f"Stages: {list(args.stages)}")
    print("=" * 100)

    if "preprocess" in args.stages:
        for dataset in datasets:
            preprocess_dataset(args, dataset)

    # Validate and persist splits before any evaluation or training.
    for spec in validation_specs:
        splits = split_metadata(spec, processed_dir(args, spec.dataset))
        save_split_audit(args, spec, splits)

    calibration_summaries = []
    if "calibrate" in args.stages:
        for spec in selected:
            calibration_summaries.append(calibrate_task(args, spec))
        passed = all(item["passed"] for item in calibration_summaries)
        save_json(
            args.output_dir / "calibration" / "gate.json",
            {"passed": passed, "tasks": {x["task"]: x["passed"] for x in calibration_summaries}},
        )
        if not passed and not args.allow_calibration_failure:
            raise RuntimeError(
                "Released-checkpoint calibration failed. CNN/official training is intentionally blocked."
            )

    training_requested = "official" in stages or run_2d or run_3d
    if training_requested and not args.smoke and not args.allow_calibration_failure:
        gate_tasks: list[TaskSpec] = []
        if "official" in stages:
            gate_tasks.extend(selected)
        if run_2d:
            gate_tasks.extend(TASKS[key] for key in args.cnn2d_tasks)
        if run_3d:
            gate_tasks.extend(TASKS[key] for key in args.cnn3d_tasks)
        gate_tasks = list(dict.fromkeys(gate_tasks))
        if not calibration_is_valid(args, gate_tasks):
            raise RuntimeError(
                "Run run.stages=[preprocess,calibrate] first. Full training is blocked until released checkpoints reproduce."
            )

    if "official" in stages:
        for spec in selected:
            for mode in args.official_modes:
                for seed in args.seeds:
                    run_official_training(args, spec, mode, seed)
        aggregate_official_results(args)

    if run_2d:
        for task_key in args.cnn2d_tasks:
            spec = TASKS[task_key]
            for backbone in args.models_2d:
                for formulation in args.formulations_2d:
                    for initialization in args.initializations_2d:
                        for seed in args.seeds:
                            train_architecture(args, spec, backbone, formulation, initialization, seed)

    if run_3d:
        for task_key in args.cnn3d_tasks:
            spec = TASKS[task_key]
            for backbone in args.models_3d:
                for formulation in args.formulations_3d:
                    for seed in args.seeds:
                        train_architecture(args, spec, backbone, formulation, "randinit", seed)

    if (run_2d or run_3d) and not args.smoke:
        aggregate_architecture_results(args)

    print(f"Done. Results: {args.output_dir}")


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    """Run the experiment using the adjacent YAML configuration."""
    run(prepare_run(config, "architecture_bank"))


if __name__ == "__main__":
    main()
