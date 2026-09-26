"""Classification training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from cinema.optim import adjust_learning_rate, get_n_accum_steps
from training_code.classification.mnms2_sax_2d.dataset import (
    load_mnms2_config,
    load_official_splits,
    make_loaders,
    preprocess_mnms2,
    resolve_processed_dir,
    save_split_audit,
    validate_inputs,
)
from training_code.classification.mnms2_sax_2d.eval import aggregate_results, evaluate, save_predictions
from training_code.classification.mnms2_sax_2d.model import MnMs2SAXMid2DClassifier
from training_code.classification.model import feature_layer_candidates
from training_code.classification.protocol import protocols
from training_code.classification.utils import apply_overrides, prepare_run
from training_code.runtime import amp_dtype_and_device, cinema_git_commit, cleanup_cuda, grad_scaler
from training_code.runtime import save_json_arrays as save_json

CLASSES = protocols["mnms2_sax_2d"]["classes"]
TASK_KEY = protocols["mnms2_sax_2d"]["task_key"]


def train_one_run(
    args: argparse.Namespace,
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    backbone: str,
    formulation: str,
    initialization: str,
    seed: int,
) -> dict[str, Any]:
    output_group = "architecture_smoke" if args.smoke else "architecture_bank"
    run_dir = (
        args.output_dir / output_group / "2d" / TASK_KEY / backbone / formulation / initialization / f"seed_{seed}"
    )
    summary_path = run_dir / "summary.json"
    if args.resume and summary_path.is_file():
        print(f"[resume] {backbone}/{formulation}/{initialization}/seed_{seed}")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)

    config = load_mnms2_config(data_root, seed)
    apply_overrides(args, config)
    train_loader, val_loader, test_loader = make_loaders(data_root, splits, config, seed)
    amp_dtype, device = amp_dtype_and_device()
    model = MnMs2SAXMid2DClassifier(backbone, formulation, initialization).to(device)
    feature_layers = feature_layer_candidates(model, backbone)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    architecture = {
        "family": "mnms2_sax_mid_2d_architecture_bank",
        "task": TASK_KEY,
        "dataset": "mnms2",
        "view": "sax_mid",
        "dimensionality": 2,
        "input_policy": "patient_mid_sax_ed_es",
        "backbone": backbone,
        "formulation": formulation,
        "initialization": initialization,
        "classes": list(CLASSES),
        "feature_dim": model.feature_dim,
        "feature_layer_candidates": feature_layers,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "cinema_commit": cinema_git_commit(),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    save_json(run_dir / "architecture.json", architecture)
    print(
        f"[architecture] mnms2/2d/{backbone}/{formulation}/{initialization}/seed{seed} "
        f"parameters={parameter_count:,} feature_layers={feature_layers}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.train.lr),
        betas=tuple(config.train.betas),
        weight_decay=float(config.train.weight_decay),
    )
    scaler = grad_scaler(args.amp)
    accumulation_steps = get_n_accum_steps(
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
        progress = tqdm(train_loader, desc=f"mnms2/{backbone}/e{epoch + 1}", leave=False)
        for step, batch in enumerate(progress):
            learning_rate = adjust_learning_rate(
                optimizer=optimizer,
                step=step / max(len(train_loader), 1) + epoch,
                warmup_steps=int(config.train.n_warmup_epochs),
                max_n_steps=int(config.train.n_epochs),
                lr=float(config.train.lr),
                min_lr=float(config.train.min_lr),
            )
            image = batch["sax_image"].to(device, non_blocking=True)
            target = batch["label"].long().to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=args.amp and device.type == "cuda",
            ):
                logits = model(image)
                raw_loss = F.cross_entropy(
                    logits,
                    target,
                    label_smoothing=float(config.train.label_smoothing),
                )
                group_start = (step // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, n_steps - group_start)
                loss = raw_loss / group_size
            scaler.scale(loss).backward()
            update = (step + 1) % accumulation_steps == 0 or (step + 1) == n_steps
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.train.clip_grad))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(raw_loss.item()) * target.size(0)
            total_correct += int((logits.argmax(1) == target).sum().item())
            total_seen += target.size(0)

        final_epoch = epoch + 1 == int(config.train.n_epochs)
        if (epoch + 1) % int(config.train.eval_interval) != 0 and not final_epoch:
            continue
        val_pids, val_y, val_probs, val_metrics = evaluate(model, val_loader, device, amp_dtype)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(total_seen, 1),
                "train_accuracy": total_correct / max(total_seen, 1),
                "lr": learning_rate,
                **{
                    f"val_{key}": val_metrics[key]
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
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(
            f"[architecture] mnms2/2d/{backbone}/{formulation}/{initialization}/seed{seed} "
            f"epoch={epoch + 1} val_MCC={val_metrics['mcc']:.4f} "
            f"val_AUROC={val_metrics['roc_auc']:.4f} val_F1={val_metrics['f1']:.4f}"
        )
        if val_metrics["mcc"] > best_mcc + float(config.train.early_stopping.min_delta):
            best_mcc = float(val_metrics["mcc"])
            best_epoch = epoch + 1
            bad_evaluations = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    **{
                        key: architecture[key]
                        for key in (
                            "family",
                            "task",
                            "classes",
                            "dimensionality",
                            "input_policy",
                            "backbone",
                            "formulation",
                            "initialization",
                            "feature_dim",
                            "feature_layer_candidates",
                            "cinema_commit",
                        )
                    },
                    "seed": seed,
                    "epoch": best_epoch,
                },
                checkpoint,
            )
            save_predictions(run_dir / "best_validation", val_pids, val_y, val_probs, val_metrics)
        else:
            bad_evaluations += 1
        if bad_evaluations >= int(config.train.early_stopping.patience):
            break

    if not checkpoint.is_file():
        raise RuntimeError(f"No checkpoint was created for {run_dir}")
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    test_pids, test_y, test_probs, test_metrics = evaluate(model, test_loader, device, amp_dtype)
    save_predictions(run_dir / "test", test_pids, test_y, test_probs, test_metrics)
    summary = {
        "family": "mnms2_sax_mid_2d_architecture_bank",
        "task": TASK_KEY,
        "dataset": "mnms2",
        "view": "sax_mid",
        "dimensionality": 2,
        "input_policy": "patient_mid_sax_ed_es",
        "backbone": backbone,
        "formulation": formulation,
        "initialization": initialization,
        "seed": seed,
        "best_val_mcc": best_mcc,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "feature_layer_candidates": feature_layers,
        "checkpoint": checkpoint,
        "test": {
            key: test_metrics[key]
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
    save_json(summary_path, summary)
    del model, optimizer, scaler
    cleanup_cuda()
    return summary


def run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages)

    print("=" * 100)
    print("M&Ms2 central-SAX patient-level 2-D architecture-bank benchmark")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CineMA commit: {cinema_git_commit()}")
    print(f"Task: {TASK_KEY}")
    print(f"Classes: {list(CLASSES)}")
    print(f"Models: {list(args.models_2d)}")
    print(f"Formulations: {list(args.formulations_2d)}")
    print(f"Initializations: {list(args.initializations_2d)}")
    print(f"Seeds: {list(args.seeds)}")
    print(f"Stages: {list(args.stages)}")
    print("=" * 100)

    if "preprocess" in stages:
        preprocess_mnms2(args)
    data_root = resolve_processed_dir(args)
    splits = load_official_splits(data_root)
    save_split_audit(args.output_dir, splits)
    config = load_mnms2_config(data_root, seed=0)
    if stages & {"validate", "train"}:
        validate_inputs(args.output_dir, data_root, splits, config)

    if "train" in stages:
        for backbone in args.models_2d:
            for formulation in args.formulations_2d:
                for initialization in args.initializations_2d:
                    for seed in args.seeds:
                        train_one_run(
                            args,
                            data_root,
                            splits,
                            backbone,
                            formulation,
                            initialization,
                            seed,
                        )
        if not args.smoke:
            aggregate_results(args)
    print(f"Done. Results: {args.output_dir}")


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    """Run the experiment using the adjacent YAML configuration."""
    run(prepare_run(config, "mnms2_sax_2d"))


if __name__ == "__main__":
    main()
