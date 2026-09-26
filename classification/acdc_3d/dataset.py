"""Patient loading, preprocessing, and transforms."""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from cinema.classification.dataset import EndDiastoleEndSystoleDataset, get_image_transforms
from training_code.classification.protocol import protocols
from training_code.classification.utils import apply_training_overrides
from training_code.runtime import cinema_git_commit, save_json
from training_code.runtime import seed_deterministic as seed_everything

CLASSES = protocols["acdc"]["classes"]
DATASET = protocols["acdc_3d"]["dataset"]
EXPECTED_SPLIT_SIZES = protocols["acdc"]["expected_split_sizes"]
TASK_KEY = protocols["acdc_3d"]["task_key"]
VIEW = protocols["shared"]["view"]


def processed_dir(args: argparse.Namespace) -> Path:
    if args.processed_dir is not None:
        return args.processed_dir.expanduser().resolve()
    return (args.output_dir / "processed" / DATASET).resolve()


def validate_raw_layout(root: Path) -> None:
    required = [
        root / "training",
        root / "testing",
        root / "training" / "patient001" / "Info.cfg",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"ACDC raw-data layout is not CineMA-compatible. Missing: {missing}")


def preprocess(args: argparse.Namespace) -> None:
    if args.raw_dir is None:
        raise ValueError("data.raw_dir is required for preprocessing ACDC")
    source = args.raw_dir.expanduser().resolve()
    validate_raw_layout(source)
    target = processed_dir(args)
    expected = [target / "train_metadata.csv", target / "test_metadata.csv"]
    if all(path.is_file() for path in expected) and not args.force_preprocess:
        print(f"[preprocess] Reusing official ACDC output: {target}")
        return
    target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "cinema.data.acdc.preprocess",
        "--data_dir",
        str(source),
        "--out_dir",
        str(target),
    ]
    print("[preprocess]", " ".join(command))
    subprocess.check_call(command, env=os.environ.copy())


def local_config(data_root: Path, seed: int) -> DictConfig:
    package = importlib.import_module("cinema.classification.acdc")
    config_path = Path(inspect.getfile(package)).resolve().parent / "config.yaml"
    config = OmegaConf.load(config_path)
    config.data.dir = str(data_root)
    config.model.views = VIEW
    config.seed = seed
    installed_classes = tuple(config.data[config.data.class_column])
    if installed_classes != CLASSES:
        raise RuntimeError(f"Installed CineMA ACDC class list changed: {installed_classes} != {CLASSES}")
    return config


def split_metadata(data_root: Path) -> dict[str, pd.DataFrame]:
    development = pd.read_csv(data_root / "train_metadata.csv", dtype={"pid": str})
    test = pd.read_csv(data_root / "test_metadata.csv", dtype={"pid": str})
    required_columns = {"pid", "pathology", "n_slices"}
    for name, frame in (("development", development), ("test", test)):
        missing = required_columns - set(frame.columns)
        if missing:
            raise RuntimeError(f"ACDC {name} metadata is missing {sorted(missing)}")

    validation_pids = development.groupby("pathology", group_keys=False).sample(n=2, random_state=0)["pid"].tolist()
    train = development[~development["pid"].isin(validation_pids)].reset_index(drop=True)
    val = development[development["pid"].isin(validation_pids)].reset_index(drop=True)
    splits = {"train": train, "val": val, "test": test.reset_index(drop=True)}

    for name, frame in splits.items():
        expected = EXPECTED_SPLIT_SIZES[name]
        if len(frame) != expected:
            raise RuntimeError(f"{TASK_KEY} {name} contains {len(frame)} patients; expected {expected}")
        observed = set(frame["pathology"].astype(str).unique())
        if observed != set(CLASSES):
            raise RuntimeError(f"{TASK_KEY} {name} classes {sorted(observed)} != {sorted(CLASSES)}")

    patient_sets = {name: set(frame["pid"].astype(str)) for name, frame in splits.items()}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patient_sets[first] & patient_sets[second]
        if overlap:
            raise RuntimeError(f"Patient leakage in {TASK_KEY}: {first}/{second}: {sorted(overlap)}")
    return splits


def save_split_audit(args: argparse.Namespace, splits: dict[str, pd.DataFrame]) -> None:
    root = args.output_dir / "splits" / TASK_KEY
    root.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "dataset": DATASET,
        "task": TASK_KEY,
        "view": VIEW,
        "dimensionality": 3,
        "input_policy": "full_sax_ed_es_volume_per_patient",
        "classes": list(CLASSES),
        "cinema_commit": cinema_git_commit(),
        "splits": {},
    }
    for name, frame in splits.items():
        frame.to_csv(root / f"{name}.csv", index=False)
        audit["splits"][name] = {
            "n": len(frame),
            "pids": frame["pid"].astype(str).tolist(),
            "class_counts": frame["pathology"].value_counts().sort_index().to_dict(),
        }
    save_json(root / "audit.json", audit)


def make_dataset(
    config: DictConfig,
    data_root: Path,
    frame: pd.DataFrame,
    split: str,
    train: bool,
) -> EndDiastoleEndSystoleDataset:
    train_transform, evaluation_transform = get_image_transforms(config)
    directory_split = "train" if split == "val" else split
    return EndDiastoleEndSystoleDataset(
        data_dir=data_root / directory_split,
        meta_df=frame,
        class_col=config.data.class_column,
        classes=list(config.data[config.data.class_column]),
        views=VIEW,
        transform=train_transform if train else evaluation_transform,
    )


def make_loaders(
    config: DictConfig,
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    seed: int,
    deterministic: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    seed_everything(seed, deterministic)
    train_dataset = make_dataset(config, data_root, splits["train"], "train", True)
    val_dataset = make_dataset(config, data_root, splits["val"], "val", False)
    test_dataset = make_dataset(config, data_root, splits["test"], "test", False)
    workers = int(config.train.n_workers)
    loader_options = {
        "pin_memory": torch.cuda.is_available(),
        "num_workers": workers,
        "persistent_workers": workers > 0,
    }
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset, generator=generator),
        batch_size=int(config.train.batch_size_per_device),
        drop_last=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=1,
        drop_last=False,
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=1,
        drop_last=False,
        **loader_options,
    )
    return train_loader, val_loader, test_loader


def validate_inputs(args: argparse.Namespace) -> None:
    data_root = processed_dir(args)
    splits = split_metadata(data_root)
    save_split_audit(args, splits)
    config = local_config(data_root, seed=0)
    apply_training_overrides(args, config)
    _, val_loader, test_loader = make_loaders(config, data_root, splits, seed=0, deterministic=True)
    patch_size = tuple(int(value) for value in config.data.sax.patch_size)
    rows: list[dict[str, Any]] = []
    for split_name, loader in (("val", val_loader), ("test", test_loader)):
        for index, batch in enumerate(loader):
            image = batch["sax_image"]
            if image.ndim != 5 or image.shape[1] != 2:
                raise RuntimeError(
                    f"{split_name} patient {batch['pid'][0]} produced {tuple(image.shape)}; expected (B,2,X,Y,Z)"
                )
            spatial_shape = tuple(int(value) for value in image.shape[2:])
            if any(size < patch for size, patch in zip(spatial_shape, patch_size, strict=True)):
                raise RuntimeError(
                    f"{split_name} patient {batch['pid'][0]} shape {spatial_shape} "
                    f"is smaller than evaluation patch {patch_size}"
                )
            if not torch.isfinite(image).all():
                raise RuntimeError(f"Non-finite input for {split_name} patient {batch['pid'][0]}")
            rows.append(
                {
                    "split": split_name,
                    "pid": str(batch["pid"][0]),
                    "shape": "x".join(str(value) for value in image.shape),
                    "minimum": float(image.min()),
                    "maximum": float(image.max()),
                    "mean": float(image.mean()),
                }
            )
            if index >= 4:
                break
    audit_path = args.output_dir / "splits" / TASK_KEY / "input_tensor_audit.csv"
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    print(f"[validate] ACDC split and full-volume tensor contract passed: {audit_path}")
