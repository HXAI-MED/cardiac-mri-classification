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
from training_code.classification.protocol import TaskSpec
from training_code.runtime import cinema_git_commit, save_json, seed_everything


def processed_dir(args: argparse.Namespace, dataset: str) -> Path:
    explicit = args.acdc_processed if dataset == "acdc" else args.mnms2_processed
    if explicit is not None:
        return explicit.expanduser().resolve()
    return (args.output_dir / "processed" / dataset).resolve()


def raw_dir(args: argparse.Namespace, dataset: str) -> Path | None:
    value = args.acdc_raw if dataset == "acdc" else args.mnms2_raw
    return None if value is None else value.expanduser().resolve()


def validate_raw_layout(dataset: str, root: Path) -> None:
    if dataset == "acdc":
        required = [root / "training", root / "testing", root / "training" / "patient001" / "Info.cfg"]
    else:
        required = [
            root / "dataset_information.csv",
            root / "dataset",
            root / "dataset" / "001" / "001_LA_ED.nii.gz",
            root / "dataset" / "001" / "001_SA_ED.nii.gz",
        ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{dataset} raw-data layout is not CineMA-compatible. Missing: {missing}")


def preprocess_dataset(args: argparse.Namespace, dataset: str) -> None:
    source = raw_dir(args, dataset)
    if source is None:
        raise ValueError(f"data.{dataset}_raw is required for preprocessing {dataset}")
    validate_raw_layout(dataset, source)
    target = processed_dir(args, dataset)

    expected_files = [target / "train_metadata.csv", target / "test_metadata.csv"]
    if dataset == "mnms2":
        expected_files.append(target / "val_metadata.csv")
    if all(p.exists() for p in expected_files) and not args.force_preprocess:
        print(f"[preprocess] Reusing existing official {dataset} output: {target}")
        return

    target.mkdir(parents=True, exist_ok=True)
    module = f"cinema.data.{dataset}.preprocess"
    command = [sys.executable, "-m", module, "--data_dir", str(source), "--out_dir", str(target)]
    print("[preprocess]", " ".join(command))
    environment = os.environ.copy()
    # Some M&Ms2 NIfTI files contain a valid but non-orthogonal sform.  ITK's
    # permissive mode preserves the official CineMA preprocessing path while
    # allowing those files to be read (ITK may still emit informational warnings).
    environment.setdefault("ITK_NIFTI_SFORM_PERMISSIVE", "1")
    subprocess.check_call(command, env=environment)


def local_config(spec: TaskSpec, data_root: Path, seed: int = 0) -> DictConfig:
    package = importlib.import_module(f"cinema.classification.{spec.dataset}")
    path = Path(inspect.getfile(package)).resolve().parent / "config.yaml"
    config = OmegaConf.load(path)
    config.data.dir = str(data_root)
    config.model.views = spec.view
    config.seed = seed
    if tuple(config.data[config.data.class_column]) != spec.classes:
        raise RuntimeError(
            f"Installed CineMA class list changed for {spec.key}: "
            f"{list(config.data[config.data.class_column])} != {list(spec.classes)}"
        )
    return config


def split_metadata(spec: TaskSpec, data_root: Path) -> dict[str, pd.DataFrame]:
    if spec.dataset == "acdc":
        development = pd.read_csv(data_root / "train_metadata.csv", dtype={"pid": str})
        test = pd.read_csv(data_root / "test_metadata.csv", dtype={"pid": str})
        val_pids = development.groupby("pathology").sample(n=2, random_state=0)["pid"].tolist()
        train = development[~development["pid"].isin(val_pids)].reset_index(drop=True)
        val = development[development["pid"].isin(val_pids)].reset_index(drop=True)
    else:
        train = pd.read_csv(data_root / "train_metadata.csv", dtype={"pid": str})
        val = pd.read_csv(data_root / "val_metadata.csv", dtype={"pid": str})
        test = pd.read_csv(data_root / "test_metadata.csv", dtype={"pid": str})
        class_col = "pathology"
        train = train[train[class_col].isin(spec.classes)].reset_index(drop=True)
        val = val[val[class_col].isin(spec.classes)].reset_index(drop=True)
        test = test[test[class_col].isin(spec.classes)].reset_index(drop=True)

    result = {"train": train, "val": val, "test": test}
    expected = {"train": spec.expected_train, "val": spec.expected_val, "test": spec.expected_test}
    for split, frame in result.items():
        if len(frame) != expected[split]:
            raise RuntimeError(
                f"{spec.key} {split} has {len(frame)} patients; expected {expected[split]}. "
                "Do not train until the official preprocessing/data release is complete."
            )
        observed = set(frame["pathology"].unique())
        if observed != set(spec.classes):
            raise RuntimeError(f"{spec.key} {split} classes {sorted(observed)} != {sorted(spec.classes)}")

    pid_sets = {k: set(v["pid"].astype(str)) for k, v in result.items()}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = pid_sets[first] & pid_sets[second]
        if overlap:
            raise RuntimeError(f"Patient leakage in {spec.key}: {first}/{second}: {sorted(overlap)}")
    return result


def save_split_audit(args: argparse.Namespace, spec: TaskSpec, splits: dict[str, pd.DataFrame]) -> None:
    root = args.output_dir / "splits" / spec.key
    root.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "task": spec.key,
        "classes": list(spec.classes),
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
    spec: TaskSpec,
    config: DictConfig,
    data_root: Path,
    frame: pd.DataFrame,
    split: str,
    train: bool,
) -> EndDiastoleEndSystoleDataset:
    train_transform, eval_transform = get_image_transforms(config)
    directory_split = "train" if spec.dataset == "acdc" and split == "val" else split
    return EndDiastoleEndSystoleDataset(
        data_dir=data_root / directory_split,
        meta_df=frame,
        class_col=config.data.class_column,
        classes=list(config.data[config.data.class_column]),
        views=spec.view,
        transform=train_transform if train else eval_transform,
    )


def make_loaders(
    spec: TaskSpec,
    config: DictConfig,
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    seed_everything(seed)
    train_ds = make_dataset(spec, config, data_root, splits["train"], "train", True)
    val_ds = make_dataset(spec, config, data_root, splits["val"], "val", False)
    test_ds = make_dataset(spec, config, data_root, splits["test"], "test", False)
    workers = int(config.train.n_workers)
    loader_kwargs = {"pin_memory": torch.cuda.is_available(), "num_workers": workers}
    train_loader = DataLoader(
        train_ds,
        sampler=RandomSampler(train_ds),
        batch_size=int(config.train.batch_size_per_device),
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, sampler=SequentialSampler(val_ds), batch_size=1, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(
        test_ds, sampler=SequentialSampler(test_ds), batch_size=1, drop_last=False, **loader_kwargs
    )
    return train_loader, val_loader, test_loader
