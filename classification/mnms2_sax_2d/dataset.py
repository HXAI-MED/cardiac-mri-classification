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

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandSpatialCropd,
    ScaleIntensityd,
    SpatialPadd,
)
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from training_code.classification.protocol import protocols
from training_code.runtime import cinema_git_commit
from training_code.runtime import save_json_arrays as save_json
from training_code.runtime import seed_cudnn as seed_everything

CLASSES = protocols["mnms2_sax_2d"]["classes"]
EXPECTED_SPLIT_SIZES = protocols["mnms2_sax_2d"]["expected_split_sizes"]
TASK_KEY = protocols["mnms2_sax_2d"]["task_key"]


def resolve_processed_dir(args: argparse.Namespace) -> Path:
    if args.mnms2_processed is not None:
        return args.mnms2_processed.expanduser().resolve()
    return (args.output_dir / "processed" / "mnms2").resolve()


def validate_raw_layout(root: Path) -> None:
    required = [
        root / "dataset_information.csv",
        root / "dataset",
        root / "dataset" / "001" / "001_SA_ED.nii.gz",
        root / "dataset" / "001" / "001_SA_ES.nii.gz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"M&Ms2 raw-data layout is invalid. Missing: {missing}")


def preprocess_mnms2(args: argparse.Namespace) -> None:
    if args.mnms2_raw is None:
        raise ValueError("data.mnms2_raw is required when run.stages includes preprocess")
    source = args.mnms2_raw.expanduser().resolve()
    validate_raw_layout(source)
    target = resolve_processed_dir(args)
    expected = [
        target / "train_metadata.csv",
        target / "val_metadata.csv",
        target / "test_metadata.csv",
    ]
    if all(path.is_file() for path in expected) and not args.force_preprocess:
        print(f"[preprocess] Reusing existing official M&Ms2 output: {target}")
        return
    target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "cinema.data.mnms2.preprocess",
        "--data_dir",
        str(source),
        "--out_dir",
        str(target),
    ]
    print("[preprocess]", " ".join(command))
    subprocess.check_call(command, env=os.environ.copy())


def load_mnms2_config(data_root: Path, seed: int) -> DictConfig:
    package = importlib.import_module("cinema.classification.mnms2")
    config_path = Path(inspect.getfile(package)).resolve().parent / "config.yaml"
    config = OmegaConf.load(config_path)
    config.data.dir = str(data_root)
    config.model.views = "sax"
    config.seed = seed
    configured_classes = tuple(config.data[config.data.class_column])
    if configured_classes != CLASSES:
        raise RuntimeError(f"Installed CineMA M&Ms2 classes changed: {configured_classes} != {CLASSES}")
    return config


def load_official_splits(data_root: Path) -> dict[str, pd.DataFrame]:
    paths = {name: data_root / f"{name}_metadata.csv" for name in EXPECTED_SPLIT_SIZES}
    missing_paths = [str(path) for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"Official processed M&Ms2 metadata not found. Missing: {missing_paths}. Run run.stages=[preprocess] first."
        )
    splits = {}
    for name, path in paths.items():
        frame = pd.read_csv(path, dtype={"pid": str})
        splits[name] = frame[frame["pathology"].isin(CLASSES)].reset_index(drop=True)

    for name, frame in splits.items():
        expected = EXPECTED_SPLIT_SIZES[name]
        if len(frame) != expected:
            raise RuntimeError(f"M&Ms2 {name} has {len(frame)} patients; expected {expected}")
        observed = set(frame["pathology"].astype(str).unique())
        if observed != set(CLASSES):
            raise RuntimeError(f"M&Ms2 {name} classes {sorted(observed)} != {sorted(CLASSES)}")
        required = {"pid", "pathology", "n_slices"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"M&Ms2 {name} metadata is missing {sorted(missing)}")

    pid_sets = {name: set(frame["pid"].astype(str)) for name, frame in splits.items()}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = pid_sets[first] & pid_sets[second]
        if overlap:
            raise RuntimeError(f"Patient leakage between {first}/{second}: {sorted(overlap)}")
    return splits


def save_split_audit(output_dir: Path, splits: dict[str, pd.DataFrame]) -> None:
    root = output_dir / "splits" / TASK_KEY
    audit: dict[str, Any] = {
        "task": TASK_KEY,
        "dataset": "M&Ms2",
        "input_policy": "one deterministic central SAX slice at ED and ES per patient",
        "classes": list(CLASSES),
        "cinema_commit": cinema_git_commit(),
        "splits": {},
    }
    for name, frame in splits.items():
        export = frame.copy()
        export["mid_sax_index_0based"] = (export["n_slices"].astype(int) - 1).clip(lower=0) // 2
        root.mkdir(parents=True, exist_ok=True)
        export.to_csv(root / f"{name}.csv", index=False)
        audit["splits"][name] = {
            "n": len(export),
            "class_counts": export["pathology"].value_counts().sort_index().to_dict(),
            "pids": export["pid"].astype(str).tolist(),
        }
    save_json(root / "audit.json", audit)


class MnMs2MidSAX2DDataset(Dataset):
    """Load one central SAX ED/ES slice for each MnMs2 patient."""

    def __init__(
        self,
        data_dir: Path,
        metadata: pd.DataFrame,
        transform: Any | None,
    ) -> None:
        self.data_dir = data_dir
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[int(index)]
        pid = str(row["pid"])
        arrays = []
        for phase in ("ed", "es"):
            path = self.data_dir / pid / f"{pid}_sax_{phase}.nii.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
            image = sitk.ReadImage(str(path))
            array = np.transpose(sitk.GetArrayFromImage(image)).astype(np.float32, copy=False)
            if array.ndim != 3:
                raise RuntimeError(f"Expected 3-D volume at {path}, got {array.shape}")
            arrays.append(array)

        available = min(int(row["n_slices"]), arrays[0].shape[-1], arrays[1].shape[-1])
        if available < 1:
            raise RuntimeError(f"No valid SAX slices for {pid}")
        slice_index = (available - 1) // 2
        image_2d = np.stack([array[..., slice_index] for array in arrays], axis=0)
        pathology = str(row["pathology"])
        sample: dict[str, Any] = {
            "pid": pid,
            "class": pathology,
            "label": torch.tensor(CLASSES.index(pathology), dtype=torch.long),
            "sax_image": torch.from_numpy(image_2d),
            "slice_index": torch.tensor(slice_index, dtype=torch.long),
        }
        return self.transform(sample) if self.transform is not None else sample


def get_2d_transforms(config: DictConfig) -> tuple[Any, Any]:
    """Convert CineMA's MnMs2 SAX augmentation settings to two spatial dimensions."""
    patch_size = tuple(int(value) for value in config.data.sax.patch_size[:2])
    rotation = float(config.transform.sax.rotate_range[-1]) / 180.0 * np.pi
    translation = tuple(float(value) for value in config.transform.sax.translate_range[:2])
    probability = float(config.transform.prob)
    train_transform = Compose(
        [
            RandAdjustContrastd(keys="sax_image", prob=probability, gamma=config.transform.gamma),
            RandGaussianNoised(keys="sax_image", prob=probability),
            ScaleIntensityd(keys="sax_image"),
            RandAffined(
                keys="sax_image",
                mode="bilinear",
                prob=probability,
                rotate_range=(rotation,),
                translate_range=translation,
                scale_range=config.transform.scale_range,
                padding_mode="zeros",
                lazy=True,
            ),
            RandSpatialCropd(keys="sax_image", roi_size=patch_size, lazy=True),
            SpatialPadd(keys="sax_image", spatial_size=patch_size, method="end", lazy=True),
        ]
    )
    eval_transform = Compose(
        [
            ScaleIntensityd(keys="sax_image"),
            SpatialPadd(keys="sax_image", spatial_size=patch_size, method="end", lazy=True),
        ]
    )
    return train_transform, eval_transform


def split_directory(data_root: Path, split: str) -> Path:
    return data_root / split


def make_datasets(
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    config: DictConfig,
) -> tuple[Dataset, Dataset, Dataset]:
    train_transform, eval_transform = get_2d_transforms(config)
    train = MnMs2MidSAX2DDataset(split_directory(data_root, "train"), splits["train"], train_transform)
    val = MnMs2MidSAX2DDataset(split_directory(data_root, "val"), splits["val"], eval_transform)
    test = MnMs2MidSAX2DDataset(split_directory(data_root, "test"), splits["test"], eval_transform)
    return train, val, test


def validate_inputs(
    output_dir: Path,
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    config: DictConfig,
) -> None:
    _, val_dataset, test_dataset = make_datasets(data_root, splits, config)
    rows = []
    for split_name, dataset in (("val", val_dataset), ("test", test_dataset)):
        for index in range(min(5, len(dataset))):
            sample = dataset[index]
            image = sample["sax_image"]
            if image.ndim != 3 or image.shape[0] != 2:
                raise RuntimeError(f"{split_name} {sample['pid']} produced {tuple(image.shape)}; expected (2,H,W)")
            if not torch.isfinite(image).all():
                raise RuntimeError(f"Non-finite input for {split_name} patient {sample['pid']}")
            rows.append(
                {
                    "split": split_name,
                    "pid": sample["pid"],
                    "slice_index": int(sample["slice_index"]),
                    "shape": "x".join(str(int(value)) for value in image.shape),
                    "minimum": float(image.min()),
                    "maximum": float(image.max()),
                    "mean": float(image.mean()),
                    "nonzero_fraction": float((image != 0).float().mean()),
                }
            )
    audit_path = output_dir / "splits" / TASK_KEY / "input_tensor_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    print(f"[validate] Patient-level 2-D input audit passed: {audit_path}")


def make_loaders(
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    config: DictConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    seed_everything(seed)
    train_dataset, val_dataset, test_dataset = make_datasets(data_root, splits, config)
    workers = int(config.train.n_workers)
    common = {"num_workers": workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=int(config.train.batch_size_per_device),
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=1,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=1,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, test_loader
