"""Dataset loading, preprocessing, transforms, and split validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from cinema.classification.dataset import (
    EndDiastoleEndSystoleDataset,
    get_image_transforms,
)
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandSpatialCropd,
    ScaleIntensityd,
    SpatialPadd,
)
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from .cinema_support import (
    load_task_config,
    preprocess_dataset,
    processed_dir,
)
from .protocol import TaskSpec, protocols
from .utils import (
    apply_training_overrides,
    cinema_git_commit,
    save_json,
    save_json_arrays,
    seed_cudnn,
    seed_deterministic,
    seed_everything,
)

# Acdc 2D

ACDC_2D_CLASSES = protocols["acdc"]["classes"]


ACDC_2D_EXPECTED_SPLIT_SIZES = protocols["acdc"]["expected_split_sizes"]


ACDC_2D_TASK_KEY = protocols["acdc_2d"]["task_key"]


def acdc_2d_resolve_processed_dir(args: argparse.Namespace) -> Path:
    return processed_dir(args, "acdc")


def preprocess_acdc(args: argparse.Namespace) -> None:
    preprocess_dataset(args, "acdc")


def load_acdc_config(data_root: Path, seed: int) -> DictConfig:
    return load_task_config("acdc", data_root, seed)


def acdc_2d_load_official_splits(data_root: Path) -> dict[str, pd.DataFrame]:
    return acdc_3d_split_metadata(data_root)


def acdc_2d_save_split_audit(output_dir: Path, splits: dict[str, pd.DataFrame]) -> None:
    root = output_dir / "splits" / ACDC_2D_TASK_KEY
    audit: dict[str, Any] = {
        "task": ACDC_2D_TASK_KEY,
        "dataset": "ACDC",
        "input_policy": "one deterministic central SAX slice at ED and ES per patient",
        "classes": list(ACDC_2D_CLASSES),
        "cinema_commit": cinema_git_commit(),
        "splits": {},
    }
    for name, frame in splits.items():
        export = frame.copy()
        export["mid_sax_index_0based"] = (export["n_slices"].astype(int) - 1).clip(
            lower=0
        ) // 2
        root.mkdir(parents=True, exist_ok=True)
        export.to_csv(root / f"{name}.csv", index=False)
        audit["splits"][name] = {
            "n": len(export),
            "class_counts": export["pathology"].value_counts().sort_index().to_dict(),
            "pids": export["pid"].astype(str).tolist(),
        }
    save_json_arrays(root / "audit.json", audit)


class MidSAX2DDataset(Dataset):
    """Load one central SAX ED/ES slice for each patient."""

    def __init__(
        self,
        data_dir: Path,
        metadata: pd.DataFrame,
        transform: Any | None,
        classes: tuple[str, ...],
    ) -> None:
        self.data_dir = data_dir
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform
        self.classes = classes

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
            array = np.transpose(sitk.GetArrayFromImage(image)).astype(
                np.float32, copy=False
            )
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
            "label": torch.tensor(self.classes.index(pathology), dtype=torch.long),
            "sax_image": torch.from_numpy(image_2d),
            "slice_index": torch.tensor(slice_index, dtype=torch.long),
        }
        return self.transform(sample) if self.transform is not None else sample


class ACDCMidSAX2DDataset(MidSAX2DDataset):
    def __init__(self, data_dir: Path, metadata: pd.DataFrame, transform: Any | None):
        super().__init__(data_dir, metadata, transform, ACDC_2D_CLASSES)


def get_mid_sax_transforms(config: DictConfig) -> tuple[Any, Any]:
    """Convert CineMA's SAX augmentation settings to two spatial dimensions."""
    patch_size = tuple(int(value) for value in config.data.sax.patch_size[:2])
    rotation = float(config.transform.sax.rotate_range[-1]) / 180.0 * np.pi
    translation = tuple(
        float(value) for value in config.transform.sax.translate_range[:2]
    )
    probability = float(config.transform.prob)
    train_transform = Compose(
        [
            RandAdjustContrastd(
                keys="sax_image", prob=probability, gamma=config.transform.gamma
            ),
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
            SpatialPadd(
                keys="sax_image", spatial_size=patch_size, method="end", lazy=True
            ),
        ]
    )
    eval_transform = Compose(
        [
            ScaleIntensityd(keys="sax_image"),
            SpatialPadd(
                keys="sax_image", spatial_size=patch_size, method="end", lazy=True
            ),
        ]
    )
    return train_transform, eval_transform


def acdc_2d_split_directory(data_root: Path, split: str) -> Path:
    # CineMA derives validation patients from its training directory.
    return data_root / ("train" if split == "val" else split)


def acdc_2d_make_datasets(
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    config: DictConfig,
) -> tuple[Dataset, Dataset, Dataset]:
    train_transform, eval_transform = get_mid_sax_transforms(config)
    train = ACDCMidSAX2DDataset(
        acdc_2d_split_directory(data_root, "train"), splits["train"], train_transform
    )
    val = ACDCMidSAX2DDataset(
        acdc_2d_split_directory(data_root, "val"), splits["val"], eval_transform
    )
    test = ACDCMidSAX2DDataset(
        acdc_2d_split_directory(data_root, "test"), splits["test"], eval_transform
    )
    return train, val, test


def acdc_2d_validate_inputs(
    output_dir: Path,
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    config: DictConfig,
) -> None:
    _, val_dataset, test_dataset = acdc_2d_make_datasets(data_root, splits, config)
    rows = []
    for split_name, dataset in (("val", val_dataset), ("test", test_dataset)):
        for index in range(min(5, len(dataset))):
            sample = dataset[index]
            image = sample["sax_image"]
            if image.ndim != 3 or image.shape[0] != 2:
                raise RuntimeError(
                    f"{split_name} {sample['pid']} produced {tuple(image.shape)}; expected (2,H,W)"
                )
            if not torch.isfinite(image).all():
                raise RuntimeError(
                    f"Non-finite input for {split_name} patient {sample['pid']}"
                )
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
    audit_path = output_dir / "splits" / ACDC_2D_TASK_KEY / "input_tensor_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    print(f"[validate] Patient-level 2-D input audit passed: {audit_path}")


def acdc_2d_make_loaders(
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    config: DictConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    seed_cudnn(seed)
    train_dataset, val_dataset, test_dataset = acdc_2d_make_datasets(
        data_root, splits, config
    )
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


# Shared


def split_metadata(spec: TaskSpec, data_root: Path) -> dict[str, pd.DataFrame]:
    if spec.dataset == "acdc":
        development = pd.read_csv(data_root / "train_metadata.csv", dtype={"pid": str})
        test = pd.read_csv(data_root / "test_metadata.csv", dtype={"pid": str})
        val_pids = (
            development.groupby("pathology").sample(n=2, random_state=0)["pid"].tolist()
        )
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
    expected = {
        "train": spec.expected_train,
        "val": spec.expected_val,
        "test": spec.expected_test,
    }
    for split, frame in result.items():
        if len(frame) != expected[split]:
            raise RuntimeError(
                f"{spec.key} {split} has {len(frame)} patients; expected {expected[split]}. "
                "Do not train until the official preprocessing/data release is complete."
            )
        observed = set(frame["pathology"].unique())
        if observed != set(spec.classes):
            raise RuntimeError(
                f"{spec.key} {split} classes {sorted(observed)} != {sorted(spec.classes)}"
            )

    pid_sets = {k: set(v["pid"].astype(str)) for k, v in result.items()}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = pid_sets[first] & pid_sets[second]
        if overlap:
            raise RuntimeError(
                f"Patient leakage in {spec.key}: {first}/{second}: {sorted(overlap)}"
            )
    return result


def save_split_audit(
    args: argparse.Namespace, spec: TaskSpec, splits: dict[str, pd.DataFrame]
) -> None:
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
    val_loader = DataLoader(
        val_ds,
        sampler=SequentialSampler(val_ds),
        batch_size=1,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        sampler=SequentialSampler(test_ds),
        batch_size=1,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader


# Acdc 3D

ACDC_3D_CLASSES = protocols["acdc"]["classes"]


ACDC_3D_DATASET = protocols["acdc_3d"]["dataset"]


ACDC_3D_EXPECTED_SPLIT_SIZES = protocols["acdc"]["expected_split_sizes"]


ACDC_3D_TASK_KEY = protocols["acdc_3d"]["task_key"]


ACDC_3D_VIEW = protocols["shared"]["view"]


def acdc_3d_processed_dir(args: argparse.Namespace) -> Path:
    return processed_dir(args, "acdc")


def acdc_3d_preprocess(args: argparse.Namespace) -> None:
    preprocess_dataset(args, "acdc")


def acdc_3d_local_config(data_root: Path, seed: int) -> DictConfig:
    return load_task_config("acdc", data_root, seed)


def acdc_3d_split_metadata(data_root: Path) -> dict[str, pd.DataFrame]:
    development = pd.read_csv(data_root / "train_metadata.csv", dtype={"pid": str})
    test = pd.read_csv(data_root / "test_metadata.csv", dtype={"pid": str})
    required_columns = {"pid", "pathology", "n_slices"}
    for name, frame in (("development", development), ("test", test)):
        missing = required_columns - set(frame.columns)
        if missing:
            raise RuntimeError(f"ACDC {name} metadata is missing {sorted(missing)}")

    validation_pids = (
        development.groupby("pathology", group_keys=False)
        .sample(n=2, random_state=0)["pid"]
        .tolist()
    )
    train = development[~development["pid"].isin(validation_pids)].reset_index(
        drop=True
    )
    val = development[development["pid"].isin(validation_pids)].reset_index(drop=True)
    splits = {"train": train, "val": val, "test": test.reset_index(drop=True)}

    for name, frame in splits.items():
        expected = ACDC_3D_EXPECTED_SPLIT_SIZES[name]
        if len(frame) != expected:
            raise RuntimeError(
                f"{ACDC_3D_TASK_KEY} {name} contains {len(frame)} patients; expected {expected}"
            )
        observed = set(frame["pathology"].astype(str).unique())
        if observed != set(ACDC_3D_CLASSES):
            raise RuntimeError(
                f"{ACDC_3D_TASK_KEY} {name} classes {sorted(observed)} != {sorted(ACDC_3D_CLASSES)}"
            )

    patient_sets = {
        name: set(frame["pid"].astype(str)) for name, frame in splits.items()
    }
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patient_sets[first] & patient_sets[second]
        if overlap:
            raise RuntimeError(
                f"Patient leakage in {ACDC_3D_TASK_KEY}: {first}/{second}: {sorted(overlap)}"
            )
    return splits


def acdc_3d_save_split_audit(
    args: argparse.Namespace, splits: dict[str, pd.DataFrame]
) -> None:
    root = args.output_dir / "splits" / ACDC_3D_TASK_KEY
    root.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "dataset": ACDC_3D_DATASET,
        "task": ACDC_3D_TASK_KEY,
        "view": ACDC_3D_VIEW,
        "dimensionality": 3,
        "input_policy": "full_sax_ed_es_volume_per_patient",
        "classes": list(ACDC_3D_CLASSES),
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


def acdc_3d_make_dataset(
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
        views=ACDC_3D_VIEW,
        transform=train_transform if train else evaluation_transform,
    )


def acdc_3d_make_loaders(
    config: DictConfig,
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    seed: int,
    deterministic: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    seed_deterministic(seed, deterministic)
    train_dataset = acdc_3d_make_dataset(
        config, data_root, splits["train"], "train", True
    )
    val_dataset = acdc_3d_make_dataset(config, data_root, splits["val"], "val", False)
    test_dataset = acdc_3d_make_dataset(
        config, data_root, splits["test"], "test", False
    )
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


def acdc_3d_validate_inputs(args: argparse.Namespace) -> None:
    data_root = acdc_3d_processed_dir(args)
    splits = acdc_3d_split_metadata(data_root)
    acdc_3d_save_split_audit(args, splits)
    config = acdc_3d_local_config(data_root, seed=0)
    apply_training_overrides(args, config)
    _, val_loader, test_loader = acdc_3d_make_loaders(
        config, data_root, splits, seed=0, deterministic=True
    )
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
            if any(
                size < patch
                for size, patch in zip(spatial_shape, patch_size, strict=True)
            ):
                raise RuntimeError(
                    f"{split_name} patient {batch['pid'][0]} shape {spatial_shape} "
                    f"is smaller than evaluation patch {patch_size}"
                )
            if not torch.isfinite(image).all():
                raise RuntimeError(
                    f"Non-finite input for {split_name} patient {batch['pid'][0]}"
                )
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
    audit_path = (
        args.output_dir / "splits" / ACDC_3D_TASK_KEY / "input_tensor_audit.csv"
    )
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    print(f"[validate] ACDC split and full-volume tensor contract passed: {audit_path}")


# Mnms2 3D

MNMS2_3D_CLASSES = protocols["mnms2_3d"]["classes"]


MNMS2_3D_DATASET = protocols["mnms2_3d"]["dataset"]


MNMS2_3D_EXPECTED_SPLIT_SIZES = protocols["mnms2_3d"]["expected_split_sizes"]


MNMS2_3D_TASK_KEY = protocols["mnms2_3d"]["task_key"]


MNMS2_3D_VIEW = protocols["shared"]["view"]


def mnms2_3d_processed_dir(args: argparse.Namespace) -> Path:
    return processed_dir(args, "mnms2")


def mnms2_3d_preprocess(args: argparse.Namespace) -> None:
    preprocess_dataset(args, "mnms2")


def mnms2_3d_local_config(data_root: Path, seed: int) -> DictConfig:
    return load_task_config("mnms2", data_root, seed)


def mnms2_3d_split_metadata(data_root: Path) -> dict[str, pd.DataFrame]:
    train = pd.read_csv(data_root / "train_metadata.csv", dtype={"pid": str})
    val = pd.read_csv(data_root / "val_metadata.csv", dtype={"pid": str})
    test = pd.read_csv(data_root / "test_metadata.csv", dtype={"pid": str})
    required_columns = {"pid", "pathology", "n_slices"}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        missing = required_columns - set(frame.columns)
        if missing:
            raise RuntimeError(f"M&Ms2 {name} metadata is missing {sorted(missing)}")

    splits = {
        "train": train[train["pathology"].isin(MNMS2_3D_CLASSES)].reset_index(
            drop=True
        ),
        "val": val[val["pathology"].isin(MNMS2_3D_CLASSES)].reset_index(drop=True),
        "test": test[test["pathology"].isin(MNMS2_3D_CLASSES)].reset_index(drop=True),
    }

    for name, frame in splits.items():
        expected = MNMS2_3D_EXPECTED_SPLIT_SIZES[name]
        if len(frame) != expected:
            raise RuntimeError(
                f"{MNMS2_3D_TASK_KEY} {name} contains {len(frame)} patients; expected {expected}"
            )
        observed = set(frame["pathology"].astype(str).unique())
        if observed != set(MNMS2_3D_CLASSES):
            raise RuntimeError(
                f"{MNMS2_3D_TASK_KEY} {name} classes {sorted(observed)} != {sorted(MNMS2_3D_CLASSES)}"
            )

    patient_sets = {
        name: set(frame["pid"].astype(str)) for name, frame in splits.items()
    }
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patient_sets[first] & patient_sets[second]
        if overlap:
            raise RuntimeError(
                f"Patient leakage in {MNMS2_3D_TASK_KEY}: {first}/{second}: {sorted(overlap)}"
            )
    return splits


def mnms2_3d_save_split_audit(
    args: argparse.Namespace, splits: dict[str, pd.DataFrame]
) -> None:
    root = args.output_dir / "splits" / MNMS2_3D_TASK_KEY
    root.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "dataset": MNMS2_3D_DATASET,
        "task": MNMS2_3D_TASK_KEY,
        "view": MNMS2_3D_VIEW,
        "dimensionality": 3,
        "input_policy": "full_sax_ed_es_volume_per_patient",
        "classes": list(MNMS2_3D_CLASSES),
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


def mnms2_3d_make_dataset(
    config: DictConfig,
    data_root: Path,
    frame: pd.DataFrame,
    split: str,
    train: bool,
) -> EndDiastoleEndSystoleDataset:
    train_transform, evaluation_transform = get_image_transforms(config)
    return EndDiastoleEndSystoleDataset(
        data_dir=data_root / split,
        meta_df=frame,
        class_col=config.data.class_column,
        classes=list(config.data[config.data.class_column]),
        views=MNMS2_3D_VIEW,
        transform=train_transform if train else evaluation_transform,
    )


def mnms2_3d_make_loaders(
    config: DictConfig,
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    seed: int,
    deterministic: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    seed_deterministic(seed, deterministic)
    train_dataset = mnms2_3d_make_dataset(
        config, data_root, splits["train"], "train", True
    )
    val_dataset = mnms2_3d_make_dataset(config, data_root, splits["val"], "val", False)
    test_dataset = mnms2_3d_make_dataset(
        config, data_root, splits["test"], "test", False
    )
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


def mnms2_3d_validate_inputs(args: argparse.Namespace) -> None:
    data_root = mnms2_3d_processed_dir(args)
    splits = mnms2_3d_split_metadata(data_root)
    mnms2_3d_save_split_audit(args, splits)
    config = mnms2_3d_local_config(data_root, seed=0)
    _, val_loader, test_loader = mnms2_3d_make_loaders(
        config, data_root, splits, seed=0, deterministic=True
    )
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
            if any(
                size < patch
                for size, patch in zip(spatial_shape, patch_size, strict=True)
            ):
                raise RuntimeError(
                    f"{split_name} patient {batch['pid'][0]} shape {spatial_shape} "
                    f"is smaller than evaluation patch {patch_size}"
                )
            if not torch.isfinite(image).all():
                raise RuntimeError(
                    f"Non-finite input for {split_name} patient {batch['pid'][0]}"
                )
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
    audit_path = (
        args.output_dir / "splits" / MNMS2_3D_TASK_KEY / "input_tensor_audit.csv"
    )
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    print(
        f"[validate] M&Ms2 split and full-volume tensor contract passed: {audit_path}"
    )


# Mnms2 Sax 2D

MNMS2_SAX_2D_CLASSES = protocols["mnms2_sax_2d"]["classes"]


MNMS2_SAX_2D_EXPECTED_SPLIT_SIZES = protocols["mnms2_sax_2d"]["expected_split_sizes"]


MNMS2_SAX_2D_TASK_KEY = protocols["mnms2_sax_2d"]["task_key"]


def mnms2_sax_2d_resolve_processed_dir(args: argparse.Namespace) -> Path:
    return processed_dir(args, "mnms2")


def preprocess_mnms2(args: argparse.Namespace) -> None:
    preprocess_dataset(args, "mnms2")


def load_mnms2_config(data_root: Path, seed: int) -> DictConfig:
    return load_task_config("mnms2", data_root, seed)


def mnms2_sax_2d_load_official_splits(data_root: Path) -> dict[str, pd.DataFrame]:
    return mnms2_3d_split_metadata(data_root)


def mnms2_sax_2d_save_split_audit(
    output_dir: Path, splits: dict[str, pd.DataFrame]
) -> None:
    root = output_dir / "splits" / MNMS2_SAX_2D_TASK_KEY
    audit: dict[str, Any] = {
        "task": MNMS2_SAX_2D_TASK_KEY,
        "dataset": "M&Ms2",
        "input_policy": "one deterministic central SAX slice at ED and ES per patient",
        "classes": list(MNMS2_SAX_2D_CLASSES),
        "cinema_commit": cinema_git_commit(),
        "splits": {},
    }
    for name, frame in splits.items():
        export = frame.copy()
        export["mid_sax_index_0based"] = (export["n_slices"].astype(int) - 1).clip(
            lower=0
        ) // 2
        root.mkdir(parents=True, exist_ok=True)
        export.to_csv(root / f"{name}.csv", index=False)
        audit["splits"][name] = {
            "n": len(export),
            "class_counts": export["pathology"].value_counts().sort_index().to_dict(),
            "pids": export["pid"].astype(str).tolist(),
        }
    save_json_arrays(root / "audit.json", audit)


class MnMs2MidSAX2DDataset(MidSAX2DDataset):
    def __init__(self, data_dir: Path, metadata: pd.DataFrame, transform: Any | None):
        super().__init__(data_dir, metadata, transform, MNMS2_SAX_2D_CLASSES)


def mnms2_sax_2d_split_directory(data_root: Path, split: str) -> Path:
    return data_root / split


def mnms2_sax_2d_make_datasets(
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    config: DictConfig,
) -> tuple[Dataset, Dataset, Dataset]:
    train_transform, eval_transform = get_mid_sax_transforms(config)
    train = MnMs2MidSAX2DDataset(
        mnms2_sax_2d_split_directory(data_root, "train"),
        splits["train"],
        train_transform,
    )
    val = MnMs2MidSAX2DDataset(
        mnms2_sax_2d_split_directory(data_root, "val"), splits["val"], eval_transform
    )
    test = MnMs2MidSAX2DDataset(
        mnms2_sax_2d_split_directory(data_root, "test"), splits["test"], eval_transform
    )
    return train, val, test


def mnms2_sax_2d_validate_inputs(
    output_dir: Path,
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    config: DictConfig,
) -> None:
    _, val_dataset, test_dataset = mnms2_sax_2d_make_datasets(data_root, splits, config)
    rows = []
    for split_name, dataset in (("val", val_dataset), ("test", test_dataset)):
        for index in range(min(5, len(dataset))):
            sample = dataset[index]
            image = sample["sax_image"]
            if image.ndim != 3 or image.shape[0] != 2:
                raise RuntimeError(
                    f"{split_name} {sample['pid']} produced {tuple(image.shape)}; expected (2,H,W)"
                )
            if not torch.isfinite(image).all():
                raise RuntimeError(
                    f"Non-finite input for {split_name} patient {sample['pid']}"
                )
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
    audit_path = (
        output_dir / "splits" / MNMS2_SAX_2D_TASK_KEY / "input_tensor_audit.csv"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    print(f"[validate] Patient-level 2-D input audit passed: {audit_path}")


def mnms2_sax_2d_make_loaders(
    data_root: Path,
    splits: dict[str, pd.DataFrame],
    config: DictConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    seed_cudnn(seed)
    train_dataset, val_dataset, test_dataset = mnms2_sax_2d_make_datasets(
        data_root, splits, config
    )
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
