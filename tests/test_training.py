"""Regression checks for CineMA reuse, CNN training and result identity (CPU only)."""

from __future__ import annotations

import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.cinema_support import (
    load_task_config,
    preprocess_dataset,
    run_official_training,
)
from src.dataset import ACDCMidSAX2DDataset, MnMs2MidSAX2DDataset, split_metadata
from src.protocol import TASKS, protocols
from src.train import fit_classifier, train_epoch
from src.utils import (
    acdc_3d_run_dir_for,
    checkpoint_root,
    grad_scaler,
    prepare_run_directory,
    save_json,
)


def small_config(directory: Path):
    return OmegaConf.create(
        {
            "data": {"dir": str(directory)},
            "train": {
                "n_epochs": 3,
                "n_warmup_epochs": 0,
                "eval_interval": 1,
                "lr": 0.05,
                "min_lr": 0.05,
                "betas": [0.9, 0.95],
                "weight_decay": 0.01,
                "label_smoothing": 0.1,
                "clip_grad": 1000000,
                "batch_size": 2,
                "batch_size_per_device": 1,
                "early_stopping": {"patience": 5, "min_delta": 0.0001},
            },
        }
    )


class TrainingTests(unittest.TestCase):
    def test_partial_accumulation_matches_an_explicit_group_average(self):
        for n_batches, n_accum in ((5, 2), (40, 16)):
            with self.subTest(n_batches=n_batches, accumulation=n_accum):
                torch.manual_seed(7)
                model = torch.nn.Linear(2, 2, bias=False)
                reference = copy.deepcopy(model)
                batches = [
                    {
                        "sax_image": torch.tensor([[i / 10, 1.0]]),
                        "label": torch.tensor([i % 2]),
                    }
                    for i in range(n_batches)
                ]
                config = small_config(Path("/tmp"))
                optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
                reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
                with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    train_epoch(
                        model,
                        batches,
                        optimizer,
                        grad_scaler(False),
                        config,
                        torch.device("cpu"),
                        torch.float32,
                        False,
                        0,
                        n_accum,
                    )
                self.assertEqual(step.call_count, (n_batches + n_accum - 1) // n_accum)
                for start in range(0, n_batches, n_accum):
                    group = batches[start : start + n_accum]
                    reference_optimizer.zero_grad(set_to_none=True)
                    loss = torch.stack(
                        [
                            F.cross_entropy(
                                reference(b["sax_image"]),
                                b["label"],
                                label_smoothing=0.1,
                            )
                            for b in group
                        ]
                    ).mean()
                    loss.backward()
                    reference_optimizer.step()
                torch.testing.assert_close(model.weight, reference.weight)

    def test_best_checkpoint_and_final_epoch_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = small_config(root)
            batches = [
                {"sax_image": torch.tensor([[1.0, 2.0]]), "label": torch.tensor([0])}
            ]
            for interval, expected_epoch in ((1, 2), (20, 3)):
                with self.subTest(eval_interval=interval):
                    config.train.eval_interval = interval
                    run_dir = root / str(interval)
                    run_dir.mkdir()
                    model = torch.nn.Linear(2, 2)
                    scores = iter([0.2, 0.6, 0.3])

                    def evaluate(current_model, loader):
                        metrics = {
                            name: 0.5 for name in protocols["shared"]["metric_names"]
                        }
                        metrics["mcc"] = next(scores)
                        return ["p"], np.array([0]), np.array([[0.5, 0.5]]), metrics

                    exporter = Mock()
                    best, epoch = fit_classifier(
                        model,
                        batches,
                        batches,
                        config,
                        run_dir,
                        torch.device("cpu"),
                        torch.float32,
                        False,
                        {"classes": ["A", "B"]},
                        evaluate,
                        exporter,
                    )
                    self.assertEqual(epoch, expected_epoch)
                    self.assertEqual(exporter.call_count, 2 if interval == 1 else 1)
                    checkpoint = torch.load(
                        run_dir / "best_val_mcc.pt", weights_only=False
                    )
                    self.assertEqual(checkpoint["epoch"], expected_epoch)
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(value, checkpoint["model"][name])
                    history = pd.read_csv(run_dir / "history.csv")
                    self.assertEqual(history["epoch"].iloc[-1], 3)

    def test_reuse_rejects_changed_settings_and_preserves_existing_results(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("src.utils.cinema_git_commit", return_value="tested"),
        ):
            root = Path(directory)
            args = SimpleNamespace(
                output_dir=root, experiment="acdc_2d", amp=False, resume=True
            )
            config = small_config(root)
            run = checkpoint_root(args) / "resnet18/seed_0"
            self.assertIsNone(prepare_run_directory(args, run, config, seed=0))
            checkpoint = run / "best_val_mcc.pt"
            checkpoint.write_bytes(b"test checkpoint")
            expected = {"checkpoint": str(checkpoint), "best_epoch": 1}
            save_json(run / "summary.json", expected)
            self.assertEqual(prepare_run_directory(args, run, config, seed=0), expected)
            original = (run / "summary.json").read_text()
            for key, value in (
                ("train.lr", 0.1),
                ("train.n_epochs", 100),
                ("data.dir", str(root / "other")),
            ):
                changed = OmegaConf.merge(
                    config, OmegaConf.from_dotlist([f"{key}={json.dumps(value)}"])
                )
                with self.subTest(setting=key), self.assertRaises(FileExistsError):
                    prepare_run_directory(args, run, changed, seed=0)
            (root / "train_metadata.csv").write_text("pid,pathology\np1,NOR\n")
            with self.assertRaises(FileExistsError):
                prepare_run_directory(args, run, config, seed=0)
            self.assertEqual((run / "summary.json").read_text(), original)
            self.assertEqual(checkpoint.read_bytes(), b"test checkpoint")
            legacy = root / "legacy"
            legacy.mkdir()
            save_json(legacy / "summary.json", expected)
            with self.assertRaises(FileExistsError):
                prepare_run_directory(args, legacy, config)

    def test_experiment_paths_do_not_collide(self):
        args = SimpleNamespace(
            output_dir=Path("outputs"), experiment="architecture_bank", smoke=False
        )
        bank = (
            checkpoint_root(args)
            / "architecture_bank/3d/acdc_sax/resnet18_3d/stacked/randinit/seed_0"
        )
        baseline = acdc_3d_run_dir_for(args, "resnet18_3d", "stacked", 0)
        self.assertNotEqual(bank, baseline)
        self.assertIn("acdc_3d", baseline.parts)


class CineMAReuseTests(unittest.TestCase):
    def test_task_defaults_and_patient_splits_match_cinema(self):
        for dataset in ("acdc", "mnms2"):
            with (
                self.subTest(dataset=dataset),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                spec = TASKS[f"{dataset}_sax"]
                counts = {
                    "train": 100 if dataset == "acdc" else 160,
                    "test": spec.expected_test,
                }
                if dataset == "mnms2":
                    counts["val"] = spec.expected_val
                for split, count in counts.items():
                    pd.DataFrame(
                        {
                            "pid": [f"{split}{i:03}" for i in range(count)],
                            "pathology": [
                                spec.classes[i % len(spec.classes)]
                                for i in range(count)
                            ],
                            "n_slices": [16] * count,
                        }
                    ).to_csv(root / f"{split}_metadata.csv", index=False)
                config = load_task_config(dataset, root, seed=7)
                package = importlib.import_module(f"cinema.classification.{dataset}")
                original = OmegaConf.load(Path(package.__file__).parent / "config.yaml")
                self.assertEqual(
                    OmegaConf.to_container(config.train),
                    OmegaConf.to_container(original.train),
                )
                self.assertEqual(
                    OmegaConf.to_container(config.transform),
                    OmegaConf.to_container(original.transform),
                )
                loader = getattr(
                    importlib.import_module(f"cinema.classification.{dataset}.train"),
                    f"load_{dataset}_dataset",
                )
                native_train, native_val = loader(config)
                splits = split_metadata(spec, root)
                self.assertEqual(
                    splits["train"]["pid"].tolist(),
                    native_train.meta_df["pid"].tolist(),
                )
                self.assertEqual(
                    splits["val"]["pid"].tolist(), native_val.meta_df["pid"].tolist()
                )

    def test_shared_slice_loader_preserves_phases_and_class_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "p1").mkdir()
            for phase, value in (("ed", 3), ("es", 6)):
                volume = np.zeros((8, 8, 3), dtype=np.float32)
                volume[..., 1] = value
                sitk.WriteImage(
                    sitk.GetImageFromArray(volume.transpose()),
                    str(root / "p1" / f"p1_sax_{phase}.nii.gz"),
                )
            frame = pd.DataFrame({"pid": ["p1"], "pathology": ["NOR"], "n_slices": [3]})
            for cls, classes in (
                (ACDCMidSAX2DDataset, TASKS["acdc_sax"].classes),
                (MnMs2MidSAX2DDataset, TASKS["mnms2_sax"].classes),
            ):
                sample = cls(root, frame, None)[0]
                self.assertEqual(tuple(sample["sax_image"].shape), (2, 8, 8))
                self.assertEqual(sample["slice_index"].item(), 1)
                self.assertEqual(sample["label"].item(), classes.index("NOR"))
                torch.testing.assert_close(
                    sample["sax_image"][0], torch.full((8, 8), 3.0)
                )
                torch.testing.assert_close(
                    sample["sax_image"][1], torch.full((8, 8), 6.0)
                )

    def test_preprocessing_uses_the_original_cinema_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw with spaces"
            (raw / "training/patient001").mkdir(parents=True)
            (raw / "testing").mkdir()
            (raw / "training/patient001/Info.cfg").touch()
            args = SimpleNamespace(
                acdc_raw=raw, acdc_processed=root / "processed", force_preprocess=False
            )
            with patch("src.cinema_support.subprocess.check_call") as command:
                preprocess_dataset(args, "acdc")
            argv = command.call_args.args[0]
            self.assertEqual(argv[1:3], ["-m", "cinema.data.acdc.preprocess"])
            self.assertEqual(argv[4], str(raw))

    def test_official_trainer_receives_saved_defaults_and_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                output_dir=root,
                experiment="reproduction",
                amp=False,
                resume=True,
                smoke=True,
                acdc_processed=root / "data",
            )

            def command(argv):
                if "cinema.classification.acdc.train" in argv:
                    run = Path(argv[argv.index("--config-path") + 1])
                    config = OmegaConf.load(run / "cinema_config.yaml")
                    self.assertEqual(config.train.n_epochs, 1)
                    self.assertEqual(config.train.n_warmup_epochs, 0)
                    self.assertEqual(config.data.max_n_samples, 24)
                    self.assertEqual(config.model.name, "resnet")
                    (run / "ckpt").mkdir()
                    (run / "ckpt/ckpt_1.pt").touch()
                else:
                    checkpoint = Path(argv[argv.index("--ckpt_path") + 1])
                    destination = checkpoint.parent / "acdc_eval_ckpt_1/test"
                    destination.mkdir(parents=True)
                    pd.DataFrame({"mcc": [0.5]}).to_csv(
                        destination / "classification_metrics.csv", index=False
                    )

            with patch(
                "src.cinema_support.subprocess.check_call", side_effect=command
            ) as subprocess:
                run_official_training(args, TASKS["acdc_sax"], "resnet50_randinit", 0)
                run_official_training(args, TASKS["acdc_sax"], "resnet50_randinit", 0)
                self.assertEqual(subprocess.call_count, 2)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
