"""Exercise every CNN training route with small CPU tensors and real exports."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src import distillation, train
from src.cinema_support import load_task_config
from src.protocol import TASKS
from src.utils import jsonable, prepare_run

ROOT = Path(__file__).resolve().parents[1]


class SmallClassifier(torch.nn.Module):
    def __init__(self, shape, n_classes):
        super().__init__()
        self.feature_dim = int(np.prod(shape))
        self.classifier = torch.nn.Linear(self.feature_dim, n_classes)

    def forward(self, image):
        return self.classifier(image.flatten(1))


def small_task_config(dataset, data_root, seed=0, view="sax"):
    config = load_task_config(dataset, data_root, seed, view)
    config.data.sax.patch_size = [8, 8, 4]
    if "lax" in config.data:
        config.data.lax.patch_size = [8, 8]
    return config


class WorkflowTests(unittest.TestCase):
    def test_train_evaluate_export_and_reuse(self):
        cases = [
            ("acdc_2d", "acdc", 2, "ACDC2DClassifier"),
            ("mnms2_sax_2d", "mnms2", 2, "MnMs2SAXMid2DClassifier"),
            ("acdc_3d", "acdc", 3, "ACDC3DClassifier"),
            ("mnms2_3d", "mnms2", 3, "MnMs2SAX3DClassifier"),
            ("architecture_bank", "mnms2", 2, "ArchitectureBank2DClassifier"),
            ("architecture_bank", "mnms2", 3, "ArchitectureBank3DClassifier"),
            ("reproduction", "mnms2", 2, "Reproduction2DClassifier"),
            ("acdc_distillation", "acdc", 3, "ACDC3DClassifier"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (experiment, dataset, dimensions, class_name) in enumerate(
                cases
            ):
                with self.subTest(experiment=experiment, dimensions=dimensions):
                    config = OmegaConf.load(ROOT / "configs" / f"{experiment}.yaml")
                    config.run.smoke = True
                    config.run.amp = False
                    config.run.seeds = [0]
                    config.logging.dir = str(root / f"case_{index}")
                    data_root = root / "data"
                    data_root.mkdir(exist_ok=True)
                    data_setting = (
                        "processed_dir"
                        if "processed_dir" in config.data
                        else f"{dataset}_processed"
                    )
                    config.data[data_setting] = str(data_root)
                    args = prepare_run(config, experiment)
                    n_classes = len(TASKS[f"{dataset}_sax"].classes)
                    shape = (2, 8, 8) if dimensions == 2 else (2, 8, 8, 4)
                    view = (
                        "lax_4c"
                        if experiment in ("architecture_bank", "reproduction")
                        and dimensions == 2
                        else "sax"
                    )
                    batches = [
                        {
                            "pid": [f"p{i}"],
                            f"{view}_image": torch.randn(1, *shape),
                            "label": torch.tensor([i]),
                        }
                        for i in range(n_classes)
                    ]
                    splits = {
                        "train": pd.DataFrame(
                            {"pid": [f"p{i}" for i in range(n_classes)]}
                        )
                    }
                    module = (
                        distillation if experiment == "acdc_distillation" else train
                    )
                    with ExitStack() as stack:
                        stack.enter_context(
                            patch.object(
                                module,
                                "amp_dtype_and_device",
                                return_value=(torch.float32, torch.device("cpu")),
                            )
                        )
                        model_name = (
                            "SAX2DClassifier"
                            if experiment in ("acdc_2d", "mnms2_sax_2d")
                            else class_name
                        )
                        stack.enter_context(
                            patch.object(
                                module,
                                model_name,
                                return_value=SmallClassifier(shape, n_classes),
                            )
                        )
                        if experiment in ("acdc_2d", "mnms2_sax_2d"):
                            stack.enter_context(
                                patch.object(
                                    train,
                                    "load_task_config",
                                    side_effect=small_task_config,
                                )
                            )
                            prefix = "acdc_2d" if dataset == "acdc" else "mnms2_sax_2d"
                            stack.enter_context(
                                patch.object(
                                    train,
                                    f"{prefix}_make_loaders",
                                    return_value=(batches, batches, batches),
                                )
                            )
                            function = getattr(train, f"{prefix}_train_one_run")
                            call_args = (
                                args,
                                data_root,
                                splits,
                                "resnet18",
                                "stacked",
                                "randinit",
                                0,
                            )
                        elif experiment in ("acdc_3d", "mnms2_3d", "acdc_distillation"):
                            prefix = "acdc_3d" if dataset == "acdc" else "mnms2_3d"
                            stack.enter_context(
                                patch.object(
                                    module,
                                    f"{prefix}_local_config",
                                    side_effect=lambda data, seed: small_task_config(
                                        dataset, data, seed
                                    ),
                                )
                            )
                            stack.enter_context(
                                patch.object(
                                    module,
                                    f"{prefix}_split_metadata",
                                    return_value=splits,
                                )
                            )
                            stack.enter_context(
                                patch.object(
                                    module,
                                    f"{prefix}_make_loaders",
                                    return_value=(batches, batches, batches),
                                )
                            )
                            function = (
                                distillation.acdc_distillation_train_one
                                if experiment == "acdc_distillation"
                                else getattr(train, f"{prefix}_train_one")
                            )
                            if experiment == "acdc_distillation":
                                teacher = root / "teacher"
                                teacher.mkdir()
                                checkpoint = teacher / "acdc_sax_0.safetensors"
                                checkpoint.touch()
                                (teacher / "config.yaml").write_text("teacher: test\n")
                                args.teacher_dir = teacher
                                targets = {
                                    f"p{i}": torch.zeros(n_classes)
                                    for i in range(n_classes)
                                }
                                stack.enter_context(
                                    patch.object(
                                        distillation,
                                        "get_teacher_targets",
                                        return_value=(targets, 0, checkpoint),
                                    )
                                )
                            call_args = (args, "resnet10_3d", "stacked", 0)
                        else:
                            spec = TASKS[
                                "mnms2_lax_4c" if dimensions == 2 else "mnms2_sax"
                            ]
                            stack.enter_context(
                                patch.object(
                                    train,
                                    "local_config",
                                    side_effect=lambda spec,
                                    data,
                                    seed: small_task_config(
                                        spec.dataset, data, seed, spec.view
                                    ),
                                )
                            )
                            stack.enter_context(
                                patch.object(
                                    train, "split_metadata", return_value=splits
                                )
                            )
                            stack.enter_context(
                                patch.object(
                                    train,
                                    "make_loaders",
                                    return_value=(batches, batches, batches),
                                )
                            )
                            function = (
                                train.train_cnn
                                if experiment == "reproduction"
                                else train.train_architecture
                            )
                            backbone = "resnet18" if dimensions == 2 else "resnet10_3d"
                            call_args = (args, spec, backbone, "stacked", "randinit", 0)
                        result = function(*call_args)
                        self.assertEqual(result["best_epoch"], 1)
                        self.assertTrue(Path(result["checkpoint"]).is_file())
                        self.assertEqual(
                            jsonable(function(*call_args)), jsonable(result)
                        )


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
