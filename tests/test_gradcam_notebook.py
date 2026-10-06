"""Check notebook preprocessing, Grad-CAM, hook cleanup, and PNG export on CPU."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from cinema.classification.dataset import (
    EndDiastoleEndSystoleDataset,
    get_image_transforms,
)

from src.dataset import (
    ACDCMidSAX2DDataset,
    acdc_2d_load_official_splits,
    acdc_2d_split_directory,
    get_mid_sax_transforms,
    make_dataset,
    split_metadata,
)
from src.protocol import TASKS, protocols

ROOT = Path(__file__).resolve().parents[1]


class GradCAMNotebookTests(unittest.TestCase):
    def test_notebook_workflow(self):
        plt.switch_backend("Agg")
        notebook = json.loads((ROOT / "notebooks/cmri-gradcma.ipynb").read_text())
        namespace = dict(
            globals(),
            F=F,
            acdc_2d_split_directory=acdc_2d_split_directory,
            DictConfig=DictConfig,
            EndDiastoleEndSystoleDataset=EndDiastoleEndSystoleDataset,
            get_image_transforms=get_image_transforms,
            split_metadata=split_metadata,
            TASKS=TASKS,
        )
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                tree = ast.parse("".join(cell["source"]))
                definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
                exec(
                    compile(
                        ast.Module(body=definitions, type_ignores=[]),
                        "<notebook>",
                        "exec",
                    ),
                    namespace,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, count in (("train", 100), ("test", 50)):
                pd.DataFrame(
                    {
                        "pid": [f"{split}{i:03}" for i in range(count)],
                        "pathology": [
                            protocols["acdc"]["classes"][i % 5] for i in range(count)
                        ],
                        "n_slices": 5,
                    }
                ).to_csv(root / f"{split}_metadata.csv", index=False)
            selected = acdc_2d_load_official_splits(root)["val"]
            selected = selected[selected["pathology"] == "HCM"].head(1)
            patient = root / "train" / selected.iloc[0]["pid"]
            patient.mkdir(parents=True)
            for index, (phase, depth) in enumerate((("ed", 5), ("es", 3))):
                image = np.arange(depth * 6 * 7, dtype=np.float32).reshape(depth, 6, 7)
                image += index * 1000
                mask = np.zeros_like(image, dtype=np.uint8)
                mask[1] = index + 1
                for suffix, array in (("", image), ("_gt", mask)):
                    sitk.WriteImage(
                        sitk.GetImageFromArray(array),
                        str(patient / f"{patient.name}_sax_{phase}{suffix}.nii.gz"),
                    )
            config = OmegaConf.create(
                {
                    "data": {"sax": {"patch_size": [8, 8, 4]}},
                    "transform": {
                        "sax": {
                            "rotate_range": [0, 0, 0],
                            "translate_range": [0, 0, 0],
                        },
                        "prob": 0,
                        "gamma": [1, 1],
                        "scale_range": 0,
                    },
                }
            )
            load_samples = namespace["load_samples"]
            acdc_task = protocols["acdc_2d"]["task_key"]
            (sample,) = load_samples(root, config, acdc_task, "val", 1, "hcm")
            raw = ACDCMidSAX2DDataset(root / "train", selected, None)[0]
            _, transform = get_mid_sax_transforms(config)
            np.testing.assert_array_equal(
                sample["image"], transform(raw)["sax_image"].numpy()
            )
            self.assertEqual(sample["image"].shape, (2, 8, 8))
            np.testing.assert_array_equal(sample["mask"][0, :7, :6], 1)
            np.testing.assert_array_equal(sample["mask"][1, :7, :6], 2)
            self.assertFalse(sample["mask"][:, 7:, :].any())
            with self.assertRaisesRegex(ValueError, "positive integer"):
                load_samples(root, config, acdc_task, n_samples=0)
            with self.assertRaisesRegex(ValueError, "Unknown split"):
                load_samples(root, config, acdc_task, split="other")
            with self.assertRaisesRegex(ValueError, "Unknown .* class"):
                load_samples(root, config, acdc_task, class_name="other")

            model = torch.nn.Sequential(
                torch.nn.Conv2d(2, 1, 1, bias=False),
                torch.nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(1, 5, bias=False),
            ).eval()
            with torch.no_grad():
                model[0].weight.fill_(1)
                model[-1].weight.copy_(torch.arange(1, 6).reshape(5, 1))
            model.requires_grad_(False)
            gradcam = namespace["gradcam"]
            with torch.no_grad():
                heatmap, class_index, probability = gradcam(sample, model, model[1])
            expected = sample["image"].sum(axis=0)
            np.testing.assert_allclose(heatmap, expected / expected.max(), atol=1e-6)
            self.assertEqual(class_index, 4)
            self.assertTrue(0 < probability <= 1)
            self.assertFalse(model[1]._forward_hooks)
            np.testing.assert_array_equal(gradcam(sample, model, model[1])[0], heatmap)
            with patch.object(
                model[-1], "forward", side_effect=RuntimeError("failure")
            ):
                with self.assertRaisesRegex(RuntimeError, "failure"):
                    gradcam(sample, model, model[1])
            self.assertFalse(model[1]._forward_hooks)
            with torch.no_grad():
                model[-1].weight.zero_()
            self.assertFalse(gradcam(sample, model, model[1])[0].any())
            self.assertTrue(
                all(parameter.grad is None for parameter in model.parameters())
            )

            figures = plt.get_fignums()
            path = root / "overlay.png"
            namespace["save_image"](
                sample["image"][0], path, heatmap, cmap="jet", alpha=0.45
            )
            self.assertTrue(path.is_file())
            self.assertEqual(plt.get_fignums(), figures)

            # LAX metadata has its own splits; the single plane is unrelated to
            # n_slices, which describes the patient's SAX stack.
            lax_root = root / "mnms2"
            lax_root.mkdir()
            spec = TASKS["mnms2_lax_4c"]
            offset = 0
            for split, count in (
                ("train", spec.expected_train),
                ("val", spec.expected_val),
                ("test", spec.expected_test),
            ):
                pd.DataFrame(
                    {
                        "pid": [f"{offset + i:03}" for i in range(count)],
                        "pathology": [spec.classes[i % 6] for i in range(count)],
                        "n_slices": 9,
                    }
                ).to_csv(lax_root / f"{split}_metadata.csv", index=False)
                offset += count
            selected = split_metadata(spec, lax_root)["train"]
            selected = selected[selected["pathology"] == "HCM"].head(1)
            patient = lax_root / "train" / selected.iloc[0]["pid"]
            patient.mkdir(parents=True)
            for index, phase in enumerate(("ed", "es")):
                image = np.arange(35, dtype=np.float32).reshape(1, 5, 7)
                image += index * 1000
                mask = np.full(image.shape, index + 1, dtype=np.uint8)
                for suffix, array in (("", image), ("_gt", mask)):
                    sitk.WriteImage(
                        sitk.GetImageFromArray(array),
                        str(patient / f"{patient.name}_lax_4c_{phase}{suffix}.nii.gz"),
                    )
            config.model = {"views": "lax_4c"}
            config.data.lax = {"patch_size": [8, 8]}
            config.data.class_column = "pathology"
            config.data.pathology = list(spec.classes)
            config.transform.lax = {"rotate_range": [0], "translate_range": [0, 0]}
            (lax_sample,) = load_samples(lax_root, config, spec.key, "train", 1, "hcm")
            expected = make_dataset(spec, config, lax_root, selected, "train", False)[0]
            self.assertEqual(lax_sample["pid"], "003")
            self.assertEqual(lax_sample["image"].shape, (2, 8, 8))
            np.testing.assert_array_equal(
                lax_sample["image"], expected["lax_4c_image"].numpy()
            )
            np.testing.assert_array_equal(lax_sample["mask"][0, :7, :5], 1)
            np.testing.assert_array_equal(lax_sample["mask"][1, :7, :5], 2)
            self.assertFalse(lax_sample["mask"][:, :, 5:].any())
            model[-1] = torch.nn.Linear(1, 6, bias=False)
            with torch.no_grad():
                model[-1].weight.copy_(torch.arange(1, 7).reshape(6, 1))
            model.requires_grad_(False)
            result = gradcam(lax_sample, model, model[1])
            self.assertEqual(result[1], 5)
            self.assertEqual(result[0].shape, (8, 8))
            namespace["concept_cmap"] = ListedColormap(
                [[0, 0, 0, 0], [1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.5]]
            )
            output_root = root / "exports/mnms2_lax_4c"
            namespace["export_samples"]([lax_sample], [result], output_root, "hcm")
            exports = sorted(output_root.rglob("*.png"))
            self.assertEqual(len(exports), 6)
            self.assertEqual(
                {path.name for path in exports}, {"003_ED.png", "003_ES.png"}
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in exports))
            self.assertEqual(plt.get_fignums(), figures)


if __name__ == "__main__":
    unittest.main()
