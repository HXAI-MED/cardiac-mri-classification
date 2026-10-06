"""Check notebook preprocessing, Grad-CAM, hook cleanup, and PNG export on CPU."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from src.dataset import (
    ACDCMidSAX2DDataset,
    acdc_2d_load_official_splits,
    acdc_2d_split_directory,
    get_mid_sax_transforms,
)
from src.protocol import protocols

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
            load_samples = namespace["load_acdc_samples"]
            (sample,) = load_samples(root, config, "val", 1, "hcm")
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
                load_samples(root, config, n_samples=0)
            with self.assertRaisesRegex(ValueError, "Unknown split"):
                load_samples(root, config, split="other")
            with self.assertRaisesRegex(ValueError, "Unknown ACDC class"):
                load_samples(root, config, class_name="other")

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
            namespace.update(
                model=model, target_layer=model[1], DEVICE=torch.device("cpu")
            )
            gradcam = namespace["gradcam"]
            with torch.no_grad():
                heatmap, class_index, probability = gradcam(sample)
            expected = sample["image"].sum(axis=0)
            np.testing.assert_allclose(heatmap, expected / expected.max(), atol=1e-6)
            self.assertEqual(class_index, 4)
            self.assertTrue(0 < probability <= 1)
            self.assertFalse(model[1]._forward_hooks)
            np.testing.assert_array_equal(gradcam(sample)[0], heatmap)
            with patch.object(
                model[-1], "forward", side_effect=RuntimeError("failure")
            ):
                with self.assertRaisesRegex(RuntimeError, "failure"):
                    gradcam(sample)
            self.assertFalse(model[1]._forward_hooks)
            with torch.no_grad():
                model[-1].weight.zero_()
            self.assertFalse(gradcam(sample)[0].any())
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


if __name__ == "__main__":
    unittest.main()
