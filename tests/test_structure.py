"""Checks for the configuration and artifact boundaries affected by restructuring."""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.metrics import (
    metrics_from_probabilities,
    mnms2_3d_metrics_from_probabilities,
    mnms2_sax_2d_metrics_from_probabilities,
)
from src.protocol import protocols
from src.utils import artifact_path, output_file, prepare_run

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_every_config_validates_and_matches_protocol(self):
        for path in sorted((ROOT / "configs").glob("*.yaml")):
            with self.subTest(experiment=path.stem):
                config = OmegaConf.load(path)
                self.assertEqual(config.experiment, path.stem)
                args = prepare_run(config, path.stem)
                self.assertEqual(args.output_dir, Path("outputs"))
                self.assertEqual(args.seeds, [0, 1, 2])
                self.assertIn(path.stem, protocols["options"])

    def test_overrides_and_invalid_seeds(self):
        config = OmegaConf.load(ROOT / "configs/acdc_2d.yaml")
        config = OmegaConf.merge(
            config,
            OmegaConf.from_dotlist(
                [
                    "train.n_epochs=3",
                    "run.seeds=[7]",
                    "run.smoke=true",
                    "data.acdc_processed=/tmp/acdc",
                    "run.stages=[validate,train]",
                ]
            ),
        )
        args = prepare_run(config, "acdc_2d")
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.acdc_processed, Path("/tmp/acdc"))
        self.assertTrue(args.smoke)
        config.run.seeds = [7, 7]
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_run(config, "acdc_2d")
        config.run.seeds = [7]
        config.run.stages = ["typo"]
        with self.assertRaisesRegex(ValueError, "Invalid stages"):
            prepare_run(config, "acdc_2d")

    def test_cli_resolves_all_configs_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in sorted((ROOT / "configs").glob("*.yaml")):
                with self.subTest(experiment=path.stem):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "main.py"),
                            "--config-name",
                            path.stem,
                            "--cfg",
                            "job",
                            "--resolve",
                        ],
                        cwd=directory,
                        text=True,
                        capture_output=True,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    config = OmegaConf.create(completed.stdout)
                    self.assertEqual(config.experiment, path.stem)
            self.assertEqual(list(Path(directory).iterdir()), [])


class MetricsTests(unittest.TestCase):
    def test_five_and_six_class_protocols(self):
        for key, score in [
            ("acdc", metrics_from_probabilities),
            ("mnms2_3d", mnms2_3d_metrics_from_probabilities),
            ("mnms2_sax_2d", mnms2_sax_2d_metrics_from_probabilities),
        ]:
            with self.subTest(protocol=key):
                classes = protocols[key]["classes"]
                labels = np.arange(len(classes))
                result = score(labels, np.eye(len(classes)))
                for metric in (
                    "accuracy",
                    "f1",
                    "macro_f1",
                    "balanced_accuracy",
                    "mcc",
                    "roc_auc",
                ):
                    self.assertEqual(result[metric], 1.0)
                np.testing.assert_array_equal(
                    result["confusion_matrix"], np.eye(len(classes))
                )
                self.assertTrue(set(classes).issubset(result["classification_report"]))

    def test_explicit_class_order_and_nonperfect_predictions(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        probabilities = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.1, 0.8, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ]
        )
        result = metrics_from_probabilities(labels, probabilities, ("Z", "A", "B"))
        self.assertAlmostEqual(result["accuracy"], 5 / 6)
        np.testing.assert_array_equal(
            result["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [0, 0, 2]]
        )
        self.assertEqual(result["classification_report"]["Z"]["recall"], 0.5)


class OutputTests(unittest.TestCase):
    def test_run_hierarchy_is_preserved_across_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "checkpoints/acdc/resnet18/seed_0"
            for filename, category in [
                ("test/predictions.csv", "predictions"),
                ("test/metrics.json", "metrics"),
                ("test/confusion_matrix.csv", "metrics"),
                ("history.csv", "logs"),
                ("command.txt", "logs"),
            ]:
                with self.subTest(filename=filename):
                    expected = root / category / "acdc/resnet18/seed_0" / filename
                    self.assertEqual(artifact_path(run / filename), expected)
                    actual = output_file(run / filename)
                    self.assertEqual(actual, expected)
                    self.assertTrue(actual.parent.is_dir())
                    # The helper is safe to apply to already categorized paths.
                    self.assertEqual(artifact_path(actual), actual)
            for filename in (
                "best_val_mcc.pt",
                "summary.json",
                "failure.json",
                "architecture.json",
            ):
                self.assertEqual(artifact_path(run / filename), run / filename)

    def test_external_legacy_prediction_paths_remain_readable(self):
        path = Path("/tmp/old_results/model/seed_0/test/predictions.csv")
        self.assertEqual(artifact_path(path), path)


class SourceTests(unittest.TestCase):
    def test_internal_imports_resolve_without_cycles(self):
        trees = {p.stem: ast.parse(p.read_text()) for p in (ROOT / "src").glob("*.py")}
        names = {}
        graph = {}
        for module, tree in trees.items():
            names[module] = set()
            graph[module] = set()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    names[module].add(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    names[module].update(a.asname or a.name for a in node.names)
                elif isinstance(node, ast.Assign):
                    names[module].update(
                        n.id
                        for target in node.targets
                        for n in ast.walk(target)
                        if isinstance(n, ast.Name)
                    )
        for module, tree in trees.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level:
                    self.assertIn(node.module, trees)
                    graph[module].add(node.module)
                    for alias in node.names:
                        self.assertIn(
                            alias.name, names[node.module], f"{module}: {alias.name}"
                        )

        def visit(module, parents):
            self.assertNotIn(module, parents, f"Import cycle: {parents} -> {module}")
            for dependency in graph[module]:
                visit(dependency, parents + [module])

        for module in graph:
            visit(module, [])


if __name__ == "__main__":
    unittest.main()
