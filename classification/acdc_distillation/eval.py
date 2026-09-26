"""Inference, metrics, and result aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training_code.classification.acdc_3d.eval import save_predictions
from training_code.classification.acdc_3d.utils import task_root as baseline_task_root
from training_code.classification.acdc_distillation.utils import task_root
from training_code.classification.eval import metrics_from_probabilities
from training_code.classification.protocol import protocols

BASELINE_PROTOCOL_VERSION = protocols["acdc_3d"]["protocol_version"]
CLASSES = protocols["acdc"]["classes"]
METRIC_NAMES = protocols["shared"]["metric_names"]
PROTOCOL_VERSION = protocols["acdc_distillation"]["protocol_version"]


def aggregate_results(args: argparse.Namespace) -> None:
    root = task_root(args, smoke=False)
    rows: list[dict[str, Any]] = []
    for path in root.rglob("summary.json") if root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("protocol_version") != PROTOCOL_VERSION:
            continue
        row = {key: value for key, value in item.items() if key != "test"}
        row.update({f"test_{key}": value for key, value in item["test"].items()})
        rows.append(row)
    if not rows:
        print(f"[aggregate] no completed distilled runs under {root}")
        return
    frame = pd.DataFrame(rows).sort_values("test_mcc", ascending=False)
    frame.to_csv(root / "all_seed_results.csv", index=False)

    group_columns = [
        "backbone",
        "formulation",
        "initialization",
        "temperature",
        "supervised_weight",
    ]
    metric_columns = [f"test_{name}" for name in METRIC_NAMES]
    summary_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys, strict=True))
        row["n_seeds"] = len(group)
        row["complete_seed_set"] = set(group["seed"].astype(int)) == set(args.seeds)
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        summary_rows.append(row)
    distilled_summary = pd.DataFrame(summary_rows).sort_values("test_mcc_mean", ascending=False)
    distilled_summary.to_csv(root / "mean_std_summary.csv", index=False)

    baseline_rows: list[dict[str, Any]] = []
    baseline_root = baseline_task_root(args, smoke=False)
    for path in baseline_root.rglob("summary.json") if baseline_root.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("protocol_version") != BASELINE_PROTOCOL_VERSION:
            continue
        baseline_rows.append(
            {
                "backbone": item["backbone"],
                "formulation": item["formulation"],
                **{f"test_{name}": item["test"][name] for name in METRIC_NAMES},
            }
        )
    if baseline_rows:
        baseline = (
            pd.DataFrame(baseline_rows).groupby(["backbone", "formulation"], as_index=False)[metric_columns].mean()
        )
        distilled = distilled_summary[
            ["backbone", "formulation", *(f"{column}_mean" for column in metric_columns)]
        ].rename(columns={f"{column}_mean": column for column in metric_columns})
        comparison = distilled.merge(
            baseline,
            on=["backbone", "formulation"],
            suffixes=("_distilled", "_randinit"),
        )
        for column in metric_columns:
            comparison[f"{column}_delta"] = comparison[f"{column}_distilled"] - comparison[f"{column}_randinit"]
        comparison.to_csv(root / "comparison_vs_randinit.csv", index=False)

    ensemble_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        predictions = [
            pd.read_csv(Path(checkpoint).parent / "test" / "predictions.csv", dtype={"pid": str})
            for checkpoint in group["checkpoint"]
        ]
        reference = predictions[0]
        probability_columns = [f"prob_{name}" for name in CLASSES]
        for prediction in predictions[1:]:
            if prediction["pid"].tolist() != reference["pid"].tolist():
                raise RuntimeError(f"Patient order differs while ensembling {keys}")
        probabilities = np.mean(np.stack([item[probability_columns].to_numpy() for item in predictions]), axis=0)
        targets = reference["target"].to_numpy(dtype=np.int64)
        metrics = metrics_from_probabilities(targets, probabilities)
        backbone, formulation, initialization, temperature, supervised_weight = keys
        output = root / str(backbone) / str(formulation) / str(initialization) / "ensemble"
        save_predictions(output, reference["pid"].tolist(), targets, probabilities, metrics)
        ensemble_rows.append(
            {
                "backbone": backbone,
                "formulation": formulation,
                "initialization": initialization,
                "temperature": temperature,
                "supervised_weight": supervised_weight,
                "n_seeds": len(group),
                **{name: metrics[name] for name in METRIC_NAMES},
            }
        )
    pd.DataFrame(ensemble_rows).sort_values("mcc", ascending=False).to_csv(
        root / "probability_ensemble_results.csv", index=False
    )
    print(f"[aggregate] wrote {len(frame)} distilled runs to {root}")
