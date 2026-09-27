"""Classification scores with fixed class order and experiment metric policies."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from .protocol import protocols


def metrics_from_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[str, ...] = protocols["acdc"]["classes"],
    *,
    micro_zero_division: str | int = "warn",
) -> dict[str, Any]:
    """Score patient probabilities using protocol class order and macro OVO AUC."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prediction = probabilities.argmax(axis=1)
    labels = list(range(len(classes)))
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "f1": float(
            f1_score(
                y_true,
                prediction,
                average="micro",
                labels=labels,
                zero_division=micro_zero_division,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true, prediction, average="macro", labels=labels, zero_division=0
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
                average="macro",
                multi_class="ovo",
                labels=labels,
            )
        ),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=labels),
        "classification_report": classification_report(
            y_true,
            prediction,
            labels=labels,
            target_names=list(classes),
            output_dict=True,
            zero_division=0,
        ),
    }


def mnms2_3d_metrics_from_probabilities(y_true, probabilities):
    return metrics_from_probabilities(
        y_true, probabilities, protocols["mnms2_3d"]["classes"]
    )


def mnms2_sax_2d_metrics_from_probabilities(y_true, probabilities):
    return metrics_from_probabilities(
        y_true, probabilities, protocols["mnms2_sax_2d"]["classes"]
    )


def architecture_bank_metrics_from_probabilities(y_true, probabilities, classes):
    return metrics_from_probabilities(
        y_true, probabilities, classes, micro_zero_division=0
    )


reproduction_metrics_from_probabilities = architecture_bank_metrics_from_probabilities
