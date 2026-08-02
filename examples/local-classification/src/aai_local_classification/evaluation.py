"""Classification metrics, cost-aware thresholding, slices, and release checks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aai_local_classification.contracts import (
    ClassificationMetrics,
    ProjectSettings,
    ThresholdSelection,
)


def _unit_interval(value: float) -> float:
    """Clamp harmless floating-point overshoot at strict metric boundaries."""

    return min(1.0, max(0.0, float(value)))


def evaluate_probabilities(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
    *,
    false_negative_cost: float,
    false_positive_cost: float,
) -> ClassificationMetrics:
    truth = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(probabilities), dtype=float)
    if len(truth) == 0 or len(truth) != len(scores):
        raise ValueError("Labels and probabilities must be non-empty and aligned")
    if set(np.unique(truth)) != {0, 1}:
        raise ValueError("Evaluation requires both binary labels")
    if np.isnan(scores).any() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    cost = (float(fn) * false_negative_cost) + (float(fp) * false_positive_cost)
    return ClassificationMetrics(
        threshold=float(threshold),
        row_count=len(truth),
        positive_rate=_unit_interval(truth.mean()),
        predicted_positive_rate=_unit_interval(predictions.mean()),
        accuracy=_unit_interval(accuracy_score(truth, predictions)),
        balanced_accuracy=_unit_interval(balanced_accuracy_score(truth, predictions)),
        precision=_unit_interval(precision_score(truth, predictions, zero_division=0)),
        recall=_unit_interval(recall_score(truth, predictions, zero_division=0)),
        specificity=_unit_interval(specificity),
        f1=_unit_interval(f1_score(truth, predictions, zero_division=0)),
        average_precision=_unit_interval(average_precision_score(truth, scores)),
        roc_auc=_unit_interval(roc_auc_score(truth, scores)),
        log_loss=float(log_loss(truth, scores, labels=[0, 1])),
        brier_score=_unit_interval(brier_score_loss(truth, scores)),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        cost_per_1000=float(cost / len(truth) * 1_000),
    )


def select_threshold(
    y_validation: Iterable[int],
    validation_probabilities: Iterable[float],
    settings: ProjectSettings,
) -> ThresholdSelection:
    """Choose a threshold on validation data only using declared action costs."""

    candidates = np.linspace(0.05, 0.85, 161)
    evaluated = [
        evaluate_probabilities(
            y_validation,
            validation_probabilities,
            float(threshold),
            false_negative_cost=settings.selection.false_negative_cost,
            false_positive_cost=settings.selection.false_positive_cost,
        )
        for threshold in candidates
    ]
    feasible = [
        metrics
        for metrics in evaluated
        if metrics.precision >= settings.selection.minimum_validation_precision
        and metrics.recall >= settings.selection.minimum_validation_recall
    ]
    if not feasible:
        raise ValueError(
            "No validation threshold satisfies the declared precision and recall "
            "constraints; the selection result is inconclusive"
        )
    selected = min(
        feasible,
        key=lambda metrics: (
            metrics.cost_per_1000,
            -metrics.f1,
            abs(metrics.threshold - 0.5),
        ),
    )
    return ThresholdSelection(
        threshold=selected.threshold,
        validation_metrics=selected,
        feasible_threshold_count=len(feasible),
        selection_rule=(
            "minimum expected validation cost subject to declared precision and "
            "recall constraints; fail inconclusive if none are feasible"
        ),
    )


def recall_slices(
    data: pd.DataFrame,
    y_true: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
    columns: tuple[str, ...] = ("plan_tier", "signup_channel"),
    minimum_positive_rows: int = 5,
) -> pd.DataFrame:
    """Report recall by operational slices; this is not a fairness certification."""

    scored = data.loc[:, list(columns)].copy()
    scored["label"] = np.asarray(list(y_true), dtype=int)
    scored["prediction"] = (
        np.asarray(list(probabilities), dtype=float) >= threshold
    ).astype(int)
    records: list[dict[str, object]] = []
    for column in columns:
        for value, group in scored.groupby(column, dropna=False):
            positives = int(group["label"].sum())
            if positives < minimum_positive_rows:
                continue
            recall = recall_score(group["label"], group["prediction"], zero_division=0)
            records.append(
                {
                    "slice_column": column,
                    "slice_value": str(value),
                    "row_count": len(group),
                    "positive_count": positives,
                    "recall": float(recall),
                }
            )
    return pd.DataFrame.from_records(records)


def maximum_recall_gap(slices: pd.DataFrame) -> float:
    if slices.empty:
        return 1.0
    gaps = slices.groupby("slice_column")["recall"].agg(
        lambda values: float(values.max() - values.min())
    )
    return float(gaps.max())


def promotion_checks(
    metrics: ClassificationMetrics,
    slice_recall_gap: float,
    settings: ProjectSettings,
) -> dict[str, bool]:
    gate = settings.promotion_gate
    return {
        "minimum_test_average_precision": (
            metrics.average_precision >= gate.minimum_test_average_precision
        ),
        "minimum_test_average_precision_lift": (
            metrics.average_precision - metrics.positive_rate
            >= gate.minimum_test_average_precision_lift
        ),
        "minimum_test_recall": metrics.recall >= gate.minimum_test_recall,
        "maximum_test_brier_score": (
            metrics.brier_score <= gate.maximum_test_brier_score
        ),
        "maximum_test_cost_per_1000": (
            metrics.cost_per_1000 <= gate.maximum_test_cost_per_1000
        ),
        "maximum_slice_recall_gap": (slice_recall_gap <= gate.maximum_slice_recall_gap),
    }


def metric_dict(metrics: ClassificationMetrics, prefix: str = "") -> dict[str, float]:
    values = metrics.model_dump(mode="python")
    return {
        f"{prefix}{name}": float(value)
        for name, value in values.items()
        if name not in {"threshold", "row_count"}
    } | {
        f"{prefix}threshold": metrics.threshold,
        f"{prefix}row_count": float(metrics.row_count),
    }
