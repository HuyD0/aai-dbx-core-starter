from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aai_local_classification.contracts import PromotionDecision
from aai_local_classification.data import generate_subscription_data
from aai_local_classification.evaluation import (
    evaluate_probabilities,
    maximum_recall_gap,
    promotion_checks,
    recall_slices,
    select_threshold,
)
from aai_local_classification.modeling import (
    build_candidate,
    build_preprocessor,
    candidate_specs,
    feature_frame,
)


def test_metric_meaning_and_confusion_counts(settings):
    truth = [0, 0, 1, 1]
    probability = [0.1, 0.8, 0.4, 0.9]
    metrics = evaluate_probabilities(
        truth,
        probability,
        0.5,
        false_negative_cost=5,
        false_positive_cost=1,
    )

    assert (
        metrics.true_negatives,
        metrics.false_positives,
        metrics.false_negatives,
        metrics.true_positives,
    ) == (1, 1, 1, 1)
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.cost_per_1000 == 1500.0


def test_threshold_search_obeys_validation_constraints(settings):
    truth = np.array([0] * 80 + [1] * 20)
    probability = np.linspace(0.01, 0.99, 100)
    selected = select_threshold(truth, probability, settings)

    assert selected.feasible_threshold_count > 0
    assert (
        selected.validation_metrics.precision
        >= settings.selection.minimum_validation_precision
    )
    assert (
        selected.validation_metrics.recall
        >= settings.selection.minimum_validation_recall
    )


def test_threshold_search_fails_inconclusive_instead_of_waiving_constraints(
    settings,
):
    strict_selection = settings.selection.model_copy(
        update={
            "minimum_validation_precision": 0.99,
            "minimum_validation_recall": 0.99,
        }
    )
    strict_settings = settings.model_copy(update={"selection": strict_selection})

    with pytest.raises(ValueError, match="selection result is inconclusive"):
        select_threshold(
            [0, 0, 1, 1],
            [0.5, 0.5, 0.5, 0.5],
            strict_settings,
        )


def test_pipeline_handles_missing_and_unseen_categories(settings):
    train = pd.DataFrame(
        {
            "tenure_months": [1, 3, 12, 24, 36, 48],
            "monthly_fee": [100, 90, 70, 60, 55, 50],
            "support_tickets_90d": [4, 3, 2, 1, 0, 0],
            "usage_hours_30d": [5, np.nan, 12, 20, 30, 40],
            "payment_failures_90d": [2, 1, 1, 0, 0, 0],
            "contract_type": [
                "month_to_month",
                "month_to_month",
                "one_year",
                "one_year",
                "two_year",
                "two_year",
            ],
            "plan_tier": ["pro", "plus", "plus", "basic", "basic", "basic"],
            "signup_channel": [
                "paid_search",
                np.nan,
                "partner",
                "organic",
                "organic",
                "partner",
            ],
            "autopay": [False, False, True, True, True, True],
            "churned_30d": [1, 1, 1, 0, 0, 0],
        }
    )
    model = build_candidate(candidate_specs()[0], settings).fit(
        feature_frame(train, settings), train.churned_30d
    )
    future = feature_frame(train.head(1), settings)
    future.loc[:, "signup_channel"] = "new_channel"

    probabilities = model.predict_proba(future)
    assert probabilities.shape == (1, 2)
    assert np.isclose(probabilities.sum(), 1.0)


def test_generated_missing_category_is_imputed_before_encoding(settings):
    generated = generate_subscription_data(settings)
    preprocessor = build_preprocessor(settings).fit(feature_frame(generated, settings))
    categorical = preprocessor.named_transformers_["categorical"]
    categories = categorical.named_steps["one_hot"].categories_
    encoded_values = [value for values in categories for value in values]
    feature_names = set(preprocessor.get_feature_names_out())

    assert generated["signup_channel"].isna().any()
    assert not any(pd.isna(value) for value in encoded_values)
    assert (
        not {
            "signup_channel_None",
            "signup_channel_nan",
            "signup_channel_<NA>",
        }
        & feature_names
    )


def test_slice_gap_and_gate_are_explicit(settings):
    data = pd.DataFrame(
        {
            "plan_tier": ["basic"] * 10 + ["pro"] * 10,
            "signup_channel": ["organic"] * 10 + ["partner"] * 10,
        }
    )
    truth = [1] * 20
    probability = [0.9] * 10 + [0.1] * 10
    slices = recall_slices(
        data,
        truth,
        probability,
        0.5,
        minimum_positive_rows=5,
    )
    assert maximum_recall_gap(slices) == 1.0

    metrics = evaluate_probabilities(
        [0, 0, *truth],
        [0.1, 0.2, *probability],
        0.5,
        false_negative_cost=5,
        false_positive_cost=1,
    )
    checks = promotion_checks(metrics, 1.0, settings)
    assert checks["maximum_slice_recall_gap"] is False
    assert PromotionDecision.REJECT.value == "reject"
