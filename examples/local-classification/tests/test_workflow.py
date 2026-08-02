from __future__ import annotations

import mlflow
import pytest

from aai_local_classification.contracts import PromotionDecision, SplitName
from aai_local_classification.data import load_split
from aai_local_classification.inference import load_champion
from aai_local_classification.tracking import local_paths
from aai_local_classification.workflow import (
    load_decision,
    load_selection,
    promote_if_approved,
    run_frozen_test_gate,
    run_full_workflow,
)


def test_complete_local_mlflow_release_and_reload(settings, tmp_path):
    result = run_full_workflow(settings, tmp_path)

    assert result["decision"]["decision"] == PromotionDecision.ADOPT.value
    assert all(result["decision"]["checks"].values())
    assert result["promotion"]["registered"] is True
    assert result["promotion"]["alias"] == "champion"
    assert result["selection"]["selected_candidate"] == "logistic-regression"

    paths = local_paths(tmp_path)
    assert paths.mlflow_root.joinpath("mlflow.db").is_file()
    predictor = load_champion(settings, tmp_path)
    validation = load_split(settings, SplitName.VALIDATION, paths.data_root)
    predictions = predictor.predict(validation.head(5), settings)
    assert list(predictions.columns) == [
        "churn_probability",
        "churn_prediction",
        "model_name",
        "model_version",
    ]
    assert predictions.churn_probability.between(0, 1).all()
    assert set(predictions.churn_prediction).issubset({0, 1})

    client = mlflow.MlflowClient(
        tracking_uri=paths.tracking_uri,
        registry_uri=paths.tracking_uri,
    )
    version = client.get_model_version_by_alias(
        settings.registered_model_name,
        "champion",
    )
    assert version.tags["validation_status"] == "passed"
    assert version.tags["dataset_sha256"] == result["dataset_sha256"]
    assert float(version.tags["decision_threshold"]) == (
        result["decision"]["threshold"]
    )

    repeated = run_full_workflow(settings, tmp_path)
    assert repeated["selection"]["selected_run_id"] == (
        result["selection"]["selected_run_id"]
    )
    assert repeated["decision"]["test_run_id"] == result["decision"]["test_run_id"]
    assert repeated["promotion"]["model_version"] == (
        result["promotion"]["model_version"]
    )


def test_consumed_test_and_promotion_linkage_fail_closed(settings, tmp_path):
    run_full_workflow(settings, tmp_path)
    selection = load_selection(tmp_path)
    decision = load_decision(tmp_path)

    changed_gate = settings.promotion_gate.model_copy(
        update={"minimum_test_recall": 0.99}
    )
    changed_settings = settings.model_copy(update={"promotion_gate": changed_gate})
    with pytest.raises(ValueError, match="frozen-test dataset version"):
        run_frozen_test_gate(changed_settings, tmp_path, selection)

    failed_checks = decision.checks.model_copy(update={"minimum_test_recall": False})
    forged_adopt = decision.model_copy(update={"checks": failed_checks})
    with pytest.raises(ValueError, match="gate outcome"):
        promote_if_approved(settings, forged_adopt, tmp_path, selection)

    other = next(
        candidate
        for candidate in selection.candidates
        if candidate.run_id != selection.selected_run_id
    )
    wrong_selection = selection.model_copy(
        update={
            "selected_candidate": other.candidate_name,
            "selected_run_id": other.run_id,
            "selected_model_id": other.model_id,
            "selected_model_uri": other.model_uri,
        }
    )
    with pytest.raises(ValueError, match="Release decision is not bound"):
        promote_if_approved(
            settings,
            decision,
            tmp_path,
            wrong_selection,
        )
