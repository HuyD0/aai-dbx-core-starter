"""The credentialed gate rejects unreviewed EvaluationDataset drift."""

from evals.evaluate import _case_key


def test_dataset_comparison_keeps_review_tags_and_ignores_mlflow_tags():
    reviewed = {
        "inputs": {"question": "What were net sales?"},
        "expectations": {"expected_response": "$10"},
        "tags": {"failure_mode": "delayed_label"},
    }
    registered = {
        **reviewed,
        "tags": {
            "failure_mode": "delayed_label",
            "mlflow.user": "workspace-user",
        },
    }
    drifted = {**reviewed, "tags": {"failure_mode": "popularity_bias"}}

    assert _case_key(reviewed) == _case_key(registered)
    assert _case_key(reviewed) != _case_key(drifted)
