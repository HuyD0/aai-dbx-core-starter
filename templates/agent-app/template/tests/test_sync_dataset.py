"""UC dataset sync handles Databricks-native errors and optional metadata."""

from types import SimpleNamespace

import pytest

from scripts import sync_dataset


class SdkNotFound(Exception):
    error_code = "NOT_FOUND"


def test_databricks_sdk_not_found_is_the_only_create_path():
    assert sync_dataset._is_missing_dataset(SdkNotFound("table missing"))
    assert not sync_dataset._is_missing_dataset(PermissionError("access denied"))


def test_missing_experiment_ids_fail_closed_without_type_error(monkeypatch):
    dataset = SimpleNamespace(experiment_ids=None)
    fake_mlflow = SimpleNamespace(
        set_experiment=lambda name: SimpleNamespace(experiment_id="experiment-1"),
        genai=SimpleNamespace(
            datasets=SimpleNamespace(get_dataset=lambda **kwargs: dataset)
        ),
    )
    context = SimpleNamespace(
        settings=SimpleNamespace(
            catalog="catalog",
            schema="schema",
            effective_experiment_name="/Shared/evaluation",
        )
    )
    monkeypatch.setattr(sync_dataset, "mlflow", fake_mlflow)
    monkeypatch.setattr(sync_dataset, "bootstrap", lambda path: context)

    with pytest.raises(RuntimeError, match="not associated"):
        sync_dataset.main()


def test_merge_refreshes_dataset_identity_from_unity_catalog(monkeypatch, capsys):
    returned = SimpleNamespace(dataset_id="dataset-returned", digest="digest-stale")
    refreshed = SimpleNamespace(dataset_id="dataset-new", digest="digest-new")
    original = SimpleNamespace(
        dataset_id="dataset-old",
        digest="digest-old",
        experiment_ids=("experiment-1",),
        merge_records=lambda records: returned,
    )
    responses = iter((original, refreshed))
    fake_mlflow = SimpleNamespace(
        set_experiment=lambda name: SimpleNamespace(experiment_id="experiment-1"),
        genai=SimpleNamespace(
            datasets=SimpleNamespace(get_dataset=lambda **kwargs: next(responses))
        ),
    )
    context = SimpleNamespace(
        settings=SimpleNamespace(
            catalog="catalog",
            schema="schema",
            effective_experiment_name="/Shared/evaluation",
        )
    )
    monkeypatch.setattr(sync_dataset, "mlflow", fake_mlflow)
    monkeypatch.setattr(sync_dataset, "bootstrap", lambda path: context)

    sync_dataset.main()

    output = capsys.readouterr().out
    assert "dataset-new" in output
    assert "digest-new" in output
    assert "digest-old" not in output
    assert "digest-stale" not in output
