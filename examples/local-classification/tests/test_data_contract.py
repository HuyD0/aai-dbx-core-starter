from __future__ import annotations

from pathlib import Path

import pytest

from aai_local_classification.contracts import SplitName
from aai_local_classification.data import (
    add_intentional_leakage,
    generate_subscription_data,
    load_split,
    prepare_dataset,
    validate_feature_contract,
)


def test_generated_categorical_missing_values_are_real_nulls(settings):
    data = generate_subscription_data(settings)
    signup_channel = data["signup_channel"]

    assert signup_channel.isna().any()
    assert not {"None", "nan", "<NA>"} & set(signup_channel.dropna().astype(str))


def test_generation_is_deterministic_and_test_labels_stay_sealed(settings, tmp_path):
    first = prepare_dataset(settings, tmp_path / "first")
    second = prepare_dataset(settings, tmp_path / "second")

    assert first.dataset_sha256 == second.dataset_sha256
    assert [item.row_count for item in first.artifacts] == [2160, 360, 360]
    by_split = {item.split: item for item in first.artifacts}
    assert by_split[SplitName.TRAIN].positive_rate is not None
    assert by_split[SplitName.VALIDATION].positive_rate is not None
    assert by_split[SplitName.TEST].positive_rate is None
    assert first.frozen_test is True
    assert all(len(item.sha256) == 64 for item in first.artifacts)


def test_time_splits_are_disjoint_and_ordered(settings, tmp_path):
    root = tmp_path / "processed"
    prepare_dataset(settings, root)
    train = load_split(settings, SplitName.TRAIN, root)
    validation = load_split(settings, SplitName.VALIDATION, root)
    test = load_split(settings, SplitName.TEST, root)

    assert train.snapshot_date.max() < validation.snapshot_date.min()
    assert validation.snapshot_date.max() < test.snapshot_date.min()
    assert set(train.account_id).isdisjoint(validation.account_id)
    assert set(train.account_id).isdisjoint(test.account_id)
    assert set(validation.account_id).isdisjoint(test.account_id)
    assert set(train[settings.data.target_column].unique()) == {0, 1}


def test_digest_mismatch_fails_before_modeling(settings, tmp_path):
    root = tmp_path / "processed"
    prepare_dataset(settings, root)
    train_path = root / "train.csv"
    train_path.write_text(train_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Digest mismatch"):
        load_split(settings, SplitName.TRAIN, root)


def test_generation_configuration_change_requires_explicit_prepare(settings, tmp_path):
    root = tmp_path / "processed"
    prepare_dataset(settings, root)
    changed_data = settings.data.model_copy(
        update={"rows_per_month": settings.data.rows_per_month + 1}
    )
    changed_settings = settings.model_copy(update={"data": changed_data})

    with pytest.raises(ValueError, match="generation configuration"):
        load_split(changed_settings, SplitName.TRAIN, root)


def test_post_outcome_feature_is_present_only_as_blocked_lesson(settings, tmp_path):
    root = tmp_path / "processed"
    prepare_dataset(settings, root)
    train = load_split(settings, SplitName.TRAIN, root)
    leaked = add_intentional_leakage(train)
    unsafe_features = settings.features.model_copy(
        update={"categorical": settings.features.categorical + ("cancellation_reason",)}
    )
    unsafe_settings = settings.model_copy(update={"features": unsafe_features})

    assert "cancellation_reason" in leaked
    with pytest.raises(ValueError, match="Leakage or identifier"):
        validate_feature_contract(unsafe_settings)


def test_generated_assets_are_ignored_by_contract():
    ignore = Path(__file__).resolve().parents[1] / ".gitignore"
    assert "data/processed/" in ignore.read_text(encoding="utf-8")
