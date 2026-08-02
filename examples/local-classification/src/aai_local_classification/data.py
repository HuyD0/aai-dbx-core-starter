"""Deterministic synthetic data, time splits, validation, and lineage manifests."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aai_local_classification.contracts import (
    DatasetManifest,
    ProjectSettings,
    SplitArtifact,
    SplitName,
)
from aai_local_classification.settings import PROJECT_ROOT


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    train: Path
    validation: Path
    test: Path
    manifest: Path


def dataset_paths(root: Path | None = None) -> DatasetPaths:
    base = root or PROJECT_ROOT / "data" / "processed"
    return DatasetPaths(
        root=base,
        train=base / "train.csv",
        validation=base / "validation.csv",
        test=base / "test.csv",
        manifest=base / "manifest.json",
    )


def generate_subscription_data(settings: ProjectSettings) -> pd.DataFrame:
    """Generate a small, realistic, non-sensitive churn learning dataset."""

    rng = np.random.default_rng(settings.random_seed)
    frames: list[pd.DataFrame] = []
    account_number = 0
    months = pd.date_range(
        settings.data.start_month,
        periods=settings.data.months,
        freq="MS",
    )

    for month_index, snapshot in enumerate(months):
        size = settings.data.rows_per_month
        contract = rng.choice(
            ["month_to_month", "one_year", "two_year"],
            size=size,
            p=[0.58, 0.29, 0.13],
        )
        plan = rng.choice(["basic", "plus", "pro"], size=size, p=[0.48, 0.37, 0.15])
        channel = rng.choice(
            ["organic", "partner", "paid_search"], size=size, p=[0.51, 0.24, 0.25]
        )
        autopay_probability = np.where(contract == "month_to_month", 0.42, 0.72)
        autopay = rng.binomial(1, autopay_probability, size=size).astype(bool)
        tenure = (
            np.clip(
                rng.gamma(shape=2.4, scale=10.0, size=size) + month_index / 3,
                1,
                72,
            )
            .round()
            .astype(int)
        )
        plan_fee = np.select(
            [plan == "basic", plan == "plus", plan == "pro"],
            [42.0, 68.0, 105.0],
        )
        monthly_fee = plan_fee + month_index * 0.35 + rng.normal(0, 6.5, size)
        support_rate = 0.7 + (contract == "month_to_month") * 0.25
        support_tickets = rng.poisson(support_rate, size=size)
        usage = np.clip(
            rng.normal(25, 8.5, size) + (plan == "plus") * 5 + (plan == "pro") * 11,
            0,
            None,
        )
        payment_failures = rng.binomial(
            2,
            np.where(autopay, 0.06, 0.14),
            size=size,
        )

        logit = (
            -3.05
            + 1.20 * (contract == "month_to_month")
            + 0.30 * (contract == "one_year")
            + 0.90 * payment_failures
            + 0.40 * support_tickets
            + 1.05 * (usage < 17)
            + 0.65 * (tenure < 6)
            - 0.62 * autopay
            + 0.48 * ((monthly_fee - 65) / 25)
            + 0.30 * (month_index >= 18)
            + 0.17 * (channel == "paid_search")
        )
        probability = 1 / (1 + np.exp(-logit))
        churn = rng.binomial(1, probability, size=size)

        account_ids = [f"SYN-{account_number + offset:06d}" for offset in range(size)]
        account_number += size
        frame = pd.DataFrame(
            {
                "account_id": account_ids,
                "snapshot_date": snapshot,
                "tenure_months": tenure,
                "monthly_fee": monthly_fee.round(2),
                "support_tickets_90d": support_tickets,
                "usage_hours_30d": usage.round(2),
                "payment_failures_90d": payment_failures,
                "contract_type": contract,
                "plan_tier": plan,
                "signup_channel": channel,
                "autopay": autopay,
                "churned_30d": churn,
            }
        )
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    missing_usage = rng.random(len(data)) < 0.035
    missing_channel = rng.random(len(data)) < 0.015
    data.loc[missing_usage, "usage_hours_30d"] = np.nan
    # SimpleImputer uses ``np.nan`` as its missing-value marker by default. Using
    # Python ``None`` here would make OneHotEncoder learn a literal ``None``
    # category instead of imputing these rows.
    data.loc[missing_channel, "signup_channel"] = np.nan
    return data


def generator_code_sha256() -> str:
    source = inspect.getsource(generate_subscription_data)
    return hashlib.sha256(source.encode()).hexdigest()


def generation_config_sha256(settings: ProjectSettings) -> str:
    payload = {
        "random_seed": settings.random_seed,
        "data": settings.data.model_dump(mode="json"),
        "features": settings.features.model_dump(mode="json"),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def validate_manifest_contract(
    manifest: DatasetManifest,
    settings: ProjectSettings,
) -> None:
    mismatches: list[str] = []
    if manifest.generator_code_sha256 != generator_code_sha256():
        mismatches.append("generator code")
    if manifest.generation_config_sha256 != generation_config_sha256(settings):
        mismatches.append("generation configuration")
    if manifest.dataset_name != settings.data.dataset_name:
        mismatches.append("dataset name")
    if manifest.target_column != settings.data.target_column:
        mismatches.append("target column")
    if manifest.feature_columns != settings.features.model_columns:
        mismatches.append("feature contract")
    if mismatches:
        raise ValueError(
            "Dataset manifest no longer matches the "
            f"{', '.join(mismatches)}; run the explicit prepare step"
        )


def split_by_time(
    data: pd.DataFrame, settings: ProjectSettings
) -> dict[SplitName, pd.DataFrame]:
    """Create train/validation/test partitions using production-like time order."""

    dates = pd.to_datetime(data[settings.data.time_column])
    validation_start = pd.Timestamp(settings.data.validation_start)
    test_start = pd.Timestamp(settings.data.test_start)
    splits = {
        SplitName.TRAIN: data.loc[dates < validation_start].copy(),
        SplitName.VALIDATION: data.loc[
            (dates >= validation_start) & (dates < test_start)
        ].copy(),
        SplitName.TEST: data.loc[dates >= test_start].copy(),
    }
    return {name: frame.reset_index(drop=True) for name, frame in splits.items()}


def validate_feature_contract(settings: ProjectSettings) -> None:
    features = settings.features.model_columns
    duplicates = sorted({name for name in features if features.count(name) > 1})
    forbidden = sorted(set(features) & set(settings.features.forbidden))
    if duplicates:
        raise ValueError(f"Duplicate model features: {duplicates}")
    if forbidden:
        raise ValueError(
            f"Leakage or identifier columns selected as features: {forbidden}"
        )


def validate_dataset(data: pd.DataFrame, settings: ProjectSettings) -> dict[str, float]:
    """Fail fast on schema, identity, target, and gross-quality violations."""

    validate_feature_contract(settings)
    required = {
        settings.data.id_column,
        settings.data.time_column,
        settings.data.target_column,
        *settings.features.model_columns,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data[settings.data.id_column].isna().any():
        raise ValueError("Account identifiers must not be null")
    if data[settings.data.id_column].duplicated().any():
        raise ValueError("Account identifiers must be unique")
    labels = set(data[settings.data.target_column].dropna().unique())
    if labels != {0, 1}:
        raise ValueError(f"Target must contain both binary labels; found {labels}")
    parsed_dates = pd.to_datetime(data[settings.data.time_column], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Snapshot dates must be valid")
    numeric = data.loc[:, list(settings.features.numeric)]
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Numeric features must not contain infinity")
    missing_rates = data.loc[:, list(settings.features.model_columns)].isna().mean()
    if float(missing_rates.max()) > 0.10:
        raise ValueError("A model feature exceeds the 10% missing-value contract")
    return {
        "row_count": float(len(data)),
        "positive_rate": float(data[settings.data.target_column].mean()),
        "maximum_feature_missing_rate": float(missing_rates.max()),
        "duplicate_id_count": float(data[settings.data.id_column].duplicated().sum()),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_dataset(
    settings: ProjectSettings,
    root: Path | None = None,
) -> DatasetManifest:
    """Generate, validate, split, persist, and fingerprint the learning data."""

    paths = dataset_paths(root)
    paths.root.mkdir(parents=True, exist_ok=True)
    data = generate_subscription_data(settings)
    validate_dataset(data, settings)
    splits = split_by_time(data, settings)
    output_paths = {
        SplitName.TRAIN: paths.train,
        SplitName.VALIDATION: paths.validation,
        SplitName.TEST: paths.test,
    }
    artifacts: list[SplitArtifact] = []
    for split_name, frame in splits.items():
        validate_dataset(frame, settings)
        output_path = output_paths[split_name]
        frame.to_csv(output_path, index=False, lineterminator="\n")
        dates = pd.to_datetime(frame[settings.data.time_column])
        artifacts.append(
            SplitArtifact(
                split=split_name,
                relative_path=output_path.relative_to(paths.root.parent).as_posix(),
                row_count=len(frame),
                start_date=dates.min().date(),
                end_date=dates.max().date(),
                positive_rate=(
                    None
                    if split_name is SplitName.TEST
                    else float(frame[settings.data.target_column].mean())
                ),
                sha256=_sha256(output_path),
            )
        )

    digest_material = "\n".join(
        f"{artifact.split.value}:{artifact.sha256}" for artifact in artifacts
    )
    manifest = DatasetManifest(
        schema_version=1,
        dataset_name=settings.data.dataset_name,
        generator_version=settings.data.generator_version,
        generator_code_sha256=generator_code_sha256(),
        generation_config_sha256=generation_config_sha256(settings),
        random_seed=settings.random_seed,
        split_strategy=(
            f"time: train < {settings.data.validation_start.isoformat()}, "
            f"validation < {settings.data.test_start.isoformat()}, test thereafter"
        ),
        target_column=settings.data.target_column,
        feature_columns=settings.features.model_columns,
        excluded_columns=settings.features.forbidden,
        artifacts=tuple(artifacts),
        dataset_sha256=hashlib.sha256(digest_material.encode()).hexdigest(),
        frozen_test=True,
    )
    paths.manifest.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_manifest(root: Path | None = None) -> DatasetManifest:
    path = dataset_paths(root).manifest
    return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_splits(
    settings: ProjectSettings,
    root: Path | None = None,
) -> dict[SplitName, pd.DataFrame]:
    return {
        split_name: load_split(settings, split_name, root) for split_name in SplitName
    }


def load_split(
    settings: ProjectSettings,
    split_name: SplitName,
    root: Path | None = None,
) -> pd.DataFrame:
    """Load and verify only the requested partition."""

    paths = dataset_paths(root)
    if not paths.manifest.exists():
        prepare_dataset(settings, root)
    manifest = load_manifest(root)
    validate_manifest_contract(manifest, settings)
    artifact = next(item for item in manifest.artifacts if item.split is split_name)
    path = {
        SplitName.TRAIN: paths.train,
        SplitName.VALIDATION: paths.validation,
        SplitName.TEST: paths.test,
    }[split_name]
    if _sha256(path) != artifact.sha256:
        raise ValueError(f"Digest mismatch for frozen {split_name.value} data")
    frame = pd.read_csv(path, parse_dates=[settings.data.time_column])
    validate_dataset(frame, settings)
    return frame


def add_intentional_leakage(data: pd.DataFrame) -> pd.DataFrame:
    """Return a teaching-only frame with a post-outcome field."""

    leaked = data.copy()
    leaked["cancellation_reason"] = np.where(
        leaked["churned_30d"] == 1,
        "account_closed",
        "not_applicable",
    )
    return leaked
