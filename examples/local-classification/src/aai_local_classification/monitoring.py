"""Small local drift simulation for the post-release learning notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aai_local_classification.contracts import MonitoringReport, ProjectSettings


def shifted_batch(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Create an explicitly simulated future batch with plausible input shift."""

    rng = np.random.default_rng(seed)
    shifted = data.sample(frac=1.0, replace=True, random_state=seed).reset_index(
        drop=True
    )
    shifted["monthly_fee"] = (shifted["monthly_fee"] * 1.10).round(2)
    shifted["usage_hours_30d"] = np.clip(
        shifted["usage_hours_30d"] - rng.normal(4.0, 1.0, len(shifted)),
        0,
        None,
    ).round(2)
    paid_mask = rng.random(len(shifted)) < 0.18
    shifted.loc[paid_mask, "signup_channel"] = "paid_search"
    return shifted


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    ref = reference.dropna().to_numpy(dtype=float)
    cur = current.dropna().to_numpy(dtype=float)
    if not len(ref) or not len(cur):
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    epsilon = 1e-6
    ref_share = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
    cur_share = np.clip(cur_counts / cur_counts.sum(), epsilon, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def categorical_total_variation(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.fillna("<missing>").astype(str).value_counts(normalize=True)
    cur = current.fillna("<missing>").astype(str).value_counts(normalize=True)
    categories = ref.index.union(cur.index)
    return float(
        0.5
        * (
            ref.reindex(categories, fill_value=0)
            - cur.reindex(categories, fill_value=0)
        )
        .abs()
        .sum()
    )


def compare_batches(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    settings: ProjectSettings,
) -> MonitoringReport:
    missing = sorted(set(settings.features.model_columns) - set(current.columns))
    if missing:
        raise ValueError(f"Current batch is missing model features: {missing}")
    numeric = {
        column: population_stability_index(reference[column], current[column])
        for column in settings.features.numeric
    }
    categorical = {
        column: categorical_total_variation(reference[column], current[column])
        for column in settings.features.categorical
    }
    missing_delta = {
        column: float(current[column].isna().mean() - reference[column].isna().mean())
        for column in settings.features.model_columns
    }
    return MonitoringReport(
        schema_version=1,
        reference_rows=len(reference),
        current_rows=len(current),
        numeric_psi=numeric,
        categorical_total_variation=categorical,
        missing_rate_delta=missing_delta,
        maximum_numeric_psi=max(numeric.values(), default=0.0),
        maximum_categorical_total_variation=max(categorical.values(), default=0.0),
        interpretation=(
            "Input drift is a diagnostic signal, not proof that model quality changed; "
            "performance and calibration require delayed outcome labels."
        ),
    )
