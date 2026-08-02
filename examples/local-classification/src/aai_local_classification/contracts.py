"""Strict persisted-evidence contracts for the local classification course."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base contract that rejects drift in configuration and evidence shapes."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SplitName(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DataSettings(StrictModel):
    dataset_name: str
    generator_version: str
    start_month: date
    months: int = Field(ge=12)
    rows_per_month: int = Field(ge=20)
    validation_start: date
    test_start: date
    target_column: str
    id_column: str
    time_column: str


class FeatureSettings(StrictModel):
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    forbidden: tuple[str, ...]

    @property
    def model_columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical


class SelectionSettings(StrictModel):
    primary_metric: Literal["average_precision"]
    simpler_model_tolerance: float = Field(ge=0, le=1)
    false_negative_cost: float = Field(gt=0)
    false_positive_cost: float = Field(gt=0)
    minimum_validation_precision: float = Field(ge=0, le=1)
    minimum_validation_recall: float = Field(ge=0, le=1)


class PromotionGateSettings(StrictModel):
    minimum_test_average_precision: float = Field(ge=0, le=1)
    minimum_test_average_precision_lift: float = Field(ge=0, le=1)
    minimum_test_recall: float = Field(ge=0, le=1)
    maximum_test_brier_score: float = Field(ge=0, le=1)
    maximum_test_cost_per_1000: float = Field(ge=0)
    maximum_slice_recall_gap: float = Field(ge=0, le=1)


class ProjectSettings(StrictModel):
    schema_version: int
    project_name: str
    experiment_name: str
    registered_model_name: str
    random_seed: int
    data: DataSettings
    features: FeatureSettings
    selection: SelectionSettings
    promotion_gate: PromotionGateSettings


class SplitArtifact(StrictModel):
    split: SplitName
    relative_path: str
    row_count: int = Field(ge=1)
    start_date: date
    end_date: date
    positive_rate: float | None = Field(default=None, ge=0, le=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetManifest(StrictModel):
    schema_version: int
    dataset_name: str
    generator_version: str
    generator_code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    random_seed: int
    split_strategy: str
    target_column: str
    feature_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    artifacts: tuple[SplitArtifact, ...]
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_test: bool


class PromotionDecision(StrEnum):
    ADOPT = "adopt"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


class PromotionChecks(StrictModel):
    minimum_test_average_precision: bool
    minimum_test_average_precision_lift: bool
    minimum_test_recall: bool
    maximum_test_brier_score: bool
    maximum_test_cost_per_1000: bool
    maximum_slice_recall_gap: bool


class GateEvidence(StrictModel):
    schema_version: int
    decision: PromotionDecision
    selected_candidate: str
    selected_run_id: str
    selected_model_id: str
    test_run_id: str
    threshold: float = Field(ge=0, le=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gate_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: dict[str, float]
    checks: PromotionChecks
    rationale: str


class ClassificationMetrics(StrictModel):
    threshold: float = Field(ge=0, le=1)
    row_count: int = Field(ge=1)
    positive_rate: float = Field(ge=0, le=1)
    predicted_positive_rate: float = Field(ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)
    balanced_accuracy: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    specificity: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    average_precision: float = Field(ge=0, le=1)
    roc_auc: float = Field(ge=0, le=1)
    log_loss: float = Field(ge=0)
    brier_score: float = Field(ge=0, le=1)
    true_negatives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    true_positives: int = Field(ge=0)
    cost_per_1000: float = Field(ge=0)


class BaselineEvidence(StrictModel):
    schema_version: int
    run_id: str
    model_id: str
    model_uri: str
    metrics: ClassificationMetrics
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ThresholdSelection(StrictModel):
    threshold: float = Field(ge=0, le=1)
    validation_metrics: ClassificationMetrics
    feasible_threshold_count: int = Field(ge=0)
    selection_rule: str


class CandidateResult(StrictModel):
    schema_version: int
    candidate_name: str
    run_id: str
    model_id: str
    model_uri: str
    threshold_selection: ThresholdSelection
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SelectionEvidence(StrictModel):
    schema_version: int
    selection_run_id: str
    primary_metric: str
    selection_rule: str
    selected_candidate: str
    selected_run_id: str
    selected_model_id: str
    selected_model_uri: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates: tuple[CandidateResult, ...]


class PromotionEvidence(StrictModel):
    schema_version: int
    registered: Literal[True]
    model_name: str
    model_version: int = Field(ge=1)
    alias: Literal["champion"]
    model_uri: str
    selected_model_id: str
    test_run_id: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gate_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MonitoringReport(StrictModel):
    schema_version: int
    reference_rows: int = Field(ge=1)
    current_rows: int = Field(ge=1)
    numeric_psi: dict[str, float]
    categorical_total_variation: dict[str, float]
    missing_rate_delta: dict[str, float]
    maximum_numeric_psi: float = Field(ge=0)
    maximum_categorical_total_variation: float = Field(ge=0, le=1)
    interpretation: str
