"""Leakage-safe sklearn pipelines and the deliberately weak baseline."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aai_local_classification.contracts import ProjectSettings


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    model_family: str
    complexity_rank: int
    rationale: str


def candidate_specs() -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec(
            name="logistic-regression",
            model_family="logistic_regression",
            complexity_rank=1,
            rationale="Interpretable linear candidate with direct probabilities.",
        ),
        CandidateSpec(
            name="random-forest",
            model_family="random_forest",
            complexity_rank=2,
            rationale="Small nonlinear challenger for interactions and thresholds.",
        ),
    )


def build_preprocessor(settings: ProjectSettings) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(settings.features.numeric)),
            ("categorical", categorical, list(settings.features.categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_baseline(settings: ProjectSettings) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(settings)),
            ("classifier", DummyClassifier(strategy="prior")),
        ]
    )


def build_candidate(spec: CandidateSpec, settings: ProjectSettings) -> Pipeline:
    if spec.model_family == "logistic_regression":
        classifier = LogisticRegression(
            C=1.0,
            max_iter=1_000,
            random_state=settings.random_seed,
        )
    elif spec.model_family == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=180,
            max_depth=8,
            min_samples_leaf=6,
            n_jobs=1,
            random_state=settings.random_seed,
        )
    else:  # pragma: no cover - CandidateSpec is controlled by this package.
        raise ValueError(f"Unsupported model family: {spec.model_family}")
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(settings)),
            ("classifier", classifier),
        ]
    )


def feature_frame(data, settings: ProjectSettings):
    """Return only declared inference features in stable order."""

    features = data.loc[:, list(settings.features.model_columns)].copy()
    # Stable floating-point numeric inputs avoid MLflow signature ambiguity if a
    # future batch represents a missing integer as NaN.
    for column in settings.features.numeric:
        features[column] = features[column].astype("float64")
    return features
