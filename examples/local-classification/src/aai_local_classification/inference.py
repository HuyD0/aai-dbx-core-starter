"""Load the locally approved alias and apply its recorded decision threshold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from aai_local_classification.contracts import ProjectSettings
from aai_local_classification.modeling import feature_frame
from aai_local_classification.tracking import configure_mlflow


@dataclass(frozen=True)
class ChampionPredictor:
    model: object
    threshold: float
    model_name: str
    model_version: str

    def predict(self, data: pd.DataFrame, settings: ProjectSettings) -> pd.DataFrame:
        features = feature_frame(data, settings)
        classes = list(self.model.classes_)
        probabilities = self.model.predict_proba(features)[:, classes.index(1)]
        return pd.DataFrame(
            {
                "churn_probability": probabilities,
                "churn_prediction": (probabilities >= self.threshold).astype(int),
                "model_name": self.model_name,
                "model_version": self.model_version,
            },
            index=data.index,
        )


def load_champion(
    settings: ProjectSettings,
    project_root: Path | None = None,
) -> ChampionPredictor:
    configure_mlflow(settings, project_root)
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(
        settings.registered_model_name,
        "champion",
    )
    threshold = float(version.tags["decision_threshold"])
    model_uri = f"models:/{settings.registered_model_name}@champion"
    model = mlflow.sklearn.load_model(model_uri)
    return ChampionPredictor(
        model=model,
        threshold=threshold,
        model_name=settings.registered_model_name,
        model_version=str(version.version),
    )
