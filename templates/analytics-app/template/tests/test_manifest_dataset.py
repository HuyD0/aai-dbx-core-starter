"""The Hub manifest and release gate share one governed dataset identity."""

from pathlib import Path

import yaml

from app.config import DATASET_NAME

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_dataset_is_the_dataset_synced_and_consumed_by_the_gate():
    platform = yaml.safe_load((ROOT / "aai-platform.yml").read_text())["platform"]
    manifest = yaml.safe_load((ROOT / "ai-app.yaml").read_text())
    expected = f"{platform['catalog']}.{platform['schema']}.{DATASET_NAME}"

    assert manifest["spec"]["evaluation"]["dataset"] == expected
    assert "DATASET_NAME" in (ROOT / "scripts" / "sync_dataset.py").read_text()
    assert "DATASET_NAME" in (ROOT / "evals" / "evaluate.py").read_text()
    assert (
        "mlflow.genai.datasets.get_dataset"
        in (ROOT / "evals" / "evaluate.py").read_text()
    )
