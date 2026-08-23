"""The Hub registration and AgentKit gate share reviewed source cases."""

from pathlib import Path

import yaml

from app.config import DATASET_NAME

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_dataset_and_agentkit_gate_share_the_reviewed_cases():
    platform = yaml.safe_load((ROOT / "aai-platform.yml").read_text())["platform"]
    manifest = yaml.safe_load((ROOT / "ai-app.yaml").read_text())
    agentkit = yaml.safe_load((ROOT / "agentkit.yaml").read_text())
    expected = f"{platform['catalog']}.{platform['schema']}.{DATASET_NAME}"
    dataset_ref = agentkit["dataset"]
    sync_source = (ROOT / "scripts" / "sync_dataset.py").read_text()

    assert manifest["spec"]["evaluation"]["dataset"] == expected
    assert dataset_ref == "evals/data/golden_cases.json"
    assert (ROOT / dataset_ref).is_file()
    assert "DATASET_NAME" in sync_source
    assert Path(dataset_ref).name in sync_source
    assert "mlflow.genai.datasets.get_dataset" in sync_source
