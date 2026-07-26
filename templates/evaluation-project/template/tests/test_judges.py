"""Judge configuration tests — construct scorers without calling a model."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from app.judges import DOMAIN_POLICY_GUIDELINES, judge_model_uri, judge_scorers

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def fake_mlflow_scorers(monkeypatch):
    class Judge:
        default_name = ""

        def __init__(self, *, name=None, model=None, guidelines=None):
            self.name = name or self.default_name
            self.model = model
            self.guidelines = guidelines

    class Correctness(Judge):
        default_name = "correctness"

    class Safety(Judge):
        default_name = "safety"

    class Guidelines(Judge):
        default_name = "guidelines"

    scorers = ModuleType("mlflow.genai.scorers")
    scorers.Correctness = Correctness
    scorers.Guidelines = Guidelines
    scorers.Safety = Safety
    genai = ModuleType("mlflow.genai")
    genai.scorers = scorers
    mlflow = ModuleType("mlflow")
    mlflow.genai = genai
    for name, module in {
        "mlflow": mlflow,
        "mlflow.genai": genai,
        "mlflow.genai.scorers": scorers,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def settings():
    return SimpleNamespace(
        models={
            "judge-model": {
                "provider": "databricks",
                "deployment": "approved-judge-endpoint",
            }
        }
    )


def test_every_llm_judge_uses_the_approved_endpoint():
    model = judge_model_uri(settings())
    scorers = judge_scorers(settings())

    assert model == "endpoints:/approved-judge-endpoint"
    assert [judge.name for judge in scorers] == [
        "correctness",
        "safety",
        "domain_policy",
    ]
    assert all(judge.model == model for judge in scorers)


def test_domain_policy_is_executable_and_report_only_until_calibrated():
    domain_policy = judge_scorers(settings())[-1]
    gate = json.loads((ROOT / "evals" / "gate_config.json").read_text("utf-8"))
    gated = {item["metric"] for item in gate["thresholds"]}

    assert domain_policy.guidelines == DOMAIN_POLICY_GUIDELINES
    assert "domain_policy/mean" not in gated
    assert "domain_policy/mean" not in gate["judge_metrics"]
    assert "domain_policy/mean" in gate["report_only_judge_metrics"]
