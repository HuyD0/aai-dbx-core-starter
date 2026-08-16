"""Judge configuration tests — resolve the plan without calling a model."""

from pathlib import Path

import pytest

from aai_core.agentkit.catalog import get_spec
from aai_core.agentkit.config import ProjectContext
from aai_core.agentkit.errors import ConfigError
from app import judges
from app.judges import judge_model_uri, judge_scorers

ROOT = Path(__file__).resolve().parents[1]


def project():
    return ProjectContext.load(ROOT / "agentkit.yaml")


def test_every_llm_judge_routes_through_the_approved_endpoint():
    model = judge_model_uri(project=project())

    assert model.startswith("endpoints:/")
    for name in ("correctness", "safety", "pension_domain_policy"):
        spec = get_spec(name)
        assert spec.judge is not None
        assert spec.judge.overridable, f"{name} must use the approved endpoint"


def test_a_non_databricks_judge_is_rejected():
    context = project()
    ungoverned = ProjectContext(
        config=context.config,
        settings=context.settings.model_copy(
            update={"models": {"judge-model": {"provider": "azure_apim"}}}
        ),
        root=context.root,
    )

    with pytest.raises(ConfigError):
        ungoverned.judge_model_uri()


def test_context_inputs_are_unambiguous_and_settings_remain_supported():
    context = project()

    with pytest.raises(ValueError, match="settings or project"):
        judge_model_uri(settings=context.settings, project=context)

    assert judge_model_uri(settings=context.settings) == judge_model_uri(
        project=context
    )


def test_judge_scorers_build_the_shared_plan_with_the_governed_model(monkeypatch):
    built = []

    def fake_build_scorer(spec, **options):
        built.append((spec, options))
        return spec.name

    monkeypatch.setattr(judges, "build_scorer", fake_build_scorer)

    scorers = judge_scorers(project=project())

    assert scorers
    assert scorers == [spec.name for spec, _ in built]
    assert all(spec.judge is not None for spec, _ in built)
    assert all(
        options["judge_model_uri"] == "endpoints:/judge-endpoint"
        for _, options in built
    )


def test_domain_policy_is_report_only_until_calibrated():
    spec = get_spec("pension_domain_policy")

    # No default threshold: the judge runs and is reported, but it does not
    # gate a release until held-out human calibration justifies one.
    assert spec.default_threshold is None
    assert spec.judge.prompt_name == "agentkit_judge_domain_policy"
    assert spec.judge.fallback_instructions
