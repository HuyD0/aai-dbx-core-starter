"""Unit tests for agentkit.yaml loading and the project composition root."""

import math

import pytest
from pydantic import ValidationError

from aai_core.agentkit.config import (
    AgentkitConfig,
    ProjectContext,
    find_agentkit_config,
    load_config,
    parse_threshold,
)
from aai_core.agentkit.errors import ConfigError, UnknownScorerError
from aai_core.evaluation import MetricDirection
from aai_core.providers.types import ProviderConfigurationError
from aai_core.testing import dev_settings

MINIMAL = """\
version: 1
agent: src/app/example_agent.py:respond
dataset: evals/data/golden_cases.json
"""


def _write(tmp_path, text=MINIMAL, name="agentkit.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _config(**overrides):
    values = {
        "version": 1,
        "agent": "src/app/example_agent.py:respond",
        "dataset": "evals/data/golden_cases.json",
    }
    values.update(overrides)
    return AgentkitConfig(**values)


def test_minimal_three_line_config_loads_with_defaults(tmp_path):
    config = load_config(_write(tmp_path))

    assert config.agent == "src/app/example_agent.py:respond"
    assert config.dataset == "evals/data/golden_cases.json"
    assert config.smoke.rows == 20
    assert config.concurrency == 8
    assert config.baseline.file == "evals/baseline.json"
    assert dict(config.thresholds) == {}
    assert config.scorers.judge_model == "judge-model"
    assert config.request_mapping.request_field == "input"


def test_unknown_keys_are_rejected(tmp_path):
    path = _write(tmp_path, MINIMAL + "surprise: true\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "surprise" in str(excinfo.value)


def test_economics_defaults_are_report_only(tmp_path):
    config = load_config(_write(tmp_path))

    assert config.economics.enabled is True
    assert config.economics.price_per_1m_input_tokens is None
    assert config.economics.price_per_1m_output_tokens is None


def test_economics_price_pair_loads_and_coerces(tmp_path):
    text = MINIMAL + (
        "economics:\n"
        "  price_per_1m_input_tokens: 2.5\n"
        "  price_per_1m_output_tokens: 10\n"
    )

    config = load_config(_write(tmp_path, text))

    assert config.economics.price_per_1m_input_tokens == 2.5
    assert config.economics.price_per_1m_output_tokens == 10.0


def test_economics_price_pair_is_both_or_neither(tmp_path):
    text = MINIMAL + "economics:\n  price_per_1m_input_tokens: 2.5\n"

    with pytest.raises(ConfigError) as excinfo:
        load_config(_write(tmp_path, text))
    assert "pair" in str(excinfo.value)


def test_economics_unknown_keys_are_rejected(tmp_path):
    text = MINIMAL + "economics:\n  price_table: builtin\n"

    with pytest.raises(ConfigError) as excinfo:
        load_config(_write(tmp_path, text))
    assert "price_table" in str(excinfo.value)


def test_missing_config_names_the_expected_file(tmp_path):
    with pytest.raises(ConfigError) as excinfo:
        load_config(tmp_path / "agentkit.yaml")
    assert "agentkit init" in str(excinfo.value)


def test_discovery_prefers_env_override_then_upward_search(tmp_path):
    config_path = _write(tmp_path)
    nested = tmp_path / "src" / "app"
    nested.mkdir(parents=True)

    found = find_agentkit_config(nested, environ={})
    assert found == config_path

    override = _write(tmp_path / "src", name="other.yaml")
    found = find_agentkit_config(nested, environ={"AGENTKIT_CONFIG": str(override)})
    assert found == override


@pytest.mark.parametrize(
    ("expression", "direction", "required"),
    [
        (">=0.7", MetricDirection.HIGHER, 0.7),
        ("<= 2", MetricDirection.LOWER, 2.0),
        (">= 1.0", MetricDirection.HIGHER, 1.0),
    ],
)
def test_threshold_expressions_parse(expression, direction, required):
    rule = parse_threshold("quality/mean", expression)

    assert rule.metric == "quality/mean"
    assert rule.direction is direction
    assert rule.required == required
    assert rule.max_regression is None


def test_strict_comparisons_move_one_ulp():
    above = parse_threshold("m", ">0.5")
    below = parse_threshold("m", "<0.5")

    assert above.direction is MetricDirection.HIGHER
    assert above.required == math.nextafter(0.5, math.inf)
    assert below.direction is MetricDirection.LOWER
    assert below.required == math.nextafter(0.5, -math.inf)


@pytest.mark.parametrize("expression", ["about 0.7", "", "=>0.7", ">=high", ">=inf"])
def test_invalid_threshold_expressions_fail(expression):
    with pytest.raises(ConfigError):
        parse_threshold("quality/mean", expression)


def test_config_with_bad_threshold_fails_at_load(tmp_path):
    path = _write(tmp_path, MINIMAL + 'thresholds:\n  correctness: "0.7"\n')

    with pytest.raises(ConfigError):
        load_config(path)


def test_unknown_scorer_in_add_lists_the_catalog(tmp_path):
    path = _write(tmp_path, MINIMAL + "scorers:\n  add: [made_up_scorer]\n")

    with pytest.raises(UnknownScorerError) as excinfo:
        load_config(path)
    message = str(excinfo.value)
    assert "made_up_scorer" in message
    assert "correctness" in message
    assert "scorers ls" in message


def test_same_scorer_cannot_be_added_and_removed_during_config_load(tmp_path):
    path = _write(
        tmp_path,
        MINIMAL
        + "scorers:\n"
        + "  add: [keyword_coverage]\n"
        + "  remove: [keyword_coverage]\n",
    )

    with pytest.raises(ConfigError, match="both scorers.add and scorers.remove"):
        load_config(path)


@pytest.mark.parametrize("key", ["keyword_coverage", "keyword_coverage/mean"])
def test_removing_a_thresholded_scorer_fails(tmp_path, key):
    path = _write(
        tmp_path,
        MINIMAL
        + "scorers:\n  remove: [keyword_coverage]\n"
        + f'thresholds:\n  {key}: ">=0.6"\n',
    )

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "keyword_coverage" in str(excinfo.value)


def test_project_context_merges_platform_settings(tmp_path):
    _write(tmp_path)
    (tmp_path / "aai-platform.yml").write_text(
        """\
platform:
  application: quality-eval
  project: agent-quality
  team: pension-ai
  owner_group: group:pension-ai-owners
  cost_center: CC-9999
  repository: example/agent-quality
providers:
  models:
    judge-model:
      provider: databricks
      deployment: judge-endpoint
""",
        encoding="utf-8",
    )

    project = ProjectContext.load(tmp_path / "agentkit.yaml", environ={})

    assert project.root == tmp_path.resolve()
    assert project.settings.resource.application == "quality-eval"
    assert project.judge_model_uri() == "endpoints:/judge-endpoint"
    assert project.baseline_path == project.root / "evals" / "baseline.json"
    assert project.results_dir == project.root / ".aai" / "agentkit" / "results"


def test_project_context_works_without_platform_file(tmp_path):
    _write(tmp_path)

    project = ProjectContext.load(tmp_path / "agentkit.yaml", environ={})

    assert project.settings.resource.environment == "dev"


def test_judge_model_uri_requires_a_databricks_deployment(tmp_path):
    project = ProjectContext(
        config=_config(),
        settings=dev_settings(
            models={"judge-model": {"provider": "azure_apim", "deployment": "j"}}
        ),
        root=tmp_path,
    )

    with pytest.raises(ConfigError) as excinfo:
        project.judge_model_uri()
    assert "provider 'databricks'" in str(excinfo.value)

    missing = ProjectContext(config=_config(), settings=dev_settings(), root=tmp_path)
    with pytest.raises(ConfigError):
        missing.judge_model_uri()


@pytest.mark.parametrize(
    ("deployment", "match"),
    (
        ("replace-with-judge-endpoint", "placeholder"),
        ("endpoints:/judge-endpoint", "endpoint name"),
        ("judge endpoint", "endpoint name"),
    ),
)
def test_agentkit_judge_uses_the_canonical_strict_resolver(tmp_path, deployment, match):
    project = ProjectContext(
        config=_config(),
        settings=dev_settings(
            models={
                "judge-model": {
                    "provider": "databricks",
                    "deployment": deployment,
                }
            }
        ),
        root=tmp_path,
    )

    with pytest.raises(ConfigError, match=match) as excinfo:
        project.judge_model_uri()

    assert isinstance(excinfo.value.__cause__, ProviderConfigurationError)


def test_experiment_manager_uses_platform_naming(tmp_path):
    project = ProjectContext(config=_config(), settings=dev_settings(), root=tmp_path)

    manager = project.experiment_manager(mlflow_module=object())
    assert manager.experiment_name == "/Shared/test-team-test-project-test-app"


def test_regression_budget_values_must_be_non_negative(tmp_path):
    path = _write(tmp_path, MINIMAL + "regression_budget:\n  quality/mean: -1\n")

    with pytest.raises(ConfigError):
        load_config(path)


def test_yaml_lists_and_integers_coerce_cleanly(tmp_path):
    path = _write(
        tmp_path,
        MINIMAL
        + "strata: [category]\n"
        + "budget:\n  judge_price_per_1m_tokens: 5\n"
        + "regression_budget:\n  quality/mean: 1\n",
    )

    config = load_config(path)
    assert config.strata == ("category",)
    assert config.budget.judge_price_per_1m_tokens == 5.0
    assert dict(config.regression_budget) == {"quality/mean": 1.0}


def test_a_regression_budget_on_a_removed_scorer_is_refused(tmp_path):
    """A budget is a gate rule, so removing its scorer is a contradiction.

    `build_policy` builds a rule from `regression_budget` whether or not
    the scorer still runs, so the run pays for every judge and then fails
    on a metric that was never going to appear. The template's default
    budget names keyword_coverage, so this is reachable by editing one
    line of generated config.
    """

    with pytest.raises(ConfigError) as excinfo:
        load_config(
            _write(
                tmp_path,
                "version: 1\n"
                "agent: src/app/agent.py:respond\n"
                "dataset: evals/data/golden_cases.json\n"
                "scorers:\n"
                "  remove: [keyword_coverage]\n"
                "regression_budget:\n"
                "  keyword_coverage/mean: 0.05\n",
            )
        )

    message = str(excinfo.value)
    assert "regression_budget" in message
    assert "keyword_coverage" in message


def test_a_threshold_on_a_removed_scorer_is_still_refused(tmp_path):
    with pytest.raises(ConfigError) as excinfo:
        load_config(
            _write(
                tmp_path,
                "version: 1\n"
                "agent: src/app/agent.py:respond\n"
                "dataset: evals/data/golden_cases.json\n"
                "scorers:\n"
                "  remove: [keyword_coverage]\n"
                "thresholds:\n"
                "  keyword_coverage: '>=0.6'\n",
            )
        )

    assert "thresholds" in str(excinfo.value)


def test_judge_identity_reads_the_served_entity(tmp_path):
    """What the endpoint serves, not what it is called."""

    from types import SimpleNamespace

    served = SimpleNamespace(
        config=SimpleNamespace(
            served_entities=[
                SimpleNamespace(entity_name="main.models.judge", entity_version="3")
            ]
        )
    )
    client = SimpleNamespace(serving_endpoints=SimpleNamespace(get=lambda name: served))
    project = ProjectContext(
        config=_config(),
        settings=dev_settings(
            models={"judge-model": {"provider": "databricks", "deployment": "j"}}
        ),
        root=tmp_path,
    )

    assert project.judge_model_identity(client=client) == "main.models.judge/3"


def test_an_unreadable_endpoint_yields_no_identity(tmp_path):
    from types import SimpleNamespace

    def _denied(name):
        raise PermissionError("requires CAN_VIEW")

    client = SimpleNamespace(serving_endpoints=SimpleNamespace(get=_denied))
    project = ProjectContext(
        config=_config(),
        settings=dev_settings(
            models={"judge-model": {"provider": "databricks", "deployment": "j"}}
        ),
        root=tmp_path,
    )

    assert project.judge_model_identity(client=client) is None


def test_integrity_config_defaults_are_inert():
    config = AgentkitConfig(
        version=1, agent="src/app.py:respond", dataset="evals/data/cases.json"
    )
    assert config.integrity.consistency_sample == 0
    assert config.integrity.require_anchors is False
    assert config.integrity.require_calibration is False
    assert config.integrity.anchors == "evals/judge_anchors.json"


def test_integrity_config_parses_and_bounds_the_knobs():
    config = AgentkitConfig(
        version=1,
        agent="src/app.py:respond",
        dataset="evals/data/cases.json",
        integrity={
            "consistency_sample": 8,
            "max_self_inconsistency": 0.25,
            "anchors": "evals/frozen.json",
            "max_anchor_drift": 0.05,
            "require_anchors": True,
        },
    )
    assert config.integrity.consistency_sample == 8
    assert config.integrity.max_anchor_drift == 0.05
    assert config.integrity.anchors == "evals/frozen.json"
    with pytest.raises(ValidationError):
        AgentkitConfig(
            version=1,
            agent="src/app.py:respond",
            dataset="evals/data/cases.json",
            integrity={"unknown_knob": True},
        )
