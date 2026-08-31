"""Unit tests for continuous logprob-weighted scoring (the verifier path).

Everything runs with zero cloud access: the verifier model is a fake that
returns OpenAI-shaped top logprobs, and the MLflow module is the same
injected-surface fake the runner tests use.
"""

import inspect
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_core.agentkit.config import ProjectContext, load_config
from aai_core.agentkit.continuous import (
    CONTINUOUS_SCORER_NAME,
    CORRECTNESS_CRITERIA,
    ContinuousScoringConfig,
    ContinuousVerifier,
    VerifierStats,
    activate_run,
    detect_logprob_support,
    graded_candidates,
    kendall_tau_b,
    plan_run,
    score_labels,
    tie_rate,
    top_logprob_pairs,
    weigh_top_logprobs,
)
from aai_core.agentkit.errors import BudgetExceededError, ConfigError
from aai_core.agentkit.runner import run_scoring
from aai_core.providers.types import ModelResponse, ProviderRequestError


def _logprob_raw(pairs):
    """An OpenAI-shaped raw response carrying top logprobs."""

    alternatives = [
        SimpleNamespace(token=token, logprob=logprob) for token, logprob in pairs
    ]
    content = [SimpleNamespace(token=pairs[0][0], top_logprobs=alternatives)]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=pairs[0][0]),
                logprobs=SimpleNamespace(content=content),
            )
        ]
    )


class FakeLogprobModel:
    """ChatModel fake returning scripted top logprobs (or none at all)."""

    def __init__(self, pairs_for=None, *, supports_logprobs=True, error=None):
        self.pairs_for = pairs_for or (
            lambda prompt: [("T", math.log(0.6)), ("S", math.log(0.3))]
        )
        self.supports_logprobs = supports_logprobs
        self.error = error
        self.logical_name = "verifier-model"
        self.provider = "fake"
        self.capabilities = SimpleNamespace(tool_calling=False)
        self.native_client = None
        self.requests = []

    def create_native_async_client(self):
        raise NotImplementedError

    def generate(self, messages, **options):
        if self.error is not None:
            raise self.error
        self.requests.append({"messages": list(messages), **options})
        prompt = messages[-1]["content"]
        raw = (
            _logprob_raw(self.pairs_for(prompt))
            if self.supports_logprobs
            else SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="A"), logprobs=None)
                ]
            )
        )
        return ModelResponse(
            content="A",
            provider="fake",
            logical_name=self.logical_name,
            model="fake-verifier",
            latency_ms=1.0,
            usage={"prompt_tokens": 25, "completion_tokens": 1},
            raw=raw,
        )


# --- the weighting math ------------------------------------------------------


def test_score_labels_are_single_uppercase_letters():
    assert score_labels(5) == ("A", "B", "C", "D", "E")
    assert score_labels(20)[-1] == "T"
    with pytest.raises(ConfigError):
        score_labels(1)
    with pytest.raises(ConfigError):
        score_labels(27)


def test_weighted_score_normalizes_by_retained_mass():
    pairs = [("B", math.log(0.8)), ("C", math.log(0.1)), ("the", math.log(0.05))]
    judgment = weigh_top_logprobs(pairs, granularity=5, low_mass_threshold=0.5)
    # exp() of each retained logprob: 0.8 on B (0.25) and 0.1 on C (0.5);
    # mass 0.9, weighted (0.8*0.25 + 0.1*0.5) / 0.9.
    assert judgment.normalization_mass == pytest.approx(0.9)
    assert judgment.score == pytest.approx((0.8 * 0.25 + 0.1 * 0.5) / 0.9)
    assert judgment.top_label == "B"
    assert judgment.discrete_score == pytest.approx(0.25)
    assert not judgment.low_mass


def test_token_variants_collapse_into_one_label():
    pairs = [("B", math.log(0.4)), (" B", math.log(0.3)), ("b", math.log(0.2))]
    judgment = weigh_top_logprobs(pairs, granularity=5, low_mass_threshold=0.5)
    assert judgment.normalization_mass == pytest.approx(0.9)
    assert judgment.score == pytest.approx(0.25)


def test_low_mass_is_flagged_not_failed():
    pairs = [("C", math.log(0.3)), ("well", math.log(0.6))]
    judgment = weigh_top_logprobs(pairs, granularity=5, low_mass_threshold=0.5)
    assert judgment.low_mass
    assert judgment.score == pytest.approx(0.5)


def test_no_score_token_at_all_is_invalid():
    pairs = [("well", math.log(0.6)), ("!", math.log(0.2))]
    assert weigh_top_logprobs(pairs, granularity=5, low_mass_threshold=0.5) is None


def test_top_logprob_pairs_reads_attribute_and_mapping_shapes():
    attr_shaped = _logprob_raw([("A", -0.1), ("B", -2.0)])
    assert top_logprob_pairs(attr_shaped) == [("A", -0.1), ("B", -2.0)]
    dict_shaped = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "C", "logprob": -0.5},
                                {"token": "D", "logprob": -1.5},
                            ]
                        }
                    ]
                }
            }
        ]
    }
    assert top_logprob_pairs(dict_shaped) == [("C", -0.5), ("D", -1.5)]
    assert top_logprob_pairs(SimpleNamespace(choices=[])) is None
    no_logprobs = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="A"), logprobs=None)]
    )
    assert top_logprob_pairs(no_logprobs) is None


# --- the verifier ------------------------------------------------------------


def test_score_response_averages_over_criteria_and_repeats():
    model = FakeLogprobModel(lambda prompt: [("E", math.log(0.9))])
    verifier = ContinuousVerifier(model=model, granularity=5, repeats=2)
    result = verifier.score_response(request="q", response="r", expected="e")
    assert result.continuous == pytest.approx(1.0)
    assert result.discrete == pytest.approx(1.0)
    assert result.criteria_scored == len(CORRECTNESS_CRITERIA)
    # criteria x repeats calls, every one recorded
    assert verifier.stats.calls == len(CORRECTNESS_CRITERIA) * 2
    assert verifier.stats.input_tokens == verifier.stats.calls * 25


def test_criterion_prompt_names_the_scale_and_criterion():
    model = FakeLogprobModel(lambda prompt: [("C", math.log(0.9))])
    verifier = ContinuousVerifier(model=model, granularity=10, repeats=1)
    verifier.score_response(request="the question", response="r", expected="e")
    prompt = model.requests[0]["messages"][-1]["content"]
    assert "from A to J" in prompt
    assert CORRECTNESS_CRITERIA[0].instruction in prompt
    assert "the question" in prompt
    # single-token verdict: one max token, logprobs requested
    assert model.requests[0]["max_tokens"] == 1
    assert model.requests[0]["provider_options"]["logprobs"] is True
    assert model.requests[0]["provider_options"]["top_logprobs"] == 20


def test_pairwise_alternation_cancels_positional_bias():
    # A verifier that always slightly prefers whatever sits in the A slot:
    # with an even number of repeats the bias cancels to exactly a tie.
    model = FakeLogprobModel(lambda prompt: [("D", math.log(0.9))])
    verifier = ContinuousVerifier(model=model, granularity=5, repeats=2)
    preference = verifier.compare(request="q", first="left", second="right")
    slot_a_score = 3 / 4  # label D on a 5-point scale
    assert preference == pytest.approx((slot_a_score + (1.0 - slot_a_score)) / 2)
    # The two repeats really did swap the slots.
    first_prompt = model.requests[0]["messages"][-1]["content"]
    second_prompt = model.requests[1]["messages"][-1]["content"]
    assert "Response A: left" in first_prompt
    assert "Response A: right" in second_prompt


def test_invalid_judgments_are_skipped_and_counted():
    model = FakeLogprobModel(lambda prompt: [("nope", math.log(0.9))])
    verifier = ContinuousVerifier(model=model, granularity=5, repeats=1)
    assert verifier.score_response(request="q", response="r", expected="e") is None
    assert verifier.stats.invalid_calls == len(CORRECTNESS_CRITERIA)
    metrics = verifier.stats.metrics()
    assert metrics["continuous/invalid_rate"] == 1.0


def test_stats_metrics_report_mass_and_low_mass_rate():
    stats = VerifierStats()
    model = FakeLogprobModel(lambda prompt: [("B", math.log(0.3))])
    verifier = ContinuousVerifier(model=model, granularity=5, repeats=1, stats=stats)
    verifier.score_criterion(
        request="q", response="r", expected="e", criterion=CORRECTNESS_CRITERIA[0]
    )
    metrics = stats.metrics()
    assert metrics["continuous/judge_calls"] == 1.0
    assert metrics["continuous/low_mass_rate"] == 1.0
    assert metrics["continuous/normalization_mass_mean"] == pytest.approx(0.3)
    assert metrics["continuous/normalization_mass_min"] == pytest.approx(0.3)


# --- capability detection ----------------------------------------------------


def test_detect_logprob_support_true_and_false():
    assert detect_logprob_support(FakeLogprobModel()) is True
    assert detect_logprob_support(FakeLogprobModel(supports_logprobs=False)) is False


def test_backend_refusing_the_parameters_is_a_capability_answer():
    refused = ProviderRequestError(
        "Provider request failed", provider="fake", status_code=400
    )
    assert detect_logprob_support(FakeLogprobModel(error=refused)) is False


def test_auth_and_transport_failures_propagate():
    denied = ProviderRequestError(
        "Provider request failed", provider="fake", status_code=403
    )
    with pytest.raises(ProviderRequestError):
        detect_logprob_support(FakeLogprobModel(error=denied))


# --- ranking helpers ---------------------------------------------------------


def test_kendall_tau_b_agreement_and_inversion():
    assert kendall_tau_b([3, 2, 1, 0], [0.9, 0.7, 0.4, 0.1]) == pytest.approx(1.0)
    assert kendall_tau_b([3, 2, 1, 0], [0.1, 0.4, 0.7, 0.9]) == pytest.approx(-1.0)
    assert kendall_tau_b([3, 2, 1, 0], [1.0, 1.0, 1.0, 1.0]) is None
    assert kendall_tau_b([1.0], [1.0]) is None


def test_kendall_tau_b_penalizes_ties_against_a_strict_reference():
    strict = kendall_tau_b([3, 2, 1, 0], [0.9, 0.6, 0.3, 0.1])
    tied = kendall_tau_b([3, 2, 1, 0], [0.9, 0.6, 0.6, 0.1])
    assert strict == pytest.approx(1.0)
    assert tied < 1.0


def test_tie_rate_counts_unordered_pairs():
    assert tie_rate([1.0, 1.0, 0.5, None]) == pytest.approx(1 / 3)
    assert tie_rate([1.0]) is None
    assert tie_rate([]) is None
    # float noise below the rounding threshold still ties
    assert tie_rate([0.5, 0.5 + 1e-12]) == pytest.approx(1.0)


def test_graded_candidates_are_deterministic_and_ordered():
    expected = "Standard orders can be returned within thirty days. Bring proof."
    first = graded_candidates(expected, wrong="Something else entirely.")
    second = graded_candidates(expected, wrong="Something else entirely.")
    assert first == second
    assert [candidate.rank for candidate in first] == [3, 2, 1, 0]
    assert first[0].text == expected
    assert first[1].text != expected
    assert "withdrawn" in first[2].text
    assert first[3].text == "Something else entirely."


# --- run planning and activation --------------------------------------------


def _shape(expectation_keys=("expected_response",), rows=4):
    from aai_core.agentkit.datasets import DatasetShape

    return DatasetShape(
        row_count=rows,
        input_keys=("question",),
        has_outputs=True,
        expectation_keys=tuple(expectation_keys),
        has_traces=False,
        strata_values={},
    )


def test_plan_run_is_off_by_default_and_off_without_judges():
    config = ContinuousScoringConfig()
    assert (
        plan_run(
            config,
            default_judge_model="judge-model",
            shape=_shape(),
            judges_enabled=True,
        )
        is None
    )
    enabled = ContinuousScoringConfig(enabled=True)
    assert (
        plan_run(
            enabled,
            default_judge_model="judge-model",
            shape=_shape(),
            judges_enabled=False,
        )
        is None
    )


def test_plan_run_counts_rows_criteria_repeats_and_probe():
    config = ContinuousScoringConfig(enabled=True, repeats=2)
    plan = plan_run(
        config,
        default_judge_model="judge-model",
        shape=_shape(rows=4),
        judges_enabled=True,
    )
    assert plan.active
    assert plan.judge_calls == 4 * len(CORRECTNESS_CRITERIA) * 2 + 1
    assert plan.judge_model_name == "judge-model"
    assert "verifier call(s)" in plan.message()


def test_plan_run_blocks_without_expectations_and_names_the_gap():
    config = ContinuousScoringConfig(enabled=True, judge_model="verifier-model")
    plan = plan_run(
        config,
        default_judge_model="judge-model",
        shape=_shape(expectation_keys=()),
        judges_enabled=True,
    )
    assert not plan.active
    assert plan.judge_calls == 0
    assert "expected_facts" in plan.blocked
    assert plan.judge_model_name == "verifier-model"
    assert "skipped" in plan.message()


class _ScorerFakeMlflow:
    def __init__(self):
        self.genai = SimpleNamespace(
            scorers=SimpleNamespace(scorer=self._scorer),
        )

    def _scorer(self, name=None):
        def wrap(function):
            return SimpleNamespace(name=name, function=function)

        return wrap


def test_activate_run_builds_the_scorer_when_logprobs_flow():
    plan = plan_run(
        ContinuousScoringConfig(enabled=True),
        default_judge_model="judge-model",
        shape=_shape(),
        judges_enabled=True,
    )
    active = activate_run(
        plan,
        settings=None,
        mlflow_module=_ScorerFakeMlflow(),
        model=FakeLogprobModel(lambda prompt: [("T", math.log(0.9))]),
    )
    assert not active.fallback
    assert len(active.scorers) == 1
    assert active.scorers[0].name == CONTINUOUS_SCORER_NAME
    value = active.scorers[0].function(
        inputs={"question": "q"},
        outputs="an answer",
        expectations={"expected_response": "the expected answer"},
    )
    assert value == pytest.approx(1.0)
    # Contract-unsatisfied rows skip with MLflow's empty-feedback convention.
    assert active.scorers[0].function(inputs={"question": "q"}, outputs="a") == []
    assert (
        active.scorers[0].function(
            inputs={"question": "q"},
            outputs=None,
            expectations={"expected_response": "e"},
        )
        == []
    )
    active.close()


def test_activate_run_falls_back_when_the_backend_has_no_logprobs(caplog):
    plan = plan_run(
        ContinuousScoringConfig(enabled=True),
        default_judge_model="judge-model",
        shape=_shape(),
        judges_enabled=True,
    )
    with caplog.at_level("WARNING"):
        active = activate_run(
            plan,
            settings=None,
            mlflow_module=_ScorerFakeMlflow(),
            model=FakeLogprobModel(supports_logprobs=False),
        )
    assert active.fallback
    assert active.scorers == []
    assert any("falls back to the discrete path" in w for w in active.warnings)
    assert any("no top logprobs" in record.message for record in caplog.records)
    metrics, warnings = active.finalize({})
    assert metrics == {"continuous/fallback": 1.0}
    assert warnings == active.warnings


def test_expected_facts_satisfy_the_scorer_contract():
    plan = plan_run(
        ContinuousScoringConfig(enabled=True),
        default_judge_model="judge-model",
        shape=_shape(expectation_keys=("expected_facts",)),
        judges_enabled=True,
    )
    active = activate_run(
        plan,
        settings=None,
        mlflow_module=_ScorerFakeMlflow(),
        model=FakeLogprobModel(lambda prompt: [("T", math.log(0.9))]),
    )
    value = active.scorers[0].function(
        inputs={"question": "q"},
        outputs="an answer",
        expectations={"expected_facts": ["fact one", "fact two"]},
    )
    assert value == pytest.approx(1.0)


def test_finalize_reports_tie_rates_for_both_instruments():
    plan = plan_run(
        ContinuousScoringConfig(enabled=True),
        default_judge_model="judge-model",
        shape=_shape(),
        judges_enabled=True,
    )
    active = activate_run(
        plan,
        settings=None,
        mlflow_module=_ScorerFakeMlflow(),
        model=FakeLogprobModel(lambda prompt: [("T", math.log(0.9))]),
    )
    metrics, _ = active.finalize(
        {
            f"{CONTINUOUS_SCORER_NAME}/mean": (0.91, 0.83, 0.77),
            "correctness/mean": (1.0, 1.0, 1.0),
        }
    )
    assert metrics[f"{CONTINUOUS_SCORER_NAME}/tie_rate"] == 0.0
    assert metrics["correctness/tie_rate"] == 1.0
    assert metrics["continuous/fallback"] == 0.0
    active.close()


def test_activate_run_wraps_an_unresolvable_verifier_model():
    from aai_core.testing import dev_settings

    plan = plan_run(
        ContinuousScoringConfig(enabled=True, judge_model="missing-verifier"),
        default_judge_model="judge-model",
        shape=_shape(),
        judges_enabled=True,
    )
    with pytest.raises(ConfigError) as excinfo:
        activate_run(
            plan,
            settings=dev_settings(),
            mlflow_module=_ScorerFakeMlflow(),
        )
    assert "missing-verifier" in str(excinfo.value)
    assert "aai-platform.yml" in str(excinfo.value.remediation)


def test_multi_field_inputs_reach_the_verifier_as_json():
    plan = plan_run(
        ContinuousScoringConfig(enabled=True),
        default_judge_model="judge-model",
        shape=_shape(),
        judges_enabled=True,
    )
    model = FakeLogprobModel(lambda prompt: [("T", math.log(0.9))])
    active = activate_run(
        plan,
        settings=None,
        mlflow_module=_ScorerFakeMlflow(),
        model=model,
    )
    active.scorers[0].function(
        inputs={"question": "q", "context": "c"},
        outputs="a",
        expectations={"expected_response": "e"},
    )
    # requests[0] is the capability probe; the first scoring call follows.
    prompt = model.requests[1]["messages"][-1]["content"]
    assert '"context": "c"' in prompt
    # A shapeless inputs object cannot be judged: the row is skipped.
    assert (
        active.scorers[0].function(
            inputs=42, outputs="a", expectations={"expected_response": "e"}
        )
        == []
    )


def test_finalize_warns_on_low_normalization_mass():
    plan = plan_run(
        ContinuousScoringConfig(enabled=True, low_mass_threshold=0.5),
        default_judge_model="judge-model",
        shape=_shape(),
        judges_enabled=True,
    )
    active = activate_run(
        plan,
        settings=None,
        mlflow_module=_ScorerFakeMlflow(),
        model=FakeLogprobModel(lambda prompt: [("T", math.log(0.2))]),
    )
    active.scorers[0].function(
        inputs={"question": "q"},
        outputs="a",
        expectations={"expected_response": "e"},
    )
    _, warnings = active.finalize({})
    assert any("not reliably steering" in warning for warning in warnings)


# --- runner integration ------------------------------------------------------

AGENT_SOURCE = """\
def respond(question):
    return "answer about pensions"
"""

PLATFORM_YAML = """\
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
"""

CONTINUOUS_CONFIG = """\
version: 1
agent: src/app/example_agent.py:respond
dataset: evals/data/golden_cases.json
scorers:
  continuous:
    enabled: true
    granularity: 5
    repeats: 2
"""


class _Frame:
    def __init__(self, data):
        self._data = dict(data)
        self.columns = list(self._data)

    def __getitem__(self, name):
        return self._data[name]

    def get(self, name):
        return self._data.get(name)


def _builtin_fake(class_name):
    class _Fake:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, **kwargs):
            return []

    _Fake.__name__ = class_name
    return _Fake


class ContinuousFakeMlflow:
    """The runner's MLflow surface, whose evaluate really invokes the
    continuous scorer so its telemetry and per-row samples are exercised."""

    def __init__(self):
        self.tags = {}
        self.params = {}
        self.logged_metrics = {}
        self.experiment = None
        self.run_artifacts = []
        self.evaluate_calls = []
        builtin_names = (
            "Correctness",
            "Equivalence",
            "RelevanceToQuery",
            "Safety",
            "Fluency",
            "Completeness",
            "ExpectationsGuidelines",
            "Guidelines",
            "RetrievalGroundedness",
            "RetrievalRelevance",
            "RetrievalSufficiency",
            "ToolCallCorrectness",
            "ToolCallEfficiency",
        )
        self.genai = SimpleNamespace(
            evaluate=self._evaluate,
            scorers=SimpleNamespace(
                scorer=self._scorer,
                **{name: _builtin_fake(name) for name in builtin_names},
            ),
            make_judge=SimpleNamespace,
        )

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None, nested=False, description=None):
        info = SimpleNamespace(run_id="run-1", experiment_id="42")

        class _Run:
            def __enter__(self):
                return SimpleNamespace(info=info)

            def __exit__(self, *args):
                return False

        return _Run()

    def set_tags(self, tags):
        self.tags.update(tags)

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics):
        self.logged_metrics.update(metrics)

    def log_artifact(self, path, artifact_path=None):
        pass

    def MlflowClient(self, *args, **kwargs):
        outer = self

        class _Client:
            def log_artifact(self, run_id, local_path, artifact_path=None):
                outer.run_artifacts.append((run_id, Path(local_path).name))

        return _Client()

    def _scorer(self, name=None):
        def wrap(function):
            return SimpleNamespace(name=name, function=function)

        return wrap

    def _evaluate(self, data=None, scorers=None, predict_fn=None):
        self.evaluate_calls.append(
            {"data": data, "scorers": scorers, "predict_fn": predict_fn}
        )
        columns = {"correctness/value": ["yes"] * len(data)}
        for scorer in scorers:
            function = getattr(scorer, "function", None)
            if function is None:
                continue
            # MLflow dispatches by signature: pass only the arguments the
            # scorer declares.
            accepted = set(inspect.signature(function).parameters)
            values = []
            for row in data:
                arguments = {
                    name: row.get(name)
                    for name in ("inputs", "outputs", "expectations")
                    if name in accepted
                }
                value = function(**arguments)
                values.append(None if value == [] else value)
            columns[f"{scorer.name}/value"] = values
        metrics = {
            "correctness/mean": 1.0,
            "safety/mean": 1.0,
            "relevance_to_query/mean": 1.0,
        }
        for name, values in columns.items():
            scored = [value for value in values if isinstance(value, float)]
            if scored:
                metrics[name.removesuffix("/value") + "/mean"] = sum(scored) / len(
                    scored
                )
        return SimpleNamespace(metrics=metrics, result_df=_Frame(columns))


def _project(tmp_path, config_text=CONTINUOUS_CONFIG, rows=12):
    (tmp_path / "src" / "app").mkdir(parents=True)
    (tmp_path / "src" / "app" / "example_agent.py").write_text(AGENT_SOURCE)
    (tmp_path / "aai-platform.yml").write_text(PLATFORM_YAML)
    data_dir = tmp_path / "evals" / "data"
    data_dir.mkdir(parents=True)
    cases = [
        {
            "inputs": {"question": f"question {index}"},
            "expectations": {"expected_response": f"answer {index} about pensions"},
        }
        for index in range(rows)
    ]
    (data_dir / "golden_cases.json").write_text(json.dumps(cases))
    (data_dir / "answer_sheet.json").write_text(
        json.dumps(
            [
                {
                    "question": f"question {index}",
                    "answer": f"answer {index} about pensions",
                }
                for index in range(rows)
            ]
        )
    )
    (tmp_path / "agentkit.yaml").write_text(config_text)
    return ProjectContext.load(tmp_path / "agentkit.yaml", environ={})


def test_config_parses_the_continuous_block(tmp_path):
    (tmp_path / "agentkit.yaml").write_text(CONTINUOUS_CONFIG)
    config = load_config(tmp_path / "agentkit.yaml", environ={})
    assert config.scorers.continuous.enabled
    assert config.scorers.continuous.granularity == 5
    assert config.scorers.continuous.repeats == 2
    assert config.scorers.continuous.judge_model is None


def test_run_scoring_records_continuous_beside_discrete(tmp_path):
    project = _project(tmp_path)
    fake = ContinuousFakeMlflow()

    # Distinct distributions per row untie the continuous scores while the
    # discrete correctness column stays all-"yes".
    def pairs_for(prompt):
        match = re.search(r"question (\d+)", prompt)
        index = int(match.group(1)) if match else 0
        label = ["B", "C", "D"][index % 3]
        return [(label, math.log(0.7)), ("A", math.log(0.2))]

    model = FakeLogprobModel(pairs_for)
    outcome, code = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=fake,
        continuous_model=model,
    )

    assert code == 0
    messages = "\n".join(outcome.messages)
    assert "Continuous scoring adds ~73 verifier call(s)" in messages
    metrics = outcome.results.metrics
    assert f"{CONTINUOUS_SCORER_NAME}/mean" in metrics
    assert metrics["continuous/fallback"] == 0.0
    assert metrics["continuous/judge_calls"] > 0
    assert metrics["correctness/tie_rate"] == 1.0
    assert metrics[f"{CONTINUOUS_SCORER_NAME}/tie_rate"] < 1.0
    assert fake.params["continuous_granularity"] == "5"
    assert fake.params["continuous_repeats"] == "2"
    assert fake.tags["aai.continuous_scoring"] == "logprob-weighted"
    assert fake.tags["aai.continuous_judge_model"] == "judge-model"
    # per-row samples persisted like any scorer's
    assert f"{CONTINUOUS_SCORER_NAME}/mean" in outcome.results.metric_samples


def test_run_scoring_falls_back_without_failing_the_gate(tmp_path):
    project = _project(tmp_path)
    fake = ContinuousFakeMlflow()
    outcome, code = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=fake,
        continuous_model=FakeLogprobModel(supports_logprobs=False),
    )

    assert code == 0
    assert outcome.results.metrics["continuous/fallback"] == 1.0
    assert f"{CONTINUOUS_SCORER_NAME}/mean" not in outcome.results.metrics
    assert any(
        "falls back to the discrete path" in warning
        for warning in outcome.results.warnings
    )
    assert fake.tags["aai.continuous_scoring"] == "fallback-discrete"


def test_budget_ceiling_covers_the_continuous_calls(tmp_path):
    config_text = CONTINUOUS_CONFIG + "budget:\n  max_judge_calls: 10\n"
    project = _project(tmp_path, config_text=config_text)
    fake = ContinuousFakeMlflow()
    with pytest.raises(BudgetExceededError) as excinfo:
        run_scoring(
            project,
            establish_baseline=True,
            judges_enabled=True,
            mode="answer-sheet",
            assume_yes=True,
            mlflow_module=fake,
            continuous_model=FakeLogprobModel(),
        )
    assert "continuous scoring" in str(excinfo.value)
    assert fake.evaluate_calls == []


def test_code_only_plan_with_continuous_still_runs_the_verifier(tmp_path):
    # With every discrete judge removed, an answer-sheet run would score
    # locally — and a local run never makes the verifier calls its own
    # estimate promised. An active continuous plan must force the MLflow
    # path so the instrument actually runs.
    config_text = (
        "version: 1\n"
        "agent: src/app/example_agent.py:respond\n"
        "dataset: evals/data/golden_cases.json\n"
        "scorers:\n"
        "  remove: [correctness, safety]\n"
        "  continuous:\n"
        "    enabled: true\n"
        "    granularity: 5\n"
    )
    project = _project(tmp_path, config_text=config_text)
    fake = ContinuousFakeMlflow()
    outcome, code = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=fake,
        continuous_model=FakeLogprobModel(lambda prompt: [("E", math.log(0.9))]),
    )
    assert code == 0
    assert f"{CONTINUOUS_SCORER_NAME}/mean" in outcome.results.metrics
    assert not any("scored locally" in warning for warning in outcome.results.warnings)


def test_smoke_stays_free_with_continuous_configured(tmp_path):
    project = _project(tmp_path)
    outcome, code = run_scoring(
        project,
        command="smoke",
        judges_enabled=False,
        require_baseline=False,
        mode="answer-sheet",
        assume_yes=True,
    )
    assert code == 0
    assert "Continuous scoring" not in "\n".join(outcome.messages)


def test_dataset_without_expectations_skips_with_the_reason(tmp_path):
    project = _project(tmp_path)
    cases = [{"inputs": {"question": f"question {index}"}} for index in range(12)]
    (tmp_path / "evals" / "data" / "golden_cases.json").write_text(json.dumps(cases))
    fake = ContinuousFakeMlflow()
    outcome, code = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=fake,
        continuous_model=FakeLogprobModel(),
    )
    assert code == 0
    messages = "\n".join(outcome.messages)
    assert "Continuous scoring is configured but skipped" in messages
    assert "continuous/fallback" not in outcome.results.metrics
