"""Unit tests for the scoring engine: governed runs, tags, baselines, gating.

The MLflow module is injected, so these run with zero cloud access. The
fake mirrors the surface the runner actually uses (see
tests/test_experiment_helpers.py for the same pattern).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_core.agentkit.baseline import load_baseline, write_baseline
from aai_core.agentkit.config import ProjectContext
from aai_core.agentkit.errors import (
    BaselineIncomparableError,
    BaselineMissingError,
    BudgetExceededError,
    ConfigError,
    EvidenceMissingError,
)
from aai_core.agentkit.gate import EXIT_ERROR, EXIT_PASS, EXIT_THRESHOLD_FAILED
from aai_core.agentkit.results import load_latest_results
from aai_core.agentkit.runner import (
    SCORER_WORKERS_ENV,
    WORKERS_ENV,
    Decision,
    run_scoring,
    set_concurrency_env,
    submit_job,
)

AGENT_SOURCE = """\
KNOWLEDGE = {
    "question 0": "answer zero about pensions",
    "question 1": "answer one about pensions",
}


def respond(question):
    return KNOWLEDGE.get(question, "I cannot help with that")
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


# What a judged run over the golden suite returns: the code scorers plus
# the judges the registry selects when expectations are present.
JUDGED_METRICS = {
    "keyword_coverage/mean": 0.9,
    "refusal_compliance/mean": 1.0,
    "response_length_ok/mean": 1.0,
    "correctness/mean": 0.9,
    "safety/mean": 1.0,
}


def _builtin_fake(class_name):
    """A stand-in for an MLflow builtin scorer *class*.

    A factory function would be simpler, but the toolkit subclasses these
    to skip rows a retrieval scorer cannot judge — and a fake that cannot
    be subclassed lets that path pass untested, which is how the wrapper
    reached review unexercised.
    """

    class _Fake:
        def __init__(self, **kwargs):
            self.class_name = class_name
            self.kwargs = kwargs

        def __call__(self, *, trace=None):
            return []

    _Fake.__name__ = class_name
    return _Fake


_BUILTIN_FAKES = {
    name: _builtin_fake(name)
    for name in (
        "Correctness",
        "Equivalence",
        "RelevanceToQuery",
        "Safety",
        "PIIDetection",
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
}


class FakeMlflow:
    """The MLflow surface the runner uses, recorded for assertions."""

    def __init__(self, metrics=None, run_id="run-1", experiment_id="42"):
        self.metrics_to_return = metrics or dict(JUDGED_METRICS)
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.experiment = None
        self.tags: dict = {}
        self.params: dict = {}
        self.logged_metrics: dict = {}
        self.artifacts: list = []
        self.run_artifacts: list = []
        self.evaluate_calls: list = []
        self.traced: list = []
        self.genai = SimpleNamespace(
            evaluate=self._evaluate,
            scorers=SimpleNamespace(scorer=self._scorer, **_BUILTIN_FAKES),
            make_judge=lambda **kwargs: SimpleNamespace(**kwargs),
        )

    # --- experiment plumbing -------------------------------------------
    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None, nested=False, description=None):
        info = SimpleNamespace(run_id=self.run_id, experiment_id=self.experiment_id)

        class _Run:
            def __enter__(inner):
                return SimpleNamespace(info=info)

            def __exit__(inner, *args):
                return False

        return _Run()

    def set_tags(self, tags):
        self.tags.update(tags)

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics):
        self.logged_metrics.update(metrics)

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((Path(path).name, artifact_path))

    def MlflowClient(self, *args, **kwargs):
        outer = self

        class _Client:
            def log_artifact(inner, run_id, local_path, artifact_path=None):
                outer.run_artifacts.append(
                    (run_id, Path(local_path).name, artifact_path)
                )

        return _Client()

    # --- genai plumbing -------------------------------------------------
    def trace(self, function):
        self.traced.append(function)
        return function

    def _scorer(self, name=None):
        def wrap(function):
            return SimpleNamespace(name=name, function=function)

        return wrap

    def _evaluate(self, data=None, scorers=None, predict_fn=None):
        self.evaluate_calls.append(
            {"data": data, "scorers": scorers, "predict_fn": predict_fn}
        )
        return SimpleNamespace(metrics=dict(self.metrics_to_return))


def _project(tmp_path, config_text=None, rows=12):
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
    (tmp_path / "agentkit.yaml").write_text(
        config_text
        or (
            "version: 1\n"
            "agent: src/app/example_agent.py:respond\n"
            "dataset: evals/data/golden_cases.json\n"
        )
    )
    return ProjectContext.load(tmp_path / "agentkit.yaml", environ={})


def test_establish_baseline_records_the_first_version(tmp_path):
    project = _project(tmp_path)
    fake = FakeMlflow()

    outcome, code = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=fake,
    )

    assert code == EXIT_PASS
    assert outcome.established_baseline
    assert "This run IS the baseline" in "\n".join(outcome.messages)

    record, warnings = load_baseline(project.baseline_path)
    assert warnings == []
    assert record.run_id == "run-1"
    assert record.dataset.digest == outcome.dataset.digest
    assert record.scope.mode == "full"
    assert dict(record.versions.scorers) == {
        "correctness": 1,
        "keyword_coverage": 1,
        "refusal_compliance": 1,
        "response_length_ok": 1,
        "safety": 1,
    }
    assert record.recorded_by == "agentkit compare --establish-baseline"


def test_governed_run_carries_the_full_tag_map(tmp_path):
    project = _project(tmp_path)
    fake = FakeMlflow()

    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=fake,
    )

    assert fake.experiment == "/Shared/pension-ai-agent-quality-quality-eval"
    # governed resource tags from ResourceContext
    assert fake.tags["aai.application"] == "quality-eval"
    assert fake.tags["aai.cost_center"] == "CC-9999"
    # lineage from ExperimentRunMetadata
    assert fake.tags["aai.run_purpose"] == "baseline"
    assert fake.tags["aai.change_summary"]
    # agentkit evidence
    assert fake.tags["aai.dataset"] == "evals/data/golden_cases.json"
    assert len(fake.tags["aai.dataset_digest"]) == 16
    assert fake.tags["aai.dataset_rows"] == "12"
    assert fake.tags["aai.agent_target"] == "src/app/example_agent.py:respond"
    assert fake.tags["aai.scorer_versions"] == (
        "correctness=1,keyword_coverage=1,refusal_compliance=1,"
        "response_length_ok=1,safety=1"
    )
    assert fake.tags["aai.gate_passed"] == "true"
    assert fake.tags["aai.decision"] == "inconclusive"
    assert fake.tags["aai.agentkit_version"]
    assert fake.tags["aai.judge_model"] == "endpoints:/judge-endpoint"


def test_compare_without_a_baseline_refuses(tmp_path):
    project = _project(tmp_path)

    with pytest.raises(BaselineMissingError) as excinfo:
        run_scoring(
            project,
            judges_enabled=False,
            mode="answer-sheet",
            assume_yes=True,
            mlflow_module=FakeMlflow(),
        )
    assert "--establish-baseline" in str(excinfo.value)


def test_compare_against_baseline_produces_a_diff(tmp_path):
    def metrics(coverage):
        return {**JUDGED_METRICS, "keyword_coverage/mean": coverage}

    project = _project(tmp_path)
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(metrics=metrics(0.6)),
    )

    outcome, code = run_scoring(
        project,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(metrics=metrics(0.9)),
    )

    assert code == EXIT_PASS
    row = next(r for r in outcome.comparison if r.metric == "keyword_coverage/mean")
    assert row.current == 0.9
    assert row.baseline == 0.6
    assert row.delta == pytest.approx(0.3)
    assert row.verdict == "pass"
    assert outcome.results.baseline_run_id == "run-1"
    assert outcome.results.established_baseline is False


def test_threshold_failure_exits_two(tmp_path):
    project = _project(tmp_path)
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )

    outcome, code = run_scoring(
        project,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(metrics={"keyword_coverage/mean": 0.1}),
    )

    assert code == EXIT_THRESHOLD_FAILED
    assert outcome.results.gate_passed is False
    assert "gate: FAILED" in "\n".join(outcome.messages)


def test_answer_sheet_mode_scores_recorded_outputs_without_a_predict_fn(tmp_path):
    project = _project(tmp_path)
    fake = FakeMlflow()

    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=fake,
    )

    call = fake.evaluate_calls[0]
    assert call["predict_fn"] is None
    assert all("outputs" in row for row in call["data"])
    assert call["data"][0]["outputs"] == "answer 0 about pensions"


def test_live_mode_passes_a_traced_predict_fn(tmp_path):
    project = _project(tmp_path)
    fake = FakeMlflow()

    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=False,
        mode="live",
        assume_yes=True,
        mlflow_module=fake,
    )

    call = fake.evaluate_calls[0]
    assert call["predict_fn"] is not None
    assert fake.traced, "the agent call must be traced exactly once"
    assert call["predict_fn"](question="question 0") == "answer zero about pensions"


def test_judged_run_records_the_governed_judge_endpoint(tmp_path):
    project = _project(tmp_path)
    fake = FakeMlflow(
        metrics={
            "keyword_coverage/mean": 0.9,
            "refusal_compliance/mean": 1.0,
            "response_length_ok/mean": 1.0,
            "correctness/mean": 0.9,
            "safety/mean": 1.0,
        }
    )

    outcome, code = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=fake,
    )

    assert code == EXIT_PASS
    assert fake.tags["aai.judge_model"] == "endpoints:/judge-endpoint"
    assert "correctness" in fake.tags["aai.scorer_versions"]
    assert outcome.cost.judge_calls > 0


def test_declined_confirmation_scores_nothing(tmp_path, monkeypatch):
    """A cancelled run is not a passed run.

    The usual cause is a CI job on a non-interactive stream with no
    `--yes`; exit 0 there would report success for an evaluation that
    never happened.
    """

    project = _project(tmp_path)
    identity_calls = []
    monkeypatch.setattr(
        project,
        "judge_model_identity",
        lambda: identity_calls.append("identity") or None,
    )
    fake = FakeMlflow()
    environ = {}

    outcome, code = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=False,
        confirm=lambda prompt: False,
        environ=environ,
        mlflow_module=fake,
    )

    assert code == EXIT_ERROR
    assert outcome.declined
    assert identity_calls == ["identity"]
    assert environ == {}
    assert fake.evaluate_calls == []
    assert not project.baseline_path.exists()
    assert any("--yes" in message for message in outcome.messages)


def test_budget_stops_the_run_before_any_call(tmp_path, monkeypatch):
    project = _project(
        tmp_path,
        config_text=(
            "version: 1\n"
            "agent: src/app/example_agent.py:respond\n"
            "dataset: evals/data/golden_cases.json\n"
            "budget:\n  max_judge_calls: 3\n"
        ),
    )
    monkeypatch.setattr(
        project,
        "judge_model_identity",
        lambda: pytest.fail("endpoint identity was resolved"),
    )
    fake = FakeMlflow()

    with pytest.raises(BudgetExceededError):
        run_scoring(
            project,
            establish_baseline=True,
            judges_enabled=True,
            mode="answer-sheet",
            assume_yes=True,
            mlflow_module=fake,
        )
    assert fake.evaluate_calls == []


def test_plan_only_prints_without_scoring(tmp_path):
    project = _project(tmp_path)
    fake = FakeMlflow()

    outcome, code = run_scoring(
        project,
        judges_enabled=False,
        mode="answer-sheet",
        plan_only=True,
        mlflow_module=fake,
    )

    assert code == EXIT_PASS
    assert outcome.plan_only
    assert fake.evaluate_calls == []
    text = "\n".join(outcome.messages)
    assert "Inferred evaluation plan" in text
    assert "0 judge calls" in text


def test_a_judged_plan_never_resolves_the_endpoint_identity(tmp_path, monkeypatch):
    project = _project(tmp_path)
    monkeypatch.setattr(
        project,
        "judge_model_identity",
        lambda: pytest.fail("endpoint identity was resolved"),
    )

    outcome, code = run_scoring(
        project,
        judges_enabled=True,
        mode="answer-sheet",
        plan_only=True,
        mlflow_module=FakeMlflow(),
    )

    assert code == EXIT_PASS
    assert outcome.plan_only
    assert outcome.cost.judge_calls > 0


def test_row_limit_samples_deterministically_and_records_scope(tmp_path):
    project = _project(tmp_path, rows=30)
    fake = FakeMlflow()

    outcome, _ = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        rows_limit=10,
        assume_yes=True,
        mlflow_module=fake,
    )

    assert outcome.results.scope.mode == "sample"
    assert outcome.results.scope.rows == 10
    assert len(fake.evaluate_calls[0]["data"]) == 10


def test_results_record_is_written_and_reloadable(tmp_path):
    project = _project(tmp_path)

    outcome, _ = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )

    assert outcome.results_path.parent == project.results_dir
    found = load_latest_results(project.results_dir)
    assert found is not None
    reloaded, path = found
    assert path == outcome.results_path
    assert reloaded.command == "compare"
    assert reloaded.is_comparison


def test_explicit_decision_is_recorded(tmp_path):
    project = _project(tmp_path)
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )
    fake = FakeMlflow()

    outcome, _ = run_scoring(
        project,
        judges_enabled=True,
        mode="answer-sheet",
        decision=Decision.ADOPT.value,
        assume_yes=True,
        mlflow_module=fake,
    )

    assert outcome.results.decision == "adopt"
    assert fake.tags["aai.decision"] == "adopt"


def _edit_an_expectation(project):
    path = project.root / "evals" / "data" / "golden_cases.json"
    cases = json.loads(path.read_text())
    cases[0]["expectations"]["expected_response"] = "a different expectation"
    path.write_text(json.dumps(cases))


def test_a_changed_dataset_refuses_the_comparison(tmp_path):
    """A delta measured on different rows is not evidence of anything."""

    project = _project(tmp_path)
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )
    _edit_an_expectation(project)
    mlflow = FakeMlflow()

    with pytest.raises(BaselineIncomparableError) as excinfo:
        run_scoring(
            project,
            judges_enabled=True,
            mode="answer-sheet",
            assume_yes=True,
            mlflow_module=mlflow,
        )

    assert "the dataset changed" in str(excinfo.value)
    assert "--establish-baseline" in str(excinfo.value)
    # Refused before the run opened, so nothing was scored or spent.
    assert mlflow.evaluate_calls == []


def test_baseline_refusal_precedes_confirmation_and_spend(tmp_path):
    """Never ask approval for a comparison that cannot produce a valid delta."""

    project = _project(tmp_path)
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )
    _edit_an_expectation(project)
    asked = []
    mlflow = FakeMlflow()

    with pytest.raises(BaselineIncomparableError):
        run_scoring(
            project,
            judges_enabled=True,
            mode="answer-sheet",
            assume_yes=False,
            confirm=lambda prompt: asked.append(prompt) or True,
            mlflow_module=mlflow,
        )

    assert asked == []
    assert mlflow.evaluate_calls == []


def test_all_comparability_follows_budget_and_precedes_confirmation_and_spend(
    tmp_path, monkeypatch
):
    from aai_core.agentkit import runner as runner_module

    project = _with_prompt_judge(tmp_path)
    _use_registered_prompt(project, monkeypatch)
    monkeypatch.setattr(
        project,
        "judge_model_identity",
        lambda: "main.models.judge/3",
    )
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )

    events = []
    enforce_budget = runner_module.enforce_budget

    def record_budget(*args, **kwargs):
        events.append("budget")
        return enforce_budget(*args, **kwargs)

    monkeypatch.setattr(runner_module, "enforce_budget", record_budget)
    monkeypatch.setattr(
        project,
        "judge_model_identity",
        lambda: events.append("identity") or "main.models.judge/3",
    )
    enforce_comparability = runner_module._enforce_comparability

    def record_comparability(*args, **kwargs):
        events.append(
            "prompt-comparability" if kwargs.get("only_prompts") else "comparability"
        )
        return enforce_comparability(*args, **kwargs)

    monkeypatch.setattr(
        runner_module,
        "_enforce_comparability",
        record_comparability,
    )
    mlflow = FakeMlflow(run_id="run-2")
    evaluate = mlflow.genai.evaluate

    def record_evaluate(*args, **kwargs):
        events.append("evaluate")
        return evaluate(*args, **kwargs)

    mlflow.genai.evaluate = record_evaluate

    run_scoring(
        project,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=False,
        confirm=lambda prompt: events.append("confirm") or True,
        mlflow_module=mlflow,
    )

    assert events[0:5] == [
        "budget",
        "identity",
        "comparability",
        "prompt-comparability",
        "confirm",
    ]
    assert events.index("identity") < events.index("comparability")
    assert events.index("comparability") < events.index("prompt-comparability")
    assert events.index("prompt-comparability") < events.index("confirm")
    assert events.index("confirm") < events.index("evaluate")


def test_allow_baseline_drift_proceeds_and_records_the_reason(tmp_path):
    project = _project(tmp_path)
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )
    _edit_an_expectation(project)

    outcome, _ = run_scoring(
        project,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        allow_baseline_drift=True,
        mlflow_module=FakeMlflow(),
    )

    assert any("--allow-baseline-drift" in w for w in outcome.warnings)
    assert any("the dataset changed" in w for w in outcome.warnings)
    # The override travels with the record, so the evidence shows it.
    assert any("the dataset changed" in w for w in outcome.results.warnings)


def test_a_changed_scorer_version_refuses_the_comparison(tmp_path):
    """Two scores from different scorer versions are not comparable."""

    project = _project(tmp_path)
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )
    baseline, _ = load_baseline(project.baseline_path)
    bumped = baseline.model_copy(
        update={
            "versions": baseline.versions.model_copy(
                update={
                    "scorers": {
                        **dict(baseline.versions.scorers),
                        "keyword_coverage": 99,
                    }
                }
            )
        }
    )
    write_baseline(project.baseline_path, bumped)

    with pytest.raises(BaselineIncomparableError) as excinfo:
        run_scoring(
            project,
            judges_enabled=True,
            mode="answer-sheet",
            assume_yes=True,
            mlflow_module=FakeMlflow(),
        )

    assert "keyword_coverage" in str(excinfo.value)


def test_concurrency_env_is_set_without_clobbering_an_explicit_value():
    environ = {}
    set_concurrency_env(12, environ)
    assert environ[WORKERS_ENV] == "12"
    assert environ[SCORER_WORKERS_ENV] == "4"

    explicit = {WORKERS_ENV: "2"}
    set_concurrency_env(12, explicit)
    assert explicit[WORKERS_ENV] == "2"


def test_submit_job_runs_the_bundle_release_gate(tmp_path):
    project = _project(tmp_path)
    commands = []

    def runner(command, cwd=None, check=False):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    code, messages = submit_job(project, target="dev", runner=runner)

    assert code == EXIT_PASS
    assert commands[0][:3] == ["databricks", "bundle", "validate"]
    assert commands[1][:4] == ["databricks", "bundle", "run", "release_gate"]
    assert any("release_gate" in message for message in messages)


def test_submit_job_propagates_failure(tmp_path):
    project = _project(tmp_path)

    def runner(command, cwd=None, check=False):
        return SimpleNamespace(returncode=2)

    code, messages = submit_job(project, runner=runner)

    assert code == 2
    assert any("exit code 2" in message for message in messages)


def test_code_only_run_scores_locally_and_opens_no_mlflow_run(tmp_path):
    """Smoke is a fast gate, not a recorded experiment.

    A code-scorer-only run over recorded answers needs nothing from MLflow,
    so it must not contact a tracking backend — that is what keeps it
    runnable on every commit in credential-free CI, and what keeps an
    afternoon of throwaway runs out of the experiment.
    """

    project = _project(tmp_path)
    fake = FakeMlflow()

    outcome, code = run_scoring(
        project,
        command="smoke",
        judges_enabled=False,
        require_baseline=False,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=fake,
    )

    assert code == EXIT_PASS
    assert fake.evaluate_calls == []
    assert fake.tags == {}
    assert fake.experiment is None
    assert outcome.results.run_id is None
    assert outcome.results.experiment_name is None
    assert any("scored locally" in warning for warning in outcome.warnings)
    # The deterministic scorers still produced real metrics.
    assert outcome.results.metrics["keyword_coverage/mean"] == pytest.approx(1.0)
    assert outcome.results.metrics["response_length_ok/mean"] == 1.0


def test_local_scoring_still_gates_and_can_fail(tmp_path):
    project = _project(tmp_path)
    sheet = tmp_path / "evals" / "data" / "answer_sheet.json"
    sheet.write_text(
        json.dumps(
            [
                {"question": f"question {index}", "answer": "unrelated"}
                for index in range(12)
            ]
        )
    )

    outcome, code = run_scoring(
        project,
        command="smoke",
        judges_enabled=False,
        require_baseline=False,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )

    assert code == EXIT_THRESHOLD_FAILED
    assert not outcome.results.gate_passed


def test_agent_override_scores_the_named_target(tmp_path):
    """The deployment gate must score the version that triggered it."""

    project = _project(tmp_path)
    fake = FakeMlflow()

    outcome, _ = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        agent="models:/main.evaluation.agent/7",
        assume_yes=True,
        mlflow_module=fake,
    )

    assert outcome.results.agent == "models:/main.evaluation.agent/7"
    assert fake.tags["aai.agent_target"] == "models:/main.evaluation.agent/7"


TRACE_ROWS = [
    {
        "inputs": {"question": f"question {index}"},
        "expectations": {"expected_response": f"answer {index} about pensions"},
        "trace": {
            "data": {
                "spans": [
                    {
                        "type": "RETRIEVER",
                        "name": "search",
                        "outputs": [{"page_content": "policy text"}],
                    }
                ]
            }
        },
    }
    for index in range(12)
]


def _trace_project(tmp_path):
    _project(tmp_path)
    (tmp_path / "evals" / "data" / "golden_cases.json").write_text(
        json.dumps(TRACE_ROWS)
    )
    return ProjectContext.load(tmp_path / "agentkit.yaml", environ={})


def test_trace_backed_dataset_scores_its_own_traces(tmp_path):
    """The recorded traces are the thing under evaluation.

    MLflow replaces a row's trace when `predict_fn` is supplied, so calling
    the agent here would score freshly generated behaviour while reporting
    it against a dataset of production traces.
    """

    project = _trace_project(tmp_path)
    mlflow = FakeMlflow(metrics={**JUDGED_METRICS, "retrieval_groundedness/mean": 0.9})

    outcome, code = run_scoring(
        project,
        command="compare",
        establish_baseline=True,
        assume_yes=True,
        mlflow_module=mlflow,
        environ={},
    )

    assert code == EXIT_PASS
    assert outcome.results.mode == "traces"
    call = mlflow.evaluate_calls[0]
    assert call["predict_fn"] is None
    assert all("trace" in row for row in call["data"])
    assert "retrieval_groundedness" in {
        entry.spec.name for entry in outcome.plan.entries
    }


def test_explicit_live_mode_on_trace_rows_warns_that_they_are_replaced(tmp_path):
    project = _trace_project(tmp_path)
    mlflow = FakeMlflow()

    outcome, _ = run_scoring(
        project,
        command="compare",
        mode="live",
        establish_baseline=True,
        assume_yes=True,
        mlflow_module=mlflow,
        environ={},
    )

    assert mlflow.evaluate_calls[0]["predict_fn"] is not None
    assert any(
        "recorded traces are not what gets scored" in warning
        for warning in outcome.warnings
    )


def test_recorded_run_attaches_its_results_record(tmp_path):
    """The approver reads evidence from the run, not from a job cluster."""

    project = _project(tmp_path)
    mlflow = FakeMlflow()

    outcome, _ = run_scoring(
        project,
        command="compare",
        establish_baseline=True,
        assume_yes=True,
        mlflow_module=mlflow,
        environ={},
    )

    assert mlflow.run_artifacts == [("run-1", outcome.results_path.name, "agentkit")]
    assert any(
        "agentkit evidence --run run-1" in message for message in outcome.messages
    )


def test_unpublishable_evidence_fails_the_run(tmp_path):
    """A run whose evidence cannot be reached is not promotion evidence.

    The deployment-job gate scores on an ephemeral job cluster: the run is
    the only durable copy of the results record, and the approval task
    depends on this task succeeding. Warning here would let a human be
    asked to approve evidence that `agentkit evidence --run` cannot find.
    """

    project = _project(tmp_path)
    mlflow = FakeMlflow()

    def _broken(*args, **kwargs):
        raise RuntimeError("tracking store unavailable")

    mlflow.MlflowClient = _broken

    with pytest.raises(EvidenceMissingError) as excinfo:
        run_scoring(
            project,
            command="compare",
            establish_baseline=True,
            assume_yes=True,
            mlflow_module=mlflow,
            environ={},
        )

    message = str(excinfo.value)
    assert "could not attach the results record" in message
    # The verdict is not hidden by the failure that follows it.
    assert "gate passed" in message


def test_local_scoring_needs_no_publication(tmp_path):
    """Smoke opens no run, so there is nothing to publish and no failure."""

    project = _project(tmp_path)
    mlflow = FakeMlflow()
    mlflow.MlflowClient = _unreachable_client

    outcome, code = run_scoring(
        project,
        command="smoke",
        judges_enabled=False,
        require_baseline=False,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=mlflow,
    )

    assert code == EXIT_PASS
    assert outcome.results.run_id is None


def _unreachable_client(*args, **kwargs):
    raise RuntimeError("tracking store unavailable")


def test_partial_traces_do_not_select_trace_mode(tmp_path):
    """Mixed rows cannot be scored as traces, and the run says so."""

    _project(tmp_path)
    mixed = list(TRACE_ROWS[:6]) + [
        {
            "inputs": {"question": f"question {index}"},
            "expectations": {"expected_response": f"answer {index} about pensions"},
        }
        for index in range(6, 12)
    ]
    (tmp_path / "evals" / "data" / "golden_cases.json").write_text(json.dumps(mixed))
    project = ProjectContext.load(tmp_path / "agentkit.yaml", environ={})
    mlflow = FakeMlflow()

    outcome, _ = run_scoring(
        project,
        command="compare",
        establish_baseline=True,
        assume_yes=True,
        mlflow_module=mlflow,
        environ={},
    )

    assert outcome.results.mode == "live"
    assert mlflow.evaluate_calls[0]["predict_fn"] is not None
    assert any("only some rows carry a trace" in w for w in outcome.warnings)


def test_the_run_records_the_rules_it_was_judged_by(tmp_path):
    project = _project(tmp_path)
    mlflow = FakeMlflow()

    outcome, _ = run_scoring(
        project,
        command="compare",
        establish_baseline=True,
        assume_yes=True,
        mlflow_module=mlflow,
        environ={},
    )

    metrics = {rule.metric for rule in outcome.results.policy_rules}
    assert "keyword_coverage/mean" in metrics
    assert "correctness/mean" in metrics
    assert outcome.results.allow_missing_regression_baseline is True


def test_the_run_records_the_baseline_it_compared_against(tmp_path):
    project = _project(tmp_path)
    run_scoring(
        project,
        command="compare",
        establish_baseline=True,
        assume_yes=True,
        mlflow_module=FakeMlflow(),
        environ={},
    )

    outcome, _ = run_scoring(
        project,
        command="compare",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
        environ={},
    )

    baseline, _ = load_baseline(project.baseline_path)
    assert outcome.results.baseline_recorded_at == baseline.recorded_at
    assert outcome.results.baseline_dataset_digest == baseline.dataset.digest


class _Frame:
    """The pandas surface the error-count reader touches: columns + getitem."""

    def __init__(self, columns):
        self._columns = dict(columns)

    @property
    def columns(self):
        return list(self._columns)

    def __getitem__(self, key):
        return self._columns[key]


def test_partial_scorer_failures_fail_the_gate(tmp_path):
    """A judge that raised on most rows must not pass on the survivors.

    MLflow reports the exception in `result_df` as `<scorer>/error_message`
    and leaves it out of `metrics` entirely, so the aggregate is computed
    over the rows that happened to work.
    """

    project = _project(tmp_path)
    mlflow = FakeMlflow()
    frame = _Frame(
        {
            "correctness/value": ["yes", None, None],
            "correctness/error_message": [None, "endpoint 429", "endpoint 429"],
            "keyword_coverage/error_message": [None, None, None],
        }
    )
    mlflow._evaluate = lambda data=None, scorers=None, predict_fn=None: (
        mlflow.evaluate_calls.append({"data": data, "predict_fn": predict_fn})
        or SimpleNamespace(metrics=dict(JUDGED_METRICS), result_df=frame)
    )
    mlflow.genai = SimpleNamespace(
        evaluate=mlflow._evaluate,
        scorers=mlflow.genai.scorers,
        make_judge=mlflow.genai.make_judge,
    )

    outcome, code = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=mlflow,
    )

    assert code == EXIT_THRESHOLD_FAILED
    assert outcome.results.metrics["correctness/error_count"] == 2.0
    assert outcome.results.metrics["keyword_coverage/error_count"] == 0.0
    assert any(
        failure["metric"] == "correctness/error_count"
        for failure in outcome.results.gate_failures
    )


def test_a_clean_result_table_adds_no_failures(tmp_path):
    project = _project(tmp_path)
    mlflow = FakeMlflow()
    frame = _Frame({"correctness/error_message": [None, float("nan"), ""]})
    mlflow._evaluate = lambda data=None, scorers=None, predict_fn=None: (
        SimpleNamespace(metrics=dict(JUDGED_METRICS), result_df=frame)
    )
    mlflow.genai = SimpleNamespace(
        evaluate=mlflow._evaluate,
        scorers=mlflow.genai.scorers,
        make_judge=mlflow.genai.make_judge,
    )

    outcome, code = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=mlflow,
    )

    assert code == EXIT_PASS
    assert outcome.results.metrics["correctness/error_count"] == 0.0


def test_explicit_traces_mode_on_partial_coverage_is_refused(tmp_path):
    from aai_core.agentkit.errors import ConfigError

    _project(tmp_path)
    mixed = list(TRACE_ROWS[:6]) + [
        {
            "inputs": {"question": f"question {index}"},
            "expectations": {"expected_response": f"answer {index} about pensions"},
        }
        for index in range(6, 12)
    ]
    (tmp_path / "evals" / "data" / "golden_cases.json").write_text(json.dumps(mixed))
    project = ProjectContext.load(tmp_path / "agentkit.yaml", environ={})
    mlflow = FakeMlflow()

    with pytest.raises(ConfigError) as excinfo:
        run_scoring(
            project,
            mode="traces",
            establish_baseline=True,
            assume_yes=True,
            mlflow_module=mlflow,
        )

    assert "only some rows carry one" in str(excinfo.value)
    assert mlflow.evaluate_calls == []


def test_a_sampled_run_records_its_scope_for_a_run_baseline(tmp_path):
    project = _project(tmp_path)
    mlflow = FakeMlflow()

    run_scoring(
        project,
        command="smoke",
        rows_limit=6,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=mlflow,
    )

    assert mlflow.tags["aai.scope_mode"] == "sample"
    assert mlflow.tags["aai.scope_rows"] == "6"


def test_a_full_run_records_a_full_scope(tmp_path):
    project = _project(tmp_path)
    mlflow = FakeMlflow()

    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=mlflow,
    )

    assert mlflow.tags["aai.scope_mode"] == "full"


def _with_prompt_judge(tmp_path):
    """A project whose plan includes the prompt-judge scorer."""

    _project(tmp_path)
    (tmp_path / "agentkit.yaml").write_text(
        "version: 1\n"
        "agent: src/app/example_agent.py:respond\n"
        "dataset: evals/data/golden_cases.json\n"
        "scorers:\n"
        "  add: [pension_domain_policy]\n"
    )
    return ProjectContext.load(tmp_path / "agentkit.yaml", environ={})


class _PromptRegistryError(RuntimeError):
    def __init__(self, message, *, error_code=""):
        super().__init__(message)
        self.error_code = error_code


class _RecordingPromptManager:
    def __init__(self, *, prompt=None, error=None):
        self.prompt = prompt
        self.error = error
        self.calls = []

    def load(self, name, *, alias):
        self.calls.append((name, alias))
        if self.error is not None:
            raise self.error
        return self.prompt

    def qualify(self, name):
        return f"cat.sch.{name}"


def _prompt_lookup_failures():
    return (
        pytest.param(
            _PromptRegistryError(
                "prompt does not exist", error_code="PERMISSION_DENIED"
            ),
            id="permission",
        ),
        pytest.param(
            _PromptRegistryError("token expired", error_code="UNAUTHENTICATED"),
            id="unauthenticated",
        ),
        pytest.param(
            _PromptRegistryError("slow down", error_code="REQUEST_LIMIT_EXCEEDED"),
            id="rate-limit",
        ),
        pytest.param(
            _PromptRegistryError(
                "service unavailable", error_code="TEMPORARILY_UNAVAILABLE"
            ),
            id="transient",
        ),
        pytest.param(
            _PromptRegistryError(
                "prompt does not exist", error_code="CUSTOMER_UNAUTHORIZED"
            ),
            id="customer-unauthorized",
        ),
        pytest.param(
            _PromptRegistryError(
                "Prompt alias production not found.",
                error_code="CUSTOMER_UNAUTHORIZED",
            ),
            id="translated-alias-auth",
        ),
        pytest.param(
            _PromptRegistryError(
                "prompt does not exist", error_code="RESOURCE_EXHAUSTED"
            ),
            id="resource-exhausted",
        ),
        pytest.param(
            _PromptRegistryError(
                "prompt does not exist", error_code="DEADLINE_EXCEEDED"
            ),
            id="deadline-exceeded",
        ),
        pytest.param(
            _PromptRegistryError("prompt does not exist", error_code="INTERNAL_ERROR"),
            id="internal-error",
        ),
        pytest.param(
            _PromptRegistryError(
                "prompt does not exist", error_code="INVALID_PARAMETER_VALUE"
            ),
            id="invalid-parameter-not-alias",
        ),
        pytest.param(
            _PromptRegistryError(
                "Prompt alias production was not found.",
                error_code="INVALID_PARAMETER_VALUE",
            ),
            id="invalid-parameter-alias-near-miss",
        ),
        pytest.param(ConnectionError("connection reset"), id="transport"),
    )


def _prompt_absence_errors():
    return (
        pytest.param(
            _PromptRegistryError("hidden", error_code="NOT_FOUND"),
            id="structured-not-found",
        ),
        pytest.param(RuntimeError("prompt does not exist"), id="code-less-marker"),
        pytest.param(
            _PromptRegistryError("prompt does not exist", error_code=None),
            id="null-code-marker",
        ),
        pytest.param(
            _PromptRegistryError(
                "Registered model alias production not found.",
                error_code="INVALID_PARAMETER_VALUE",
            ),
            id="mlflow-pre-translation-alias",
        ),
        pytest.param(
            _PromptRegistryError(
                "Prompt alias production not found.",
                error_code="INVALID_PARAMETER_VALUE",
            ),
            id="mlflow-translated-alias",
        ),
        pytest.param(
            _PromptRegistryError(
                "INVALID_PARAMETER_VALUE: Prompt alias production not found.",
                error_code="INVALID_PARAMETER_VALUE",
            ),
            id="mlflow-rest-translated-alias",
        ),
    )


def _use_registered_prompt(project, monkeypatch):
    manager = _RecordingPromptManager(
        prompt=SimpleNamespace(
            uri="prompts:/cat.sch.agentkit_judge_domain_policy/3",
            template="registered domain policy",
        )
    )
    monkeypatch.setattr(
        project,
        "prompt_manager",
        lambda mlflow_module=None: manager,
    )
    return manager


def test_pinned_mlflow_translator_shapes_are_missing_aliases():
    pytest.importorskip("mlflow")
    from mlflow.exceptions import MlflowException, RestException
    from mlflow.prompt.registry_utils import translate_prompt_exception

    from aai_core.prompts import is_missing_prompt_error

    errors = (
        (
            MlflowException.invalid_parameter_value(
                "Registered model alias production not found."
            ),
            "Prompt alias production not found.",
        ),
        (
            RestException(
                {
                    "error_code": "INVALID_PARAMETER_VALUE",
                    "message": "Registered model alias production not found.",
                }
            ),
            "INVALID_PARAMETER_VALUE: Prompt alias production not found.",
        ),
    )
    for error, translated_message in errors:

        @translate_prompt_exception
        def load_prompt(error=error):
            raise error

        with pytest.raises(MlflowException) as excinfo:
            load_prompt()

        assert str(excinfo.value) == translated_message
        assert excinfo.value.error_code == "INVALID_PARAMETER_VALUE"
        assert is_missing_prompt_error(excinfo.value)


@pytest.mark.parametrize("missing", _prompt_absence_errors())
def test_a_missing_judge_prompt_uses_one_cached_bundled_fallback(
    tmp_path, monkeypatch, missing
):
    project = _with_prompt_judge(tmp_path)
    manager = _RecordingPromptManager(error=missing)
    monkeypatch.setattr(
        project,
        "prompt_manager",
        lambda mlflow_module=None: manager,
    )
    mlflow = FakeMlflow()

    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=mlflow,
    )

    assert manager.calls == [("agentkit_judge_domain_policy", "production")]
    scorer = next(
        scorer
        for scorer in mlflow.evaluate_calls[0]["scorers"]
        if getattr(scorer, "name", None) == "pension_domain_policy"
    )
    assert "Never disclose personal contact information" in scorer.instructions
    assert "aai.judge_prompt_versions" not in mlflow.tags
    baseline, _ = load_baseline(project.baseline_path)
    assert dict(baseline.versions.judge_prompts) == {}


@pytest.mark.parametrize("error", _prompt_lookup_failures())
def test_prompt_lookup_failures_refuse_baseline_before_scoring_or_evidence(
    tmp_path, monkeypatch, error
):
    from aai_core.agentkit import catalog as catalog_module

    project = _with_prompt_judge(tmp_path)
    manager = _RecordingPromptManager(error=error)
    monkeypatch.setattr(
        project,
        "prompt_manager",
        lambda mlflow_module=None: manager,
    )
    monkeypatch.setattr(
        catalog_module,
        "build_scorer",
        lambda *args, **kwargs: pytest.fail("a scorer was constructed"),
    )
    mlflow = FakeMlflow()

    with pytest.raises(type(error)) as excinfo:
        run_scoring(
            project,
            establish_baseline=True,
            judges_enabled=True,
            mode="answer-sheet",
            assume_yes=True,
            mlflow_module=mlflow,
        )

    assert excinfo.value is error
    assert manager.calls == [("agentkit_judge_domain_policy", "production")]
    assert mlflow.evaluate_calls == []
    assert mlflow.experiment is None
    assert mlflow.tags == {}
    assert mlflow.run_artifacts == []
    assert not project.results_dir.exists()
    assert not project.baseline_path.exists()


@pytest.mark.parametrize("error", _prompt_lookup_failures())
def test_prompt_lookup_failures_refuse_comparison_without_new_evidence(
    tmp_path, monkeypatch, error
):
    from aai_core.agentkit import catalog as catalog_module

    project = _with_prompt_judge(tmp_path)
    registered = _RecordingPromptManager(
        prompt=SimpleNamespace(
            uri="prompts:/cat.sch.agentkit_judge_domain_policy/3",
            template="registered domain policy",
        )
    )
    monkeypatch.setattr(
        project,
        "prompt_manager",
        lambda mlflow_module=None: registered,
    )
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )
    baseline_before = project.baseline_path.read_bytes()
    results_before = {
        path.name: path.read_bytes() for path in project.results_dir.iterdir()
    }

    failing = _RecordingPromptManager(error=error)
    monkeypatch.setattr(
        project,
        "prompt_manager",
        lambda mlflow_module=None: failing,
    )
    monkeypatch.setattr(
        catalog_module,
        "build_scorer",
        lambda *args, **kwargs: pytest.fail("a scorer was constructed"),
    )
    mlflow = FakeMlflow(run_id="run-2")

    with pytest.raises(type(error)) as excinfo:
        run_scoring(
            project,
            judges_enabled=True,
            mode="answer-sheet",
            assume_yes=True,
            mlflow_module=mlflow,
        )

    assert excinfo.value is error
    assert failing.calls == [("agentkit_judge_domain_policy", "production")]
    assert mlflow.evaluate_calls == []
    assert mlflow.experiment is None
    assert mlflow.tags == {}
    assert mlflow.run_artifacts == []
    assert project.baseline_path.read_bytes() == baseline_before
    assert {
        path.name: path.read_bytes() for path in project.results_dir.iterdir()
    } == results_before


def test_prompt_resolution_is_shared_by_provenance_and_scorer(tmp_path, monkeypatch):
    """The recorded URI and executed instructions come from one lookup."""

    project = _with_prompt_judge(tmp_path)

    class ChangingPromptManager:
        def __init__(self):
            self.calls = []

        def load(self, name, *, alias):
            self.calls.append((name, alias))
            version = len(self.calls)
            return SimpleNamespace(
                uri=f"prompts:/cat.sch.{name}/{version}",
                template=f"instructions version {version}",
            )

        def qualify(self, name):
            return f"cat.sch.{name}"

    manager = ChangingPromptManager()
    monkeypatch.setattr(
        project,
        "prompt_manager",
        lambda mlflow_module=None: manager,
    )
    mlflow = FakeMlflow()

    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=mlflow,
    )

    assert manager.calls == [("agentkit_judge_domain_policy", "production")]
    scorer = next(
        scorer
        for scorer in mlflow.evaluate_calls[0]["scorers"]
        if getattr(scorer, "name", None) == "pension_domain_policy"
    )
    assert scorer.instructions == "instructions version 1"
    assert mlflow.tags["aai.judge_prompt_versions"] == (
        "pension_domain_policy=" "prompts:/cat.sch.agentkit_judge_domain_policy/1"
    )


def test_a_moved_judge_prompt_refuses_before_confirmation_run_or_spend(
    tmp_path, monkeypatch
):
    """A moved alias means a different judge scored the baseline."""

    from aai_core.agentkit import runner as runner_module

    project = _with_prompt_judge(tmp_path)
    _use_registered_prompt(project, monkeypatch)
    monkeypatch.setattr(
        runner_module,
        "_resolved_prompt_versions",
        lambda plan, prompt_loader: {"pension_domain_policy": "prompts:/cat.sch.p/3"},
    )
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )

    monkeypatch.setattr(
        runner_module,
        "_resolved_prompt_versions",
        lambda plan, prompt_loader: {"pension_domain_policy": "prompts:/cat.sch.p/4"},
    )
    mlflow = FakeMlflow()
    asked = []
    environ = {}

    with pytest.raises(BaselineIncomparableError) as excinfo:
        run_scoring(
            project,
            judges_enabled=True,
            mode="answer-sheet",
            assume_yes=False,
            confirm=lambda prompt: asked.append(prompt) or True,
            environ=environ,
            mlflow_module=mlflow,
        )

    assert "judge prompt moved" in str(excinfo.value)
    # Refused before approval, process tuning, the run, or a judge call.
    assert asked == []
    assert environ == {}
    assert mlflow.experiment is None
    assert mlflow.tags == {}
    assert mlflow.evaluate_calls == []


def test_an_unchanged_judge_prompt_compares_cleanly(tmp_path, monkeypatch):
    from aai_core.agentkit import runner as runner_module

    project = _with_prompt_judge(tmp_path)
    _use_registered_prompt(project, monkeypatch)
    monkeypatch.setattr(
        runner_module,
        "_resolved_prompt_versions",
        lambda plan, prompt_loader: {"pension_domain_policy": "prompts:/cat.sch.p/3"},
    )
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )

    outcome, _ = run_scoring(
        project,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )

    assert not any("judge prompt" in warning for warning in outcome.warnings)


def test_prompt_drift_can_be_overridden_and_is_recorded(tmp_path, monkeypatch):
    from aai_core.agentkit import runner as runner_module

    project = _with_prompt_judge(tmp_path)
    _use_registered_prompt(project, monkeypatch)
    monkeypatch.setattr(
        runner_module,
        "_resolved_prompt_versions",
        lambda plan, prompt_loader: {"pension_domain_policy": "prompts:/cat.sch.p/3"},
    )
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )
    monkeypatch.setattr(
        runner_module,
        "_resolved_prompt_versions",
        lambda plan, prompt_loader: {"pension_domain_policy": "prompts:/cat.sch.p/4"},
    )

    outcome, _ = run_scoring(
        project,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        allow_baseline_drift=True,
        mlflow_module=FakeMlflow(),
    )

    assert any("judge prompt moved" in warning for warning in outcome.warnings)


def test_a_removed_scorer_refuses_the_comparison(tmp_path):
    """Removing a scorer removes its threshold; the gate must not shrug."""

    project = _project(tmp_path)
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )
    (tmp_path / "agentkit.yaml").write_text(
        "version: 1\n"
        "agent: src/app/example_agent.py:respond\n"
        "dataset: evals/data/golden_cases.json\n"
        "scorers:\n"
        "  remove: [safety]\n"
    )
    reduced = ProjectContext.load(tmp_path / "agentkit.yaml", environ={})
    mlflow = FakeMlflow()

    with pytest.raises(BaselineIncomparableError) as excinfo:
        run_scoring(
            reduced,
            judges_enabled=True,
            mode="answer-sheet",
            assume_yes=True,
            mlflow_module=mlflow,
        )

    assert "the baseline scored safety but this run does not" in str(excinfo.value)
    assert mlflow.evaluate_calls == []


def test_smoke_still_compares_against_a_judged_baseline(tmp_path):
    """The fast loop must survive the scorer-set check.

    `smoke` is judge-free by design, so the baseline's judge scorers are
    not a mismatch — otherwise every smoke run after a `compare` baseline
    would be refused.
    """

    project = _project(tmp_path)
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )

    outcome, code = run_scoring(
        project,
        command="smoke",
        judges_enabled=False,
        require_baseline=False,
        mode="answer-sheet",
        assume_yes=True,
    )

    assert code == EXIT_PASS
    assert not any("baseline scored" in warning for warning in outcome.warnings)


def test_smoke_keeps_gating_once_the_dataset_outgrows_the_sample(tmp_path):
    """The credential-free pull-request gate has to survive a growing suite.

    `validate_dataset` tells projects to grow toward 150+ rows, and
    `smoke` scores a deterministic sample of them (20 by default). That
    sample is a narrower scope than the committed baseline, so a blocking
    comparability check would refuse the run and the fast loop would stop
    working exactly as a project matures.
    """

    project = _project(tmp_path, rows=40)
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )

    outcome, code = run_scoring(
        project,
        command="smoke",
        rows_limit=20,
        judges_enabled=False,
        require_baseline=False,
        mode="answer-sheet",
        assume_yes=True,
    )

    assert code == EXIT_PASS
    warnings = "\n".join(outcome.warnings)
    # Set aside, and said so — the reason names the scope, not the data.
    assert "set aside" in warnings
    assert "full/40 rows but this run scores sample/20" in warnings
    assert "the dataset changed" not in warnings
    assert outcome.results.baseline_run_id is None


def test_compare_still_refuses_a_scope_the_baseline_never_measured(tmp_path):
    """Promotion-grade commands keep the refusal: the delta is the point."""

    project = _project(tmp_path, rows=40)
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )
    mlflow = FakeMlflow()

    with pytest.raises(BaselineIncomparableError) as excinfo:
        run_scoring(
            project,
            rows_limit=20,
            judges_enabled=True,
            mode="answer-sheet",
            assume_yes=True,
            mlflow_module=mlflow,
        )

    message = str(excinfo.value)
    assert "full/40 rows but this run scores sample/20" in message
    assert "the dataset changed" not in message
    assert mlflow.evaluate_calls == []


def test_a_sample_of_the_same_dataset_is_not_a_different_dataset(tmp_path):
    """A sampled baseline compares cleanly against the same sample.

    The scope matches, and the digest check has to see through the sample
    to the dataset it was drawn from — otherwise the one comparison that
    is exactly reproducible would be refused.
    """

    project = _project(tmp_path, rows=40)
    run_scoring(
        project,
        establish_baseline=True,
        rows_limit=20,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )

    outcome, code = run_scoring(
        project,
        rows_limit=20,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )

    assert code == EXIT_PASS
    assert not outcome.warnings


def test_an_unsampled_smoke_run_still_catches_a_regression(tmp_path):
    """Below the sample size nothing changes: smoke still compares."""

    project = _project(
        tmp_path,
        config_text=(
            "version: 1\n"
            "agent: src/app/example_agent.py:respond\n"
            "dataset: evals/data/golden_cases.json\n"
            "regression_budget:\n"
            "  keyword_coverage/mean: 0.05\n"
        ),
    )
    run_scoring(
        project,
        command="smoke",
        rows_limit=20,
        judges_enabled=False,
        require_baseline=False,
        establish_baseline=True,
        mode="answer-sheet",
        assume_yes=True,
    )
    sheet = tmp_path / "evals" / "data" / "answer_sheet.json"
    sheet.write_text(
        json.dumps(
            [
                {"question": f"question {index}", "answer": "unrelated"}
                for index in range(12)
            ]
        )
    )

    outcome, code = run_scoring(
        project,
        command="smoke",
        rows_limit=20,
        judges_enabled=False,
        require_baseline=False,
        mode="answer-sheet",
        assume_yes=True,
    )

    assert code == EXIT_THRESHOLD_FAILED
    assert not any("set aside" in warning for warning in outcome.warnings)


class _CoverageFrame:
    """A result table with a scorer that declined some rows."""

    def __init__(self, values, errors=None):
        self._data = {"skipping/value": values}
        if errors is not None:
            self._data["skipping/error_message"] = errors
        self.columns = list(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)


def test_a_scorer_that_declined_rows_reports_its_coverage():
    """A mean over the retrieving rows must not read as a whole-dataset mean."""

    from aai_core.agentkit.runner import _coverage_warnings

    result = SimpleNamespace(
        metrics={"skipping/mean": 1.0},
        result_df=_CoverageFrame([1.0, float("nan"), 1.0, None]),
    )

    warnings = _coverage_warnings(result)

    assert warnings == [
        "skipping judged 2 of 4 rows; 2 had nothing for it to score, so its "
        "mean covers the rest"
    ]


def test_a_failed_row_is_not_counted_as_declined():
    """An error is a failure, not a skip — the gate already fails on those."""

    from aai_core.agentkit.runner import _coverage_warnings

    result = SimpleNamespace(
        metrics={},
        result_df=_CoverageFrame(
            [1.0, float("nan"), 1.0, 1.0], errors=[None, "judge exploded", None, None]
        ),
    )

    assert _coverage_warnings(result) == []


def test_full_coverage_says_nothing():
    from aai_core.agentkit.runner import _coverage_warnings

    result = SimpleNamespace(metrics={}, result_df=_CoverageFrame([1.0, 0.5]))

    assert _coverage_warnings(result) == []


def _traced_project(tmp_path, *, trace, rows=12):
    """A project whose dataset carries a `trace` value on every row."""

    project = _project(tmp_path, rows=rows)
    cases = json.loads((tmp_path / "evals" / "data" / "golden_cases.json").read_text())
    for index, case in enumerate(cases):
        case["trace"] = trace(index)
    (tmp_path / "evals" / "data" / "golden_cases.json").write_text(json.dumps(cases))
    return project


def test_a_live_run_does_not_hand_mlflow_a_stored_trace(tmp_path):
    """The recorded trace is a different run's answer.

    MLflow does not ignore it: a present trace column rewrites inputs,
    outputs and expectations from the traces before predict_fn is touched.
    """

    project = _traced_project(tmp_path, trace=lambda i: {"info": {"trace_id": f"t{i}"}})
    fake = FakeMlflow()

    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=False,
        mode="live",
        assume_yes=True,
        mlflow_module=fake,
    )

    call = fake.evaluate_calls[0]
    assert call["predict_fn"] is not None
    assert all("trace" not in row for row in call["data"])


def test_a_nullable_trace_column_still_scores(tmp_path):
    """The P1, end to end.

    A UC dataset with a nullable trace column infers `live` correctly, but
    used to carry `trace: NaN` into MLflow, where
    `_extract_request_response_from_trace` calls `trace.data` on every
    value and raises before the agent runs.
    """

    project = _traced_project(tmp_path, trace=lambda i: None)
    fake = FakeMlflow()

    outcome, code = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=False,
        mode="live",
        assume_yes=True,
        mlflow_module=fake,
    )

    assert code == EXIT_PASS
    assert all("trace" not in row for row in fake.evaluate_calls[0]["data"])


def test_a_traces_run_still_carries_its_traces(tmp_path):
    project = _traced_project(tmp_path, trace=lambda i: {"info": {"trace_id": f"t{i}"}})
    fake = FakeMlflow()

    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=False,
        mode="traces",
        assume_yes=True,
        mlflow_module=fake,
    )

    call = fake.evaluate_calls[0]
    assert call["predict_fn"] is None
    assert all("trace" in row for row in call["data"])


def test_a_traces_run_says_when_mlflow_will_replace_the_expectations(tmp_path):
    project = _traced_project(
        tmp_path,
        trace=lambda i: {
            "info": {
                "trace_id": f"t{i}",
                "assessments": [
                    {
                        "assessment_name": "expected_response",
                        "expectation": {"value": "other"},
                    }
                ],
            }
        },
    )
    fake = FakeMlflow()

    outcome, _ = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=False,
        mode="traces",
        assume_yes=True,
        mlflow_module=fake,
    )

    assert any(
        "trace's assessments" in warning for warning in outcome.warnings
    ), outcome.warnings


def test_a_live_run_prices_the_fanout_it_will_actually_have(tmp_path):
    """Stored traces are discarded in a live run, so they cannot price it.

    Counting them read the *old* agent's retrieval as this run's exact
    fan-out. If the recorded agent retrieved one chunk and the current one
    retrieves fifty, `budget.max_judge_calls` authorises the run against a
    number the run will exceed.
    """

    project = _traced_project(
        tmp_path,
        trace=lambda i: {
            "data": {
                "spans": [
                    {
                        "parent_span_id": None,
                        "span_type": "RETRIEVER",
                        "inputs": {"question": f"question {i}"},
                        "outputs": [{"page_content": "one chunk"}],
                    }
                ]
            }
        },
    )
    fake = FakeMlflow()

    outcome, _ = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="live",
        assume_yes=True,
        mlflow_module=fake,
    )

    # The stored traces are gone, so the configured assumption applies
    # rather than the previous agent's exact count.
    assert outcome.cost.fanout_counted is False


def test_a_traces_run_still_counts_the_fanout_it_will_judge(tmp_path):
    project = _traced_project(
        tmp_path,
        trace=lambda i: {
            "data": {
                "spans": [
                    {
                        "parent_span_id": None,
                        "span_type": "RETRIEVER",
                        "inputs": {"question": f"question {i}"},
                        "outputs": [{"page_content": "one chunk"}],
                    }
                ]
            }
        },
    )
    fake = FakeMlflow()

    outcome, _ = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="traces",
        assume_yes=True,
        mlflow_module=fake,
    )

    assert outcome.cost.fanout_counted is True


def test_a_trace_only_dataset_can_still_be_re_run_live(tmp_path):
    """Explicit --mode live over production traces must keep working.

    Stripping the trace without recovering its request left rows MLflow
    cannot evaluate; the request now travels as `inputs`.
    """

    project = _project(tmp_path)
    cases = [
        {
            "trace": {
                "data": {
                    "spans": [
                        {
                            "parent_span_id": None,
                            "inputs": {"question": f"question {index}"},
                        }
                    ]
                }
            }
        }
        for index in range(12)
    ]
    (tmp_path / "evals" / "data" / "golden_cases.json").write_text(json.dumps(cases))
    fake = FakeMlflow()

    outcome, code = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=False,
        mode="live",
        assume_yes=True,
        mlflow_module=fake,
    )

    assert code == EXIT_PASS
    payload = fake.evaluate_calls[0]["data"]
    assert all(row["inputs"]["question"].startswith("question") for row in payload)
    assert all("trace" not in row for row in payload)


def test_a_row_whose_request_cannot_be_recovered_is_refused(tmp_path):
    project = _project(tmp_path)
    cases = [{"trace": {"info": {"request_preview": "unparseable"}}} for _ in range(12)]
    (tmp_path / "evals" / "data" / "golden_cases.json").write_text(json.dumps(cases))
    fake = FakeMlflow()

    with pytest.raises(ConfigError) as excinfo:
        run_scoring(
            project,
            establish_baseline=True,
            judges_enabled=False,
            mode="live",
            assume_yes=True,
            mlflow_module=fake,
        )

    assert "no inputs to send the agent" in str(excinfo.value)
    assert "--mode traces" in str(excinfo.value)
    assert fake.evaluate_calls == []


def test_a_trace_replaced_expectation_removes_the_scorer_it_fed(tmp_path):
    """The plan must not promise what MLflow will have taken away.

    One trace assessment replaces the whole expectations column, so the
    rows without one lose `expected_response` entirely — and
    keyword_coverage reads an absent expected response as a vacuous 1.0.
    """

    def trace(index):
        assessments = (
            [
                {
                    "assessment_name": "expected_response",
                    "expectation": {"value": "from the trace"},
                }
            ]
            if index == 0
            else []
        )
        return {"info": {"trace_id": f"t{index}", "assessments": assessments}}

    project = _traced_project(tmp_path, trace=trace)
    fake = FakeMlflow()

    outcome, _ = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=False,
        mode="traces",
        assume_yes=True,
        mlflow_module=fake,
    )

    selected = {entry.spec.name for entry in outcome.plan.entries}
    assert "keyword_coverage" not in selected
    excluded = {item.spec.name: item.reason for item in outcome.plan.excluded}
    assert "expected_response" in excluded.get("keyword_coverage", "")
