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
    def factory(**kwargs):
        return SimpleNamespace(class_name=class_name, kwargs=kwargs)

    return factory


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


def test_declined_confirmation_scores_nothing(tmp_path):
    """A cancelled run is not a passed run.

    The usual cause is a CI job on a non-interactive stream with no
    `--yes`; exit 0 there would report success for an evaluation that
    never happened.
    """

    project = _project(tmp_path)
    fake = FakeMlflow()

    outcome, code = run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=False,
        confirm=lambda prompt: False,
        mlflow_module=fake,
    )

    assert code == EXIT_ERROR
    assert outcome.declined
    assert fake.evaluate_calls == []
    assert not project.baseline_path.exists()
    assert any("--yes" in message for message in outcome.messages)


def test_budget_stops_the_run_before_any_call(tmp_path):
    project = _project(
        tmp_path,
        config_text=(
            "version: 1\n"
            "agent: src/app/example_agent.py:respond\n"
            "dataset: evals/data/golden_cases.json\n"
            "budget:\n  max_judge_calls: 3\n"
        ),
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


def test_the_refusal_precedes_the_budget_and_the_prompt(tmp_path):
    """Refusing after paying for judge calls would be worthless."""

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

    with pytest.raises(BaselineIncomparableError):
        run_scoring(
            project,
            judges_enabled=True,
            mode="answer-sheet",
            assume_yes=False,
            confirm=lambda prompt: asked.append(prompt) or True,
            mlflow_module=FakeMlflow(),
        )

    assert asked == []


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


def test_publish_failure_warns_without_failing_the_run(tmp_path):
    project = _project(tmp_path)
    mlflow = FakeMlflow()

    def _broken(*args, **kwargs):
        raise RuntimeError("tracking store unavailable")

    mlflow.MlflowClient = _broken

    outcome, code = run_scoring(
        project,
        command="compare",
        establish_baseline=True,
        assume_yes=True,
        mlflow_module=mlflow,
        environ={},
    )

    assert code == EXIT_PASS
    assert any("could not attach the results record" in w for w in outcome.warnings)


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


def test_a_moved_judge_prompt_refuses_before_any_judge_call(tmp_path, monkeypatch):
    """A moved alias means a different judge scored the baseline."""

    from aai_core.agentkit import runner as runner_module

    project = _with_prompt_judge(tmp_path)
    monkeypatch.setattr(
        runner_module,
        "_resolved_prompt_versions",
        lambda plan, project_, mlflow: {
            "pension_domain_policy": "prompts:/cat.sch.p/3"
        },
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
        lambda plan, project_, mlflow: {
            "pension_domain_policy": "prompts:/cat.sch.p/4"
        },
    )
    mlflow = FakeMlflow()

    with pytest.raises(BaselineIncomparableError) as excinfo:
        run_scoring(
            project,
            judges_enabled=True,
            mode="answer-sheet",
            assume_yes=True,
            mlflow_module=mlflow,
        )

    assert "judge prompt moved" in str(excinfo.value)
    # Refused before the run opened and before a single judge call.
    assert mlflow.evaluate_calls == []


def test_an_unchanged_judge_prompt_compares_cleanly(tmp_path, monkeypatch):
    from aai_core.agentkit import runner as runner_module

    project = _with_prompt_judge(tmp_path)
    monkeypatch.setattr(
        runner_module,
        "_resolved_prompt_versions",
        lambda plan, project_, mlflow: {
            "pension_domain_policy": "prompts:/cat.sch.p/3"
        },
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
    monkeypatch.setattr(
        runner_module,
        "_resolved_prompt_versions",
        lambda plan, project_, mlflow: {
            "pension_domain_policy": "prompts:/cat.sch.p/3"
        },
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
        lambda plan, project_, mlflow: {
            "pension_domain_policy": "prompts:/cat.sch.p/4"
        },
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
