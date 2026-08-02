"""Unit tests for the scoring engine: governed runs, tags, baselines, gating.

The MLflow module is injected, so these run with zero cloud access. The
fake mirrors the surface the runner actually uses (see
tests/test_experiment_helpers.py for the same pattern).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_core.agentkit.baseline import load_baseline
from aai_core.agentkit.config import ProjectContext
from aai_core.agentkit.errors import BaselineMissingError, BudgetExceededError
from aai_core.agentkit.gate import EXIT_PASS, EXIT_THRESHOLD_FAILED
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

    assert code == EXIT_PASS
    assert outcome.declined
    assert fake.evaluate_calls == []
    assert not project.baseline_path.exists()


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


def test_dataset_drift_against_the_baseline_warns(tmp_path):
    project = _project(tmp_path)
    run_scoring(
        project,
        establish_baseline=True,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )
    cases = json.loads(
        (project.root / "evals" / "data" / "golden_cases.json").read_text()
    )
    cases[0]["expectations"]["expected_response"] = "a different expectation"
    (project.root / "evals" / "data" / "golden_cases.json").write_text(
        json.dumps(cases)
    )

    outcome, _ = run_scoring(
        project,
        judges_enabled=True,
        mode="answer-sheet",
        assume_yes=True,
        mlflow_module=FakeMlflow(),
    )

    assert any("different dataset version" in w for w in outcome.warnings)


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
