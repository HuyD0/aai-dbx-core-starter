"""Tests for the clone-friendly learning-example runner."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "examples.py"


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("aai_examples_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def lifecycle_support():
    path = ROOT / "examples" / "lifecycle_support.py"
    spec = importlib.util.spec_from_file_location("aai_lifecycle_support", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_progressive_curriculum_uses_stable_fictional_earnings_cases(
    lifecycle_support,
):
    cases = lifecycle_support.CASES

    assert lifecycle_support.EXPERIMENT_PURPOSE == "earnings-summary-quality-cost"
    assert lifecycle_support.PROMPT_NAME == "earnings_summary"
    assert lifecycle_support.BASELINE_NAME == "baseline-earnings-summary-prompt-v1"
    assert lifecycle_support.CHANGE_NAME == "change-cited-earnings-summary-prompt-v2"
    assert lifecycle_support.RELEASE_NAME == "earnings-summary-prompt-v2"
    assert lifecycle_support.DATASET_NAME == "fictional-earnings-summary-regression-v1"
    assert len(cases) == 3
    assert [case.case_id for case in cases] == [
        "quarterly-revenue-and-margin",
        "forward-revenue-and-margin-guidance",
        "cash-flow-inventory-and-supplier-risk",
    ]
    assert [case.source_id for case in cases] == [
        "ARS-FY25-Q2-RESULTS",
        "ARS-FY25-Q2-GUIDANCE",
        "ARS-FY25-Q2-CASH-RISK",
    ]

    records = [case.evaluation_record() for case in cases]
    assert all(
        set(record["inputs"])
        == {"case_id", "question", "earnings_excerpt", "source_id"}
        for record in records
    )
    assert all(
        set(record["expectations"])
        == {"source_id", "required_facts", "investment_recommendation_prohibited"}
        for record in records
    )

    curriculum_text = " ".join(
        [
            lifecycle_support.BASELINE_PROMPT,
            lifecycle_support.CHANGE_PROMPT,
            *(str(value) for case in cases for value in vars(case).values()),
        ]
    ).lower()
    for concept in (
        "aster ridge systems",
        "fictional",
        "revenue",
        "operating margin",
        "guidance",
        "free cash flow",
        "inventory",
        "supplier",
    ):
        assert concept in curriculum_text
    for prompt in (
        lifecycle_support.BASELINE_PROMPT,
        lifecycle_support.CHANGE_PROMPT,
    ):
        normalized = prompt.lower()
        assert "investment advice" in normalized
        assert all(action in normalized for action in ("buying", "selling", "holding"))
    assert "yfinance" not in curriculum_text
    assert "market data" not in curriculum_text


def test_prompt_change_only_adds_exact_source_citation(lifecycle_support):
    baseline_lines = lifecycle_support.BASELINE_PROMPT.splitlines()
    change_lines = lifecycle_support.CHANGE_PROMPT.splitlines()
    added_lines = [line for line in change_lines if line not in baseline_lines]

    baseline_index = 0
    for line in change_lines:
        if (
            baseline_index < len(baseline_lines)
            and line == baseline_lines[baseline_index]
        ):
            baseline_index += 1
    assert baseline_index == len(baseline_lines)
    assert added_lines
    assert all(
        "source" in line.lower() or "{{source_id}}" in line for line in added_lines
    )
    assert "exactly once" in " ".join(added_lines).lower()

    for case in lifecycle_support.CASES:
        inputs = case.evaluation_record()["inputs"]
        baseline = lifecycle_support.generate_response("baseline", **inputs)
        change = lifecycle_support.generate_response("change", **inputs)
        assert baseline["answer"].count(case.source_id) == 0
        assert change["answer"].count(case.source_id) == 1
        assert (
            lifecycle_support.recommendation_policy_score(
                baseline,
                case.evaluation_record()["expectations"],
            )
            == 1.0
        )
        assert (
            lifecycle_support.recommendation_policy_score(
                change,
                case.evaluation_record()["expectations"],
            )
            == 1.0
        )

    baseline_metrics = lifecycle_support.metrics_for("baseline")
    change_metrics = lifecycle_support.metrics_for("change")
    decision = lifecycle_support.release_decision(baseline_metrics, change_metrics)
    assert baseline_metrics["citation_rate"] == 0.0
    assert change_metrics["citation_rate"] == 1.0
    assert decision["decision"] == "adopt"
    assert decision["release"] == "earnings-summary-prompt-v2"
    assert all(decision["checks"].values())


def test_fact_matching_ignores_formatting_but_source_id_remains_exact(
    lifecycle_support,
):
    case = lifecycle_support.CASES[2]
    expectations = case.evaluation_record()["expectations"]
    formatted_answer = {
        "answer": (
            "Free cash flow was **$21.7\u202fmillion**; inventory grew "
            "28\u202f%, and there was a single\u2011source supplier risk. "
            "(ARS\u2011FY25\u2011Q2\u2011CASH\u2011RISK)"
        )
    }

    assert lifecycle_support.fact_coverage(formatted_answer, expectations) == 1.0
    assert lifecycle_support.citation_score(formatted_answer, expectations) == 0.0

    exact_answer = {
        "answer": f"{formatted_answer['answer']} {case.source_id}",
    }
    assert lifecycle_support.citation_score(exact_answer, expectations) == 1.0

    extended_id_answer = {
        "answer": f"Source: {case.source_id}-DRAFT",
    }
    assert lifecycle_support.citation_score(extended_id_answer, expectations) == 0.0


@pytest.mark.parametrize(
    "answer",
    [
        "You should buy Aster Ridge Systems.",
        "Sell Aster Ridge Systems now.",
        "Investors ought to hold the stock.",
        "I recommend buying shares.",
        "The stock is overweight.",
    ],
)
def test_recommendation_policy_rejects_direct_investment_advice(
    lifecycle_support,
    answer,
):
    expectations = lifecycle_support.CASES[0].evaluation_record()["expectations"]
    assert (
        lifecycle_support.recommendation_policy_score(
            {"answer": answer},
            expectations,
        )
        == 0.0
    )


def test_catalog_separates_offline_connected_and_interactive_examples(runner):
    assert runner.EXAMPLES["offline_hello_world"].connected is False
    for name in (
        "first_trace",
        "first_experiment",
        "first_prompt",
        "first_evaluation",
    ):
        assert runner.EXAMPLES[name].connected is True
        assert runner.EXAMPLES[name].local is True
    assert runner.EXAMPLES["connected_setup"].connected is True
    assert runner.EXAMPLES["connected_setup"].interactive is True
    assert runner.EXAMPLES["first_llm_call"].interactive is True
    assert runner.EXAMPLES["connected_first_call"].connected is True
    assert runner.EXAMPLES["connected_first_call"].interactive is False
    for name in (
        "tool_trajectory_evaluation",
        "multi_turn_session_evaluation",
        "layered_judges",
        "cost_quality_tradeoff",
        "agent_alignment_optimization",
    ):
        assert runner.EXAMPLES[name].connected is False
        assert runner.EXAMPLES[name].local is True
        assert runner.EXAMPLES[name].interactive is True

    numbered_paths = [example.path for example in runner.EXAMPLES.values()]
    assert [Path(path).name[:2] for path in numbered_paths] == [
        f"{number:02d}" for number in range(13)
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("first_trace", "first_trace"),
        ("first_trace.py", "first_trace"),
        ("01_first_trace.py", "first_trace"),
        ("08-tool-trajectory-evaluation.ipynb", "tool_trajectory_evaluation"),
    ),
)
def test_example_name_normalization_keeps_cli_aliases_and_accepts_numbered_files(
    runner,
    value,
    expected,
):
    assert runner._normalize_example_name(value) == expected


def test_prompt_registration_normalizes_unity_catalog_not_found(
    lifecycle_support, monkeypatch
):
    mlflow = pytest.importorskip("mlflow")

    class MissingPromptError(Exception):
        error_code = "NOT_FOUND"

    class PromptClient:
        def get_prompt(self, name):
            raise MissingPromptError(f"Prompt {name} does not exist")

    registered = SimpleNamespace(version=1)
    prompts = SimpleNamespace(
        qualify=lambda name: f"dbx_dev.default.{name}",
        register=lambda *args, **kwargs: registered,
    )
    monkeypatch.setattr(mlflow, "MlflowClient", lambda: PromptClient())

    result = lifecycle_support.ensure_prompt_version(
        prompts,
        role="baseline",
        template=lifecycle_support.BASELINE_PROMPT,
    )

    assert result is registered


def test_connected_environment_routes_mlflow_to_databricks(runner, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///wrong.db")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "sqlite:///wrong.db")
    monkeypatch.setenv("DATABRICKS_AUTH_TYPE", "pat")

    environment = runner._connected_environment()

    assert environment["MLFLOW_TRACKING_URI"] == "databricks"
    assert environment["MLFLOW_REGISTRY_URI"] == "databricks-uc"
    assert environment["DATABRICKS_AUTH_TYPE"] == "azure-cli"


def test_local_environment_uses_isolated_store(runner, tmp_path, monkeypatch):
    local_dir = tmp_path / ".aai" / "local"
    monkeypatch.setattr(runner, "LOCAL_DIR", local_dir)
    monkeypatch.setattr(runner, "LOCAL_DB", local_dir / "mlflow.db")
    monkeypatch.setattr(runner, "LOCAL_ARTIFACTS", local_dir / "mlruns")

    environment = runner._local_environment()

    assert environment["MLFLOW_TRACKING_URI"] == (
        f"sqlite:///{(local_dir / 'mlflow.db').resolve()}"
    )
    assert environment["MLFLOW_REGISTRY_URI"] == environment["MLFLOW_TRACKING_URI"]
    assert environment["AAI_PLATFORM_CONFIG"] == str(runner.CONFIG_EXAMPLE)
    assert environment["AAI_EXAMPLE_LOCAL_DIR"] == str(local_dir.resolve())
    assert environment["AAI_EXAMPLE_ARTIFACT_ROOT"] == str(
        (local_dir / "mlruns").resolve()
    )


def test_config_preflight_checks_only_fields_used_by_the_example(
    runner, tmp_path, monkeypatch
):
    config = tmp_path / "aai-platform.yml"
    config.write_text(
        """
platform:
  experiment_name: /Shared/learning
  catalog: main
  schema: example_ai
providers:
  models:
    general-chat:
      deployment: replace-with-serving-endpoint
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "CONFIG", config)

    assert runner._config_issues(runner.EXAMPLES["first_trace"]) == []
    assert runner._config_issues(runner.EXAMPLES["first_llm_call"]) == [
        "Configure `providers.models.general-chat.deployment` in "
        "aai-platform.yml (current value: 'replace-with-serving-endpoint')."
    ]


def test_connect_creates_local_config_once(runner, tmp_path, monkeypatch, capsys):
    config = tmp_path / "aai-platform.yml"
    example = tmp_path / "aai-platform.example.yml"
    example.write_text(
        """
platform:
  experiment_name: /Shared/learning
providers:
  models:
    general-chat:
      deployment: ready-endpoint
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "CONFIG", config)
    monkeypatch.setattr(runner, "CONFIG_EXAMPLE", example)
    monkeypatch.setattr(runner, "_cloud_issues", lambda environment: [])

    assert runner.connect() == 0
    assert config.read_text(encoding="utf-8") == example.read_text(encoding="utf-8")
    assert "Created local configuration" in capsys.readouterr().out

    customized = config.read_text(encoding="utf-8").replace(
        "ready-endpoint", "my-endpoint"
    )
    config.write_text(customized, encoding="utf-8")
    assert runner.connect() == 0
    assert config.read_text(encoding="utf-8") == customized
    assert "Using existing local configuration" in capsys.readouterr().out


def test_connect_returns_nonzero_when_authentication_is_blocked(
    runner, tmp_path, monkeypatch, capsys
):
    config = tmp_path / "aai-platform.yml"
    config.write_text(
        """
platform:
  experiment_name: /Shared/learning
providers:
  models:
    general-chat:
      deployment: ready-endpoint
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "CONFIG", config)
    monkeypatch.setattr(
        runner,
        "_cloud_issues",
        lambda environment: ["Azure CLI is not authenticated; run `az login`."],
    )

    assert runner.connect() == 2
    output = capsys.readouterr().out
    assert "Authentication still needed" in output
    assert "az login" in output


def test_connected_run_stops_before_cloud_call_when_config_is_missing(
    runner, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(runner, "CONFIG", tmp_path / "missing.yml")
    monkeypatch.setattr(runner, "_module_issues", lambda example: [])

    def unexpected_cloud_check(environment):
        raise AssertionError("cloud check must not run before local preflight passes")

    monkeypatch.setattr(runner, "_cloud_issues", unexpected_cloud_check)

    assert runner.run_example("first_trace.py") == 2
    output = capsys.readouterr().out
    assert "aai-platform.yml is missing" in output
    assert "make workspace-connect" in output


@pytest.mark.parametrize(
    "example_name",
    ["first_trace", "first_experiment", "first_prompt", "first_evaluation"],
)
def test_local_run_never_checks_cloud_and_reports_workspace_path(
    runner, tmp_path, monkeypatch, capsys, example_name
):
    local_dir = tmp_path / ".aai" / "local"
    monkeypatch.setattr(runner, "LOCAL_DIR", local_dir)
    monkeypatch.setattr(runner, "LOCAL_DB", local_dir / "mlflow.db")
    monkeypatch.setattr(runner, "LOCAL_ARTIFACTS", local_dir / "mlruns")
    monkeypatch.setattr(runner, "_module_issues", lambda example: [])

    def unexpected_cloud_check(environment):
        raise AssertionError("local execution must not check cloud access")

    monkeypatch.setattr(runner, "_cloud_issues", unexpected_cloud_check)
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["cwd"] = kwargs["cwd"]
        observed["environment"] = kwargs["env"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.run_example(example_name, destination="local") == 0
    assert observed["command"][0] == sys.executable
    assert observed["cwd"] == local_dir
    assert observed["environment"]["AAI_PLATFORM_CONFIG"] == str(runner.CONFIG_EXAMPLE)
    assert observed["environment"]["MLFLOW_TRACKING_URI"].endswith(
        "/.aai/local/mlflow.db"
    )
    output = capsys.readouterr().out
    assert "make local-ui" in output
    assert "make workspace-connect" in output


def test_progressive_examples_execute_offline_with_connected_lineage(tmp_path, runner):
    mlflow = pytest.importorskip("mlflow")
    local_dir = tmp_path / "local"
    tracking_uri = f"sqlite:///{local_dir / 'mlflow.db'}"
    environment = dict(os.environ)
    environment.update(
        {
            "AAI_EXAMPLE_LOCAL_DIR": str(local_dir),
            "AAI_EXAMPLE_ARTIFACT_ROOT": str(local_dir / "artifacts"),
            "AAI_PLATFORM_CONFIG": str(ROOT / "aai-platform.example.yml"),
            "MLFLOW_TRACKING_URI": tracking_uri,
            "MLFLOW_REGISTRY_URI": tracking_uri,
            "PYTHONHASHSEED": "0",
        }
    )
    environment.pop("DATABRICKS_HOST", None)
    environment.pop("DATABRICKS_AUTH_TYPE", None)

    def execute(name: str) -> dict:
        example_path = runner.EXAMPLES[name].path
        completed = subprocess.run(
            [sys.executable, str(ROOT / example_path)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        result_lines = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("LIFECYCLE_RESULT=")
        ]
        assert len(result_lines) == 1, completed.stdout
        return json.loads(result_lines[0].split("=", 1)[1])

    names = (
        "first_trace",
        "first_experiment",
        "first_prompt",
        "first_evaluation",
    )
    payloads = {name: execute(name) for name in names}

    common_keys = {
        "hypothesis",
        "baseline",
        "change",
        "result",
        "decision",
        "release",
        "dataset_digest_sha256",
    }
    assert all(common_keys <= payload.keys() for payload in payloads.values())
    assert len({payload["dataset_digest_sha256"] for payload in payloads.values()}) == 1
    assert len({payload["experiment"] for payload in payloads.values()}) == 1
    assert {payload["experiment"] for payload in payloads.values()} == {
        "/Shared/example-ai-earnings-summary-quality-cost"
    }
    assert all(
        payloads[name]["release"] == "blocked_until_evaluated"
        for name in ("first_trace", "first_experiment", "first_prompt")
    )
    assert payloads["first_prompt"]["result"] == {
        "autolog": "not_applicable_no_model_or_framework_call",
        "exact_versions_linked": True,
        "manual_prompt_span": True,
        "measurement_status": {
            "cost": "pending_model_execution",
            "cost_coverage": "pending_model_execution",
            "latency": "pending_model_execution",
            "quality": "pending_full_evaluation",
            "tokens": "pending_model_execution",
        },
    }
    assert payloads["first_evaluation"]["result"]["gate_passed"] is True
    assert all(payloads["first_evaluation"]["result"]["checks"].values())
    assert payloads["first_evaluation"]["decision"] == "adopt"
    assert payloads["first_evaluation"]["release"] == "earnings-summary-prompt-v2"
    assert payloads["first_evaluation"]["baseline"]["name"] == (
        "baseline-earnings-summary-prompt-v1"
    )
    assert payloads["first_evaluation"]["change"]["name"] == (
        "change-cited-earnings-summary-prompt-v2"
    )
    evaluation_source = (ROOT / runner.EXAMPLES["first_evaluation"].path).read_text(
        encoding="utf-8"
    )
    assert evaluation_source.index("gate.require_passed()") < evaluation_source.index(
        'alias="production"'
    )

    repeated_prompt = execute("first_prompt")
    assert (
        repeated_prompt["baseline"]["prompt_uri"]
        == payloads["first_prompt"]["baseline"]["prompt_uri"]
    )
    assert (
        repeated_prompt["change"]["prompt_uri"]
        == payloads["first_prompt"]["change"]["prompt_uri"]
    )

    client = mlflow.MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
    )
    mlflow.set_registry_uri(tracking_uri)
    production_prompt = mlflow.genai.load_prompt(
        "prompts:/main.example_ai.earnings_summary@production"
    )
    assert str(production_prompt.version) == str(
        payloads["first_evaluation"]["change"]["prompt_uri"].rsplit("/", 1)[-1]
    )
    experiment_change = client.get_run(payloads["first_experiment"]["change"]["run_id"])
    assert experiment_change.data.tags["aai.run_purpose"] == "change"
    assert (
        experiment_change.data.tags["aai.baseline_run_id"]
        == payloads["first_experiment"]["baseline"]["run_id"]
    )
    experiment_traces = client.search_traces(
        locations=[experiment_change.info.experiment_id],
        run_id=experiment_change.info.run_id,
        include_spans=True,
        flush=True,
    )
    experiment_spans = [
        span.name for trace in experiment_traces for span in trace.data.spans
    ]
    assert experiment_spans.count("earnings_summary.experiment") == 3
    prompt_versions = client.search_prompt_versions("main.example_ai.earnings_summary")
    assert len(prompt_versions) == 2

    baseline_run_id = payloads["first_trace"]["baseline"]["run_id"]
    experiment_location = client.get_run(baseline_run_id).info.experiment_id
    baseline_traces = client.search_traces(
        locations=[experiment_location],
        run_id=baseline_run_id,
        include_spans=True,
        flush=True,
    )
    assert len(baseline_traces) == 1
    baseline_spans = baseline_traces[0].data.spans
    assert [span.name for span in baseline_spans].count(
        "earnings_summary.baseline"
    ) == 1
    assert [span.name for span in baseline_spans].count("deterministic.generate") == 1

    prompt_traces = client.search_traces(
        locations=[experiment_location],
        run_id=payloads["first_prompt"]["run_id"],
        include_spans=True,
        flush=True,
    )
    prompt_spans = [
        span
        for trace in prompt_traces
        for span in trace.data.spans
        if span.name == "prompt.register_load_render"
    ]
    assert len(prompt_spans) == 1
    assert prompt_spans[0].span_type == "PROMPT"


def test_interactive_workspace_example_prints_configured_exports(
    runner, tmp_path, monkeypatch, capsys
):
    config = tmp_path / "aai-platform.yml"
    config.write_text(
        """
platform:
  catalog: main
  schema: example_ai
providers:
  models:
    general-chat:
      deployment: ready-endpoint
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "CONFIG", config)
    monkeypatch.setattr(runner, "_module_issues", lambda example: [])
    monkeypatch.setattr(runner, "_cloud_issues", lambda environment: [])

    assert runner.run_example("first_llm_call") == 0
    output = capsys.readouterr().out
    assert (
        f"export DATABRICKS_HOST={runner._identifiers()['databricks_host']}" in output
    )
    assert "export DATABRICKS_AUTH_TYPE=azure-cli" in output
    assert "export MLFLOW_TRACKING_URI" not in output
    assert "export MLFLOW_REGISTRY_URI" not in output
    assert f"export AAI_PLATFORM_CONFIG={config}" in output
    assert (
        f"Open {runner.ROOT / runner.EXAMPLES['first_llm_call'].path} "
        "in your preferred notebook editor."
    ) in output
    assert f"Select this Python kernel: {sys.executable}" in output
    assert "A Databricks CLI profile is not required" in output
    assert "SEND_EVIDENCE_TO_DATABRICKS = False" in output
    assert "store prompts, runs, and traces in Databricks" in output


def test_interactive_local_lab_prints_only_local_evidence_exports(
    runner, tmp_path, monkeypatch, capsys
):
    local_dir = tmp_path / ".aai" / "local"
    monkeypatch.setattr(runner, "LOCAL_DIR", local_dir)
    monkeypatch.setattr(runner, "LOCAL_DB", local_dir / "mlflow.db")
    monkeypatch.setattr(runner, "LOCAL_ARTIFACTS", local_dir / "mlruns")
    monkeypatch.setattr(runner, "_module_issues", lambda example: [])

    assert (
        runner.run_example(
            "08_tool_trajectory_evaluation.ipynb",
            destination="local",
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "export MLFLOW_TRACKING_URI=" in output
    assert "export MLFLOW_REGISTRY_URI=" in output
    assert "export AAI_PLATFORM_CONFIG=" in output
    assert "export DATABRICKS_HOST=" not in output
    assert "keyless Azure CLI" not in output
    assert "credential-free and makes no model request" in output


def test_makefile_exposes_single_command_onboarding():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "check-uv:" in makefile
    assert "quickstart: install" in makefile
    assert "local-start: examples-install" in makefile
    assert "local-lifecycle: examples-install" in makefile
    assert "local-ui: examples-install" in makefile
    assert "workspace-connect: examples-install" in makefile
    assert "workspace-example: examples-install" in makefile
    assert "--extra databricks --extra genai --extra examples --locked" in makefile
    assert "import ipykernel; import jupyterlab" in makefile
    assert "$(PYTHON) scripts/examples.py local" in makefile
    assert "$(PYTHON) scripts/examples.py workspace" in makefile
    assert "Example dependencies ready in" in makefile
