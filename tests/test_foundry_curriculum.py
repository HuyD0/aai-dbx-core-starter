"""Credential-free checks for the Microsoft Foundry notebook curriculum."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import inspect
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CURRICULUM = ROOT / "examples" / "foundry-curriculum"
SETUP = CURRICULUM / "notebook_setup.py"

_spec = importlib.util.spec_from_file_location("foundry_notebook_setup", SETUP)
assert _spec is not None and _spec.loader is not None
setup = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = setup
_spec.loader.exec_module(setup)
labs = setup.load_offline_labs(CURRICULUM)

CORE_NOTEBOOKS = tuple(
    sorted(path for path in (CURRICULUM / "notebooks").glob("0[0-7]_*.ipynb"))
)
ADVANCED_NOTEBOOKS = tuple(
    sorted(
        path
        for path in (CURRICULUM / "notebooks").glob("*.ipynb")
        if path.name[:2] in {"08", "09", "10", "11", "12"}
    )
)


def _fake_context(config_path: Path):
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return SimpleNamespace(
        settings=SimpleNamespace(
            models=document["providers"]["models"],
            azure_identity=document["platform"]["azure_identity"],
            raw=document,
        )
    )


def _execute_notebook(path: Path) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": f"offline_{path.stem}"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"{path.name}:{cell['id']}", "exec"), namespace)
    return namespace


async def _execute_async_notebook(path: Path) -> dict[str, object]:
    """Execute a clean notebook namespace, including top-level-await cells."""

    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": f"offline_{path.stem}"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        code = compile(
            "".join(cell["source"]),
            f"{path.name}:{cell['id']}",
            "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            dont_inherit=True,
        )
        result = eval(code, namespace)
        if inspect.isawaitable(result):
            await result
    return namespace


def test_example_configuration_is_portable_and_project_scoped():
    path = CURRICULUM / "config" / "aai-platform.dev.example.yml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    platform = document["platform"]
    model = document["providers"]["models"]["foundry-chat"]

    assert platform["repository"] == "replace-with-owner/replace-with-repository"
    assert platform["catalog"] == "replace-with-catalog"
    assert model["provider"] == "foundry"
    assert "/api/projects/" in model["endpoint"]
    assert model["deployment"].startswith("replace-")
    assert document.get("secrets") == {}
    assert document["foundry"]["a2a"]["protocol_version"] == "1.0"
    assert document["foundry"]["agent"]["version"].startswith("replace-")
    assert document["foundry"]["evaluation"]["evaluator_model"].startswith("replace-")
    assert document["foundry"]["observability"][
        "application_insights_resource_id"
    ].startswith("replace-")


def test_readme_copies_the_portable_example_before_opening_notebooks():
    readme = (CURRICULUM / "README.md").read_text(encoding="utf-8")
    copy_source = "examples/foundry-curriculum/config/aai-platform.dev.example.yml"
    copy_target = "examples/foundry-curriculum/config/aai-platform.dev.yml"

    assert f"cp {copy_source} \\\n  {copy_target}" in readme
    assert f"Edit `{copy_target}`, not the tracked" in readme
    assert readme.index(copy_source) < readme.index("Open the repository")

    ignore = (CURRICULUM / "config" / ".gitignore").read_text(encoding="utf-8")
    assert "aai-platform.*.yml" in ignore
    assert "!aai-platform.*.example.yml" in ignore


def test_session_loads_endpoint_only_from_selected_configuration(tmp_path):
    config = tmp_path / "aai-platform.yml"
    config.write_text(
        """
platform:
  azure_identity: azure_cli
providers:
  models:
    foundry-chat:
      provider: foundry
      endpoint: https://account.services.ai.azure.com/api/projects/project-dev
      deployment: chat-model
""",
        encoding="utf-8",
    )

    session = setup.load_session(
        CURRICULUM,
        config_path=config,
        bootstrap_fn=_fake_context,
    )

    assert session.project_endpoint.endswith("/api/projects/project-dev")
    assert session.deployment == "chat-model"
    assert session.connected_ready
    assert session.safe_summary()["azure_identity"] == "azure_cli"
    assert not session.agent_ready
    assert not session.a2a_ready


def test_session_loads_advanced_identifiers_and_derives_a2a_urls(tmp_path):
    config = tmp_path / "aai-platform.yml"
    app_insights_resource_id = (
        "/subscriptions/00000000-0000-0000-0000-000000000000/"
        "resourceGroups/rg-foundry/providers/microsoft.insights/"
        "components/appi-foundry"
    )
    config.write_text(
        f"""
platform:
  azure_identity: azure_cli
providers:
  models:
    foundry-chat:
      provider: foundry
      endpoint: https://account.services.ai.azure.com/api/projects/project-dev
      deployment: chat-model
foundry:
  agent:
    name: curriculum-agent
    version: "2"
    id: curriculum-agent:2
  evaluation:
    evaluator_model: judge-model
  memory:
    store_name: learner-memory
  a2a:
    remote_agent_name: policy-specialist
    connection_name: policy-specialist-a2a
    protocol_version: "1.0"
  observability:
    application_insights_resource_id: {app_insights_resource_id}
""",
        encoding="utf-8",
    )

    session = setup.load_session(
        CURRICULUM,
        config_path=config,
        bootstrap_fn=_fake_context,
    )

    assert session.agent_ready
    assert session.trace_ready
    assert session.evaluation_ready
    assert session.a2a_ready
    assert session.observability_ready
    assert session.agent_reference() == {
        "type": "agent_reference",
        "name": "curriculum-agent",
        "version": "2",
    }
    assert session.a2a_base_url == (
        "https://account.services.ai.azure.com/api/projects/project-dev/agents/"
        "policy-specialist/endpoint/protocols/a2a"
    )
    assert session.a2a_agent_card_url.endswith("/agentCard/v1.0")


@pytest.mark.parametrize(
    "endpoint",
    (
        "http://account.services.ai.azure.com/api/projects/project-dev",
        "https://account.services.ai.azure.com",
        "https://example.com/api/projects/project-dev",
        "https://user:password@account.services.ai.azure.com/api/projects/project-dev",
    ),
)
def test_session_rejects_non_project_or_unsafe_endpoints(tmp_path, endpoint):
    config = tmp_path / "aai-platform.yml"
    config.write_text(
        f"""
platform:
  azure_identity: azure_cli
providers:
  models:
    foundry-chat:
      provider: foundry
      endpoint: {endpoint}
      deployment: chat-model
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="endpoint|HTTPS|host|project"):
        setup.load_session(
            CURRICULUM,
            config_path=config,
            bootstrap_fn=_fake_context,
        )


def test_connected_call_requires_an_explicit_network_opt_in():
    session = setup.FoundryNotebookSession(
        curriculum_root=CURRICULUM,
        config_path=Path("config.yml"),
        logical_model="foundry-chat",
        project_endpoint="https://account.services.ai.azure.com/api/projects/dev",
        deployment="chat-model",
        context=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="disabled"):
        setup.create_text_response(session, "hello")


@pytest.mark.parametrize(
    ("project_endpoint", "deployment"),
    (
        (
            "https://replace-with-foundry-account.services.ai.azure.com/"
            "api/projects/project-dev",
            "chat-model",
        ),
        (
            "https://account.services.ai.azure.com/api/projects/"
            "replace-with-project",
            "chat-model",
        ),
        (
            "https://account.services.ai.azure.com/api/projects/project-dev",
            "replace-with-model-deployment",
        ),
    ),
)
def test_connected_call_rejects_placeholder_configuration(project_endpoint, deployment):
    session = setup.FoundryNotebookSession(
        curriculum_root=CURRICULUM,
        config_path=Path("config.yml"),
        logical_model="foundry-chat",
        project_endpoint=project_endpoint,
        deployment=deployment,
        context=SimpleNamespace(),
    )

    assert not session.connected_ready
    with pytest.raises(RuntimeError, match="endpoint.*deployment"):
        setup.create_text_response(session, "hello", allow_network=True)


def test_curriculum_has_thirteen_clean_compilable_notebooks():
    notebooks = sorted((CURRICULUM / "notebooks").glob("*.ipynb"))
    assert [path.name for path in notebooks] == [
        "00_setup_and_architecture.ipynb",
        "01_models_and_prompting.ipynb",
        "02_responses_and_structured_outputs.ipynb",
        "03_rag_and_retrieval_security.ipynb",
        "04_agents_tools_and_mcp.ipynb",
        "05_evaluation_safety_and_red_team.ipynb",
        "06_observability_and_genaiops.ipynb",
        "07_capstone_release_gate.ipynb",
        "08_context_engineering_and_memory.ipynb",
        "09_foundry_a2a_and_handoffs.ipynb",
        "10_foundry_native_evaluation.ipynb",
        "11_mlflow_tracing_and_genai_evaluation.ipynb",
        "12_dual_otel_export_foundry_and_mlflow.ipynb",
    ]

    lesson_ids = set()
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        lesson = notebook["metadata"]["aai_lesson"]
        assert set(lesson) >= {
            "audience",
            "cleanup",
            "duration_minutes",
            "evidence",
            "execution_modes",
            "id",
            "level",
            "next_lesson",
            "objectives",
            "prerequisites",
        }
        assert lesson["duration_minutes"] > 0
        assert "offline" in lesson["execution_modes"]
        assert lesson["evidence"] and lesson["objectives"]
        assert lesson["id"] not in lesson_ids
        lesson_ids.add(lesson["id"])
        ids = [cell.get("id") for cell in notebook["cells"]]
        assert all(ids)
        assert len(ids) == len(set(ids))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert "load_session(" in source
        assert ".services.ai.azure.com/api/projects/" not in source
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile(
                "".join(cell.get("source", [])),
                f"{path.name}:code-cell-{index}",
                "exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )


def test_core_lab_cells_stay_focused():
    for path in CORE_NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            nonblank_lines = sum(bool(line.strip()) for line in cell.get("source", []))
            limit = 40 if cell["id"] == "load-config" else 25
            assert nonblank_lines <= limit, (
                f"{path.name}:{cell['id']} has {nonblank_lines} nonblank lines; "
                f"limit is {limit}"
            )


def test_core_labs_execute_twice_offline_from_clean_state(monkeypatch):
    config = CURRICULUM / "config" / "aai-platform.dev.example.yml"
    monkeypatch.setenv("AAI_PLATFORM_CONFIG", str(config))
    monkeypatch.chdir(ROOT)

    executions = []
    for _ in range(2):
        executions.append(
            {path.name: _execute_notebook(path) for path in CORE_NOTEBOOKS}
        )

    observed = executions[-1]
    assert (
        sum(
            cell["cell_type"] == "code"
            for path in CORE_NOTEBOOKS
            for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
        )
        == 40
    )
    assert observed["00_setup_and_architecture.ipynb"]["risk_evidence"]["complete"]
    model_lab = observed["01_models_and_prompting.ipynb"]
    assert model_lab["model_decision"].decision == "adopt"
    assert model_lab["underpowered_decision"].decision == "inconclusive"
    structured = observed["02_responses_and_structured_outputs.ipynb"]
    assert structured["safe_validation_view"]["valid"]["accepted"]
    assert not structured["safe_validation_view"]["oversized"]["accepted"]
    retrieval = observed["03_rag_and_retrieval_security.ipynb"]
    assert retrieval["retrieval_evidence"]["indirect_injection_gate"] == "pass"
    tools = observed["04_agents_tools_and_mcp.ipynb"]
    assert tools["published"].attempts == 2 and tools["replayed"].replayed
    evaluation = observed["05_evaluation_safety_and_red_team.ipynb"]
    assert evaluation["evaluation_decision"].decision == "adopt"
    operations = observed["06_observability_and_genaiops.ipynb"]
    assert operations["release_router"].active_version == "baseline-v1"
    capstone = observed["07_capstone_release_gate.ipynb"]
    assert capstone["release_decision"].decision == "inconclusive"
    assert capstone["complete_fixture_decision"].decision == "adopt"
    assert capstone["failed_fixture_decision"].decision == "reject"


def test_offline_lab_boundaries_fail_closed():
    risks = ("prompt injection",)
    incomplete = {
        "prompt injection": labs.RiskTreatment(
            owner="group:security", control="instruction boundary", verification=""
        )
    }
    assert not labs.assess_risk_treatments(risks, incomplete)["complete"]

    with pytest.raises(ValueError, match="limit"):
        labs.hybrid_retrieve("query", (), frozenset(), limit=0)

    backend = labs.SimulatedToolBackend({"publish_release": 3})
    gateway = labs.ToolGateway(
        {
            "publish_release": labs.ToolPolicy(
                frozenset({"release-owners"}), side_effect=True, max_attempts=2
            )
        },
        backend,
    )
    exhausted = gateway.execute(
        "publish_release",
        {"version": "v2"},
        caller_groups=frozenset({"release-owners"}),
        approved=True,
        idempotency_key="release-exhausted",
    )
    assert exhausted.status == "failed" and exhausted.attempts == 2
    assert not backend.committed_keys

    router = labs.SimulatedReleaseRouter({"baseline-v1"}, "baseline-v1")
    with pytest.raises(ValueError, match="known immutable"):
        router.rollback("unknown-v2")

    evidence = {
        "review": labs.EvidenceItem("pass", None, "group:reviewers"),
        "safety": labs.EvidenceItem("fail", "run://failed", "group:security"),
    }
    decision = labs.decide_evidence_map(evidence, ("review", "safety"))
    assert decision.decision == "reject"
    assert decision.failed == ("safety",) and decision.missing == ("review",)


def test_offline_model_selection_requires_aligned_threshold_evidence():
    thresholds = labs.ModelThresholds(0.9, 500, 1.0, 2)
    eligible = labs.ModelResult("a", 1.0, 100, 0.5, ("one", "two"))
    misaligned = labs.ModelResult("b", 1.0, 100, 0.5, ("one", "three"))
    ineligible = labs.ModelResult("b", 0.5, 900, 2.0, ("one", "two"))

    assert labs.select_model((), thresholds).decision == "inconclusive"
    assert (
        labs.select_model((eligible, misaligned), thresholds).decision == "inconclusive"
    )
    assert labs.select_model((ineligible,), thresholds).decision == "reject"
    assert not labs.citations_resolve({}, ())


def test_tool_and_evaluation_policies_reject_invalid_evidence():
    with pytest.raises(ValueError, match="max_attempts"):
        labs.ToolPolicy(frozenset({"owners"}), side_effect=True, max_attempts=0)

    gateway = labs.ToolGateway(
        {"read": labs.ToolPolicy(frozenset({"readers"}), False, 1)},
        labs.SimulatedToolBackend(),
    )
    blank_key = gateway.execute(
        "read",
        {},
        caller_groups=frozenset({"readers"}),
        approved=False,
        idempotency_key=" ",
    )
    assert blank_key.reason == "idempotency key is required"

    with pytest.raises(ValueError, match="same non-empty"):
        labs.calibrate_binary_judge({"one": True}, {})

    baseline = labs.EvaluationSummary("v1", "same", 0.8, 1.0, {}, ())
    improved = labs.EvaluationSummary("v2", "same", 0.9, 1.0, {}, ())
    thresholds = labs.EvaluationThresholds(0.85, 1.0, 0.8)
    calibrated = labs.JudgeCalibration(0.9, (), ())
    weak_calibration = labs.JudgeCalibration(0.5, (), ())

    different_data = labs.EvaluationSummary("v2", "different", 0.9, 1.0, {}, ())
    assert (
        labs.decide_evaluation(
            baseline, different_data, calibrated, thresholds
        ).decision
        == "inconclusive"
    )
    assert (
        labs.decide_evaluation(
            baseline, improved, weak_calibration, thresholds
        ).decision
        == "inconclusive"
    )
    unsafe = labs.EvaluationSummary("v2", "same", 0.9, 0.75, {}, ())
    assert (
        labs.decide_evaluation(baseline, unsafe, calibrated, thresholds).decision
        == "reject"
    )
    stronger_baseline = labs.EvaluationSummary("v1", "same", 0.9, 1.0, {}, ())
    unchanged = labs.EvaluationSummary("v2", "same", 0.9, 1.0, {}, ())
    assert (
        labs.decide_evaluation(
            stronger_baseline, unchanged, calibrated, thresholds
        ).decision
        == "inconclusive"
    )


def test_release_router_rejects_unknown_versions_and_blank_requests():
    with pytest.raises(ValueError, match="active version"):
        labs.SimulatedReleaseRouter({"baseline-v1"}, "unknown-v2")

    router = labs.SimulatedReleaseRouter({"baseline-v1"}, "baseline-v1")
    with pytest.raises(ValueError, match="must not be blank"):
        router.invoke(" ", correlation_id="trace-1", dependency_state="healthy")


def test_advanced_notebooks_are_opt_in_and_do_not_provision_a2a():
    sources = {}
    for path in ADVANCED_NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        sources[path.name] = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

    assert "RUN_CONNECTED = False" in sources["08_context_engineering_and_memory.ipynb"]
    assert "RUN_A2A_CONNECTED = False" in sources["09_foundry_a2a_and_handoffs.ipynb"]
    assert "RUN_FOUNDRY_EVAL = False" in sources["10_foundry_native_evaluation.ipynb"]
    assert (
        "RUN_MLFLOW_EVAL = False"
        in sources["11_mlflow_tracing_and_genai_evaluation.ipynb"]
    )
    assert (
        "RUN_DUAL_EXPORT = False"
        in sources["12_dual_otel_export_foundry_and_mlflow.ipynb"]
    )
    assert "canceled" in sources["10_foundry_native_evaluation.ipynb"]
    assert (
        "AZURE_EXPERIMENTAL_ENABLE_GENAI_TRACING"
        in sources["12_dual_otel_export_foundry_and_mlflow.ipynb"]
    )
    assert all("update_details(" not in source for source in sources.values())
    assert all(
        "project.agents.create_version(" not in source for source in sources.values()
    )


def test_advanced_notebooks_execute_twice_offline_without_network(
    monkeypatch,
):
    """Exercise every default advanced-lab cell in the certified extras lane."""

    pytest.importorskip(
        "agent_framework",
        reason="advanced Foundry execution runs in the provider-extras CI lane",
    )
    pytest.importorskip(
        "mlflow",
        reason="advanced Foundry execution runs in the provider-extras CI lane",
    )
    pytest.importorskip(
        "opentelemetry.sdk.trace",
        reason="advanced Foundry execution runs in the provider-extras CI lane",
    )

    def deny_network(*_args, **_kwargs):
        raise AssertionError("advanced offline notebooks attempted network access")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setenv(
        "AAI_PLATFORM_CONFIG",
        str(CURRICULUM / "config" / "aai-platform.dev.example.yml"),
    )
    monkeypatch.chdir(ROOT)

    previous_setup = sys.modules.get("foundry_curriculum_setup")
    executions: list[dict[str, dict[str, object]]] = []

    async def execute_all() -> dict[str, dict[str, object]]:
        observed: dict[str, dict[str, object]] = {}
        for path in ADVANCED_NOTEBOOKS:
            namespace = await _execute_async_notebook(path)
            observed[path.name] = namespace
            provider = namespace.get("offline_provider")
            if provider is not None:
                provider.shutdown()
        return observed

    try:
        for _ in range(2):
            executions.append(asyncio.run(execute_all()))
    finally:
        if previous_setup is None:
            sys.modules.pop("foundry_curriculum_setup", None)
        else:
            sys.modules["foundry_curriculum_setup"] = previous_setup

    observed = executions[-1]
    context = observed["08_context_engineering_and_memory.ipynb"]
    assert [item.context_id for item in context["envelope"].selected] == [
        "policy-current"
    ]
    routing = observed["09_foundry_a2a_and_handoffs.ipynb"]["routing_evidence"]
    assert all(item["expected"] == item["actual"] for item in routing)
    assert (
        observed["10_foundry_native_evaluation.ipynb"]["offline_evidence"].status
        == "completed"
    )
    assert (
        len(
            observed["11_mlflow_tracing_and_genai_evaluation.ipynb"][
                "evaluation_records"
            ]
        )
        == 20
    )
    dual = observed["12_dual_otel_export_foundry_and_mlflow.ipynb"]
    assert dual["foundry_shape"] == dual["mlflow_shape"]


def test_dual_export_offline_probe_preserves_native_sibling_topology():
    pytest.importorskip("opentelemetry.sdk.trace")
    notebook = json.loads(
        (
            CURRICULUM / "notebooks" / "12_dual_otel_export_foundry_and_mlflow.ipynb"
        ).read_text(encoding="utf-8")
    )
    cell = next(
        cell for cell in notebook["cells"] if cell["id"] == "dual-offline-proof"
    )
    namespace = {"__name__": "notebook_dual_export_probe"}

    exec(compile("".join(cell["source"]), "dual-offline-proof", "exec"), namespace)

    foundry_spans = namespace["foundry_probe"].get_finished_spans()
    invoke_span = next(span for span in foundry_spans if span.name == "invoke_agent")
    native_children = [
        span for span in foundry_spans if span.name in {"chat", "execute_tool"}
    ]
    assert [span.name for span in native_children] == [
        "chat",
        "execute_tool",
        "chat",
    ]
    assert all(
        span.parent.span_id == invoke_span.context.span_id for span in native_children
    )
    assert namespace["foundry_shape"] == namespace["mlflow_shape"]
    assert (
        invoke_span.attributes["aai.measurement_source"] == "synthetic_offline_topology"
    )


def test_dual_export_teaches_assurance_ownership_routing_and_review_loop():
    readme = (CURRICULUM / "README.md").read_text(encoding="utf-8")
    practices = (CURRICULUM / "CURRENT_PRACTICES.md").read_text(encoding="utf-8")
    notebook = (
        CURRICULUM / "notebooks" / "12_dual_otel_export_foundry_and_mlflow.ipynb"
    ).read_text(encoding="utf-8")
    combined = "\n".join((readme, practices, notebook))

    assert "authoritative assurance" in combined
    assert "Application Insights" in combined
    assert "hidden chain-of-thought" in combined
    assert "/v1/traces" in combined
    assert "x-mlflow-experiment-id" in combined
    assert "/api/2.0/otel/v1/traces" in combined
    assert "X-Databricks-UC-Table-Name" in combined
    assert "collector/gateway" in combined
    assert "renewable" in combined
    assert "invoke_agent" in combined
    assert "execute_tool" in combined
    assert "direct siblings" in combined or "direct_sibling_children" in combined
    assert "live-validate" in combined
    assert "EvaluationDataset" in combined
    assert "Feedback" in combined
    assert "Debug" in combined
    assert "dual-export/correlation smoke test" in combined
    assert "does not invoke Agent Framework" in combined
    assert "authenticated live validation" in combined


def test_evaluation_starter_has_twenty_cases_and_four_attacks():
    records = [
        json.loads(line)
        for line in (CURRICULUM / "data" / "evaluation_cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert len(records) == 20
    assert len({record["case_id"] for record in records}) == 20
    assert sum(record["category"] == "adversarial" for record in records) == 4
    assert all("expectations" in record for record in records)


@pytest.mark.parametrize("dataset_name", ("context_cases.jsonl", "a2a_cases.jsonl"))
def test_advanced_datasets_use_mlflow_standard_shape(dataset_name):
    records = [
        json.loads(line)
        for line in (CURRICULUM / "data" / dataset_name)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert len(records) == 8
    assert len({record["case_id"] for record in records}) == 8
    assert {record["split"] for record in records} == {"regression", "validation"}
    assert all(
        set(record) >= {"inputs", "expectations", "critical"} for record in records
    )
    assert any(record["critical"] for record in records)


def test_current_practices_cites_primary_foundry_and_mlflow_sources():
    guide = (CURRICULUM / "CURRENT_PRACTICES.md").read_text(encoding="utf-8")

    assert "last_verified: 2026-08-09" in guide
    assert "review_by: 2026-11-07" in guide
    assert "live_validation: required before production use" in guide
    for dependency in (
        "agent-framework-core: 1.12.1",
        "azure-ai-projects: 2.4.0",
        "mlflow: 3.15.1",
        "opentelemetry-sdk: 1.43.0",
    ):
        assert dependency in guide
    assert "2026-08-09" in guide
    assert "learn.microsoft.com/en-us/azure/foundry" in guide
    assert "mlflow.org/docs/latest" in guide
    assert "learn.microsoft.com/en-us/azure/databricks" in guide
    assert "Application Insights" in guide
    assert "backend synchronization" in guide
