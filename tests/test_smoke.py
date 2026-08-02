"""Credential-free regression tests for the platform's security boundaries."""

import ast
import asyncio
import importlib.util
import json
import os
import re
import runpy
import subprocess
import sys
from ast import PyCF_ALLOW_TOP_LEVEL_AWAIT
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

# The identifier-stamping map lives in the sync script; loading it here keeps the
# check and the writer from drifting apart.
_sync_spec = importlib.util.spec_from_file_location(
    "sync_template_shared", ROOT / "scripts" / "sync_template_shared.py"
)
sync_module = importlib.util.module_from_spec(_sync_spec)
_sync_spec.loader.exec_module(sync_module)
WORKFLOWS = ROOT / ".github" / "workflows"
SHA_PIN = re.compile(
    r"^\s*(?:-\s+)?uses:\s*[^@\s]+@([0-9a-f]{40})(?:\s+#.*)?$",
    re.MULTILINE,
)
USES = re.compile(r"^\s*(?:-\s+)?uses:\s*[^@\s]+@([^\s]+)", re.MULTILINE)
# Single source of truth for environment-specific identifiers. These tests
# cross-check every other occurrence against it so a clone that edits the
# fixture is pointed at each file that must agree.
IDENTIFIERS = json.loads((ROOT / "platform-identifiers.json").read_text())


def load_yaml(relative_path):
    with (ROOT / relative_path).open() as stream:
        return yaml.safe_load(stream)


def test_sample_notebook_runs(capsys):
    runpy.run_path(str(ROOT / "src" / "notebooks" / "sample_etl.py"))
    assert capsys.readouterr().out.strip().endswith("package import verified")


def test_learning_artifacts_have_one_contiguous_numbered_order():
    artifacts = sorted(
        path
        for path in (ROOT / "examples").iterdir()
        if re.fullmatch(r"\d{2}_[a-z0-9_]+\.(?:py|ipynb)", path.name)
    )

    assert [path.name[:2] for path in artifacts] == [
        f"{number:02d}" for number in range(15)
    ]
    assert all(
        not re.match(r"\d{2}_", helper)
        for helper in ("lifecycle_support.py", "notebook_setup.py")
    )


def test_all_numbered_example_notebooks_are_safe_clean_and_compilable():
    notebooks = sorted((ROOT / "examples").glob("[0-9][0-9]_*.ipynb"))
    assert [path.name for path in notebooks] == [
        "05_connected_setup.ipynb",
        "07_first_llm_call.ipynb",
        "08_tool_trajectory_evaluation.ipynb",
        "09_multi_turn_session_evaluation.ipynb",
        "10_layered_judges.ipynb",
        "11_cost_quality_tradeoff.ipynb",
        "12_agent_alignment_optimization.ipynb",
        "13_decision_and_promotion_lifecycle.ipynb",
        "14_platform_llm_operations.ipynb",
    ]

    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        cell_ids = [cell.get("id") for cell in notebook["cells"]]
        assert all(cell_ids)
        assert len(cell_ids) == len(set(cell_ids))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert "DATABRICKS_TOKEN" not in source
        assert "AZURE_CLIENT_SECRET" not in source
        assert "OPENAI_API_KEY" not in source
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile(
                "".join(cell.get("source", [])),
                f"{path.name}:code-cell-{index}",
                "exec",
                flags=PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )


def test_advanced_notebooks_preserve_release_guardrails():
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in (ROOT / "examples").glob("*.ipynb")
    }

    trajectory = sources["08_tool_trajectory_evaluation.ipynb"]
    assert "right-answer-wrong-trajectory" in trajectory
    assert "tool_trajectory_exact" in trajectory
    assert "TraceIntegration.MLFLOW_LANGCHAIN" in trajectory

    multi_turn = sources["09_multi_turn_session_evaluation.ipynb"]
    assert "mlflow.trace.session" not in multi_turn
    assert "mlflow.tracing.context(session_id=opaque_session_id)" in multi_turn
    assert "tag.aai.eval_batch" in multi_turn
    assert "predict_fn=" not in multi_turn

    judges = sources["10_layered_judges.ipynb"]
    assert 'source_id=\\"group:domain-reviewers\\"' in judges
    assert "MINIMUM_TOTAL_LABELS = 50" in judges
    assert '\\"report_only\\"' in judges

    cost = sources["11_cost_quality_tradeoff.ipynb"]
    assert "target_inference_cost_usd" in cost
    assert "evaluation_judge_cost_usd" in cost
    assert "cost_coverage" in cost
    assert "vendor model IDs" in cost

    optimization = sources["12_agent_alignment_optimization.ipynb"]
    assert "RUN_EXPERIMENTAL_OPTIMIZATION = False" in optimization
    assert "active_prompt = mlflow.genai.load_prompt(prompt_uri)" in optimization
    assert 'train_data=datasets[\\"optimizer_training\\"]' in optimization
    assert 'data=datasets[\\"held_out_release\\"]' in optimization
    assert "optimization_result.optimized_prompts[0]" in optimization
    assert "link_prompt_versions_to_trace(" in optimization
    assert "max_metric_calls" in optimization
    assert "set_prompt_alias" not in optimization


def test_advanced_notebooks_offer_governed_dataset_and_run_evidence():
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in (ROOT / "examples").glob("*.ipynb")
    }

    for notebook_name in (
        "07_first_llm_call.ipynb",
        "08_tool_trajectory_evaluation.ipynb",
        "09_multi_turn_session_evaluation.ipynb",
        "10_layered_judges.ipynb",
        "11_cost_quality_tradeoff.ipynb",
        "12_agent_alignment_optimization.ipynb",
    ):
        source = sources[notebook_name]
        assert "get_or_create_uc_evaluation_dataset(" in source
        assert "mlflow.log_input(" in source
        assert "description=(" in source

    assert (
        "fictional_earnings_summary_regression_v1" in sources["07_first_llm_call.ipynb"]
    )
    assert (
        "fictional_agent_tool_trajectory_regression_v1"
        in sources["08_tool_trajectory_evaluation.ipynb"]
    )
    assert (
        "fictional_multi_turn_session_regression_v1"
        in sources["09_multi_turn_session_evaluation.ipynb"]
    )
    assert "fictional_layered_judge_cases_v1" in sources["10_layered_judges.ipynb"]
    assert "fictional_judge_calibration_labels_v1" in sources["10_layered_judges.ipynb"]
    assert (
        "fictional_cost_quality_regression_v1"
        in sources["11_cost_quality_tradeoff.ipynb"]
    )
    optimization = sources["12_agent_alignment_optimization.ipynb"]
    assert 'dataset_name=f\\"fictional_{split}_v1\\"' in optimization
    for split in ("judge_calibration", "optimizer_training", "held_out_release"):
        assert f'\\"{split}\\"' in optimization


def test_advanced_notebooks_run_all_on_the_credential_free_default_path():
    pytest.importorskip("pandas")
    # 14 is connected-guarded like 05/07 and deliberately excluded here.
    for path in sorted((ROOT / "examples").glob("0[89]_*.ipynb")) + sorted(
        (ROOT / "examples").glob("1[0-3]_*.ipynb")
    ):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        namespace = {"__name__": f"notebook_{path.stem}"}
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            code = compile(
                "".join(cell.get("source", [])),
                f"{path.name}:code-cell-{index}",
                "exec",
            )
            exec(code, namespace)


def test_offline_example_runs_with_zero_credentials():
    environment = dict(os.environ)
    for name in (
        "AZURE_CLIENT_SECRET",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_TOKEN",
        "OPENAI_API_KEY",
    ):
        environment.pop(name, None)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "00_offline_hello_world.py")],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    assert "completed with zero credentials" in output
    assert "not-a-real-secret" not in output


def test_first_llm_notebook_is_valid_safe_and_output_free():
    notebook = json.loads(
        (ROOT / "examples" / "07_first_llm_call.ipynb").read_text(encoding="utf-8")
    )
    setup_helper_source = (ROOT / "examples" / "notebook_setup.py").read_text(
        encoding="utf-8"
    )
    assert notebook["nbformat"] == 4
    assert notebook["cells"]

    markdown_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    ]
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    markdown_source = "\n".join(markdown_cells)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    code_source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(cell_ids)
    assert len(cell_ids) == len(set(cell_ids))

    assert "prepare_notebook_environment(" in code_source
    assert "SEND_EVIDENCE_TO_DATABRICKS = False" in code_source
    assert '"databricks" if SEND_EVIDENCE_TO_DATABRICKS else "local"' in code_source
    assert "preflight_databricks(environment)" in code_source
    assert "importlib.reload(lifecycle_support)" in code_source
    assert 'context.providers.model("general-chat")' in setup_helper_source
    assert "subprocess.run(" not in code_source
    assert "workspace.serving_endpoints.list()" not in code_source
    assert "integration=TraceIntegration.MLFLOW_OPENAI" in source
    assert "capture_mode=TraceCaptureMode.FULL" in source
    assert "model.create_native_async_client()" in source
    assert "native_async_client.chat.completions.create(" in source
    assert 'stream_options={"include_usage": True}' in code_source
    assert "await stream.close()" in code_source
    assert "link_prompt_versions_to_trace(" in source
    assert 'application_span.set_attribute("mlflow.message.format", "openai")' in (
        source
    )
    assert 'application_span.set_outputs({"content": content})' in source
    assert "mlflow.update_current_trace(request_preview=rendered_prompt)" in source
    assert "mlflow.update_current_trace(response_preview=content)" in source
    assert '@traced(name=\\"earnings_summary.prompt_evaluation' not in source
    assert "mlflow.log_input(registered_dataset" in source
    assert 'trace_metadata.get("mlflow.sourceRun")' in code_source
    assert "client.search_traces(" in code_source
    assert "include_spans=False" in code_source
    assert 'trace_metadata.get("mlflow.trace.sizeStats"' in code_source
    assert 'lineage_tags.get("mlflow.linkedPrompts"' in code_source
    assert 'client.get_run(record["run_id"])' in code_source

    mermaid_blocks = re.findall(
        r"```mermaid\n(.*?)```",
        markdown_source,
        flags=re.DOTALL,
    )
    assert len(mermaid_blocks) == 5
    assert all("subgraph" not in block for block in mermaid_blocks)
    assert all(block.count("-->") <= 5 for block in mermaid_blocks)
    assert len(markdown_cells) > len(code_cells)
    assert markdown_source.lower().count("why this matters") >= 4
    assert markdown_source.lower().count("what risk it prevents") >= 4
    assert markdown_source.lower().count("what evidence we will collect") >= 4
    assert "fictional earnings packet" in markdown_source.lower()
    assert "Aster Ridge Systems" in markdown_source
    for source_id in (
        "ARS-FY25-Q2-RESULTS",
        "ARS-FY25-Q2-GUIDANCE",
        "ARS-FY25-Q2-CASH-RISK",
    ):
        assert source_id in markdown_source
    assert "investment advice" in markdown_source.lower()
    assert markdown_source.count("Diagram in words") == 5
    assert markdown_source.count("### Interpret the") >= 7
    assert "| v1 — evidence-only baseline | v2 — cited change |" in markdown_source
    assert "MLflow is the evidence system and data model" in markdown_source
    assert "Databricks can host its managed tracking service" in markdown_source
    assert "Unity Catalog is the governance layer" in markdown_source
    assert "publishing prompts to Unity Catalog does **not** automatically move" in (
        markdown_source
    )
    assert "Unity Catalog OpenTelemetry tables" in markdown_source
    assert "tracking URI" in markdown_source
    assert "registry URI" in markdown_source

    glossary_start = markdown_source.lower().find("glossary")
    assert glossary_start >= 0
    glossary = markdown_source[glossary_start:].lower()
    for term in (
        "prompt template",
        "prompt registry",
        "trace",
        "span",
        "run",
        "experiment",
    ):
        assert term in glossary

    for checkpoint in (
        "SETUP PASSED",
        "PREFLIGHT PASSED",
        "PROMPT VERSIONS READY",
        "A/B CALLS SUCCEEDED",
        "TRACES VERIFIED",
        "COMPARISON RECORDED",
    ):
        assert checkpoint in source

    assert "earnings_summary" in source
    assert "baseline-earnings-summary-prompt-v1" in source
    assert "change-cited-earnings-summary-prompt-v2" in source
    assert "fictional-earnings-summary-regression-v1" in source
    assert "earnings_excerpt" in code_source
    assert "source_id" in code_source
    assert "len(CASES) * 2" in code_source
    assert "fact_coverage" in code_source
    assert "citation" in code_source
    assert "recommendation_policy" in code_source
    assert "latency" in code_source
    assert "input_tokens" in code_source
    assert "output_tokens" in code_source
    assert "cost_coverage" in code_source

    assert code_source.count(".load(") >= 2
    assert code_source.count("version=int(") >= 2
    assert "loaded_baseline" in code_source
    assert "loaded_change" in code_source
    assert "PUBLISH_PROMPTS_TO_DATABRICKS = False" in code_source
    assert "This exploratory notebook moves no alias" in markdown_source
    assert "PromptManager.set_alias()" in markdown_source
    assert "databricks-uc" in source
    assert "cross-store" in source.lower()
    assert "FULL DATABRICKS EVIDENCE MODE ALREADY ACTIVE" in code_source
    assert "experiment, runs, traces, and exact prompt versions" in markdown_source
    assert 'local_mlflow_dir = root / ".aai" / "local"' in setup_helper_source
    assert 'os.environ["MLFLOW_TRACKING_URI"] = tracking_uri' in setup_helper_source
    assert "inconclusive" in source.lower()
    assert "run full evaluation" in source.lower()
    assert "What this proved" in markdown_source
    assert "did not prove" in markdown_source.lower()
    assert "Troubleshooting" in markdown_source
    assert "`make local-ui`" in source

    assert "PREFLIGHT PASSED" in source
    assert "A Databricks CLI profile" in source
    assert "workspace.current_user.me()" in setup_helper_source
    assert "workspace.serving_endpoints.get(deployment)" in setup_helper_source
    assert "_ready_chat_endpoints(workspace)" in setup_helper_source
    assert "CAN_QUERY" in source
    assert "DATABRICKS_TOKEN" not in source
    assert "AZURE_CLIENT_SECRET" not in source

    assert "model.generate(" not in code_source
    for intent_comment in (
        "Registration is idempotent",
        "Render the registry-loaded object",
        "Missing cost is unknown, never zero",
        "cannot authorize release",
        "Only prompt storage changes here",
    ):
        assert intent_comment in code_source
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile(
            "".join(cell.get("source", [])),
            f"07_first_llm_call.ipynb:code-cell-{index}",
            "exec",
            flags=PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )


def test_first_llm_trace_displays_content_without_losing_telemetry():
    notebook = json.loads(
        (ROOT / "examples" / "07_first_llm_call.ipynb").read_text(encoding="utf-8")
    )
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("id") == "ab-code"
    )
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"response_text", "invoke_prompt"}
    ]

    class FakeSpan:
        def __init__(self):
            self.attributes = {}
            self.inputs = None
            self.outputs = None

        def set_attribute(self, name, value):
            self.attributes[name] = value

        def set_inputs(self, value):
            self.inputs = value

        def set_outputs(self, value):
            self.outputs = value

    class FakeMlflow:
        def __init__(self):
            self.span = FakeSpan()
            self.trace_updates = []

        @contextmanager
        def start_span(self, **kwargs):
            assert kwargs == {
                "name": "earnings_summary.prompt_evaluation",
                "span_type": "CHAIN",
            }
            yield self.span

        def update_current_trace(self, **kwargs):
            self.trace_updates.append(kwargs)

    class FakeStream:
        def __init__(self):
            self.closed = False
            self.events = iter(
                [
                    SimpleNamespace(
                        model="served-model",
                        usage=SimpleNamespace(
                            model_dump=lambda: {
                                "prompt_tokens": 8,
                                "completion_tokens": 3,
                            }
                        ),
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="assistant answer")
                            )
                        ],
                    )
                ]
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration as error:
                raise StopAsyncIteration from error

        async def close(self):
            self.closed = True

    stream = FakeStream()
    native_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: _async_value(stream))
        )
    )
    fake_mlflow = FakeMlflow()
    namespace = {
        "mlflow": fake_mlflow,
        "model": SimpleNamespace(model="configured-model"),
        "ctx": SimpleNamespace(tags=SimpleNamespace()),
        "monotonic": __import__("time").monotonic,
        "set_trace_resource_context": lambda context: None,
    }
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            "07_first_llm_call.ipynb:trace-functions",
            "exec",
        ),
        namespace,
    )

    result = asyncio.run(
        namespace["invoke_prompt"](
            native_async_client=native_client,
            rendered_prompt="synthetic user request",
        )
    )

    assert fake_mlflow.span.inputs == {
        "messages": [{"role": "user", "content": "synthetic user request"}]
    }
    assert fake_mlflow.span.outputs == {"content": "assistant answer"}
    assert fake_mlflow.span.attributes["mlflow.message.format"] == "openai"
    assert fake_mlflow.trace_updates == [
        {"request_preview": "synthetic user request"},
        {"response_preview": "assistant answer"},
    ]
    assert result["content"] == "assistant answer"
    assert result["usage"] == {"prompt_tokens": 8, "completion_tokens": 3}
    assert result["model"] == "served-model"
    assert result["latency_ms"] >= 0
    assert stream.closed


async def _async_value(value):
    return value


def test_setup_notebook_is_explicit_output_free_and_uses_shared_helper():
    notebook = json.loads(
        (ROOT / "examples" / "05_connected_setup.ipynb").read_text(encoding="utf-8")
    )
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    code_source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)

    assert notebook["nbformat"] == 4
    assert "prepare_notebook_environment(repo_root)" in code_source
    assert "preflight_databricks(environment)" in code_source
    assert "makes no LLM calls and publishes no prompts" in source
    assert "do not use `%run`" in source
    assert "SETUP PASSED" in source
    assert "PREFLIGHT PASSED" in source
    assert "07_first_llm_call.ipynb" in source
    assert "DATABRICKS_TOKEN" not in source
    assert "AZURE_CLIENT_SECRET" not in source
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile(
            "".join(cell.get("source", [])),
            f"05_connected_setup.ipynb:code-cell-{index}",
            "exec",
        )


def test_connected_first_call_uses_stable_adapter_and_bounded_sdk_trace():
    source = (ROOT / "examples" / "06_connected_first_call.py").read_text(
        encoding="utf-8"
    )

    assert "ctx = bootstrap()" in source
    assert 'ctx.providers.model("general-chat")' in source
    assert "response = model.generate(" in source
    assert "integration=TraceIntegration.SDK" in source
    assert "model.native_client" not in source
    assert "create_native_async_client" not in source
    assert '"aai.cost_status": "unknown"' in source
    assert 'run_name="connected-general-chat-grounded-summary-baseline"' in source


def test_connected_setup_notebook_is_diagnostic_and_output_free():
    notebook = json.loads(
        (ROOT / "examples" / "05_connected_setup.ipynb").read_text(encoding="utf-8")
    )
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert 'import_module("examples.notebook_setup")' in source
    assert "prepare_notebook_environment(" in source
    assert "preflight_databricks(" in source
    assert "model.generate(" not in source
    assert "chat.completions.create(" not in source
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)


def test_dev_target_is_pinned_to_dev_workspace():
    bundle = load_yaml("databricks.yml")
    host = bundle["targets"]["dev"]["workspace"]["host"]
    # workspace.host must stay a literal (the Databricks CLI forbids variable
    # interpolation in authentication fields), so this cross-check keeps it in
    # sync with the identifiers fixture.
    assert host == IDENTIFIERS["databricks_host"]
    assert host.startswith("https://") and host.endswith(".azuredatabricks.net")


def test_sample_job_uses_constrained_job_compute_policy():
    bundle = load_yaml("databricks.yml")
    resources = load_yaml("resources/sample_job.yml")
    cluster = resources["resources"]["jobs"]["aai_dbx_base_template_sample"][
        "job_clusters"
    ][0]["new_cluster"]
    assert cluster["policy_id"] == "${var.job_compute_policy_id}"
    assert (
        bundle["variables"]["job_compute_policy_id"]["default"]
        == IDENTIFIERS["job_compute_policy_id"]
    )
    assert cluster["num_workers"] == 1
    assert cluster["spark_version"] == "18.0.x-scala2.13"
    assert "spark_conf" not in cluster


def _discovered_templates():
    templates_dir = ROOT / "templates"
    return sorted(
        entry
        for entry in templates_dir.iterdir()
        if entry.is_dir() and (entry / "databricks_template_schema.json").is_file()
    )


def test_identifier_fixture_is_the_single_source_of_truth():
    """Every other file holding an environment identifier must agree with
    platform-identifiers.json; a clone edits the fixture and this test lists
    each remaining literal that must follow."""
    templates = _discovered_templates()
    assert templates, "no bundle templates discovered"

    # The stamping map lives in the sync script so the check cannot drift from
    # the thing that writes the values. `aai_core_pip_source` is in it too: it
    # is the one default that points at a *repository*, so a clone that misses
    # it makes every generated project's CI install the SDK from upstream.
    drift = sync_module.schema_default_drift()
    assert not drift, "run `make sync-templates`: " + "; ".join(drift)
    stamped = {prop for _, prop, _ in sync_module.planned_schema_defaults()}
    assert stamped == set(sync_module.IDENTIFIER_DEFAULTS), (
        "every identifier-owned schema property must exist in every template; "
        f"stamped={sorted(stamped)}"
    )

    verify = (ROOT / "scripts" / "cloud-verify.sh").read_text()
    assert "platform-identifiers.json" in verify
    for value in (
        IDENTIFIERS["azure_tenant_id"],
        IDENTIFIERS["azure_subscription_id"],
        IDENTIFIERS["databricks_host"],
    ):
        assert (
            value not in verify
        ), "cloud-verify.sh must read the fixture, not inline ids"


#: Every key the fixture must carry. A downstream clone keeps its own copy of
#: platform-identifiers.json (docs/enterprise-clone-runbook.md recommends a
#: `merge=keepours` driver so upstream merges never prompt on it) — and the cost
#: of that is that a *new* key added upstream would be silently dropped there.
#: This list is the guard: it travels with the merge, so the clone fails loudly
#: on the next test run instead of rendering an empty default into a command.
REQUIRED_IDENTIFIER_KEYS = {
    "azure_tenant_id",
    "azure_subscription_id",
    "databricks_host",
    "job_compute_policy_id",
    "sdk_artifact_volume",
    "sdk_pip_source",
    "template_repo",
}


def test_identifier_fixture_carries_every_required_key():
    present = {key for key in IDENTIFIERS if not key.startswith("$")}
    missing = REQUIRED_IDENTIFIER_KEYS - present
    assert not missing, (
        "platform-identifiers.json is missing key(s) that this version of the "
        "repository requires: " + ", ".join(sorted(missing)) + ". A clone that "
        "keeps its own fixture must add them with its own values."
    )
    for key in sorted(REQUIRED_IDENTIFIER_KEYS):
        assert str(IDENTIFIERS[key]).strip(), f"{key} is empty"


#: Prose is the fourth copy of the identifier values and the one nothing used to
#: check, which is how the SDK artifact volume in AGENTS.md drifted away from the
#: fixture while every test stayed green. Markdown may *describe* these values but
#: must not restate them, so a clone never has to hunt through docs.
_MARKDOWN_FORBIDDEN = (
    "azure_tenant_id",
    "azure_subscription_id",
    "databricks_host",
    "job_compute_policy_id",
    "sdk_artifact_volume",
)
# The audit report deliberately quotes the drift it found, and the clone runbook
# needs to name the fixture keys it walks you through.
_MARKDOWN_EXEMPT = {"docs/platform-audit.md"}


def test_documented_bundle_init_never_hardcodes_a_repository():
    """`databricks bundle init <url>` decides which repository a developer's next
    project comes from. A clone whose docs still named the upstream URL would send
    every developer upstream, so the documented form must resolve it at run time."""
    offenders = []
    for path in sorted(ROOT.glob("**/*.md")):
        if any(part in {".claude", ".git", ".venv"} for part in path.parts):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in _MARKDOWN_EXEMPT:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            # Only actual commands, not prose that mentions `bundle init`.
            stripped = line.strip()
            if not stripped.startswith("databricks bundle init"):
                continue
            arguments = stripped[len("databricks bundle init") :].split()
            if arguments and arguments[0].startswith(("http://", "https://", "git@")):
                offenders.append(f"{relative}: {stripped}")
    assert not offenders, (
        "documented `bundle init` must take the repository from "
        "`platform-identifiers.json` (source scripts/platform-env.sh, then use "
        '"$AAI_TEMPLATE_REPO"): ' + "; ".join(offenders)
    )


def test_markdown_does_not_restate_environment_identifiers():
    offenders = []
    for path in sorted(ROOT.glob("**/*.md")):
        if any(
            part in {".claude", ".git", ".venv", "node_modules"} for part in path.parts
        ):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in _MARKDOWN_EXEMPT:
            continue
        text = path.read_text(encoding="utf-8")
        for key in _MARKDOWN_FORBIDDEN:
            if IDENTIFIERS[key] in text:
                offenders.append(f"{relative} restates {key}")
    assert not offenders, (
        "documentation must point at platform-identifiers.json rather than copy it "
        "(a clone would otherwise have to edit prose too): " + "; ".join(offenders)
    )


def test_bundle_and_compute_use_required_platform_tags():
    bundle = load_yaml("databricks.yml")
    resources = load_yaml("resources/sample_job.yml")
    bundle_tags = bundle["targets"]["dev"]["presets"]["tags"]
    compute_tags = resources["resources"]["jobs"]["aai_dbx_base_template_sample"][
        "job_clusters"
    ][0]["new_cluster"]["custom_tags"]
    required = {
        "application",
        "project",
        "environment",
        "team",
        "owner_group",
        "cost_center",
        "data_classification",
        "lifecycle",
        "tag_schema_version",
    }
    assert required.issubset(bundle_tags)
    assert required.issubset(compute_tags)


def test_all_github_actions_are_commit_pinned():
    for workflow in WORKFLOWS.glob("*.yml"):
        text = workflow.read_text()
        references = USES.findall(text)
        pins = SHA_PIN.findall(text)
        assert references, f"{workflow.name} has no action references"
        assert len(pins) == len(
            references
        ), f"{workflow.name} contains a mutable action reference"


def test_pr_ci_is_credential_free():
    text = (WORKFLOWS / "ci.yml").read_text()
    workflow = yaml.safe_load(text)
    assert workflow["permissions"] == {"contents": "read"}
    assert all(
        not reference.lower().startswith("azure/login@")
        for reference in USES.findall(text)
    )
    assert "${{ secrets." not in text


def test_credentialed_jobs_do_not_use_github_environments_or_secrets():
    for name in ("auth-smoke.yml", "deploy.yml", "publish-sdk.yml"):
        text = (WORKFLOWS / name).read_text()
        workflow = yaml.safe_load(text)
        workflow_permissions = workflow.get("permissions", {})
        credentialed_jobs = [
            job
            for job in workflow["jobs"].values()
            if (
                workflow_permissions.get("id-token") == "write"
                or job.get("permissions", {}).get("id-token") == "write"
            )
        ]
        assert credentialed_jobs
        assert "${{ secrets." not in text
        for job in workflow["jobs"].values():
            assert "environment" not in job


def test_cloud_environment_is_reproducible_and_credential_free():
    setup = (ROOT / "scripts" / "codex-cloud-setup.sh").read_text()
    maintenance = (ROOT / "scripts" / "codex-cloud-maintenance.sh").read_text()
    verify = (ROOT / "scripts" / "cloud-verify.sh").read_text()
    ci = (WORKFLOWS / "ci.yml").read_text()

    for version in ("0.8.23", "2.88.0"):
        assert version in setup

    assert "sha256sum --check" in setup
    assert "codex-cloud-setup.sh" in maintenance
    assert "./scripts/cloud-verify.sh" in ci
    assert "AZURE_CLIENT_SECRET" in verify
    assert "DATABRICKS_TOKEN" in verify
    assert "azure/login" not in verify.lower()
    assert "az login" not in verify.lower()


def test_repository_setup_does_not_provision_infrastructure():
    setup_paths = (
        "Makefile",
        ".gitignore",
        ".vscode/extensions.json",
        ".github/workflows/ci.yml",
        "scripts/codex-cloud-setup.sh",
        "scripts/cloud-verify.sh",
        "scripts/pre-commit.sh",
        "scripts/pre-push.sh",
    )
    for relative_path in setup_paths:
        text = (ROOT / relative_path).read_text()
        assert "terraform" not in text.lower(), relative_path

    assert not (ROOT / "infra").exists()


def test_databricks_cli_version_is_in_lockstep_everywhere():
    """The Codex setup pin and every databricks/setup-cli reference (this repo
    and the template's generated workflows) must agree, or bundle behavior
    diverges between local/Codex and CI."""

    setup = (ROOT / "scripts" / "codex-cloud-setup.sh").read_text()
    script_version = re.search(r'DATABRICKS_CLI_VERSION="([0-9.]+)"', setup).group(1)

    workflow_files = list(WORKFLOWS.glob("*.yml"))
    for template in _discovered_templates():
        workflow_files.extend(
            (template / "template" / ".github" / "workflows").glob("*.yml")
        )
    pins = []
    for workflow in workflow_files:
        pins.extend(
            re.findall(
                r"databricks/setup-cli@[0-9a-f]{40}\s+#\s*v([0-9.]+)",
                workflow.read_text(),
            )
        )
    assert pins, "no databricks/setup-cli pins found"
    assert set(pins) == {script_version}
