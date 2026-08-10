"""Credential-free regression tests for the platform's security boundaries."""

import ast
import asyncio
import importlib.util
import json
import os
import re
import runpy
import socket
import subprocess
import sys
from ast import PyCF_ALLOW_TOP_LEVEL_AWAIT
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

ADVANCED_SUPPORT_BY_NOTEBOOK = {
    "07_first_llm_call.ipynb": ("connected_llm.py",),
    "08_tool_trajectory_evaluation.ipynb": ("agent_assurance.py",),
    "09_multi_turn_session_evaluation.ipynb": ("agent_assurance.py",),
    "10_layered_judges.ipynb": ("agent_assurance.py",),
    "11_cost_quality_tradeoff.ipynb": ("cost_quality.py",),
    "12_agent_alignment_optimization.ipynb": ("optimization.py",),
    "15_compare_and_select_llms.ipynb": ("model_selection.py",),
}


def advanced_implementation_source(notebook_name):
    """Return a thin lesson together with the mechanics it invokes."""

    sources = [(ROOT / "examples" / notebook_name).read_text(encoding="utf-8")]
    sources.extend(
        (ROOT / "examples" / "support" / helper).read_text(encoding="utf-8")
        for helper in ADVANCED_SUPPORT_BY_NOTEBOOK[notebook_name]
    )
    return "\n".join(sources)


def deny_example_credentials_and_network(monkeypatch):
    for name in (
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    def unexpected_network(*args, **kwargs):
        raise AssertionError(f"credential-free example attempted network I/O: {args}")

    monkeypatch.setattr(socket, "create_connection", unexpected_network)


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
        f"{number:02d}" for number in range(16)
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
        "15_compare_and_select_llms.ipynb",
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


def test_advanced_notebook_cells_keep_one_teaching_step_visible():
    required_metadata = {
        "id",
        "audience",
        "level",
        "prerequisites",
        "duration_minutes",
        "execution_modes",
        "objectives",
        "evidence",
        "cleanup",
        "next_lesson",
    }
    for notebook_name in ADVANCED_SUPPORT_BY_NOTEBOOK:
        path = ROOT / "examples" / notebook_name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert required_metadata <= notebook["metadata"]["aai_lesson"].keys()
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            nonblank_lines = sum(
                bool(line.strip()) for line in "".join(cell["source"]).splitlines()
            )
            tags = cell.get("metadata", {}).get("tags", [])
            maximum = 40 if "setup" in tags or cell["id"] == "setup-code" else 25
            assert nonblank_lines <= maximum, (
                f"{path.name}:{cell['id']} has {nonblank_lines} nonblank lines; "
                f"maximum is {maximum}"
            )


def test_model_selection_workshop_has_the_required_interactive_pattern():
    notebook = json.loads(
        (ROOT / "examples" / "15_compare_and_select_llms.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    markdown_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )

    assert markdown_source.startswith(
        "# Comparing and Selecting LLMs for Enterprise Processes"
    )
    assert "Estimated duration:" in markdown_source
    assert "## Module Guide" in markdown_source
    assert "## Learning Objectives" in markdown_source
    assert "%pip install" in source
    assert "get_ipython().run_line_magic" in source
    assert "configured keyless identity" in source
    assert "keyvault://" in source
    assert "databricks-secret://" in source
    assert "simulated_offline_fixture" in source
    assert markdown_source.count("**Practical Application.**") == 4

    required_headings = (
        "Routing 2 Models through a Golden Dataset",
        "Automating Side-by-Side Scoring with LLM-as-a-Judge",
        "Evaluating TCO & Token Economics",
        "Enterprise Governance & Liability Checks",
    )
    assert all(heading in markdown_source for heading in required_headings)

    concept_indexes = []
    for index, cell in enumerate(notebook["cells"]):
        tags = cell.get("metadata", {}).get("tags", [])
        if "concept" in tags:
            concept_indexes.append(index)
            concept_source = "".join(cell.get("source", []))
            assert "| Strategy | Best use case | Weak point |" in concept_source
    assert len(concept_indexes) == 4

    expected_sequence = (
        "concept",
        "working-example",
        "exercise",
        "reference-solution",
        "verification",
    )
    for concept_index in concept_indexes:
        cells = notebook["cells"][concept_index : concept_index + 5]
        assert len(cells) == len(expected_sequence)
        for cell, expected_tag in zip(cells, expected_sequence, strict=True):
            assert expected_tag in cell.get("metadata", {}).get("tags", [])

    solutions = [
        cell
        for cell in notebook["cells"]
        if "reference-solution" in cell.get("metadata", {}).get("tags", [])
    ]
    verifications = [
        cell
        for cell in notebook["cells"]
        if "verification" in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(solutions) == len(verifications) == 4
    assert "TODO" not in source
    assert "NotImplementedError" not in source
    assert all(
        "".join(cell.get("source", [])).startswith("# Verification -- do not modify")
        and "assert " in "".join(cell.get("source", []))
        for cell in verifications
    )

    workshop = json.loads(
        (
            ROOT
            / "examples"
            / "workshops"
            / "15_compare_and_select_llms_exercises.ipynb"
        ).read_text(encoding="utf-8")
    )
    workshop_stubs = [
        cell
        for cell in workshop["cells"]
        if "exercise-stub" in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(workshop_stubs) == 4
    assert all(
        "TODO" in "".join(cell.get("source", []))
        and "NotImplementedError" in "".join(cell.get("source", []))
        for cell in workshop_stubs
    )


def test_model_selection_working_examples_execute_without_credentials(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    deny_example_credentials_and_network(monkeypatch)
    notebook = json.loads(
        (ROOT / "examples" / "15_compare_and_select_llms.ipynb").read_text(
            encoding="utf-8"
        )
    )
    working_cells = [
        cell
        for cell in notebook["cells"]
        if "working-example" in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(working_cells) == 4
    for repeat in range(2):
        for cell in working_cells:
            namespace = {"__name__": f"notebook_{cell['id']}_{repeat}"}
            exec(
                compile(
                    "".join(cell.get("source", [])),
                    f"15_compare_and_select_llms.ipynb:{cell['id']}",
                    "exec",
                ),
                namespace,
            )

    solution_indexes = [
        index
        for index, cell in enumerate(notebook["cells"])
        if "reference-solution" in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(solution_indexes) == 4
    for index in solution_indexes:
        namespace = {"__name__": f"notebook_{notebook['cells'][index]['id']}"}
        for cell in notebook["cells"][index : index + 2]:
            exec(
                compile(
                    "".join(cell.get("source", [])),
                    f"15_compare_and_select_llms.ipynb:{cell['id']}",
                    "exec",
                ),
                namespace,
            )


def test_advanced_notebooks_preserve_release_guardrails():
    sources = {
        path.name: advanced_implementation_source(path.name)
        for path in (ROOT / "examples").glob("*.ipynb")
        if path.name in ADVANCED_SUPPORT_BY_NOTEBOOK
    }

    trajectory = sources["08_tool_trajectory_evaluation.ipynb"]
    assert "right-answer-wrong-trajectory" in trajectory
    assert "correct-tool-failed-safe-fallback" in trajectory
    assert "tool_trajectory_exact" in trajectory
    assert "decision_action_consistency" in trajectory
    assert "decision_tool_appropriateness" in trajectory
    assert "safe_fallback_observed" in trajectory
    assert "assurance_report" in trajectory
    assert "operations_evidence" in trajectory
    assert "not a fabricated MLflow trace" in trajectory
    assert "does not claim" in trajectory
    assert "default recovery" in trajectory
    assert "mlflow.start_span(" not in trajectory
    assert "@mlflow.trace" not in trajectory
    assert "TraceIntegration.MLFLOW_LANGCHAIN" in trajectory

    multi_turn = sources["09_multi_turn_session_evaluation.ipynb"]
    assert "mlflow.trace.session" not in multi_turn
    assert "mlflow.tracing.context(session_id=opaque_session_id)" in multi_turn
    assert "tag.aai.eval_batch" in multi_turn
    assert "predict_fn=" not in multi_turn

    judges = sources["10_layered_judges.ipynb"]
    assert 'source_id="group:domain-reviewers"' in judges
    assert "MINIMUM_TOTAL_LABELS = 50" in judges
    assert '"report_only"' in judges

    cost = sources["11_cost_quality_tradeoff.ipynb"]
    assert "target_inference_cost_usd" in cost
    assert "evaluation_judge_cost_usd" in cost
    assert "cost_coverage" in cost
    assert "vendor model IDs" in cost

    optimization = sources["12_agent_alignment_optimization.ipynb"]
    assert "RUN_EXPERIMENTAL_OPTIMIZATION = False" in optimization
    assert "active_prompt = mlflow.genai.load_prompt(prompt_uri)" in optimization
    assert 'train_data=datasets["optimizer_training"]' in optimization
    assert 'dataset=datasets["held_out_release"]' in optimization
    assert "data=dataset" in optimization
    assert "optimization_result.optimized_prompts[0]" in optimization
    assert "link_prompt_versions_to_trace(" in optimization
    assert "max_metric_calls" in optimization
    assert "set_prompt_alias" not in optimization

    # Lesson 14 intentionally stays self-contained rather than using the
    # refactored support-module pattern enforced for the advanced notebooks.
    operations = (ROOT / "examples" / "14_platform_llm_operations.ipynb").read_text(
        encoding="utf-8"
    )
    operations_notebook = json.loads(operations)
    provenance_source = next(
        "".join(cell["source"])
        for cell in operations_notebook["cells"]
        if "provenance_stamp =" in "".join(cell.get("source", []))
    )
    provenance_tree = ast.parse(provenance_source)
    provenance_assignment = next(
        node
        for node in provenance_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "provenance_stamp"
            for target in node.targets
        )
    )
    provenance_stamp = ast.literal_eval(provenance_assignment.value)
    generated_stamp = json.loads(
        (ROOT / "templates/prompt-app/template/.aai-template.json.tmpl").read_text()
    )
    generated_stamp["generated_with"] = {
        "project_name": "fictional-earnings",
        "application_name": "earnings-summary",
        "team": "fictional-app-team",
        "model_provider": "databricks",
        "prompt_name": "earnings_summary",
        "aai_core_version": "0.4.0",
    }
    assert provenance_stamp == generated_stamp


def test_tool_trajectory_fixture_separates_decisions_execution_and_assessment():
    pytest.importorskip("pandas")
    notebook = json.loads(
        (ROOT / "examples" / "08_tool_trajectory_evaluation.ipynb").read_text(
            encoding="utf-8"
        )
    )
    namespace = {"__name__": "notebook_tool_trajectory_assurance"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        exec(
            compile(
                "".join(cell.get("source", [])),
                f"08_tool_trajectory_evaluation.ipynb:code-cell-{index}",
                "exec",
            ),
            namespace,
        )

    report = namespace["trajectory_report"].set_index("case_id")
    assurance_report = namespace["assurance_report"].set_index("case_id")
    wrong_path = report.loc["right-answer-wrong-trajectory"]
    safe_fallback = report.loc["correct-tool-failed-safe-fallback"]

    assert len(namespace["EVAL_CASES"]) == 3
    assert report["final_answer_correct"].tolist() == [True, True, True]
    assert report["decision_action_consistency"].tolist() == [True, True, True]
    assert report["decision_tool_appropriateness"].tolist() == [True, False, True]
    assert report["tool_trajectory_exact"].tolist() == [True, False, True]
    assert report["tool_execution_succeeded"].tolist() == [True, True, False]
    assert report["safe_fallback_observed"].tolist() == [True, True, True]
    assert report["operations_evidence"].tolist() == [
        "partial",
        "partial",
        "partial",
    ]
    assert assurance_report["outcome_assessment"].tolist() == [
        "PASS",
        "PASS",
        "PASS",
    ]
    assert assurance_report["behavior_assessment"].tolist() == [
        "PASS",
        "FAIL",
        "PASS",
    ]
    assert assurance_report["operations_assessment"].tolist() == [
        "PASS",
        "PASS",
        "FAIL",
    ]
    assert bool(wrong_path["final_answer_correct"])
    assert bool(wrong_path["decision_action_consistency"])
    assert not bool(wrong_path["decision_tool_appropriateness"])
    assert not bool(wrong_path["tool_trajectory_exact"])
    assert not bool(safe_fallback["tool_execution_succeeded"])
    assert bool(safe_fallback["decision_tool_appropriateness"])
    assert bool(safe_fallback["safe_fallback_observed"])

    reordered_fallback_case = json.loads(json.dumps(namespace["EVAL_CASES"][2]))
    reordered_events = reordered_fallback_case["observed"]["trajectory_events"]
    reordered_events[2], reordered_events[3] = reordered_events[3], reordered_events[2]
    reordered_score = namespace["score_case"](reordered_fallback_case)
    assert not reordered_score["safe_fallback_observed"]
    assert reordered_score["behavior_assessment"] == "FAIL"

    no_operations_case = json.loads(json.dumps(namespace["EVAL_CASES"][0]))
    no_operations_case["observed"]["tool_results"] = []
    no_operations_score = namespace["score_case"](no_operations_case)
    assert no_operations_score["operations_evidence"] == "unknown"
    assert no_operations_score["operations_assessment"] == "UNKNOWN"
    assert not no_operations_score["tool_execution_succeeded"]

    for case in namespace["EVAL_CASES"]:
        decision_types = [
            decision["decision_type"] for decision in case["agent_decisions"]
        ]
        if case["case_id"] == "correct-tool-failed-safe-fallback":
            assert decision_types == ["tool_selection", "fallback", "answer_readiness"]
            assert case["observed"]["tool_results"] == [
                {
                    "name": "lookup_earnings_source",
                    "status": "error",
                    "error_type": "SourceUnavailable",
                }
            ]
            fallback_reason = case["agent_decisions"][1]["reason"]
            assert "SourceUnavailable" in fallback_reason
            assert "succeed" not in fallback_reason.lower()
        else:
            assert decision_types == ["tool_selection", "evidence_sufficiency"]
        assert all(value is None for value in case["observed"]["operations"].values())


def test_agent_behavior_assurance_teaching_keeps_evidence_layers_separate():
    sources = (
        (ROOT / "README.md").read_text(encoding="utf-8"),
        (ROOT / "examples" / "README.md").read_text(encoding="utf-8"),
        (ROOT / "examples" / "08_tool_trajectory_evaluation.ipynb").read_text(
            encoding="utf-8"
        ),
    )

    for source in sources:
        assert "Outcome" in source
        assert "Behavior" in source
        assert "Operations" in source
        assert "Optional internal diagnostics" in source
        assert "Assessment" in source
        assert "chain-of-thought" in source

    for source in sources[:2]:
        assert "Code tells us what could happen" in source
        assert re.search(r"runtime\s+agent decision", source)
        assert "lifecycle" in source


def test_assurance_docs_keep_mlflow_authoritative_and_production_cases_reviewed():
    sources = (
        (ROOT / "README.md").read_text(encoding="utf-8"),
        (ROOT / "docs" / "developer-guide.md").read_text(encoding="utf-8"),
        (ROOT / "examples" / "README.md").read_text(encoding="utf-8"),
    )

    for source in sources:
        assert "authoritative assurance" in source
        assert "Application Insights" in source
        assert "EvaluationDataset" in source
        assert "Feedback" in source
        assert "unreviewed" in source or "human review" in source

    developer_guide = sources[1]
    assert "invoke_agent" in developer_guide
    assert re.search(r"direct\s+sibling children", developer_guide)
    assert "execute_tool" in developer_guide
    assert "renewable" in developer_guide


def test_agent_monitoring_fails_closed_before_managed_preflight():
    notebook = (
        ROOT
        / "templates"
        / "agent-app"
        / "template"
        / "notebooks"
        / "02_enable_monitoring.py"
    ).read_text(encoding="utf-8")
    readme = (
        ROOT / "templates" / "agent-app" / "template" / "README.md.tmpl"
    ).read_text(encoding="utf-8")

    for source in (notebook, readme):
        assert "Production Monitoring (Beta)" in source
        assert "serverless budget policy" in source
        assert re.search(r"SQL\s+warehouse", source)
        assert "Unity Catalog" in source
        assert re.search(r"trace-table\s+permissions", source)

    assert "MANAGED_MONITORING_PREFLIGHT_COMPLETE = False" in notebook
    preflight_call = notebook.index(
        "require_managed_monitoring_preflight(MANAGED_MONITORING_PREFLIGHT_COMPLETE)"
    )
    assert preflight_call < notebook.rindex(".register(")
    assert preflight_call < notebook.rindex(".start(")
    assert "does not enable Beta" in readme
    assert "does not provision them" in notebook
    for source in (notebook, readme):
        assert re.search(r"receives (?:the )?(?:sampled )?trace", source)
        assert "benchmark" in source
        assert "expectations" in source
        assert "decision_action_consistency" in source
        assert "self-contained" in source
        assert "@scorer" in source
        assert "decision_tool_appropriateness" in source
        assert "cannot be registered unchanged" in source
    assert "from app.tool_scoring import" not in notebook


def test_trace_to_dataset_teaching_uses_reviewed_native_mlflow_boundary():
    template_readme = (
        ROOT / "templates" / "agent-app" / "template" / "README.md.tmpl"
    ).read_text(encoding="utf-8")
    lifecycle = (ROOT / "docs" / "genai-lifecycle.md").read_text(encoding="utf-8")

    for source in (template_readme, lifecycle):
        assert "no public `promote_trace`" in source
        assert "dataset.merge_records(reviewed_traces)" in source
        assert re.search(r"root\s+inputs/outputs", source)
        assert re.search(r"expectation\s+Assessments", source)
        assert re.search(r"source\s+trace/session lineage", source)
        assert re.search(r"full\s+span tree", source)
        assert "expected_tool_calls" in source
        assert "live-validate" in source


def test_advanced_notebooks_offer_governed_dataset_and_run_evidence():
    sources = {
        path.name: advanced_implementation_source(path.name)
        for path in (ROOT / "examples").glob("*.ipynb")
        if path.name in ADVANCED_SUPPORT_BY_NOTEBOOK
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
        assert re.search(r"mlflow(?:_module)?\.log_input\(", source)
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
    assert 'dataset_name=f"fictional_{split}_v1"' in optimization
    for split in ("judge_calibration", "optimizer_training", "held_out_release"):
        assert f'"{split}"' in optimization


def test_advanced_notebooks_run_all_on_the_credential_free_default_path(monkeypatch):
    pytest.importorskip("pandas")
    deny_example_credentials_and_network(monkeypatch)
    # 14 is connected-guarded like 05/07 and deliberately excluded here.
    paths = (
        sorted((ROOT / "examples").glob("0[89]_*.ipynb"))
        + sorted((ROOT / "examples").glob("1[0-3]_*.ipynb"))
        + [ROOT / "examples" / "15_compare_and_select_llms.ipynb"]
    )
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for repeat in range(2):
            notebook_shell = SimpleNamespace(
                run_line_magic=lambda *_args, **_kwargs: None
            )
            namespace = {
                "__name__": f"notebook_{path.stem}_{repeat}",
                "get_ipython": lambda shell=notebook_shell: shell,
            }
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
    assert "'secret_redaction_verified': True" in output
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
    implementation_source = advanced_implementation_source("07_first_llm_call.ipynb")
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
    assert "integration=TraceIntegration.MLFLOW_OPENAI" in implementation_source
    assert "capture_mode=TraceCaptureMode.FULL" in implementation_source
    assert "model.create_native_async_client()" in implementation_source
    assert "native_async_client.chat.completions.create(" in implementation_source
    assert 'stream_options={"include_usage": True}' in implementation_source
    assert "await stream.close()" in implementation_source
    assert "link_prompt_versions_to_trace(" in implementation_source
    assert 'application_span.set_attribute("mlflow.message.format", "openai")' in (
        implementation_source
    )
    assert 'application_span.set_outputs({"content": content})' in implementation_source
    assert "mlflow_module.update_current_trace(request_preview=rendered_prompt)" in (
        implementation_source
    )
    assert "mlflow_module.update_current_trace(response_preview=content)" in (
        implementation_source
    )
    assert (
        '@traced(name="earnings_summary.prompt_evaluation' not in implementation_source
    )
    assert "mlflow_module.log_input(registered_dataset" in implementation_source
    assert 'trace_metadata.get("mlflow.sourceRun")' in implementation_source
    assert "client.search_traces(" in implementation_source
    assert "include_spans=False" in implementation_source
    assert 'get("mlflow.trace.sizeStats", "{}")' in implementation_source
    assert 'lineage_tags.get("mlflow.linkedPrompts"' in implementation_source
    assert 'client.get_run(record["run_id"])' in implementation_source

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
    assert "earnings_excerpt" in implementation_source
    assert "source_id" in implementation_source
    assert "len(CASES) * 2" in code_source
    assert "fact_coverage" in implementation_source
    assert "citation" in implementation_source
    assert "recommendation_policy" in implementation_source
    assert "latency" in implementation_source
    assert "input_tokens" in implementation_source
    assert "output_tokens" in implementation_source
    assert "cost_coverage" in implementation_source

    assert implementation_source.count(".load(") >= 1
    assert implementation_source.count("version=int(") >= 1
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
    assert "CAN_QUERY" in implementation_source
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
        assert intent_comment in implementation_source
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile(
            "".join(cell.get("source", [])),
            f"07_first_llm_call.ipynb:code-cell-{index}",
            "exec",
            flags=PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )


class _TraceNotebookSpan:
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


class _TraceNotebookMlflow:
    def __init__(self):
        self.span = _TraceNotebookSpan()
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


class _TraceNotebookStream:
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


class _FailingTraceNotebookStream(_TraceNotebookStream):
    async def __anext__(self):
        raise RuntimeError("synthetic stream failure")


def test_first_llm_trace_displays_content_without_losing_telemetry(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from examples.support import connected_llm

    stream = _TraceNotebookStream()
    native_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: _async_value(stream))
        )
    )
    fake_mlflow = _TraceNotebookMlflow()
    monkeypatch.setattr(
        connected_llm,
        "set_trace_resource_context",
        lambda context: None,
    )
    result = asyncio.run(
        connected_llm.invoke_prompt(
            native_async_client=native_client,
            rendered_prompt="synthetic user request",
            mlflow_module=fake_mlflow,
            model=SimpleNamespace(model="configured-model"),
            resource_context=SimpleNamespace(),
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


def test_first_llm_trace_closes_stream_after_iteration_failure(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from examples.support import connected_llm

    stream = _FailingTraceNotebookStream()
    native_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: _async_value(stream))
        )
    )
    monkeypatch.setattr(
        connected_llm,
        "set_trace_resource_context",
        lambda context: None,
    )

    with pytest.raises(RuntimeError, match="synthetic stream failure"):
        asyncio.run(
            connected_llm.invoke_prompt(
                native_async_client=native_client,
                rendered_prompt="synthetic user request",
                mlflow_module=_TraceNotebookMlflow(),
                model=SimpleNamespace(model="configured-model"),
                resource_context=SimpleNamespace(),
            )
        )

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
    "app_usage_policy_id",
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
    assert bundle_tags["tag_schema_version"] == "2"
    assert compute_tags["tag_schema_version"] == "2"


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
