"""Strict, credential-free tests for the email-support teaching path."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ACCELERATOR = ROOT / "examples" / "email-support-agent"
WORKSHOP = ACCELERATOR / "workshop"
SOURCE = ACCELERATOR / "src"
_ADDED_SOURCE = str(SOURCE) not in sys.path
if _ADDED_SOURCE:
    sys.path.insert(0, str(SOURCE))

from email_support_agent.workshop import (  # noqa: E402
    LESSON_SPECS,
    TRACE_SPAN_CONTRACT,
    WorkshopResult,
    run_lesson,
)

if _ADDED_SOURCE:
    # Keep this standalone example out of unrelated tests' ambient import path.
    sys.path.remove(str(SOURCE))

LESSON_FILES = {
    "graph_basics": "01_graph_basics.py",
    "reliability_hitl_idempotency": "02_reliability_hitl_idempotency.py",
    "mlflow_trace_evaluation": "03_mlflow_trace_evaluation.py",
    "improvement_release_decision": "04_improvement_release_decision.py",
}


@pytest.fixture(scope="module")
def lesson_results() -> dict[str, WorkshopResult]:
    return {slug: run_lesson(slug) for slug in LESSON_SPECS}


def test_curriculum_has_four_ordered_credential_free_levels(
    lesson_results: dict[str, WorkshopResult],
):
    assert list(LESSON_SPECS) == list(LESSON_FILES)
    assert [result.level for result in lesson_results.values()] == [1, 2, 3, 4]

    for slug, result in lesson_results.items():
        assert result.slug == slug
        assert result.credential_mode == "credential_free"
        assert result.expected_observations
        assert result.failure_exercise.startswith("Predict ")
        assert all(
            "TEST-ONLY" in boundary or "NO REMOTE" in boundary
            for boundary in result.fake_boundaries
        )
        # Revalidate exactly what the executable lesson publishes.
        assert (
            WorkshopResult.model_validate_json(
                result.model_dump_json(),
                strict=True,
            )
            == result
        )


def test_level_1_teaches_strict_state_and_prepare_without_writes(
    lesson_results: dict[str, WorkshopResult],
):
    observed = lesson_results["graph_basics"].observations

    assert observed["state_contract"] == "PreparedCase"
    assert observed["state_json_round_trip"] is True
    assert observed["route"] == "knowledge_reply"
    assert observed["conditional_edge"] == "review"
    assert observed["node_sequence"] == TRACE_SPAN_CONTRACT
    assert observed["evidence_document_ids"] == ("kb-password-reset-v3",)
    assert observed["planned_action_kinds"] == ("enqueue_reply",)
    assert observed["outbox_writes_before_commit"] == 0
    assert observed["strict_admission_failure_observed"] is True


def test_level_2_teaches_interrupt_authorization_and_idempotency(
    lesson_results: dict[str, WorkshopResult],
):
    observed = lesson_results["reliability_hitl_idempotency"].observations

    assert observed["pending_review"] is True
    assert observed["preapproval_commit_blocked"] is True
    assert observed["outbox_writes_before_approval"] == 0
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", observed["proposal_digest"])
    assert observed["first_receipts_duplicate"] == (False, False)
    assert observed["retry_receipts_duplicate"] == (True, True)
    assert observed["unique_outbox_actions"] == 2
    assert observed["outbox_attempts"] == 4
    assert observed["verified_reviewer_group"] == ("group:support-quality-reviewers")
    assert observed["forged_authorization_blocked"] is True
    assert observed["forged_outbox_actions"] == 0


def test_level_3_teaches_mlflow_shapes_trace_policy_and_release_gate(
    lesson_results: dict[str, WorkshopResult],
):
    observed = lesson_results["mlflow_trace_evaluation"].observations

    assert observed["evaluation_case_count"] >= 10
    assert observed["mlflow_row_top_level_keys"] == ("expectations", "inputs")
    assert observed["mlflow_row_inputs_are_nested"] is True
    assert observed["mlflow_row_expectations_are_nested"] is True
    assert observed["span_contract"] == TRACE_SPAN_CONTRACT
    assert observed["trace_claim"] == "offline_contract_only_no_trace_id"
    assert observed["trace_capture_mode"] == "metadata_only"
    assert observed["trace_payload_shape"] == {
        "type": "mapping",
        "keys": ("body", "retrieved_documents", "subject"),
        "size": 3,
        "truncated": False,
    }
    assert observed["retriever_document_top_level_keys"] == (
        "id",
        "metadata",
        "page_content",
    )
    assert {"doc_uri", "chunk_id"}.issubset(
        observed["retriever_document_metadata_keys"]
    )
    assert observed["gate_passed"] is True
    assert observed["false_auto_send_failure_gate_passed"] is False
    assert observed["outbox_writes_from_prepare"] == 0
    assert observed["selected_metrics"] == {
        "classification/critical_recall": 1.0,
        "safety/false_auto_send_rate": 0.0,
        "retrieval/recall_at_k": 1.0,
        "trajectory/idempotency": 1.0,
        "cost/coverage": 0.0,
    }


def test_level_4_teaches_signals_and_conservative_release_decision(
    lesson_results: dict[str, WorkshopResult],
):
    observed = lesson_results["improvement_release_decision"].observations
    lifecycle = observed["lifecycle"]

    assert lifecycle["baseline"] == "review_every_reply"
    assert lifecycle["change"] == "policy_qualified_low_risk_canary"
    assert lifecycle["result"] == {
        "deterministic_gate_passed": True,
        "false_auto_send_rate": 0.0,
        "cost_coverage": 0.0,
    }
    assert lifecycle["decision"] == "inconclusive"
    assert lifecycle["release"] == "blocked_until_connected_evidence"
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        observed["application_release_digest"],
    )
    assert observed["review_feedback_action"] == "edit"
    assert observed["review_feedback_edit_distance"] == 0.25
    assert observed["outcome_feedback"] == {
        "delivery": "delivered",
        "resolved_first_contact": True,
        "customer_reopened_7d": False,
    }
    assert observed["adopt_with_failing_gate_blocked"] is True
    assert observed["production_promotion_authorized"] is False


@pytest.mark.parametrize(("slug", "script_name"), LESSON_FILES.items())
def test_each_lesson_executes_from_repository_root(slug: str, script_name: str):
    completed = subprocess.run(
        [sys.executable, str(WORKSHOP / script_name)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result_lines = [
        line.removeprefix("WORKSHOP_RESULT=")
        for line in completed.stdout.splitlines()
        if line.startswith("WORKSHOP_RESULT=")
    ]

    assert completed.stderr == ""
    assert len(result_lines) == 1
    result = WorkshopResult.model_validate_json(result_lines[0], strict=True)
    assert result.slug == slug
    assert result.credential_mode == "credential_free"


def test_workshop_outputs_do_not_expose_obvious_identity_or_secret_values(
    lesson_results: dict[str, WorkshopResult],
):
    serialized = json.dumps(
        {
            slug: result.model_dump(mode="json")
            for slug, result in lesson_results.items()
        },
        sort_keys=True,
    )

    assert not re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", serialized)
    assert not re.search(
        r"(?i)(api[_-]?key|client[_-]?secret|bearer)\s*[:=]\s*[^, }]+",
        serialized,
    )


def test_workshop_reuses_accelerator_contracts_and_has_no_remote_client_imports():
    module_path = SOURCE / "email_support_agent" / "workshop.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    defined_classes = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    forbidden_forks = {
        "RedactedEmail",
        "PreparedCase",
        "ReviewDecision",
        "EvidenceDocument",
        "EmailSupportWorkflow",
    }
    imported_roots = {
        alias.name.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert defined_classes == {"WorkshopResult"}
    assert defined_classes.isdisjoint(forbidden_forks)
    assert {"mlflow", "databricks", "azure", "requests", "httpx", "urllib"}.isdisjoint(
        imported_roots
    )
    source = module_path.read_text(encoding="utf-8")
    assert "from email_support_agent.contracts import" in source
    assert "from email_support_agent.offline import build_offline_workflow" in source
    assert "from email_support_agent.evaluation import" in source


def test_workshop_readme_is_complete_and_does_not_add_an_install_path():
    readme = (WORKSHOP / "README.md").read_text(encoding="utf-8")

    for script_name in LESSON_FILES.values():
        assert f"python examples/email-support-agent/workshop/{script_name}" in readme
        assert (WORKSHOP / script_name).is_file()
    assert readme.count("Expected observations:") == 4
    assert readme.count("Failure exercise:") == 4
    assert "TEST-ONLY" in readme
    assert "NO REMOTE" in readme
    assert "schema validation is not\nauthentication" in readme
    assert "pip install" not in readme
