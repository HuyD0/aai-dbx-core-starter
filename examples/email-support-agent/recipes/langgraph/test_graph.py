"""Behavioral contract for the accelerator's optional native LangGraph adapter."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from email_support_agent import (  # noqa: E402
    ReviewAction,
    ReviewDecision,
    ReviewReason,
)
from email_support_agent.evaluation import load_release_cases  # noqa: E402
from email_support_agent.offline import build_offline_workflow  # noqa: E402

GRAPH_PATH = Path(__file__).with_name("graph.py")
SPEC = importlib.util.spec_from_file_location("email_support_langgraph", GRAPH_PATH)
assert SPEC is not None and SPEC.loader is not None
graph_recipe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = graph_recipe
SPEC.loader.exec_module(graph_recipe)


def _case(case_id: str):
    cases = load_release_cases(ROOT / "evals" / "data" / "release_cases.jsonl")
    return next(item for item in cases if item.inputs.case_id == case_id)


def _graph(workflow):
    # Both in-memory dependencies are deliberately confined to this test.
    return graph_recipe.build_graph(
        workflow,
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
    )


def _decision(
    pending,
    *,
    case_id: str,
    action: ReviewAction,
    reason: ReviewReason,
    edited_response: str | None = None,
) -> ReviewDecision:
    review_request = pending["__interrupt__"][0].value
    return ReviewDecision(
        case_id=case_id,
        proposal_digest=review_request["proposal_digest"],
        application_release=review_request["application_release"],
        authorization_ref="synthetic://review/support-quality",
        action=action,
        reason=reason,
        edited_response=edited_response,
    )


def test_interrupt_resume_and_duplicate_delivery_are_idempotent():
    async def scenario():
        workflow, outbox = build_offline_workflow(ROOT)
        graph = _graph(workflow)
        case = _case("case-bug-crash")
        for thread_id in ("delivery-one", "delivery-two"):
            config = {"configurable": {"thread_id": thread_id}}
            pending = await graph.ainvoke(
                graph_recipe.initial_state(case.inputs), config
            )
            assert pending["__interrupt__"]
            assert len(outbox.actions) == (0 if thread_id == "delivery-one" else 2)
            decision = _decision(
                pending,
                case_id=case.inputs.case_id,
                action=ReviewAction.APPROVE,
                reason=ReviewReason.APPROVED,
            )

            completed = await graph.ainvoke(
                Command(resume=decision.model_dump(mode="json")), config
            )
            result = graph_recipe.final_result(completed)
            assert result.disposition.value == "queued"

        assert len(outbox.actions) == 2
        assert outbox.attempts == 4

    asyncio.run(scenario())


def test_rejection_never_enqueues_a_ticket_or_reply():
    async def scenario():
        workflow, outbox = build_offline_workflow(ROOT)
        graph = _graph(workflow)
        case = _case("case-billing-refund")
        config = {"configurable": {"thread_id": "rejected-delivery"}}

        pending = await graph.ainvoke(graph_recipe.initial_state(case.inputs), config)
        assert pending["__interrupt__"]
        completed = await graph.ainvoke(
            Command(
                resume=_decision(
                    pending,
                    case_id=case.inputs.case_id,
                    action=ReviewAction.REJECT,
                    reason=ReviewReason.NEEDS_INVESTIGATION,
                ).model_dump(mode="json")
            ),
            config,
        )
        result = graph_recipe.final_result(completed)
        assert result.disposition.value == "handled_by_human"
        assert not outbox.actions

    asyncio.run(scenario())


def test_safe_edit_after_abstention_is_bound_to_the_interrupted_proposal():
    async def scenario():
        workflow, outbox = build_offline_workflow(ROOT)
        graph = _graph(workflow)
        case = _case("case-critical-outage")
        config = {"configurable": {"thread_id": "edited-outage"}}

        pending = await graph.ainvoke(graph_recipe.initial_state(case.inputs), config)
        review_request = pending["__interrupt__"][0].value
        assert review_request["draft"]["abstained"] is True
        assert review_request["planned_actions"] == []
        assert review_request["proposal_digest"].startswith("sha256:")

        completed = await graph.ainvoke(
            Command(
                resume=_decision(
                    pending,
                    case_id=case.inputs.case_id,
                    action=ReviewAction.EDIT,
                    reason=ReviewReason.FACTUAL_EDIT,
                    edited_response=(
                        "A support specialist reviewed the outage and is "
                        "investigating. Verified updates will appear on the "
                        "status page."
                    ),
                ).model_dump(mode="json")
            ),
            config,
        )

        result = graph_recipe.final_result(completed)
        assert result.disposition.value == "queued"
        assert len(outbox.actions) == 1

    asyncio.run(scenario())


def test_rejected_sensitive_edit_never_enters_checkpoint_state():
    async def scenario():
        workflow, outbox = build_offline_workflow(ROOT)
        graph = _graph(workflow)
        case = _case("case-billing-refund")
        config = {"configurable": {"thread_id": "unsafe-edit"}}

        pending = await graph.ainvoke(graph_recipe.initial_state(case.inputs), config)
        unsafe = _decision(
            pending,
            case_id=case.inputs.case_id,
            action=ReviewAction.EDIT,
            reason=ReviewReason.POLICY_EDIT,
            edited_response="Safe placeholder.",
        ).model_dump(mode="json")
        unsafe["edited_response"] = "Email person@example.test now."

        with pytest.raises(ValueError, match="strict admission"):
            await graph.ainvoke(Command(resume=unsafe), config)
        snapshot = await graph.aget_state(config)
        serialized = json.dumps(snapshot.values)
        assert "person@example.test" not in serialized
        assert "person@example.test" not in repr(snapshot.tasks)
        assert "review" not in snapshot.values
        assert not outbox.actions

    asyncio.run(scenario())


def test_unreviewed_canary_path_commits_once_without_an_interrupt():
    async def scenario():
        workflow, outbox = build_offline_workflow(ROOT, auto_send_low_risk=True)
        graph = _graph(workflow)
        case = _case("case-faq-reset")
        config = {"configurable": {"thread_id": "canary-delivery"}}

        completed = await graph.ainvoke(graph_recipe.initial_state(case.inputs), config)
        assert "__interrupt__" not in completed
        assert graph_recipe.final_result(completed).disposition.value == "queued"
        assert len(outbox.actions) == 1

    asyncio.run(scenario())
