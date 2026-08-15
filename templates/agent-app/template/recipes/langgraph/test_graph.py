"""Behavioral contract for the optional durable async LangGraph recipe."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import ValidationError

RECIPE = Path(__file__).with_name("graph.py")
SPEC = importlib.util.spec_from_file_location("aai_langgraph_recipe", RECIPE)
assert SPEC is not None and SPEC.loader is not None
recipe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = recipe
SPEC.loader.exec_module(recipe)

APPROVE = {"approved": True, "reason_code": "approved"}
REJECT_MODEL_ERROR = {
    "approved": False,
    "reason_code": "model_error",
    "note": "The proposed case references the wrong account.",
}
REJECT_AMBIGUOUS = {
    "approved": False,
    "reason_code": "ambiguous_intent",
    "note": "Actually, change the date.",
}


class FakeDependencies:
    def __init__(self) -> None:
        self.execute_attempts = 0
        self.side_effects = 0
        self.results: dict[str, dict] = {}
        self.proposals: list[dict | None] = []

    async def propose(self, request, *, feedback=None):
        await asyncio.sleep(0)
        self.proposals.append(feedback)
        return {"action": "open_case", "question": request.question}

    async def execute_once(self, *, idempotency_key, action):
        await asyncio.sleep(0)
        self.execute_attempts += 1
        if idempotency_key not in self.results:
            self.side_effects += 1
            self.results[idempotency_key] = {
                "status": "completed",
                "idempotency_key": idempotency_key,
                **action,
            }
        return self.results[idempotency_key]


def _graph(dependencies, **kwargs):
    # In-memory persistence is deliberately confined to this local test.
    return recipe.build_graph(
        dependencies,
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        **kwargs,
    )


def test_ainvoke_interrupt_resume_and_duplicate_delivery_are_safe():
    async def scenario():
        dependencies = FakeDependencies()
        graph = _graph(dependencies)
        request = recipe.SupportRequest(
            conversation_id="conversation-1",
            request_id="request-1",
            question="Open a support case",
        )

        for thread_id in ("delivery-1", "delivery-2"):
            config = {"configurable": {"thread_id": thread_id}}
            interrupted = await graph.ainvoke(recipe.initial_state(request), config)
            assert interrupted["__interrupt__"]
            assert dependencies.side_effects == (0 if thread_id == "delivery-1" else 1)

            completed = await graph.ainvoke(Command(resume=APPROVE), config)
            assert completed["result"]["status"] == "completed"
            assert completed["decision"]["reason_code"] == "approved"

        assert dependencies.execute_attempts == 2
        assert dependencies.side_effects == 1

    asyncio.run(scenario())


def test_astream_interrupt_and_resume_are_separate_invocations():
    async def scenario():
        dependencies = FakeDependencies()
        graph = _graph(dependencies)
        config = {"configurable": {"thread_id": "streamed-delivery"}}
        request = recipe.SupportRequest(
            conversation_id="conversation-2",
            request_id="request-2",
            question="Open another support case",
        )

        first = [
            event
            async for event in graph.astream(
                recipe.initial_state(request),
                config,
                stream_mode="values",
            )
        ]
        assert first[-1]["__interrupt__"]
        assert dependencies.side_effects == 0

        resumed = [
            event
            async for event in graph.astream(
                Command(resume=APPROVE),
                config,
                stream_mode="values",
            )
        ]
        assert resumed[-1]["result"]["status"] == "completed"
        assert dependencies.side_effects == 1

    asyncio.run(scenario())


def test_rejection_never_executes_and_preserves_the_reason():
    async def scenario():
        dependencies = FakeDependencies()
        graph = _graph(dependencies)
        config = {"configurable": {"thread_id": "rejected-delivery"}}
        request = recipe.SupportRequest(
            conversation_id="conversation-3",
            request_id="request-3",
            question="Open another support case",
        )

        interrupted = await graph.ainvoke(recipe.initial_state(request), config)
        assert interrupted["__interrupt__"]
        rejected = await graph.ainvoke(Command(resume=REJECT_MODEL_ERROR), config)

        # The override keeps its reason: the trace and the final state carry
        # why the reviewer said no, not only that they did.
        assert rejected["result"] == {
            "status": "rejected",
            "reason_code": "model_error",
            "note": REJECT_MODEL_ERROR["note"],
            "attempts": 1,
        }
        assert dependencies.execute_attempts == 0

    asyncio.run(scenario())


def test_ambiguous_rejection_replans_once_with_feedback_then_executes():
    async def scenario():
        dependencies = FakeDependencies()
        graph = _graph(dependencies)
        config = {"configurable": {"thread_id": "replanned-delivery"}}
        request = recipe.SupportRequest(
            conversation_id="conversation-4",
            request_id="request-4",
            question="Open a support case",
        )

        interrupted = await graph.ainvoke(recipe.initial_state(request), config)
        assert interrupted["__interrupt__"]

        replanned = await graph.ainvoke(Command(resume=REJECT_AMBIGUOUS), config)
        # The graph re-proposed instead of terminating and is waiting for a
        # fresh review of attempt two.
        assert replanned["__interrupt__"]
        assert replanned["attempts"] == 2
        assert dependencies.side_effects == 0

        completed = await graph.ainvoke(Command(resume=APPROVE), config)
        assert completed["result"]["status"] == "completed"
        assert completed["attempts"] == 2
        assert dependencies.side_effects == 1

        # The second proposal received the reviewer's correction as feedback.
        assert dependencies.proposals[0] is None
        assert dependencies.proposals[1]["reason_code"] == "ambiguous_intent"
        assert dependencies.proposals[1]["note"] == REJECT_AMBIGUOUS["note"]

    asyncio.run(scenario())


def test_replanning_is_bounded_by_the_attempt_cap():
    async def scenario():
        dependencies = FakeDependencies()
        graph = _graph(dependencies, max_proposal_attempts=2)
        config = {"configurable": {"thread_id": "capped-delivery"}}
        request = recipe.SupportRequest(
            conversation_id="conversation-5",
            request_id="request-5",
            question="Open a support case",
        )

        interrupted = await graph.ainvoke(recipe.initial_state(request), config)
        assert interrupted["__interrupt__"]

        replanned = await graph.ainvoke(Command(resume=REJECT_AMBIGUOUS), config)
        assert replanned["__interrupt__"]

        exhausted = await graph.ainvoke(Command(resume=REJECT_AMBIGUOUS), config)
        assert exhausted["result"]["status"] == "rejected"
        assert exhausted["result"]["reason_code"] == "ambiguous_intent"
        assert exhausted["result"]["attempts"] == 2
        assert dependencies.execute_attempts == 0

    asyncio.run(scenario())


def test_resume_boundary_rejects_malformed_decisions_then_recovers():
    async def scenario():
        dependencies = FakeDependencies()
        graph = _graph(dependencies)
        config = {"configurable": {"thread_id": "malformed-resume"}}
        request = recipe.SupportRequest(
            conversation_id="conversation-6",
            request_id="request-6",
            question="Open a support case",
        )

        interrupted = await graph.ainvoke(recipe.initial_state(request), config)
        assert interrupted["__interrupt__"]

        # The resume payload is untrusted input. A submitted resume value is
        # durable, so the graph must not raise: each malformed decision —
        # coerced booleans, unknown fields, unknown reasons, legacy bare
        # booleans — re-interrupts with the validation problem instead.
        for malformed in (
            {"approved": 1, "reason_code": "approved"},
            {"approved": True, "reason_code": "approved", "unexpected": True},
            {"approved": True, "reason_code": "made_up_reason"},
            True,
        ):
            reprompted = await graph.ainvoke(Command(resume=malformed), config)
            (pending,) = reprompted["__interrupt__"]
            assert "invalid decision payload" in pending.value["error"]
            assert dependencies.execute_attempts == 0

        # Malformed attempts never corrupt durable state: the thread is still
        # interrupted and a valid decision completes it.
        completed = await graph.ainvoke(Command(resume=APPROVE), config)
        assert completed["result"]["status"] == "completed"
        assert dependencies.side_effects == 1

    asyncio.run(scenario())


def test_decision_model_requires_reason_consistent_with_outcome():
    with pytest.raises(ValidationError):
        recipe.ApprovalDecision.model_validate(
            {"approved": True, "reason_code": "model_error"}
        )
    with pytest.raises(ValidationError):
        recipe.ApprovalDecision.model_validate(
            {"approved": False, "reason_code": "approved"}
        )


def test_build_rejects_sync_only_checkpointer():
    class SyncOnly:
        pass

    with pytest.raises(TypeError, match="async-compatible checkpointer"):
        recipe.build_graph(
            FakeDependencies(),
            checkpointer=SyncOnly(),
            store=InMemoryStore(),
        )


def test_boundary_rejects_coercion_and_unknown_fields():
    with pytest.raises(ValidationError):
        recipe.initial_state(
            {
                "conversation_id": "conversation-7",
                "request_id": 7,
                "question": "Invalid request",
                "unexpected": True,
            }
        )


def test_mlflow_autolog_keeps_concurrent_and_resumed_async_traces_separate(tmp_path):
    """Canary MLflow's native LangGraph async callback and session context."""

    mlflow = pytest.importorskip("mlflow")
    from mlflow import MlflowClient
    from mlflow.tracking import fluent

    original_tracking_uri = mlflow.get_tracking_uri()
    original_experiment_id = fluent._active_experiment_id
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    experiment_name = "langgraph-async-compatibility"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment(
        experiment_name,
        artifact_location=(tmp_path / "artifacts").resolve().as_uri(),
    )
    reject_policy = {
        "approved": False,
        "reason_code": "policy_boundary",
        "note": "Requires an approver from the risk group.",
    }
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.langchain.autolog(run_tracer_inline=True)
        graph = _graph(FakeDependencies())

        async def invoke(thread_id, value):
            config = {"configurable": {"thread_id": thread_id}}
            with mlflow.tracing.context(session_id=thread_id):
                result = await graph.ainvoke(value, config)
            return result

        async def stream_invoke(thread_id, value):
            config = {"configurable": {"thread_id": thread_id}}
            with mlflow.tracing.context(session_id=thread_id):
                events = [
                    event
                    async for event in graph.astream(
                        value,
                        config,
                        stream_mode="values",
                    )
                ]
            return events[-1]

        async def scenario():
            initial = await asyncio.gather(
                invoke(
                    "thread-a",
                    recipe.initial_state(
                        recipe.SupportRequest(
                            conversation_id="thread-a",
                            request_id="request-a",
                            question="Open case A",
                        )
                    ),
                ),
                stream_invoke(
                    "thread-b",
                    recipe.initial_state(
                        recipe.SupportRequest(
                            conversation_id="thread-b",
                            request_id="request-b",
                            question="Open case B",
                        )
                    ),
                ),
            )
            assert all(result["__interrupt__"] for result in initial)
            resumed = await asyncio.gather(
                invoke("thread-a", Command(resume=reject_policy)),
                stream_invoke("thread-b", Command(resume=reject_policy)),
            )
            assert all(result["result"]["status"] == "rejected" for result in resumed)
            assert all(
                result["result"]["reason_code"] == "policy_boundary"
                for result in resumed
            )

        asyncio.run(scenario())
        mlflow.flush_trace_async_logging()

        traces = list(
            client.search_traces(
                locations=[experiment_id],
                flush=True,
            )
        )
        assert len(traces) == 4
        sessions = [
            trace.info.trace_metadata["mlflow.trace.session"] for trace in traces
        ]
        assert sessions.count("thread-a") == 2
        assert sessions.count("thread-b") == 2
    finally:
        mlflow.langchain.autolog(disable=True)
        mlflow.set_tracking_uri(original_tracking_uri)
        fluent._active_experiment_id = original_experiment_id
