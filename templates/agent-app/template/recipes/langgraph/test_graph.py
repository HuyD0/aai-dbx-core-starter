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


class FakeDependencies:
    def __init__(self) -> None:
        self.execute_attempts = 0
        self.side_effects = 0
        self.results: dict[str, dict] = {}

    async def propose(self, request):
        await asyncio.sleep(0)
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


def _graph(dependencies):
    # In-memory persistence is deliberately confined to this local test.
    return recipe.build_graph(
        dependencies,
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
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

            completed = await graph.ainvoke(Command(resume=True), config)
            assert completed["result"]["status"] == "completed"

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
                Command(resume=True),
                config,
                stream_mode="values",
            )
        ]
        assert resumed[-1]["result"]["status"] == "completed"
        assert dependencies.side_effects == 1

    asyncio.run(scenario())


def test_rejection_never_executes_the_side_effect():
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
        rejected = await graph.ainvoke(Command(resume=False), config)

        assert rejected["result"] == {"status": "rejected"}
        assert dependencies.execute_attempts == 0

    asyncio.run(scenario())


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
                "conversation_id": "conversation-4",
                "request_id": 4,
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
                invoke("thread-a", Command(resume=False)),
                stream_invoke("thread-b", Command(resume=False)),
            )
            assert all(result["result"]["status"] == "rejected" for result in resumed)

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
