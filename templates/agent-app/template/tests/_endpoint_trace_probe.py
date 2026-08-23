"""Isolated real-MLflow canary for AgentServer trace export privacy."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import httpx
import mlflow
from mlflow import MlflowClient
from mlflow.genai.agent_server import AgentServer

from aai_core import tracing
from aai_core.agents import AgentResponse
from aai_core.tags import ResourceContext
from app import endpoint

PROMPT_SECRET = "restricted-prompt-canary-7dbf"
ANSWER_SECRET = "restricted-answer-canary-a4ce"
ANSWER_PARTS = ("restricted-answer-", "canary-a4ce")
CONVERSATION_SECRET = "restricted-conversation-canary-b190"
EXCEPTION_SECRET = "provider rejected patient-secret-exception-9917"
ERROR_TRIGGER = "trigger-private-provider-error"


class FakeApplication:
    def __init__(self, tags: ResourceContext) -> None:
        self.context = type("Context", (), {"tags": tags})()

    async def ainvoke(self, request):
        if any(ERROR_TRIGGER in item.get("content", "") for item in request.messages):
            raise ValueError(EXCEPTION_SECRET)
        return AgentResponse(content=ANSWER_SECRET)

    async def astream_text(self, request):
        if any(ERROR_TRIGGER in item.get("content", "") for item in request.messages):
            raise ValueError(EXCEPTION_SECRET)
        for part in ANSWER_PARTS:
            yield part


async def exercise_http() -> None:
    transport = httpx.ASGITransport(
        app=AgentServer("ResponsesAgent").app,
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://agent.test",
    ) as client:
        invoke_response = await client.post(
            "/invocations",
            json={
                "input": [{"role": "user", "content": PROMPT_SECRET}],
                "context": {"conversation_id": CONVERSATION_SECRET},
            },
        )
        assert invoke_response.status_code == 200, invoke_response.text
        assert ANSWER_SECRET in invoke_response.text

        async with client.stream(
            "POST",
            "/invocations",
            json={
                "input": [{"role": "user", "content": PROMPT_SECRET}],
                "context": {"conversation_id": CONVERSATION_SECRET},
                "stream": True,
            },
        ) as stream_response:
            assert stream_response.status_code == 200
            stream_body = (await stream_response.aread()).decode("utf-8")
        assert ANSWER_SECRET in stream_body

        failed_invoke = await client.post(
            "/invocations",
            json={"input": [{"role": "user", "content": ERROR_TRIGGER}]},
        )
        assert failed_invoke.status_code >= 500
        assert EXCEPTION_SECRET not in failed_invoke.text

        async with client.stream(
            "POST",
            "/invocations",
            json={
                "input": [{"role": "user", "content": ERROR_TRIGGER}],
                "stream": True,
            },
        ) as failed_stream:
            failed_stream_body = (await failed_stream.aread()).decode("utf-8")
        assert EXCEPTION_SECRET not in failed_stream_body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("restricted", "off"))
    parser.add_argument("database", type=Path)
    arguments = parser.parse_args()

    context = ResourceContext(
        application="endpoint-trace-probe",
        project="agent-test",
        environment="test",
        team="ai-platform",
        owner_group="group:ai-platform-owners",
        cost_center="CC-1234",
        data_classification=(
            "restricted" if arguments.mode == "restricted" else "internal"
        ),
        lifecycle="experimental",
        repository="example/agent-test",
        release="test",
    )
    capture_mode = (
        tracing.TraceCaptureMode.METADATA_ONLY
        if arguments.mode == "restricted"
        else tracing.TraceCaptureMode.OFF
    )
    tracking_uri = f"sqlite:///{arguments.database}"
    tracing.configure_tracing(
        context,
        experiment_name="agent-server-privacy-canary",
        tracking_uri=tracking_uri,
        integration=tracing.TraceIntegration.MLFLOW_AGENT_SERVER,
        policy=tracing.TracePolicy(capture_mode=capture_mode),
    )
    endpoint._application = lambda: FakeApplication(context)  # type: ignore[method-assign]

    asyncio.run(exercise_http())
    mlflow.flush_trace_async_logging()
    experiment = mlflow.get_experiment_by_name("agent-server-privacy-canary")
    assert experiment is not None
    traces = MlflowClient(tracking_uri=tracking_uri).search_traces(
        locations=[experiment.experiment_id],
        include_spans=True,
        flush=True,
    )
    if arguments.mode == "off":
        assert not traces, "OFF tracing persisted an AgentServer trace"
        return

    assert len(traces) == 4, "success and failure invoke/stream calls must persist"
    serialized = "\n".join(trace.to_json() for trace in traces)
    assert PROMPT_SECRET not in serialized
    assert ANSWER_SECRET not in serialized
    assert all(part not in serialized for part in ANSWER_PARTS)
    assert CONVERSATION_SECRET not in serialized
    assert EXCEPTION_SECRET not in serialized
    assert ERROR_TRIGGER not in serialized
    sessions = {
        session
        for trace in traces
        if (session := trace.info.trace_metadata.get("mlflow.trace.session"))
        is not None
    }
    assert len(sessions) == 1
    session = sessions.pop()
    assert isinstance(session, str) and session.startswith("sha256:")


if __name__ == "__main__":
    main()
