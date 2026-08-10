"""The release judge is independent without conflating provider namespaces."""

import asyncio
from types import SimpleNamespace

import pytest

from aai_core import tracing
from aai_core.agents import AgentDecision, AgentDecisionType, AgentResponse
from aai_core.tags import ResourceContext
from evals.evaluate import (
    _build_predict_fn,
    _case_key,
    _evaluation_model_identities,
    load_thresholds,
)


def _settings(
    *,
    target_provider: str,
    target: str,
    judge: str,
    endpoint: str = "https://foundry-a.example.invalid/api/projects/project-a",
):
    return SimpleNamespace(
        models={
            "general-chat": {
                "provider": target_provider,
                "deployment": target,
                "endpoint": endpoint,
            },
            "judge-model": {
                "provider": "databricks",
                "deployment": judge,
            },
        }
    )


def test_cross_provider_deployment_names_do_not_imply_same_target():
    target, judge = _evaluation_model_identities(
        _settings(target_provider="foundry", target="chat", judge="chat")
    )

    assert target.startswith("foundry:chat@endpoint-sha256:")
    assert judge == "databricks:chat"


def test_foundry_endpoint_origin_is_part_of_target_identity():
    first, _ = _evaluation_model_identities(
        _settings(
            target_provider="foundry",
            target="chat",
            judge="judge",
            endpoint="https://foundry.example.invalid/api/projects/a/",
        )
    )
    second, _ = _evaluation_model_identities(
        _settings(
            target_provider="foundry",
            target="chat",
            judge="judge",
            endpoint="https://foundry.example.invalid/api/projects/b",
        )
    )
    equivalent_first, _ = _evaluation_model_identities(
        _settings(
            target_provider="foundry",
            target="chat",
            judge="judge",
            endpoint="https://FOUNDRY.example.invalid:443/api//projects/a",
        )
    )

    assert first != second
    assert first == equivalent_first
    assert "https://" not in first
    assert "/api/projects" not in first


def test_same_provider_and_deployment_cannot_self_judge():
    with pytest.raises(ValueError, match="cannot rely"):
        _evaluation_model_identities(
            _settings(target_provider="databricks", target="chat", judge="CHAT")
        )


def test_dataset_comparison_includes_review_tags_but_not_mlflow_system_tags():
    reviewed = {
        "inputs": {"question": "Q"},
        "expectations": {"expected_response": "A"},
        "tags": {"failure_mode": "delayed_label"},
    }
    registered = {
        **reviewed,
        "tags": {
            "failure_mode": "delayed_label",
            "mlflow.user": "workspace-user",
        },
    }
    drifted = {**reviewed, "tags": {"failure_mode": "popularity_bias"}}

    assert _case_key(reviewed) == _case_key(registered)
    assert _case_key(reviewed) != _case_key(drifted)


def test_release_gate_covers_outcome_behavior_and_operations():
    metrics = {rule.metric for rule in load_thresholds()}

    assert {
        "correctness/mean",
        "tool_call_correctness/mean",
        "decision_action_consistency/mean",
        "decision_tool_appropriateness/mean",
        "trace_execution_success/mean",
    }.issubset(metrics)


def test_eval_predictor_keeps_model_and_tool_spans_in_one_trace(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    original_tracking_uri = mlflow.get_tracking_uri()
    default_state = tracing.TraceState(
        metadata={},
        policy=tracing.TracePolicy(capture_mode=tracing.TraceCaptureMode.OFF),
    )
    monkeypatch.setattr(tracing, "_DEFAULT_TRACE_STATE", default_state)
    monkeypatch.setattr(tracing, "_PROCESS_TRACE_CONFIGURATION", None)
    token = tracing._TRACE_STATE.set(None)

    class FakeAgent:
        async def ainvoke(self, _request):
            with tracing.provider_span("model.generate", span_type="LLM") as span:
                assert span is not None
                span.set_outputs({"tool_call": "lookup_order_status"})
            tracing.record_agent_decision(
                AgentDecision(
                    decision_type=AgentDecisionType.TOOL_SELECTION,
                    goal="Obtain the required order status.",
                    selected_action="lookup_order_status",
                    reason="The provider explicitly requested this tool.",
                    evidence_refs=("provider_tool_calls",),
                )
            )
            with tracing.provider_span("lookup_order_status", span_type="TOOL") as span:
                assert span is not None
                span.set_inputs({"order_id": "A-1001"})
                span.set_outputs({"status": "shipped"})
            return AgentResponse(content="Order A-1001 has shipped.")

    try:
        tracing.configure_tracing(
            ResourceContext(
                application="eval-trace-test",
                project="agent-test",
                environment="test",
                team="ai-platform",
                owner_group="group:ai-platform-owners",
                cost_center="CC-1234",
                data_classification="internal",
                lifecycle="experimental",
                repository="example/agent-test",
                release="test",
            ),
            experiment_name="eval-trace-regression",
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
            integration=tracing.TraceIntegration.SDK,
            policy=tracing.TracePolicy(capture_mode=tracing.TraceCaptureMode.BOUNDED),
        )
        with asyncio.Runner() as runner:
            prediction = _build_predict_fn(FakeAgent(), runner)
            assert prediction("Where is order A-1001?") == ("Order A-1001 has shipped.")

        mlflow.flush_trace_async_logging()
        trace_id = mlflow.get_last_active_trace_id()
        assert trace_id is not None
        trace = mlflow.get_trace(trace_id, flush=True)
        spans = {span.name: span for span in trace.data.spans}

        assert set(spans) == {
            "agent.evaluate",
            "model.generate",
            "decision.tool_selection",
            "lookup_order_status",
        }
        assert spans["model.generate"].parent_id == spans["agent.evaluate"].span_id
        assert (
            spans["decision.tool_selection"].parent_id
            == spans["agent.evaluate"].span_id
        )
        assert spans["lookup_order_status"].parent_id == spans["agent.evaluate"].span_id
        assert len(trace.search_spans(span_type="TOOL")) == 1
        decision = spans["decision.tool_selection"]
        assert decision.get_attribute("agent.decision.selected_action") == (
            "lookup_order_status"
        )
    finally:
        tracing._TRACE_STATE.reset(token)
        mlflow.set_tracking_uri(original_tracking_uri)
