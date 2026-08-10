import pytest
from pydantic import ValidationError

from aai_core.agents import (
    AgentDecision,
    AgentDecisionType,
    AgentRequest,
    AgentResponse,
)
from aai_core.contracts import FrozenMapping


def test_agent_contracts_are_strict_immutable_and_serializable():
    request = AgentRequest(
        messages=[{"role": "user", "content": "hello"}],
        session_id="conversation-1",
        metadata={"channel": "test"},
    )
    response = AgentResponse(
        content="ready",
        citations=[{"doc_uri": "https://example.test/doc"}],
        metadata={"quality": "checked"},
    )

    assert isinstance(request.messages[0], FrozenMapping)
    assert isinstance(request.metadata, FrozenMapping)
    assert request.model_dump(mode="json")["messages"][0]["role"] == "user"
    assert response.model_dump(mode="json")["citations"][0]["doc_uri"].startswith(
        "https://"
    )

    try:
        request.session_id = "other"
    except ValidationError:
        pass
    else:
        raise AssertionError("AgentRequest must be frozen")


def test_agent_contracts_forbid_unknown_fields():
    try:
        AgentRequest(
            messages=[{"role": "user", "content": "hello"}],
            unsupported=True,
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("AgentRequest must reject unknown fields")


def test_agent_decision_is_strict_frozen_and_serializable():
    decision = AgentDecision(
        decision_type="tool_selection",
        goal="Answer with authoritative order status",
        selected_action="lookup_order_status",
        reason="The request requires current order data.",
        evidence_refs=["user_request", "tool_schema:lookup_order_status"],
        confidence=0.94,
        alternatives_considered=["answer_without_tool"],
        expected_result="Return the current order status.",
    )

    assert decision.decision_type is AgentDecisionType.TOOL_SELECTION
    assert decision.evidence_refs == (
        "user_request",
        "tool_schema:lookup_order_status",
    )
    assert decision.alternatives_considered == ("answer_without_tool",)
    assert decision.model_dump(mode="json") == {
        "decision_type": "tool_selection",
        "goal": "Answer with authoritative order status",
        "selected_action": "lookup_order_status",
        "reason": "The request requires current order data.",
        "evidence_refs": ["user_request", "tool_schema:lookup_order_status"],
        "confidence": 0.94,
        "alternatives_considered": ["answer_without_tool"],
        "expected_result": "Return the current order status.",
    }

    with pytest.raises(ValidationError):
        decision.confidence = 0.5


@pytest.mark.parametrize(
    "confidence",
    [-0.01, 1.01, float("nan"), float("inf"), True, "0.94"],
)
def test_agent_decision_rejects_invalid_confidence(confidence):
    with pytest.raises(ValidationError):
        AgentDecision(
            decision_type=AgentDecisionType.ANSWER_READINESS,
            goal="Answer the request",
            selected_action="answer",
            reason="The available evidence is sufficient.",
            evidence_refs=("tool_result:call-1",),
            confidence=confidence,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("goal", "leading whitespace "),
        ("reason", "First line\nsecond line"),
        ("selected_action", "run arbitrary action"),
        ("evidence_refs", ("full evidence payload",)),
        ("alternatives_considered", ("answer from memory",)),
        ("reason", "x" * 513),
        ("evidence_refs", ()),
        ("evidence_refs", tuple(f"source:{index}" for index in range(17))),
        (
            "alternatives_considered",
            tuple(f"alternative_{index}" for index in range(9)),
        ),
    ],
)
def test_agent_decision_rejects_unbounded_or_non_reference_content(field, value):
    values = {
        "decision_type": AgentDecisionType.TOOL_SELECTION,
        "goal": "Choose a source of current information",
        "selected_action": "lookup_order_status",
        "reason": "The request requires current order data.",
        "evidence_refs": ("user_request",),
    }
    values[field] = value

    with pytest.raises(ValidationError):
        AgentDecision(**values)


def test_agent_decision_rejects_unknown_fields_and_unknown_types():
    with pytest.raises(ValidationError):
        AgentDecision(
            decision_type="private_reasoning",
            goal="Answer the request",
            selected_action="answer",
            reason="The available evidence is sufficient.",
            evidence_refs=("user_request",),
        )

    with pytest.raises(ValidationError):
        AgentDecision(
            decision_type=AgentDecisionType.ANSWER_READINESS,
            goal="Answer the request",
            selected_action="answer",
            reason="The available evidence is sufficient.",
            evidence_refs=("user_request",),
            hidden_reasoning="not allowed",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decision_type", 123),
        ("evidence_refs", "user_request"),
        ("alternatives_considered", "answer_without_tool"),
    ],
)
def test_agent_decision_malformed_boundary_types_raise_validation_error(field, value):
    values = {
        "decision_type": AgentDecisionType.TOOL_SELECTION,
        "goal": "Choose a source of current information",
        "selected_action": "lookup_order_status",
        "reason": "The request requires current order data.",
        "evidence_refs": ("user_request",),
    }
    values[field] = value

    with pytest.raises(ValidationError):
        AgentDecision(**values)
