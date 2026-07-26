from pydantic import ValidationError

from aai_core.agents import AgentRequest, AgentResponse
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
