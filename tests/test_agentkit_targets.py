"""Unit tests for target detection and the predict-fn adapters."""

import json
import sys
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from aai_core.agentkit.config import AgentkitConfig, ProjectContext
from aai_core.agentkit.errors import (
    ConfigError,
    MissingExtraError,
    TargetContractError,
    TargetInvocationError,
    TargetResolutionError,
)
from aai_core.agentkit.targets import (
    TargetKind,
    build_predict_fn,
    resolve_target,
)
from aai_core.testing import dev_settings

FAKE_MLFLOW = SimpleNamespace(trace=lambda fn: fn)


def _project(tmp_path, **config_overrides):
    values = {
        "version": 1,
        "agent": "src/app/example_agent.py:respond",
        "dataset": "evals/data/golden_cases.json",
    }
    values.update(config_overrides)
    return ProjectContext(
        config=AgentkitConfig(**values),
        settings=dev_settings(),
        root=tmp_path,
    )


def test_detection_table(tmp_path):
    (tmp_path / "agent.py").write_text("def respond(q):\n    return q\n")
    (tmp_path / "answers.json").write_text("[]")

    cases = [
        ("endpoints:/serving", TargetKind.SERVING_ENDPOINT, "endpoints:/serving"),
        ("models:/main.eval.agent", TargetKind.UC_MODEL, "models:/main.eval.agent"),
        ("https://host/score", TargetKind.HTTP, "https://host/score"),
        ("agent.py:respond", TargetKind.LOCAL_CALLABLE, "agent.py:respond"),
        ("json.tool:main", TargetKind.LOCAL_CALLABLE, "json.tool:main"),
        ("answers.json", TargetKind.ANSWER_SHEET, "answers.json"),
        ("my-endpoint", TargetKind.SERVING_ENDPOINT, "endpoints:/my-endpoint"),
    ]
    for reference, kind, normalized in cases:
        target = resolve_target(reference, root=tmp_path)
        assert target.kind is kind, reference
        assert target.normalized == normalized, reference


def test_logical_model_wins_over_bare_endpoint(tmp_path):
    settings = dev_settings(
        models={"target-model": {"provider": "databricks", "deployment": "x"}}
    )

    target = resolve_target("target-model", root=tmp_path, settings=settings)
    assert target.kind is TargetKind.LOGICAL_MODEL

    bare = resolve_target("other-endpoint", root=tmp_path, settings=settings)
    assert bare.kind is TargetKind.SERVING_ENDPOINT


def test_missing_local_file_is_an_error(tmp_path):
    with pytest.raises(TargetResolutionError) as excinfo:
        resolve_target("missing.py:respond", root=tmp_path)
    assert "does not exist" in str(excinfo.value)


def test_unresolvable_reference_lists_supported_shapes(tmp_path):
    with pytest.raises(TargetResolutionError) as excinfo:
        resolve_target("!!definitely not a target!!", root=tmp_path)
    message = str(excinfo.value)
    assert "endpoints:/" in message
    assert "models:/" in message
    assert "answer sheet" in message


def test_answer_sheet_target_builds_no_predict_fn(tmp_path):
    (tmp_path / "answers.json").write_text("[]")
    target = resolve_target("answers.json", root=tmp_path)

    predict = build_predict_fn(
        target, project=_project(tmp_path), mlflow_module=FAKE_MLFLOW
    )
    assert predict is None


def test_local_callable_single_argument(tmp_path):
    (tmp_path / "agent.py").write_text(
        "def respond(question):\n    return f'echo {question}'\n"
    )
    target = resolve_target("agent.py:respond", root=tmp_path)

    predict = build_predict_fn(
        target, project=_project(tmp_path), mlflow_module=FAKE_MLFLOW
    )
    assert predict(question="hello") == "echo hello"
    # sole input under a different key still maps to the single parameter
    assert predict(prompt="hi") == "echo hi"


def test_local_callable_keyword_arguments(tmp_path):
    (tmp_path / "agent.py").write_text(
        "def respond(question, tone):\n    return f'{tone}: {question}'\n"
    )
    target = resolve_target("agent.py:respond", root=tmp_path)

    predict = build_predict_fn(
        target, project=_project(tmp_path), mlflow_module=FAKE_MLFLOW
    )
    assert predict(question="q", tone="formal") == "formal: q"


def test_local_callable_contract_mismatch(tmp_path):
    (tmp_path / "agent.py").write_text("def respond(question):\n    return question\n")
    target = resolve_target("agent.py:respond", root=tmp_path)
    predict = build_predict_fn(
        target, project=_project(tmp_path), mlflow_module=FAKE_MLFLOW
    )

    with pytest.raises(TargetContractError) as excinfo:
        predict(alpha="a", beta="b")
    assert "question" in str(excinfo.value)


def test_local_callable_runtime_error_wraps(tmp_path):
    (tmp_path / "agent.py").write_text(
        "def respond(question):\n    raise RuntimeError('boom')\n"
    )
    target = resolve_target("agent.py:respond", root=tmp_path)
    predict = build_predict_fn(
        target, project=_project(tmp_path), mlflow_module=FAKE_MLFLOW
    )

    with pytest.raises(TargetInvocationError) as excinfo:
        predict(question="q")
    assert "boom" in str(excinfo.value)


def test_local_callable_missing_attribute(tmp_path):
    (tmp_path / "agent.py").write_text("VALUE = 1\n")
    target = resolve_target("agent.py:VALUE", root=tmp_path)

    with pytest.raises(TargetResolutionError) as excinfo:
        build_predict_fn(target, project=_project(tmp_path), mlflow_module=FAKE_MLFLOW)
    assert "not a callable" in str(excinfo.value)


def test_http_adapter_maps_request_and_response(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_TOKEN", "token-value")
    captured = {}

    def transport(request):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return json.dumps({"result": {"text": "the answer"}}).encode("utf-8")

    project = _project(
        tmp_path,
        request_mapping={
            "request_field": "payload.input",
            "response_field": "result.text",
            "extra_body": {"payload": {"mode": "chat"}, "version": 2},
            "auth_env": "AGENT_TOKEN",
        },
    )
    target = resolve_target("https://host/score", root=tmp_path)
    predict = build_predict_fn(
        target, project=project, transport=transport, mlflow_module=FAKE_MLFLOW
    )

    assert predict(question="what is my balance") == "the answer"
    assert captured["url"] == "https://host/score"
    assert captured["body"] == {
        "payload": {"mode": "chat", "input": "what is my balance"},
        "version": 2,
    }
    assert captured["headers"].get("Authorization") == "Bearer token-value"


def test_http_adapter_missing_auth_env(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_TOKEN", raising=False)
    project = _project(tmp_path, request_mapping={"auth_env": "AGENT_TOKEN"})
    target = resolve_target("https://host/score", root=tmp_path)
    predict = build_predict_fn(
        target,
        project=project,
        transport=lambda request: b"{}",
        mlflow_module=FAKE_MLFLOW,
    )

    with pytest.raises(TargetInvocationError) as excinfo:
        predict(question="q")
    assert "AGENT_TOKEN" in str(excinfo.value)


def test_http_adapter_missing_response_path_lists_keys_not_bodies(tmp_path):
    def transport(request):
        return json.dumps({"unexpected": "secret-content", "status": "ok"}).encode(
            "utf-8"
        )

    target = resolve_target("https://host/score", root=tmp_path)
    predict = build_predict_fn(
        target,
        project=_project(tmp_path),
        transport=transport,
        mlflow_module=FAKE_MLFLOW,
    )

    with pytest.raises(TargetContractError) as excinfo:
        predict(question="q")
    message = str(excinfo.value)
    assert "unexpected" in message
    assert "status" in message
    assert "secret-content" not in message


def test_http_adapter_translates_http_errors(tmp_path):
    def transport(request):
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", None, None)

    target = resolve_target("https://host/score", root=tmp_path)
    predict = build_predict_fn(
        target,
        project=_project(tmp_path),
        transport=transport,
        mlflow_module=FAKE_MLFLOW,
    )

    with pytest.raises(TargetInvocationError) as excinfo:
        predict(question="q")
    assert "401" in str(excinfo.value)
    assert "auth" in str(excinfo.value).lower()


def test_http_adapter_rejects_non_json(tmp_path):
    target = resolve_target("https://host/score", root=tmp_path)
    predict = build_predict_fn(
        target,
        project=_project(tmp_path),
        transport=lambda request: b"<html>oops</html>",
        mlflow_module=FAKE_MLFLOW,
    )

    with pytest.raises(TargetContractError):
        predict(question="q")


def test_serving_endpoint_without_sdk_names_the_extra(tmp_path, monkeypatch):
    # A None entry makes the import fail the way an absent package does,
    # so this holds whether or not the extras are installed here.
    monkeypatch.setitem(sys.modules, "databricks_openai", None)
    target = resolve_target("endpoints:/serving", root=tmp_path)

    with pytest.raises(MissingExtraError) as excinfo:
        build_predict_fn(target, project=_project(tmp_path), mlflow_module=FAKE_MLFLOW)
    assert "aai-core[databricks]" in str(excinfo.value)


def test_serving_endpoint_without_credentials_explains_login(tmp_path):
    pytest.importorskip("databricks_openai")
    target = resolve_target("endpoints:/serving", root=tmp_path)

    with pytest.raises(TargetResolutionError) as excinfo:
        build_predict_fn(target, project=_project(tmp_path), mlflow_module=FAKE_MLFLOW)
    assert "az login" in str(excinfo.value)


def test_openai_shaped_request_mapping_builds_an_array(tmp_path):
    """`messages.0.content` is the template's documented mapping.

    Treating every path segment as a dict key would replace the messages
    list from extra_body with `{"0": {...}}`, and an OpenAI-compatible or
    Foundry endpoint rejects that on every call.
    """

    captured = {}

    def transport(request):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return json.dumps({"choices": [{"message": {"content": "the answer"}}]}).encode(
            "utf-8"
        )

    project = _project(
        tmp_path,
        request_mapping={
            "request_field": "messages.0.content",
            "response_field": "choices.0.message.content",
            "extra_body": {"messages": [{"role": "user"}], "model": "my-model"},
        },
    )
    target = resolve_target("https://host/score", root=tmp_path)
    predict = build_predict_fn(
        target, project=project, transport=transport, mlflow_module=FAKE_MLFLOW
    )

    assert predict(question="what is my balance") == "the answer"
    assert captured["body"] == {
        # The list stays a list, and the role extra_body set survives.
        "messages": [{"role": "user", "content": "what is my balance"}],
        "model": "my-model",
    }


def test_request_mapping_builds_a_list_without_extra_body(tmp_path):
    captured = {}

    def transport(request):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return json.dumps({"output": "ok"}).encode("utf-8")

    project = _project(
        tmp_path, request_mapping={"request_field": "messages.0.content"}
    )
    target = resolve_target("https://host/score", root=tmp_path)
    predict = build_predict_fn(
        target, project=project, transport=transport, mlflow_module=FAKE_MLFLOW
    )

    predict(question="hello")

    assert captured["body"] == {"messages": [{"content": "hello"}]}


def _http_predict(tmp_path, url, transport, **mapping):
    project = _project(tmp_path, request_mapping=mapping)
    return build_predict_fn(
        resolve_target(url, root=tmp_path),
        project=project,
        transport=transport,
        mlflow_module=FAKE_MLFLOW,
    )


def test_a_token_is_never_sent_over_cleartext_http(tmp_path, monkeypatch):
    """An http:// target plus a token puts the credential on the wire.

    `resolve_target` accepts http:// because an unauthenticated local stub
    is a real target shape. Adding request_mapping.auth_env changes what a
    plain hop costs: the bearer token is readable by anyone on the path.
    """

    monkeypatch.setenv("AGENT_TOKEN", "token-value")
    calls = []

    with pytest.raises(ConfigError) as excinfo:
        _http_predict(
            tmp_path,
            "http://agent.internal/score",
            lambda request: calls.append(request) or b"{}",
            auth_env="AGENT_TOKEN",
        )

    message = str(excinfo.value)
    assert "unencrypted" in message
    assert "token-value" not in message
    # Refused where the call is built, so nothing was ever sent.
    assert calls == []


def test_cleartext_is_allowed_without_a_token_and_on_loopback(tmp_path, monkeypatch):
    """Only the credential needs the encrypted hop."""

    monkeypatch.setenv("AGENT_TOKEN", "token-value")

    def transport(request):
        return json.dumps({"output": "ok"}).encode("utf-8")

    unauthenticated = _http_predict(tmp_path, "http://agent.internal/score", transport)
    loopback = _http_predict(
        tmp_path,
        "http://localhost:8000/score",
        transport,
        auth_env="AGENT_TOKEN",
    )

    assert unauthenticated(question="q") == "ok"
    assert loopback(question="q") == "ok"


def _redirect(from_url, to_url, *, authorization=None):
    """Ask the redirect handler what it would do, without a network."""

    from aai_core.agentkit.targets import _CredentialSafeRedirects

    headers = {"Authorization": authorization} if authorization else {}
    request = urllib.request.Request(from_url, headers=headers, method="POST")
    fp = SimpleNamespace(read=lambda: b"")
    return _CredentialSafeRedirects().redirect_request(
        request, fp, 302, "Found", {}, to_url
    )


def test_a_token_never_follows_a_redirect_to_another_origin():
    """urllib copies every header onto the redirected request.

    A 301/302/303 to another host therefore hands the bearer token to
    whoever answers there, and an endpoint that can be misconfigured can
    also be compromised.
    """

    with pytest.raises(TargetInvocationError) as excinfo:
        _redirect(
            "https://agent.internal/score",
            "https://attacker.example/collect",
            authorization="Bearer token-value",
        )

    message = str(excinfo.value)
    assert "different origin" in message
    assert "attacker.example" in message
    # The token itself is never echoed into the error.
    assert "token-value" not in message


def test_a_same_origin_redirect_still_follows_with_the_token():
    redirected = _redirect(
        "https://agent.internal/score",
        "https://agent.internal/v2/score",
        authorization="Bearer token-value",
    )

    assert redirected is not None
    assert redirected.full_url == "https://agent.internal/v2/score"


def test_an_unauthenticated_redirect_is_not_our_business():
    """No credential, nothing to leak."""

    redirected = _redirect("https://agent.internal/score", "https://cdn.example/score")

    assert redirected is not None
