"""Resolve the thing under evaluation into a callable the harness can score.

The ``agent:`` value resolves by shape — a logical model from
``aai-platform.yml``, a Databricks serving endpoint, a UC registered model,
any HTTP/JSON endpoint (a Foundry hosted agent included), a local Python
callable, or a recorded answer sheet. Detection is pure string/filesystem
logic; adapter construction imports heavy clients lazily.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from aai_core.agentkit.config import ProjectContext, RequestMapping
from aai_core.agentkit.errors import (
    ConfigError,
    TargetContractError,
    TargetInvocationError,
    TargetResolutionError,
    missing_extra,
)
from aai_core.contracts import thaw_value

_ENDPOINT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_MODULE_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_PREFERRED_INPUT_KEYS = ("question", "input", "query", "prompt", "message")
_HTTP_TIMEOUT_SECONDS = 60
_AUTH_REMEDIATIONS = {
    401: "The request was not authenticated. Check the token in the "
    "configured request_mapping.auth_env variable.",
    403: "The identity is authenticated but not authorized for this "
    "endpoint. Ask the endpoint owner for query access.",
}

SUPPORTED_SHAPES = (
    ("logical model", "target-model  (a providers.models entry in aai-platform.yml)"),
    ("serving endpoint", "endpoints:/agent-serving  (or a bare endpoint name)"),
    ("UC model", "models:/catalog.schema.agent_model"),
    ("HTTP endpoint", "https://host/score  (request_mapping maps the fields)"),
    ("local callable", "src/app/example_agent.py:respond  or  pkg.module:respond"),
    ("answer sheet", "evals/data/answer_sheet.json  (recorded outputs)"),
)


class TargetKind(StrEnum):
    LOGICAL_MODEL = "logical-model"
    SERVING_ENDPOINT = "serving-endpoint"
    UC_MODEL = "uc-model"
    HTTP = "http"
    LOCAL_CALLABLE = "local-callable"
    ANSWER_SHEET = "answer-sheet"


@dataclass(frozen=True)
class Target:
    kind: TargetKind
    ref: str
    normalized: str
    path: Path | None = None
    attribute: str | None = None


def resolve_target(
    ref: str,
    *,
    root: Path,
    settings: Any | None = None,
) -> Target:
    """Detect the target shape — first match wins, files beat bare names."""

    reference = str(ref).strip()
    if not reference:
        raise TargetResolutionError("agent must not be blank")
    if reference.startswith("endpoints:/"):
        return Target(TargetKind.SERVING_ENDPOINT, reference, reference)
    if reference.startswith("models:/"):
        return Target(TargetKind.UC_MODEL, reference, reference)
    if reference.startswith(("http://", "https://")):
        return Target(TargetKind.HTTP, reference, reference)
    if ":" in reference and not reference.startswith(("/", "\\")):
        location, _, attribute = reference.rpartition(":")
        if location.endswith(".py"):
            path = root / location
            if not path.is_file():
                raise TargetResolutionError(
                    f"agent {reference!r} points at {path}, which does not " "exist"
                )
            if not attribute.isidentifier():
                raise TargetResolutionError(
                    f"agent {reference!r} needs a function name after ':'"
                )
            return Target(
                TargetKind.LOCAL_CALLABLE,
                reference,
                reference,
                path=path,
                attribute=attribute,
            )
        if _MODULE_PATH.match(location) and attribute.isidentifier():
            return Target(
                TargetKind.LOCAL_CALLABLE,
                reference,
                reference,
                attribute=attribute,
            )
    candidate = root / reference
    if candidate.is_file() and candidate.suffix in {".json", ".jsonl"}:
        return Target(TargetKind.ANSWER_SHEET, reference, reference, path=candidate)
    if settings is not None and reference in getattr(settings, "models", {}):
        return Target(TargetKind.LOGICAL_MODEL, reference, reference)
    if _ENDPOINT_NAME.match(reference):
        return Target(TargetKind.SERVING_ENDPOINT, reference, f"endpoints:/{reference}")
    shapes = "\n".join(f"  - {kind}: {example}" for kind, example in SUPPORTED_SHAPES)
    raise TargetResolutionError(
        f"could not resolve agent {reference!r}. Supported shapes:\n{shapes}",
        remediation="Set `agent:` in agentkit.yaml to one of the shapes above.",
    )


def build_predict_fn(
    target: Target,
    *,
    project: ProjectContext,
    transport: Callable[[urllib.request.Request], bytes] | None = None,
    mlflow_module: Any | None = None,
) -> Callable[..., Any] | None:
    """Build the traced predict function for a live target.

    Returns ``None`` for answer-sheet targets — recorded outputs are scored
    directly, no live call happens. The returned callable accepts the
    dataset's ``inputs`` keys as keyword arguments and emits exactly one
    trace per call, which is the ``mlflow.genai.evaluate`` contract.
    """

    if target.kind is TargetKind.ANSWER_SHEET:
        return None
    mlflow = _mlflow(mlflow_module)
    if target.kind is TargetKind.LOCAL_CALLABLE:
        call = _local_call(target)
    elif target.kind is TargetKind.HTTP:
        call = _http_call(target, project.config.request_mapping, transport)
    elif target.kind is TargetKind.LOGICAL_MODEL:
        call = _logical_model_call(target, project)
    elif target.kind is TargetKind.SERVING_ENDPOINT:
        call = _serving_call(target, project)
    else:
        call = _uc_model_call(target, mlflow)

    def predict(**inputs: Any) -> Any:
        return call(inputs)

    predict.__name__ = "agent"
    traced = mlflow.trace(predict)
    return traced


def _local_call(target: Target) -> Callable[[Mapping[str, Any]], Any]:
    if target.path is not None:
        specification = importlib.util.spec_from_file_location(
            "aai_agentkit_target", target.path
        )
        if specification is None or specification.loader is None:
            raise TargetResolutionError(f"could not load {target.path}")
        module = importlib.util.module_from_spec(specification)
        try:
            specification.loader.exec_module(module)
        except Exception as error:
            raise TargetResolutionError(
                f"loading {target.path} failed: {error}"
            ) from error
    else:
        module_name = target.ref.rpartition(":")[0]
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise TargetResolutionError(
                f"could not import module {module_name!r}: {error}"
            ) from error
    function = getattr(module, target.attribute or "", None)
    if not callable(function):
        raise TargetResolutionError(
            f"{target.ref!r}: {target.attribute!r} is not a callable in the " "module"
        )
    signature = inspect.signature(function)
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    ]
    accepts_var_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    single_argument = len(parameters) == 1 and not accepts_var_keyword

    def call(inputs: Mapping[str, Any]) -> Any:
        try:
            if single_argument:
                name = parameters[0].name
                if name in inputs:
                    return function(inputs[name])
                if len(inputs) == 1:
                    return function(next(iter(inputs.values())))
                raise TargetContractError(
                    f"{target.ref!r} takes one argument {name!r} but the "
                    f"dataset inputs have keys {sorted(inputs)}"
                )
            return function(**inputs)
        except TargetContractError:
            raise
        except Exception as error:
            raise TargetInvocationError(
                f"agent call failed for {target.ref!r}: {error}"
            ) from error

    return call


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _require_encrypted_transport(target: Target, auth_env: str) -> None:
    """A bearer token does not travel in the clear.

    ``resolve_target`` accepts ``http://`` because an unauthenticated local
    stub is a legitimate target. Adding ``request_mapping.auth_env`` changes
    that: the token goes into an ``Authorization`` header, and on a plain
    HTTP hop anyone on the path can read it. Loopback stays allowed —
    development against a local stub never leaves the machine.

    Checked where the call is built, so it fails before the first request
    and before ``mlflow.genai.evaluate`` spends anything.
    """

    parts = urllib.parse.urlsplit(target.normalized)
    if parts.scheme == "https":
        return
    host = (parts.hostname or "").lower()
    if host in _LOOPBACK_HOSTS or host.endswith(".localhost"):
        return
    raise ConfigError(
        f"agent {target.ref!r} uses {parts.scheme}://, but "
        f"request_mapping.auth_env names {auth_env!r} - the token would be "
        "sent unencrypted",
        remediation=(
            "Point `agent:` at the https:// URL for this endpoint, or drop "
            "request_mapping.auth_env if it needs no token."
        ),
    )


def _http_call(
    target: Target,
    mapping: RequestMapping,
    transport: Callable[[urllib.request.Request], bytes] | None,
) -> Callable[[Mapping[str, Any]], Any]:
    send = transport or _urllib_transport
    if mapping.auth_env:
        _require_encrypted_transport(target, mapping.auth_env)

    def call(inputs: Mapping[str, Any]) -> Any:
        body: dict[str, Any] = thaw_value(mapping.extra_body)
        _set_path(body, mapping.request_field, _primary_input(inputs))
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if mapping.auth_env:
            token = os.environ.get(mapping.auth_env)
            if not token:
                raise TargetInvocationError(
                    f"request_mapping.auth_env names {mapping.auth_env!r} "
                    "but the variable is not set",
                    remediation=f"export {mapping.auth_env}=<token> before " "running.",
                )
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            target.normalized,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            raw = send(request)
        except urllib.error.HTTPError as error:
            raise TargetInvocationError(
                f"HTTP target returned {error.code} for {target.normalized}",
                remediation=_AUTH_REMEDIATIONS.get(error.code),
            ) from error
        except urllib.error.URLError as error:
            raise TargetInvocationError(
                f"could not reach {target.normalized}: {error.reason}"
            ) from error
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TargetContractError(
                f"{target.normalized} did not return JSON"
            ) from error
        found, value = _get_path(document, mapping.response_field)
        if not found:
            keys = sorted(document) if isinstance(document, Mapping) else type(document)
            raise TargetContractError(
                f"response has no {mapping.response_field!r} field; "
                f"top-level keys: {keys}",
                remediation="Set request_mapping.response_field in "
                "agentkit.yaml to the dot-path of the answer.",
            )
        return value

    return call


def _logical_model_call(
    target: Target, project: ProjectContext
) -> Callable[[Mapping[str, Any]], Any]:
    from aai_core.context import PlatformContext

    model = PlatformContext(project.settings).providers.model(target.ref)

    def call(inputs: Mapping[str, Any]) -> Any:
        question = _primary_text(inputs, target)
        response = model.generate([{"role": "user", "content": question}])
        return response.content

    return call


def _serving_call(
    target: Target, project: ProjectContext
) -> Callable[[Mapping[str, Any]], Any]:
    endpoint = target.normalized.removeprefix("endpoints:/")
    try:
        from databricks_openai import DatabricksOpenAI
    except ImportError as error:
        raise missing_extra(
            f"Calling the serving endpoint {endpoint!r}", "databricks"
        ) from error
    from aai_core.tags import databricks_ai_gateway_request_headers

    try:
        client = DatabricksOpenAI(
            default_headers=databricks_ai_gateway_request_headers(
                project.settings.resource
            )
        )
    except Exception as error:
        raise TargetResolutionError(
            f"could not create a client for {endpoint!r}: {error}",
            remediation="Authenticate to the workspace first (`az login` "
            "with DATABRICKS_AUTH_TYPE=azure-cli, or the Databricks CLI "
            "profile the platform documents).",
        ) from error

    def call(inputs: Mapping[str, Any]) -> Any:
        question = _primary_text(inputs, target)
        try:
            completion = client.chat.completions.create(
                model=endpoint,
                messages=[{"role": "user", "content": question}],
            )
        except Exception as error:
            raise TargetInvocationError(
                f"serving endpoint {endpoint!r} call failed: {error}",
                remediation="Check the endpoint name and that your identity "
                "holds CAN_QUERY on it.",
            ) from error
        return completion.choices[0].message.content

    return call


def _uc_model_call(target: Target, mlflow: Any) -> Callable[[Mapping[str, Any]], Any]:
    try:
        model = mlflow.pyfunc.load_model(target.normalized)
    except Exception as error:
        raise TargetResolutionError(
            f"could not load {target.normalized}: {error}",
            remediation="For served agents prefer `endpoints:/<name>`; "
            "models:/ loads the model locally through mlflow.pyfunc.",
        ) from error

    def call(inputs: Mapping[str, Any]) -> Any:
        try:
            return model.predict(dict(inputs))
        except Exception as error:
            raise TargetInvocationError(
                f"UC model prediction failed for {target.normalized}: {error}"
            ) from error

    return call


class _CredentialSafeRedirects(urllib.request.HTTPRedirectHandler):
    """Never carry the bearer token to a different origin.

    urllib copies every header onto a redirected request, so a 301/302/303
    to another host hands the ``Authorization`` value to whoever answers
    there — and an endpoint that can be misconfigured can also be
    compromised. Refusing rather than stripping is the honest option: a
    target that redirects elsewhere is no longer the target the project
    configured, so scoring whatever answers would produce evidence about
    something nobody named.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None or not req.get_header("Authorization"):
            return redirected
        if _origin(req.full_url) != _origin(redirected.full_url):
            raise TargetInvocationError(
                f"{req.full_url} redirected to a different origin "
                f"({_origin(redirected.full_url)}) while carrying the token "
                f"from request_mapping.auth_env",
                remediation=(
                    "Point `agent:` at the endpoint that answers directly, or "
                    "drop request_mapping.auth_env if it needs no token."
                ),
            )
        return redirected


def _origin(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}".lower()


_OPENER = urllib.request.build_opener(_CredentialSafeRedirects)


def _urllib_transport(request: urllib.request.Request) -> bytes:
    with _OPENER.open(  # noqa: S310 - https target configured by user
        request, timeout=_HTTP_TIMEOUT_SECONDS
    ) as response:
        return response.read()


def _primary_input(inputs: Mapping[str, Any]) -> Any:
    if len(inputs) == 1:
        return next(iter(inputs.values()))
    for key in _PREFERRED_INPUT_KEYS:
        if key in inputs:
            return inputs[key]
    return dict(inputs)


def _primary_text(inputs: Mapping[str, Any], target: Target) -> str:
    value = _primary_input(inputs)
    if isinstance(value, Mapping):
        raise TargetContractError(
            f"chat target {target.ref!r} needs a single text input; the "
            f"dataset row has keys {sorted(inputs)}",
            remediation="Use a local callable or an HTTP target with "
            "request_mapping for multi-field inputs.",
        )
    return str(value)


def _set_path(document: dict[str, Any], path: str, value: Any) -> None:
    """Place ``value`` at a dot path, honouring numeric segments as indices.

    ``messages.0.content`` is the documented mapping for an OpenAI-shaped
    endpoint, so the segment after ``messages`` has to build a list. It
    also has to *keep* whatever ``extra_body`` already put there —
    replacing the list would turn ``{"messages": [{"role": "user"}]}`` into
    ``{"messages": {"0": ...}}`` and every call would be rejected.
    """

    keys = path.split(".")
    cursor: Any = document
    for position, key in enumerate(keys[:-1]):
        wants_list = keys[position + 1].isdigit()
        child = _child(cursor, key)
        if wants_list and not isinstance(child, list):
            child = []
        elif not wants_list and not isinstance(child, dict):
            child = {}
        _assign(cursor, key, child)
        cursor = child
    _assign(cursor, keys[-1], value)


def _child(cursor: Any, key: str) -> Any:
    if isinstance(cursor, list):
        index = int(key)
        return cursor[index] if index < len(cursor) else None
    return cursor.get(key) if isinstance(cursor, dict) else None


def _assign(cursor: Any, key: str, value: Any) -> None:
    if isinstance(cursor, list):
        index = int(key)
        # Grow rather than fail: a mapping may address messages.1 on a
        # body that only carries messages.0.
        while len(cursor) <= index:
            cursor.append({})
        cursor[index] = value
    else:
        cursor[key] = value


def _get_path(document: Any, path: str) -> tuple[bool, Any]:
    cursor = document
    for key in path.split("."):
        if isinstance(cursor, Mapping) and key in cursor:
            cursor = cursor[key]
        elif isinstance(cursor, list) and key.isdigit() and int(key) < len(cursor):
            cursor = cursor[int(key)]
        else:
            return False, None
    return True, cursor


def _mlflow(mlflow_module: Any | None) -> Any:
    if mlflow_module is not None:
        return mlflow_module
    try:
        import mlflow
    except ImportError as error:
        raise missing_extra("Live evaluation", "genai") from error
    return mlflow
