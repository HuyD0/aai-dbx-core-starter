"""Resolve the thing under evaluation into a callable the harness can score.

The ``agent:`` value resolves by shape — a logical model from
``aai-platform.yml``, a Databricks serving endpoint, a UC registered model,
any HTTP/JSON endpoint (a Foundry hosted agent included), a local Python
callable, or a recorded answer sheet. Detection is pure string/filesystem
logic; adapter construction imports heavy clients lazily.
"""

from __future__ import annotations

import ast
import importlib
import importlib.machinery
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
    (
        "UC model",
        "models:/catalog.schema.agent_model/7  (or @champion)",
    ),
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
        normalized = _validated_model_uri(reference)
        return Target(TargetKind.UC_MODEL, normalized, normalized)
    if reference.startswith(("http://", "https://")):
        normalized = _validated_http_url(reference)
        return Target(TargetKind.HTTP, normalized, normalized)
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


def preflight_target(
    target: Target, *, project: ProjectContext, require_invocation: bool = True
) -> None:
    """Refuse locally knowable target failures without constructing a client.

    This deliberately performs no network or provider-client operation. It
    runs before endpoint identity, prompt registry reads, and confirmation,
    so users are not asked to approve a run whose local target cannot work.
    """

    if target.kind is TargetKind.LOCAL_CALLABLE and require_invocation:
        if target.path is not None:
            _preflight_local_file(target)
        else:
            _preflight_local_module(target)
    elif target.kind is TargetKind.HTTP:
        _validated_http_url(target.normalized)
        if require_invocation:
            _preflight_http(target, project.config.request_mapping)
    elif target.kind is TargetKind.UC_MODEL:
        _validated_model_uri(target.normalized)


def _validated_model_uri(reference: str) -> str:
    """Reject an ambiguous registered-model URI before loading MLflow.

    MLflow 3 interprets ``models:/<value>`` without a selector as a logged
    model id. A three-part Unity Catalog name therefore does *not* name the
    registered model a developer likely intended unless it carries a numeric
    version or an ``@alias``. Logged-model ids remain valid, while UC
    references must select the scored artifact explicitly.
    """

    if not reference.startswith("models:/"):
        raise TargetResolutionError(f"agent {reference!r} is not an MLflow model URI")
    remainder = reference.removeprefix("models:/")
    name = remainder
    selector: str | None = None
    if "/" in remainder:
        name, separator, selector = remainder.partition("/")
        if "/" in selector:
            selector = None
        selector_kind = "version"
    elif "@" in remainder:
        name, separator, selector = remainder.partition("@")
        if "@" in selector:
            selector = None
        selector_kind = "alias"
    else:
        separator = ""
        selector_kind = "version or alias"

    is_three_part_uc_name = len(name.split(".")) == 3 and all(name.split("."))
    if not remainder or (
        is_three_part_uc_name
        and (not separator or not selector or not selector.strip())
    ):
        raise TargetResolutionError(
            f"agent {reference!r} is an incomplete Unity Catalog model reference",
            remediation=(
                "Select the exact registered-model version or alias, for example "
                f"models:/{name}/7 or models:/{name}@champion."
            ),
        )
    if (
        is_three_part_uc_name
        and selector_kind == "version"
        and selector is not None
        and not selector.isdigit()
    ):
        raise TargetResolutionError(
            f"agent {reference!r} has a nonnumeric Unity Catalog model version",
            remediation=(
                f"Use a numeric version such as models:/{name}/7. "
                f"To select an alias, use models:/{name}@champion."
            ),
        )
    if separator and (not name or selector is None or not selector.strip()):
        raise TargetResolutionError(
            f"agent {reference!r} has an invalid model {selector_kind} selector"
        )
    return reference


def _preflight_local_file(target: Target) -> None:
    try:
        source = target.path.read_text(encoding="utf-8")  # type: ignore[union-attr]
        module = ast.parse(source, filename=str(target.path))
    except (OSError, SyntaxError) as error:
        raise TargetResolutionError(
            f"could not inspect local agent {target.ref!r}: {error}"
        ) from error

    attribute = target.attribute or ""
    binding = "absent"
    dynamic_attributes = False
    for node in module.body:
        outcome = _top_level_binding(node, attribute)
        if outcome is not None:
            binding = outcome
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == "__getattr__":
                dynamic_attributes = True
            if _definition_has_import_time_effect(node):
                binding = "unknown"
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is not None and any(
                isinstance(item, (ast.Call, ast.NamedExpr))
                for item in ast.walk(node.value)
            ):
                binding = "unknown"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and any(
                alias.name == "*" for alias in node.names
            ):
                binding = "unknown"
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            # Module docstring or another inert literal.
            continue
        elif isinstance(node, ast.Pass):
            continue
        else:
            # Conditional definitions, try/except imports, exec-like calls,
            # module __getattr__, and other dynamic module code make absence
            # unknowable without executing application code.
            binding = "unknown"

    if binding in {"callable", "unknown"}:
        return
    if binding == "async":
        raise TargetContractError(
            f"{target.ref!r}: {attribute!r} is async, but AgentKit's "
            "local target adapter is synchronous",
            remediation="Expose a synchronous wrapper that returns the final answer.",
        )
    if binding == "absent" and dynamic_attributes:
        return
    detail = (
        "is declared but is not callable"
        if binding == "non-callable"
        else "is not declared"
    )
    raise TargetResolutionError(
        f"{target.ref!r}: {attribute!r} {detail} in {target.path}"
    )


def _preflight_local_module(target: Target) -> None:
    module_name = target.ref.rpartition(":")[0]
    parts = module_name.split(".")
    specification = importlib.machinery.PathFinder.find_spec(parts[0])
    if specification is None:
        try:
            specification = importlib.util.find_spec(parts[0])
        except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
            # A custom finder or a pre-populated sys.modules entry can make
            # the answer unknowable without importing application code.
            return
    if specification is None:
        raise TargetResolutionError(
            f"could not find local agent module {module_name!r}"
        )

    qualified = parts[0]
    for part in parts[1:]:
        locations = specification.submodule_search_locations
        if locations is None:
            return
        qualified = f"{qualified}.{part}"
        try:
            specification = importlib.machinery.PathFinder.find_spec(
                qualified, list(locations)
            )
        except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
            return
        if specification is None:
            # Importing a package can extend or replace __path__ (for example
            # via pkgutil.extend_path), so a miss in its static location is
            # not proof that the nested module cannot be imported.
            return
    origin = getattr(specification, "origin", None)
    if isinstance(origin, str) and origin.endswith(".py"):
        _preflight_local_file(
            Target(
                kind=target.kind,
                ref=target.ref,
                normalized=target.normalized,
                path=Path(origin),
                attribute=target.attribute,
            )
        )


def _top_level_binding(node: ast.stmt, name: str) -> str | None:
    """Classify a statement's final direct binding without executing it.

    Only outcomes Python makes syntactically certain are rejected. Decorators,
    imports, destructuring, calls, and control flow stay unknown and are left
    to the runtime adapter.
    """

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if node.name != name:
            return None
        if node.decorator_list:
            return "unknown"
        return "async" if isinstance(node, ast.AsyncFunctionDef) else "callable"
    if isinstance(node, ast.Assign):
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return _assigned_value_kind(node.value)
        if any(_binds_name(target, name) for target in node.targets):
            return "unknown"
        return None
    if isinstance(node, ast.AnnAssign):
        if not isinstance(node.target, ast.Name) or node.target.id != name:
            return None
        # A bare annotation does not bind the name at runtime.
        return None if node.value is None else _assigned_value_kind(node.value)
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        if any(
            (alias.asname or alias.name.rpartition(".")[2]) == name
            for alias in node.names
        ):
            return "unknown"
        return None
    if isinstance(node, ast.AugAssign) and _binds_name(node.target, name):
        return "unknown"
    if isinstance(node, ast.Delete) and any(
        _binds_name(target, name) for target in node.targets
    ):
        return "absent"
    return None


def _assigned_value_kind(value: ast.expr) -> str:
    if isinstance(value, ast.Lambda):
        return "callable"
    if isinstance(value, (ast.Constant, ast.Dict, ast.List, ast.Set, ast.Tuple)):
        return "non-callable"
    return "unknown"


def _definition_has_import_time_effect(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> bool:
    """Ignore dormant function bodies while inspecting import-time code."""

    # Applying even a bare-name decorator executes arbitrary code.
    if node.decorator_list:
        return True
    type_parameters = tuple(getattr(node, "type_params", ()))
    if isinstance(node, ast.ClassDef):
        # Resolving a base/metaclass and constructing a subclass can invoke
        # user code. The class body itself also executes during import.
        if node.bases or node.keywords:
            return True
        if any(_expression_has_effect(item) for item in type_parameters):
            return True
        return any(_class_statement_has_import_time_effect(item) for item in node.body)

    expressions: list[ast.AST] = [*node.args.defaults, *type_parameters]
    expressions.extend(item for item in node.args.kw_defaults if item is not None)
    expressions.extend(
        argument.annotation
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        if argument.annotation is not None
    )
    for argument in (node.args.vararg, node.args.kwarg):
        if argument is not None and argument.annotation is not None:
            expressions.append(argument.annotation)
    if node.returns is not None:
        expressions.append(node.returns)
    return any(_expression_has_effect(item) for item in expressions)


def _class_statement_has_import_time_effect(node: ast.stmt) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return _definition_has_import_time_effect(node)
    if isinstance(node, (ast.Pass, ast.Import, ast.ImportFrom)):
        return False
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return False
    if isinstance(node, ast.Assign):
        simple_targets = all(isinstance(target, ast.Name) for target in node.targets)
        return not simple_targets or _expression_has_effect(node.value)
    if isinstance(node, ast.AnnAssign):
        if not isinstance(node.target, ast.Name):
            return True
        return _expression_has_effect(node.annotation) or (
            node.value is not None and _expression_has_effect(node.value)
        )
    # Control flow, global writes, comprehensions used as statements, and
    # exec-like constructs can affect the module namespace when the class
    # body runs. Runtime loading remains authoritative for those shapes.
    return True


def _expression_has_effect(node: ast.AST) -> bool:
    """Find executed calls/bindings without descending into lambda bodies."""

    class EffectVisitor(ast.NodeVisitor):
        found = False

        def visit_Call(self, call: ast.Call) -> None:  # noqa: N802
            self.found = True

        def visit_NamedExpr(self, expression: ast.NamedExpr) -> None:  # noqa: N802
            self.found = True

        def visit_Lambda(self, expression: ast.Lambda) -> None:  # noqa: N802
            for default in expression.args.defaults:
                self.visit(default)
            for default in expression.args.kw_defaults:
                if default is not None:
                    self.visit(default)

    visitor = EffectVisitor()
    visitor.visit(node)
    return visitor.found


def _binds_name(target: ast.AST, name: str) -> bool:
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_binds_name(item, name) for item in target.elts)
    if isinstance(target, ast.Starred):
        return _binds_name(target.value, name)
    return False


def _validated_http_url(reference: str) -> str:
    """Return a tag-safe invocation URL or reject it before any cloud read."""

    try:
        parts = urllib.parse.urlsplit(reference)
        hostname = parts.hostname
        # Accessing port validates malformed/non-numeric/out-of-range values.
        _ = parts.port
    except ValueError as error:
        raise TargetResolutionError("agent HTTP URL is malformed") from error
    if parts.scheme not in {"http", "https"} or not hostname:
        raise TargetResolutionError(
            "agent HTTP URL needs an http(s) scheme and a hostname"
        )
    if any(character.isspace() or ord(character) < 32 for character in reference):
        raise TargetResolutionError("agent HTTP URL contains invalid whitespace")
    if parts.username is not None or parts.password is not None:
        raise TargetResolutionError(
            "agent HTTP URL must not contain user information",
            remediation=(
                "Put the credential in request_mapping.auth_env instead; "
                "URLs are written to MLflow tags and results evidence."
            ),
        )
    if parts.query or parts.fragment:
        raise TargetResolutionError(
            "agent HTTP URL must not contain a query string or fragment",
            remediation=(
                "Put credentials in request_mapping.auth_env and request "
                "values in request_mapping.extra_body; URLs are written to "
                "MLflow tags and results evidence."
            ),
        )
    return urllib.parse.urlunsplit(
        (parts.scheme.lower(), parts.netloc, parts.path, "", "")
    )


def _preflight_http(target: Target, mapping: RequestMapping) -> None:
    _validated_http_url(target.normalized)
    if not mapping.auth_env:
        return
    _require_encrypted_transport(target, mapping.auth_env)
    if not os.environ.get(mapping.auth_env):
        raise TargetInvocationError(
            f"request_mapping.auth_env names {mapping.auth_env!r} but the "
            "variable is not set",
            remediation=f"export {mapping.auth_env}=<token> before running.",
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
    if inspect.iscoroutinefunction(function):
        raise TargetContractError(
            f"{target.ref!r} is async, but AgentKit's local target adapter "
            "is synchronous",
            remediation="Expose a synchronous wrapper that returns the final answer.",
        )
    try:
        signature = inspect.signature(function)
    except Exception as error:
        raise TargetContractError(
            f"could not inspect the callable signature for {target.ref!r}",
            remediation=(
                "Expose a Python callable with an inspectable signature so "
                "AgentKit can map dataset inputs before evaluation."
            ),
        ) from error
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
                parameter = parameters[0]
                name = parameter.name
                if len(inputs) == 1:
                    value = (
                        inputs[name] if name in inputs else next(iter(inputs.values()))
                    )
                    if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                        arguments: tuple[Any, ...] = ()
                        keywords = {name: value}
                    else:
                        arguments = (value,)
                        keywords = {}
                else:
                    # Do not discard row fields merely because the target's
                    # declared argument is present. Binding the complete
                    # mapping makes an incomplete agent contract fail before
                    # the target or any judges are invoked.
                    arguments = ()
                    keywords = dict(inputs)
            else:
                arguments = ()
                keywords = dict(inputs)
            try:
                signature.bind(*arguments, **keywords)
            except TypeError as error:
                raise TargetContractError(
                    f"{target.ref!r} cannot accept dataset input keys "
                    f"{sorted(inputs)}: {error}",
                    remediation=(
                        "Align the callable parameters with the dataset's "
                        "inputs fields."
                    ),
                ) from error
            result = function(*arguments, **keywords)
            if inspect.isawaitable(result):
                close = getattr(result, "close", None)
                if callable(close):
                    close()
                raise TargetContractError(
                    f"{target.ref!r} returned an awaitable, but AgentKit's "
                    "local target adapter is synchronous",
                    remediation=(
                        "Await the operation inside a synchronous wrapper and "
                        "return the final answer."
                    ),
                )
            return result
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
    _preflight_http(target, mapping)

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
            ) from None
        except urllib.error.URLError:
            raise TargetInvocationError(
                f"could not reach {target.normalized}"
            ) from None
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
        source = _redirect_origin(req.full_url)
        destination = _redirect_origin(newurl)
        credentialed = bool(req.get_header("Authorization"))
        if credentialed and (
            source is None
            or destination is None
            or source != destination
            or _url_has_userinfo(newurl)
        ):
            destination_text = _format_origin(destination)
            raise TargetInvocationError(
                "credentialed HTTP target redirected to a different origin"
                f"{destination_text}; redirect refused",
                remediation=(
                    "Point `agent:` at the endpoint that answers directly, or "
                    "drop request_mapping.auth_env if it needs no token."
                ),
            )
        try:
            return super().redirect_request(req, fp, code, msg, headers, newurl)
        except (TypeError, ValueError, urllib.error.HTTPError):
            raise TargetInvocationError(
                "HTTP target redirect could not be followed"
                f"{_format_origin(destination)}"
            ) from None


def _redirect_origin(url: str) -> tuple[str, str, int] | None:
    try:
        parts = urllib.parse.urlsplit(url)
        scheme = parts.scheme.lower()
        hostname = parts.hostname
        if scheme not in {"http", "https"} or not hostname:
            return None
        port = parts.port or (443 if scheme == "https" else 80)
    except (AttributeError, TypeError, ValueError):
        return None
    return scheme, hostname.lower(), port


def _format_origin(origin: tuple[str, str, int] | None) -> str:
    if origin is None:
        return ""
    scheme, hostname, port = origin
    safe_hostname = f"[{hostname}]" if ":" in hostname else hostname
    return f" ({scheme}://{safe_hostname}:{port})"


def _url_has_userinfo(url: str) -> bool:
    try:
        parts = urllib.parse.urlsplit(url)
        return parts.username is not None or parts.password is not None
    except (AttributeError, TypeError, ValueError):
        return True


_OPENER = urllib.request.build_opener(_CredentialSafeRedirects)


def _urllib_transport(request: urllib.request.Request) -> bytes:
    with _OPENER.open(  # noqa: S310 - https target configured by user
        request, timeout=_HTTP_TIMEOUT_SECONDS
    ) as response:
        return response.read()


def _primary_input(inputs: Mapping[str, Any]) -> Any:
    if len(inputs) == 1:
        return next(iter(inputs.values()))
    # An HTTP request mapping is the adapter for structured input.  Preserve
    # the complete row rather than choosing a familiar-looking key and
    # silently dropping context, history, tenant, or another required field.
    # Chat-style adapters call `_primary_text`, which rejects this mapping.
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
