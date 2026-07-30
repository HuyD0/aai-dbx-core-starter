"""Fail-closed orchestration boundary for governed Databricks job launches.

The Hub records evaluation and promotion workflow state separately from the Jobs
service.  This module is the narrow boundary between those workflows and a job
runner:

* requests are strict, immutable, and limited to non-secret job parameters;
* an unavailable runner raises instead of manufacturing a successful run;
* the local preview runner is deterministic and clearly identifies preview runs;
* the Databricks adapter imports the SDK and creates its client only when called.

The live adapter deliberately relies on Databricks unified authentication.  In a
Databricks App that means the runtime-provided app service-principal environment;
this module never reads, copies, or logs the injected OAuth secret.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from enum import StrEnum
from threading import Lock
from types import MappingProxyType
from typing import Annotated, Any, Protocol, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

_MAX_JOB_ID = 9_223_372_036_854_775_807
_MAX_PARAMETERS = 64
_MAX_PARAMETER_BYTES = 10_000
_PARAMETER_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_IDEMPOTENCY_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SENSITIVE_KEY_PARTS = frozenset(
    {
        "apikey",
        "authorization",
        "clientsecret",
        "credential",
        "credentials",
        "password",
        "passwd",
        "privatekey",
        "refreshtoken",
        "secret",
        "token",
    }
)
_SENSITIVE_VALUE = re.compile(
    r"(?i)(?:"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|(?:password|passwd|client[_-]?secret|access[_-]?token|refresh[_-]?token"
    r"|api[_-]?key|authorization)\s*[:=]"
    r"|(?:bearer|basic)\s+[A-Za-z0-9+/_.=-]+"
    r"|(?<![A-Za-z0-9])(?:dapi|github_pat_|gh[oprsu]_|sk-)[A-Za-z0-9_-]{8,}"
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r"|(?:[?&]sig=|accountkey=|sharedaccesssignature=)"
    r")"
)


class _JobModel(BaseModel):
    """Strict defaults for request, response, and capability boundaries."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        hide_input_in_errors=True,
    )


class JobRunnerMode(StrEnum):
    UNAVAILABLE = "unavailable"
    PREVIEW = "preview"
    DATABRICKS = "databricks"


class JobRunState(StrEnum):
    QUEUED = "QUEUED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    TERMINATING = "TERMINATING"
    TERMINATED = "TERMINATED"
    SKIPPED = "SKIPPED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    BLOCKED = "BLOCKED"
    WAITING_FOR_RETRY = "WAITING_FOR_RETRY"
    UNKNOWN = "UNKNOWN"


class JobExecutionCapability(_JobModel):
    """Projection used by APIs to distinguish preview from live execution."""

    mode: JobRunnerMode
    enabled: bool
    remote_execution: bool
    detail: Annotated[str, StringConstraints(min_length=1, max_length=256)]

    @model_validator(mode="after")
    def mode_matches_flags(self) -> Self:
        if self.mode is JobRunnerMode.UNAVAILABLE and (
            self.enabled or self.remote_execution
        ):
            raise ValueError("an unavailable runner cannot execute jobs")
        if self.mode is JobRunnerMode.PREVIEW and (
            not self.enabled or self.remote_execution
        ):
            raise ValueError("a preview runner must be enabled and local")
        if self.mode is JobRunnerMode.DATABRICKS and (
            not self.enabled or not self.remote_execution
        ):
            raise ValueError("a Databricks runner must execute remote jobs")
        return self


class JobLaunchRequest(_JobModel):
    """Validated arguments for one idempotent Jobs ``run-now`` request."""

    job_id: str
    idempotency_token: str
    parameters: Mapping[str, str] = Field(default_factory=dict)

    @field_validator("job_id", mode="before")
    @classmethod
    def validate_job_id(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        job_id = value.strip()
        if not job_id.isascii() or not job_id.isdigit():
            raise ValueError("job_id must be a positive numeric Databricks job ID")
        if job_id.startswith("0") or int(job_id) > _MAX_JOB_ID:
            raise ValueError("job_id must be a positive signed 64-bit integer")
        return job_id

    @field_validator("idempotency_token", mode="before")
    @classmethod
    def validate_idempotency_token(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        token = value.strip()
        if not _IDEMPOTENCY_TOKEN.fullmatch(token):
            raise ValueError("idempotency_token must be 1-64 URL-safe ASCII characters")
        return token

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: Any) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise ValueError("parameters must be a string-to-string mapping")
        if len(value) > _MAX_PARAMETERS:
            raise ValueError(f"parameters may contain at most {_MAX_PARAMETERS} items")

        validated: dict[str, str] = {}
        for key, parameter_value in value.items():
            if not isinstance(key, str) or not isinstance(parameter_value, str):
                raise ValueError("parameter keys and values must be strings")
            if not _PARAMETER_KEY.fullmatch(key):
                raise ValueError(
                    "parameter keys must start with a letter and contain only "
                    "letters, digits, dot, hyphen, or underscore"
                )
            normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
            if any(part in normalized_key for part in _SENSITIVE_KEY_PARTS):
                raise ValueError("credential-bearing job parameters are forbidden")
            if (
                not parameter_value
                or len(parameter_value) > 2_048
                or not parameter_value.isascii()
                or not parameter_value.isprintable()
            ):
                raise ValueError(
                    "parameter values must be 1-2048 printable ASCII characters"
                )
            if contains_credential_material(parameter_value):
                raise ValueError("credential-bearing job parameters are forbidden")
            validated[key] = parameter_value

        canonical = dict(sorted(validated.items()))
        encoded = json.dumps(
            canonical,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        if len(encoded) > _MAX_PARAMETER_BYTES:
            raise ValueError(
                f"encoded parameters must not exceed {_MAX_PARAMETER_BYTES} bytes"
            )
        return canonical

    @field_validator("parameters")
    @classmethod
    def freeze_parameters(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_serializer("parameters")
    def serialize_parameters(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)


class JobLaunchResult(_JobModel):
    """The durable identifiers returned immediately after a launch request."""

    run_id: str
    run_page_url: str | None
    state: JobRunState
    preview: bool = False

    @field_validator("run_id", mode="before")
    @classmethod
    def validate_run_id(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        run_id = value.strip()
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run_id has an unsupported format")
        return run_id

    @field_validator("run_page_url", mode="before")
    @classmethod
    def validate_run_page_url(cls, value: Any) -> Any:
        if value is None or not isinstance(value, str):
            return value
        url = value.strip()
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(
                "run_page_url must be an absolute credential-free HTTPS URL"
            )
        return url


class JobExecutionError(RuntimeError):
    """Base class for safe-to-render job orchestration errors."""


class JobRunnerUnavailableError(JobExecutionError):
    """Raised when no real or explicit preview runner can accept a launch."""


class JobLaunchError(JobExecutionError):
    """Raised when the remote Jobs service does not accept a launch."""


class JobIdempotencyConflictError(JobExecutionError):
    """Raised when a preview token is reused for different launch arguments."""


class JobRunner(Protocol):
    """Execution capability consumed by evaluation and promotion workflows."""

    @property
    def capability(self) -> JobExecutionCapability: ...

    def launch(self, request: JobLaunchRequest) -> JobLaunchResult: ...


_UNAVAILABLE_CAPABILITY = JobExecutionCapability(
    mode=JobRunnerMode.UNAVAILABLE,
    enabled=False,
    remote_execution=False,
    detail="Databricks Jobs execution is not configured.",
)
_PREVIEW_CAPABILITY = JobExecutionCapability(
    mode=JobRunnerMode.PREVIEW,
    enabled=True,
    remote_execution=False,
    detail="Local preview only; no Databricks job is launched.",
)
_DATABRICKS_CAPABILITY = JobExecutionCapability(
    mode=JobRunnerMode.DATABRICKS,
    enabled=True,
    remote_execution=True,
    detail="Launches Databricks Jobs with the app service principal.",
)


class UnavailableJobRunner:
    """Fail closed when job execution has not been explicitly configured."""

    @property
    def capability(self) -> JobExecutionCapability:
        return _UNAVAILABLE_CAPABILITY

    def launch(self, request: JobLaunchRequest) -> JobLaunchResult:
        del request
        raise JobRunnerUnavailableError("Databricks Jobs execution is not configured.")


class RecordingJobRunner:
    """Deterministic, thread-safe runner for tests and explicit local preview."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: list[JobLaunchRequest] = []
        self._results_by_token: dict[str, tuple[JobLaunchRequest, JobLaunchResult]] = {}

    @property
    def capability(self) -> JobExecutionCapability:
        return _PREVIEW_CAPABILITY

    @property
    def requests(self) -> tuple[JobLaunchRequest, ...]:
        with self._lock:
            return tuple(self._requests)

    def launch(self, request: JobLaunchRequest) -> JobLaunchResult:
        with self._lock:
            previous = self._results_by_token.get(request.idempotency_token)
            if previous is not None:
                previous_request, previous_result = previous
                if previous_request != request:
                    raise JobIdempotencyConflictError(
                        "idempotency token was already used for another job launch"
                    )
                return previous_result

            sequence = len(self._requests) + 1
            run_id = f"preview-run-{sequence:06d}"
            result = JobLaunchResult(
                run_id=run_id,
                run_page_url=(
                    f"https://local.invalid/jobs/{request.job_id}/runs/{run_id}"
                ),
                state=JobRunState.QUEUED,
                preview=True,
            )
            self._requests.append(request)
            self._results_by_token[request.idempotency_token] = (request, result)
            return result


def _default_workspace_client() -> Any:
    """Build a client lazily from Databricks unified-auth environment variables."""

    from databricks.sdk import WorkspaceClient

    # Do not pass the runtime OAuth secret. Unified authentication reads the app
    # service-principal environment directly, keeping credential values out of
    # arguments, tracebacks, and this module's state.
    return WorkspaceClient()


class DatabricksJobRunner:
    """Thin, non-blocking adapter over Databricks Jobs ``run-now``."""

    def __init__(self, client_factory: Callable[[], Any] | None = None) -> None:
        # A factory, rather than a client, keeps SDK import and authentication out of
        # module import and runner construction. The default factory stores no secret.
        self._client_factory = client_factory or _default_workspace_client

    @property
    def capability(self) -> JobExecutionCapability:
        return _DATABRICKS_CAPABILITY

    def launch(self, request: JobLaunchRequest) -> JobLaunchResult:
        try:
            client = self._client_factory()
        except Exception:
            raise JobRunnerUnavailableError(
                "Databricks Jobs execution could not authenticate."
            ) from None

        try:
            waiter = client.jobs.run_now(
                job_id=int(request.job_id),
                idempotency_token=request.idempotency_token,
                job_parameters=dict(request.parameters),
            )
        except Exception:
            raise JobLaunchError(
                "Databricks Jobs did not accept the launch request."
            ) from None

        raw_run_id = getattr(waiter, "run_id", None)
        if raw_run_id is None:
            response = getattr(waiter, "response", None)
            raw_run_id = getattr(response, "run_id", None)
        if raw_run_id is None:
            raise JobLaunchError(
                "Databricks Jobs accepted the request without returning a run ID."
            )

        run_id = str(raw_run_id)
        run_page_url: str | None = None
        state = JobRunState.QUEUED
        try:
            run = client.jobs.get_run(run_id=int(run_id))
        except Exception:
            # The launch has already succeeded and returned a stable run ID. A
            # transient read-after-write failure must not encourage a duplicate retry;
            # callers can poll the run later.
            pass
        else:
            raw_page_url = getattr(run, "run_page_url", None)
            if isinstance(raw_page_url, str):
                run_page_url = raw_page_url
            state = _job_run_state(getattr(run, "state", None))

        return JobLaunchResult(
            run_id=run_id,
            run_page_url=run_page_url,
            state=state,
            preview=False,
        )


def contains_credential_material(value: str) -> bool:
    """Return whether untrusted text contains a strong credential signature."""

    if _SENSITIVE_VALUE.search(value):
        return True
    if "://" not in value:
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return True
    return parsed.username is not None or parsed.password is not None


def _job_run_state(raw_state: Any) -> JobRunState:
    life_cycle_state = getattr(raw_state, "life_cycle_state", None)
    raw_value = getattr(life_cycle_state, "value", life_cycle_state)
    if not isinstance(raw_value, str):
        return JobRunState.UNKNOWN
    try:
        return JobRunState(raw_value.upper())
    except ValueError:
        return JobRunState.UNKNOWN


__all__ = [
    "DatabricksJobRunner",
    "JobExecutionCapability",
    "JobExecutionError",
    "JobIdempotencyConflictError",
    "JobLaunchError",
    "JobLaunchRequest",
    "JobLaunchResult",
    "JobRunState",
    "JobRunner",
    "JobRunnerMode",
    "JobRunnerUnavailableError",
    "RecordingJobRunner",
    "UnavailableJobRunner",
]
