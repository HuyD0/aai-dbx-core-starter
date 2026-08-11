"""The results record — one scoring run's evidence, written to disk.

Every completed scoring command writes one under ``.aai/agentkit/results/``
and atomically points the local gate at it. ``gate`` and ``evidence`` read
that completed attempt rather than re-running an evaluation; a failed newer
attempt cannot silently fall back to an older passing record.

A recorded run also attaches the record to its MLflow run as an artifact.
The local directory is whatever filesystem the run happened on — for the
deployment-job gate that is an ephemeral job cluster the approver will
never see. Attaching it to the run makes ``agentkit evidence --run <id>``
the answer to "show me what this version scored", from any machine.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Literal, cast

try:  # POSIX is the deployment/runtime platform; keep Windows import-safe.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on Windows
    fcntl = None  # type: ignore[assignment]

from pydantic import (
    Field,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from aai_core.agentkit.baseline import BaselineDataset, BaselineScope, BaselineVersions
from aai_core.agentkit.errors import ConfigError
from aai_core.agentkit.statistics import StatisticalEvidence
from aai_core.contracts import ContractModel, freeze_value, thaw_value
from aai_core.evaluation import MetricRule

RESULTS_GLOB = "*.json"
RESULTS_ARTIFACT_DIR = "agentkit"
RESULTS_ARTIFACT_FILE = "results.json"
RESULTS_ARTIFACT_PATH = f"{RESULTS_ARTIFACT_DIR}/{RESULTS_ARTIFACT_FILE}"
RESULTS_ATTEMPT_FILE = ".latest-attempt"
RESULTS_ATTEMPT_STATE_PREFIX = ".attempt-"
RESULTS_ATTEMPT_LOCK_FILE = ".attempt-lock"
RESULTS_ATTEMPT_TRANSITION_FILE = ".attempt-transition"


class ResultsAttemptPointer(ContractModel):
    schema_version: Literal[1] = 1
    attempt_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    command: str = Field(min_length=1)


class ResultsAttempt(ContractModel):
    """Atomic pointer to the only local result a bare gate may consume."""

    schema_version: Literal[1] = 1
    attempt_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    command: str = Field(min_length=1)
    status: Literal["pending", "complete"]
    results_file: str | None = None
    results_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def completed_attempt_has_bound_results(self) -> ResultsAttempt:
        if self.status == "complete":
            if not self.results_file or not self.results_sha256:
                raise ValueError("a complete attempt needs its results file and digest")
            if Path(self.results_file).name != self.results_file:
                raise ValueError("results_file must be a basename")
        elif self.results_file is not None or self.results_sha256 is not None:
            raise ValueError("a pending attempt cannot name completed results")
        return self


class ResultsRecord(ContractModel):
    """What one ``compare``/``smoke``/``eval`` run produced."""

    schema_version: Literal[1] = 1
    # New records bind the durable evidence to the local attempt that
    # produced it.  ``None`` keeps already-published v1 artifacts readable;
    # a result completed through the attempt protocol must always set it.
    attempt_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    command: str = Field(min_length=1)
    recorded_at: str = Field(min_length=1)
    run_id: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    agent: str = Field(min_length=1)
    dataset: BaselineDataset
    scope: BaselineScope
    mode: str = Field(min_length=1)
    metrics: Mapping[str, float] = Field(default_factory=dict)
    # Per-row numeric scorer values in dataset order. Content never travels
    # with them; they exist solely to make paired uncertainty reproducible.
    metric_samples: Mapping[str, tuple[float | None, ...]] = Field(default_factory=dict)
    statistics: StatisticalEvidence | None = None
    versions: BaselineVersions
    baseline_run_id: str | None = None
    baseline_metrics: Mapping[str, float] = Field(default_factory=dict)
    # The baseline's own lineage, snapshotted. Reading it from the local
    # `evals/baseline.json` at render time would let evidence pair this
    # run's deltas with a baseline that has since been re-established.
    baseline_recorded_at: str | None = None
    baseline_dataset_digest: str | None = None
    established_baseline: bool = False
    # The rules this run was actually judged by. A record is evidence, and
    # evidence is only evidence if reopening it cannot change the verdict:
    # without this, relaxing a threshold in agentkit.yaml turns a failed
    # run into a passing one with no re-scoring.
    policy_rules: tuple[MetricRule, ...] = ()
    allow_missing_regression_baseline: bool = False
    decision: str = Field(min_length=1)
    change_id: str = Field(min_length=1)
    gate_passed: bool
    gate_failures: tuple[Mapping[str, str], ...] = ()
    warnings: tuple[str, ...] = ()
    judges_enabled: bool = False

    @field_validator("gate_failures", "warnings", "policy_rules", mode="before")
    @classmethod
    def coerce_sequences(cls, value: Any) -> Any:
        # Round-tripping through JSON turns tuples into lists; strict mode
        # would otherwise refuse to reload a record it just wrote.
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("metrics", "baseline_metrics", mode="before")
    @classmethod
    def coerce_metrics(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: float(item) if isinstance(item, int) else item
                for key, item in value.items()
                if not isinstance(item, bool)
            }
        return value

    @field_validator("metric_samples", mode="before")
    @classmethod
    def coerce_metric_samples(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {
            str(name): tuple(samples) if isinstance(samples, list | tuple) else samples
            for name, samples in value.items()
        }

    @field_validator("metrics", "baseline_metrics", mode="after")
    @classmethod
    def freeze_metrics(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        non_finite = sorted(
            name for name, item in value.items() if not math.isfinite(item)
        )
        if non_finite:
            raise ValueError(
                "metric values must be finite (invalid: " + ", ".join(non_finite) + ")"
            )
        return cast(Mapping[str, float], freeze_value(value))

    @field_validator("metric_samples", mode="after")
    @classmethod
    def freeze_metric_samples(
        cls, value: Mapping[str, tuple[float | None, ...]]
    ) -> Mapping[str, tuple[float | None, ...]]:
        for name, samples in value.items():
            if not name.strip():
                raise ValueError("metric sample keys must name a metric")
            if any(item is not None and not math.isfinite(item) for item in samples):
                raise ValueError(f"metric samples for {name!r} must be finite")
        return cast(Mapping[str, tuple[float | None, ...]], freeze_value(value))

    @field_validator("gate_failures", mode="after")
    @classmethod
    def freeze_failures(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
        return cast(tuple[Any, ...], freeze_value(value))

    @field_serializer("metrics", "baseline_metrics")
    def serialize_metrics(self, value: Mapping[str, float]) -> dict[str, float]:
        return cast(dict[str, float], thaw_value(value))

    @field_serializer("metric_samples")
    def serialize_metric_samples(
        self, value: Mapping[str, tuple[float | None, ...]]
    ) -> dict[str, list[float | None]]:
        return cast(dict[str, list[float | None]], thaw_value(value))

    @field_serializer("gate_failures")
    def serialize_failures(self, value: tuple[Any, ...]) -> list[dict[str, str]]:
        return cast(list[dict[str, str]], thaw_value(value))

    @property
    def is_comparison(self) -> bool:
        """True when this run named what it was scored against.

        A run that established the baseline is itself the named reference;
        anything else must link a baseline run or carry baseline metrics.
        """

        return bool(
            self.established_baseline or self.baseline_run_id or self.baseline_metrics
        )


def _results_document(record: ResultsRecord) -> str:
    return (
        json.dumps(
            record.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _write_results_file(path: Path, record: ResultsRecord) -> None:
    path.write_text(_results_document(record), encoding="utf-8")


def write_results(
    directory: Path,
    record: ResultsRecord,
    *,
    attempt: ResultsAttempt | None = None,
) -> Path:
    """Atomically expose a completed local results record."""

    directory.mkdir(parents=True, exist_ok=True)
    if attempt is not None:
        _require_attempt_binding(record, attempt)
        owner = attempt.attempt_id
    else:
        # Legacy/test callers that do not participate in the attempt protocol
        # still receive a collision-free path.  Production scoring always
        # passes its attempt and therefore uses the attempt id as the owner.
        owner = uuid.uuid4().hex
    stamp = record.recorded_at.replace(":", "").replace("-", "")
    path = directory / f"{stamp}-{record.command}-{owner}.json"
    descriptor, scratch_name = tempfile.mkstemp(
        dir=directory, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    scratch = Path(scratch_name)
    try:
        _write_results_file(scratch, record)
        os.replace(scratch, path)
    finally:
        scratch.unlink(missing_ok=True)
    return path


def begin_results_attempt(directory: Path, *, command: str) -> ResultsAttempt:
    """Invalidate stale gate evidence before an evaluation can start."""

    attempt = ResultsAttempt(
        attempt_id=uuid.uuid4().hex,
        command=command,
        status="pending",
    )
    pointer = ResultsAttemptPointer(
        attempt_id=attempt.attempt_id,
        command=attempt.command,
    )
    pointer_path = directory / RESULTS_ATTEMPT_FILE
    transition_path = directory / RESULTS_ATTEMPT_TRANSITION_FILE
    with _attempt_protocol_lock(directory, shared=False):
        # Moving the prior pointer is the first mutation: it atomically makes
        # an old pass ineligible without depending on a new write succeeding.
        # Rewriting the marker binds it to this attempt; if that or any later
        # step fails, the marker remains and readers refuse.
        with suppress(FileNotFoundError):
            os.replace(pointer_path, transition_path)
        _write_contract_file(transition_path, pointer.model_dump())
        _write_attempt_state(directory, attempt)
        _write_contract_file(pointer_path, pointer.model_dump())
        transition_path.unlink()
    return attempt


def complete_results_attempt(
    directory: Path, attempt: ResultsAttempt, results_path: Path
) -> None:
    """Atomically bind the newest attempt to its completed result bytes."""

    if results_path.parent.resolve() != directory.resolve():
        raise ConfigError(
            "completed results must live in the project results directory"
        )
    document = _read_results_bytes(results_path)
    record = _parse_results_bytes(results_path, document)
    _require_attempt_binding(record, attempt)
    completed = ResultsAttempt(
        attempt_id=attempt.attempt_id,
        command=attempt.command,
        status="complete",
        results_file=results_path.name,
        results_sha256=hashlib.sha256(document).hexdigest(),
    )
    _write_attempt_state(directory, completed)


def load_gate_results(directory: Path) -> tuple[ResultsRecord, Path] | None:
    """The result bound to the latest attempt, with legacy fallback."""

    # Hold the same lock as begin_results_attempt across the entire evidence
    # read. A reader therefore returns the old completed attempt before a new
    # begin, or observes the new pending attempt after it — never a mixture.
    with _attempt_protocol_lock(directory, shared=True):
        return _load_gate_results_locked(directory)


def _load_gate_results_locked(directory: Path) -> tuple[ResultsRecord, Path] | None:
    transition_path = directory / RESULTS_ATTEMPT_TRANSITION_FILE
    if _path_exists(transition_path, purpose="evaluation-attempt transition"):
        raise ConfigError(
            "the latest evaluation attempt did not finish initializing",
            remediation=(
                "Fix the local results-directory error and rerun the evaluation "
                "before invoking `agentkit gate`."
            ),
        )
    pointer_path = directory / RESULTS_ATTEMPT_FILE
    try:
        pointer_path.stat()
    except FileNotFoundError:
        if _has_attempt_metadata(directory):
            raise ConfigError(
                "evaluation-attempt metadata exists but its latest pointer is "
                "missing",
                remediation=(
                    "Rerun the evaluation before invoking `agentkit gate`; "
                    "legacy result fallback is only available to directories "
                    "that have never used the attempt protocol."
                ),
            ) from None
        return load_latest_results(directory)
    except OSError as error:
        raise ConfigError(
            f"could not inspect evaluation-attempt pointer {pointer_path}: {error}"
        ) from error
    try:
        pointer = ResultsAttemptPointer.model_validate(
            json.loads(pointer_path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as error:
        raise ConfigError(
            f"{pointer_path} is not a valid evaluation-attempt pointer: {error}"
        ) from error
    state_path = _attempt_state_path(directory, pointer.attempt_id)
    try:
        attempt = ResultsAttempt.model_validate(
            json.loads(state_path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as error:
        raise ConfigError(
            f"{state_path} is not a valid evaluation-attempt record: {error}"
        ) from error
    if attempt.attempt_id != pointer.attempt_id or attempt.command != pointer.command:
        raise ConfigError("the latest evaluation-attempt pointer is inconsistent")
    if attempt.status != "complete":
        raise ConfigError(
            "the latest evaluation attempt did not publish a results record",
            remediation=(
                "Fix the scoring or MLflow artifact error and rerun the "
                "evaluation before invoking `agentkit gate`."
            ),
        )
    path = directory / str(attempt.results_file)
    try:
        document = path.read_bytes()
    except FileNotFoundError as error:
        raise ConfigError(
            f"the latest evaluation result {path} is missing",
            remediation="Run the evaluation again before invoking `agentkit gate`.",
        ) from error
    except OSError as error:
        raise ConfigError(
            f"the latest evaluation result {path} could not be read: {error}",
            remediation="Run the evaluation again before invoking `agentkit gate`.",
        ) from error
    digest = hashlib.sha256(document).hexdigest()
    if digest != attempt.results_sha256:
        raise ConfigError(
            f"the latest evaluation result {path} changed after it was recorded",
            remediation="Run the evaluation again before invoking `agentkit gate`.",
        )
    # Parse the exact byte string whose digest was compared.  A second read
    # here would reopen a check/use race and could validate different
    # evidence from the bytes the completed attempt actually bound.
    record = _parse_results_bytes(path, document)
    _require_attempt_binding(record, attempt)
    return record, path


def _attempt_state_path(directory: Path, attempt_id: str) -> Path:
    return directory / f"{RESULTS_ATTEMPT_STATE_PREFIX}{attempt_id}"


@contextmanager
def _attempt_protocol_lock(directory: Path, *, shared: bool) -> Iterator[None]:
    """Serialize attempt transitions with full gate-evidence reads."""

    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / RESULTS_ATTEMPT_LOCK_FILE
    with lock_path.open("a+b") as lock:
        if fcntl is not None:
            operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(lock.fileno(), operation)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _path_exists(path: Path, *, purpose: str) -> bool:
    try:
        path.stat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ConfigError(f"could not inspect {purpose} {path}: {error}") from error
    return True


def _has_attempt_metadata(directory: Path) -> bool:
    """Whether legacy newest-file fallback would cross a protocol boundary."""

    try:
        return any(_is_attempt_state_name(entry.name) for entry in directory.iterdir())
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ConfigError(
            f"could not inspect evaluation-attempt metadata in {directory}: {error}"
        ) from error


def _is_attempt_state_name(name: str) -> bool:
    attempt_id = name.removeprefix(RESULTS_ATTEMPT_STATE_PREFIX)
    return (
        name.startswith(RESULTS_ATTEMPT_STATE_PREFIX)
        and len(attempt_id) == 32
        and all(character in "0123456789abcdef" for character in attempt_id)
    )


def _write_attempt_state(directory: Path, attempt: ResultsAttempt) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_contract_file(
        _attempt_state_path(directory, attempt.attempt_id), attempt.model_dump()
    )


def _write_contract_file(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, scratch_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    scratch = Path(scratch_name)
    try:
        scratch.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(scratch, path)
    finally:
        scratch.unlink(missing_ok=True)


def read_results(path: Path) -> ResultsRecord:
    return _parse_results_bytes(path, _read_results_bytes(path))


def _read_results_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except (OSError, ValueError) as error:
        raise ConfigError(f"could not read results record {path}: {error}") from error


def _parse_results_bytes(path: Path, payload: bytes) -> ResultsRecord:
    try:
        document = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ConfigError(f"{path} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(document, Mapping):
        raise ConfigError(f"{path} must contain a JSON object")
    try:
        return ResultsRecord.model_validate(document)
    except ValidationError as error:
        raise ConfigError(f"{path} is not a valid results record: {error}") from error


def _require_attempt_binding(
    record: ResultsRecord,
    attempt: ResultsAttempt,
) -> None:
    if record.attempt_id != attempt.attempt_id or record.command != attempt.command:
        raise ConfigError(
            "the results record is not bound to the evaluation attempt that "
            "is completing"
        )


def publish_results(
    mlflow_module: Any, run_id: str, record: ResultsRecord
) -> str | None:
    """Attach the results record to its MLflow run.

    The timestamped local filename is useful for retaining several runs, but
    the run artifact has one stable contract: ``agentkit/results.json``.
    Stage that canonical basename outside the gate-visible local results
    directory so a failed upload cannot leave a passing record behind.
    """

    try:
        with tempfile.TemporaryDirectory(prefix="aai-agentkit-results-") as staged:
            path = Path(staged) / RESULTS_ARTIFACT_FILE
            _write_results_file(path, record)
            client = mlflow_module.MlflowClient()
            client.log_artifact(run_id, str(path), artifact_path=RESULTS_ARTIFACT_DIR)
    except Exception as error:  # pragma: no cover - network/credential paths
        return f"could not attach the results record to run {run_id}: {error}"
    return None


def fetch_results(run_id: str, *, mlflow_module: Any | None = None) -> ResultsRecord:
    """Read a run's results record back out of MLflow."""

    if mlflow_module is None:
        try:
            import mlflow
        except ImportError as error:
            from aai_core.agentkit.errors import missing_extra

            raise missing_extra(
                "reading results from an MLflow run", "genai"
            ) from error
    else:
        mlflow = mlflow_module
    try:
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=RESULTS_ARTIFACT_PATH
        )
    except Exception as error:
        raise ConfigError(
            f"run {run_id} has no agentkit results record ({error})",
            remediation=(
                "Check the run id. Only runs recorded by `agentkit compare` "
                "or `agentkit eval` carry one; `agentkit smoke` scores "
                "locally and opens no run."
            ),
        ) from error
    try:
        path = Path(local)
    except (TypeError, ValueError, OSError) as error:
        raise ConfigError(
            f"run {run_id} returned an invalid agentkit results artifact path"
        ) from error
    record = read_results(path)
    if record.run_id != run_id:
        raise ConfigError(
            f"run {run_id} carries an agentkit results record for "
            f"{record.run_id!r}, not that run",
            remediation=(
                "Re-run the evaluation so MLflow receives evidence bound "
                "to the exact run that produced it."
            ),
        )
    return record


def load_latest_results(directory: Path) -> tuple[ResultsRecord, Path] | None:
    """Newest results record in the directory, or None when none exist."""

    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob(RESULTS_GLOB))
    if not candidates:
        return None
    newest = max(candidates, key=lambda item: (item.stat().st_mtime, item.name))
    return read_results(newest), newest
