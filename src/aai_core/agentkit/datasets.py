"""Dataset loading, shape inference, digesting, and deterministic sampling.

Local JSON/JSONL files load with base dependencies only; Unity Catalog
evaluation datasets (``catalog.schema.table``) load lazily through MLflow.
The digest and the sampled subset are deterministic so a comparison always
names exactly which rows it scored.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from aai_core.agentkit._values import is_missing_scalar
from aai_core.agentkit.errors import ConfigError, missing_extra

PLACEHOLDER_MARKERS = ("replace this", "replace-with", "todo", "changeme")
DEFAULT_MINIMUM_ROWS = 10
# The modes in which a row's *stored* trace is the thing being scored, and
# so the only ones whose payload carries it. `live` produces fresh traces
# from the agent; `answer-sheet` replays recorded outputs.
STORED_TRACE_MODES = frozenset({"traces"})
SMOKE_SEED = 20260802
_STRATA_CARDINALITY_LIMIT = 8
_LISTED_FAILURES = 5


@dataclass(frozen=True)
class DatasetShape:
    """What the rows contain — drives scorer auto-selection."""

    row_count: int
    input_keys: tuple[str, ...]
    has_outputs: bool
    # Expectation keys populated on EVERY row — the only ones a scorer can
    # be applied to without silently skipping rows.
    expectation_keys: tuple[str, ...]
    has_traces: bool
    strata_values: Mapping[str, tuple[str, ...]]
    # Present on some rows but not all; reported so the plan can say why a
    # scorer was not selected instead of leaving it a mystery.
    partial_expectation_keys: tuple[str, ...] = ()
    # The populated expectation keys of each row, in row order. A scorer
    # whose contract is a choice — correctness accepts expected_response
    # OR expected_facts, per row — is satisfied when every row provides
    # one of them, which the intersection above cannot express: rows split
    # between the two alternatives intersect to nothing.
    expectation_rows: tuple[tuple[str, ...], ...] = ()
    has_retrieval_spans: bool = False
    has_tool_spans: bool = False
    # Some rows carry a trace and some do not: the dataset cannot be
    # scored as traces, and saying so beats silently scoring a subset.
    partial_traces: bool = False


@dataclass(frozen=True)
class LoadedDataset:
    ref: str
    source: str
    rows: tuple[Mapping[str, Any], ...]
    digest: str
    shape: DatasetShape
    # The digest of the dataset this one was sampled from, when it was.
    # A sample asks a subset of the same questions, so a baseline recorded
    # on the whole file is still a baseline for the same *data* — only the
    # scope differs. Without this the comparability check reports "the
    # dataset changed", which is both a refusal and a false statement.
    sampled_from: str | None = None


def load_dataset(
    ref: str,
    *,
    root: Path,
    mlflow_module: Any | None = None,
) -> LoadedDataset:
    """Load rows from a repo-relative JSON/JSONL file or a UC dataset."""

    reference = str(ref).strip()
    if not reference:
        raise ConfigError("dataset must not be blank")
    path = (root / reference).resolve() if not Path(reference).is_absolute() else None
    if path is not None and path.is_file():
        rows, source = _load_file_rows(path)
    elif _looks_like_uc_dataset(reference, root):
        rows, source = _load_uc_rows(reference, mlflow_module), "uc-dataset"
    elif path is not None:
        raise ConfigError(
            f"dataset {reference!r} does not exist (looked for {path})",
            remediation=(
                "Point `dataset` at a JSON/JSONL file relative to "
                "agentkit.yaml, or at a Unity Catalog dataset named "
                "catalog.schema.table."
            ),
        )
    else:
        raise ConfigError(f"dataset {reference!r} must be a relative path")
    return _build_dataset(reference, source, rows)


_ANSWER_FIELDS = ("outputs", "trace")


def dataset_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Stable 16-hex digest identifying the questions this dataset asks.

    ``outputs`` and ``trace`` are both excluded, because both are the
    answer rather than the question. ``attach_answer_sheet`` merges
    outputs in, and a trace is recorded behaviour carrying its own id,
    timestamps, and responses — so hashing either would give a new dataset
    identity to the very thing a comparison exists to measure, and the
    comparability check would reject it as different data.

    A row that carries only a trace still needs an identity, so the
    request is extracted from it. Two runs over the same production
    questions therefore agree on the digest even though every trace
    differs.
    """

    canonical = json.dumps(
        [_row_identity(row) for row in rows],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        key: _plain(value) for key, value in row.items() if key not in _ANSWER_FIELDS
    }
    if not _is_populated(identity.get("inputs")):
        request = _trace_request(row.get("trace"))
        if request is not None:
            identity["inputs"] = request
    return identity


def _trace_request(trace: Any) -> Any:
    """The question a recorded trace answered, if it can be recovered.

    The root span's inputs come first, and ``request_preview`` last.
    MLflow documents the preview as "equivalent to the input of the root
    span but JSON-encoded and **can be truncated**", so two different long
    questions sharing a prefix would land on one dataset digest — and the
    comparability check would then wave through a run that scored
    different questions than the baseline did. The preview stays as a last
    resort because a trace without spans still needs an identity, and a
    truncated one beats none.
    """

    document = _trace_document(trace)
    if document is None:
        return None
    for span in _root_spans(trace):
        inputs = _span_field(span, ("inputs", "input"), "mlflow.spanInputs")
        if inputs is not None:
            return _plain(inputs)
    info = document.get("info")
    if isinstance(info, Mapping):
        for key in ("request", "request_preview"):
            value = info.get(key)
            if _is_populated(value):
                return _plain(value)
    return None


def _trace_response(trace: Any) -> Any:
    """What the trace answered, preferring the untruncated form.

    ``response_preview`` carries the same truncation caveat the request
    preview does, so a long answer read from it would be under-counted in
    the judge-token estimate the developer approves.
    """

    for span in _root_spans(trace):
        outputs = _span_outputs(span)
        if outputs is not None:
            return _plain(outputs)
    document = _trace_document(trace)
    info = document.get("info") if document is not None else None
    if isinstance(info, Mapping):
        for key in ("response", "response_preview"):
            value = info.get(key)
            if _is_populated(value):
                return _plain(value)
    return None


def _root_spans(trace: Any) -> list[Mapping[str, Any]]:
    """Spans with no parent from the MLflow v2/v3 data envelope.

    Going through ``_spans`` keeps the v2/v3 span parsing decision in one
    place and prevents an ignored top-level field from changing identity or
    cost calculations.
    """

    return [span for span in _spans(trace) if _is_root_span(span)]


def _is_root_span(span: Mapping[str, Any]) -> bool:
    """Whether a span has no populated parent identifier."""

    parent = span.get(
        "parent_span_id",
        span.get("parentSpanId", span.get("parent_id")),
    )
    return not _is_populated(parent)


def _span_field(
    span: Mapping[str, Any], keys: Sequence[str], attribute: str
) -> Any | None:
    """A span field, from the plain key or the serialized attribute."""

    candidates = [span.get(key) for key in keys]
    attributes = span.get("attributes")
    if isinstance(attributes, Mapping):
        candidates.append(attributes.get(attribute))
    for candidate in candidates:
        if isinstance(candidate, (str, bytes)):
            try:
                candidate = json.loads(candidate)
            except (ValueError, TypeError):
                continue
        if _is_populated(candidate):
            return candidate
    return None


def smoke_sample(
    dataset: LoadedDataset,
    n: int,
    *,
    strata: tuple[str, ...] = (),
    seed: int = SMOKE_SEED,
) -> LoadedDataset:
    """Deterministic (optionally stratified) sample of at most ``n`` rows."""

    if n <= 0:
        raise ConfigError("smoke sample size must be positive")
    if n >= dataset.shape.row_count:
        return dataset
    rng = random.Random(seed)
    if strata:
        selected = _stratified_indices(dataset.rows, n, strata, rng)
    else:
        indices = list(range(dataset.shape.row_count))
        rng.shuffle(indices)
        selected = sorted(indices[:n])
    rows = [dataset.rows[index] for index in selected]
    return _build_dataset(
        dataset.ref,
        f"{dataset.source}+sample",
        rows,
        # Carry the parent identity: a sample of this dataset is the same
        # questions asked of fewer rows, not different data.
        sampled_from=dataset.sampled_from or dataset.digest,
    )


def attach_answer_sheet(dataset: LoadedDataset, sheet_path: Path) -> LoadedDataset:
    """Replay recorded outputs onto the rows (answer-sheet mode).

    Accepts the template's ``{"question": ..., "answer": ...}`` records and
    the generic ``{"inputs": {...}, "outputs": ...}`` shape.
    """

    sheet_path = Path(sheet_path)
    try:
        text = sheet_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ConfigError(
            f"answer sheet {sheet_path} does not exist",
            remediation=(
                "Record the current outputs to the answer sheet, or run in "
                "live mode against a callable target."
            ),
        ) from error
    except UnicodeError as error:
        raise ConfigError(f"answer sheet {sheet_path} is not valid UTF-8") from error
    except OSError as error:
        raise ConfigError(f"answer sheet {sheet_path} could not be read") from error

    if sheet_path.suffix == ".jsonl":
        parsed = _load_jsonl_objects(
            text,
            subject=f"answer sheet {sheet_path}",
        )
        if not parsed:
            raise ConfigError(
                f"answer sheet {sheet_path} must contain at least one JSON object"
            )
        located_records = [(record, f"line {line}") for line, record in parsed]
    else:
        try:
            records = json.loads(text)
        except json.JSONDecodeError as error:
            raise ConfigError(
                f"answer sheet {sheet_path} is not valid JSON "
                f"(line {error.lineno}, column {error.colno}: {error.msg})"
            ) from error
        if not isinstance(records, list):
            raise ConfigError(f"answer sheet {sheet_path} must contain a JSON list")
        located_records = [
            (record, f"row {index}") for index, record in enumerate(records)
        ]

    answers: dict[str, Any] = {}
    answer_locations: dict[str, str] = {}
    for record, location in located_records:
        if not isinstance(record, Mapping):
            raise ConfigError(
                f"answer sheet {sheet_path} {location} must be a JSON object"
            )
        if "question" in record and "answer" in record:
            if not _is_populated(record["question"]):
                raise ConfigError(
                    f"answer sheet {sheet_path} {location} needs a populated question"
                )
            inputs = {"question": record["question"]}
            output = record["answer"]
            output_name = "answer"
        elif "inputs" in record and "outputs" in record:
            inputs = record["inputs"]
            if not isinstance(inputs, Mapping) or not inputs:
                raise ConfigError(
                    f"answer sheet {sheet_path} {location} needs a non-empty "
                    "inputs object"
                )
            output = record["outputs"]
            output_name = "outputs"
        else:
            raise ConfigError(
                f"answer sheet {sheet_path} {location} needs question/answer "
                "or inputs/outputs fields"
            )
        if not _is_populated(output):
            raise ConfigError(
                f"answer sheet {sheet_path} {location} needs populated "
                f"{output_name}"
            )
        key = _inputs_key(inputs)
        if key in answers:
            raise ConfigError(
                f"answer sheet {sheet_path} has duplicate inputs at {location} "
                f"(first seen at {answer_locations[key]})",
                remediation="Keep exactly one recorded answer for each input.",
            )
        answers[key] = output
        answer_locations[key] = location

    rows: list[Mapping[str, Any]] = []
    missing: list[str] = []
    for row in dataset.rows:
        key = _inputs_key(row.get("inputs", {}))
        if key not in answers:
            missing.append(_row_label(row))
            continue
        merged = dict(row)
        merged["outputs"] = answers[key]
        rows.append(merged)
    if missing:
        listed = ", ".join(repr(item) for item in missing[:_LISTED_FAILURES])
        raise ConfigError(
            f"answer sheet {sheet_path} lacks answers for {len(missing)} "
            f"row(s): {listed}",
            remediation=("Re-record the answer sheet so it covers every dataset row."),
        )
    return _build_dataset(
        dataset.ref,
        f"{dataset.source}+answers",
        rows,
        sampled_from=dataset.sampled_from,
    )


def effective_dataset(dataset: LoadedDataset, *, mode: str) -> LoadedDataset:
    """The dataset as MLflow will actually see it, for this mode.

    The authored rows and the scored rows are not the same thing, and
    everything downstream — which scorers apply, what the judge calls will
    cost, what the payload contains — has to be decided from the second.
    Deciding from the first is how a plan promises a scorer whose field
    MLflow will have replaced, and how a budget is approved against a
    fan-out that will not happen.

    Three transformations, each mirroring something MLflow does:

    **A stored trace travels only in ``traces`` mode.** It is the recorded
    answer, so in ``live`` the answer comes from ``predict_fn`` and in
    ``answer-sheet`` from the sheet. It is not inert baggage either:
    ``_convert_to_eval_set`` pipes any present trace column through
    ``_extract_request_response_from_trace``, which calls
    ``trace.data._get_root_span()`` on every value — one ``NaN`` raises
    before the agent is called — and through
    ``_extract_expectations_from_trace``, which overwrites expectations.
    MLflow's own validation agrees the column is unwanted here: it adds
    ``trace`` to the satisfied columns whenever a ``predict_fn`` is given.

    **Dropping the trace must not drop the question with it.** A
    trace-only row keeps its request, recovered the same way the dataset
    digest recovers it, because otherwise removing the trace would leave a
    row MLflow cannot evaluate at all.

    **In ``traces`` mode, expectations come from the traces** whenever any
    one of them carries an expectation assessment — MLflow replaces the
    whole column, so a row whose trace has no assessment ends up with
    *none*, whatever the dataset wrote. Mirroring that here is what stops
    a scorer being selected against a curated field the run will not have:
    `keyword_coverage` reads an absent expected response as a vacuous
    1.0, and a gate passing on that is the exact failure this toolkit
    exists to prevent.

    **A missing value is an absent key.** A Unity Catalog dataset arrives
    through ``to_dict("records")``, where an absent field is ``NaN``; only
    dropping the key says "absent" rather than handing MLflow a float
    where a mapping belongs. That alone would not fix a partly traced
    dataset — pandas refills the column for the rows that dropped the key
    — which is why the mode rule exists as well.

    The digest is recomputed from the effective rows. That makes identity
    mode-aware: ``traces`` binds the expectation assessments MLflow actually
    scores, while live and answer-sheet modes discard the trace and retain
    authored expectations. The digest still excludes trace outputs, ids, and
    timestamps through ``dataset_digest``. Ref and sampling provenance stay
    attached to the source dataset.
    """

    keep_trace = mode in STORED_TRACE_MODES
    # MLflow replaces the column wholesale, or not at all.
    replace_expectations = keep_trace and any(
        _trace_expectation_names(row.get("trace")) for row in dataset.rows
    )
    rows: list[Mapping[str, Any]] = []
    for row in dataset.rows:
        kept: dict[str, Any] = {}
        for key, value in row.items():
            if key == "trace" and not keep_trace:
                continue
            if _is_missing(value):
                continue
            kept[key] = value
        if not keep_trace and not _is_populated(kept.get("inputs")):
            request = _trace_inputs(row.get("trace"))
            if request is not None:
                kept["inputs"] = request
        if replace_expectations:
            kept["expectations"] = _trace_expectations(row.get("trace"))
        rows.append(kept)
    frozen = tuple(rows)
    shape = _infer_shape(frozen)
    if not keep_trace:
        # The span *kinds* stay the authored dataset's. A live run's traces
        # do not exist yet, but a suite recorded against a retrieving agent
        # is still a retrieval suite, and inferring otherwise from rows we
        # just emptied would silently drop the retrieval judges from every
        # live run — a control removed by a fix, which is not a fix. What
        # must not carry over is the *count*: the estimate reads the rows,
        # finds no traces, and applies the configured assumption instead of
        # the previous agent's fan-out.
        shape = replace(
            shape,
            has_retrieval_spans=dataset.shape.has_retrieval_spans,
            has_tool_spans=dataset.shape.has_tool_spans,
        )
    return LoadedDataset(
        ref=dataset.ref,
        source=dataset.source,
        rows=frozen,
        digest=dataset_digest(frozen),
        shape=shape,
        sampled_from=dataset.sampled_from,
    )


def evaluation_rows(dataset: LoadedDataset, *, mode: str) -> list[dict[str, Any]]:
    """The rows handed to ``mlflow.genai.evaluate``."""

    return [dict(row) for row in effective_dataset(dataset, mode=mode).rows]


def rows_missing_inputs(dataset: LoadedDataset) -> tuple[int, ...]:
    """Rows MLflow cannot evaluate without a trace to fall back on.

    ``inputs`` must be a mapping once the trace is gone — MLflow says so
    itself, and only skips the check when a trace column is present.
    """

    return tuple(
        index
        for index, row in enumerate(dataset.rows)
        if not isinstance(row.get("inputs"), Mapping) or not row["inputs"]
    )


def _trace_inputs(trace: Any) -> Mapping[str, Any] | None:
    """The trace's request, but only when it can serve as ``inputs``.

    MLflow requires a mapping there. ``_trace_request`` may recover a
    JSON string from ``info.request`` instead, which is worth decoding,
    but anything that is not a mapping in the end is not usable and the
    row is reported rather than passed on to fail obscurely.
    """

    request = _trace_request(trace)
    if isinstance(request, str):
        try:
            request = json.loads(request)
        except (ValueError, TypeError):
            return None
    if isinstance(request, Mapping) and request:
        return _plain(request)
    return None


def _has_usable_trace(trace: Any, *, authored_inputs: Any) -> bool:
    """Whether a populated trace is safe to hand to MLflow.

    A non-empty string or mapping is not necessarily a trace. MLflow only
    discovers that after planning and spend confirmation, when it deserializes
    the evaluation frame. Require the local reader to decode the value and to
    recover either a request or a root span before treating the row as traced.
    The latter preserves recorded traces whose authored ``inputs`` column is
    already present while accepting MLflow's supported v2 and v3 envelopes.
    """

    document = _trace_document(trace)
    if document is None or not _trace_envelope_is_deserializable(document):
        return False

    # Trace.from_dict supports the v2 and v3 envelopes in the same layout:
    # info plus data.spans. A top-level `spans` key is ignored by MLflow and
    # must never rescue an empty or malformed data section locally.
    spans = _spans(document)

    # Read the trace-info request directly. Going through `_trace_inputs`
    # would let the inputs on an id-less mapping masquerading as a root span
    # satisfy this branch before its span structure is checked.
    info_has_request = False
    info = document.get("info")
    if isinstance(info, Mapping):
        for key in ("request", "request_preview"):
            request = info.get(key)
            if isinstance(request, (str, bytes)):
                try:
                    request = json.loads(request)
                except (ValueError, TypeError):
                    continue
            if isinstance(request, Mapping) and request:
                info_has_request = True
                break

    roots = [span for span in spans if _is_root_span(span)]
    if not roots:
        # MLflow accepts a complete envelope with an empty span list. It still
        # needs a recoverable request to represent an evaluation row. A
        # non-empty graph made only of children has a dangling parent chain.
        return not spans and info_has_request
    identified_roots = all(_is_populated(_span_identifier(span)) for span in roots)
    if not identified_roots:
        return False
    # A span id proves structure, not that the row has a question to score.
    # The request may come from the authored row or the identified root span.
    row_has_request = (
        isinstance(authored_inputs, Mapping) and bool(authored_inputs)
    ) or info_has_request
    if row_has_request:
        return True
    return all(
        isinstance(
            request := _span_field(root, ("inputs", "input"), "mlflow.spanInputs"),
            Mapping,
        )
        and bool(request)
        for root in roots
    )


_TRACE_STATES = frozenset({"STATE_UNSPECIFIED", "OK", "ERROR", "IN_PROGRESS"})
_TRACE_V2_STATUSES = frozenset(
    {"TRACE_STATUS_UNSPECIFIED", "OK", "ERROR", "IN_PROGRESS"}
)
_SPAN_STATUSES = frozenset({"UNSET", "OK", "ERROR"})
_SPAN_V3_STATUSES = _SPAN_STATUSES | frozenset(
    {"STATUS_CODE_UNSET", "STATUS_CODE_OK", "STATUS_CODE_ERROR"}
)
_ASSESSMENT_SOURCE_TYPES = frozenset(
    {"SOURCE_TYPE_UNSPECIFIED", "LLM_JUDGE", "AI_JUDGE", "HUMAN", "CODE"}
)
_ASSESSMENT_KIND_KEYS = frozenset({"expectation", "feedback", "issue"})
_ASSESSMENT_KEYS = frozenset(
    {
        "assessment_id",
        "assessment_name",
        "trace_id",
        "run_id",
        "source",
        "create_time",
        "last_update_time",
        "expectation",
        "feedback",
        "issue",
        "rationale",
        "metadata",
        "span_id",
        "overrides",
        "valid",
    }
)
_EXPECTATION_KEYS = frozenset({"value", "serialized_value"})
_SERIALIZED_EXPECTATION_KEYS = frozenset({"serialization_format", "value"})
_FEEDBACK_KEYS = frozenset({"value", "error"})
_FEEDBACK_ERROR_KEYS = frozenset({"error_code", "error_message", "stack_trace"})
_ISSUE_KEYS = frozenset({"issue_name"})
_TRACE_LOCATION_SHAPES = {
    "TRACE_LOCATION_TYPE_UNSPECIFIED": (None, frozenset(), frozenset()),
    "MLFLOW_EXPERIMENT": (
        "mlflow_experiment",
        frozenset({"experiment_id"}),
        frozenset(),
    ),
    "INFERENCE_TABLE": (
        "inference_table",
        frozenset({"full_table_name"}),
        frozenset(),
    ),
    "UC_SCHEMA": (
        "uc_schema",
        frozenset({"catalog_name", "schema_name"}),
        frozenset({"otel_spans_table_name", "otel_logs_table_name"}),
    ),
    "UC_TABLE_PREFIX": (
        "uc_table_prefix",
        frozenset({"catalog_name", "schema_name"}),
        frozenset(
            {
                "table_prefix",
                "otel_spans_table_name",
                "otel_logs_table_name",
                "annotations_table_name",
            }
        ),
    ),
}
_TRACE_INFO_V2_KEYS = frozenset(
    {
        "request_id",
        "experiment_id",
        "timestamp_ms",
        "execution_time_ms",
        "status",
        "request_metadata",
        "tags",
        "assessments",
    }
)
_TRACE_INFO_V3_KEYS = frozenset(
    {
        "trace_id",
        "trace_location",
        "request_time",
        "state",
        "request_preview",
        "response_preview",
        "client_request_id",
        "execution_duration",
        "execution_duration_ms",
        "trace_metadata",
        "tags",
        "assessments",
    }
)


def _trace_envelope_is_deserializable(document: Mapping[str, Any]) -> bool:
    """Whether MLflow can deserialize this complete v2/v3 trace envelope.

    The pure structural contract is intentionally conservative and complete
    for the canonical dictionaries emitted by MLflow 2/3. The SDK's base
    install therefore rejects malformed traces without importing an optional
    dependency. When MLflow is present, pinned ``Trace.from_dict`` remains the
    final parity check.
    """

    if not _complete_trace_envelope(document):
        return False
    try:
        from mlflow.entities import Trace
    except ModuleNotFoundError as error:
        # The SDK's base install intentionally has no MLflow dependency. A
        # broken installed MLflow is different from an absent optional extra.
        return error.name == "mlflow"
    except ImportError:
        return False

    try:
        Trace.from_dict(copy.deepcopy(_plain(document)))
    except Exception:
        return False
    return True


def _complete_trace_envelope(document: Mapping[str, Any]) -> bool:
    """Dependency-free contract for canonical MLflow v2/v3 trace dictionaries."""

    try:
        info = document.get("info")
        data = document.get("data")
        if not isinstance(info, Mapping) or not isinstance(data, Mapping):
            return False
        schema = "v2" if "request_id" in info else "v3"
        if not _complete_trace_info(info, schema=schema):
            return False
        spans = data.get("spans", [])
        if not isinstance(spans, list):
            return False
        request_id = info["request_id" if schema == "v2" else "trace_id"]
        return all(
            isinstance(span, Mapping)
            and _complete_span(
                span,
                schema=schema,
                expected_request_id=request_id,
            )
            for span in spans
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        # This is a boundary over untrusted JSON. Shape errors are governed
        # validation failures, never raw Python exceptions.
        return False


def _complete_trace_info(info: Mapping[str, Any], *, schema: str) -> bool:
    if schema == "v2":
        if not set(info) <= _TRACE_INFO_V2_KEYS:
            return False
        request_id = info.get("request_id")
        experiment_id = info.get("experiment_id")
        status = info.get("status")
        if not (
            _non_empty_string(request_id)
            and _non_empty_string(experiment_id)
            and isinstance(status, str)
            and status in _TRACE_V2_STATUSES
            and _non_negative_integer(info.get("timestamp_ms"))
            and "execution_time_ms" in info
            and _optional_non_negative_integer(info.get("execution_time_ms"))
            and _optional_string_map(info, "request_metadata")
            and _optional_string_map(info, "tags")
        ):
            return False
    else:
        if not set(info) <= _TRACE_INFO_V3_KEYS:
            return False
        request_id = info.get("trace_id")
        state = info.get("state")
        if not (
            _non_empty_string(request_id)
            and isinstance(state, str)
            and state in _TRACE_STATES
            and _rfc3339_timestamp(info.get("request_time"))
            and _complete_trace_location(info.get("trace_location"))
            and _optional_string(info, "request_preview")
            and _optional_string(info, "response_preview")
            and _optional_string(info, "client_request_id")
            and _optional_non_negative_integer_field(info, "execution_duration")
            and _optional_non_negative_integer_field(info, "execution_duration_ms")
            and _optional_string_map(info, "trace_metadata")
            and _optional_string_map(info, "tags")
        ):
            return False

    assessments = info.get("assessments", [])
    return isinstance(assessments, list) and all(
        _complete_assessment(item, expected_request_id=request_id)
        for item in assessments
    )


def _complete_trace_location(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    location_type = value.get("type")
    if (
        not isinstance(location_type, str)
        or location_type not in _TRACE_LOCATION_SHAPES
    ):
        return False

    field, required, optional = _TRACE_LOCATION_SHAPES[location_type]
    expected_keys = {"type"} if field is None else {"type", field}
    if set(value) != expected_keys:
        return False
    if field is None:
        return True

    details = value.get(field)
    return (
        isinstance(details, Mapping)
        and required <= set(details) <= required | optional
        and all(_non_empty_string(details.get(key)) for key in details)
    )


def _complete_assessment(assessment: Any, *, expected_request_id: str) -> bool:
    if not isinstance(assessment, Mapping):
        return False
    source = assessment.get("source")
    if not (
        set(assessment) <= _ASSESSMENT_KEYS
        and _non_empty_string(assessment.get("assessment_name"))
        and _rfc3339_timestamp(assessment.get("create_time"))
        and _rfc3339_timestamp(assessment.get("last_update_time"))
        and isinstance(source, Mapping)
        and set(source) <= {"source_type", "source_id"}
        and isinstance(source.get("source_type"), str)
        and source["source_type"].upper() in _ASSESSMENT_SOURCE_TYPES
        and _optional_string(source, "source_id")
        and _optional_string(assessment, "assessment_id")
        and _optional_string(assessment, "trace_id")
        and _optional_string(assessment, "run_id")
        and _optional_string(assessment, "rationale")
        and _optional_string(assessment, "span_id")
        and _optional_string(assessment, "overrides")
        and _optional_string_map(assessment, "metadata")
        and ("valid" not in assessment or isinstance(assessment.get("valid"), bool))
    ):
        return False
    assessment_trace_id = assessment.get("trace_id")
    if assessment_trace_id is not None and assessment_trace_id != expected_request_id:
        return False

    kinds = set(assessment) & _ASSESSMENT_KIND_KEYS
    if len(kinds) != 1:
        return False
    kind = next(iter(kinds))
    value = assessment[kind]
    if not isinstance(value, Mapping):
        return False
    if kind == "expectation":
        return _complete_expectation(value)
    if kind == "feedback":
        return _complete_feedback(value)
    return set(value) == _ISSUE_KEYS and _non_empty_string(value.get("issue_name"))


def _complete_expectation(value: Mapping[str, Any]) -> bool:
    arms = set(value) & _EXPECTATION_KEYS
    if not set(value) <= _EXPECTATION_KEYS or len(arms) != 1:
        return False
    if arms == {"value"}:
        direct = value["value"]
        return direct is not None and _json_value(direct)

    serialized = value["serialized_value"]
    if (
        not isinstance(serialized, Mapping)
        or set(serialized) != _SERIALIZED_EXPECTATION_KEYS
        or serialized.get("serialization_format") != "JSON_FORMAT"
    ):
        return False
    encoded = serialized.get("value")
    if not isinstance(encoded, str):
        return False
    try:
        decoded = json.loads(encoded)
    except (TypeError, ValueError):
        return False
    return decoded is not None and _json_value(decoded)


def _complete_feedback(value: Mapping[str, Any]) -> bool:
    if (
        not set(value) <= _FEEDBACK_KEYS
        or "value" not in value
        or not _json_value(value.get("value"))
    ):
        return False
    if "error" not in value:
        return True
    error = value["error"]
    return (
        isinstance(error, Mapping)
        and set(error) <= _FEEDBACK_ERROR_KEYS
        and _non_empty_string(error.get("error_code"))
        and _optional_string(error, "error_message")
        and _optional_string(error, "stack_trace")
    )


def _complete_span(
    span: Mapping[str, Any], *, schema: str, expected_request_id: str
) -> bool:
    attributes = span.get("attributes")
    if not (
        _otel_attributes(attributes)
        and _request_id_attribute_matches(attributes, expected_request_id)
        and _non_empty_string(span.get("name"))
    ):
        return False
    if schema == "v2":
        return _complete_v2_span(span)
    return _complete_v3_span(span)


def _complete_v2_span(span: Mapping[str, Any]) -> bool:
    context = span.get("context")
    status = span.get("status_code")
    return (
        isinstance(context, Mapping)
        and _hex_id(context.get("trace_id"), width=32)
        and _hex_id(context.get("span_id"), width=16)
        and "parent_id" in span
        and _optional_hex_parent(span.get("parent_id"))
        and _non_negative_integer(span.get("start_time"))
        and _non_negative_integer(span.get("end_time"))
        and isinstance(status, str)
        and status in _SPAN_STATUSES
        and isinstance(span.get("status_message"), str)
        and _complete_events(span.get("events"), schema="v2")
        and _complete_links(span.get("links", []))
    )


def _complete_v3_span(span: Mapping[str, Any]) -> bool:
    trace_id = span.get("trace_id")
    byte_ids = _base64_id(trace_id, width=16)
    ids_are_valid = (
        _base64_id(span.get("span_id"), width=8)
        and _optional_base64_parent(span.get("parent_span_id"))
        if byte_ids
        else _property_trace_id(trace_id)
        and _hex_id(span.get("span_id"), width=16)
        and _optional_hex_parent(span.get("parent_span_id"))
    )
    status = span.get("status")
    end_time = span.get("end_time_unix_nano")
    return (
        ids_are_valid
        and "parent_span_id" in span
        and _non_negative_integer(span.get("start_time_unix_nano"))
        and (end_time is None or _non_negative_integer(end_time))
        and isinstance(status, Mapping)
        and isinstance(status.get("code"), str)
        and status["code"] in _SPAN_V3_STATUSES
        and ("message" not in status or isinstance(status.get("message"), str))
        and _complete_events(span.get("events", []), schema="v3")
        and _complete_links(span.get("links", []))
    )


def _complete_events(value: Any, *, schema: str) -> bool:
    if not isinstance(value, list):
        return False
    time_key = "timestamp" if schema == "v2" else "time_unix_nano"
    return all(
        isinstance(event, Mapping)
        and _non_empty_string(event.get("name"))
        and _non_negative_integer(event.get(time_key))
        and (
            "attributes" in event
            and (
                event.get("attributes") is None
                or _otel_attributes(event.get("attributes"))
            )
            if schema == "v2"
            else _otel_attributes(event.get("attributes", {}))
        )
        for event in value
    )


def _complete_links(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(link, Mapping)
        and _linked_trace_id(link.get("trace_id"))
        and _hex_id(link.get("span_id"), width=16)
        and (
            link.get("attributes") is None
            or _json_scalar_mapping(link.get("attributes"))
        )
        for link in value
    )


def _request_id_attribute_matches(
    attributes: Mapping[str, Any], expected_request_id: str
) -> bool:
    encoded = attributes.get("mlflow.traceRequestId")
    if not isinstance(encoded, str):
        return False
    try:
        decoded = json.loads(encoded)
    except (TypeError, ValueError):
        return False
    return isinstance(decoded, str) and decoded == expected_request_id


def _rfc3339_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    body = value[:-1]
    timestamp, separator, fraction = body.partition(".")
    if separator and (not 1 <= len(fraction) <= 9 or not fraction.isdigit()):
        return False
    try:
        datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return False
    return True


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _non_negative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _optional_non_negative_integer(value: Any) -> bool:
    return value is None or _non_negative_integer(value)


def _optional_non_negative_integer_field(mapping: Mapping[str, Any], key: str) -> bool:
    return key not in mapping or _optional_non_negative_integer(mapping.get(key))


def _optional_string(mapping: Mapping[str, Any], key: str) -> bool:
    return (
        key not in mapping
        or mapping.get(key) is None
        or isinstance(mapping.get(key), str)
    )


def _optional_string_map(mapping: Mapping[str, Any], key: str) -> bool:
    return key not in mapping or _string_map(mapping.get(key))


def _string_map(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    )


def _otel_attributes(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and _otel_attribute_value(item)
        for key, item in value.items()
    )


def _otel_attribute_value(value: Any) -> bool:
    if isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if not isinstance(value, (list, tuple)):
        return False
    if not value:
        return True
    value_type = type(value[0])
    return value_type in {str, bool, int, float} and all(
        type(item) is value_type
        and (not isinstance(item, float) or math.isfinite(item))
        for item in value
    )


def _json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _json_value(item) for key, item in value.items()
        )
    return False


def _json_scalar_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(key, str)
        and (item is None or isinstance(item, (str, bool, int, float)))
        and (not isinstance(item, float) or math.isfinite(item))
        for key, item in value.items()
    )


def _base64_id(value: Any, *, width: int) -> bool:
    if not isinstance(value, str):
        return False
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        return False
    return len(decoded) == width and any(decoded)


def _hex_id(value: Any, *, width: int) -> bool:
    if not isinstance(value, str) or len(value) != width:
        return False
    try:
        return int(value, 16) > 0
    except ValueError:
        return False


def _optional_base64_parent(value: Any) -> bool:
    return value is None or value == "" or _base64_id(value, width=8)


def _optional_hex_parent(value: Any) -> bool:
    return value is None or value == "" or _hex_id(value, width=16)


def _property_trace_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    identifier = value
    if value.startswith("trace:/"):
        parts = value.removeprefix("trace:/").split("/")
        if len(parts) != 2 or not all(parts):
            return False
        identifier = parts[1]
    if identifier.startswith("tr-"):
        identifier = identifier.removeprefix("tr-")
    return _hex_id(identifier, width=32)


def _linked_trace_id(value: Any) -> bool:
    return _property_trace_id(value) or _hex_id(value, width=32)


def _span_identifier(span: Mapping[str, Any]) -> Any:
    context = span.get("context")
    if isinstance(context, Mapping):
        return context.get("span_id")
    return span.get("span_id", span.get("spanId"))


def _trace_expectations(trace: Any) -> dict[str, Any]:
    return dict(_trace_expectation_items(trace))


def _trace_expectation_items(trace: Any) -> tuple[tuple[str, Any], ...]:
    document = _trace_document(trace)
    info = document.get("info") if document is not None else None
    if not isinstance(info, Mapping):
        return ()
    expectations: list[tuple[str, Any]] = []
    for assessment in info.get("assessments") or ():
        if not isinstance(assessment, Mapping):
            continue
        expectation = assessment.get("expectation")
        if not _is_populated(expectation):
            continue
        name = assessment.get("assessment_name") or assessment.get("name")
        if not name:
            continue
        expectations.append(
            (str(name), _expectation_value(expectation, name=str(name)))
        )
    return tuple(expectations)


def _expectation_value(expectation: Any, *, name: str) -> Any:
    """Decode the expectation shapes emitted by MLflow ``Trace.to_dict``.

    Scalar expectations arrive directly or under ``value``. Non-scalars
    such as ``expected_facts`` lists use
    ``serialized_value.value`` containing JSON. Treating that wrapper like
    a direct value returns ``None`` and silently removes the scorer contract
    after MLflow replaces the authored expectations column, so malformed
    wrappers fail before planning or spend.
    """

    if not isinstance(expectation, Mapping):
        return _plain(expectation)

    direct = expectation.get("value")
    if not _is_missing(direct):
        return _plain(direct)

    serialized = expectation.get("serialized_value")
    if not isinstance(serialized, Mapping) or "value" not in serialized:
        raise _malformed_trace_expectation(
            name,
            "expected a direct value or serialized_value.value",
        )
    encoded = serialized["value"]
    if not isinstance(encoded, str):
        raise _malformed_trace_expectation(
            name,
            "serialized_value.value must be a JSON string",
        )
    try:
        value = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise _malformed_trace_expectation(
            name,
            "serialized_value.value is not valid JSON",
        ) from error
    if _is_missing(value):
        raise _malformed_trace_expectation(
            name,
            "serialized_value.value decoded to a missing value",
        )
    return _plain(value)


def _malformed_trace_expectation(name: str, detail: str) -> ConfigError:
    return ConfigError(
        f"trace expectation {name!r} is malformed: {detail}",
        remediation=(
            "Re-record the trace expectation with MLflow 3.14 or provide a "
            "direct scalar value; do not run a gate after ground truth could "
            "not be decoded."
        ),
    )


def trace_expectation_overrides(dataset: LoadedDataset) -> tuple[str, ...]:
    """Rows whose curated expectations MLflow will replace with the trace's.

    ``_extract_expectations_from_trace`` documents itself as filling the
    column "if it is not already present", but it has no such check: any
    trace carrying an expectation assessment rewrites the whole column. In
    ``traces`` mode that is MLflow's behaviour and may be what the project
    wants — so this reports rather than refuses, naming the expectations
    whose value comes from the trace instead of the dataset.
    """

    overridden: set[str] = set()
    for row in dataset.rows:
        if not _is_populated(row.get("expectations")):
            continue
        overridden.update(_trace_expectation_names(row.get("trace")))
    return tuple(sorted(overridden))


def _trace_expectation_names(trace: Any) -> tuple[str, ...]:
    # Use the same strict reader as `_trace_expectations`: a malformed value
    # must not count as an override here and then disappear when the effective
    # rows are built. `Trace.to_dict()` writes `assessment_name`; the
    # in-memory entity attribute is `name`, and the shared reader accepts both.
    return tuple(name for name, _ in _trace_expectation_items(trace))


def validate_dataset(
    dataset: LoadedDataset,
    *,
    minimum_rows: int = DEFAULT_MINIMUM_ROWS,
) -> list[str]:
    """Structural failures that should stop an evaluation before any spend."""

    failures: list[str] = []
    if dataset.shape.row_count < minimum_rows:
        failures.append(
            f"dataset has {dataset.shape.row_count} rows; keep at least "
            f"{minimum_rows} (grow toward 150+ as the suite matures)"
        )
    for index, row in enumerate(dataset.rows):
        inputs = row.get("inputs")
        expectations = row.get("expectations")
        trace = row.get("trace")
        has_trace = _is_populated(trace)
        # Checked before the trace shortcut below: shape inference reads a
        # malformed expectations value as *absent*, which silently drops
        # the scorers and thresholds that depend on it while the value
        # still travels to MLflow. A traced row is exempt from needing
        # inputs, not from being well formed.
        if not _is_missing(expectations) and not isinstance(expectations, Mapping):
            failures.append(f"row {index} expectations must be an object")
            continue
        # A trace replaces a missing request, not a malformed authored one.
        # MLflow retains an explicit ``inputs`` value on the evaluation row,
        # so accepting a scalar or list here only defers the row-contract
        # failure until after planning and spend confirmation.
        if has_trace and not _is_missing(inputs) and not isinstance(inputs, Mapping):
            failures.append(f"row {index} inputs must be an object")
            continue
        if has_trace and not _has_usable_trace(trace, authored_inputs=inputs):
            failures.append(
                f"row {index} trace must be decodable and contain a usable "
                "request or root span"
            )
            continue
        # A trace exempts the row from supplying authored inputs, but MLflow
        # still scores explicit inputs and curated expectations when they are
        # present. When they are absent, the recoverable trace request is the
        # input that represents the row. Validate exactly that content before
        # the trace shortcut can accept a placeholder into a paid run.
        checked_inputs = inputs
        if has_trace and not _is_populated(checked_inputs):
            checked_inputs = _trace_request(trace)
        if has_trace or (isinstance(inputs, Mapping) and inputs):
            text = json.dumps(_plain(checked_inputs), default=str).lower()
            if _is_populated(expectations):
                text += json.dumps(_plain(expectations), default=str).lower()
            if any(marker in text for marker in PLACEHOLDER_MARKERS):
                failures.append(f"row {index} still contains placeholder text")
        if has_trace:
            continue
        if not isinstance(inputs, Mapping) or not inputs:
            failures.append(f"row {index} is missing a non-empty inputs object")
            continue
    return failures


def _build_dataset(
    ref: str,
    source: str,
    rows: Sequence[Mapping[str, Any]],
    sampled_from: str | None = None,
) -> LoadedDataset:
    frozen_rows = tuple(rows)
    return LoadedDataset(
        ref=ref,
        source=source,
        rows=frozen_rows,
        digest=dataset_digest(frozen_rows),
        shape=_infer_shape(frozen_rows),
        sampled_from=sampled_from,
    )


def _infer_shape(rows: Sequence[Mapping[str, Any]]) -> DatasetShape:
    input_keys: set[str] = set()
    # Expectation coverage is per-row, not per-dataset. A scorer is only
    # applicable when EVERY row can satisfy its contract: a row with no
    # expected response cannot be scored against one, and letting it
    # through means the scorer silently returns a perfect score for a row
    # it never checked, inflating the aggregate the gate reads.
    expectation_keys: set[str] | None = None
    partial_expectation_keys: set[str] = set()
    expectation_rows: list[tuple[str, ...]] = []
    # Trace coverage is per-row for the same reason expectations are: a
    # traces run supplies no predict_fn, so a row without a populated
    # trace has no answer at all — it cannot be scored, only skipped or
    # errored. A nullable Unity Catalog trace column makes `trace: null`
    # the ordinary shape of that, not an exotic one.
    traced_rows = 0
    has_retrieval_spans = False
    has_tool_spans = False
    has_outputs = bool(rows)
    candidate_strata: dict[str, set[str]] = {}
    for row in rows:
        inputs = row.get("inputs")
        if isinstance(inputs, Mapping):
            input_keys.update(str(key) for key in inputs)
            for key, value in inputs.items():
                if isinstance(value, str):
                    candidate_strata.setdefault(str(key), set()).add(value)
                else:
                    candidate_strata.setdefault(str(key), set()).update(
                        {f"<{type(value).__name__}>", "<mixed>"}
                    )
        expectations = row.get("expectations")
        present = (
            {str(key) for key in expectations if _is_populated(expectations[key])}
            if isinstance(expectations, Mapping)
            else set()
        )
        partial_expectation_keys |= present
        expectation_keys = (
            present if expectation_keys is None else expectation_keys & present
        )
        expectation_rows.append(tuple(sorted(present)))
        if _is_populated(row.get("trace")):
            traced_rows += 1
            retrieval, tools = _trace_span_kinds(row["trace"])
            has_retrieval_spans = has_retrieval_spans or retrieval
            has_tool_spans = has_tool_spans or tools
        if _is_missing(row.get("outputs")):
            has_outputs = False
    strata_values = {
        key: tuple(sorted(values))
        for key, values in sorted(candidate_strata.items())
        if 1 < len(values) <= _STRATA_CARDINALITY_LIMIT
        and not any(value.startswith("<") for value in values)
    }
    complete = expectation_keys or set()
    has_traces = bool(rows) and traced_rows == len(rows)
    return DatasetShape(
        row_count=len(rows),
        input_keys=tuple(sorted(input_keys)),
        has_outputs=has_outputs,
        expectation_keys=tuple(sorted(complete)),
        partial_expectation_keys=tuple(sorted(partial_expectation_keys - complete)),
        expectation_rows=tuple(expectation_rows),
        has_traces=has_traces,
        partial_traces=0 < traced_rows < len(rows),
        has_retrieval_spans=has_retrieval_spans,
        has_tool_spans=has_tool_spans,
        strata_values=strata_values,
    )


def _is_missing(value: Any) -> bool:
    """The several ways a value can be absent once a dataframe is involved.

    A Unity Catalog dataset arrives through ``DataFrame.to_dict("records")``,
    which represents a nullable column as ``NaN``/``NaT``/``pd.NA`` rather
    than ``None``. Treating those as present is how a nullable ``trace``
    column makes every row look traced — selecting the traces mode, which
    supplies no predict_fn, for rows that have no answer at all.

    Duck-typed on purpose: pandas is MLflow's dependency, not the SDK's, and
    ``bool(pd.NA)`` raises, so its type name is the safe thing to check.
    """

    return is_missing_scalar(value)


def _is_populated(value: Any) -> bool:
    """An empty expectation cannot satisfy a scorer's input contract."""

    if _is_missing(value):
        return False
    if isinstance(value, (str, list, tuple, dict, set)):
        return len(value) > 0
    return True


_RETRIEVER_SPAN_TYPE = "RETRIEVER"
_TOOL_SPAN_TYPE = "TOOL"
_SPAN_TYPE_KEYS = ("type", "span_type", "spanType")


def _trace_span_kinds(trace: Any) -> tuple[bool, bool]:
    """Which span kinds a row's trace carries.

    Retrieval judges need RETRIEVER spans and tool judges need tool spans;
    selecting both because a trace merely exists spends judge calls on
    scorers whose contract was never present.

    The span types are read from the spans, not from the serialized trace
    text. Scanning the payload for "retriever" or "tool" matches an answer
    that happens to use the word, and an LLM-only trace about retrieval
    tools then buys both sets of judges — which cannot score it, so the
    calls are wasted and the scorer errors fail the gate.

    Nesting does not matter here, only presence: MLflow judges top-level
    retriever spans, and a nested retriever always has one above it. A
    trace whose structure cannot be read reports neither kind — a scorer
    the toolkit cannot prove is applicable is not auto-selected.
    """

    types = _span_types(_spans(trace))
    return _RETRIEVER_SPAN_TYPE in types, _TOOL_SPAN_TYPE in types


def _span_types(spans: Iterable[Mapping[str, Any]]) -> set[str]:
    """The upper-cased span types these spans declare."""

    found: set[str] = set()
    for span in spans:
        for key in _SPAN_TYPE_KEYS:
            value = span.get(key)
            if isinstance(value, str) and value.strip():
                found.add(_span_type_name(value))
        attributes = span.get("attributes")
        if isinstance(attributes, Mapping):
            value = attributes.get("mlflow.spanType")
            if isinstance(value, str) and value.strip():
                found.add(_span_type_name(value))
    return found


def _span_type_name(value: str) -> str:
    """``"RETRIEVER"``, ``'"RETRIEVER"'`` and ``SpanType.RETRIEVER`` alike.

    MLflow stores the span type as a JSON-encoded attribute value, so the
    quotes are part of the string; an enum repr arrives dotted.
    """

    return value.strip().strip('"').rpartition(".")[2].upper()


@dataclass(frozen=True)
class RetrievalFanout:
    """How much retrieval the rows' own traces contain.

    A judge call is not one per row for the retrieval scorers: MLflow calls
    the judge once per RETRIEVER span (groundedness, sufficiency) and once
    per retrieved chunk (relevance). When the rows carry traces, that is
    countable rather than guessable, which is the difference between a
    budget ceiling and a hopeful number.
    """

    rows_counted: int = 0
    retriever_spans: int = 0
    retrieved_chunks: int = 0
    # Rows whose trace could be read at all. A traced row with no
    # retriever span is *counted as zero*, not unknown: the retrieval
    # scorers skip it, so assuming a span and a page of chunks for it
    # would refuse a budget the real run comes in well under.
    rows_with_traces: int = 0
    # Input characters in exactly the retriever spans/chunks the built-in
    # scorers judge. Cost estimation converts these aggregates to tokens;
    # keeping them beside the fan-out prevents non-retrieving rows from
    # diluting the price of a large retrieved context.
    retriever_span_input_characters: int = 0
    retrieval_sufficiency_input_characters: int = 0
    retrieved_chunk_input_characters: int = 0


def retrieval_fanout(rows: Sequence[Mapping[str, Any]]) -> RetrievalFanout:
    """Count retriever spans and retrieved chunks in the rows' traces."""

    rows_counted = spans = chunks = traced = 0
    span_input_characters = sufficiency_input_characters = 0
    chunk_input_characters = 0
    for row in rows:
        trace = row.get("trace") if isinstance(row, Mapping) else None
        if _is_missing(trace):
            continue
        # Readable structure is what makes "no retrieval here" a fact
        # rather than a guess; an unparseable trace still gets the
        # assumption, because rounding a budget down is what breaks it.
        if _spans(trace):
            traced += 1
        row_spans = _retriever_spans(trace)
        if not row_spans:
            continue
        rows_counted += 1
        spans += len(row_spans)
        chunks += sum(_chunk_count(span) for span in row_spans)
        (
            span_characters,
            sufficiency_characters,
            chunk_characters,
        ) = _retrieval_judge_input_characters(row, row_spans)
        span_input_characters += span_characters
        sufficiency_input_characters += sufficiency_characters
        chunk_input_characters += chunk_characters
    return RetrievalFanout(
        rows_counted=rows_counted,
        retriever_spans=spans,
        retrieved_chunks=chunks,
        rows_with_traces=traced,
        retriever_span_input_characters=span_input_characters,
        retrieval_sufficiency_input_characters=sufficiency_input_characters,
        retrieved_chunk_input_characters=chunk_input_characters,
    )


def _retrieval_judge_input_characters(
    row: Mapping[str, Any],
    spans: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int]:
    """Characters sent to span- and chunk-fan-out retrieval judges.

    Groundedness judges the request, recorded answer, and one retriever
    span's documents. Sufficiency judges the request, effective expected
    facts/response, and those documents instead; sharing groundedness's
    payload would omit the potentially large ground truth. Relevance judges
    one document at a time with the request. Counting these actual payloads
    is both more faithful and safer than multiplying an all-row average,
    which lets unrelated conversational rows make an expensive retrieval
    look artificially cheap.
    """

    trace = row.get("trace")
    request = _trace_request(trace)
    if request is None:
        request = row.get("inputs")
    response = _trace_response(trace)
    if response is None:
        response = row.get("outputs")
    expectations = row.get("expectations")
    ground_truth = None
    if isinstance(expectations, Mapping):
        selected = {
            key: _plain(expectations[key])
            for key in ("expected_facts", "expected_response")
            if key in expectations and _is_populated(expectations[key])
        }
        if selected:
            ground_truth = selected

    span_characters = sufficiency_characters = chunk_characters = 0
    for span in spans:
        outputs = _span_outputs(span)
        span_characters += _judge_payload_characters(request, response, outputs)
        sufficiency_characters += _judge_payload_characters(
            request, ground_truth, outputs
        )
        if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
            # An explicitly empty list is known zero retrieval relevance
            # calls. Only a missing/unreadable output remains unknown and
            # receives the conservative one-call assumption below.
            documents = list(outputs)
        else:
            documents = [outputs]
        for document in documents:
            chunk_characters += _judge_payload_characters(request, document)
    return span_characters, sufficiency_characters, chunk_characters


def _judge_payload_characters(*parts: Any) -> int:
    payload = [_plain(part) for part in parts if part is not None]
    return len(
        json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
    )


def _trace_document(trace: Any) -> Mapping[str, Any] | None:
    """A trace as a plain mapping, whatever form it arrived in.

    MLflow serialises a dataframe's ``trace`` column as a JSON string
    (``search_traces`` has done so since 3.2) and hands back ``Trace``
    objects elsewhere. Inspecting only mappings would leave the ordinary
    case uncounted, and an uncounted retrieval fan-out silently falls back
    to the assumed chunk count — which is how a budget stops being one.
    """

    if isinstance(trace, Mapping):
        return trace
    if isinstance(trace, (str, bytes)):
        try:
            loaded = json.loads(trace)
        except (ValueError, TypeError):
            return None
        return loaded if isinstance(loaded, Mapping) else None
    to_dict = getattr(trace, "to_dict", None)
    if callable(to_dict):
        try:
            document = to_dict()
        except Exception:  # pragma: no cover - exotic trace objects
            return None
        return document if isinstance(document, Mapping) else None
    return None


def trace_judge_text(trace: Any) -> str:
    """The part of a recorded trace a judge is actually shown.

    A trace-backed row can carry no ``inputs``/``outputs`` of its own while
    the trace holds the request, the response, and every retrieved chunk —
    which is precisely what the retrieval judges are handed. Estimating
    tokens from the row alone reports nearly nothing for exactly the runs
    that cost the most, so the cost estimate reads this instead.

    Span ids, timestamps, and attributes are left out: they are not sent to
    a judge, and counting them would replace an under-estimate with an
    over-estimate.
    """

    document = _trace_document(trace)
    if document is None:
        return ""
    parts: list[Any] = []
    request = _trace_request(trace)
    if request is not None:
        parts.append(request)
    response = _trace_response(trace)
    if response is not None:
        parts.append(response)
    for span in _spans(trace):
        if _is_retriever(span):
            documents = _span_outputs(span)
            if documents is not None:
                parts.append(_plain(documents))
    if not parts:
        return ""
    return json.dumps(parts, default=str)


def _spans(trace: Any) -> list[Mapping[str, Any]]:
    """The span records a trace carries, whatever form it arrived in.

    Both MLflow v2 and v3 ``Trace.to_dict()`` envelopes nest them under
    ``data``. A top-level ``spans`` field is not a supported alternative:
    pinned MLflow ignores it during ``Trace.from_dict``.
    """

    document = _trace_document(trace)
    if document is None:
        return []
    data = document.get("data")
    spans: Any = data.get("spans") if isinstance(data, Mapping) else None
    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)):
        return []
    return [span for span in spans if isinstance(span, Mapping)]


def _retriever_spans(trace: Any) -> list[Mapping[str, Any]]:
    """Top-level retriever spans, mirroring what MLflow actually judges.

    MLflow's ``_get_top_level_retrieval_spans`` skips a retriever span
    nested under another retriever, so counting every one would overstate
    the fan-out for a trace that retrieves inside a retriever.
    """

    spans = _spans(trace)
    by_id: dict[Any, Mapping[str, Any]] = {}
    for span in spans:
        identifier = _span_identifier(span)
        if identifier is not None:
            by_id[identifier] = span
    return [span for span in spans if _is_retriever(span) and not _nested(span, by_id)]


def _nested(span: Mapping[str, Any], by_id: Mapping[Any, Mapping[str, Any]]) -> bool:
    """True when an ancestor of ``span`` is itself a retriever span."""

    seen: set[Any] = set()
    parent_id = span.get(
        "parent_span_id",
        span.get("parentSpanId", span.get("parent_id")),
    )
    while parent_id is not None and parent_id not in seen:
        seen.add(parent_id)
        parent = by_id.get(parent_id)
        if parent is None:
            return False
        if _is_retriever(parent):
            return True
        parent_id = parent.get(
            "parent_span_id",
            parent.get("parentSpanId", parent.get("parent_id")),
        )
    return False


def _is_retriever(span: Mapping[str, Any]) -> bool:
    return _RETRIEVER_SPAN_TYPE in _span_types([span])


def _span_outputs(span: Mapping[str, Any]) -> Any | None:
    """What a span returned, wherever this serialization put it.

    ``Span.to_dict()`` stores it in ``attributes["mlflow.spanOutputs"]`` as
    a JSON string; a hand-written or partly-normalised span carries a plain
    ``outputs``. Both readers of this field — the chunk count and the token
    estimate — go through here, because the bug worth preventing is the two
    of them disagreeing about where to look.
    """

    attributes = span.get("attributes")
    candidates: list[Any] = [span.get("outputs"), span.get("output")]
    if isinstance(attributes, Mapping):
        candidates.append(attributes.get("mlflow.spanOutputs"))
    found: list[Any] = []
    for candidate in candidates:
        if isinstance(candidate, (str, bytes)):
            try:
                candidate = json.loads(candidate)
            except (ValueError, TypeError):
                # Not the serialized-JSON form, so it is a plain answer
                # string. Dropping it here lost the whole response from
                # the token estimate.
                pass
        if (
            isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes))
        ) or _is_populated(candidate):
            found.append(candidate)
    # A retriever can expose the same output in several MLflow shapes. An
    # empty lower-priority representation must not erase a populated one and
    # turn a real judge call into zero. Prefer the representation with the
    # most known documents, using serialized size to break equal-count ties.
    sequences = [
        candidate
        for candidate in found
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes))
    ]
    populated_sequences = [candidate for candidate in sequences if candidate]
    if populated_sequences:
        return max(
            populated_sequences,
            key=lambda candidate: (
                len(candidate),
                _judge_payload_characters(candidate),
            ),
        )
    non_sequences = [
        candidate
        for candidate in found
        if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes))
    ]
    if non_sequences:
        # A scalar/mapping conflicting with an empty list is not a known-zero
        # retrieval shape, so retain the conservative one-call path.
        return non_sequences[0]
    return sequences[0] if sequences else None


def _chunk_count(span: Mapping[str, Any]) -> int:
    """Documents returned; exact zero when empty, one when unknown."""

    outputs = _span_outputs(span)
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        return len(outputs)
    return 1


def _stratified_indices(
    rows: Sequence[Mapping[str, Any]],
    n: int,
    strata: tuple[str, ...],
    rng: random.Random,
) -> list[int]:
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, row in enumerate(rows):
        inputs = row.get("inputs")
        source = inputs if isinstance(inputs, Mapping) else {}
        key = tuple(str(source.get(name, "")) for name in strata)
        groups.setdefault(key, []).append(index)
    ordered_groups = [groups[key] for key in sorted(groups)]
    for group in ordered_groups:
        rng.shuffle(group)
    selected: list[int] = []
    position = 0
    while len(selected) < n:
        progressed = False
        for group in ordered_groups:
            if position < len(group):
                selected.append(group[position])
                progressed = True
                if len(selected) == n:
                    break
        if not progressed:
            break
        position += 1
    return sorted(selected)


def _load_file_rows(path: Path) -> tuple[list[Mapping[str, Any]], str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows_raw = [row for _, row in _load_jsonl_objects(text, subject=str(path))]
        source = "local-jsonl"
    else:
        try:
            rows_raw = json.loads(text)
        except json.JSONDecodeError as error:
            raise ConfigError(f"{path} is not valid JSON: {error}") from error
        source = "local-json"
    if not isinstance(rows_raw, list):
        raise ConfigError(f"{path} must contain a JSON list of rows")
    rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows_raw):
        if not isinstance(row, Mapping):
            raise ConfigError(f"{path} row {index} must be a JSON object")
        rows.append(row)
    return rows, source


def _load_jsonl_objects(
    text: str, *, subject: str
) -> list[tuple[int, Mapping[str, Any]]]:
    """Parse nonblank JSONL lines without echoing their untrusted contents."""

    rows: list[tuple[int, Mapping[str, Any]]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ConfigError(
                f"{subject} line {line_number} is not valid JSON "
                f"(column {error.colno}: {error.msg})"
            ) from error
        if not isinstance(row, Mapping):
            raise ConfigError(f"{subject} line {line_number} must be a JSON object")
        rows.append((line_number, row))
    return rows


def _looks_like_uc_dataset(reference: str, root: Path) -> bool:
    if "/" in reference or "\\" in reference:
        return False
    parts = reference.split(".")
    return len(parts) == 3 and all(part.strip() for part in parts)


def _load_uc_rows(reference: str, mlflow_module: Any | None) -> list[Mapping[str, Any]]:
    mlflow = mlflow_module
    if mlflow is None:
        try:
            import mlflow  # type: ignore[no-redef]
        except ImportError as error:
            raise missing_extra(
                f"Loading the Unity Catalog dataset {reference!r}", "genai"
            ) from error
    try:
        dataset = mlflow.genai.datasets.get_dataset(name=reference)
        frame = dataset.to_df()
        records = frame.to_dict("records")
    except Exception as error:
        raise ConfigError(
            f"could not load Unity Catalog dataset {reference!r}: {error}",
            remediation=(
                "Check the catalog.schema.table name, your workspace "
                "authentication, and that you hold SELECT on the table."
            ),
        ) from error
    return [row if isinstance(row, Mapping) else {"inputs": row} for row in records]


def _inputs_key(inputs: Any) -> str:
    return json.dumps(_plain(inputs), sort_keys=True, default=str)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _row_label(row: Mapping[str, Any]) -> str:
    inputs = row.get("inputs")
    if isinstance(inputs, Mapping):
        for key in ("question", "input", "query", "prompt"):
            if key in inputs:
                return str(inputs[key])[:80]
    return _inputs_key(inputs)[:80]
