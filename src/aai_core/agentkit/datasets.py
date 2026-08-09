"""Dataset loading, shape inference, digesting, and deterministic sampling.

Local JSON/JSONL files load with base dependencies only; Unity Catalog
evaluation datasets (``catalog.schema.table``) load lazily through MLflow.
The digest and the sampled subset are deterministic so a comparison always
names exactly which rows it scored.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aai_core.agentkit.errors import ConfigError, missing_extra

PLACEHOLDER_MARKERS = ("replace this", "replace-with", "todo", "changeme")
DEFAULT_MINIMUM_ROWS = 10
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
    """The question a recorded trace answered, if it can be recovered."""

    document = _trace_document(trace)
    if document is None:
        return None
    info = document.get("info")
    if isinstance(info, Mapping):
        for key in ("request_preview", "request"):
            value = info.get(key)
            if _is_populated(value):
                return _plain(value)
    data = document.get("data")
    spans = data.get("spans") if isinstance(data, Mapping) else None
    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)):
        return None
    for span in spans:
        if not isinstance(span, Mapping):
            continue
        if span.get("parent_span_id", span.get("parentSpanId")) is not None:
            continue
        for candidate in (
            span.get("inputs"),
            (
                (span.get("attributes") or {}).get("mlflow.spanInputs")
                if isinstance(span.get("attributes"), Mapping)
                else None
            ),
        ):
            if _is_populated(candidate):
                return _plain(candidate)
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

    try:
        records = json.loads(Path(sheet_path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(
            f"answer sheet {sheet_path} does not exist",
            remediation=(
                "Record the current outputs to the answer sheet, or run in "
                "live mode against a callable target."
            ),
        ) from error
    if not isinstance(records, list):
        raise ConfigError(f"answer sheet {sheet_path} must contain a JSON list")
    answers: dict[str, Any] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ConfigError(f"answer sheet {sheet_path} rows must be objects")
        if "question" in record and "answer" in record:
            answers[_inputs_key({"question": record["question"]})] = record["answer"]
        elif "inputs" in record and "outputs" in record:
            answers[_inputs_key(record["inputs"])] = record["outputs"]
        else:
            raise ConfigError(
                f"answer sheet {sheet_path} rows need question/answer or "
                "inputs/outputs fields"
            )

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
        if _is_populated(row.get("trace")):
            continue
        if not isinstance(inputs, Mapping) or not inputs:
            failures.append(f"row {index} is missing a non-empty inputs object")
            continue
        expectations = row.get("expectations")
        if expectations is not None and not isinstance(expectations, Mapping):
            failures.append(f"row {index} expectations must be an object")
            continue
        text = json.dumps(_plain(inputs), default=str).lower()
        if expectations:
            text += json.dumps(_plain(expectations), default=str).lower()
        if any(marker in text for marker in PLACEHOLDER_MARKERS):
            failures.append(f"row {index} still contains placeholder text")
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

    if value is None:
        return True
    if isinstance(value, float):
        return value != value  # NaN is the only float unequal to itself
    return type(value).__name__ in {"NAType", "NaTType"}


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


def retrieval_fanout(rows: Sequence[Mapping[str, Any]]) -> RetrievalFanout:
    """Count retriever spans and retrieved chunks in the rows' traces."""

    rows_counted = spans = chunks = 0
    for row in rows:
        trace = row.get("trace") if isinstance(row, Mapping) else None
        if _is_missing(trace):
            continue
        row_spans = _retriever_spans(trace)
        if not row_spans:
            continue
        rows_counted += 1
        spans += len(row_spans)
        chunks += sum(_chunk_count(span) for span in row_spans)
    return RetrievalFanout(
        rows_counted=rows_counted, retriever_spans=spans, retrieved_chunks=chunks
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
    info = document.get("info")
    if isinstance(info, Mapping):
        for key in ("response_preview", "response"):
            value = info.get(key)
            if _is_populated(value):
                parts.append(_plain(value))
                break
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

    ``Trace.to_dict()`` nests them under ``data``; some payloads carry them
    at the top level. A shape with neither yields nothing, which is what
    makes every span question answerable without guessing from text.
    """

    document = _trace_document(trace)
    if document is None:
        return []
    spans: Any = None
    data = document.get("data")
    if isinstance(data, Mapping):
        spans = data.get("spans")
    if spans is None:
        spans = document.get("spans")
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
        identifier = span.get("span_id", span.get("spanId"))
        if identifier is not None:
            by_id[identifier] = span
    return [span for span in spans if _is_retriever(span) and not _nested(span, by_id)]


def _nested(span: Mapping[str, Any], by_id: Mapping[Any, Mapping[str, Any]]) -> bool:
    """True when an ancestor of ``span`` is itself a retriever span."""

    seen: set[Any] = set()
    parent_id = span.get("parent_span_id", span.get("parentSpanId"))
    while parent_id is not None and parent_id not in seen:
        seen.add(parent_id)
        parent = by_id.get(parent_id)
        if parent is None:
            return False
        if _is_retriever(parent):
            return True
        parent_id = parent.get("parent_span_id", parent.get("parentSpanId"))
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
                continue
        if _is_populated(candidate):
            found.append(candidate)
    # A list of documents is the shape both callers want; prefer it over a
    # scalar or mapping that some other key happened to hold.
    for candidate in found:
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            return candidate
    return found[0] if found else None


def _chunk_count(span: Mapping[str, Any]) -> int:
    """Documents a retriever span returned; 1 when the shape is unknown.

    Never zero: a span that returned nothing countable is still one judge
    call's worth of uncertainty, and rounding a cost estimate down is the
    direction that breaks a budget.
    """

    outputs = _span_outputs(span)
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        return max(1, len(outputs))
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
        rows_raw = [json.loads(line) for line in text.splitlines() if line.strip()]
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
