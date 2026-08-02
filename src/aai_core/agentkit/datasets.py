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
from collections.abc import Mapping, Sequence
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
    has_retrieval_spans: bool = False
    has_tool_spans: bool = False


@dataclass(frozen=True)
class LoadedDataset:
    ref: str
    source: str
    rows: tuple[Mapping[str, Any], ...]
    digest: str
    shape: DatasetShape


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


def dataset_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Stable 16-hex digest of the canonical JSON form of the rows."""

    canonical = json.dumps(
        [_plain(row) for row in rows],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


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
    return _build_dataset(dataset.ref, f"{dataset.source}+sample", rows)


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
    return _build_dataset(dataset.ref, f"{dataset.source}+answers", rows)


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
        if "trace" in row:
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
    ref: str, source: str, rows: Sequence[Mapping[str, Any]]
) -> LoadedDataset:
    frozen_rows = tuple(rows)
    return LoadedDataset(
        ref=ref,
        source=source,
        rows=frozen_rows,
        digest=dataset_digest(frozen_rows),
        shape=_infer_shape(frozen_rows),
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
    has_traces = False
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
        if "trace" in row:
            has_traces = True
            retrieval, tools = _trace_span_kinds(row["trace"])
            has_retrieval_spans = has_retrieval_spans or retrieval
            has_tool_spans = has_tool_spans or tools
        if row.get("outputs") is None:
            has_outputs = False
    strata_values = {
        key: tuple(sorted(values))
        for key, values in sorted(candidate_strata.items())
        if 1 < len(values) <= _STRATA_CARDINALITY_LIMIT
        and not any(value.startswith("<") for value in values)
    }
    complete = expectation_keys or set()
    return DatasetShape(
        row_count=len(rows),
        input_keys=tuple(sorted(input_keys)),
        has_outputs=has_outputs,
        expectation_keys=tuple(sorted(complete)),
        partial_expectation_keys=tuple(sorted(partial_expectation_keys - complete)),
        has_traces=has_traces,
        has_retrieval_spans=has_retrieval_spans,
        has_tool_spans=has_tool_spans,
        strata_values=strata_values,
    )


def _is_populated(value: Any) -> bool:
    """An empty expectation cannot satisfy a scorer's input contract."""

    if value is None:
        return False
    if isinstance(value, (str, list, tuple, dict, set)):
        return len(value) > 0
    return True


_RETRIEVAL_SPAN_MARKERS = ("RETRIEVER", '"retriever"')
_TOOL_SPAN_MARKERS = ("TOOL", '"tool"', "tool_calls")


def _trace_span_kinds(trace: Any) -> tuple[bool, bool]:
    """Which span kinds a row's trace carries.

    Retrieval judges need RETRIEVER spans and tool judges need tool spans;
    selecting both because a trace merely exists spends judge calls on
    scorers whose contract was never present. Detection is deliberately
    tolerant of trace shape, and unknown shapes report neither — a scorer
    the toolkit cannot prove is applicable is not auto-selected.
    """

    try:
        payload = json.dumps(_plain(trace), default=str)
    except (TypeError, ValueError):  # pragma: no cover - exotic trace objects
        payload = str(trace)
    upper = payload.upper()
    retrieval = any(marker.upper() in upper for marker in _RETRIEVAL_SPAN_MARKERS)
    tools = any(marker.upper() in upper for marker in _TOOL_SPAN_MARKERS)
    return retrieval, tools


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
        if trace is None:
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


def _retriever_spans(trace: Any) -> list[Any]:
    plain = _plain(trace)
    spans: Any = None
    if isinstance(plain, Mapping):
        data = plain.get("data")
        if isinstance(data, Mapping):
            spans = data.get("spans")
        if spans is None:
            spans = plain.get("spans")
    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)):
        return []
    found = []
    for span in spans:
        if isinstance(span, Mapping) and _is_retriever(span):
            found.append(span)
    return found


def _is_retriever(span: Mapping[str, Any]) -> bool:
    for key in ("type", "span_type", "spanType"):
        value = span.get(key)
        if isinstance(value, str) and value.upper() == "RETRIEVER":
            return True
    attributes = span.get("attributes")
    if isinstance(attributes, Mapping):
        value = attributes.get("mlflow.spanType")
        if isinstance(value, str) and "RETRIEVER" in value.upper():
            return True
    return False


def _chunk_count(span: Mapping[str, Any]) -> int:
    """Documents a retriever span returned; 1 when the shape is unknown.

    Never zero: a span that returned nothing countable is still one judge
    call's worth of uncertainty, and rounding a cost estimate down is the
    direction that breaks a budget.
    """

    candidates: list[Any] = [span.get("outputs")]
    attributes = span.get("attributes")
    if isinstance(attributes, Mapping):
        candidates.append(attributes.get("mlflow.spanOutputs"))
    for candidate in candidates:
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            return max(1, len(candidate))
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
