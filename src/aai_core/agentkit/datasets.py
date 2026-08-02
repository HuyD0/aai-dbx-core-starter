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
    expectation_keys: tuple[str, ...]
    has_traces: bool
    strata_values: Mapping[str, tuple[str, ...]]


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
    expectation_keys: set[str] = set()
    has_traces = False
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
        if isinstance(expectations, Mapping):
            expectation_keys.update(str(key) for key in expectations)
        if "trace" in row:
            has_traces = True
        if row.get("outputs") is None:
            has_outputs = False
    strata_values = {
        key: tuple(sorted(values))
        for key, values in sorted(candidate_strata.items())
        if 1 < len(values) <= _STRATA_CARDINALITY_LIMIT
        and not any(value.startswith("<") for value in values)
    }
    return DatasetShape(
        row_count=len(rows),
        input_keys=tuple(sorted(input_keys)),
        has_outputs=has_outputs,
        expectation_keys=tuple(sorted(expectation_keys)),
        has_traces=has_traces,
        strata_values=strata_values,
    )


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
