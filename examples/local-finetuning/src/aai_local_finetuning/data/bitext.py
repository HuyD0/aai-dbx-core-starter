"""Local-only adapter for the Bitext customer-support CSV and Kaggle ZIP."""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from .schemas import BitextLoadResult, RawBitextRow

BITEXT_COLUMNS = ("flags", "instruction", "category", "intent", "response")
_MAX_ZIP_MEMBER_BYTES = 1_000_000_000


def load_bitext(path: str | Path) -> BitextLoadResult:
    """Read Bitext records from an existing local CSV or Kaggle ZIP.

    This function never downloads data and never extracts a ZIP onto disk. A ZIP must
    contain exactly one CSV whose header includes the five documented Bitext fields.
    """

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Bitext input does not exist: {source}")
    if not source.is_file():
        raise ValueError(f"Bitext input must be a file: {source}")
    suffix = source.suffix.lower()
    if suffix not in {".csv", ".zip"}:
        raise ValueError("Bitext input must be a local .csv or .zip file")

    source_member = _select_zip_member(source) if suffix == ".zip" else None
    with _open_text(source, source_member) as stream:
        reader = csv.DictReader(stream)
        headers = _validated_headers(reader.fieldnames)
        records: list[RawBitextRow] = []
        invalid_csv_rows = 0
        for source_row, row in enumerate(reader, start=2):
            if None in row:
                invalid_csv_rows += 1
            selected = {
                column: _csv_string(row.get(column)) for column in BITEXT_COLUMNS
            }
            records.append(RawBitextRow(source_row=source_row, **selected))

    return BitextLoadResult(
        records=tuple(records),
        headers=headers,
        invalid_csv_rows=invalid_csv_rows,
        source_member=source_member,
    )


def _select_zip_member(path: Path) -> str:
    candidates: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                if info.is_dir() or not info.filename.lower().endswith(".csv"):
                    continue
                if info.file_size > _MAX_ZIP_MEMBER_BYTES:
                    raise ValueError(
                        "ZIP CSV exceeds the "
                        f"{_MAX_ZIP_MEMBER_BYTES}-byte safety limit: "
                        f"{info.filename}"
                    )
                with archive.open(info, "r") as raw:
                    with io.TextIOWrapper(
                        raw, encoding="utf-8-sig", newline=""
                    ) as text:
                        reader = csv.reader(text)
                        try:
                            headers = next(reader)
                        except StopIteration:
                            continue
                if set(BITEXT_COLUMNS).issubset(headers):
                    candidates.append(info.filename)
    except zipfile.BadZipFile as error:
        raise ValueError(f"Invalid ZIP file: {path}") from error

    if not candidates:
        joined = ", ".join(BITEXT_COLUMNS)
        raise ValueError(f"ZIP contains no CSV with required columns: {joined}")
    if len(candidates) > 1:
        names = ", ".join(candidates)
        raise ValueError(
            "ZIP contains multiple matching Bitext CSV files; provide one CSV "
            f"directly instead: {names}"
        )
    return candidates[0]


@contextmanager
def _open_text(path: Path, member: str | None) -> Iterator[TextIO]:
    if member is None:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            yield stream
        return

    with zipfile.ZipFile(path) as archive:
        with archive.open(member, "r") as raw:
            with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as stream:
                yield stream


def _validated_headers(headers: list[str] | None) -> tuple[str, ...]:
    if headers is None:
        raise ValueError("Bitext CSV has no header row")
    missing = [column for column in BITEXT_COLUMNS if column not in headers]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Bitext CSV is missing required columns: {joined}")
    return tuple(headers)


def _csv_string(value: str | list[str] | None) -> str:
    return value if isinstance(value, str) else ""
