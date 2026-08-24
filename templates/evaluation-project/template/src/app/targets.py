"""Strict project datasets and adapters for things under evaluation.

AgentKit owns general target resolution. The Pydantic models here deliberately
remain project-local: the bundled golden suite and answer sheet have a narrower
contract with stable case IDs, bounded text, and an exact one-to-one mapping.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aai_core import PlatformContext
from aai_core.agentkit.config import ProjectContext
from aai_core.agentkit.targets import build_predict_fn, resolve_target

MAX_QUESTION_CHARS = 8_000
MAX_ANSWER_CHARS = 8_192


class AnswerRecord(BaseModel):
    """One reviewed, bounded answer keyed by stable case identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    answer: str = Field(min_length=1, max_length=MAX_ANSWER_CHARS)

    @field_validator("question", "answer")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class CaseInputs(BaseModel):
    """Strict inputs for a reviewed golden case."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class CaseExpectations(BaseModel):
    """Strict expected output for a reviewed case."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    expected_response: str = Field(min_length=1, max_length=MAX_ANSWER_CHARS)

    @field_validator("expected_response")
    @classmethod
    def reject_blank_expectation(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("expected response must not be blank")
        return value


class CaseTags(BaseModel):
    """Stable identity carried with a reviewed case."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")


class EvaluationCase(BaseModel):
    """One golden evaluation case."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    inputs: CaseInputs
    expectations: CaseExpectations
    tags: CaseTags

    @property
    def case_id(self) -> str:
        return self.tags.case_id


class EdgeInputs(BaseModel):
    """Inputs for a deliberately adversarial edge case."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    # Empty questions are intentional adversarial cases and remain valid here.
    question: str = Field(max_length=MAX_QUESTION_CHARS)


class EdgeCase(BaseModel):
    """One bounded edge case; edge cases remain report-only."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    inputs: EdgeInputs
    expectations: CaseExpectations


def load_evaluation_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load the golden suite and require unique stable case IDs."""

    payload = _json_records(path)
    cases = [EvaluationCase.model_validate(record, strict=True) for record in payload]
    _require_unique((case.case_id for case in cases), "case id")
    return [case.model_dump(mode="json") for case in cases]


def load_edge_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load bounded adversarial cases and reject duplicate questions."""

    payload = _json_records(path)
    cases = [EdgeCase.model_validate(record, strict=True) for record in payload]
    _require_unique((case.inputs.question for case in cases), "edge-case question")
    return [case.model_dump(mode="json") for case in cases]


def answer_sheet_predict_fn(
    path: str | Path,
    *,
    expected_cases: Sequence[Mapping[str, Any]] | None = None,
) -> Callable[[str], str]:
    """Replay strictly validated, uniquely keyed answers offline."""

    records = [
        AnswerRecord.model_validate(record, strict=True)
        for record in _json_records(path)
    ]
    _require_unique((record.case_id for record in records), "case id")
    _require_unique((record.question for record in records), "question")
    if expected_cases is not None:
        _require_answer_sheet_matches_cases(records, expected_cases)
    answers = {record.question: record.answer for record in records}

    def predict(question: str) -> str:
        _validate_question(question)
        return answers.get(question, "")

    return predict


def validate_bundled_data(root: str | Path, *, include_answer_sheet: bool) -> None:
    """Validate the generated project's reviewed data before evaluation.

    AgentKit additionally validates whichever dataset is selected in
    ``agentkit.yaml``. This narrower preflight protects the bundled examples'
    stable case-ID and answer-sheet contract before any target or judge call.
    """

    project_root = Path(root)
    data = project_root / "evals" / "data"
    cases = load_evaluation_cases(data / "golden_cases.json")
    load_edge_cases(data / "edge_cases.json")
    if include_answer_sheet:
        answer_sheet_predict_fn(data / "answer_sheet.json", expected_cases=cases)


def endpoint_predict_fn(
    context: PlatformContext,
    logical_name: str = "target-model",
) -> Callable[[str], str]:
    """Call a logical model with bounded input, tokens, and output."""

    model = context.providers.model(logical_name)

    def predict(question: str) -> str:
        _validate_question(question)
        response = model.generate(
            [{"role": "user", "content": question}],
            max_tokens=1_024,
        )
        if not response.content.strip() or len(response.content) > MAX_ANSWER_CHARS:
            raise RuntimeError("target returned an empty or oversized response")
        return response.content

    return predict


def agent_predict_fn(
    project: ProjectContext | None = None,
) -> Callable[..., Any]:
    """Resolve and build the target selected by ``agentkit.yaml``."""

    context = project or ProjectContext.load(
        Path(__file__).resolve().parents[2] / "agentkit.yaml"
    )
    target = resolve_target(
        context.config.agent,
        root=context.root,
        settings=context.settings,
    )
    predict = build_predict_fn(target, project=context)
    if predict is None:
        raise ValueError("the configured target is an answer sheet, not a live agent")
    return predict


def _json_records(path: str | Path) -> Sequence[Mapping[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(record, Mapping) for record in payload
    ):
        raise TypeError("evaluation data must be a JSON array of objects")
    return payload


def _validate_question(question: str) -> None:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"question exceeds the {MAX_QUESTION_CHARS}-character bound")


def _require_answer_sheet_matches_cases(
    records: Sequence[AnswerRecord],
    expected_cases: Sequence[Mapping[str, Any]],
) -> None:
    cases = [
        EvaluationCase.model_validate(case, strict=True) for case in expected_cases
    ]
    answers_by_id = {record.case_id: record.question for record in records}
    cases_by_id = {case.case_id: case.inputs.question for case in cases}
    if answers_by_id.keys() != cases_by_id.keys():
        missing = sorted(cases_by_id.keys() - answers_by_id.keys())
        extra = sorted(answers_by_id.keys() - cases_by_id.keys())
        raise ValueError(
            "answer sheet case ids differ from the reviewed cases "
            f"(missing={missing}, extra={extra})"
        )
    mismatched = sorted(
        case_id
        for case_id, question in cases_by_id.items()
        if answers_by_id[case_id] != question
    )
    if mismatched:
        raise ValueError(f"answer sheet questions differ for case ids: {mismatched}")


def _require_unique(values: Iterable[str], label: str) -> None:
    observed: set[str] = set()
    for value in values:
        if value in observed:
            raise ValueError(f"duplicate evaluation {label}: {value!r}")
        observed.add(value)
