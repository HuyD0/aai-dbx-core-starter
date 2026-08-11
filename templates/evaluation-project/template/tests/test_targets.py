import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aai_core.providers import ModelResponse
from app.targets import (
    MAX_QUESTION_CHARS,
    answer_sheet_predict_fn,
    endpoint_predict_fn,
    load_edge_cases,
    load_evaluation_cases,
)

ROOT = Path(__file__).resolve().parents[1]


def test_versioned_evaluation_data_has_unique_stable_case_ids():
    cases = load_evaluation_cases(ROOT / "evals" / "data" / "golden_cases.json")
    predict = answer_sheet_predict_fn(ROOT / "evals" / "data" / "answer_sheet.json")

    assert len(cases) == 10
    assert len({case["tags"]["case_id"] for case in cases}) == len(cases)
    assert "thirty days" in predict(cases[0]["inputs"]["question"])


def test_answer_sheet_rejects_duplicates_and_extra_fields(tmp_path):
    duplicate = [
        {"case_id": "one", "question": "Question?", "answer": "Answer."},
        {"case_id": "two", "question": "Question?", "answer": "Other."},
    ]
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate evaluation question"):
        answer_sheet_predict_fn(path)

    duplicate[1]["question"] = "Another?"
    duplicate[1]["unexpected"] = "field"
    path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValidationError):
        answer_sheet_predict_fn(path)


def test_golden_cases_require_case_ids_and_forbid_extra_fields(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "inputs": {"question": "Question?", "extra": "no"},
                    "expectations": {"expected_response": "Answer."},
                    "tags": {"case_id": "case-1"},
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_evaluation_cases(path)


def test_answer_sheet_must_match_reviewed_case_identity(tmp_path):
    cases = [
        {
            "inputs": {"question": "Question?"},
            "expectations": {"expected_response": "Answer."},
            "tags": {"case_id": "case-1"},
        }
    ]
    path = tmp_path / "answers.json"
    path.write_text(
        json.dumps(
            [{"case_id": "case-2", "question": "Question?", "answer": "Answer."}]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="case ids differ"):
        answer_sheet_predict_fn(path, expected_cases=cases)

    path.write_text(
        json.dumps(
            [{"case_id": "case-1", "question": "Different?", "answer": "Answer."}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="questions differ"):
        answer_sheet_predict_fn(path, expected_cases=cases)


def test_edge_cases_allow_empty_questions_but_reject_unknown_fields(tmp_path):
    path = tmp_path / "edge.json"
    path.write_text(
        json.dumps(
            [
                {
                    "inputs": {"question": ""},
                    "expectations": {"expected_response": "Ask for a question."},
                }
            ]
        ),
        encoding="utf-8",
    )
    assert load_edge_cases(path)[0]["inputs"]["question"] == ""

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[0]["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_edge_cases(path)


def test_endpoint_target_bounds_input_and_output():
    class Model:
        def __init__(self):
            self.requests = []

        def generate(self, messages, **options):
            self.requests.append((messages, options))
            return ModelResponse(
                content="bounded answer",
                provider="fake",
                logical_name="target-model",
                model="fake",
                latency_ms=1,
            )

    model = Model()
    context = SimpleNamespace(
        providers=SimpleNamespace(model=lambda logical_name: model)
    )
    predict = endpoint_predict_fn(context)

    assert predict("Question?") == "bounded answer"
    assert model.requests[0][1]["max_tokens"] == 1024
    with pytest.raises(ValueError, match="character bound"):
        predict("x" * (MAX_QUESTION_CHARS + 1))
    assert len(model.requests) == 1
