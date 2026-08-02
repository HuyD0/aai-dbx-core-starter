"""Unit tests for dataset loading, digesting, sampling, and validation."""

import json
from types import SimpleNamespace

import pytest

from aai_core.agentkit.datasets import (
    attach_answer_sheet,
    dataset_digest,
    load_dataset,
    smoke_sample,
    validate_dataset,
)
from aai_core.agentkit.errors import ConfigError


def _rows(count=12, category=None):
    rows = []
    for index in range(count):
        row = {
            "inputs": {"question": f"question {index}"},
            "expectations": {"expected_response": f"answer {index}"},
        }
        if category is not None:
            row["inputs"]["category"] = category[index % len(category)]
        rows.append(row)
    return rows


def _write_dataset(tmp_path, rows, name="golden.json"):
    path = tmp_path / name
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_load_json_and_jsonl(tmp_path):
    rows = _rows(3)
    _write_dataset(tmp_path, rows)
    jsonl = tmp_path / "cases.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    from_json = load_dataset("golden.json", root=tmp_path)
    from_jsonl = load_dataset("cases.jsonl", root=tmp_path)

    assert from_json.source == "local-json"
    assert from_jsonl.source == "local-jsonl"
    assert from_json.digest == from_jsonl.digest
    assert from_json.shape.row_count == 3
    assert from_json.shape.input_keys == ("question",)
    assert from_json.shape.expectation_keys == ("expected_response",)
    assert from_json.shape.has_outputs is False
    assert from_json.shape.has_traces is False


def test_rows_must_be_objects(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(["not-a-row"]), encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        load_dataset("bad.json", root=tmp_path)
    assert "row 0" in str(excinfo.value)


def test_missing_file_error_names_the_path(tmp_path):
    with pytest.raises(ConfigError) as excinfo:
        load_dataset("evals/data/nope.json", root=tmp_path)
    assert "nope.json" in str(excinfo.value)


def test_uc_reference_loads_through_mlflow(tmp_path):
    frame = SimpleNamespace(
        to_dict=lambda orient: [
            {"inputs": {"question": "q"}, "expectations": {"expected_response": "a"}}
        ]
    )
    fake_mlflow = SimpleNamespace(
        genai=SimpleNamespace(
            datasets=SimpleNamespace(
                get_dataset=lambda name: SimpleNamespace(to_df=lambda: frame)
            )
        )
    )

    dataset = load_dataset(
        "main.evaluation.golden_set", root=tmp_path, mlflow_module=fake_mlflow
    )

    assert dataset.source == "uc-dataset"
    assert dataset.shape.row_count == 1


def test_digest_is_stable_and_content_sensitive():
    rows = _rows(5)

    assert dataset_digest(rows) == dataset_digest([dict(row) for row in rows])
    assert dataset_digest(rows) != dataset_digest(rows[:4])
    assert len(dataset_digest(rows)) == 16


def test_smoke_sample_is_deterministic(tmp_path):
    _write_dataset(tmp_path, _rows(30))
    dataset = load_dataset("golden.json", root=tmp_path)

    first = smoke_sample(dataset, 10)
    second = smoke_sample(dataset, 10)

    assert first.shape.row_count == 10
    assert first.digest == second.digest
    assert first.source == "local-json+sample"
    assert smoke_sample(dataset, 100) is dataset


def test_smoke_sample_stratifies(tmp_path):
    rows = _rows(30, category=["billing", "policy", "safety"])
    _write_dataset(tmp_path, rows)
    dataset = load_dataset("golden.json", root=tmp_path)

    sample = smoke_sample(dataset, 6, strata=("category",))

    categories = {row["inputs"]["category"] for row in sample.rows}
    assert categories == {"billing", "policy", "safety"}
    assert sample.shape.row_count == 6


def test_strata_values_respect_cardinality_limit(tmp_path):
    rows = _rows(12, category=["a", "b", "c"])
    _write_dataset(tmp_path, rows)

    dataset = load_dataset("golden.json", root=tmp_path)

    assert dataset.shape.strata_values["category"] == ("a", "b", "c")
    # question is unique per row -> above the cardinality limit -> not strata
    assert "question" not in dataset.shape.strata_values


def test_attach_answer_sheet_template_shape(tmp_path):
    rows = _rows(3)
    _write_dataset(tmp_path, rows)
    sheet = tmp_path / "answers.json"
    sheet.write_text(
        json.dumps(
            [
                {"question": f"question {index}", "answer": f"recorded {index}"}
                for index in range(3)
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_dataset("golden.json", root=tmp_path)
    replayed = attach_answer_sheet(dataset, sheet)

    assert replayed.shape.has_outputs is True
    assert replayed.rows[0]["outputs"] == "recorded 0"
    assert replayed.source.endswith("+answers")


def test_attach_answer_sheet_generic_shape(tmp_path):
    rows = _rows(2)
    _write_dataset(tmp_path, rows)
    sheet = tmp_path / "answers.json"
    sheet.write_text(
        json.dumps(
            [
                {"inputs": row["inputs"], "outputs": f"generic {index}"}
                for index, row in enumerate(rows)
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_dataset("golden.json", root=tmp_path)
    replayed = attach_answer_sheet(dataset, sheet)

    assert replayed.rows[1]["outputs"] == "generic 1"


def test_attach_answer_sheet_reports_missing_rows(tmp_path):
    _write_dataset(tmp_path, _rows(3))
    sheet = tmp_path / "answers.json"
    sheet.write_text(
        json.dumps([{"question": "question 0", "answer": "only one"}]),
        encoding="utf-8",
    )

    dataset = load_dataset("golden.json", root=tmp_path)
    with pytest.raises(ConfigError) as excinfo:
        attach_answer_sheet(dataset, sheet)
    message = str(excinfo.value)
    assert "2 row(s)" in message
    assert "question 1" in message


def test_validate_dataset_reports_structural_failures(tmp_path):
    rows = _rows(3)
    rows.append({"inputs": {}})
    rows.append({"inputs": {"question": "replace this with a real question"}})
    _write_dataset(tmp_path, rows)

    dataset = load_dataset("golden.json", root=tmp_path)
    failures = validate_dataset(dataset, minimum_rows=10)

    assert any("5 rows" in failure for failure in failures)
    assert any("row 3" in failure for failure in failures)
    assert any("placeholder" in failure for failure in failures)


def test_validate_dataset_passes_clean_rows(tmp_path):
    _write_dataset(tmp_path, _rows(12))

    dataset = load_dataset("golden.json", root=tmp_path)

    assert validate_dataset(dataset) == []


def test_expectation_keys_require_every_row(tmp_path):
    """Coverage is per row: a field half the rows carry is not available.

    keyword_coverage scores a missing expected response as a vacuous 1.0,
    so treating a partially-present field as dataset-wide inflates the
    aggregate the gate reads.
    """

    rows = [
        {"inputs": {"question": "a"}, "expectations": {"expected_response": "yes"}},
        {"inputs": {"question": "b"}},
    ]
    _write_dataset(tmp_path, rows)

    dataset = load_dataset("golden.json", root=tmp_path)

    assert dataset.shape.expectation_keys == ()
    assert dataset.shape.partial_expectation_keys == ("expected_response",)


def test_empty_expectation_values_do_not_count_as_present(tmp_path):
    rows = [
        {"inputs": {"question": "a"}, "expectations": {"expected_response": "yes"}},
        {"inputs": {"question": "b"}, "expectations": {"expected_response": ""}},
    ]
    _write_dataset(tmp_path, rows)

    dataset = load_dataset("golden.json", root=tmp_path)

    assert dataset.shape.expectation_keys == ()
    assert dataset.shape.partial_expectation_keys == ("expected_response",)


def test_fully_covered_expectations_are_available(tmp_path):
    _write_dataset(tmp_path, _rows(3))

    dataset = load_dataset("golden.json", root=tmp_path)

    assert dataset.shape.expectation_keys == ("expected_response",)
    assert dataset.shape.partial_expectation_keys == ()


def test_span_kinds_are_distinguished(tmp_path):
    retrieval = [
        {
            "inputs": {"question": "a"},
            "trace": {"data": {"spans": [{"type": "RETRIEVER", "name": "search"}]}},
        }
    ]
    tools = [
        {
            "inputs": {"question": "a"},
            "trace": {"data": {"spans": [{"type": "TOOL", "name": "lookup"}]}},
        }
    ]
    _write_dataset(tmp_path, retrieval, name="retrieval.json")
    _write_dataset(tmp_path, tools, name="tools.json")

    retrieval_shape = load_dataset("retrieval.json", root=tmp_path).shape
    tool_shape = load_dataset("tools.json", root=tmp_path).shape

    assert retrieval_shape.has_traces and tool_shape.has_traces
    assert retrieval_shape.has_retrieval_spans and not retrieval_shape.has_tool_spans
    assert tool_shape.has_tool_spans and not tool_shape.has_retrieval_spans


def test_retrieval_fanout_counts_spans_and_chunks():
    """Judge calls fan out per span and per chunk, so they get counted.

    MLflow calls the retrieval-relevance judge once per retrieved chunk;
    a per-row count would understate a RAG run's cost several times over.
    """

    from aai_core.agentkit.datasets import retrieval_fanout

    rows = [
        {
            "inputs": {"question": "a"},
            "trace": {
                "data": {
                    "spans": [
                        {"type": "RETRIEVER", "outputs": [{}, {}, {}]},
                        {"type": "LLM", "outputs": [{}]},
                    ]
                }
            },
        },
        {
            "inputs": {"question": "b"},
            "trace": {
                "data": {
                    "spans": [
                        {"span_type": "RETRIEVER", "outputs": [{}, {}]},
                        {"span_type": "RETRIEVER", "outputs": [{}]},
                    ]
                }
            },
        },
        {"inputs": {"question": "c"}},
    ]

    fanout = retrieval_fanout(rows)

    assert fanout.rows_counted == 2
    assert fanout.retriever_spans == 3
    assert fanout.retrieved_chunks == 6


def test_retrieval_fanout_counts_a_span_of_unknown_shape_as_one_chunk():
    from aai_core.agentkit.datasets import retrieval_fanout

    rows = [
        {
            "inputs": {"question": "a"},
            "trace": {
                "data": {
                    "spans": [
                        {"attributes": {"mlflow.spanType": '"RETRIEVER"'}},
                    ]
                }
            },
        }
    ]

    fanout = retrieval_fanout(rows)

    assert fanout.retriever_spans == 1
    assert fanout.retrieved_chunks == 1


def test_retrieval_fanout_ignores_rows_without_retrieval():
    from aai_core.agentkit.datasets import retrieval_fanout

    assert retrieval_fanout([{"inputs": {"q": "a"}}]).rows_counted == 0
