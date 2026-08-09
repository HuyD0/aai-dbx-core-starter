"""Unit tests for dataset loading, digesting, sampling, and validation."""

import base64
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from aai_core.agentkit.datasets import (
    attach_answer_sheet,
    dataset_digest,
    effective_dataset,
    evaluation_rows,
    load_dataset,
    rows_missing_inputs,
    smoke_sample,
    trace_expectation_overrides,
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


def _encoded_id(value, width):
    return base64.b64encode(int(value).to_bytes(width, "big")).decode("ascii")


def _mlflow_span(
    index=0,
    *,
    question="q",
    output="a",
    span_type="LLM",
    parent_span_id=None,
):
    """A complete MLflow 3.14 v3 span dictionary."""

    request_id = f"tr-{index:032x}"
    attributes = {
        "mlflow.traceRequestId": json.dumps(request_id),
        "mlflow.spanType": json.dumps(span_type),
        "mlflow.spanOutputs": json.dumps(output),
    }
    if question is not None:
        attributes["mlflow.spanInputs"] = json.dumps({"question": question})
    return {
        "trace_id": _encoded_id(index + 1, 16),
        "span_id": _encoded_id(index + 1, 8),
        "parent_span_id": parent_span_id,
        "name": f"span-{index}",
        "start_time_unix_nano": 1_786_000_000_000_000_000 + index,
        "end_time_unix_nano": 1_786_000_000_001_000_000 + index,
        "events": [],
        "status": {"code": "STATUS_CODE_OK", "message": ""},
        "attributes": attributes,
        "links": [],
    }


def _mlflow_expectation(index, name, expectation):
    return {
        "assessment_id": f"a-{index:032x}",
        "assessment_name": name,
        "trace_id": f"tr-{index:032x}",
        "source": {"source_type": "HUMAN", "source_id": "test"},
        "create_time": "2026-08-09T00:00:00Z",
        "last_update_time": "2026-08-09T00:00:00Z",
        "expectation": expectation,
    }


def _mlflow_trace(
    index=0,
    *,
    question="q",
    output="a",
    spans=None,
    assessments=(),
    request_preview=True,
):
    """A complete MLflow 3.14 v3 Trace.to_dict-compatible envelope."""

    info = {
        "trace_id": f"tr-{index:032x}",
        "trace_location": {
            "type": "MLFLOW_EXPERIMENT",
            "mlflow_experiment": {"experiment_id": "0"},
        },
        "request_time": "2026-08-09T00:00:00Z",
        "state": "OK",
        "trace_metadata": {},
        "tags": {},
        "assessments": list(assessments),
    }
    if request_preview:
        info["request_preview"] = json.dumps({"question": question})
        info["response_preview"] = json.dumps(output)
    return {
        "info": info,
        "data": {
            "spans": (
                [_mlflow_span(index, question=question, output=output)]
                if spans is None
                else spans
            )
        },
    }


def _mlflow_v2_trace():
    request_id = "tr-0123456789abcdef0123456789abcdef"
    return {
        "info": {
            "request_id": request_id,
            "experiment_id": "0",
            "timestamp_ms": 1,
            "execution_time_ms": 1,
            "status": "OK",
            "request_metadata": {},
            "tags": {},
            "assessments": [],
        },
        "data": {
            "spans": [
                {
                    "context": {
                        "trace_id": "0123456789abcdef0123456789abcdef",
                        "span_id": "0123456789abcdef",
                    },
                    "parent_id": None,
                    "name": "agent",
                    "start_time": 1,
                    "end_time": 2,
                    "attributes": {
                        "mlflow.traceRequestId": json.dumps(request_id),
                        "mlflow.spanInputs": json.dumps({"question": "q"}),
                        "mlflow.spanOutputs": json.dumps("a"),
                    },
                    "status_code": "OK",
                    "status_message": "",
                    "events": [],
                    "links": [],
                }
            ]
        },
    }


def _trace_with_event_and_link(trace):
    document = json.loads(json.dumps(trace))
    span = document["data"]["spans"][0]
    if "context" in span:
        span["events"] = [
            {"name": "retrieved", "timestamp": 2, "attributes": {"count": 1}}
        ]
    else:
        span["events"] = [
            {
                "name": "retrieved",
                "time_unix_nano": 1_786_000_000_000_000_001,
                "attributes": {"count": 1},
            }
        ]
    span["links"] = [
        {
            "trace_id": "tr-11111111111111111111111111111111",
            "span_id": "0123456789abcdef",
            "attributes": {"relationship": "handoff"},
        }
    ]
    return document


def _malformed_complete_trace(case):
    document = _mlflow_v2_trace() if case.startswith("v2-") else _mlflow_trace()
    span = document["data"]["spans"][0]
    if case == "bad-request-time":
        document["info"]["request_time"] = "not-a-timestamp"
    elif case == "list-state":
        document["info"]["state"] = ["OK"]
    elif case == "mapping-location-type":
        document["info"]["trace_location"]["type"] = {"name": "experiment"}
    elif case == "bad-trace-id":
        span["trace_id"] = "not-base64!"
    elif case == "bad-span-id":
        span["span_id"] = "not-base64!"
    elif case == "bad-span-status":
        span["status"]["code"] = "STATUS_CODE_UNKNOWN"
    elif case == "bad-event":
        span["events"] = [{"time_unix_nano": 1, "attributes": {}}]
    elif case == "bad-link":
        span["links"] = [{"trace_id": "tr-" + "1" * 32}]
    elif case == "bad-request-id-json":
        span["attributes"]["mlflow.traceRequestId"] = json.dumps(["not", "an-id"])
    elif case == "bad-request-id-encoding":
        span["attributes"]["mlflow.traceRequestId"] = "not-json"
    elif case == "mismatched-request-id":
        span["attributes"]["mlflow.traceRequestId"] = json.dumps("tr-other")
    elif case == "v2-bad-timestamp":
        document["info"]["timestamp_ms"] = "not-an-integer"
    elif case == "v2-bad-trace-id":
        span["context"]["trace_id"] = "not-hex"
    elif case == "v2-bad-span-status":
        span["status_code"] = "UNKNOWN"
    elif case == "v2-bad-event":
        span["events"] = [{"timestamp": 1, "attributes": {}}]
    elif case == "v2-bad-link":
        span["links"] = [{"trace_id": "tr-" + "1" * 32}]
    elif case == "v2-bad-request-id-json":
        span["attributes"]["mlflow.traceRequestId"] = json.dumps({"id": "bad"})
    else:  # pragma: no cover - test table and builder must stay in lockstep
        raise AssertionError(case)
    return document


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


def test_trace_coverage_is_per_row(tmp_path):
    """One traced row does not make a dataset trace-backed.

    A traces run supplies no predict_fn, so an untraced row has no answer
    at all — it can only be skipped or error. Partial coverage must not
    select the mode.
    """

    rows = [
        {
            "inputs": {"question": "a"},
            "trace": {"data": {"spans": [{"type": "RETRIEVER"}]}},
        },
        {"inputs": {"question": "b"}},
    ]
    _write_dataset(tmp_path, rows)

    shape = load_dataset("golden.json", root=tmp_path).shape

    assert shape.has_traces is False
    assert shape.partial_traces is True


def test_null_trace_column_does_not_count_as_traced(tmp_path):
    """A nullable Unity Catalog trace column yields `trace: null`."""

    rows = [
        {
            "inputs": {"question": "a"},
            "trace": {"data": {"spans": [{"type": "RETRIEVER"}]}},
        },
        {"inputs": {"question": "b"}, "trace": None},
    ]
    _write_dataset(tmp_path, rows)

    shape = load_dataset("golden.json", root=tmp_path).shape

    assert shape.has_traces is False
    assert shape.partial_traces is True


def test_fully_traced_rows_are_trace_backed(tmp_path):
    rows = [
        {
            "inputs": {"question": name},
            "trace": {"data": {"spans": [{"type": "RETRIEVER"}]}},
        }
        for name in ("a", "b")
    ]
    _write_dataset(tmp_path, rows)

    shape = load_dataset("golden.json", root=tmp_path).shape

    assert shape.has_traces is True
    assert shape.partial_traces is False


def test_a_row_with_a_null_trace_still_needs_inputs(tmp_path):
    """`trace: null` must not exempt a row from structural validation."""

    rows = [{"inputs": {}, "trace": None}] + _rows(11)
    _write_dataset(tmp_path, rows)

    failures = validate_dataset(load_dataset("golden.json", root=tmp_path))

    assert any("row 0" in failure for failure in failures)


def test_trace_payload_is_not_dataset_identity(tmp_path):
    """A trace is the answer, not the question.

    Two runs over the same production questions carry different trace ids,
    timestamps, and responses. Hashing those would give the very behaviour
    under comparison a new dataset identity, and the comparability check
    would reject it as different data.
    """

    def _rows_with(trace_id, answer):
        return [
            {
                "inputs": {"question": "how do I retire early?"},
                "trace": {
                    "info": {"trace_id": trace_id},
                    "data": {"spans": [{"type": "LLM", "outputs": answer}]},
                },
            }
        ]

    _write_dataset(tmp_path, _rows_with("tr-1", "first answer"), name="a.json")
    _write_dataset(tmp_path, _rows_with("tr-2", "second answer"), name="b.json")

    first = load_dataset("a.json", root=tmp_path)
    second = load_dataset("b.json", root=tmp_path)

    assert first.digest == second.digest


def test_trace_expectation_assessments_are_dataset_identity(tmp_path):
    """Trace ground truth is identity only in the mode that scores it."""

    def _rows_with(expected_response):
        return [
            {
                "inputs": {"question": "how do I retire early?"},
                "trace": {
                    "info": {
                        "trace_id": "volatile-id",
                        "assessments": [
                            {
                                "assessment_name": "expected_response",
                                "expectation": {"value": expected_response},
                            }
                        ],
                    },
                    "data": {"spans": [{"type": "LLM", "outputs": "volatile answer"}]},
                },
            }
        ]

    _write_dataset(tmp_path, _rows_with("age 60"), name="a.json")
    _write_dataset(tmp_path, _rows_with("age 65"), name="b.json")

    first = load_dataset("a.json", root=tmp_path)
    second = load_dataset("b.json", root=tmp_path)

    # The authored dataset identity remains trace-free: live and
    # answer-sheet modes discard the recorded trace entirely.
    assert first.digest == second.digest
    assert (
        effective_dataset(first, mode="live").digest
        == effective_dataset(second, mode="live").digest
    )
    # Traces mode replaces authored expectations with these assessments, so
    # changing one changes the evidence that is actually scored.
    assert (
        effective_dataset(first, mode="traces").digest
        != effective_dataset(second, mode="traces").digest
    )


def test_a_different_question_still_changes_the_digest(tmp_path):
    def _rows_with(question):
        return [{"inputs": {"question": question}, "trace": {"data": {"spans": []}}}]

    _write_dataset(tmp_path, _rows_with("question one"), name="a.json")
    _write_dataset(tmp_path, _rows_with("question two"), name="b.json")

    assert (
        load_dataset("a.json", root=tmp_path).digest
        != load_dataset("b.json", root=tmp_path).digest
    )


def test_trace_only_rows_take_their_identity_from_the_request(tmp_path):
    """Rows with no `inputs` must not all digest to the same thing."""

    def _rows_with(question):
        return [
            {
                "trace": {
                    "info": {"request_preview": question},
                    "data": {"spans": [{"type": "LLM"}]},
                }
            }
        ]

    _write_dataset(tmp_path, _rows_with("about pensions"), name="a.json")
    _write_dataset(tmp_path, _rows_with("about pensions"), name="same.json")
    _write_dataset(tmp_path, _rows_with("about something else"), name="b.json")

    same = load_dataset("same.json", root=tmp_path).digest
    assert load_dataset("a.json", root=tmp_path).digest == same
    assert load_dataset("b.json", root=tmp_path).digest != same


def test_trace_only_rows_fall_back_to_the_root_span_inputs(tmp_path):
    def _rows_with(question):
        return [
            {
                "trace": {
                    "data": {
                        "spans": [
                            {"span_id": "root", "inputs": {"question": question}},
                            {"span_id": "child", "parent_span_id": "root"},
                        ]
                    }
                }
            }
        ]

    _write_dataset(tmp_path, _rows_with("first"), name="a.json")
    _write_dataset(tmp_path, _rows_with("second"), name="b.json")

    assert (
        load_dataset("a.json", root=tmp_path).digest
        != load_dataset("b.json", root=tmp_path).digest
    )


def test_span_kinds_come_from_the_spans_not_the_text(tmp_path):
    """An answer about retriever tools does not buy retriever judges.

    Scanning the serialized trace for "retriever" or "tool" matches the
    words wherever they appear, including in the question and the answer.
    The retrieval and tool judges cannot score an LLM-only trace, so the
    calls are spent and then reported as scorer errors that fail the gate.
    """

    rows = [
        {
            "inputs": {"question": "which retriever tool should I use?"},
            "outputs": "Use the retriever tool with tool_calls enabled.",
            "trace": {
                "data": {
                    "spans": [
                        {
                            "type": "LLM",
                            "name": "answer",
                            "outputs": "Use the retriever tool.",
                        }
                    ]
                }
            },
        }
    ]
    _write_dataset(tmp_path, rows)

    shape = load_dataset("golden.json", root=tmp_path).shape

    assert shape.has_traces
    assert not shape.has_retrieval_spans
    assert not shape.has_tool_spans


def test_span_kinds_read_mlflow_attribute_span_types(tmp_path):
    """MLflow stores the span type as a JSON-quoted attribute value."""

    rows = [
        {
            "inputs": {"question": "a"},
            "trace": {
                "data": {
                    "spans": [
                        {
                            "name": "search",
                            "attributes": {"mlflow.spanType": '"RETRIEVER"'},
                        },
                        {
                            "name": "lookup",
                            "attributes": {"mlflow.spanType": '"TOOL"'},
                        },
                    ]
                }
            },
        }
    ]
    _write_dataset(tmp_path, rows)

    shape = load_dataset("golden.json", root=tmp_path).shape

    assert shape.has_retrieval_spans and shape.has_tool_spans


def test_a_sample_records_the_dataset_it_was_drawn_from(tmp_path):
    """Provenance is what keeps a sample from looking like changed data."""

    _write_dataset(tmp_path, _rows(10))
    dataset = load_dataset("golden.json", root=tmp_path)

    sample = smoke_sample(dataset, 4)

    assert sample.shape.row_count == 4
    assert sample.digest != dataset.digest
    assert sample.sampled_from == dataset.digest
    # A sample of a sample still names the original.
    assert smoke_sample(sample, 2).sampled_from == dataset.digest
    # An unsampled dataset claims no parent.
    assert dataset.sampled_from is None
    assert smoke_sample(dataset, 50) is dataset


def test_expectation_rows_record_each_row_alternative(tmp_path):
    """The OR contract is a per-row question, so the shape keeps per-row data."""

    rows = [
        {"inputs": {"question": "a"}, "expectations": {"expected_response": "x"}},
        {"inputs": {"question": "b"}, "expectations": {"expected_facts": ["y"]}},
        {"inputs": {"question": "c"}, "expectations": {"expected_response": ""}},
    ]
    _write_dataset(tmp_path, rows)

    shape = load_dataset("golden.json", root=tmp_path).shape

    assert shape.expectation_rows == (("expected_response",), ("expected_facts",), ())
    # The dataset-wide views are unchanged: nothing is on every row.
    assert shape.expectation_keys == ()
    assert shape.partial_expectation_keys == ("expected_facts", "expected_response")


class NAType:
    """Stands in for pandas' NA, whose truthiness deliberately raises.

    Named to match `type(pd.NA).__name__`: the check is by type name,
    because pandas is MLflow's dependency and not the SDK's, and
    `bool(pd.NA)` raises rather than answering.
    """

    def __bool__(self):
        raise TypeError("boolean value of NA is ambiguous")


def test_dataframe_nulls_read_as_missing(tmp_path):
    """A UC dataset arrives via to_dict("records"), so nulls are NaN.

    Treating NaN as present makes every row of a nullable `trace` column
    look traced, which selects the traces mode — and that mode supplies no
    predict_fn, so those rows would have no answer at all.
    """

    rows = [
        {
            "inputs": {"question": f"q{index}"},
            "expectations": {"expected_response": "a"},
            "trace": float("nan"),
            "outputs": float("nan"),
        }
        for index in range(10)
    ]
    _write_dataset(tmp_path, rows)
    shape = load_dataset("golden.json", root=tmp_path).shape

    assert not shape.has_traces
    assert not shape.partial_traces
    assert not shape.has_outputs


def test_missing_recognises_every_null_sentinel():
    from aai_core.agentkit.datasets import _is_missing, _is_populated

    absent = [None, float("nan"), Decimal("NaN"), Decimal("sNaN"), NAType()]
    try:
        import numpy as np

        absent += [
            np.float32("nan"),
            np.float64("nan"),
            np.datetime64("NaT", "ns"),
        ]
    except ImportError:
        pass
    try:
        import pandas as pd

        # The real sentinels, when pandas is installed: the stub above only
        # proves the name check, not that the name is the right one.
        absent += [pd.NA, pd.NaT, float("nan")]
    except ImportError:
        pass
    for value in absent:
        assert _is_missing(value), value
        assert not _is_populated(value)
    for present in (0, 0.0, False, "x", ["x"], {"a": 1}):
        assert not _is_missing(present), present


def test_a_null_expectation_does_not_satisfy_a_contract(tmp_path):
    rows = [
        {"inputs": {"question": f"q{index}"}, "expectations": {"expected_response": v}}
        for index, v in enumerate(["a"] * 9 + [float("nan")])
    ]
    _write_dataset(tmp_path, rows)

    shape = load_dataset("golden.json", root=tmp_path).shape

    assert shape.expectation_keys == ()
    assert shape.expectation_rows[-1] == ()


def test_serialized_span_outputs_reach_the_token_estimate(tmp_path):
    """`Span.to_dict()` puts them in attributes as a JSON string.

    The chunk count already read that form; the token estimate did not, so
    the retrieved documents contributed nothing to the number a developer
    approves — in exactly the serialized case the estimate exists for.
    """

    from aai_core.agentkit.datasets import _chunk_count, trace_judge_text

    body = "Contributions vest after two years of continuous service. " * 20
    documents = [{"page_content": body}, {"page_content": body}]
    serialized = {
        "info": {"request_preview": "when do contributions vest?"},
        "data": {
            "spans": [
                {
                    "span_id": "1",
                    "attributes": {
                        "mlflow.spanType": '"RETRIEVER"',
                        "mlflow.spanOutputs": json.dumps(documents),
                    },
                }
            ]
        },
    }
    plain = {
        "info": {"request_preview": "when do contributions vest?"},
        "data": {
            "spans": [{"span_id": "1", "type": "RETRIEVER", "outputs": documents}]
        },
    }

    assert body in trace_judge_text(serialized)
    # The two shapes describe the same retrieval, so they agree on both.
    assert trace_judge_text(serialized) == trace_judge_text(plain)
    assert _chunk_count(serialized["data"]["spans"][0]) == 2
    assert _chunk_count(plain["data"]["spans"][0]) == 2


def test_dataset_identity_uses_full_inputs_not_a_truncated_preview(tmp_path):
    """MLflow documents request_preview as truncatable.

    Two different long questions sharing a prefix would then land on one
    digest, and the comparability check would accept a run that scored
    different questions than the baseline did — the one thing the digest
    exists to prevent.
    """

    prefix = "What is the vesting rule for " + "x" * 300

    def _row(tail):
        return {
            "trace": {
                "info": {"request_preview": prefix},
                "data": {
                    "spans": [
                        {
                            "span_id": "1",
                            "type": "LLM",
                            "inputs": {"question": prefix + tail},
                        }
                    ]
                },
            }
        }

    _write_dataset(tmp_path, [_row(" part-time staff?")] * 10, name="a.json")
    _write_dataset(tmp_path, [_row(" seasonal staff?")] * 10, name="b.json")

    first = load_dataset("a.json", root=tmp_path)
    second = load_dataset("b.json", root=tmp_path)

    assert first.digest != second.digest


def test_the_preview_is_still_used_when_there_are_no_spans(tmp_path):
    """A trace without spans still needs an identity."""

    from aai_core.agentkit.datasets import _trace_request

    assert _trace_request({"info": {"request_preview": "the question"}}) == (
        "the question"
    )


def test_trace_readers_ignore_top_level_spans_like_mlflow_from_dict():
    """A top-level span field cannot override the v2/v3 data envelope."""

    from aai_core.agentkit.datasets import _trace_request

    trace = _mlflow_trace(spans=[])
    trace["spans"] = [_mlflow_span(question="ignored question")]

    assert _trace_request(trace) == json.dumps({"question": "q"})


def test_the_token_estimate_reads_the_full_response(tmp_path):
    """A truncated response_preview under-counts the tokens a judge sees."""

    from aai_core.agentkit.datasets import trace_judge_text

    answer = "The full answer that the preview would have cut short. " * 20
    trace = {
        "info": {"response_preview": "The full answer that the previe..."},
        "data": {
            "spans": [
                {"span_id": "1", "type": "LLM", "inputs": {"q": "x"}, "outputs": answer}
            ]
        },
    }

    text = trace_judge_text(trace)

    assert answer in text
    assert len(text) > len(answer)


def test_the_response_preview_is_still_the_fallback():
    from aai_core.agentkit.datasets import trace_judge_text

    text = trace_judge_text(
        {"info": {"request_preview": "q", "response_preview": "the only answer"}}
    )

    assert "the only answer" in text


def test_a_traced_row_still_has_its_expectations_checked(tmp_path):
    """A trace exempts a row from needing inputs, not from being well formed.

    Shape inference reads a malformed expectations value as *absent*, so
    the scorers and thresholds that depend on it are silently dropped
    while the value still travels to MLflow. The check therefore runs
    before the traced-row shortcut, not after it.
    """

    rows = [
        {"trace": {"info": {"trace_id": "t0"}}, "expectations": "yes"},
        {"trace": {"info": {"trace_id": "t1"}}, "expectations": ["yes"]},
    ]
    _write_dataset(tmp_path, rows)

    dataset = load_dataset("golden.json", root=tmp_path)
    failures = validate_dataset(dataset, minimum_rows=1)

    assert any("row 0 expectations must be an object" in f for f in failures)
    assert any("row 1 expectations must be an object" in f for f in failures)


def test_a_traced_row_without_expectations_is_still_valid(tmp_path):
    rows = [{"trace": _mlflow_trace(index, question=f"q{index}")} for index in range(3)]
    _write_dataset(tmp_path, rows)

    dataset = load_dataset("golden.json", root=tmp_path)

    assert validate_dataset(dataset, minimum_rows=1) == []


@pytest.mark.parametrize(
    "trace",
    [
        "not-json",
        {"unrelated": "object"},
        {"info": {"trace_id": "info-only"}},
        {"spans": [{}]},
        {"spans": [{"inputs": {"question": "q"}}]},
    ],
    ids=(
        "invalid-json",
        "unrelated-object",
        "info-only-no-data",
        "top-level-empty-span",
        "top-level-id-less-root-span",
    ),
)
def test_populated_malformed_traces_fail_local_validation(tmp_path, trace):
    rows = [{"inputs": {"question": "q"}, "trace": trace}]
    _write_dataset(tmp_path, rows)

    failures = validate_dataset(
        load_dataset("golden.json", root=tmp_path), minimum_rows=1
    )

    assert failures == [
        "row 0 trace must be decodable and contain a usable request or root span"
    ]


def test_an_identified_root_without_any_request_fails_local_validation(tmp_path):
    _write_dataset(
        tmp_path,
        [
            {
                "trace": _mlflow_trace(
                    question=None,
                    request_preview=False,
                    spans=[_mlflow_span(question=None)],
                )
            }
        ],
    )

    failures = validate_dataset(
        load_dataset("golden.json", root=tmp_path), minimum_rows=1
    )

    assert failures == [
        "row 0 trace must be decodable and contain a usable request or root span"
    ]


def test_an_identified_root_can_use_non_empty_authored_inputs(tmp_path):
    _write_dataset(
        tmp_path,
        [
            {
                "inputs": {"question": "q"},
                "trace": _mlflow_trace(
                    question=None,
                    request_preview=False,
                    spans=[_mlflow_span(question=None)],
                ),
            }
        ],
    )

    dataset = load_dataset("golden.json", root=tmp_path)

    assert validate_dataset(dataset, minimum_rows=1) == []


@pytest.mark.parametrize(
    "trace",
    [
        {
            "info": {"request": {"question": "q"}},
            "spans": [{}],
        },
        {
            "info": {"request_preview": json.dumps({"question": "q"})},
            "data": {"spans": "not-a-list"},
        },
        _mlflow_trace(
            spans=[
                _mlflow_span(
                    parent_span_id=_encoded_id(999, 8),
                )
            ]
        ),
    ],
    ids=("unidentified-root", "non-sequence-spans", "child-only-graph"),
)
def test_trace_info_request_does_not_hide_malformed_spans(tmp_path, trace):
    _write_dataset(tmp_path, [{"trace": trace}])

    failures = validate_dataset(
        load_dataset("golden.json", root=tmp_path), minimum_rows=1
    )

    assert failures == [
        "row 0 trace must be decodable and contain a usable request or root span"
    ]


@pytest.mark.parametrize(
    ("parent_key", "parent_value"),
    [
        ("parent_span_id", None),
        ("parent_span_id", ""),
    ],
)
def test_empty_parent_identifier_is_a_valid_root(tmp_path, parent_key, parent_value):
    root = _mlflow_span(question="q")
    root[parent_key] = parent_value
    _write_dataset(
        tmp_path,
        [
            {
                "trace": _mlflow_trace(
                    question="q",
                    request_preview=False,
                    spans=[root],
                )
            }
        ],
    )

    dataset = load_dataset("golden.json", root=tmp_path)

    assert validate_dataset(dataset, minimum_rows=1) == []


@pytest.mark.parametrize(
    "nested_spans",
    [
        [],
        [{"span_id": "child", "parent_span_id": "missing-root"}],
    ],
    ids=("empty-nested", "child-only-nested"),
)
def test_nested_spans_cannot_borrow_a_top_level_root(tmp_path, nested_spans):
    if nested_spans:
        nested_spans = [_mlflow_span(parent_span_id=_encoded_id(999, 8), question=None)]
    top_level = _mlflow_span(question="q")
    trace = _mlflow_trace(
        question=None,
        request_preview=False,
        spans=nested_spans,
    )
    trace["spans"] = [top_level]
    _write_dataset(
        tmp_path,
        [{"trace": trace}],
    )

    failures = validate_dataset(
        load_dataset("golden.json", root=tmp_path), minimum_rows=1
    )

    assert failures == [
        "row 0 trace must be decodable and contain a usable request or root span"
    ]


def test_consistent_dual_span_layout_uses_the_nested_root(tmp_path):
    root = _mlflow_span(question="q")
    trace = _mlflow_trace(request_preview=False, spans=[root])
    trace["spans"] = [dict(root)]
    _write_dataset(
        tmp_path,
        [{"trace": trace}],
    )
    dataset = load_dataset("golden.json", root=tmp_path)

    assert validate_dataset(dataset, minimum_rows=1) == []
    assert effective_dataset(dataset, mode="live").rows[0]["inputs"] == {
        "question": "q"
    }


def test_null_nested_spans_do_not_fall_back_to_an_ignored_top_level_layout(tmp_path):
    root = _mlflow_span(question="q")
    trace = _mlflow_trace(request_preview=False, spans=[])
    trace["data"]["spans"] = None
    trace["spans"] = [root]
    _write_dataset(
        tmp_path,
        [{"trace": trace}],
    )
    dataset = load_dataset("golden.json", root=tmp_path)

    assert validate_dataset(dataset, minimum_rows=1) == [
        "row 0 trace must be decodable and contain a usable request or root span"
    ]


@pytest.mark.parametrize(
    "trace",
    [
        _mlflow_trace(spans=[]),
        _mlflow_trace(request_preview=False),
        _mlflow_v2_trace(),
        json.dumps(
            _mlflow_trace(
                assessments=[
                    _mlflow_expectation(
                        0,
                        "expected_response",
                        {"value": "a"},
                    )
                ]
            )
        ),
    ],
    ids=(
        "info-request-preview",
        "root-span-request",
        "v2-root-span-request",
        "serialized-with-expectation",
    ),
)
def test_supported_trace_shapes_pass_local_validation(tmp_path, trace):
    rows = [{"inputs": {"question": "q"}, "trace": trace}]
    _write_dataset(tmp_path, rows)
    dataset = load_dataset("golden.json", root=tmp_path)

    assert validate_dataset(dataset, minimum_rows=1) == []
    if isinstance(trace, str):
        assert effective_dataset(dataset, mode="traces").rows[0]["expectations"] == {
            "expected_response": "a"
        }


def test_trace_validation_matches_locked_mlflow_from_dict(tmp_path):
    """Pin the local structural guard to MLflow's real v3 deserializer."""

    pytest.importorskip("mlflow")
    from mlflow.entities import Trace
    from mlflow.exceptions import MlflowException

    valid = _mlflow_trace()
    assert Trace.from_dict(valid).info.trace_id == valid["info"]["trace_id"]
    _write_dataset(tmp_path, [{"inputs": {"question": "q"}, "trace": valid}])
    assert (
        validate_dataset(load_dataset("golden.json", root=tmp_path), minimum_rows=1)
        == []
    )

    info_only = {"info": valid["info"]}
    with pytest.raises(MlflowException, match="Expected keys: 'info' and 'data'"):
        Trace.from_dict(info_only)
    _write_dataset(
        tmp_path,
        [{"inputs": {"question": "q"}, "trace": info_only}],
        "invalid.json",
    )
    assert validate_dataset(
        load_dataset("invalid.json", root=tmp_path), minimum_rows=1
    ) == ["row 0 trace must be decodable and contain a usable request or root span"]


_MALFORMED_TRACE_CASES = (
    "bad-request-time",
    "list-state",
    "mapping-location-type",
    "bad-trace-id",
    "bad-span-id",
    "bad-span-status",
    "bad-event",
    "bad-link",
    "bad-request-id-json",
    "bad-request-id-encoding",
    "mismatched-request-id",
    "v2-bad-timestamp",
    "v2-bad-trace-id",
    "v2-bad-span-status",
    "v2-bad-event",
    "v2-bad-link",
    "v2-bad-request-id-json",
)


@pytest.mark.parametrize("case", _MALFORMED_TRACE_CASES)
def test_dependency_free_trace_contract_fails_closed_and_is_total(tmp_path, case):
    """Every JSON-valid malformed value becomes a governed row failure."""

    from aai_core.agentkit.datasets import _complete_trace_envelope

    trace = _malformed_complete_trace(case)
    assert _complete_trace_envelope(trace) is False
    _write_dataset(tmp_path, [{"inputs": {"question": "q"}, "trace": trace}])

    assert validate_dataset(
        load_dataset("golden.json", root=tmp_path), minimum_rows=1
    ) == ["row 0 trace must be decodable and contain a usable request or root span"]


@pytest.mark.parametrize("trace", [_mlflow_trace(), _mlflow_v2_trace()])
def test_dependency_free_trace_contract_accepts_canonical_v2_and_v3(trace):
    from aai_core.agentkit.datasets import _complete_trace_envelope

    assert _complete_trace_envelope(_trace_with_event_and_link(trace)) is True


@pytest.mark.parametrize(
    "case",
    [
        "bad-request-time",
        "list-state",
        "mapping-location-type",
        "bad-trace-id",
        "bad-span-id",
        "bad-span-status",
        "bad-event",
        "bad-link",
        "bad-request-id-json",
        "v2-bad-trace-id",
        "v2-bad-span-status",
        "v2-bad-event",
        "v2-bad-link",
        "v2-bad-request-id-json",
    ],
)
def test_structural_rejections_match_locked_mlflow(case):
    pytest.importorskip("mlflow")
    from mlflow.entities import Trace
    from mlflow.exceptions import MlflowException

    with pytest.raises((ValueError, TypeError, MlflowException)):
        Trace.from_dict(_malformed_complete_trace(case))


@pytest.mark.parametrize("trace", [_mlflow_trace(), _mlflow_v2_trace()])
def test_canonical_event_and_link_shapes_match_locked_mlflow(trace):
    pytest.importorskip("mlflow")
    from mlflow.entities import Trace

    document = _trace_with_event_and_link(trace)
    assert Trace.from_dict(document).data.spans


@pytest.mark.parametrize("inputs", ["scalar request", ["list request"]])
def test_a_traced_row_rejects_authored_non_object_inputs(tmp_path, inputs):
    """A trace fills in absence; it cannot legalize a malformed row field."""

    _write_dataset(
        tmp_path,
        [{"inputs": inputs, "trace": _mlflow_trace()}],
    )

    failures = validate_dataset(
        load_dataset("golden.json", root=tmp_path), minimum_rows=1
    )

    assert failures == ["row 0 inputs must be an object"]


def test_traced_rows_still_reject_placeholder_content(tmp_path):
    rows = [
        {
            "inputs": {"question": "TODO replace this question"},
            "expectations": {"expected_response": "a real answer"},
            "trace": _mlflow_trace(0),
        },
        {
            "inputs": {"question": "a real question"},
            "expectations": {"expected_response": "TODO write the answer"},
            "trace": _mlflow_trace(1),
        },
        {"trace": _mlflow_trace(2, question="changeme")},
    ]
    _write_dataset(tmp_path, rows)

    failures = validate_dataset(
        load_dataset("golden.json", root=tmp_path), minimum_rows=1
    )

    assert len(failures) == 3
    assert all(
        f"row {index} still contains placeholder text" in failures for index in range(3)
    )


def test_a_null_expectations_column_is_not_malformed(tmp_path):
    """A dataframe null is a missing value, not a bad one."""

    rows = _rows(3)
    rows[0]["expectations"] = float("nan")
    _write_dataset(tmp_path, rows)

    dataset = load_dataset("golden.json", root=tmp_path)
    failures = validate_dataset(dataset, minimum_rows=1)

    assert not any("must be an object" in failure for failure in failures)


def _traced_rows(tmp_path, *, trace=None, **extra):
    rows = []
    for index in range(3):
        row = {
            "inputs": {"question": f"q{index}"},
            "expectations": {"expected_response": f"a{index}"},
            **extra,
        }
        if trace is not None:
            row["trace"] = trace(index)
        rows.append(row)
    _write_dataset(tmp_path, rows)
    return load_dataset("golden.json", root=tmp_path)


def test_a_stored_trace_travels_only_in_traces_mode(tmp_path):
    """The trace is the recorded answer, so it belongs to one mode.

    In live the answer comes from predict_fn and in answer-sheet from the
    sheet; passing a stored trace there hands MLflow a different run's
    answer — and MLflow does not ignore it, it rewrites inputs, outputs
    and expectations from it.
    """

    dataset = _traced_rows(tmp_path, trace=lambda i: {"info": {"trace_id": f"t{i}"}})

    for mode in ("live", "answer-sheet"):
        payload = evaluation_rows(dataset, mode=mode)
        assert all("trace" not in row for row in payload), mode
        assert all(row["inputs"] for row in payload), mode
    traced = evaluation_rows(dataset, mode="traces")
    assert all("trace" in row for row in traced)


def test_a_null_trace_never_reaches_the_payload(tmp_path):
    """The P1: MLflow calls trace.data on every value of a trace column.

    `_extract_request_response_from_trace` does it unconditionally, so one
    NaN raises AttributeError before the agent is ever called.
    """

    dataset = _traced_rows(tmp_path, trace=lambda i: float("nan"))

    for mode in ("live", "answer-sheet", "traces"):
        payload = evaluation_rows(dataset, mode=mode)
        assert all("trace" not in row for row in payload), mode


def test_a_partly_traced_dataset_carries_no_trace_outside_traces_mode(tmp_path):
    """Dropping null keys alone is not enough here.

    Pandas refills the column with NaN for the rows that dropped the key,
    so the mode rule is what closes the partial case.
    """

    rows = [
        {"inputs": {"question": "q0"}, "trace": {"info": {"trace_id": "t0"}}},
        {"inputs": {"question": "q1"}},
    ]
    _write_dataset(tmp_path, rows)
    dataset = load_dataset("golden.json", root=tmp_path)

    assert dataset.shape.partial_traces
    assert all("trace" not in row for row in evaluation_rows(dataset, mode="live"))


def test_a_missing_value_is_an_absent_key(tmp_path):
    rows = [
        {
            "inputs": {"question": f"q{index}"},
            "expectations": float("nan"),
            "outputs": float("nan"),
        }
        for index in range(3)
    ]
    _write_dataset(tmp_path, rows)
    dataset = load_dataset("golden.json", root=tmp_path)

    payload = evaluation_rows(dataset, mode="live")

    assert all(set(row) == {"inputs"} for row in payload)


def test_building_the_payload_does_not_alter_the_dataset(tmp_path):
    dataset = _traced_rows(tmp_path, trace=lambda i: {"info": {"trace_id": f"t{i}"}})
    digest = dataset.digest

    evaluation_rows(dataset, mode="live")

    assert dataset.digest == digest
    assert all("trace" in row for row in dataset.rows)


def test_trace_expectation_overrides_names_the_expectation(tmp_path):
    """MLflow rewrites the whole expectations column from trace assessments.

    Its docstring says it fills the column only when absent; the code has
    no such check.
    """

    def trace(index):
        assessments = (
            [
                {
                    "assessment_name": "expected_response",
                    "expectation": {"value": "from trace"},
                }
            ]
            if index == 1
            else []
        )
        return {"info": {"trace_id": f"t{index}", "assessments": assessments}}

    dataset = _traced_rows(tmp_path, trace=trace)

    assert trace_expectation_overrides(dataset) == ("expected_response",)


def test_no_override_reported_without_assessments(tmp_path):
    dataset = _traced_rows(tmp_path, trace=lambda i: {"info": {"trace_id": f"t{i}"}})

    assert trace_expectation_overrides(dataset) == ()


def test_mlflow_really_does_raise_on_a_null_trace(tmp_path):
    """The reason `evaluation_rows` drops the column, checked not assumed.

    Reaches into MLflow's private `_convert_to_eval_set` on purpose: that
    is the function `evaluate` calls on its data (`evaluation/base.py`),
    before `predict_fn` is touched, and pinning it here is what tells us
    if the behaviour this fix exists for ever changes.
    """

    pytest.importorskip("mlflow")
    from mlflow.genai.evaluation.utils import _convert_to_eval_set

    dataset = _traced_rows(tmp_path, trace=lambda i: float("nan"))
    raw = [dict(row) for row in dataset.rows]

    with pytest.raises(AttributeError):
        _convert_to_eval_set(raw)

    frame = _convert_to_eval_set(evaluation_rows(dataset, mode="live"))
    assert "trace" not in frame.columns


def test_mlflow_really_does_overwrite_curated_expectations(tmp_path):
    """`_extract_expectations_from_trace` documents a guard it does not have.

    Built through MLflow rather than hand-rolled JSON: the serialized
    assessment carries `assessment_name`, `create_time` and
    `last_update_time`, and a hand-written shape both fails to load and
    proves nothing about the real one.
    """

    pytest.importorskip("mlflow")
    import mlflow
    from mlflow.genai.evaluation.utils import _convert_to_eval_set

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("expectation-override")

    @mlflow.trace
    def agent(question):
        return "an answer"

    agent("q0")
    mlflow.flush_trace_async_logging()
    recorded = mlflow.search_traces(return_type="list")[0]
    mlflow.log_expectation(
        trace_id=recorded.info.trace_id,
        name="expected_response",
        value="from the trace",
    )
    trace = mlflow.get_trace(recorded.info.trace_id).to_dict()

    rows = [
        {
            "inputs": {"question": "q0"},
            "expectations": {"expected_response": "from the dataset"},
            "trace": trace,
        }
    ]
    _write_dataset(tmp_path, rows)
    dataset = load_dataset("golden.json", root=tmp_path)

    frame = _convert_to_eval_set([dict(row) for row in dataset.rows])

    assert frame["expectations"][0] == {"expected_response": "from the trace"}
    assert trace_expectation_overrides(dataset) == ("expected_response",)


def test_dropping_the_trace_keeps_the_question(tmp_path):
    """Removing the answer must not remove the question with it.

    A trace-only row has no `inputs` of its own; stripping the trace for a
    live run left MLflow a row it cannot evaluate at all.
    """

    rows = [
        {
            "trace": {
                "data": {
                    "spans": [
                        {
                            "parent_span_id": None,
                            "inputs": {"question": f"q{index}"},
                        }
                    ]
                }
            }
        }
        for index in range(3)
    ]
    _write_dataset(tmp_path, rows)
    dataset = load_dataset("golden.json", root=tmp_path)

    scored = effective_dataset(dataset, mode="live")

    assert rows_missing_inputs(scored) == ()
    assert [row["inputs"] for row in scored.rows] == [
        {"question": "q0"},
        {"question": "q1"},
        {"question": "q2"},
    ]
    assert all("trace" not in row for row in scored.rows)


def test_an_unrecoverable_request_is_reported_not_passed(tmp_path):
    """MLflow needs a mapping; a preview string is not one."""

    rows = [{"trace": {"info": {"request_preview": "just some text"}}}]
    _write_dataset(tmp_path, rows)
    dataset = load_dataset("golden.json", root=tmp_path)

    scored = effective_dataset(dataset, mode="live")

    assert rows_missing_inputs(scored) == (0,)


def test_traces_mode_plans_against_the_expectations_mlflow_will_use(tmp_path):
    """One assessment replaces the whole column, including the empty ones.

    A row whose trace carries no assessment ends up with no expectations
    at all, whatever the dataset wrote — so a scorer chosen from the
    curated field would score nothing and report a vacuous pass.
    """

    def trace(index):
        assessments = (
            [
                {
                    "assessment_name": "expected_response",
                    "expectation": {"value": "from the trace"},
                }
            ]
            if index == 0
            else []
        )
        return {"info": {"trace_id": f"t{index}", "assessments": assessments}}

    dataset = _traced_rows(tmp_path, trace=trace)
    assert dataset.shape.expectation_keys == ("expected_response",)

    scored = effective_dataset(dataset, mode="traces")

    assert scored.rows[0]["expectations"] == {"expected_response": "from the trace"}
    assert scored.rows[1]["expectations"] == {}
    # No longer on every row, so the contract check will exclude the
    # scorers that need it instead of letting them score vacuously.
    assert scored.shape.expectation_keys == ()
    assert scored.shape.partial_expectation_keys == ("expected_response",)


def test_traces_mode_decodes_serialized_expected_facts_for_plan_and_digest(tmp_path):
    """MLflow serializes non-scalar expectation values inside the trace."""

    from aai_core.agentkit.catalog import select_scorers
    from aai_core.agentkit.config import AgentkitConfig

    def _rows_with(facts):
        return [
            {
                "inputs": {"question": "What is the vesting rule?"},
                "expectations": {"expected_facts": ["authored fallback"]},
                "trace": {
                    "info": {
                        "trace_id": "tr-realistic",
                        "assessments": [
                            {
                                "assessment_name": "expected_facts",
                                "expectation": {
                                    # Trace.to_dict leaves the scalar arm
                                    # empty when the serialized arm is used.
                                    "value": None,
                                    "serialized_value": {
                                        "value": json.dumps(facts),
                                        "serialization_format": "JSON",
                                    },
                                },
                            },
                            {
                                "assessment_name": "expected_response",
                                "expectation": {"value": "Direct scalar value."},
                            },
                            {
                                "assessment_name": "guidelines",
                                "expectation": "Direct bare scalar.",
                            },
                        ],
                    },
                    "data": {
                        "spans": [
                            {
                                "span_id": "search",
                                "type": "RETRIEVER",
                                "outputs": [{"page_content": "Policy context."}],
                            }
                        ]
                    },
                },
            }
        ]

    _write_dataset(
        tmp_path, _rows_with(["Two years.", "Continuous service."]), "a.json"
    )
    _write_dataset(tmp_path, _rows_with(["Three years."]), "b.json")
    first = load_dataset("a.json", root=tmp_path)
    second = load_dataset("b.json", root=tmp_path)

    # Trace behaviour remains absent from authored identity.
    assert first.digest == second.digest
    scored = effective_dataset(first, mode="traces")

    assert scored.rows[0]["expectations"] == {
        "expected_facts": ["Two years.", "Continuous service."],
        "expected_response": "Direct scalar value.",
        "guidelines": "Direct bare scalar.",
    }
    assert scored.digest != effective_dataset(second, mode="traces").digest
    plan = select_scorers(
        scored.shape,
        AgentkitConfig(
            version=1,
            agent="agent.py:respond",
            dataset="golden.json",
            scorers={"add": ["retrieval_sufficiency"]},
        ),
        mode="traces",
        judges_enabled=True,
    )
    selected = {spec.name for spec in plan.specs}
    assert {"correctness", "retrieval_sufficiency"} <= selected


@pytest.mark.parametrize(
    "serialized_value",
    [None, {}, {"value": ["not", "json"]}, {"value": "[not valid JSON"}],
)
def test_malformed_serialized_trace_expectation_fails_closed(
    tmp_path, serialized_value
):
    rows = [
        {
            "inputs": {"question": "What is the vesting rule?"},
            "trace": {
                "info": {
                    "assessments": [
                        {
                            "assessment_name": "expected_facts",
                            "expectation": {
                                "value": None,
                                "serialized_value": serialized_value,
                            },
                        }
                    ]
                }
            },
        }
    ]
    _write_dataset(tmp_path, rows)

    with pytest.raises(
        ConfigError,
        match="trace expectation 'expected_facts' is malformed",
    ):
        # Runner computes the effective dataset before scorer selection,
        # budget confirmation, or any provider call.
        effective_dataset(load_dataset("golden.json", root=tmp_path), mode="traces")


def test_curated_expectations_survive_traces_without_assessments(tmp_path):
    dataset = _traced_rows(tmp_path, trace=lambda i: {"info": {"trace_id": f"t{i}"}})

    scored = effective_dataset(dataset, mode="traces")

    assert scored.shape.expectation_keys == ("expected_response",)
    assert scored.rows[0]["expectations"] == {"expected_response": "a0"}


def test_the_effective_dataset_keeps_the_authored_identity(tmp_path):
    dataset = _traced_rows(tmp_path, trace=lambda i: {"info": {"trace_id": f"t{i}"}})

    scored = effective_dataset(dataset, mode="live")

    assert scored.digest == dataset.digest
    assert scored.ref == dataset.ref
    assert scored.sampled_from == dataset.sampled_from


def test_a_live_run_keeps_the_span_kinds_but_not_the_counts(tmp_path):
    """A suite recorded against a retrieving agent is still a retrieval suite.

    Inferring the span kinds from rows the strip just emptied would drop
    the retrieval judges from every live run — a control removed by a fix.
    What must not carry over is the count, which the estimate reads from
    the rows themselves.
    """

    dataset = _traced_rows(
        tmp_path,
        trace=lambda i: {
            "data": {
                "spans": [
                    {
                        "parent_span_id": None,
                        "span_type": "RETRIEVER",
                        "outputs": [{"page_content": "chunk"}],
                    }
                ]
            }
        },
    )
    assert dataset.shape.has_retrieval_spans

    scored = effective_dataset(dataset, mode="live")

    assert scored.shape.has_retrieval_spans
    assert not scored.shape.has_traces
    assert all("trace" not in row for row in scored.rows)


def test_a_trace_body_is_not_scanned_for_placeholders(tmp_path):
    """The placeholder scan is request-side on purpose.

    A production answer may legitimately say "todo" or "changeme", and
    failing a real trace dataset for that would teach developers to stop
    trusting the check. Pinned here so widening the scan to the whole row
    is caught by this suite rather than by someone's evaluation run.
    """

    rows = [
        {
            "inputs": {"question": "what should I do about my pension?"},
            "expectations": {"expected_response": "review your contributions"},
            "trace": _mlflow_trace(
                question="what should I do?",
                output="Add it to your TODO list and changeme later",
            ),
        }
    ]
    _write_dataset(tmp_path, rows)

    failures = validate_dataset(
        load_dataset("golden.json", root=tmp_path), minimum_rows=1
    )

    assert failures == []
