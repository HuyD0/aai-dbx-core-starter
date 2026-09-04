"""Genie as an evaluation target, and the text-to-SQL scorers that make it gradable.

Credential-free like every other test here: target resolution and preflight are
pure string work by design, and the call factory is exercised through a fake
workspace client, so nothing reaches a workspace.

Two things these tests exist to pin:

* **Preflight performs no network operation.** A malformed ``genie:/`` reference
  must be refused locally, before a client is constructed and before a developer
  is asked to approve a run that cannot work.
* **The structured answer keeps working with the scorers that already exist.**
  A Genie turn is a mapping, not a string; ``response`` is a recognised text
  field, so the prose-based scorers read it unchanged while the SQL scorers grade
  the statement the prose is supposed to rest on.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_core.agentkit.catalog import CATALOG, ScorerKind, get_spec
from aai_core.agentkit.errors import TargetResolutionError, UnknownScorerError
from aai_core.agentkit.targets import (
    GENIE_MAX_RESULT_ROWS,
    SUPPORTED_SHAPES,
    TargetKind,
    _genie_answer,
    preflight_target,
    resolve_target,
)
from aai_core.scorers import _output_text, sql_claim_scope, sql_read_only

ROOT = Path(__file__).resolve().parents[1]


# --- Target resolution -----------------------------------------------------


@pytest.mark.parametrize("space_id", ["01ef2b3c4d5e", "my-space_1", "A1"])
def test_a_genie_reference_resolves_to_its_own_kind(space_id):
    target = resolve_target(f"genie:/{space_id}", root=ROOT)
    assert target.kind is TargetKind.GENIE_SPACE
    assert target.normalized == f"genie:/{space_id}"


@pytest.mark.parametrize(
    "reference",
    [
        "genie:/",
        "genie:/   ",
        "genie:/https://host/spaces/abc",
        "genie:/spaces/abc",
        "genie:/a b",
        "genie:/" + "x" * 200,
    ],
)
def test_a_malformed_genie_reference_is_refused_locally(reference):
    """No network, no client: a bad reference fails before anything is built."""

    with pytest.raises(TargetResolutionError):
        resolve_target(reference, root=ROOT)


def test_the_genie_shape_is_documented_for_the_resolution_error():
    """The error message lists supported shapes; Genie has to be among them."""

    kinds = [kind for kind, _example in SUPPORTED_SHAPES]
    assert "Genie space" in kinds
    with pytest.raises(TargetResolutionError) as error:
        resolve_target("not a target at all!", root=ROOT)
    assert "Genie space" in str(error.value)


def test_preflight_accepts_a_valid_space_and_makes_no_call():
    target = resolve_target("genie:/01ef2b3c4d5e", root=ROOT)
    project = SimpleNamespace(config=SimpleNamespace(request_mapping=None))
    preflight_target(target, project=project)  # must not raise, must not connect


def test_a_serving_endpoint_is_not_mistaken_for_a_genie_space():
    assert (
        resolve_target("endpoints:/agent", root=ROOT).kind
        is TargetKind.SERVING_ENDPOINT
    )
    assert (
        resolve_target("some_endpoint", root=ROOT).kind is TargetKind.SERVING_ENDPOINT
    )


# --- Flattening a turn -----------------------------------------------------


def _attachment(*, attachment_id="a1", text=None, sql=None, truncated=None):
    return SimpleNamespace(
        attachment_id=attachment_id,
        text=SimpleNamespace(content=text) if text is not None else None,
        query=(
            SimpleNamespace(
                query=sql,
                query_result_metadata=(
                    SimpleNamespace(is_truncated=truncated)
                    if truncated is not None
                    else None
                ),
            )
            if sql is not None
            else None
        ),
    )


class _FakeClient:
    def __init__(self, rows=None, truncated=False, fail=False):
        outer = self

        class _Genie:
            @staticmethod
            def get_message_attachment_query_result(*_args):
                if outer.fail:
                    raise RuntimeError("result expired")
                return SimpleNamespace(
                    statement_response=SimpleNamespace(
                        result=SimpleNamespace(data_array=outer.rows),
                        manifest=SimpleNamespace(truncated=outer.truncated),
                    )
                )

        self.rows = rows if rows is not None else []
        self.truncated = truncated
        self.fail = fail
        self.genie = _Genie()


def _message(attachments, *, error=None):
    return SimpleNamespace(
        conversation_id="c1", message_id="m1", error=error, attachments=attachments
    )


def test_a_turn_flattens_into_the_structured_answer():
    answer = _genie_answer(
        _FakeClient(rows=[["West", "4248193.00"]]),
        "sp",
        _message(
            [
                _attachment(text="Revenue was $4,248,193.00."),
                _attachment(
                    attachment_id="a2",
                    sql="SELECT region, sum(amount) FROM sales GROUP BY 1",
                    truncated=False,
                ),
            ]
        ),
    )
    assert answer["response"] == "Revenue was $4,248,193.00."
    assert answer["generated_sql"].startswith("SELECT region")
    assert answer["query_result"] == [["West", "4248193.00"]]
    assert answer["truncated"] is False
    assert answer["error"] is None


def test_the_prose_stays_readable_by_the_scorers_that_already_exist():
    """`response` is a recognised text field, so nothing downstream changes."""

    answer = _genie_answer(
        _FakeClient(), "sp", _message([_attachment(text="Revenue was flat.")])
    )
    assert _output_text(answer) == "Revenue was flat."


def test_our_own_row_cap_is_reported_as_truncation():
    answer = _genie_answer(
        _FakeClient(rows=[[str(i)] for i in range(GENIE_MAX_RESULT_ROWS + 50)]),
        "sp",
        _message([_attachment(sql="SELECT n FROM t")]),
    )
    assert len(answer["query_result"]) == GENIE_MAX_RESULT_ROWS
    assert answer["truncated"] is True


def test_an_unavailable_result_keeps_the_prose_and_sql():
    answer = _genie_answer(
        _FakeClient(fail=True),
        "sp",
        _message([_attachment(text="Revenue rose.", sql="SELECT 1")]),
    )
    assert answer["response"] == "Revenue rose."
    assert answer["generated_sql"] == "SELECT 1"
    assert answer["query_result"] == []


def test_a_message_error_is_carried_through():
    answer = _genie_answer(
        _FakeClient(), "sp", _message([], error=SimpleNamespace(error="warehouse down"))
    )
    assert answer["error"] == "warehouse down"


# --- The registry entries --------------------------------------------------


@pytest.mark.parametrize("name", ["sql_read_only", "sql_claim_scope"])
def test_the_sql_scorers_are_versioned_registry_assets(name):
    spec = get_spec(name)
    assert spec.kind is ScorerKind.CODE
    assert spec.judge is None
    assert spec.judge_overhead_tokens == 0  # code scorers cost nothing to run
    assert spec.metric == f"{name}/mean"


def test_read_only_gates_by_default_and_claim_scope_reports():
    """A safety property gates; a prose heuristic reports until calibrated."""

    assert get_spec("sql_read_only").default_threshold == ">=1.0"
    assert get_spec("sql_claim_scope").default_threshold is None


def test_the_sql_scorers_are_not_auto_selected():
    """They are meaningless for an agent that never writes SQL.

    Both declare no expectation and no trace need, which is what keeps them out
    of automatic selection — a project opts in through scorers.add.
    """

    for name in ("sql_read_only", "sql_claim_scope"):
        spec = get_spec(name)
        assert spec.needs_expectations == ()
        assert spec.needs_trace.value == "none"


def test_registry_names_stay_unique():
    names = [spec.name for spec in CATALOG]
    assert len(names) == len(set(names))


def test_an_unknown_scorer_is_still_refused():
    with pytest.raises(UnknownScorerError):
        get_spec("sql_something_invented")


# --- Scorer behaviour ------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT sum(amount) FROM sales",
        "WITH t AS (SELECT 1) SELECT * FROM t",
        "-- DELETE FROM sales\nSELECT 1",
        "DESCRIBE TABLE sales",
        "SELECT update_ts, create_date FROM sales",
    ],
)
def test_read_only_accepts_reads(sql):
    assert sql_read_only({"response": "ok", "generated_sql": sql}, None) == 1.0


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM sales WHERE id = 1",
        "UPDATE sales SET amount = 0",
        "DROP TABLE sales",
        "CREATE TABLE t AS SELECT 1",
        "GRANT SELECT ON sales TO `x`",
        "SELECT 1; DROP TABLE sales",
    ],
)
def test_read_only_refuses_writes_and_multiple_statements(sql):
    assert sql_read_only({"response": "ok", "generated_sql": sql}, None) == 0.0


def test_read_only_reads_a_fenced_block_in_prose():
    """A provenance footer shows its SQL in prose rather than a field."""

    assert sql_read_only("Here:\n```sql\nSELECT 1 FROM t\n```", None) == 1.0
    assert sql_read_only("Here:\n```sql\nUPDATE t SET a = 1\n```", None) == 0.0


def test_read_only_passes_vacuously_without_sql():
    """An agent that did not query is not an agent that queried unsafely."""

    assert sql_read_only("Just prose.", None) == 1.0
    assert sql_read_only(None, None) == 1.0
    assert sql_read_only({"response": "prose only"}, None) == 1.0


def test_a_trend_claim_needs_a_time_predicate():
    assert (
        sql_claim_scope(
            {
                "response": "Revenue grew steadily.",
                "generated_sql": "SELECT sum(a) FROM s",
            },
            None,
        )
        == 0.0
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT date_trunc('month', ts), sum(a) FROM s GROUP BY 1",
        "SELECT sum(a) FROM s WHERE order_date > '2024-01-01'",
        "SELECT sum(a) FROM s WHERE ts >= current_date - INTERVAL 30 DAYS",
    ],
)
def test_a_scoped_trend_claim_passes(sql):
    assert (
        sql_claim_scope(
            {"response": "Revenue grew steadily.", "generated_sql": sql}, None
        )
        == 1.0
    )


def test_a_comparison_is_not_a_trend():
    """Demanding a date filter for "higher than plan" would be noise, not signal."""

    assert (
        sql_claim_scope(
            {
                "response": "West is higher than plan.",
                "generated_sql": "SELECT region FROM s",
            },
            None,
        )
        == 1.0
    )


def test_claim_scope_passes_vacuously_without_a_claim_or_an_answer():
    assert sql_claim_scope({"response": "The regions are West and East."}, None) == 1.0
    assert sql_claim_scope(None, None) == 1.0
