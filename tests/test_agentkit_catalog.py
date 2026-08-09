"""Unit tests for the shared scorer registry: integrity, selection, building."""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from aai_core.agentkit.catalog import (
    CATALOG,
    CODE_SCORER_FUNCTIONS,
    JudgeBinding,
    ScorerKind,
    build_scorer,
    effective_threshold,
    get_spec,
    keyword_coverage,
    refusal_compliance,
    render_plan,
    response_length_ok,
    score_all,
    select_scorers,
)
from aai_core.agentkit.config import AgentkitConfig
from aai_core.agentkit.datasets import DatasetShape
from aai_core.agentkit.errors import ConfigError, UnknownScorerError
from aai_core.scorers import (
    keyword_coverage as shared_keyword_coverage,
)
from aai_core.scorers import (
    refusal_compliance as shared_refusal_compliance,
)
from aai_core.scorers import (
    response_length_ok as shared_response_length_ok,
)


def _shape(
    expectation_keys=("expected_response",),
    has_traces=False,
    row_count=10,
    has_outputs=True,
    partial_expectation_keys=(),
    has_retrieval_spans=None,
    has_tool_spans=None,
    expectation_rows=(),
):
    return DatasetShape(
        row_count=row_count,
        input_keys=("question",),
        has_outputs=has_outputs,
        expectation_keys=tuple(expectation_keys),
        has_traces=has_traces,
        strata_values={},
        partial_expectation_keys=tuple(partial_expectation_keys),
        expectation_rows=tuple(expectation_rows),
        has_retrieval_spans=(
            has_traces if has_retrieval_spans is None else has_retrieval_spans
        ),
        has_tool_spans=has_traces if has_tool_spans is None else has_tool_spans,
    )


def _config(**overrides):
    values = {"version": 1, "agent": "agent.py:respond", "dataset": "golden.json"}
    values.update(overrides)
    return AgentkitConfig(**values)


def _selected_names(plan):
    return {entry.spec.name for entry in plan.entries}


def _excluded(plan, name):
    for item in plan.excluded:
        if item.spec.name == name:
            return item.reason
    return None


def test_catalog_integrity():
    names = [spec.name for spec in CATALOG]
    assert len(names) == len(set(names)), "scorer names must be unique"
    metrics = [spec.metric for spec in CATALOG]
    assert len(metrics) == len(set(metrics)), "metric keys must be unique"
    for spec in CATALOG:
        assert spec.version >= 1
        if spec.kind in {ScorerKind.BUILTIN, ScorerKind.PROMPT_JUDGE}:
            assert spec.judge is not None, spec.name
        if spec.kind is ScorerKind.PROMPT_JUDGE:
            assert spec.judge.prompt_name, spec.name
            assert spec.judge.fallback_instructions, spec.name
        if spec.kind is ScorerKind.CODE:
            assert spec.judge_overhead_tokens == 0, spec.name


def test_get_spec_unknown_name_lists_registry():
    with pytest.raises(UnknownScorerError) as excinfo:
        get_spec("made_up")
    assert "correctness" in str(excinfo.value)


def test_auto_selection_with_expected_response_and_judges():
    plan = select_scorers(_shape(), _config(), mode="live", judges_enabled=True)

    names = _selected_names(plan)
    assert {
        "response_length_ok",
        "keyword_coverage",
        "refusal_compliance",
        "correctness",
        "safety",
        "latency_seconds",
    } <= names
    assert "relevance" not in names  # expectations exist
    assert "equivalence" not in names  # add-only scorer
    assert "fluency" not in names  # add-only scorer


def test_code_only_smoke_excludes_judges_with_reason():
    plan = select_scorers(
        _shape(),
        _config(),
        mode="answer-sheet",
        judges_enabled=False,
        judge_note="smoke runs code scorers only; use --live for judges",
    )

    names = _selected_names(plan)
    assert names == {"response_length_ok", "keyword_coverage", "refusal_compliance"}
    assert "code scorers only" in _excluded(plan, "correctness")
    assert "needs a trace" in _excluded(plan, "latency_seconds")


def test_guidelines_rows_select_expectations_guidelines():
    plan = select_scorers(
        _shape(expectation_keys=("guidelines",)),
        _config(),
        mode="live",
        judges_enabled=True,
    )

    assert "expectations_guidelines" in _selected_names(plan)
    assert "correctness" not in _selected_names(plan)


def test_bare_inputs_select_relevance_with_warning_reason():
    plan = select_scorers(
        _shape(expectation_keys=()), _config(), mode="live", judges_enabled=True
    )

    names = _selected_names(plan)
    assert "relevance" in names
    assert "keyword_coverage" not in names
    entry = next(e for e in plan.entries if e.spec.name == "relevance")
    assert "no expectations" in entry.reason


def test_trace_rows_select_trace_dependent_scorers():
    plan = select_scorers(
        _shape(has_traces=True),
        _config(),
        mode="traces",
        judges_enabled=True,
    )

    names = _selected_names(plan)
    assert "retrieval_groundedness" in names
    assert "tool_call_correctness" in names


def test_requested_trace_scorer_on_plain_rows_fails_before_any_spend():
    with pytest.raises(ConfigError) as excinfo:
        select_scorers(
            _shape(),
            _config(scorers={"add": ["retrieval_groundedness"]}),
            mode="traces",
            judges_enabled=True,
        )

    assert "scorers.add requests 'retrieval_groundedness'" in str(excinfo.value)
    assert "RETRIEVER spans" in str(excinfo.value)


def test_trace_scorer_added_in_live_mode_is_conditional():
    plan = select_scorers(
        _shape(),
        _config(scorers={"add": ["retrieval_groundedness"]}),
        mode="live",
        judges_enabled=True,
    )

    entry = next(e for e in plan.entries if e.spec.name == "retrieval_groundedness")
    assert "conditional" in entry.reason


def test_add_violating_expectation_contract_fails_before_any_spend():
    with pytest.raises(ConfigError) as excinfo:
        select_scorers(
            _shape(expectation_keys=()),
            _config(scorers={"add": ["equivalence"]}),
            mode="live",
            judges_enabled=True,
        )
    assert "expectations.expected_response" in str(excinfo.value)


def test_add_with_partial_single_field_coverage_fails_before_any_spend():
    with pytest.raises(ConfigError) as excinfo:
        select_scorers(
            _shape(
                expectation_keys=(),
                partial_expectation_keys=("expected_response",),
                expectation_rows=(("expected_response",), (), ("expected_response",)),
            ),
            _config(scorers={"add": ["equivalence"]}),
            mode="answer-sheet",
            judges_enabled=True,
        )

    message = str(excinfo.value)
    assert "scorers.add requests 'equivalence'" in message
    assert "only some rows have expectations.expected_response" in message
    assert "every applicable row" in message


def test_add_with_partial_or_contract_coverage_fails_before_any_spend():
    with pytest.raises(ConfigError) as excinfo:
        select_scorers(
            _shape(
                expectation_keys=(),
                partial_expectation_keys=("expected_facts", "expected_response"),
                expectation_rows=(("expected_response",), (), ("expected_facts",)),
            ),
            _config(scorers={"add": ["correctness"]}),
            mode="answer-sheet",
            judges_enabled=True,
        )

    message = str(excinfo.value)
    assert "scorers.add requests 'correctness'" in message
    assert "expected_facts" in message
    assert "expected_response" in message
    assert "only some rows" in message


def test_global_judge_disable_still_wins_over_scorers_add():
    plan = select_scorers(
        _shape(expectation_keys=()),
        _config(scorers={"add": ["correctness"]}),
        mode="answer-sheet",
        judges_enabled=False,
        judge_note="smoke runs code scorers only",
    )

    assert "correctness" not in _selected_names(plan)
    assert _excluded(plan, "correctness") == "smoke runs code scorers only"


def test_same_scorer_cannot_be_both_added_and_removed():
    with pytest.raises(ConfigError, match="both scorers.add and scorers.remove"):
        select_scorers(
            _shape(),
            _config(
                scorers={
                    "add": ["keyword_coverage"],
                    "remove": ["keyword_coverage"],
                }
            ),
            mode="answer-sheet",
            judges_enabled=False,
        )


def test_remove_wins_over_auto_selection():
    plan = select_scorers(
        _shape(),
        _config(scorers={"remove": ["response_length_ok"]}),
        mode="live",
        judges_enabled=True,
    )

    assert "response_length_ok" not in _selected_names(plan)
    assert _excluded(plan, "response_length_ok") == "removed by scorers.remove"


def test_config_guidelines_select_guidelines_scorer():
    plan = select_scorers(
        _shape(),
        _config(scorers={"guidelines": ["Always cite the policy number."]}),
        mode="live",
        judges_enabled=True,
    )

    assert "guidelines" in _selected_names(plan)


def test_effective_threshold_precedence():
    spec = get_spec("correctness")

    by_name = _config(thresholds={"correctness": ">=0.9"})
    by_metric = _config(thresholds={"correctness/mean": ">=0.8"})
    default = _config()

    assert effective_threshold(spec, by_name) == ">=0.9"
    assert effective_threshold(spec, by_metric) == ">=0.8"
    assert effective_threshold(spec, default) == ">=0.7"
    assert effective_threshold(get_spec("fluency"), default) is None


def test_plan_render_names_versions_thresholds_and_exclusions():
    plan = select_scorers(
        _shape(),
        _config(),
        mode="answer-sheet",
        judges_enabled=False,
        judge_note="smoke runs code scorers only",
    )

    text = render_plan(plan, judge_model_uri="endpoints:/judge")
    assert "keyword_coverage" in text
    assert ">=0.6" in text
    assert "excluded: correctness" in text
    assert "note: smoke runs code scorers only" in text


def test_scorer_versions_tag_is_sorted_and_compact():
    plan = select_scorers(
        _shape(), _config(), mode="answer-sheet", judges_enabled=False
    )

    tag = plan.scorer_versions_tag()
    assert tag == "keyword_coverage=2,refusal_compliance=2,response_length_ok=2"


def test_shared_code_scorer_specs_publish_the_new_semantics_as_v2():
    assert {
        name: get_spec(name).version
        for name in (
            "keyword_coverage",
            "refusal_compliance",
            "response_length_ok",
        )
    } == {
        "keyword_coverage": 2,
        "refusal_compliance": 2,
        "response_length_ok": 2,
    }


@pytest.mark.parametrize(
    ("outputs", "expectations", "expected"),
    [
        (
            "The capital of France is Paris",
            {"expected_response": "Paris is the capital of France"},
            1.0,
        ),
        ("Nothing relevant here", {"expected_response": ""}, 0.0),
        ("Nothing relevant at all", {"expected_response": "quarterly report"}, 0.0),
    ],
)
def test_keyword_coverage_values(outputs, expectations, expected):
    assert keyword_coverage(outputs, expectations) == expected


def test_refusal_compliance_values():
    refusal_expected = {"expected_response": "The agent should refuse politely."}
    assert refusal_compliance("I cannot help with that", refusal_expected) == 1.0
    assert refusal_compliance("Sure, here it is", refusal_expected) == 0.0
    normal = {"expected_response": "The report is due Friday."}
    assert refusal_compliance("It is due Friday", normal) == 1.0
    assert refusal_compliance("I cannot help with that", normal) == 0.0


def test_answer_length_values():
    assert response_length_ok("", {}) == 0.0
    assert response_length_ok("ok", {}) == 1.0
    assert response_length_ok("x" * 2001, {}) == 0.0


def test_agentkit_uses_the_canonical_shared_code_scorers():
    assert keyword_coverage is shared_keyword_coverage
    assert refusal_compliance is shared_refusal_compliance
    assert response_length_ok is shared_response_length_ok


def test_score_all_covers_every_row_level_code_scorer():
    scores = score_all("answer", {"expected_response": "answer"})
    assert set(scores) == set(CODE_SCORER_FUNCTIONS)


def test_code_scorers_never_turn_a_missing_output_into_text():
    missing = [
        None,
        Decimal("NaN"),
        Decimal("1"),
        0,
        False,
        {},
        {"status": "ok"},
        [],
        (),
        SimpleNamespace(status="ok"),
    ]
    try:
        import numpy as np

        missing.extend((np.float32("nan"), np.datetime64("NaT", "ns")))
    except ImportError:
        pass
    try:
        import pandas as pd

        missing.extend((pd.NA, pd.NaT))
    except ImportError:
        pass
    for value in missing:
        with pytest.raises(ConfigError, match="no output to score"):
            score_all(value, {"expected_response": "None"})


def test_code_scorers_extract_text_from_provider_shapes():
    outputs = {"choices": [{"message": {"content": "The answer is Paris."}}]}

    assert score_all(outputs, {"expected_response": "Paris"}) == {
        "keyword_coverage": 1.0,
        "refusal_compliance": 1.0,
        "response_length_ok": 1.0,
    }


def test_blank_recorded_output_reaches_the_gate_as_zero_scores():
    assert score_all("", {"expected_response": "Paris"}) == {
        "keyword_coverage": 0.0,
        "refusal_compliance": 0.0,
        "response_length_ok": 0.0,
    }


def _fake_mlflow(make_judge=None):
    def scorer_decorator(name=None):
        def wrap(function):
            return SimpleNamespace(name=name, function=function)

        return wrap

    def builtin(class_name):
        # A class, not a factory: the toolkit subclasses retrieval scorers
        # so a row with nothing retrieved is skipped instead of raising.
        class _Fake:
            def __init__(self, **kwargs):
                self.class_name = class_name
                self.kwargs = kwargs

            def __call__(self, *, trace=None):
                return []

        _Fake.__name__ = class_name
        return _Fake

    scorers = SimpleNamespace(
        scorer=scorer_decorator,
        Correctness=builtin("Correctness"),
        Equivalence=builtin("Equivalence"),
        RelevanceToQuery=builtin("RelevanceToQuery"),
        Safety=builtin("Safety"),
        Fluency=builtin("Fluency"),
        Completeness=builtin("Completeness"),
        ExpectationsGuidelines=builtin("ExpectationsGuidelines"),
        Guidelines=builtin("Guidelines"),
        PIIDetection=builtin("PIIDetection"),
        RetrievalGroundedness=builtin("RetrievalGroundedness"),
        RetrievalRelevance=builtin("RetrievalRelevance"),
        RetrievalSufficiency=builtin("RetrievalSufficiency"),
        ToolCallCorrectness=builtin("ToolCallCorrectness"),
        ToolCallEfficiency=builtin("ToolCallEfficiency"),
    )
    genai = SimpleNamespace(scorers=scorers)
    if make_judge is not None:
        genai.make_judge = make_judge
    return SimpleNamespace(genai=genai)


def test_build_code_scorer_wraps_the_pure_function():
    built = build_scorer(get_spec("keyword_coverage"), mlflow_module=_fake_mlflow())

    assert built.name == "keyword_coverage"
    value = built.function(outputs="Paris", expectations={"expected_response": "Paris"})
    assert value == 1.0


def test_native_code_scorer_fails_closed_on_a_missing_output():
    built = build_scorer(get_spec("response_length_ok"), mlflow_module=_fake_mlflow())

    with pytest.raises(ConfigError, match="no output to score"):
        built.function(outputs=None, expectations={})


def test_latency_scorer_fails_when_trace_duration_is_missing():
    built = build_scorer(get_spec("latency_seconds"), mlflow_module=_fake_mlflow())

    with pytest.raises(ConfigError, match="execution_duration"):
        built.function(trace=SimpleNamespace(info=SimpleNamespace()))
    assert (
        built.function(
            trace=SimpleNamespace(info=SimpleNamespace(execution_duration=1250))
        )
        == 1.25
    )


def test_every_judge_routes_through_the_governed_endpoint():
    fake = _fake_mlflow()

    for name in ("correctness", "safety", "relevance", "equivalence"):
        built = build_scorer(
            get_spec(name), judge_model_uri="endpoints:/judge", mlflow_module=fake
        )
        assert built.kwargs.get("model") == "endpoints:/judge", name


def test_a_non_overridable_binding_keeps_the_platform_default_judge():
    spec = get_spec("safety").model_copy(
        update={"judge": JudgeBinding(overridable=False)}
    )

    built = build_scorer(
        spec, judge_model_uri="endpoints:/judge", mlflow_module=_fake_mlflow()
    )

    assert built.kwargs == {}


def test_build_guidelines_scorer_requires_project_text():
    fake = _fake_mlflow()

    with pytest.raises(ConfigError):
        build_scorer(get_spec("guidelines"), mlflow_module=fake)

    built = build_scorer(
        get_spec("guidelines"),
        guidelines=("Always cite the policy number.",),
        judge_model_uri="endpoints:/judge",
        mlflow_module=fake,
    )
    assert built.kwargs["name"] == "guidelines"
    assert built.kwargs["guidelines"] == ["Always cite the policy number."]


def test_build_prompt_judge_uses_make_judge_with_fallback_instructions():
    captured = {}

    def make_judge(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(name=kwargs["name"])

    build_scorer(
        get_spec("pension_domain_policy"),
        judge_model_uri="endpoints:/judge",
        mlflow_module=_fake_mlflow(make_judge=make_judge),
    )

    assert captured["name"] == "pension_domain_policy"
    assert "{{ inputs }}" in captured["instructions"]
    assert "official support channels" in captured["instructions"]
    assert captured["model"] == "endpoints:/judge"


def test_build_prompt_judge_prefers_registry_prompt_text():
    captured = {}

    def make_judge(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(name=kwargs["name"])

    def prompt_loader(name, alias):
        assert name == "agentkit_judge_domain_policy"
        assert alias == "production"
        return SimpleNamespace(template="Registry rules {{ inputs }} {{ outputs }}")

    build_scorer(
        get_spec("pension_domain_policy"),
        prompt_loader=prompt_loader,
        mlflow_module=_fake_mlflow(make_judge=make_judge),
    )

    assert captured["instructions"].startswith("Registry rules")


def test_build_prompt_judge_falls_back_to_guidelines_without_make_judge():
    built = build_scorer(
        get_spec("pension_domain_policy"),
        judge_model_uri="endpoints:/judge",
        mlflow_module=_fake_mlflow(),
    )

    assert built.class_name == "Guidelines"
    assert built.kwargs["name"] == "pension_domain_policy"
    assert len(built.kwargs["guidelines"]) == 3


def test_retrieval_traces_do_not_select_tool_scorers():
    """A retrieval trace must not buy tool-judge calls it cannot satisfy."""

    plan = select_scorers(
        _shape(has_traces=True, has_retrieval_spans=True, has_tool_spans=False),
        _config(),
        mode="traces",
        judges_enabled=True,
    )

    names = _selected_names(plan)
    assert "retrieval_groundedness" in names
    assert "tool_call_correctness" not in names
    assert "no tool-call spans" in _excluded(plan, "tool_call_correctness")


def test_tool_traces_do_not_select_retrieval_scorers():
    plan = select_scorers(
        _shape(has_traces=True, has_retrieval_spans=False, has_tool_spans=True),
        _config(),
        mode="traces",
        judges_enabled=True,
    )

    names = _selected_names(plan)
    assert "tool_call_correctness" in names
    assert "retrieval_groundedness" not in names


def test_partially_present_expectations_are_not_scored_dataset_wide():
    """A field only some rows carry cannot be scored as if all rows had it.

    keyword_coverage returns a vacuous 1.0 when there is nothing to check,
    so running it on rows without an expected response would inflate the
    aggregate the gate reads.
    """

    plan = select_scorers(
        _shape(expectation_keys=(), partial_expectation_keys=("expected_response",)),
        _config(),
        mode="answer-sheet",
        judges_enabled=False,
    )

    names = _selected_names(plan)
    assert "keyword_coverage" not in names
    reason = _excluded(plan, "keyword_coverage")
    assert "only some rows" in reason


def test_live_plan_names_the_retrieval_scorers_it_cannot_decide():
    """A live RAG run must not silently drop groundedness.

    Auto-selection reads the dataset, and a live run's traces do not exist
    until the agent produces them, so plain question rows cannot show that
    the agent retrieves. Dropping the scorer without a word would let a RAG
    comparison pass while its default groundedness threshold never ran.
    """

    plan = select_scorers(
        _shape(has_traces=False), _config(), mode="live", judges_enabled=True
    )

    assert "retrieval_groundedness" not in _selected_names(plan)
    reason = _excluded(plan, "retrieval_groundedness")
    assert "whether this agent retrieves" in reason
    assert "name them in scorers.add" in reason
    assert "calls tools" in _excluded(plan, "tool_call_correctness")
    # One line per reason, naming every scorer it covers.
    rendered = [
        line for line in render_plan(plan).splitlines() if line.startswith("excluded:")
    ]
    retrieval = next(line for line in rendered if "retrieval_groundedness" in line)
    assert "retrieval_relevance" in retrieval
    assert "retrieval_sufficiency" in retrieval
    assert sum("whether this agent retrieves" in line for line in rendered) == 1


def test_answer_sheet_plan_does_not_suggest_trace_scorers():
    plan = select_scorers(
        _shape(has_traces=False), _config(), mode="answer-sheet", judges_enabled=False
    )

    assert "whether this agent retrieves" not in render_plan(plan)


def test_traces_mode_scores_the_spans_the_rows_carry():
    shape = _shape(has_traces=True, has_retrieval_spans=True, has_tool_spans=False)

    plan = select_scorers(shape, _config(), mode="traces", judges_enabled=True)

    names = _selected_names(plan)
    assert "retrieval_groundedness" in names
    assert "latency_seconds" in names
    assert "tool_call_correctness" not in names
    assert "carry no tool-call spans" in _excluded(plan, "tool_call_correctness")


def test_a_plan_with_no_scorers_is_refused():
    """ "Evaluated nothing" must never be a passing verdict.

    An empty plan produces no metrics, an empty policy has nothing to fail
    on, and the gate would pass a run that scored not one row.
    """

    with pytest.raises(ConfigError) as excinfo:
        select_scorers(
            _shape(expectation_keys=()),
            _config(scorers={"remove": ["response_length_ok", "relevance"]}),
            mode="answer-sheet",
            judges_enabled=False,
            judge_note="smoke runs code scorers only",
        )

    message = str(excinfo.value)
    assert "would evaluate nothing" in message
    # It names what was dropped, so the developer can act on it.
    assert "response_length_ok" in message
    assert "scorers.remove" in message


def test_rows_may_satisfy_an_or_contract_with_different_fields():
    """`correctness` reads expected_response OR expected_facts, per row.

    A dataset whose rows are split between the two alternatives satisfies
    that contract, but the keys present on *every* row intersect to
    nothing — so asking the intersection would drop the scorer and its
    default >=0.7 threshold from a dataset that scores perfectly well.
    """

    shape = _shape(
        expectation_keys=(),
        partial_expectation_keys=("expected_facts", "expected_response"),
        expectation_rows=(
            ("expected_response",),
            ("expected_facts",),
            ("expected_response",),
        ),
    )

    plan = select_scorers(shape, _config(), mode="answer-sheet", judges_enabled=True)

    names = _selected_names(plan)
    assert "correctness" in names
    # A scorer needing one specific field is still blocked by the split.
    assert "keyword_coverage" not in names
    assert "one of them" in _excluded(plan, "keyword_coverage")


def test_a_row_satisfying_neither_alternative_still_blocks():
    shape = _shape(
        expectation_keys=(),
        partial_expectation_keys=("expected_facts", "expected_response"),
        expectation_rows=(("expected_response",), (), ("expected_facts",)),
    )

    plan = select_scorers(shape, _config(), mode="answer-sheet", judges_enabled=True)

    assert "correctness" not in _selected_names(plan)
    assert "only some rows have" in _excluded(plan, "correctness")


def test_retrieval_scorers_skip_rows_with_nothing_retrieved():
    """MLflow raises there; the toolkit turns that into a skipped row.

    A conditionally retrieving agent would otherwise error on every
    non-retrieval row, and scorer errors fail the gate — so it could
    never pass.
    """

    from aai_core.agentkit.catalog import build_scorer, get_spec

    mlflow = _fake_mlflow()

    class _Raising:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, *, trace=None):
            if trace == "no-retrieval":
                raise RuntimeError(
                    "No retrieval context found in the trace. The "
                    "RetrievalGroundedness scorer requires the trace to "
                    "contain at least one span with type 'RETRIEVER'."
                )
            if trace == "boom":
                raise RuntimeError("the judge endpoint refused the request")
            return ["scored"]

    mlflow.genai.scorers.RetrievalGroundedness = _Raising
    built = build_scorer(
        get_spec("retrieval_groundedness"),
        judge_model_uri="endpoints:/judge",
        mlflow_module=mlflow,
    )

    assert built(trace="has-retrieval") == ["scored"]
    assert built(trace="no-retrieval") == []
    # Any other failure is still a failure.
    with pytest.raises(RuntimeError, match="judge endpoint"):
        built(trace="boom")


def test_the_plan_says_retrieval_coverage_varies_per_row():
    plan = select_scorers(
        _shape(has_traces=False, has_retrieval_spans=False, has_tool_spans=False),
        _config(scorers={"add": ["retrieval_groundedness"]}),
        mode="live",
        judges_enabled=True,
    )

    reason = next(
        entry.reason
        for entry in plan.entries
        if entry.spec.name == "retrieval_groundedness"
    )
    assert "vary per row" in reason
    assert "skip" not in reason


def test_a_split_expectation_suite_does_not_also_buy_relevance():
    """Every row has expectations; none of them is on every row.

    Reading the empty intersection as "no expectations" would add the
    thresholded relevance judge on top of correctness — judge calls
    nobody asked for, on a gate that fails closed.
    """

    plan = select_scorers(
        _shape(
            expectation_keys=(),
            partial_expectation_keys=("expected_facts", "expected_response"),
            expectation_rows=(("expected_response",), ("expected_facts",)),
        ),
        _config(),
        mode="answer-sheet",
        judges_enabled=True,
    )

    names = _selected_names(plan)
    assert "correctness" in names
    assert "relevance" not in names


def test_a_dataset_with_no_expectations_still_selects_relevance():
    plan = select_scorers(
        _shape(expectation_keys=(), partial_expectation_keys=()),
        _config(),
        mode="answer-sheet",
        judges_enabled=True,
    )

    assert "relevance" in _selected_names(plan)


def test_answer_sheet_mode_refuses_trace_scorers_and_names_the_mode():
    """The rows carry traces, but an answer-sheet run does not pass them.

    Selecting these on the strength of the dataset's stored spans would
    pair one run's recorded answers with another run's retrieval — and
    since the payload no longer carries the trace, they would have nothing
    to read either way.
    """

    plan = select_scorers(
        _shape(has_traces=True, has_retrieval_spans=True, has_tool_spans=True),
        _config(),
        mode="answer-sheet",
        judges_enabled=True,
    )

    names = _selected_names(plan)
    assert "retrieval_groundedness" not in names
    assert "tool_call_correctness" not in names
    for scorer in ("retrieval_groundedness", "tool_call_correctness"):
        reason = _excluded(plan, scorer)
        assert reason is not None
        assert "--mode traces" in reason


def test_a_requested_trace_scorer_still_refuses_in_answer_sheet_mode():
    with pytest.raises(ConfigError) as excinfo:
        select_scorers(
            _shape(has_traces=True, has_retrieval_spans=True),
            _config(scorers={"add": ["retrieval_groundedness"]}),
            mode="answer-sheet",
            judges_enabled=True,
        )

    assert "--mode traces" in str(excinfo.value)


def test_live_mode_still_selects_trace_scorers():
    """Fresh traces come from the agent, so live is untouched."""

    plan = select_scorers(
        _shape(),
        _config(scorers={"add": ["retrieval_groundedness"]}),
        mode="live",
        judges_enabled=True,
    )

    assert "retrieval_groundedness" in _selected_names(plan)


def test_a_trace_free_dataset_is_not_told_to_use_mode_traces():
    """Advice a dataset cannot take is worse than no advice.

    The answer-sheet reason names `--mode traces`, which only helps rows
    that actually carry traces; trace-free rows keep the message about the
    spans they lack.
    """

    with pytest.raises(ConfigError) as excinfo:
        select_scorers(
            _shape(),
            _config(scorers={"add": ["retrieval_groundedness"]}),
            mode="answer-sheet",
            judges_enabled=True,
        )

    reason = str(excinfo.value)
    assert "--mode traces" not in reason
    assert "RETRIEVER spans" in reason
