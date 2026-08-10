"""Tier-1 gate: the whole semantic contract, verified with zero credentials.

Pull-request CI proves here that (1) the semantic model validates, (2) the
knowledge docs have not drifted from it — the anti-staleness check that
keeps definitions, docs, and data model in one reviewed diff, (3) every
golden case compiles through the semantic layer and reproduces its pinned
expected value on the snapshot executor, and (4) the recorded answer sheet
passes the deterministic provenance scorers at gate thresholds. Judges and
the live warehouse run only in evals/evaluate.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from aai_core.evaluation import GatePolicy, MetricRule, apply_gate

ROOT = Path(__file__).resolve().parents[1]
MIN_GOLDEN_CASES = 12
PLACEHOLDER_MARKERS = ("replace this", "replace-with", "todo", "changeme")
REQUIRED_CATEGORIES = {
    "metric_lookup",
    "grouped_metric",
    "encoding_trap",
    "ambiguity_clarify",
    "freshness",
    "raw_fallback",
    "out_of_scope_refusal",
}
VALID_TIERS = {"semantic_layer", "curated_reference", "raw_table"}


def load_gate() -> tuple[list[MetricRule], set[str], set[str], set[str]]:
    config = json.loads((ROOT / "evals" / "gate_config.json").read_text("utf-8"))
    thresholds = [MetricRule(**threshold) for threshold in config["thresholds"]]
    return (
        thresholds,
        set(config["code_metrics"]),
        set(config["judge_metrics"]),
        set(config["report_only_judge_metrics"]),
    )


def main() -> int:  # noqa: C901 - linear, independent contract assertions
    sys.path.insert(0, str(ROOT / "src"))
    from app.knowledge import KnowledgeRouter
    from app.scorers import score_all
    from app.semantics.compiler import (
        QueryFilter,
        SemanticQuery,
        TimeGrain,
        compile_query,
    )
    from app.semantics.executor import FakeWarehouseExecutor, ensure_read_only
    from app.semantics.models import load_semantic_model

    failures: list[str] = []

    # 1. The semantic contract validates strictly.
    try:
        model = load_semantic_model(ROOT / "semantics" / "semantic_model.yml")
    except Exception as error:  # noqa: BLE001 - report every contract break
        print(f"FAIL: semantic model does not validate: {error}", file=sys.stderr)
        return 1

    # 2. Knowledge docs must not drift from the semantic model.
    router = KnowledgeRouter(ROOT / "knowledge")
    for issue in router.cross_reference_issues(model):
        failures.append(f"knowledge drift: {issue}")

    # 3. Gate category invariants (identical discipline to the other
    # templates: every gated metric categorized exactly once).
    thresholds, code_metrics, judge_metrics, report_only_judge_metrics = load_gate()
    gated = {threshold.metric for threshold in thresholds}
    gated_categories = code_metrics | judge_metrics
    all_categories = [code_metrics, judge_metrics, report_only_judge_metrics]
    overlap = set()
    for index, category in enumerate(all_categories):
        for other in all_categories[index + 1 :]:
            overlap.update(category & other)
    if overlap:
        failures.append(
            "metrics must appear in exactly one code, judge, or report-only "
            f"category: {sorted(overlap)}"
        )
    for metric in gated_categories:
        if metric not in gated:
            failures.append(f"categorized metric {metric} is not gated")
    for metric in gated - gated_categories:
        failures.append(f"gated metric {metric} is not categorized")
    for metric in report_only_judge_metrics & gated:
        failures.append(f"report-only judge metric {metric} must not be gated")

    # 4 + 5. Golden cases: shape, coverage, and snapshot-pinned truth. Every
    # expected_query must compile read-only against the model and reproduce
    # its expected_value on the offline snapshot executor.
    golden = json.loads(
        (ROOT / "evals" / "data" / "golden_cases.json").read_text("utf-8")
    )
    if len(golden) < MIN_GOLDEN_CASES:
        failures.append(
            f"golden_cases.json has {len(golden)} cases; keep at least "
            f"{MIN_GOLDEN_CASES} (grow toward dozens per domain as corrections "
            "are harvested)"
        )
    executor = FakeWarehouseExecutor(ROOT / "evals" / "data" / "seed_data.json")
    seen_categories: set[str] = set()
    for index, case in enumerate(golden):
        question = case.get("inputs", {}).get("question", "")
        expectations = case.get("expectations", {})
        expected = expectations.get("expected_response", "")
        if not question.strip() or not str(expected).strip():
            failures.append(f"golden case {index} is missing question/expectation")
            continue
        if any(m in f"{question} {expected}".lower() for m in PLACEHOLDER_MARKERS):
            failures.append(f"golden case {index} still contains placeholder text")
        tier = expectations.get("expected_tier")
        if tier not in VALID_TIERS:
            failures.append(f"golden case {index} has invalid expected_tier {tier!r}")
        seen_categories.add(str(expectations.get("category", "")))
        plan = expectations.get("expected_query")
        if plan is None:
            continue
        try:
            query = SemanticQuery(
                metrics=tuple(plan.get("metrics", ())),
                dimensions=tuple(plan.get("dimensions", ())),
                filters=tuple(
                    QueryFilter(
                        dimension=item["dimension"],
                        value=item["value"],
                        grain=(TimeGrain(item["grain"]) if item.get("grain") else None),
                    )
                    for item in plan.get("filters", ())
                ),
                time_dimension=plan.get("time_dimension"),
                time_grain=(
                    TimeGrain(plan["time_grain"]) if plan.get("time_grain") else None
                ),
            )
            compiled = compile_query(model, query)
            ensure_read_only(compiled.sql)
            for source in compiled.sources:
                if source.count(".") != 2:
                    failures.append(
                        f"golden case {index} compiled a non-fully-qualified "
                        f"source {source!r}"
                    )
            result = executor.run_plan(model, query)
        except Exception as error:  # noqa: BLE001 - every case must compile
            failures.append(f"golden case {index} expected_query fails: {error}")
            continue
        expected_value = expectations.get("expected_value")
        if expected_value is not None:
            observed = result.scalar
            if observed is None:
                failures.append(
                    f"golden case {index} expected a scalar value but the plan "
                    f"returned {len(result.rows)} row(s)"
                )
            elif abs(float(observed) - float(expected_value)) > 1e-6:
                failures.append(
                    f"golden case {index} expected_value {expected_value} does "
                    f"not match the snapshot result {observed}; seed data and "
                    "answers must change together"
                )
    missing_categories = REQUIRED_CATEGORIES - seen_categories
    if missing_categories:
        failures.append(
            f"golden cases must cover categories {sorted(missing_categories)}"
        )

    # 6. The answer sheet covers every golden question.
    sheet = json.loads(
        (ROOT / "evals" / "data" / "answer_sheet.json").read_text("utf-8")
    )
    answers = {record["question"]: record["answer"] for record in sheet}
    missing = [
        case["inputs"]["question"]
        for case in golden
        if case["inputs"]["question"] not in answers
    ]
    if missing:
        failures.append(f"answer_sheet.json lacks answers for: {missing}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    # 7. Deterministic provenance scoring over the recorded answer sheet.
    totals: dict[str, float] = {}
    for case in golden:
        question = case["inputs"]["question"]
        scores = score_all(answers[question], case["expectations"])
        for name, value in scores.items():
            totals[f"{name}/mean"] = totals.get(f"{name}/mean", 0.0) + value
    metrics = {name: value / len(golden) for name, value in totals.items()}
    code_thresholds = [t for t in thresholds if t.metric in code_metrics]
    report = apply_gate(
        metrics,
        policy=GatePolicy(
            rules=tuple(code_thresholds),
            # Tier 1 enforces absolute deterministic thresholds only. The
            # governed tier-2 run owns comparison to the approved baseline.
            allow_missing_regression_baseline=True,
        ),
    )
    print({"tier": "offline", "metrics": report.metrics})
    report.require_passed()
    print(f"offline release-gate checks passed ({len(golden)} golden cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
