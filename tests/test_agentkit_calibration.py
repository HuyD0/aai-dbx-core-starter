"""Unit tests for judge calibration: κ, consensus, ceiling, records."""

import json

import pytest
from pydantic import ValidationError

from aai_core.agentkit.calibration import (
    DEFAULT_MINIMUM_KAPPA,
    MINIMUM_LABELS,
    AnnotatorVerdict,
    CalibrationLabel,
    CalibrationRecord,
    calibrate,
    calibration_failures,
    calibration_path,
    calibration_status,
    canonical_verdict,
    cohen_kappa,
    consensus_verdicts,
    human_ceiling,
    labels_digest,
    load_calibration,
    load_labels,
    percent_agreement,
    write_calibration,
)
from aai_core.agentkit.errors import ConfigError


def _label(example_id, judge, *votes, annotators=("group:rev-a", "group:rev-b")):
    return CalibrationLabel(
        example_id=example_id,
        judge_value=judge,
        annotations=tuple(
            AnnotatorVerdict(annotator=annotators[index], value=vote)
            for index, vote in enumerate(votes)
        ),
    )


def _labels(judge_values, human_values):
    return tuple(
        _label(f"case-{index}", judge, human, human)
        for index, (judge, human) in enumerate(
            zip(judge_values, human_values, strict=True)
        )
    )


# --- verdict normalization --------------------------------------------------


def test_canonical_verdict_collapses_synonyms():
    assert canonical_verdict("yes") == "pass"
    assert canonical_verdict(True) == "pass"
    assert canonical_verdict(1) == "pass"
    assert canonical_verdict("no") == "fail"
    assert canonical_verdict(False) == "fail"
    assert canonical_verdict(0.0) == "fail"
    assert canonical_verdict(0.5) == "0.5"
    assert canonical_verdict("Partial ") == "partial"


# --- agreement math ---------------------------------------------------------


def test_perfect_agreement_is_kappa_one():
    labels = ["pass", "fail", "pass", "fail"]
    assert percent_agreement(labels, labels) == 1.0
    assert cohen_kappa(labels, labels) == 1.0


def test_kappa_matches_the_hand_computed_two_by_two():
    judge = ["pass", "pass", "fail", "fail", "pass", "fail", "pass", "pass"]
    humans = ["pass", "fail", "fail", "fail", "pass", "fail", "pass", "pass"]
    # observed 7/8 = 0.875; chance (5/8)(4/8) + (3/8)(4/8) = 0.5
    assert percent_agreement(judge, humans) == pytest.approx(0.875)
    assert cohen_kappa(judge, humans) == pytest.approx(0.75)


def test_chance_agreement_deflates_raw_agreement():
    # 90% raw agreement on a 90/10 skewed set is barely better than chance.
    judge = ["pass"] * 10
    humans = ["pass"] * 9 + ["fail"]
    assert percent_agreement(judge, humans) == pytest.approx(0.9)
    assert cohen_kappa(judge, humans) == pytest.approx(0.0)


def test_degenerate_single_category_is_read_conservatively():
    assert cohen_kappa(["pass", "pass"], ["pass", "pass"]) == 1.0


def test_agreement_refuses_mismatched_sequences():
    with pytest.raises(ConfigError):
        percent_agreement(["pass"], [])


# --- consensus and ceiling --------------------------------------------------


def test_consensus_majority_wins_and_ties_are_counted_out():
    labels = (
        _label("a", "yes", "yes", "yes"),
        _label("b", "yes", "yes", "no"),  # tie: excluded
        _label(
            "c",
            "no",
            "no",
            "no",
            "yes",
            annotators=("group:rev-a", "group:rev-b", "group:rev-c"),
        ),
    )
    judge, humans, ties = consensus_verdicts(labels)
    assert ties == 1
    assert judge == ["pass", "fail"]
    assert humans == ["pass", "fail"]


def test_human_ceiling_is_pairwise_kappa():
    labels = _labels(["yes"] * 4 + ["no"] * 4, ["yes"] * 4 + ["no"] * 4)
    assert human_ceiling(labels) == 1.0


def test_single_annotator_has_no_ceiling():
    labels = tuple(
        _label(f"case-{index}", "yes", "yes", annotators=("group:rev-a",))
        for index in range(4)
    )
    assert human_ceiling(labels) is None


# --- record construction ----------------------------------------------------


def _many_labels(agreeing=20, disagreeing=0):
    judges = ["yes", "no"] * (agreeing // 2) + ["yes"] * (agreeing % 2)
    humans = list(judges)
    for index in range(disagreeing):
        humans[index] = "no" if judges[index] == "yes" else "yes"
    return _labels(judges, humans)


def test_calibrate_builds_a_passing_record():
    record = calibrate(
        scorer="correctness",
        scorer_version=1,
        labels=_many_labels(),
        recorded_at="2026-08-19T10:00:00Z",
        decided_by="group:pension-ai-owners",
    )
    assert record.passed
    assert record.kappa == 1.0
    assert record.human_ceiling_kappa == 1.0
    assert record.sample_size == 20
    assert record.annotator_count == 2
    assert record.minimum_kappa == DEFAULT_MINIMUM_KAPPA


def test_calibrate_fails_the_bar_without_hiding_the_number():
    record = calibrate(
        scorer="correctness",
        scorer_version=1,
        labels=_many_labels(disagreeing=8),
        recorded_at="2026-08-19T10:00:00Z",
    )
    assert not record.passed
    assert record.kappa < DEFAULT_MINIMUM_KAPPA


def test_calibrate_refuses_a_tiny_label_set():
    with pytest.raises(ConfigError, match=str(MINIMUM_LABELS)):
        calibrate(
            scorer="correctness",
            scorer_version=1,
            labels=_many_labels()[:5],
            recorded_at="2026-08-19T10:00:00Z",
        )


def test_calibrate_refuses_an_all_tie_rubric():
    labels = tuple(
        _label(f"case-{index}", "yes", "yes", "no") for index in range(MINIMUM_LABELS)
    )
    with pytest.raises(ConfigError, match="rubric"):
        calibrate(
            scorer="correctness",
            scorer_version=1,
            labels=labels,
            recorded_at="2026-08-19T10:00:00Z",
        )


def test_record_refuses_personal_identities():
    with pytest.raises(ValidationError, match="email"):
        CalibrationRecord(
            scorer="correctness",
            scorer_version=1,
            labels_digest="0" * 64,
            sample_size=20,
            annotator_count=2,
            percent_agreement=1.0,
            kappa=1.0,
            passed=True,
            recorded_at="2026-08-19T10:00:00Z",
            decided_by="someone@example.com",
        )
    with pytest.raises(ValidationError, match="email"):
        AnnotatorVerdict(annotator="reviewer@example.com", value="yes")


def test_record_round_trips_through_the_committed_file(tmp_path):
    record = calibrate(
        scorer="correctness",
        scorer_version=1,
        labels=_many_labels(),
        recorded_at="2026-08-19T10:00:00Z",
        judge_model="endpoints:/judge-endpoint",
        judge_model_identity="catalog.schema.judge/3",
        judge_prompt_uri="prompts:/cat.sch.judge_prompt/2",
    )
    path = calibration_path(tmp_path, "evals/judges", "correctness")
    write_calibration(path, record)
    assert path == tmp_path / "evals" / "judges" / "correctness.json"
    loaded = load_calibration(path)
    assert loaded == record


def test_labels_digest_changes_when_a_label_changes(tmp_path):
    first = _many_labels()
    second = _many_labels(disagreeing=1)
    assert labels_digest(first) != labels_digest(second)
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([label.model_dump(mode="json") for label in first]))
    assert load_labels(path) == first


# --- enforcement ------------------------------------------------------------


def _committed(tmp_path, scorer="correctness", version=1, passed=True):
    record = CalibrationRecord(
        scorer=scorer,
        scorer_version=version,
        labels_digest="0" * 64,
        sample_size=20,
        annotator_count=2,
        percent_agreement=0.9,
        kappa=0.8 if passed else 0.2,
        minimum_kappa=DEFAULT_MINIMUM_KAPPA,
        passed=passed,
        recorded_at="2026-08-19T10:00:00Z",
    )
    write_calibration(calibration_path(tmp_path, "evals/judges", scorer), record)
    return record


def test_missing_record_is_a_failure_naming_the_command(tmp_path):
    failures = calibration_failures(
        root=tmp_path, directory="evals/judges", judge_scorers={"correctness": 1}
    )
    assert len(failures) == 1
    assert "agentkit judge calibrate" in failures[0]


def test_version_mismatch_and_failed_records_block(tmp_path):
    _committed(tmp_path, version=1)
    assert (
        calibration_failures(
            root=tmp_path,
            directory="evals/judges",
            judge_scorers={"correctness": 1},
        )
        == []
    )
    stale = calibration_failures(
        root=tmp_path, directory="evals/judges", judge_scorers={"correctness": 2}
    )
    assert len(stale) == 1 and "re-calibrate" in stale[0]
    _committed(tmp_path, scorer="safety", passed=False)
    failed = calibration_failures(
        root=tmp_path, directory="evals/judges", judge_scorers={"safety": 1}
    )
    assert len(failed) == 1 and "failed calibration" in failed[0]


def test_status_reports_staleness_without_blocking(tmp_path):
    _committed(tmp_path)
    rows = calibration_status(
        root=tmp_path,
        directory="evals/judges",
        judge_scorers={"correctness": 1, "safety": 1},
        judge_prompts={"correctness": "prompts:/cat.sch.p/3"},
        judge_model_identity="catalog.schema.other-judge/9",
    )
    by_scorer = {row["scorer"]: row for row in rows}
    assert by_scorer["safety"]["status"] == "uncalibrated"
    assert by_scorer["correctness"]["status"] == "passed"
    # The committed record carries no judge identity, so nothing is stale.
    assert "stale_judge" not in by_scorer["correctness"]
