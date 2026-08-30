"""Tests for the granularity/repeats sweep script (simulate mode only —
the real mode needs a logprob-capable endpoint and is exercised manually)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sweep_continuous_scoring.py"


@pytest.fixture(scope="module")
def sweep():
    spec = importlib.util.spec_from_file_location("aai_sweep_continuous", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["aai_sweep_continuous"] = module
    spec.loader.exec_module(module)
    return module


def test_gold_items_carry_four_ordered_candidates(sweep):
    items = sweep.load_gold_items(ROOT / sweep.DEFAULT_DATASET, max_rows=None)
    assert len(items) >= 2
    for item in items:
        assert item.request
        assert [candidate.rank for candidate in item.candidates] == [3, 2, 1, 0]
        # the wrong candidate is another row's answer, never this row's
        assert item.candidates[3].text != item.expected


def test_simulated_sweep_recovers_the_known_ordering(sweep):
    items = sweep.load_gold_items(ROOT / sweep.DEFAULT_DATASET, max_rows=4)
    model = sweep.SimulatedVerifierModel()
    for item in items:
        for candidate in item.candidates:
            model.register(item.request, candidate)
    report = sweep.sweep_combination(
        model, items, granularity=5, repeats=1, low_mass_threshold=0.5
    )
    assert report.tau_continuous == pytest.approx(1.0)
    assert report.tau_discrete == pytest.approx(1.0)
    assert report.tie_rate_continuous == pytest.approx(0.0)
    assert report.rows_skipped == 0
    assert report.judge_calls == len(items) * 4 * 3  # candidates x criteria
    assert report.input_tokens > 0


def test_same_text_ranks_differently_under_different_questions(sweep):
    # A row's expected answer is a later row's "wrong" candidate. The
    # simulator must key on (request, response), or the collision quietly
    # inverts half the reference ordering.
    items = sweep.load_gold_items(ROOT / sweep.DEFAULT_DATASET, max_rows=2)
    model = sweep.SimulatedVerifierModel()
    for item in items:
        for candidate in item.candidates:
            model.register(item.request, candidate)
    first, second = items[0], items[1]
    assert first.candidates[3].text == second.candidates[0].text
    assert model.rank_by_key[(first.request, first.candidates[3].text)] == 0
    assert model.rank_by_key[(second.request, second.candidates[0].text)] == 3


def test_main_simulate_writes_the_json_report(sweep, tmp_path, capsys):
    output = tmp_path / "sweep.json"
    code = sweep.main(
        [
            "--simulate",
            "--granularities",
            "5,10",
            "--repeats",
            "1",
            "--max-rows",
            "3",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "tau_cont" in printed
    payload = json.loads(output.read_text())
    assert payload["model"] == "simulated"
    assert len(payload["combinations"]) == 2
    for combination in payload["combinations"]:
        assert combination["tau_continuous"] == pytest.approx(1.0)
        assert combination["judge_calls"] > 0


def test_int_list_flags_reject_garbage(sweep):
    with pytest.raises(SystemExit):
        sweep._parse_int_list("5,banana", flag="--granularities")
    with pytest.raises(SystemExit):
        sweep._parse_int_list("", flag="--repeats")
