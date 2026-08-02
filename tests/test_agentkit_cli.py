"""Unit tests for the CLI surface and the exit-code contract."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_core.agentkit.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]

AGENT_SOURCE = """\
KNOWLEDGE = {f"question {i}": f"answer {i} about pensions" for i in range(12)}


def respond(question):
    return KNOWLEDGE.get(question, "I cannot help with that")
"""

PLATFORM_YAML = """\
platform:
  application: quality-eval
  project: agent-quality
  team: pension-ai
  owner_group: group:pension-ai-owners
  cost_center: CC-9999
  repository: example/agent-quality
providers:
  models:
    judge-model:
      provider: databricks
      deployment: judge-endpoint
"""


@pytest.fixture
def project_dir(tmp_path, monkeypatch):
    (tmp_path / "src" / "app").mkdir(parents=True)
    (tmp_path / "src" / "app" / "example_agent.py").write_text(AGENT_SOURCE)
    (tmp_path / "aai-platform.yml").write_text(PLATFORM_YAML)
    data = tmp_path / "evals" / "data"
    data.mkdir(parents=True)
    cases = [
        {
            "inputs": {"question": f"question {index}"},
            "expectations": {"expected_response": f"answer {index} about pensions"},
        }
        for index in range(12)
    ]
    (data / "golden_cases.json").write_text(json.dumps(cases))
    (data / "answer_sheet.json").write_text(
        json.dumps(
            [
                {
                    "question": f"question {index}",
                    "answer": f"answer {index} about pensions",
                }
                for index in range(12)
            ]
        )
    )
    (tmp_path / "agentkit.yaml").write_text(
        "version: 1\n"
        "agent: src/app/example_agent.py:respond\n"
        "dataset: evals/data/golden_cases.json\n"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _config_flag(project_dir):
    return ["--config", str(project_dir / "agentkit.yaml")]


def test_smoke_runs_on_base_dependencies(project_dir, capsys):
    code = main(["smoke", *_config_flag(project_dir)])

    output = capsys.readouterr().out
    assert code == 0
    assert "Inferred evaluation plan" in output
    assert "keyword_coverage" in output
    assert "0 judge calls" in output
    assert "gate: PASSED" in output


def test_smoke_establishes_and_then_compares(project_dir, capsys):
    assert main(["smoke", "--establish-baseline", *_config_flag(project_dir)]) == 0
    capsys.readouterr()

    assert main(["smoke", *_config_flag(project_dir)]) == 0
    output = capsys.readouterr().out
    assert "Compared against" in output


def test_gate_without_results_exits_one(project_dir, capsys):
    code = main(["gate", *_config_flag(project_dir)])

    assert code == 1
    assert "agentkit compare" in capsys.readouterr().err


def test_gate_after_smoke_exits_zero(project_dir, capsys):
    main(["smoke", "--establish-baseline", *_config_flag(project_dir)])
    capsys.readouterr()

    code = main(["gate", *_config_flag(project_dir)])

    output = capsys.readouterr().out
    assert code == 0
    assert "gate: PASSED" in output
    assert "this run IS the recorded baseline" in output


def test_gate_exits_two_when_a_threshold_fails(project_dir, capsys):
    # The recorded answers no longer contain the expected content, so the
    # deterministic coverage scorer drops below its registry threshold.
    (project_dir / "evals" / "data" / "answer_sheet.json").write_text(
        json.dumps(
            [
                {"question": f"question {index}", "answer": "unrelated text"}
                for index in range(12)
            ]
        )
    )
    main(["smoke", "--establish-baseline", *_config_flag(project_dir)])
    capsys.readouterr()

    code = main(["gate", *_config_flag(project_dir)])

    output = capsys.readouterr().out
    assert code == 2
    assert "gate: FAILED" in output
    assert "keyword_coverage/mean" in output


def test_gate_fails_closed_on_a_missing_thresholded_metric(project_dir, capsys):
    main(["smoke", "--establish-baseline", *_config_flag(project_dir)])
    (project_dir / "agentkit.yaml").write_text(
        "version: 1\n"
        "agent: src/app/example_agent.py:respond\n"
        "dataset: evals/data/golden_cases.json\n"
        "thresholds:\n"
        '  correctness: ">=0.7"\n'
    )
    capsys.readouterr()

    code = main(["gate", *_config_flag(project_dir)])

    assert code == 2
    assert "missing" in capsys.readouterr().out


def test_broken_agent_fails_the_gate_with_exit_two(project_dir, capsys):
    main(["smoke", "--establish-baseline", *_config_flag(project_dir)])
    (project_dir / "evals" / "data" / "answer_sheet.json").write_text(
        json.dumps(
            [{"question": f"question {index}", "answer": ""} for index in range(12)]
        )
    )
    capsys.readouterr()

    code = main(["smoke", *_config_flag(project_dir)])

    assert code == 2
    assert "gate: FAILED" in capsys.readouterr().out


def test_plan_flag_prints_without_scoring(project_dir, capsys):
    code = main(["smoke", "--plan", *_config_flag(project_dir)])

    output = capsys.readouterr().out
    assert code == 0
    assert "Inferred evaluation plan" in output
    assert "gate:" not in output
    assert not (project_dir / ".aai").exists()


def test_json_output_is_one_document(project_dir, capsys):
    code = main(["smoke", "--json", *_config_flag(project_dir)])

    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["exit_code"] == 0
    assert document["cost"]["judge_calls"] == 0
    assert any(entry["scorer"] == "keyword_coverage" for entry in document["plan"])


def test_json_without_yes_refuses_a_judged_run(project_dir, capsys):
    code = main(["compare", "--json", *_config_flag(project_dir)])

    assert code == 1
    assert "--yes" in capsys.readouterr().err


def test_compare_without_a_baseline_reports_the_refusal(project_dir, capsys):
    code = main(
        ["compare", "--yes", "--mode", "answer-sheet", *_config_flag(project_dir)]
    )

    assert code == 1
    error = capsys.readouterr().err
    assert "--establish-baseline" in error


def test_invalid_config_exits_one(project_dir, capsys):
    (project_dir / "agentkit.yaml").write_text("version: 1\nagent: a\n")

    code = main(["smoke", *_config_flag(project_dir)])

    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_unknown_scorer_exits_one_and_names_the_registry(project_dir, capsys):
    (project_dir / "agentkit.yaml").write_text(
        "version: 1\n"
        "agent: src/app/example_agent.py:respond\n"
        "dataset: evals/data/golden_cases.json\n"
        "scorers:\n  add: [not_a_scorer]\n"
    )

    code = main(["smoke", *_config_flag(project_dir)])

    assert code == 1
    assert "correctness" in capsys.readouterr().err


def test_evidence_writes_the_record(project_dir, capsys):
    main(["smoke", "--establish-baseline", *_config_flag(project_dir)])
    capsys.readouterr()

    code = main(["evidence", *_config_flag(project_dir)])

    output = capsys.readouterr().out
    assert code == 0
    assert "# Release evidence" in output
    written = project_dir / ".aai" / "agentkit" / "evidence" / "evidence.json"
    assert json.loads(written.read_text())["decision"] == "inconclusive"


def test_evidence_without_results_exits_one(project_dir, capsys):
    code = main(["evidence", *_config_flag(project_dir)])

    assert code == 1
    assert "agentkit compare" in capsys.readouterr().err


def test_scorers_ls_lists_the_registry_offline(capsys):
    code = main(["scorers", "ls"])

    output = capsys.readouterr().out
    assert code == 0
    for name in ("correctness", "safety", "keyword_coverage", "retrieval_groundedness"):
        assert name in output
    assert "never redefines one" in output


def test_scorers_ls_json(capsys):
    code = main(["scorers", "ls", "--json"])

    assert code == 0
    document = json.loads(capsys.readouterr().out)
    names = {entry["name"] for entry in document}
    assert "pension_domain_policy" in names
    assert all("version" in entry for entry in document)


def test_init_print_only_uses_the_platform_template(capsys, monkeypatch):
    monkeypatch.setenv("AAI_TEMPLATE_REPO", "https://example.invalid/org/repo")

    code = main(["init", "--name", "pension-agent", "--print-only"])

    output = capsys.readouterr().out
    assert code == 0
    assert "databricks bundle init https://example.invalid/org/repo" in output
    assert "--template-dir templates/evaluation-project" in output
    assert "agentkit compare --establish-baseline" in output


def test_init_without_a_template_source_explains_platform_env(capsys, monkeypatch):
    monkeypatch.delenv("AAI_TEMPLATE_REPO", raising=False)

    code = main(["init", "--name", "pension-agent", "--print-only"])

    assert code == 1
    assert "platform-env.sh" in capsys.readouterr().err


def test_init_rejects_an_invalid_project_name(capsys, monkeypatch):
    monkeypatch.setenv("AAI_TEMPLATE_REPO", "https://example.invalid/org/repo")

    code = main(["init", "--name", "Not Valid", "--print-only"])

    assert code == 1
    assert "lowercase" in capsys.readouterr().err


def test_eval_submit_runs_the_bundle_job(project_dir, capsys, monkeypatch):
    commands = []

    def fake_run(command, cwd=None, check=False):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    code = main(["eval", "--submit", *_config_flag(project_dir)])

    assert code == 0
    assert commands[1][:4] == ["databricks", "bundle", "run", "release_gate"]


def test_cli_imports_no_heavy_dependencies():
    """`agentkit smoke` must work in an environment without MLflow."""

    script = (
        "import sys; import aai_core.agentkit.cli;"
        "heavy = [m for m in ('mlflow', 'databricks', 'openai', 'azure') "
        "if m in sys.modules];"
        "print(heavy)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.stdout.strip() == "[]"


def test_agent_override_flag_is_accepted(project_dir, capsys):
    code = main(
        [
            "smoke",
            "--agent",
            "evals/data/answer_sheet.json",
            "--plan",
            *_config_flag(project_dir),
        ]
    )

    assert code == 0
    assert "Inferred evaluation plan" in capsys.readouterr().out


def test_evidence_reports_why_approval_is_unknown(project_dir, capsys):
    main(["smoke", "--establish-baseline", *_config_flag(project_dir)])
    capsys.readouterr()

    code = main(["evidence", "--json", *_config_flag(project_dir)])

    assert code == 0
    document = json.loads(capsys.readouterr().out)
    # The shipped lookup runs; with no registered model it explains itself
    # instead of silently reporting a bare "unknown".
    assert document["approver"]["status"] == "unknown"
    assert "registered_model" in document["approver"]["reason"]
