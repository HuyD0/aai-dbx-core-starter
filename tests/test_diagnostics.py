import json

from aai_core.diagnostics import _module_available, main, run_doctor

VALID_CONFIG = """
platform:
  application: doctor-test
  project: doctor
  environment: dev
  team: test-team
  owner_group: group:test-owners
  cost_center: CC-0000
  azure_identity: azure_cli
"""


def test_missing_nested_module_is_reported_without_crashing():
    assert not _module_available("definitely_missing_parent.child")


def test_doctor_passes_on_valid_config_without_cloud(tmp_path):
    config = tmp_path / "aai-platform.yml"
    config.write_text(VALID_CONFIG, encoding="utf-8")

    checks = run_doctor(config_path=config)

    by_name = {check.name: check for check in checks}
    assert by_name["configuration"].status == "pass"
    # Optional provider modules absent in the dev env degrade to actionable
    # skips, never failures.
    dependency_checks = [c for c in checks if c.name.startswith("dependency:")]
    assert dependency_checks
    assert all(c.status in {"pass", "skip"} for c in dependency_checks)
    skipped = [check for check in dependency_checks if check.status == "skip"]
    assert all("install aai-core[" in check.detail for check in skipped)


def test_doctor_reports_lifecycle_readiness_as_skips_not_failures(tmp_path):
    config = tmp_path / "aai-platform.yml"
    config.write_text(VALID_CONFIG, encoding="utf-8")

    checks = run_doctor(config_path=config)

    by_name = {check.name: check for check in checks}
    assert by_name["lifecycle:experiment"].status == "pass"
    assert by_name["lifecycle:experiment"].detail.startswith("/Shared/")
    assert by_name["lifecycle:prompt-registry"].status == "skip"
    assert "platform.catalog" in by_name["lifecycle:prompt-registry"].detail
    assert by_name["lifecycle:judge-model"].status == "skip"
    assert "judge-model" in by_name["lifecycle:judge-model"].detail
    lifecycle_checks = [c for c in checks if c.name.startswith("lifecycle:")]
    assert all(c.status in {"pass", "skip", "info"} for c in lifecycle_checks)


def test_doctor_passes_lifecycle_checks_when_configured(tmp_path):
    config = tmp_path / "aai-platform.yml"
    config.write_text(
        VALID_CONFIG
        + """
  catalog: main
  schema: app

providers:
  models:
    judge-model:
      provider: databricks
      deployment: judge-ep
""",
        encoding="utf-8",
    )

    checks = run_doctor(config_path=config)

    by_name = {check.name: check for check in checks}
    assert by_name["lifecycle:prompt-registry"].status == "pass"
    assert by_name["lifecycle:prompt-registry"].detail == "main.app"
    assert by_name["lifecycle:judge-model"].status == "pass"
    assert by_name["lifecycle:judge-model"].detail == "endpoints:/judge-ep"


def test_doctor_cli_exit_codes_and_json_output(tmp_path, capsys):
    valid = tmp_path / "valid.yml"
    valid.write_text(VALID_CONFIG, encoding="utf-8")
    # A prod environment with placeholder settings must fail validation.
    invalid = tmp_path / "invalid.yml"
    invalid.write_text(
        VALID_CONFIG.replace("environment: dev", "environment: prod"),
        encoding="utf-8",
    )

    assert main(["doctor", "--config", str(valid)]) == 0
    checks = json.loads(capsys.readouterr().out)
    assert {"name", "status", "detail"} <= set(checks[0])

    assert main(["doctor", "--config", str(invalid)]) == 1
    failed = json.loads(capsys.readouterr().out)
    assert failed[0]["status"] == "fail"
