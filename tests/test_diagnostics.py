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
