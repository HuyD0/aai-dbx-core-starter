import stat
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_hooks_are_local_credential_free_and_cover_both_git_stages():
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )

    assert config["default_install_hook_types"] == ["pre-commit", "pre-push"]
    assert [repository["repo"] for repository in config["repos"]] == ["local"]

    hooks = {
        hook["id"]: hook
        for repository in config["repos"]
        for hook in repository["hooks"]
    }
    assert hooks["aai-pre-commit"]["entry"] == "./scripts/pre-commit.sh"
    assert hooks["aai-pre-commit"]["stages"] == ["pre-commit"]
    assert hooks["aai-pre-push"]["entry"] == "./scripts/pre-push.sh"
    assert hooks["aai-pre-push"]["stages"] == ["pre-push"]

    combined = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            ".pre-commit-config.yaml",
            "scripts/pre-commit.sh",
            "scripts/pre-push.sh",
        )
    )
    for forbidden in (
        "azure/login",
        "az login",
        "DATABRICKS_TOKEN",
        "AZURE_CLIENT_SECRET",
        "id-token: write",
    ):
        assert forbidden not in combined


def test_commit_and_push_scripts_reuse_repository_verification_contracts():
    commit_script = (ROOT / "scripts/pre-commit.sh").read_text(encoding="utf-8")
    push_script = (ROOT / "scripts/pre-push.sh").read_text(encoding="utf-8")

    assert "git diff --cached --check" in commit_script
    assert "make check-templates format-check typecheck" in commit_script
    assert '-m "not generated_project"' in commit_script
    assert "./scripts/cloud-verify.sh" in push_script


def test_hook_scripts_are_executable_and_generated_tier_is_registered():
    for relative in ("scripts/pre-commit.sh", "scripts/pre-push.sh"):
        mode = (ROOT / relative).stat().st_mode
        assert mode & stat.S_IXUSR, f"{relative} must remain executable"

    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"generated_project:' in project
