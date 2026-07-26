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

    assert "make check-templates format-check test" in commit_script
    assert "./scripts/cloud-verify.sh" in push_script
