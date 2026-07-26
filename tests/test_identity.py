"""Identity factory tests — security-sensitive mode dispatch, no cloud."""

import pytest
from conftest import install_fake_module

from aai_core.identity import azure_credential, identity_summary


class _Recorder:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return (self.name, kwargs)


@pytest.mark.parametrize(
    ("mode", "class_name"),
    [
        ("auto", "DefaultAzureCredential"),
        ("azure_cli", "AzureCliCredential"),
        ("azure-cli", "AzureCliCredential"),
        ("managed_identity", "ManagedIdentityCredential"),
        ("workload_identity", "WorkloadIdentityCredential"),
    ],
)
def test_each_identity_mode_builds_the_matching_credential(
    monkeypatch, mode, class_name
):
    recorders = {
        name: _Recorder(name)
        for name in (
            "DefaultAzureCredential",
            "AzureCliCredential",
            "ManagedIdentityCredential",
            "WorkloadIdentityCredential",
        )
    }
    install_fake_module(monkeypatch, "azure.identity", **recorders)

    result_name, kwargs = azure_credential(mode)

    assert result_name == class_name
    if class_name == "DefaultAzureCredential":
        # Interactive browser login must never enter the chain.
        assert kwargs["exclude_interactive_browser_credential"] is True


def test_unknown_identity_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported Azure identity mode"):
        azure_credential("client_secret")


def test_identity_summary_is_non_sensitive(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.azuredatabricks.net")
    monkeypatch.setenv("AZURE_CLIENT_ID", "00000000-0000-0000-0000-000000000000")

    summary = identity_summary("azure_cli")

    assert summary["azure_identity"] == "azure_cli"
    assert summary["databricks_host_configured"] == "true"
    assert summary["azure_client_id_configured"] == "true"
    # Presence booleans only — never the values themselves.
    assert "example.azuredatabricks.net" not in str(summary.values())
    assert "00000000" not in str(summary.values())
