"""Native Azure and Databricks identity factories."""

from __future__ import annotations

import os
from typing import Any

__all__ = ["azure_credential", "databricks_workspace_client", "identity_summary"]


def azure_credential(mode: str = "auto", **kwargs: Any) -> object:
    """Create an Azure credential without accepting a client secret."""

    normalized = mode.replace("-", "_").lower()
    if normalized == "auto":
        from azure.identity import DefaultAzureCredential

        return DefaultAzureCredential(
            exclude_interactive_browser_credential=True, **kwargs
        )
    if normalized == "azure_cli":
        from azure.identity import AzureCliCredential

        return AzureCliCredential(**kwargs)
    if normalized == "managed_identity":
        from azure.identity import ManagedIdentityCredential

        return ManagedIdentityCredential(**kwargs)
    if normalized == "workload_identity":
        from azure.identity import WorkloadIdentityCredential

        return WorkloadIdentityCredential(**kwargs)
    raise ValueError(f"Unsupported Azure identity mode: {mode}")


def databricks_workspace_client(**kwargs: Any) -> object:
    """Use Databricks unified authentication with no SDK-specific token logic."""

    from databricks.sdk import WorkspaceClient

    return WorkspaceClient(**kwargs)


def identity_summary(azure_identity: str) -> dict[str, str]:
    """Return non-sensitive identity configuration for diagnostics."""

    return {
        "azure_identity": azure_identity,
        "databricks_auth_type": os.getenv("DATABRICKS_AUTH_TYPE", "default"),
        "databricks_host_configured": str(bool(os.getenv("DATABRICKS_HOST"))).lower(),
        "azure_client_id_configured": str(bool(os.getenv("AZURE_CLIENT_ID"))).lower(),
    }
