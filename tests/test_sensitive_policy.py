"""Cross-boundary sensitive-name normalization and drift guards."""

from __future__ import annotations

import pytest

from aai_core.experiments import _safe_parameters
from aai_core.manifest import _reject_secret_like_keys
from aai_core.tracing import TracePolicy, sanitize_trace_payload


@pytest.mark.parametrize(
    "field_name",
    [
        "authorization",
        "authorizationHeader",
        "vendorApiKey",
        "client-secret",
        "access.token",
        "refresh/token",
        "databasePassword",
        "service_secret_reference",
    ],
)
def test_sensitive_names_are_consistent_across_evidence_boundaries(field_name):
    raw_value = "credential-value-must-not-leak"

    with pytest.raises(ValueError, match="sensitive") as experiment_error:
        _safe_parameters({field_name: raw_value})
    assert raw_value not in str(experiment_error.value)

    sanitized = sanitize_trace_payload(
        {field_name: raw_value},
        policy=TracePolicy(),
    )
    assert sanitized[field_name] == "[REDACTED]"

    if field_name == "authorization":
        # This exact key is a policy section in the manifest, never a header.
        _reject_secret_like_keys({"authorization": {"mode": "application"}})
    else:
        with pytest.raises(ValueError, match="secret-like") as manifest_error:
            _reject_secret_like_keys({field_name: raw_value})
        assert raw_value not in str(manifest_error.value)


def test_usage_counts_are_not_misclassified_as_credentials():
    assert _safe_parameters({"token_count": 42}) == {"token_count": 42}
    assert sanitize_trace_payload(
        {"token_count": 42},
        policy=TracePolicy(),
    ) == {"token_count": 42}
    _reject_secret_like_keys({"token_count": 42})
