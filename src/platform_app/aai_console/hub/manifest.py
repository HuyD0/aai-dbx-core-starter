"""Compatibility exports for the SDK-owned application manifest contract.

The manifest is consumed by generated projects as well as the platform Hub, so
its authoritative implementation lives in :mod:`aai_core.manifest`.  Keep this
module as a stable import path for existing Hub integrations.
"""

from aai_core.manifest import (
    MANIFEST_API_VERSION,
    MANIFEST_KIND,
    SUPPORTED_READINESS_PROFILES,
    AIApplicationManifest,
    ApplicationSpec,
    AuthorizationSpec,
    CostControlsSpec,
    EnvironmentSpec,
    EvaluationSpec,
    ManifestEnvelope,
    ManifestMetadata,
    ReadinessSpec,
    RepositorySpec,
    ResourceBindings,
    ServiceLevels,
    build_manifest_envelope,
    canonical_manifest_json,
    load_manifest,
    manifest_json_schema,
)

__all__ = [
    "AIApplicationManifest",
    "ApplicationSpec",
    "AuthorizationSpec",
    "CostControlsSpec",
    "EnvironmentSpec",
    "EvaluationSpec",
    "MANIFEST_API_VERSION",
    "MANIFEST_KIND",
    "ManifestEnvelope",
    "ManifestMetadata",
    "ReadinessSpec",
    "RepositorySpec",
    "ResourceBindings",
    "ServiceLevels",
    "SUPPORTED_READINESS_PROFILES",
    "build_manifest_envelope",
    "canonical_manifest_json",
    "load_manifest",
    "manifest_json_schema",
]
