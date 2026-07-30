"""AI Platform Hub domain contracts.

The Hub is deliberately separate from the guided-onboarding modules.  This package
contains versioned, persistence-safe boundaries that can be reused by API and storage
adapters without coupling them to the current server-rendered UI.
"""

from .manifest import (
    AIApplicationManifest,
    ManifestEnvelope,
    build_manifest_envelope,
    canonical_manifest_json,
    load_manifest,
    manifest_json_schema,
)

__all__ = [
    "AIApplicationManifest",
    "ManifestEnvelope",
    "build_manifest_envelope",
    "canonical_manifest_json",
    "load_manifest",
    "manifest_json_schema",
]
