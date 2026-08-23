"""Forward-only Lakebase schema migrations for the Platform Hub.

The app service principal creates and owns one dedicated schema.  Migrations are
transactional, additive, and checksum protected; changing an applied migration is a
startup error rather than an implicit rewrite of operational evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class LakebaseMigration:
    version: int
    description: str
    statements: tuple[str, ...]

    @property
    def checksum(self) -> str:
        payload = "\n-- statement --\n".join(
            statement.strip() for statement in self.statements
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def render(self, quoted_schema: str) -> tuple[str, ...]:
        return tuple(
            statement.replace("__HUB_SCHEMA__", quoted_schema)
            for statement in self.statements
        )


INITIAL_SCHEMA = LakebaseMigration(
    version=1,
    description="durable Hub registry, workflow, and audit records",
    statements=(
        """
        CREATE TABLE __HUB_SCHEMA__.applications (
            application_id TEXT PRIMARY KEY,
            row_version BIGINT NOT NULL CHECK (row_version >= 1),
            updated_at TIMESTAMPTZ NOT NULL,
            record JSONB NOT NULL
        )
        """,
        """
        CREATE TABLE __HUB_SCHEMA__.application_versions (
            version_id TEXT PRIMARY KEY,
            application_id TEXT NOT NULL REFERENCES
                __HUB_SCHEMA__.applications(application_id) ON DELETE RESTRICT,
            environment TEXT NOT NULL,
            git_repository TEXT NOT NULL,
            git_commit_sha_normalized TEXT NOT NULL,
            manifest_hash TEXT NOT NULL,
            deployment_target TEXT NOT NULL,
            registered_at TIMESTAMPTZ NOT NULL,
            is_current BOOLEAN NOT NULL,
            record JSONB NOT NULL,
            CONSTRAINT uq_hub_application_version_evidence UNIQUE (
                application_id,
                environment,
                git_commit_sha_normalized,
                manifest_hash
            )
        )
        """,
        """
        CREATE UNIQUE INDEX uq_hub_current_application_version
        ON __HUB_SCHEMA__.application_versions (application_id, environment)
        WHERE is_current
        """,
        """
        CREATE INDEX ix_hub_application_versions_registered
        ON __HUB_SCHEMA__.application_versions
            (application_id, registered_at, version_id)
        """,
        """
        CREATE TABLE __HUB_SCHEMA__.application_principals (
            application_id TEXT NOT NULL REFERENCES
                __HUB_SCHEMA__.applications(application_id) ON DELETE CASCADE,
            principal_type TEXT NOT NULL,
            principal_name_normalized TEXT NOT NULL,
            record JSONB NOT NULL,
            PRIMARY KEY (
                application_id,
                principal_type,
                principal_name_normalized
            )
        )
        """,
        """
        CREATE INDEX ix_hub_application_principals_lookup
        ON __HUB_SCHEMA__.application_principals
            (principal_type, principal_name_normalized, application_id)
        """,
        """
        CREATE TABLE __HUB_SCHEMA__.resource_bindings (
            binding_id TEXT PRIMARY KEY,
            application_id TEXT NOT NULL REFERENCES
                __HUB_SCHEMA__.applications(application_id) ON DELETE CASCADE,
            environment TEXT NOT NULL,
            record JSONB NOT NULL
        )
        """,
        """
        CREATE INDEX ix_hub_resource_bindings_application
        ON __HUB_SCHEMA__.resource_bindings
            (application_id, environment, binding_id)
        """,
        """
        CREATE TABLE __HUB_SCHEMA__.evaluation_runs (
            evaluation_run_id TEXT PRIMARY KEY,
            application_id TEXT NOT NULL REFERENCES
                __HUB_SCHEMA__.applications(application_id) ON DELETE RESTRICT,
            environment TEXT NOT NULL,
            application_version_id TEXT NOT NULL REFERENCES
                __HUB_SCHEMA__.application_versions(version_id) ON DELETE RESTRICT,
            status TEXT NOT NULL,
            requested_at TIMESTAMPTZ NOT NULL,
            row_version BIGINT NOT NULL CHECK (row_version >= 1),
            record JSONB NOT NULL
        )
        """,
        """
        CREATE UNIQUE INDEX uq_hub_active_evaluation
        ON __HUB_SCHEMA__.evaluation_runs
            (application_id, environment, application_version_id)
        WHERE status IN ('REQUESTED', 'QUEUED', 'RUNNING')
        """,
        """
        CREATE INDEX ix_hub_evaluations_application
        ON __HUB_SCHEMA__.evaluation_runs
            (application_id, requested_at, evaluation_run_id)
        """,
        """
        CREATE TABLE __HUB_SCHEMA__.promotion_requests (
            promotion_request_id TEXT PRIMARY KEY,
            application_id TEXT NOT NULL REFERENCES
                __HUB_SCHEMA__.applications(application_id) ON DELETE RESTRICT,
            application_version_id TEXT NOT NULL REFERENCES
                __HUB_SCHEMA__.application_versions(version_id) ON DELETE RESTRICT,
            target_environment TEXT NOT NULL,
            status TEXT NOT NULL,
            requested_at TIMESTAMPTZ NOT NULL,
            row_version BIGINT NOT NULL CHECK (row_version >= 1),
            record JSONB NOT NULL
        )
        """,
        """
        CREATE UNIQUE INDEX uq_hub_active_promotion
        ON __HUB_SCHEMA__.promotion_requests
            (application_id, application_version_id, target_environment)
        WHERE status IN (
            'PENDING_REVIEW',
            'CHANGES_REQUESTED',
            'APPROVED',
            'EXECUTING'
        )
        """,
        """
        CREATE INDEX ix_hub_promotions_application
        ON __HUB_SCHEMA__.promotion_requests
            (application_id, requested_at, promotion_request_id)
        """,
        """
        CREATE TABLE __HUB_SCHEMA__.action_events (
            event_id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            event_time TIMESTAMPTZ NOT NULL,
            record JSONB NOT NULL
        )
        """,
        """
        CREATE INDEX ix_hub_action_events_entity
        ON __HUB_SCHEMA__.action_events
            (entity_type, entity_id, event_time, event_id)
        """,
    ),
)


MIGRATIONS = (INITIAL_SCHEMA,)
