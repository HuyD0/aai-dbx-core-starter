---
name: aai-change-sdk
description: Safely change the aai-core Python SDK, including public contracts, configuration, providers, tracing, secrets, evaluation, MLflow integration, concurrency, packaging, and compatibility. Use for work under src/aai_core, SDK-facing tests, runtime dependencies, or behavior consumed by examples and generated projects.
---

# Change the AAI SDK

Evolve the SDK without hiding native provider capabilities or breaking generated
projects. Prefer the smallest policy-bearing change over a new framework.

## Workflow

1. Read `AGENTS.md`, `pyproject.toml`, `compatibility.json`, and the affected
   domain tests. Inspect `git status` and preserve unrelated work.
2. Locate every consumer in examples and rendered templates before changing a
   public symbol, model, exception, tag, or evidence shape.
3. Classify the change:
   - Public contract: preserve compatibility or follow the declared release
     migration.
   - Boundary data: use strict Pydantic v2 models and platform-owned enums.
   - Internal transient state: use ordinary Python types.
   - Provider feature: expose it through `native_client` unless two providers
     share a stable capability.
4. Implement explicit ownership, timeouts, cancellation, and cleanup. Close only
   SDK-created resources. Keep caller-owned clients open.
5. Add focused tests before broad verification. Include failure, concurrency,
   redaction, and compatibility cases when applicable.
6. Run the narrow tests, then `make check`. Run `./scripts/cloud-verify.sh` before
   handing off a release-ready change.

## Design guardrails

- Keep the top-level entry point limited to `bootstrap`, `PlatformSettings`,
  `PlatformContext`, and domain modules.
- Keep synchronous stable adapters small. Do not invent a universal async or
  streaming event model; preserve native async clients and stream types.
- Fail configuration and capability checks before expensive provider calls.
- Keep authentication native to Azure, Databricks, and provider SDKs.
- Never expose `SecretValue` through strings, logs, exceptions, traces, tags, or
  MLflow parameters. Test the final formatted exception and traceback.
- Keep classic MLflow evaluation separate from `mlflow.genai.evaluate()`.
- Require stable case IDs, deterministic ordering, and immutable evidence at
  release boundaries.
- Reject abstractions that neither protect an external boundary nor have two
  real consumers. Do not add a DI container, ORM, event bus, or rules DSL.

## Dependency changes

When a runtime dependency changes, update its policy entry, `uv.lock`, affected
template locks, and `compatibility.json` together as required by `AGENTS.md`.
Use the repository lock scripts; do not hand-edit generated lock files.

## Handoff

Report the public behavior changed, compatibility impact, ownership semantics,
tests run, and any live validation still required. Never publish or make cloud
identity changes unless the user explicitly authorizes them.
