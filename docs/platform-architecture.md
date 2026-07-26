# Platform architecture

AAI Core is one part of the AI/ML developer platform:

```text
Development team
  -> generated project template
  -> aai-core SDK contracts
  -> native MLflow, Databricks, Azure, and OpenAI-compatible clients
  -> platform-managed identities, policies, catalogs, endpoints, and monitoring
```

## Responsibility boundaries

The SDK owns runtime configuration, logical resource resolution, secret
redaction, tag propagation, trace policy, evaluation gates, and release
metadata. Persisted and untrusted boundaries use strict Pydantic models and
small enums. Internal orchestration may use ordinary Python types when another
schema would add no safety.

Templates own project structure, examples, tests, bundle resources, and the
recommended development lifecycle.

The platform team owns identities, Key Vaults, Unity Catalog, compute and
serverless policies, model deployments, search services/indexes, network
controls, budgets, dashboards, and incident response.

Application teams own business behavior, domain data, prompts, tools,
evaluation cases, quality thresholds, and on-call response for their
applications.

## Provider boundary

Applications reference logical resources such as `general-chat` and
`product-knowledge`. `aai-platform.yml` resolves those names to environment
resources.

Portable contracts cover synchronous non-streaming chat generation, embeddings,
and retrieval. Provider capabilities are explicit. Every model exposes its
actual synchronous `native_client`, and `create_native_async_client()` creates
an event-loop-local native client with the same governed identity, gateway,
timeout, and retry configuration. Callers own async client and stream cleanup.
MLflow helpers likewise retain native runs, traces, evaluation results, prompts,
and logged models. The SDK adds governance and safe defaults; it does not rename
or mirror provider async, streaming, Responses API, graph, or state types.
Index provisioning, Foundry-hosted agents, Databricks Delta Sync, and provider
administration remain provider-specific.

This boundary is intentionally escape-friendly:

```text
application code
  -> aai-core governed capability (recommended path)
  -> native_client / create_native_async_client / native result
```

Application teams may use the native SDK directly when the stable capability
does not cover a feature. They still own configuration, identity, tagging,
tracing, evaluation, and release evidence required by platform policy.

## Release unit

For GenAI, the model is not the complete release. An `ApplicationRelease`
binds code, SDK, model deployment, prompt version, retrieval index, embedding
and chunking profiles, evaluation evidence, and environment into one
immutable, hashable record.
