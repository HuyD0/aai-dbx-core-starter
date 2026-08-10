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

Schema version 2 groups that evidence into three independently comparable
clocks without adding another runtime framework:

- **World:** governed source, dataset, index, and freshness snapshots.
- **Learning:** source commit, SDK, model, prompt, retrieval configuration, and
  tool schemas that determine adaptive behavior.
- **Control:** evaluation, baseline, manifest, readiness, budget-policy, and
  service-level evidence.

Each clock has a deterministic digest. Runtime traces keep the ordinary
`aai.release` join key rather than copying the full release document into every
request; the release record is the durable explanation of which three clock
versions produced that behavior.

## Evidence and enforcement boundary

Generated applications call native MLflow APIs for traces, Assessments,
Evaluation Datasets, Prompt Registry versions, scorers, and evaluation. The
starter supplies small examples and safe defaults; it never creates a second
feedback store or evaluation result type. Human interventions and observed
business outcomes are attached to the originating trace, and reviewed
corrections may become expectations before a trace is curated into the Unity
Catalog evaluation dataset. Production feedback never becomes training or
release data automatically.

For MLflow Agent Server, the SDK owns one pre-export span processor because
the framework writes its root output after the application handler returns.
That processor applies the classification policy to invoke and streaming root
spans; metadata-only mode preserves typed operational evidence and hashes
session IDs, while tracing-off disables native export. Agent Server uses this
as its single payload-sanitization boundary, clears payloads before processing
so an unexpected processor error fails closed, and records only generic error
categories rather than provider/tool exception messages. It is appended after
preinstalled platform processors instead of replacing them, and full-capture
mode leaves the processor list untouched. SDK-decorated async generators keep
the trace and request context open through iteration and cancellation, so
stream children remain in the same trace. Other integrations keep their
existing native processor ownership.

The SDK cannot enforce a budget against clients that bypass it. Hard controls
therefore remain platform-owned: approved gateway routing, request tags, rate
limits, usage policies, authoritative billing, and a readiness rule that fails
closed when budget policy or cost attribution cannot be verified. Generated
agent limits (tool turns, output tokens, tool timeouts, and request deadlines)
are defense in depth and keep local/test behavior predictable.

This split keeps the new-team path small: declare ownership and a cost center,
write the behavior and evaluation cases, and use the generated workflow. The
platform validates the manifest and enforces deployment policy; advanced teams
can still use native Databricks, MLflow, or provider capabilities directly.
