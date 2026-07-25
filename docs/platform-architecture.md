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
redaction, tag propagation, trace schemas, evaluation gates, and release
metadata.

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

Portable contracts cover chat generation, embeddings, and retrieval. Provider
capabilities are explicit, and every adapter exposes `native_client` for
advanced features. Index provisioning, Foundry-hosted agents, Databricks
Delta Sync, and provider administration remain provider-specific.

## Release unit

For GenAI, the model is not the complete release. An `ApplicationRelease`
binds code, SDK, model deployment, prompt version, retrieval index, embedding
and chunking profiles, evaluation evidence, and environment into one
immutable, hashable record.
