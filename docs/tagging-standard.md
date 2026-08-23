# Tagging standard

Every application supplies the following non-secret fields:

| Field | Purpose |
|---|---|
| `application` | Runtime application identity |
| `project` | Delivery or funding project |
| `environment` | Deployment environment |
| `team` | Owning delivery team |
| `owner_group` | Non-personal support/ownership group |
| `cost_center` | Finance attribution |
| `data_classification` | Handling category: `public`, `internal`, `confidential`, or `restricted` |
| `lifecycle` | Schema-v2 application maturity: `experimental`, `validation`, `production`, or `retired` |
| `repository` | Source repository |
| `release` | Application release |
| `tag_schema_version` | Tag-contract version |

AAI Core projects one validated context to MLflow, traces, Databricks
resources, Azure resources, and structured logs.

`data_classification` also sets the MLflow trace-capture boundary. Public and
internal applications default to bounded/redacted payload capture.
Confidential and restricted applications permit only metadata-only or disabled
tracing; application code cannot weaken that platform boundary with an
explicit policy. Metadata-only mode suppresses prompt and response content,
drops arbitrary SDK attributes, and reduces framework-owned attribute values
to shape-only metadata while retaining typed model/tool, lineage, token, and
cost evidence needed for operational reliability. Conversation/session IDs
are deterministically hashed under metadata-only or tracing-off policy, so
requests remain groupable without persisting the caller-supplied identifier.
The same policy drops arbitrary request metadata; only `request_id` and
`correlation_id` are retained, and both are deterministically hashed. Provider
and tool failures are re-raised to application code, but their original
messages and chained tracebacks never cross the governed MLflow span boundary.

`lifecycle` is deliberately a plain maturity tag, not a new name for the
engineering practice. The repository calls the wider practice the **AI
application lifecycle**. Use:

- `experimental` while exploring or establishing a baseline.
- `validation` while a proposed release is gathering release evidence.
- `production` only for a supported deployed application.
- `retired` when the resource remains for lineage but is no longer active.

Historical tag-schema-v1 evidence may contain `candidate`. The SDK keeps that
evidence readable and emits a deprecation warning; new configuration and
generated projects use schema 2 and `validation` without silently rewriting
historical records.

Do not introduce `AIMLOps` or another practice name as a lifecycle value. In
comparisons, call the variant being tested a **change**; after evaluation,
record an explicit `adopt`, `reject`, or `inconclusive` decision.

Application code cannot override controlled fields. Custom application tags
may be added only with new keys.

Tags are plain-text metadata. Never place secrets, tokens, prompts, user
content, personal email addresses, or other sensitive information in them.

## Enforcement

- Bundle presets tag jobs and pipelines.
- Cluster `custom_tags` support classic compute and Azure VM attribution.
- Serverless usage policies support serverless cost attribution.
- Unity Catalog governed tags enforce allowed ownership, classification, and
  lifecycle values on supported securables. The SDK uses matching closed
  `DataClassification` and `LifecycleStage` enums.
- CI validates mandatory bundle and compute tags.
- Generated starters keep ownership tags and the approved compute policy as
  rendered, cross-file-validated values rather than runtime bundle variables.
- Billing dashboards query `system.billing.usage.custom_tags`.
