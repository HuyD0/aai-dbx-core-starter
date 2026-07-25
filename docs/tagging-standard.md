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
| `data_classification` | Information handling category |
| `lifecycle` | Experimental, candidate, or production |
| `repository` | Source repository |
| `release` | Application release |
| `tag_schema_version` | Tag-contract version |

AAI Core projects one validated context to MLflow, traces, Databricks
resources, Azure resources, and structured logs.

Application code cannot override controlled fields. Custom application tags
may be added only with new keys.

Tags are plain-text metadata. Never place secrets, tokens, prompts, user
content, personal email addresses, or other sensitive information in them.

## Enforcement

- Bundle presets tag jobs and pipelines.
- Cluster `custom_tags` support classic compute and Azure VM attribution.
- Serverless usage policies support serverless cost attribution.
- Unity Catalog governed tags enforce allowed ownership, classification, and
  lifecycle values on supported securables.
- CI validates mandatory bundle and compute tags.
- Billing dashboards query `system.billing.usage.custom_tags`.
