# Enterprise clone runbook

Use this checklist to connect a clone of this repository to resources managed
by another GitHub organization, Azure tenant, and Databricks workspace. The
repository does not provision those resources.

## 1. Capture the clone's immutable GitHub IDs

```bash
gh api users/<org> --jq .id
gh api repos/<org>/<repo> --jq .id
```

The federated credential subject embeds these numeric owner and repository
IDs. A clone cannot reuse the current repository's subject.

## 2. Request the external platform prerequisites

Ask the enterprise identity and platform owners for:

- a repository-specific Entra application and service principal;
- a `main` branch federated credential using the new immutable IDs;
- registration in the target Databricks workspace;
- `CAN_USE` on the approved job-compute policy;
- least-privilege access to the SDK artifact volume.

The identity must have no client secret and should have no Azure ARM role
unless an independently reviewed workload requirement needs one. Do not add
infrastructure provisioning to this repository or its CI.

## 3. Update repository identifiers

Edit `platform-identifiers.json` first:

- `azure_tenant_id`
- `azure_subscription_id`
- `databricks_host`
- `job_compute_policy_id`
- `sdk_artifact_volume`

Then update the human-readable table in `AGENTS.md` and the values identified
by the smoke tests:

- the literal dev workspace host and compute-policy default in
  `databricks.yml`;
- `workspace_host`, `compute_policy_id`, and `aai_core_volume` defaults in
  every `templates/*/databricks_template_schema.json`.

Run:

```bash
pytest -q tests/test_smoke.py
```

The identifier cross-checks report each remaining value that must agree.

## 4. Configure repository variables

Follow `docs/cloud-setup.md` with the enterprise client ID, tenant,
subscription, workspace, artifact path, and attribution values. Use GitHub
repository variables, never secrets.

## 5. Configure model access

Enterprise LLM access should flow through Azure API Management or Databricks AI
Gateway. This is application configuration in `aai-platform.yml`; application
code continues to use logical resource names.

- For Databricks AI Gateway, configure the logical model's
  `provider: databricks` deployment as the gateway-enabled serving endpoint.
- For Azure API Management, use `provider: azure_apim`, the enterprise
  `base_url`, and `token_scope`. If a subscription key is mandatory, store
  only a secret reference such as `keyvault://...` or
  `databricks-secret://...`.

## 6. Protect and verify

1. Protect `main` and require code-owner review.
2. Run `gh workflow run auth-smoke.yml --ref main`.
3. Run or merge into `deploy.yml`; a green deployment is the definitive
   authorization test.
4. Run `./scripts/cloud-verify.sh` for the credential-free local checks.

Each additional staging or production target needs its externally managed
federated credential and workspace registration before its GitHub environment
or deployment job is enabled.
