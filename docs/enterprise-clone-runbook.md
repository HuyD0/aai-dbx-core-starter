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

## 6. Decide whether to run the platform console

The guided console (`src/platform_app`) is **optional and off by default in a
clone**. Its bundle resource lives at `resources/optional/platform_console.yml`,
deliberately outside the `resources/*.yml` glob in `databricks.yml`, so a clone
deploys nothing app-related until you opt in with an explicit `include:`.

Leave it off unless all of the following hold in the target workspace:

- Databricks Apps is enabled.
- The workspace setting *"Only allow app deployments from Git"* is **off**
  (otherwise a workspace-folder `source_code_path` deploy fails and the resource
  needs `git_source` instead).
- Your platform identity owner has created the app out-of-band and granted the
  CI principal `CAN MANAGE` on it. Creating an app auto-provisions a workspace
  service principal, which AGENTS.md section 4 rule 8 reserves for the human-run
  platform process — CI must only ever update an existing app.
- You have a serverless usage policy id for cost attribution. An app resource
  has no `tags` field, so `app_usage_policy_id` is how its spend is attributed.

The exact grant request to send is in `docs/platform-console.md`. Note the
console bills continuously while running and is stopped by default; use
`make app-start` / `make app-stop`.

## 7. Protect and verify

1. Protect `main` and require code-owner review.
2. Run `gh workflow run auth-smoke.yml --ref main`.
3. Run or merge into `deploy.yml`; a green deployment is the definitive
   authorization test.
4. Run `./scripts/cloud-verify.sh` for the credential-free local checks.

Each additional staging or production target needs its externally managed
federated credential and workspace registration before its GitHub environment
or deployment job is enabled.
