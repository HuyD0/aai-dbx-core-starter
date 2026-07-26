# Enterprise clone runbook

Ordered checklist to stand this repository up in another GitHub org and Azure
tenant (for example, cloning the personal proving ground into the enterprise).
Nothing in this flow creates or stores a secret; the clone re-establishes the
same keyless chain against the new tenant.

**Why identity cannot be reused:** the federated credential subject embeds the
*immutable* GitHub owner and repo ids
(`repo:<owner>@<owner_id>/<repo>@<repo_id>:ref:refs/heads/main`). A clone has
new ids, so the existing app registration and FIC are useless to it — the
bootstrap must be re-run, which is by design (AGENTS.md rule 8: bootstrap is
human-run).

Estimated effort once prerequisites are met: under an hour of hands-on work.

---

## 1. Clone and capture the new immutable ids

```bash
# [you] after creating <org>/<repo> in the enterprise
gh api users/<org>  --jq .id          # or orgs/<org> for an organization
gh api repos/<org>/<repo> --jq .id
```

## 2. Edit the identifier sources (two files + one table)

1. `platform-identifiers.json` — tenant id, subscription id, workspace host,
   job compute policy id, SDK artifact volume. **Edit this first**; the smoke
   tests then fail on every other file that must agree until step 3 is done.
2. `infra/terraform.tfvars` — `tenant_id`, `subscription_id`, `github_owner`,
   `repo_name`, `github_owner_id`, `repo_id` (from step 1), app display name,
   cost tags, region.
3. `AGENTS.md` §3 — the human-readable identifier table.

Also update, as pointed out by the failing cross-check tests:

- `databricks.yml`: `targets.dev.workspace.host` (must be a literal — the CLI
  forbids variables in authentication fields) and the
  `job_compute_policy_id` variable default.
- `templates/agentic-rag/databricks_template_schema.json`: `workspace_host`,
  `compute_policy_id`, and `aai_core_volume` defaults.
- `infra/backend.tf`: enterprise state storage account/container/key — or
  delete the file to bootstrap with local state.

Run `pytest -q tests/test_smoke.py` until green: the identifier fixture test
enumerates anything still inconsistent.

## 3. Bootstrap the OIDC identity (human-run Terraform)

Follow `docs/cloud-setup.md` §2 against the enterprise tenant: `terraform
init` (multi-platform provider lock), `plan`, `apply`. Output: one secretless
app registration + one federated credential whose subject uses the new
immutable ids. Verify with `az ad app federated-credential list`.

## 4. Register the CI service principal in the enterprise workspace

Follow `docs/cloud-setup.md` §3 with the enterprise workspace host and the new
client id: create the SP in the workspace, grant `CAN_USE` on the enterprise
job-compute policy (the id you put in `platform-identifiers.json`), and grant
`READ VOLUME`/`WRITE VOLUME` on the enterprise SDK artifact volume. Do not
grant admin or cluster-create.

## 5. Set repository variables (non-secret)

Follow `docs/cloud-setup.md` §4 substituting the enterprise values:
`AZURE_CLIENT_ID` (new app), `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`,
`DATABRICKS_HOST`, `SDK_ARTIFACT_VOLUME`, `COST_CENTER`, `TEAM`,
`OWNER_GROUP`. No `gh secret set`, ever.

## 6. Wire model access through the enterprise gateway

The enterprise reaches LLMs through Azure APIM or Databricks AI Gateway —
never direct Foundry. This is configuration only (`aai-platform.yml` in
consuming projects); application code keeps using logical names:

- **Databricks AI Gateway**: point the logical model's `provider: databricks`
  `deployment` at the gateway-enabled serving endpoint name. Nothing else
  changes.
- **Azure APIM**: use `provider: azure_apim` with the APIM `base_url`, the
  enterprise `token_scope` (the APIM app registration audience), and — only if
  the enterprise mandates subscription keys for chargeback — a
  `subscription_key` secret *reference* (`keyvault://…` or
  `databricks-secret://…`), never a value.

See `aai-platform.example.yml` for the same logical name resolved all three
ways (direct dev, APIM, gateway-fronted serving endpoint).

## 7. Protect `main`, then verify end-to-end

1. Apply branch protection + CODEOWNERS per `docs/cloud-setup.md` §8.1 (and
   enterprise org policy: secret scanning, push protection).
2. `gh workflow run auth-smoke.yml --ref main` — proves the OIDC exchange.
3. Merge to `main` (or dispatch `deploy.yml`) — the `deploy` job going green is
   the definitive authorization test.
4. `./scripts/cloud-verify.sh` locally for the offline checks.

## Enterprise deltas

- **Multiple workspaces (staging/prod)**: each additional target needs its own
  federated credential first (environment-subject form), then the GitHub
  protected environment, then the target in `databricks.yml` — in that order.
  See `docs/cloud-setup.md` §7.
- **Enterprise Key Vaults / AI Search**: `aai-platform.yml` entries only
  (secret *references* and endpoint identifiers). The SDK is logical-name
  based; no code changes.
- **Sandboxed shells**: if `databricks` CLI calls fail on TLS while `az`
  works, the network blocks `*.azuredatabricks.net`; use a GitHub runner
  (AGENTS.md §7).
