# UAT promotion

This generated project supports `dev -> UAT`. UAT carries lifecycle
`validation`; no production target is declared.

The workflow does not provision identity or infrastructure. Before setting
`UAT_DEPLOYMENT_ENABLED=true`, the platform and identity owners must:

1. Register the project's dedicated CI service principal in the UAT workspace
   with least-privilege compute, data, model, search, and application grants.
   Do not reuse or mutate a shared legacy principal.
2. Verify the existing `main` branch-ref federated credential has the immutable
   repository subject approved by the identity owner. The credentialed jobs
   deliberately have no GitHub `environment:` because no matching
   environment-subject credential has been verified.
3. Protect `main`, restrict workflow dispatch to trusted maintainers, and set
   the non-secret repository variable `DATABRICKS_UAT_HOST` to the exact UAT
   host recorded when this project was generated. Keep the repository-level
   `AZURE_CLIENT_ID` and `AZURE_TENANT_ID` values for the dedicated CI identity
   registered in both workspaces.
4. Externally create `<application>-uat` when this project includes a
   Databricks App, and grant the CI principal `CAN MANAGE`. CI may bind and
   update the App but may not create it or its service principal.
5. Set the repository variable `UAT_DEPLOYMENT_ENABLED=true` after review.

Promote with:

```bash
gh workflow run deploy.yml --ref main -f target=uat
```

For an App's one-time bundle binding:

```bash
gh workflow run deploy.yml --ref main -f target=uat \
  -f bind_app=<application>-uat
```

The workflow builds one wheel, records its source commit and SHA-256 digest,
deploys it to dev, runs the dev release gate, applies the manual UAT prerequisite
gate, verifies the same evidence, deploys to UAT, and reruns the release gate
there. The UAT runtime receives `environment=uat`, `lifecycle=validation`, and
the immutable source commit as its release. No artifact is rebuilt between
environments.

`UAT_DEPLOYMENT_ENABLED` is a reviewed enablement flag, not a substitute for
branch protection. Do not add a GitHub environment gate until its matching
environment-subject federated credential has been provisioned and verified
externally.

Rollback is another reviewed promotion of a known-good commit. Do not use
`bundle destroy` as rollback.
