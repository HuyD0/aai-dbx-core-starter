# ----------------------------------------------------------------------------
# Reused CI identity (keyless OIDC)
# ----------------------------------------------------------------------------
# We do NOT create or own the app registration. We reference the existing
# `github-actions-dbx-platform` app — a Databricks-only OIDC service principal
# with zero ARM RBAC — by its client id, and attach ONE new federated
# credential scoped to this repo's main branch.
#
#   * No client secret is ever created.
#   * `terraform destroy` removes ONLY this credential, never the shared app.
#   * The subject is the branch-ref form, so the deploy/smoke jobs must NOT set
#     a GitHub `environment:` (that would change the token subject and break the
#     exchange). To gate on an environment later, add a second FIC with subject
#     "repo:<owner>/<repo>:environment:<name>".
data "azuread_application" "cicd" {
  # azuread ~> 3.0 looks apps up by `client_id`.
  # azuread 2.x fallback: rename this argument to `application_id`.
  client_id = var.cicd_app_client_id
}

resource "azuread_application_federated_identity_credential" "gha_main" {
  # azuread ~> 3.0: application_id is the application RESOURCE id, of the form
  # "/applications/<object-id>". Constructing it from object_id avoids ambiguity
  # over what the data source's `.id` returns across provider patch versions.
  # azuread 2.x fallback: replace this line with
  #   application_object_id = data.azuread_application.cicd.object_id
  application_id = "/applications/${data.azuread_application.cicd.object_id}"
  display_name   = "gh-${var.repo_name}-main"
  description    = "GitHub Actions OIDC — ${var.github_owner}/${var.repo_name} push to ${var.main_branch}"
  audiences      = ["api://AzureADTokenExchange"]
  issuer         = "https://token.actions.githubusercontent.com"
  subject        = "repo:${var.github_owner}/${var.repo_name}:ref:refs/heads/${var.main_branch}"
}
