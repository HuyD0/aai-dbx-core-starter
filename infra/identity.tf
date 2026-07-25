# ----------------------------------------------------------------------------
# Dedicated CI identity (target state)
# ----------------------------------------------------------------------------
# This application is owned by this repository and has no client secret, API
# permissions, or Azure RBAC. During migration it exists alongside the legacy
# shared identity below so the currently working deployment is not interrupted.
#
# After a human applies this configuration, register ONLY this service
# principal in dbx-dev, grant it CAN_USE on the constrained Job Compute policy,
# update AZURE_CLIENT_ID, and verify deployment before removing the legacy FIC.
resource "azuread_application" "dedicated_cicd" {
  display_name            = var.dedicated_cicd_app_display_name
  sign_in_audience        = "AzureADMyOrg"
  prevent_duplicate_names = true
}

resource "azuread_service_principal" "dedicated_cicd" {
  client_id       = azuread_application.dedicated_cicd.client_id
  account_enabled = true
}

resource "azuread_application_federated_identity_credential" "dedicated_gha_main" {
  application_id = azuread_application.dedicated_cicd.id
  display_name   = "gh-${var.repo_name}-main"
  description    = "GitHub Actions OIDC — ${var.github_owner}/${var.repo_name} push to ${var.main_branch}"
  audiences      = ["api://AzureADTokenExchange"]
  issuer         = "https://token.actions.githubusercontent.com"
  subject        = "repo:${var.github_owner}@${var.github_owner_id}/${var.repo_name}@${var.repo_id}:ref:refs/heads/${var.main_branch}"
}

# ----------------------------------------------------------------------------
# Legacy shared CI identity (temporary migration compatibility)
# ----------------------------------------------------------------------------
# We do NOT create or own the app registration. We reference the existing
# `github-actions-dbx-platform` app — a Databricks-only OIDC service principal
# with zero ARM RBAC — by its client id, and attach ONE new federated
# credential scoped to this repo's main branch.
#
#   * No client secret is ever created.
#   * `terraform destroy` removes ONLY this credential, never the shared app.
#   * Subject uses GitHub's IMMUTABLE id form
#     (repo:<owner>@<owner_id>/<repo>@<repo_id>:ref:refs/heads/<branch>), because
#     that is what this account's runner presents in the OIDC token. Azure does
#     an EXACT match — the readable form will NOT match here.
#   * It is a branch-ref subject, so the deploy/smoke jobs must NOT set a GitHub
#     `environment:` (that would change the token subject to :environment:<name>
#     and break the exchange). To gate on an environment later, add a second FIC
#     with subject
#     "repo:<owner>@<owner_id>/<repo>@<repo_id>:environment:<name>".
#
# Remove this data source and FIC in a follow-up Terraform apply ONLY after the
# dedicated identity is verified. Never delete the shared application or SP.
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
  # Immutable subject form. Must match the token's `sub` claim EXACTLY.
  subject = "repo:${var.github_owner}@${var.github_owner_id}/${var.repo_name}@${var.repo_id}:ref:refs/heads/${var.main_branch}"
}
