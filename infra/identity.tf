# This application is owned by this repository and has no client secret, API
# permissions, or Azure RBAC. It is registered only in dbx-dev and receives
# CAN_USE on the constrained Job Compute policy.
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
