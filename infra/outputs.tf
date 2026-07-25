output "cicd_client_id" {
  description = "Client id CI uses for azure/login — set as repo variable AZURE_CLIENT_ID."
  value       = var.cicd_app_client_id
}

output "federated_credential_subject" {
  description = "The OIDC subject this credential trusts. Must match the token GitHub mints."
  value       = azuread_application_federated_identity_credential.gha_main.subject
}

output "tenant_id" {
  value = var.tenant_id
}

output "subscription_id" {
  value = var.subscription_id
}

output "project_resource_group" {
  description = "Optional project RG name, or null when create_project_rg = false."
  value       = var.create_project_rg ? azurerm_resource_group.project[0].name : null
}
