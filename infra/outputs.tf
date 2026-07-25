output "cicd_client_id" {
  description = "Legacy shared client id currently used by CI during migration."
  value       = var.cicd_app_client_id
}

output "dedicated_cicd_client_id" {
  description = "Dedicated client id to set as AZURE_CLIENT_ID after Databricks registration."
  value       = azuread_application.dedicated_cicd.client_id
}

output "dedicated_cicd_sp_object_id" {
  description = "Dedicated Entra service-principal object id; use only for verification."
  value       = azuread_service_principal.dedicated_cicd.object_id
}

output "federated_credential_subject" {
  description = "The OIDC subject both migration credentials trust."
  value       = azuread_application_federated_identity_credential.dedicated_gha_main.subject
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
