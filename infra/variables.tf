variable "tenant_id" {
  type        = string
  description = "Entra ID (Azure AD) tenant id."
}

variable "subscription_id" {
  type        = string
  description = "Azure subscription id for the optional project resource group."
}

variable "github_owner" {
  type        = string
  description = "GitHub org/user that owns the repository (case-sensitive)."
}

variable "repo_name" {
  type        = string
  description = "GitHub repository name, without the owner."
}

variable "cicd_app_client_id" {
  type        = string
  description = <<-EOT
    AppId (client id) of the EXISTING app registration reused as the CI OIDC
    identity. This module does NOT own the app — it only attaches a new
    federated credential to it. Default is github-actions-dbx-platform.
  EOT
}

variable "project_name" {
  type        = string
  description = "Logical project/template name; used for tags and the RG name."
  default     = "aai-dbx-base-template"
}

variable "location" {
  type        = string
  description = "Azure region for the optional project resource group."
  default     = "eastus2"
}

variable "create_project_rg" {
  type        = bool
  description = <<-EOT
    Create an (empty, tagged) resource group as a home for future project Azure
    resources. In the Databricks-only model nothing is deployed here and CI has
    no access to it. Set to false to skip entirely.
  EOT
  default     = true
}

variable "main_branch" {
  type        = string
  description = "Branch whose pushes are allowed to assume the CI identity."
  default     = "main"
}
