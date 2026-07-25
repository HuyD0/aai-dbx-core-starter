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

variable "github_owner_id" {
  type        = string
  description = <<-EOT
    GitHub owner's IMMUTABLE numeric id. GitHub now mints OIDC subjects in the
    immutable form (repo:<owner>@<owner_id>/<repo>@<repo_id>:...). Get it from a
    failing azure/login "subject claim" log, or `gh api users/<owner> --jq .id`.
  EOT
}

variable "repo_id" {
  type        = string
  description = <<-EOT
    GitHub repository's IMMUTABLE numeric id. Get it from the OIDC "subject
    claim" log, or `gh api repos/<owner>/<repo> --jq .id`.
  EOT
}

variable "cicd_app_client_id" {
  type        = string
  description = <<-EOT
    AppId (client id) of the EXISTING app registration reused as the CI OIDC
    identity. This module does NOT own the app — it only attaches a new
    federated credential to it. Default is github-actions-dbx-platform.
  EOT
}

variable "dedicated_cicd_app_display_name" {
  type        = string
  description = "Display name for this repository's dedicated, secretless CI application."
  default     = "github-actions-aai-dbx-core-starter"
}

variable "project_name" {
  type        = string
  description = "Logical project/template name; used for tags and the RG name."
  default     = "aai-dbx-base-template"
}

# Cost-attribution tags. Mirror the Databricks bundle variables of the same
# names (databricks.yml) so Azure resources and Databricks compute attribute to
# the same cost center/team. Set per instance in terraform.tfvars.
variable "cost_center" {
  type        = string
  description = "Finance cost center these resources are charged to. Set per instance."
  default     = "CC-1234"
}

variable "team" {
  type        = string
  description = "Team that owns these resources. Set per instance."
  default     = "data-platform"
}

variable "owner" {
  type        = string
  description = "Primary owner (email or alias) for these resources."
  default     = "huyydo@gmail.com"
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
