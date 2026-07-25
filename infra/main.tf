locals {
  tags = {
    project     = var.project_name
    managed_by  = "terraform"
    repo        = "${var.github_owner}/${var.repo_name}"
    template    = "aai-dbx-core-starter"
    cost_center = var.cost_center
    team        = var.team
    owner       = var.owner
  }
}

# Optional landing zone for future project Azure resources. Empty by design in
# the Databricks-only model: CI deploys bundles to an existing workspace and
# holds no rights here. Set create_project_rg = false to skip.
resource "azurerm_resource_group" "project" {
  count    = var.create_project_rg ? 1 : 0
  name     = "rg-${var.project_name}-dev"
  location = var.location
  tags     = local.tags
}
