# Remote state in the existing Terraform state account (house style).
#
# Prereqs (run once, from a shell that can reach the storage data plane):
#   az storage container create \
#     --account-name tfstatee18f8286 --name tfstate --auth-mode login
#   # You need "Storage Blob Data Contributor" on the account for use_azuread_auth.
#
# Prefer a quick local-state bootstrap instead? Delete this file, run
# `terraform init`, and state stays in infra/terraform.tfstate (git-ignored).
terraform {
  backend "azurerm" {
    resource_group_name  = "rg-terraform-state"
    storage_account_name = "tfstatee18f8286"
    container_name       = "tfstate"
    key                  = "aai-dbx-base-template/dev.tfstate"
    use_azuread_auth     = true
  }
}
