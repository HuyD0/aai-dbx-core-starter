# Both providers authenticate through your existing `az login` session
# (Azure CLI auth). No client secret is created, stored, or referenced anywhere.
provider "azuread" {
  tenant_id = var.tenant_id
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
  tenant_id       = var.tenant_id
}
