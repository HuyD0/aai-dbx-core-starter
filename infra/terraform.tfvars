# Concrete values for this instance of the template.
# All non-secret identifiers — safe to commit.

tenant_id       = "7f6a2cf9-5e4e-46ae-95d4-74016c1df1a6"
subscription_id = "ea936670-dda1-4884-8467-49c225bf3e83"

github_owner = "HuyD0"
repo_name    = "aai-dbx-core-starter"
# Immutable numeric ids — GitHub mints OIDC subjects in the immutable form here.
# Source: the azure/login "subject claim" log, or
#   gh api users/HuyD0 --jq .id            -> 151226205
#   gh api repos/HuyD0/aai-dbx-core-starter --jq .id  -> 1311037530
github_owner_id = "151226205"
repo_id         = "1311037530"

# Legacy shared app registration. Retained only until the dedicated identity
# below is applied, registered in dbx-dev, and verified.
cicd_app_client_id = "b74a6820-d0ac-454f-8c32-02141cba3c8a"

dedicated_cicd_app_display_name = "github-actions-aai-dbx-core-starter"

project_name      = "aai-dbx-base-template"
location          = "eastus2"
create_project_rg = true
main_branch       = "main"
