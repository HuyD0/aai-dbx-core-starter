# shellcheck shell=bash
#
# Export this environment's non-secret Databricks/Azure identifiers.
#
#     source scripts/platform-env.sh
#
# Every value comes from platform-identifiers.json, the single source a clone
# edits. Documentation sources this instead of restating literals, so a clone
# never has to find and edit a workspace host inside a code block — and
# tests/test_smoke.py enforces that by scanning every *.md for these values.
#
# This exports identifiers only. It performs no login and reads no credential;
# run `az login` yourself. DATABRICKS_AUTH_TYPE=azure-cli tells the Databricks
# CLI to use that Azure CLI session, which is why no token is ever needed.

_aai_identifier() {
    python3 -c "import json,sys; print(json.load(open('$1'))['$2'])"
}

_aai_platform_env() {
    local fixture="platform-identifiers.json"
    local root="${AAI_REPO_ROOT:-}"

    if [[ -z "$root" ]]; then
        # Works from any subdirectory of the checkout, and when sourced from a
        # shell whose cwd is elsewhere but BASH_SOURCE points into the repo.
        root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi

    if [[ ! -f "$root/$fixture" ]]; then
        echo "platform-env: cannot find $fixture (looked in $root)" >&2
        return 1
    fi

    DATABRICKS_HOST="$(_aai_identifier "$root/$fixture" databricks_host)"
    AZURE_TENANT_ID="$(_aai_identifier "$root/$fixture" azure_tenant_id)"
    AZURE_SUBSCRIPTION_ID="$(_aai_identifier "$root/$fixture" azure_subscription_id)"
    DATABRICKS_AUTH_TYPE="azure-cli"
    # Which repository `databricks bundle init` generates projects from. Exported
    # so documented commands never hard-code it: a clone whose docs still named
    # the upstream URL would send its developers upstream for every new project.
    AAI_TEMPLATE_REPO="$(_aai_identifier "$root/$fixture" template_repo)"
    export DATABRICKS_HOST AZURE_TENANT_ID AZURE_SUBSCRIPTION_ID DATABRICKS_AUTH_TYPE
    export AAI_TEMPLATE_REPO

    echo "platform-env: DATABRICKS_HOST=$DATABRICKS_HOST (auth: azure-cli)"
}

_aai_platform_env
