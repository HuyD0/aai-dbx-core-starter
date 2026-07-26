# Secrets and identity

## Preference order

1. Managed or workload identity.
2. Databricks unified authentication.
3. Unity Catalog service credentials.
4. Azure Key Vault for credentials that cannot be replaced by identity.

Secret values must not appear in GitHub secrets for this repository, source,
bundle variables, notebooks, tags, MLflow parameters, traces, or logs.

## SDK usage

Configuration stores references:

```yaml
secrets:
  vendor_key: keyvault://application-vault/vendor-api-key
```

Resolution is explicit:

```python
value = ctx.secrets.resolve(ctx.settings.secrets["vendor_key"])
native_client = VendorClient(api_key=value.reveal())
```

`SecretValue` always renders as `[REDACTED]`, and resolution registers the raw
value with the process log redactor.

## Key Vault-backed Databricks scopes

These scopes are read-only views over a vault and currently require the Key
Vault access-policy permission model. A principal with access to the scope can
access every secret in the backing vault. Use separate vaults for separate
application or trust boundaries.

## Gateway authentication

Enterprise LLM traffic goes through Azure APIM or Databricks AI Gateway. Both
stay keyless by default:

- `provider: databricks` uses Databricks unified authentication; AI Gateway
  features on the serving endpoint need nothing extra from the application.
- `provider: azure_apim` sends an Entra bearer token for the gateway's
  configured `token_scope` (pair with APIM's `validate-azure-ad-token`
  policy). If the enterprise additionally requires a per-team subscription
  key for chargeback, configure it as `subscription_key:
  keyvault://…`/`databricks-secret://…` — a raw value is rejected and never
  echoed into errors or logs.

## Production policy

`azure_identity: auto` is rejected for test, staging, and production settings.
Select `workload_identity` or `managed_identity` so an accidental local
credential cannot become the production identity. The configured mode is
honored everywhere a credential is built — provider clients and the Key Vault
secret provider alike.

For local development, `AZURE_TOKEN_CREDENTIALS=dev` (azure-identity ≥ 1.23)
constrains the `auto` chain to developer credentials, keeping the fallback
order deterministic.

Vault, role, credential, and scope provisioning remains human-reviewed
infrastructure work; application code only consumes references. Key Vaults
should use the RBAC permission model (the API default from 2026-02-01);
never switch a shared vault to access policies for a Databricks scope.
