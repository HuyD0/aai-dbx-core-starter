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

## Production policy

`azure_identity: auto` is rejected for test and production settings. Select
`workload_identity` or `managed_identity` so an accidental local credential
cannot become the production identity.

Vault, role, credential, and scope provisioning remains human-reviewed
infrastructure work; application code only consumes references.
