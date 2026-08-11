# Security policy

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. In the current clone,
open **Security → Advisories → New draft security advisory** and include a
minimal reproduction, affected version, impact, and suggested mitigation. Use
synthetic values only; never include a real secret, customer payload, access
token, or live trace. If private reporting is disabled, contact the owning
platform security team through its approved internal channel rather than using
the upstream repository.

## Supported versions

The versions and Python runtimes currently receiving fixes are declared in
`compatibility.json`. Pre-1.0 compatibility follows `docs/versioning.md`.

## Security boundaries

- Pull requests are credential-free.
- Protected `main` and repository-specific OIDC are the deployment boundary.
- Provider identity and cloud infrastructure are provisioned externally.
- SDK and template releases are immutable and dependency-locked.
- Secret values must not appear in logs, exceptions, traces, tags, or MLflow
  parameters.

See `AGENTS.md` for the enforceable repository rules and
`docs/secrets-and-identity.md` for the threat model and operational guidance.
