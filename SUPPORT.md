# Support

This repository supports the `aai-core` SDK, generated project templates, and
the documented credential-free examples on the Python and dependency versions
listed in `compatibility.json`.

Before opening a support request:

1. Run `aai-core doctor` for local configuration, then `aai-core doctor --cloud`
   only from an approved keyless environment.
2. Run `make verify` in this repository, or `make check` in a generated project.
3. Capture the failing command, Python version, SDK/template version, provider,
   sanitized exception type, and whether the issue is local or connected.

Use a GitHub issue for reproducible defects and documentation gaps. Do not post
credentials, tokens, prompt or response content, customer data, full trace
payloads, or environment dumps. Use the private security-reporting path in
`SECURITY.md` when a report could expose a vulnerability or sensitive data.

Cloud resource provisioning, identity grants, quota, and workspace permissions
are owned by the external platform process described in `docs/cloud-setup.md`;
the repository intentionally does not grant or mutate them.
