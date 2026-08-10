# Contributing

Changes should preserve the repository's keyless identity model, native provider
escape hatches, and credential-free pull-request boundary. Read `AGENTS.md`
before changing authentication, deployment, dependencies, tags, tracing, or
release behavior.

## Local workflow

```bash
make install
make hooks-install
make check
make verify
```

`make check` runs scaffold drift, formatting, SDK type checks, tests, and the
wheel build. `make verify` is the complete credential-free CI path, including
coverage, workflow security lint, and template checks. Provider integrations
must retain deterministic fake-client tests; live checks belong only on the
protected credentialed path.

The installed pre-commit hook runs staged-whitespace, scaffold, format, type,
and non-generated test checks. The pre-push hook runs the complete offline
verification path, including every rendered-project combination. Hook
definitions are repository-local and never download third-party hook code.

## Change expectations

- Add behavior-focused tests for fixes and public behavior changes.
- Keep public SDK entry points small and document exported symbols.
- Prefer native Databricks, MLflow, and Foundry APIs over another abstraction
  layer. Add a shared abstraction only after at least two real consumers have
  the same stable contract.
- Keep synchronous batch templates synchronous. Async is required only at I/O
  concurrency or streaming boundaries, with deadlines and deterministic close.
- Update dependency policy, exact locks, compatibility metadata, generated
  scaffold, and migration notes together when their contracts change.
- Never add credentials, personal data, prompts, or secret values to fixtures,
  tags, logs, traces, screenshots, or test output.

Pull requests should describe the user-visible outcome, compatibility impact,
verification performed, and any follow-up that is intentionally out of scope.
