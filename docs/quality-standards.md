# SDK and template quality standard

This document is the acceptance contract for `aai-core`, generated projects,
and examples. The platform console is intentionally outside this quality
initiative except where shared release or security boundaries require it.

## Required gates

| Area | Required evidence |
|---|---|
| SDK correctness | Python 3.11 and 3.12 tests; an enforced 86% combined line/branch coverage ratchet, raised toward 90% before 1.0; focused security, cancellation, concurrency, and cleanup tests |
| Types and readability | Mypy on the SDK; Ruff complexity ceiling 12 for new or changed functions; documented exported APIs; no reusable business logic hidden in notebooks |
| Dependencies | Exact root and template locks, `pip check`, advisory audit, and minimum/latest compatibility canaries |
| Static security | CodeQL Python and GitHub Actions analysis with the `security-extended` and `security-and-quality` suites; SHA-pinned actions; credential-free pull requests |
| Templates | Every meaningful provider/feature render shape parses and validates; generated tests run against the built SDK wheel rather than the source checkout |
| Streaming | Ordered deltas, bounded output and memory, slow-consumer backpressure, caller cancellation, deadline propagation, and deterministic stream/client cleanup |
| Remote analytics work | Structured allowlisted query plans, parameterized values, whole-request and statement deadlines, bounded concurrency, and explicit remote cancellation |
| Examples | Every offline path executes without network access; connected paths have fake-provider contract tests and a small protected live canary |

Coverage is a regression signal, not a substitute for assertions. Tests should
prefer observable behavior and boundary contracts over source-text matching.
Any temporary exception needs a narrow scope, an owner, and a removal condition.

The SDK ratchet is 86%, below the measured 86.27% baseline in the root
SDK-measuring test process with the locked `dev` and `all` extras installed;
`make coverage` and the mandatory cloud gate reproduce that environment.
Generated-project coverage runs separately. SDK maintainers own the ratchet:
coverage may not fall, and the floor must increase whenever new tests move the
stable baseline across the next whole percentage point. The exception ends at
90% or the 1.0 release, whichever comes first. This avoids presenting an
aspirational number as a passing control.

Generated projects use their checked-in `.coveragerc` as the single source for
their own branch-coverage floor. The initial measured floors are agent 85%,
analytics 81%, evaluation 84%, experiment 93%, prompt 96%, and RAG 86%; every
meaningful render is tested against its template's floor. Template maintainers
own the same non-regression rule: advance a floor at the next whole percentage
point and bring every template to at least 90% before the SDK 1.0 release. A
single global floor is deliberately avoided because it would either fail honest
projects or silently weaken the stronger ones.

The complexity ceiling applies to reusable logic and decision-bearing runtime
paths. A function-level exception is allowed for a linear validation
orchestrator whose branches only append independent failures and whose complete
behavior is exercised by tests. Refactor the exception before any branch shares
mutable state, becomes reusable, or introduces nested operational control flow.

## Architecture guardrails

Use provider-neutral contracts only for capabilities the SDK actually owns.
Keep native Databricks, MLflow, OpenAI, Azure AI Search, and Foundry clients
available when their APIs are the clearer boundary.

The following are deliberately not part of the architecture:

- a dependency-injection container, ORM, rules DSL, or event bus;
- a universal async or streaming wrapper over unlike provider APIs;
- async conversions for batch-only prompt, RAG build, experiment, or
  evaluation jobs;
- a shared runtime package for generated applications with independent release
  lifecycles;
- repository-specific copies of vendor documentation or agent frameworks.

Create a shared helper when two production consumers need the same semantics
and the helper removes more policy drift than it adds indirection. Keep small
workload-specific controls with the generated application.

## Databricks, MLflow, and Foundry conventions

- Databricks bundles own logical resource configuration; infrastructure and
  identity remain external. Use serverless policy IDs and required cost tags.
- MLflow 3 traces, runs, datasets, prompt versions, Feedback, and Assessments
  remain the evidence system. Do not invent parallel evidence objects.
- Agent serving keeps MLflow `ResponsesAgent` wire contracts and native async
  streaming ownership.
- Foundry examples use maintained Azure SDKs and the repository's opt-in
  curriculum. Preview A2A and telemetry features remain explicitly optional,
  bounded, and cleanup-aware.

## Release completion

A release is ready only after the root workflow, CodeQL, provider/Foundry lanes,
template render matrix, wheel validation, and relevant credentialed Databricks
validation are green. The most recent manually dispatched dependency canary for
the frozen candidate must also pass all four Python 3.11/3.12 × lowest/latest
lanes; a previous-version or partially green run is not release evidence.
Publishing must consume the same verified wheel and must never overwrite an
existing version. A commit-pinned generated-project source is release-candidate
CI evidence only, never evidence of a published runtime artifact.
