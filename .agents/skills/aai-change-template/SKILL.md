---
name: aai-change-template
description: Safely create or modify the repository's Databricks project templates, shared scaffold, bundle resources, generated Python, template locks, and agent, analytics, prompt, RAG, experiment, or evaluation recipes. Use for work under templates, template schemas, synchronization tooling, render matrices, or generated-project contracts.
---

# Change an AAI Template

Keep generated projects portable, independently operable, and aligned with the
candidate `aai-core` wheel.

## Workflow

1. Read `AGENTS.md`, the target schema and README,
   `templates/_shared/manifest.json`, and its cases in
   `tests/template_matrix.py`. Preserve unrelated work.
2. Decide whether the behavior is shared or template-specific:
   - Put genuinely shared files in the canonical shared scaffold and manifest.
   - Keep application runtime behavior inside that template.
   - Do not couple generated applications through a new shared runtime package.
3. Implement production logic under generated `src/`; keep notebooks thin and
   instructional. Preserve native provider escape hatches.
4. Render every affected schema branch. Verify conditional files are both
   present and absent in the intended combinations.
5. Run `make sync-templates` only after changing canonical shared content, then
   review every generated diff. Run `make check-templates` in all cases.
6. Run focused template tests, then `make check`. Use `make validate-templates`
   only with explicit authorization and an existing keyless workspace session.

## Generated-project contract

- Import a pinned compatible `aai-core` version and install the validated wheel.
- Include unit tests, evaluation data, an evaluation gate, bundle resources,
  keyless setup guidance, and the complete platform tag set.
- Keep logical resource names in application code and physical names in
  environment configuration.
- Bound requests, output, tokens, tool calls, retrieval context, concurrency,
  and whole-operation time. Propagate cancellation and close owned resources.
- Never expose model-authored raw SQL. Compile typed semantic queries with
  allowlisted identifiers and parameterized values.
- Treat retrieved text as untrusted data, preserve authorization filters, and
  cite only documents actually retrieved.
- Never provision identities, catalogs, volumes, endpoints, indexes, or other
  external infrastructure from a template or its CI.
- Pin every generated GitHub Action to a full commit SHA.

## Provider-specific work

Use the maintained `azure-ai` skill when current Azure AI behavior is
material. Apply this repository's portability, evidence, and security rules
around that provider guidance; do not copy provider manuals into this skill.

## Handoff

Report affected render combinations, generated contracts, tests run, lock or
compatibility changes, and credentialed validation still pending. Do not deploy
or mutate cloud resources unless explicitly authorized.
