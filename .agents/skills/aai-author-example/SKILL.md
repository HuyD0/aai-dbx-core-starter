---
name: aai-author-example
description: Create or revise executable AAI learning examples, notebooks, workshops, and local-classification lessons. Use for work under examples that teaches SDK, MLflow, Databricks, agentic, RAG, evaluation, streaming, or classical-ML lifecycle practices.
---

# Author an AAI Example

Produce a complete reference lesson that is readable first and reproducible by
automation. Keep exercises separate from the canonical solution.

## Workflow

1. Read `AGENTS.md`, `examples/README.md`, the neighboring lessons, the example
   runner catalog, and affected tests. Preserve the numbered learning sequence.
2. Define lesson metadata: audience, level, prerequisites, duration, execution
   mode, objectives, evidence, cleanup, and next lesson.
3. Put reusable mechanics in a typed support module. Keep notebook cells focused
   on one teaching step; target at most 25 nonblank lines for normal cells and 40
   for setup cells when that improves comprehension.
4. Make the canonical numbered lesson complete and runnable. Put learner TODOs
   and `NotImplementedError` only in a clearly separate workshop copy, with a
   tested reference solution.
5. Keep the default path credential-free, deterministic, output-free, and safe
   to rerun. Label synthetic latency, token, cost, and quality values as
   `simulated_offline_fixture`.
6. Compile and execute the affected offline path from clean state. Run the
   focused example tests and repeat the execution to detect hidden state.

## Teaching guardrails

- Introduce one concept at a time in early lessons; reserve production limits,
  cleanup, cancellation, adversarial cases, and evaluation depth for advanced
  lessons.
- Use fixed ordered cases, stable IDs, explicit digests, and
  `baseline -> change -> result -> decision` evidence.
- Keep offline fixtures distinct from provider performance evidence.
- Default every connected switch to `False`; require configured keyless
  identity and pre-provisioned resources. Never provision cloud resources.
- Close streams and clients in `finally`, propagate cancellation, and set a
  whole-operation deadline in async lessons.
- Never teach static secrets, hidden chain-of-thought capture, model-authored raw
  SQL, mutable prompt aliases as lineage, or unreviewed traces as ground truth.
- Keep notebooks free of stored output, duplicate IDs, and hidden execution
  order. A restart-and-run-all path must work.

## Domain routing

Use the maintained `azure-ai` skill for current Azure AI Search behavior. Add
only this repository's lesson, governance, portability, and evidence rules
around it.

## Verification

Run the affected subset of:

```text
uv run pytest tests/test_smoke.py tests/test_examples_runner.py -q
make -C examples/local-classification check
```

Finish with `make check` when the lesson is ready for review. Report which paths
were executed offline and which connected behavior remains protected-canary
work.
