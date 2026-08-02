# Local curriculum sequence

## Phase 1 — Bitext structured support

Start with data provenance and immutable hashes, then inspect quality without
printing large raw text tables. Build the versioned portable splits and prove
that exact duplicates, inferred templates, near duplicates, target text, and
test demonstrations cannot cross evidence boundaries.

Run the two deterministic baselines before loading a model. Then compare the
same untouched model with a basic prompt, constrained prompt, and training-only
few-shot prompt. These are separate baseline runs—not informal prompt tweaks
made after looking at frozen test failures.

Run a short MLX-LM check, train the LoRA change, and evaluate all methods with
the framework-neutral scorer. Treat the successful-training manifest as an
immutable lineage handle: it binds the expected model, exact training files,
effective configuration, and adapter outputs, and the same fingerprint must
survive inference, tracking, and the decision. Required measures:

- intent accuracy, macro precision/recall/F1, weighted F1, and per-intent F1;
- category and escalation accuracy;
- JSON parse, strict schema validity, and unsupported-label rates;
- response-policy compliance reported separately from classification;
- latency, output tokens, and measured peak memory;
- intent, category, language-variation flag, ambiguity, and difficulty slices.

Inspect bounded masked errors. Adopt only if the change beats the strongest
meaningful baseline and clears every absolute gate.

## Phase 2 — Enterprise-style ticket routing

The adapter seam is documented, but this first release does not pretend that an
uninspected optional dataset is executable. After a fresh source/license/schema
review, start with English records and a limited queue/priority/type/review
contract. Add imbalance, source-aware splitting, queue/priority slices, and
language slices incrementally. Multilingual inference is an extension.

## Phase 3 — Application-readiness capstone

Generate reviewed domain data from a versioned deterministic policy engine, not
from an LLM. The engine owns ground-truth checks, severities, remediation IDs,
failure combinations, and rule provenance. External lookups and human judgment
route to review instead of becoming invented facts.

Compare the policy engine, untouched prompts, a fine-tuned model, and a hybrid
where deterministic checks remain authoritative and a model may improve
normalization or wording. Ask whether a model is necessary for each behavior.

## Phase 4 — One optional advanced project

Choose invoice extraction, prompt transformation, or MiniZinc generation only
after Phase 1. Their evaluation strategies differ:

- invoice: field exactness/F1, normalized money/date/currency, missing and
  hallucinated fields, schema and human-review routing;
- prompt transformation: component rubrics, human calibration, multiple valid
  outputs, and report-only judges until calibrated;
- MiniZinc: parsing, compilation, execution, constraints, objective values,
  test cases, runtime, and invalid code.

The dataset catalog is a review queue, not a production suitability claim.
