# Governed batch inference

A reference implementation of batch LLM inference with `ai_query` that treats
the result the way it deserves to be treated: as **a model deployment wearing a
SQL query's clothes**. It produces a derived dataset other things consume, so
it gets the controls a deployment would get — a declared spec, a cost decision
before execution, a stratified sampled gate, provenance that survives joins,
and monitoring — without over-controlling exploratory work.

```text
DECLARE → ESTIMATE → SAMPLE → EVALUATE → GATE → EXECUTE → LAND → MONITOR
```

## What is here

| File | Role |
|---|---|
| `governed_batch_inference.py` | The reusable module: `BatchInferenceSpec` (strict, frozen, YAML-round-trip), Wilson intervals, stratified sample allocation, per-field per-stratum precision/recall scoring, the gate, cost estimation, and SQL builders for execution and provenance. Pure Python — no Spark or MLflow imports at module scope — so every statistical claim is unit-testable. |
| `example_notebook.py` | A Databricks notebook (`# COMMAND ----------` source format) running the whole pattern end to end on synthetic T4/T5/1099-DIV/K-1 documents, teaching the reasoning at each stage. |
| `../../tests/test_governed_batch_inference.py` | Unit tests for the parts most likely to be quietly simplified later: Wilson lower bounds, allocation floors, the worst-stratum rule, tier-1 sign-off. |

## Running the notebook

Import the directory into a Databricks workspace (Repos or workspace files) and
open `example_notebook.py` on a recent runtime (15.4+ or serverless) with Unity
Catalog. Set the widgets:

- `catalog` / `schema` — somewhere you can create tables.
- `inference_mode` — `simulated` (default) runs with **no endpoint and no
  cost**: a deterministic seeded extractor stands in for the model so the
  statistical lessons reproduce exactly. `live` calls `ai_query` against the
  endpoint named in the spec; the cost estimate stage gates the spend first.

The demonstration is built around a corpus whose minority stratum
(`legacy_scan`, ~2% of rows) is genuinely harder. With prompt v1, a naive
proportional sample scored in aggregate **passes** every declared tolerance
while the stratified gate **rejects** on the worst stratum — the clustered
failure this pattern exists to catch. Prompt v2 (targeted guidance plus an
abstention path) then passes the same gate on the same terms.

## The rules the implementation refuses to bend

1. **Tolerances are declared before results exist.** The gate reads them from
   the spec, which is authored, reviewed, and digested first. A threshold set
   after looking at output is not a threshold.
2. **The gate compares the Wilson interval's lower bound to the tolerance —
   never the point estimate.** 97/100 correct *fails* a 95% tolerance: with
   n=100 the lower bound is ≈0.915. Gating on point estimates is the most
   common silent failure of this pattern.
3. **`criticality: high` fields gate on the worst-performing stratum, never
   the weighted average.** Aggregate accuracy is irrelevant when the failures
   concentrate in the one segment that matters.
4. **Precision and recall are reported separately.** Extraction fails two
   ways — hallucinating and missing — and a single accuracy figure hides
   whichever one your consumers care about. Abstention hurts recall, never
   precision: that asymmetry is what makes the abstention path the single
   highest-value change to an extraction pipeline.
5. **Insufficient evidence is `inconclusive`, not `reject` — and not a pass.**
   If even a flawless sample of the stratum's size could not clear the bar
   (lower bound of n/n caps at n/(n+z²)), the verdict is "label more rows".
6. **Tier 1 cannot be auto-approved.** A fully passing result returns
   `pending_approval`; a named human accepts residual risk via
   `approve_gate`, abstentions get human review, and a rollback path is on
   file. Approval can never resurrect a rejection.
7. **Execution is refused without an adopting gate for the exact spec
   digest** (`require_executable`), and the run is idempotent: an anti-join
   selects only unlanded rows, so a partial failure restarts by re-running
   the same statement.
8. **Evidence belongs to the release that produced it.** Scores carry a
   release stamp (spec digest, model version, prompt version) and the
   declared confidence level; the gate refuses anything else. Re-using the
   last passing evaluation for a changed prompt — "we tested this, it was
   fine" — is exactly the shortcut this blocks. Re-scoring is arithmetic
   over records you already hold, so the strict rule is cheap to satisfy.
9. **A new release reprocesses the table.** The restart anti-join matches
   on the key *and* the model and prompt versions, and the write is a
   `MERGE`. Matching on the key alone would let a newly gated release
   report success while every row still carried the previous release's
   values and provenance.

## Adapting it to a new use case

Work through the spec first — most adaptation is spec, not code:

- **Fields and tolerances.** Set `tolerable_error_rate` per field *with the
  consumers named in `consumed_by`*, not alone at a keyboard. Then check the
  feasibility number `min_labelled_rows_for_tolerance` prints: a 1% tolerance
  demands ≥381 flawless labelled rows per gated group — if the labelling
  budget cannot support the tolerance, that conversation happens now, not
  after a mystery `inconclusive`.
- **Strata.** Choose columns your failure modes actually track (document
  type, source system, format vintage, language). Run the probe, look at
  where errors cluster, and revisit at every re-sample. Strata multiply the
  labelling floor: budget ≈ floors × strata count.
- **Use tier.** Set by what consumes the output, not table size. Tier 3
  (exploratory, notebook-only) gets cost tracking and *no* gate — if the
  governed path is annoying for throwaway work, people will export data to a
  chat tool and you will have traded a tracked spend problem for an untracked
  egress problem. Tier 1 additionally needs a rollback plan and named
  approver — the spec model enforces both.
- **Matching.** `values_match` is deliberately simple (numeric-aware, case
  and whitespace tolerant). Real fields need real comparators — dates,
  currency codes, name normalisation. Keep them deterministic; scoring must
  be reproducible.
- **The prompt and the schema.** `response_format` builds the structured
  output schema from the spec, nullable via `[type, "null"]` (the only union
  Databricks structured outputs support — no `anyOf`). Keep the abstention
  instruction in the prompt aligned with `abstain_threshold`.
- **Gold labels are an asset.** Version the adjudicated sample, reuse and
  extend it at every re-evaluation and scheduled re-sample instead of paying
  the labelling cost again.

## Supporting controls outside the pipeline

The pipeline binds only if the platform does:

- Restrict endpoint creation and `CAN MANAGE` to admins; grant query
  permission on approved endpoints only. A gateway is only binding if people
  cannot create their own endpoint around it.
- Make the default endpoint a small model; wanting a frontier model for a
  50-million-row scan should require asking — that is the conversation you
  want. This repository's rules already require keyless identity and
  least-privilege grants (see `AGENTS.md`); batch inference adds no secrets.
- Serverless budget policies and spend alerting on the compute; costs land in
  `system.billing.usage` under `MODEL_SERVING` / offering type
  `BATCH_INFERENCE`. Query usage *first* to learn whether you have a real
  problem or two people experimenting.
- Add a batch inference section to the design review checklist — today it
  falls in the gap between "model" and "data pipeline", which is why nobody
  gets asked about it.
- Tag every scheduled job cluster with the platform tag set from
  `docs/tagging-standard.md`, values via bundle variables.

## Relationship to the rest of this repository

The module is deliberately standalone (pydantic + PyYAML + stdlib) so a team
can copy this directory into any Databricks project. It speaks the platform's
decision vocabulary — `adopt` / `reject` / `inconclusive` — and its gate
evidence lands in MLflow runs like the numbered curriculum's evaluation gates
(`04_first_evaluation.py`). API surfaces referenced (verified against current
documentation, August 2026): `ai_query` `responseFormat`/`failOnError`,
Unity Catalog `ALTER COLUMN … SET TAGS`, and Lakehouse Monitoring's
`data_quality.create_monitor` (which replaced the deprecated
`quality_monitors` API). Where current guidance differs from older advice —
notably "submit one query and let AI Functions manage parallelism" instead of
hand-chunked batches — the notebook says so inline.

Out of scope, on purpose: no UI, no orchestration framework, no generic ETL
abstraction. One clear worked example a team can copy and adapt.
