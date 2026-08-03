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
   an aggregate.** Aggregate accuracy is irrelevant when the failures
   concentrate in the one segment that matters — and a population aggregate
   *cannot* fail on a segment holding 2% of the rows, however broken it is.
4. **The all-strata figure is population-weighted, never pooled.** The
   sample deliberately over-samples rare strata, so pooling its rows would
   describe a population that does not exist. Each stratum is weighted back
   to its true share and the interval widened for the unequal sampling (a
   design effect, applied by converting the weighted variance to an
   effective sample size). Medium- and low-criticality fields gate on that.
5. **Precision and recall are reported separately.** Extraction fails two
   ways — hallucinating and missing — and a single accuracy figure hides
   whichever one your consumers care about. Abstention hurts recall, never
   precision: that asymmetry is what makes the abstention path the single
   highest-value change to an extraction pipeline.
6. **Insufficient evidence is `inconclusive`; a demonstrated failure is
   `reject`; neither is a pass.** If the interval straddles the bar and even
   a flawless sample of that size could not clear it (the lower bound of n/n
   caps at n/(n+z²)), the verdict is "label more rows". But if the whole
   interval sits *below* the bar, that is a rejection however small the
   sample: 0/30 and 30/30 are the same size and not the same evidence.
7. **Tier 1 cannot be auto-approved.** A fully passing result returns
   `pending_approval`; a named human accepts residual risk via
   `approve_gate`, abstentions get human review, and a rollback path is on
   file. Approval can never resurrect a rejection.
8. **Execution is refused without an adopting gate for the exact spec
   digest** (`require_executable`), and the run is idempotent: an anti-join
   selects only unlanded rows, so a partial failure restarts by re-running
   the same statement.
9. **Evidence belongs to what produced it, and must be complete.** The
   binding starts at the record: each `EvaluationRecord` names the
   *inference identity* — endpoint, model version, prompt version, **the
   prompt text itself**, abstention threshold, and the fields that build
   the response schema — that produced its prediction, and
   `score_extraction` refuses to score it against a different one. The
   text and not merely the label, because `prompt_version` is a string
   someone types: nothing stops it reading "1.0.0" across an edit. The
   template therefore lives in the spec, and `build_execute_sql` takes no
   prompt argument at all — there is no way to run one prompt and stamp
   another. Taking the stamp from the spec at scoring
   time instead would let v1 output certify itself as v2 evidence: the
   gate's release check would be reading a label the same call had just
   written.
   Predictions bind to the inference identity rather than the whole spec
   because the spec holds two different things. Tier, consumers,
   tolerances, strata and the release sequence change how output is
   *judged*, not what the model returns — binding predictions to them
   would force a paid re-run to obtain byte-identical results. Scores
   still carry the full release, because re-judging under a new policy
   does require re-scoring; that is arithmetic over records you hold.
   Scores additionally carry the declared confidence level (checked on
   every interval, not just the score) and the sample's stratum manifest,
   so quietly dropping the failing stratum before gating fails too.
   Re-scoring is arithmetic over records you already hold, so the strict
   rule is cheap to satisfy.
   The cost estimate is bound the same way: a longer prompt is a different
   budget, and `log_gate_evidence` refuses an estimate measured for
   another release rather than let one clear the ceiling on another's
   assumptions.
10. **"Done" means this content, by this release or newer.** The restart
    anti-join matches on the key, a digest of the source document, *and*
    the full release identity — spec digest, model version, prompt
    version — and the write is a `MERGE`. The content digest is what makes
    an edit-in-place reprocess instead of leaving the target serving values
    derived from text that no longer exists. Stratum labels are row
    metadata rather than model output, so a corrected label is resynced by
    `resync_strata_sql` instead of triggering a paid re-run that would
    regenerate identical values — and, like the MERGE, it refuses to
    relabel a row a newer release owns. Ordering is on the **pair**
    `(release_sequence, source_version)`: the release says *what* ran, the
    source version says *over which rows*, and neither alone orders two
    runs. A nightly job unchanged for months ties with itself on sequence,
    so without the version a delayed Monday run finishing after Tuesday's
    would write its older content over Tuesday's for every document edited
    in between. A strictly newer `release_sequence`
    counts as done **on its own, before the content digest is considered**;
    the MERGE separately updates a row only when its sequence is not being
    lowered, so an old job resuming after a newer release has landed cannot
    roll production back to its own stale output. The ordering inside that
    predicate is not cosmetic: test the digest first and a row the newer
    release landed from *edited* text stops matching, so the old job pays
    for inference the MERGE then correctly discards — a second full bill
    per edited document. Matching on the key alone would let a newly gated release
    report success while every row still carried the previous release's
    values and provenance;
    matching on model and prompt alone would do the same whenever the spec
    changed while those labels held still. Bump `spec_version` when a code
    change alters what the pipeline produces, since the digest covers the
    spec rather than the module. A field-set change also needs the target
    table migrated (`require_migrated_target`) — `CREATE TABLE IF NOT
    EXISTS` will not touch a table that already exists. Additive changes
    are applied; a column the release no longer produces *blocks* the run,
    because `INSERT *` expands over the target's columns and cannot
    resolve one the source lacks. Dropping it destroys data, so a human
    decides.
11. **An abstention is enforced, not merely recorded.** A value the model
    both returned and declared abstained, answered below the declared
    threshold, or answered with a confidence outside [0, 1] — structured
    output constrains the JSON type, not the range — is nulled rather than
    landed, because scoring treated it as an abstention and it never went
    through the precision gate. The same rule runs in evaluation and in
    execution (`apply_abstention_policy` and the generated SQL), so what
    was measured is what lands.
12. **Persisted evidence is verified, not trusted — all the way down.**
    Everything the gate reads round-trips through MLflow as JSON, and a
    report is what authorises a paid, table-mutating run, so nothing in it
    is taken at its word. The chain reconstructs from the raw counts up:

    - a `ConfidenceInterval` is checked against its own successes and trials;
    - a physical `FieldStratumScore`'s intervals against the counts printed
      beside them;
    - the population-weighted row by **recomputing** it from the physical
      rows and the weights it carries — its intervals are an effective
      sample size rather than a row count, so they cannot be checked
      against one, and it is the only row a medium- or low-criticality
      field is gated on, which makes it the row worth forging;
    - each `FieldGateResult` by re-running the gate over those scores;
    - and the report's aggregate `decision` from the field results.

    `require_executable` closes the loop by binding those scores to the
    spec — the one thing a self-contained report cannot know about itself —
    by checking the report judged every field, by checking the report's
    tier is this run's tier (tier selects which invariants the report's own
    validator enforced, so a relabelled tier switches the source pin off),
    and by checking it judged them under *this* spec's policy.

    **`build_execute_sql` re-runs all of it, plus the cost ceiling.** It is
    the function that emits the paid statement, so it does not assume the
    caller checked first: it calls `require_executable` and
    `require_within_ceiling`, and requires an estimate measured for this
    release *and* this snapshot, against the ceiling *the spec* declares.
    A guard that holds only because the notebook happens to call things in
    the right order protects the notebook, not the pipeline.

    The estimate is evidence too, and is treated like the rest: a
    `CostEstimate` recomputes its projection from the token counts, row
    count and prices it records, so a persisted one cannot declare itself
    free. And because every paid run reads a pinned snapshot, tier 3 —
    which has no gate report to pin it — is pinned by its estimate
    instead. Cost is that tier's only control, so an unpinned read would
    let the table outgrow the projection that authorised the spend. The spec owns policy, the report owns
    outcomes: a persisted report that quietly lowers a high field's
    required rate, or relabels it medium so the gate stops being
    worst-stratum, derives `adopt` perfectly honestly from real evidence
    and is refused here rather than by any internal check. Each link is computed by
    the *same* function that produced it (`_gate_field`, `_weighted_row`),
    never a second implementation written for checking: two implementations
    are two things to keep in sync, and the drift between them is the bug.
13. **The run reads the Delta version the evidence describes.** Population,
    sample, gate and cost estimate are all computed against one snapshot of
    the source, and a tier 1 spec then waits for a human. `build_execute_sql`
    therefore takes the report that authorised the run and reads the pinned
    version off it — never a separate argument that could disagree. Rows
    that land during review are not lost; they are the next cycle's work,
    with evidence of their own. (Time travel needs the files to still exist:
    a review outlasting `VACUUM` retention fails loudly and asks for a
    re-gate, which is the right answer.)
14. **The run record is written once per `run_id`.** It is a keyed `MERGE`,
    not an `INSERT`: `ai_run_id` is the join key every landed row uses to
    reach the run metadata, so a duplicate from a retried cell would fan
    out downstream joins and tie one run to two table versions.

## Adapting it to a new use case

Work through the spec first — most adaptation is spec, not code:

- **A unique, non-null key and a non-null document are preconditions, not
  niceties.** Every idempotence guarantee here rests on key equality.
  `NULL = NULL` is not true, so a null-keyed row is re-inferred and
  re-inserted on every run while the restart logic appears to work; a
  duplicated key matches twice, so the MERGE cannot resolve it and "one
  current row per key" — which every provenance join assumes — was never
  true. A null document fails a third way and even more quietly: its content
  digest is null, so the restart anti-join can never match it and each run
  pays to infer over an empty request again. `require_usable_source_rows`
  checks all three before any spend, and refuses rather than filtering:
  skipping those rows would shrink coverage of the table the gate just
  certified. Fix them upstream, or narrow the source view.
- **`release_sequence` must increase with every release.** It is what lets
  the pipeline tell *newer* from merely *different*. Identity alone cannot,
  and without an ordering a delayed retry or an overlapping deploy would see
  newer rows as unprocessed and write older output back over them — a silent
  rollback of the production table.
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
