# Databricks notebook source
# MAGIC %md
# MAGIC # Governed batch inference — reference implementation
# MAGIC
# MAGIC Batch inference over a large table is **a model deployment wearing a SQL
# MAGIC query's clothes**. One `ai_query` statement produces a derived dataset
# MAGIC that other things consume, but it passes through none of the controls a
# MAGIC deployment would. Four specific failures follow:
# MAGIC
# MAGIC 1. **Unbounded output error.** Nobody can inspect a million rows. Without
# MAGIC    a sampled estimate there is no error rate at all, only an impression.
# MAGIC 2. **Clustered failure, not random failure.** LLM extraction fails in
# MAGIC    patterns — unusual formats, edge-case entities, under-represented
# MAGIC    populations. Aggregate accuracy can look excellent while every error
# MAGIC    sits in the one segment that matters.
# MAGIC 3. **Provenance evaporates.** Three joins downstream, an AI-generated
# MAGIC    column looks like any other column.
# MAGIC 4. **Cost is discovered, not decided.** Rate limits do not stop an
# MAGIC    `ai_query` batch workload on a pay-per-token endpoint — only usage
# MAGIC    tracking sees it, after the fact.
# MAGIC
# MAGIC This notebook runs the whole governed pipeline end to end on synthetic
# MAGIC tax and investment documents (T4, T5, 1099-DIV, K-1):
# MAGIC
# MAGIC ```
# MAGIC DECLARE → ESTIMATE → SAMPLE → EVALUATE → GATE → EXECUTE → LAND → MONITOR
# MAGIC ```
# MAGIC
# MAGIC Each stage produces an artifact; the gate is where it bites. The
# MAGIC reusable pieces (spec model, statistics, gate, SQL builders) live in
# MAGIC `governed_batch_inference.py` next to this notebook and are unit-tested
# MAGIC in the repository (`tests/test_governed_batch_inference.py`).
# MAGIC
# MAGIC **How to run.** Works standalone on a recent Databricks Runtime (15.4+
# MAGIC or serverless) with Unity Catalog. Set the `catalog`/`schema` widgets to
# MAGIC a location you can write to. `inference_mode`:
# MAGIC
# MAGIC - `simulated` (default): a deterministic, seeded extractor plays the
# MAGIC   model, so the notebook runs with **no endpoint, no cost**, and the
# MAGIC   statistical lessons reproduce exactly. Every other line of the
# MAGIC   pipeline — spec, sampling, scoring, gate, provenance — is the real
# MAGIC   path, identical to what a live run uses.
# MAGIC - `live`: the evaluate and execute stages call `ai_query` against the
# MAGIC   endpoint named in the spec. This spends real money (estimated and
# MAGIC   gated in stage 2) and your results will differ from the walkthrough
# MAGIC   numbers.
# MAGIC
# MAGIC All issuers, amounts, and identifiers below are synthetic teaching data.

# COMMAND ----------

# MAGIC %md
# MAGIC `faker` generates the synthetic document text. It is deliberately not in
# MAGIC the repository's certified locks — it is demo scaffolding, not a runtime
# MAGIC dependency. A bounded range keeps the install predictable; a real
# MAGIC project pins exact versions through its own lockfile or cluster policy.

# COMMAND ----------

# MAGIC %pip install --quiet "faker>=33,<40"

# COMMAND ----------

import hashlib
import importlib
import os
import random
import sys

import mlflow
import pandas as pd
from databricks.sdk.runtime import dbutils, display, spark
from faker import Faker
from pyspark.sql import functions as F

# The module sits next to this notebook; for workspace files / Repos the
# working directory is the notebook directory on recent runtimes.
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
gbi = importlib.import_module("governed_batch_inference")

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "governed_batch_demo")
dbutils.widgets.dropdown("inference_mode", "simulated", ["simulated", "live"])

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SIMULATED = dbutils.widgets.get("inference_mode") == "simulated"

SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.tax_documents"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.tax_document_gold_labels"
SAMPLE_TABLE = f"{CATALOG}.{SCHEMA}.tax_document_eval_sample"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.tax_document_entities"
RUNS_TABLE = f"{CATALOG}.{SCHEMA}.batch_inference_runs"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"mode={'simulated' if SIMULATED else 'live'}  schema={CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 0 — synthetic corpus with a genuinely harder minority stratum
# MAGIC
# MAGIC 25,000 documents. ~98% are `standard` layouts: clean, labelled lines
# MAGIC that any competent model reads correctly. ~2% are `legacy_scan`: OCR-ish
# MAGIC noise, misspelled labels, issuer names split across lines, box-label
# MAGIC synonyms. **The minority stratum being genuinely harder is the whole
# MAGIC point of the demonstration** — it is where extraction quietly fails
# MAGIC while aggregate numbers stay excellent.
# MAGIC
# MAGIC Because the corpus is synthetic we hold perfect gold labels for every
# MAGIC row. In production the gold table is built by *human adjudication* of
# MAGIC sampled rows — that scarcity is modelled honestly in stage 3, where the
# MAGIC sample is sized by labelling capacity, not by compute.
# MAGIC
# MAGIC Reproducibility note: every random draw below is keyed on the document
# MAGIC id (`random.Random(f"{doc_id}|purpose")`), so layouts, gold values, and
# MAGIC the simulated extractor's behaviour are stable across machines and
# MAGIC library versions. Faker supplies only display strings (issuer names),
# MAGIC never correctness.

# COMMAND ----------

N_DOCS = 25_000
DOC_TYPES = ["T4", "T5", "1099-DIV", "K-1"]
FIELD_NAMES = ["issuer_name", "tax_year", "box_amount", "account_id"]

Faker.seed(2026)
_fake = Faker("en_CA")
ISSUER_POOL = [_fake.company() for _ in range(400)]


def doc_rng(doc_id: str, purpose: str) -> random.Random:
    """Deterministic per-document randomness, independent of Faker."""
    return random.Random(f"{doc_id}|{purpose}")


def make_gold(index: int, doc_id: str) -> dict:
    return {
        "issuer_name": ISSUER_POOL[index % len(ISSUER_POOL)],
        "tax_year": str(2019 + index % 7),
        # Some documents genuinely lack a value: the correct answer is
        # null, and asserting anything is a hallucination.
        "box_amount": (
            None
            if doc_rng(doc_id, "box_gold").random() < 0.06
            else f"{(index % 90000) / 7:.2f}"
        ),
        "account_id": (
            None
            if doc_rng(doc_id, "account_gold").random() < 0.04
            else f"AC{index:08d}"
        ),
    }


def render_document(doc_id: str, doc_type: str, layout: str, gold: dict) -> str:
    if layout == "standard":
        lines = [
            f"{doc_type} STATEMENT OF INCOME",
            f"Issuer: {gold['issuer_name']}",
            f"Tax year: {gold['tax_year']}",
            f"Box 14 amount: {gold['box_amount'] or 'N/A'}",
            f"Account: {gold['account_id'] or 'N/A'}",
        ]
        return "\n".join(lines)
    noise = doc_rng(doc_id, "noise")
    issuer = gold["issuer_name"]
    split_at = max(1, len(issuer) // 2 + noise.randint(-3, 3))
    issuer_broken = issuer[:split_at] + "\n" + issuer[split_at:]
    box_label = noise.choice(["Bx14", "BOX-14 amt", "14.", "Box14(see notes)"])
    account_label = noise.choice(["Acct #", "ACCT", "A/C no."])
    lines = [
        f"{doc_type}  *scanned copy*",
        noise.choice(["Issu3r:", "lssuer:", "Payer/lssuer"]),
        issuer_broken,
        f"Yr {gold['tax_year']}  (assessment period)",
        f"{box_label} {gold['box_amount'] or ''}".rstrip(),
        f"{account_label} {gold['account_id'] or '---'}",
        "" if noise.random() < 0.5 else "~~~ page 1 of 1 ~~~",
    ]
    return "\n".join(lines)


documents = []
for index in range(N_DOCS):
    doc_id = f"DOC-{index:06d}"
    layout = "legacy_scan" if doc_rng(doc_id, "layout").random() < 0.02 else "standard"
    doc_type = doc_rng(doc_id, "doctype").choice(DOC_TYPES)
    gold = make_gold(index, doc_id)
    documents.append(
        {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "layout": layout,
            "doc_text": render_document(doc_id, doc_type, layout, gold),
            **{f"gold_{name}": gold[name] for name in FIELD_NAMES},
        }
    )

corpus = spark.createDataFrame(pd.DataFrame(documents))
source_columns = ["doc_id", "doc_type", "layout", "doc_text"]
corpus.select(*source_columns).write.mode("overwrite").saveAsTable(SOURCE_TABLE)
corpus.select("doc_id", "layout", *[f"gold_{name}" for name in FIELD_NAMES]).write.mode(
    "overwrite"
).saveAsTable(GOLD_TABLE)

display(spark.table(SOURCE_TABLE).groupBy("layout").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1 — DECLARE: the spec exists before any result does
# MAGIC
# MAGIC The single most important property of this stage is *when* it happens.
# MAGIC Tolerable error rates are declared per field, committed to the
# MAGIC repository, and reviewed **before the first inference runs**. A
# MAGIC threshold set after looking at the output is not a threshold — it is a
# MAGIC rationalisation of whatever the output happened to be.
# MAGIC
# MAGIC Choices being made explicit here:
# MAGIC
# MAGIC - **`use_tier: 2` (operational).** The output feeds an internal
# MAGIC   reconciliation process with a human downstream. The tier is set by
# MAGIC   what *consumes* the output, not by how big the table is. (Stage 5
# MAGIC   also demonstrates the tier 1 path, which cannot be auto-approved.)
# MAGIC - **`consumed_by` is named.** Forces the author to know who is
# MAGIC   downstream — if you cannot fill this field in, that is a finding.
# MAGIC - **Per-field criticality and tolerance.** `issuer_name`, `tax_year`
# MAGIC   and `box_amount` are `high`: they flow into reconciliation matching,
# MAGIC   so the gate will hold their **worst stratum** to the declared bar.
# MAGIC   `account_id` is `medium` with a looser tolerance: it is
# MAGIC   cross-checked against a reference table downstream, so the pooled
# MAGIC   sample decides.
# MAGIC - **The endpoint is a small model.** Cost per token varies by an order
# MAGIC   of magnitude; anyone who wants a frontier model for a full-table
# MAGIC   scan should have to ask. The name below is a placeholder for
# MAGIC   whichever governed endpoint your platform team approves.
# MAGIC - **`cost_ceiling_cad` is a hard stop**, enforced in stage 2 before
# MAGIC   anything runs.

# COMMAND ----------

# The prompt lives *in* the spec, not beside it. `prompt_version` is a
# string someone types and nothing stops it staying "1.0.0" across an
# edit, so the identity that guards evidence covers this text itself.
# Its length is also an input to the cost estimate: a prompt edit is a
# new release *and* a new budget.
PROMPT_V1 = (
    "Extract the following fields from the tax document below and answer "
    "in the required JSON shape: issuer_name, tax_year, box_amount, "
    "account_id. Use null for a field that is not present.\n\nDOCUMENT:\n"
)

spec_v1 = gbi.BatchInferenceSpec(
    name="tax_document_extraction",
    source_table=SOURCE_TABLE,
    target_table=TARGET_TABLE,
    run_metadata_table=RUNS_TABLE,
    document_column="doc_text",
    key_column="doc_id",
    use_tier=gbi.UseTier.OPERATIONAL,
    consumed_by=("income_reconciliation_pipeline", "annual_slip_qa_review"),
    fields=(
        gbi.FieldSpec(
            name="issuer_name",
            description="Legal name of the organisation issuing the slip.",
            criticality=gbi.Criticality.HIGH,
            tolerable_error_rate=0.05,
        ),
        gbi.FieldSpec(
            name="tax_year",
            description="Four-digit tax year the document reports on.",
            criticality=gbi.Criticality.HIGH,
            tolerable_error_rate=0.05,
        ),
        gbi.FieldSpec(
            name="box_amount",
            description="Amount reported in box 14 (or equivalent), as printed.",
            criticality=gbi.Criticality.HIGH,
            tolerable_error_rate=0.05,
        ),
        gbi.FieldSpec(
            name="account_id",
            description="Account identifier printed on the slip, if any.",
            criticality=gbi.Criticality.MEDIUM,
            tolerable_error_rate=0.10,
        ),
    ),
    strata=("layout",),
    endpoint="databricks-gpt-oss-20b",
    model_version="gpt-oss-20b-2025-08",
    prompt_version="1.0.0",
    prompt_template=PROMPT_V1,
    # Bumped on every release. Identity says two runs differ; only an
    # ordering says which is later, and without that an old job resuming
    # late would overwrite newer rows with its own stale output.
    release_sequence=1,
    cost_ceiling_cad=50.0,
    abstain_threshold=0.70,
)

spec_yaml = spec_v1.to_yaml()
print(spec_yaml)
print(f"spec digest: {spec_v1.spec_digest}")

# In a real project this YAML is a file in the repo, changed by pull
# request. Loading it back through the same strict model is the point:
# the gate will read tolerances from this object and nowhere else.
spec_v1 = gbi.BatchInferenceSpec.from_yaml(spec_yaml)

# COMMAND ----------

# MAGIC %md
# MAGIC The declared tolerances immediately imply a minimum evidence size, via
# MAGIC the Wilson lower bound (stage 4 explains the interval itself). Even a
# MAGIC **flawless** sample of n rows cannot prove "error rate ≤ t" until n
# MAGIC reaches the figure below — labelling fewer rows per stratum buys no
# MAGIC decision at all. This is the number that turns "how many can the team
# MAGIC adjudicate?" into an explicit trade against the declared tolerance.

# COMMAND ----------

for field in spec_v1.fields:
    minimum = gbi.min_labelled_rows_for_tolerance(
        field.tolerable_error_rate, spec_v1.confidence_level
    )
    print(
        f"{field.name:12s} criticality={field.criticality:6s} "
        f"tolerable_error_rate={field.tolerable_error_rate:.2f} "
        f"→ ≥{minimum} labelled rows per gated group, even if flawless"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2 — ESTIMATE: approve a budget instead of discovering a bill
# MAGIC
# MAGIC A probe over a handful of documents measures tokens per row; multiplied
# MAGIC out by row count and the endpoint's rate, that is the projected spend —
# MAGIC compared against the declared ceiling **before** anything executes.
# MAGIC
# MAGIC Two things happen before the probe, both cheap and both refusals:
# MAGIC
# MAGIC - **Null source keys are rejected.** Everything restartable here rests
# MAGIC   on key equality, and `NULL = NULL` is not true, so a null-keyed row
# MAGIC   never matches the restart anti-join (it is re-inferred, and paid for,
# MAGIC   every run) and never matches the MERGE (a fresh duplicate every run).
# MAGIC   The guarantee does not degrade gracefully — it stops holding while
# MAGIC   still looking like it works.
# MAGIC - **The estimate is per release.** It carries the prompt, model, and
# MAGIC   spec revision it was measured for, and the instruction tokens come
# MAGIC   from the real prompt rather than a constant. Prompt v2 in stage 5 is
# MAGIC   longer than v1, so it gets its own estimate and its own ceiling
# MAGIC   check; carrying v1's number forward would authorise the run against
# MAGIC   a cheaper release's assumptions.
# MAGIC
# MAGIC Two honesty notes:
# MAGIC
# MAGIC - `ai_query` does not return token usage, so the probe estimates input
# MAGIC   tokens from text length (~4 characters/token) and output tokens from
# MAGIC   the response schema. After the first real run, refine both from
# MAGIC   `system.billing.usage` (AI Functions costs appear under the
# MAGIC   `MODEL_SERVING` product, `BATCH_INFERENCE` offering type).
# MAGIC - The CAD-per-million-token rates below are **placeholders**. Take real
# MAGIC   rates from your account's price sheet for the endpoint in the spec.

# COMMAND ----------

# Everything from here — the population counts, the sample, the gate and
# the cost estimate — describes the source *as it is right now*. Record
# which version that is, so the run can read those same rows back after
# labelling and review have moved on. `DESCRIBE HISTORY` is newest-first.
SOURCE_SNAPSHOT = gbi.SourceSnapshot(
    table=SOURCE_TABLE,
    version=spark.sql(f"DESCRIBE HISTORY {SOURCE_TABLE}").first().version,
)
print(f"evidence describes {SOURCE_TABLE} at version {SOURCE_SNAPSHOT.version}")

# From this point the live table is never read again. Every count, draw
# and probe below goes through one of these two, because evidence about
# rows the run will not process — or missing rows it will — is not
# evidence about the run. `PINNED_SOURCE` is the SQL spelling; the
# `pinned_source()` reader is the DataFrame one.
PINNED_SOURCE = f"{SOURCE_TABLE} VERSION AS OF {SOURCE_SNAPSHOT.version}"


def pinned_source():
    return spark.read.option("versionAsOf", SOURCE_SNAPSHOT.version).table(SOURCE_TABLE)


# Pre-flight, before anything is paid for: every idempotence guarantee
# here rests on key equality, and `NULL = NULL` is not true. A null-keyed
# row would be re-inferred and re-inserted on every single run while the
# restart logic appeared to work. A null document does the same via its
# null content digest, and pays an endpoint to read nothing each time.
rows = spark.sql(gbi.source_preflight_sql(spec_v1, SOURCE_SNAPSHOT)).first()
gbi.require_usable_source_rows(
    spec_v1, rows.null_keys, rows.duplicate_keys, rows.null_documents
)
print(
    f"source rows usable: {rows.null_keys} null and "
    f"{rows.duplicate_keys} duplicate {spec_v1.key_column} values, "
    f"{rows.null_documents} null {spec_v1.document_column} values"
)

CAD_PER_M_INPUT = 0.20  # placeholder — use your negotiated list price
CAD_PER_M_OUTPUT = 0.60  # placeholder

probe = pinned_source().limit(32).collect()
row_count = pinned_source().count()


def estimate_for(spec) -> "gbi.CostEstimate":
    """Cost of running *this* release over the whole table.

    The instruction tokens come from the actual prompt, so a longer
    prompt shows up as a larger budget instead of hiding behind a
    constant. The returned estimate carries the release it was measured
    for, and the gate refuses to log an estimate from a different one.

    It carries the snapshot too. The release fixes the price per row; the
    snapshot fixes how many rows there are, and a projection made over a
    smaller, older image of the table would clear a ceiling the run then
    blows through.
    """
    instruction_tokens = gbi.estimate_tokens_from_text(spec.prompt_template)
    return gbi.estimate_cost(
        spec,
        row_count=row_count,
        probe_input_tokens=[
            gbi.estimate_tokens_from_text(row.doc_text) + instruction_tokens
            for row in probe
        ],
        probe_output_tokens=[130] * len(probe),  # structured response, per schema
        cad_per_million_input_tokens=CAD_PER_M_INPUT,
        cad_per_million_output_tokens=CAD_PER_M_OUTPUT,
        source_snapshot=SOURCE_SNAPSHOT,
    )


estimate_v1 = gbi.require_within_ceiling(estimate_for(spec_v1))
print(
    f"{estimate_v1.row_count:,} rows × "
    f"(~{estimate_v1.mean_input_tokens_per_row:.0f} in + "
    f"~{estimate_v1.mean_output_tokens_per_row:.0f} out tokens) × safety "
    f"{estimate_v1.safety_factor} → projected "
    f"{estimate_v1.projected_cost_cad:.2f} CAD ≤ ceiling "
    f"{estimate_v1.cost_ceiling_cad:.2f} CAD — approved to proceed"
)

# COMMAND ----------

# What the stop looks like: the same table pointed at a frontier-priced
# endpoint blows through the ceiling, and the pipeline refuses to start.
# That refusal — before the run — is the conversation you want to force.
instruction_tokens = gbi.estimate_tokens_from_text(spec_v1.prompt_template)
try:
    gbi.require_within_ceiling(
        gbi.estimate_cost(
            spec_v1,
            row_count=row_count,
            probe_input_tokens=[
                gbi.estimate_tokens_from_text(row.doc_text) + instruction_tokens
                for row in probe
            ],
            probe_output_tokens=[130] * len(probe),
            cad_per_million_input_tokens=7.00,  # frontier-class placeholder
            cad_per_million_output_tokens=21.00,
            source_snapshot=SOURCE_SNAPSHOT,
        )
    )
except gbi.CostCeilingExceeded as refusal:
    print(f"CostCeilingExceeded: {refusal}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 3 — SAMPLE: stratified, sized by human labelling capacity
# MAGIC
# MAGIC The failure modes of LLM extraction track document properties, so the
# MAGIC sample is stratified on the spec's `strata` columns (`layout` here —
# MAGIC chosen because that is where probe errors clustered; revisit the choice
# MAGIC at every re-sample).
# MAGIC
# MAGIC A **proportional** 400-row sample of this corpus would contain roughly
# MAGIC 8 `legacy_scan` rows — almost no information about the hard case, which
# MAGIC is the case that matters. So rare strata are deliberately over-sampled:
# MAGIC every stratum gets a floor of 150 rows (comfortably above the 73-row
# MAGIC feasibility minimum for the 5% tolerance, so a small number of real
# MAGIC errors does not automatically starve the gate), and the remainder is
# MAGIC allocated proportionally.
# MAGIC
# MAGIC The binding constraint is the **labelling budget**: 400 rows is roughly
# MAGIC two days of careful adjudication for one analyst. This is also why the
# MAGIC gold set is an asset — build it once, version it, reuse and extend it
# MAGIC at every re-evaluation instead of paying the two days again.

# COMMAND ----------

population = {
    row["layout"]: row["count"]
    for row in pinned_source().groupBy("layout").count().collect()
}
LABELLING_BUDGET = 400
allocation = gbi.allocate_stratified_sample(
    population, LABELLING_BUDGET, min_per_stratum=150
)

comparison = pd.DataFrame(
    [
        {
            "stratum": stratum,
            "population": population[stratum],
            "proportional_share": round(
                LABELLING_BUDGET * population[stratum] / sum(population.values())
            ),
            "stratified_allocation": allocation[stratum],
        }
        for stratum in sorted(population)
    ]
)
display(comparison)

# Deterministic draw: rank rows inside each stratum by a hash of the key
# and keep the first `allocation[stratum]`. Re-running selects the same
# rows, so labelling work is never invalidated by a re-run.
pinned_source().createOrReplaceTempView("source_docs")
# Stratum values are table data, not configuration, so they are escaped
# rather than pasted between quotes — a value containing an apostrophe is
# ordinary in real data and must not be able to reshape the statement.
allocation_case = " ".join(
    f"WHEN {gbi.sql_string_literal(stratum)} THEN {quota}"
    for stratum, quota in allocation.items()
)
spark.sql(f"""
    CREATE OR REPLACE TABLE {SAMPLE_TABLE} AS
    SELECT doc_id, doc_type, layout, doc_text
    FROM (
      SELECT *,
             row_number() OVER (
               PARTITION BY layout ORDER BY sha2(doc_id, 256)
             ) AS rank_in_stratum
      FROM source_docs
    )
    WHERE rank_in_stratum <= CASE layout {allocation_case} END
    """)
display(spark.table(SAMPLE_TABLE).groupBy("layout").count())

# COMMAND ----------

# MAGIC %md
# MAGIC Cheap what-if before spending labelling effort: had the floor been 30
# MAGIC rows, even a *flawless* 30/30 stratum result has a Wilson lower bound
# MAGIC of ~0.886 — mathematically unable to clear a 95% bar. The gate would
# MAGIC return `inconclusive` and the labelling work would buy no decision.

# COMMAND ----------

flawless_30 = gbi.wilson_interval(30, 30, spec_v1.confidence_level)
print(
    f"30/30 correct → lower bound {flawless_30.lower:.4f} < 0.95 → "
    "inconclusive by construction; the floor must respect the tolerance"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 4 — EVALUATE: per field, per stratum, precision *and* recall
# MAGIC
# MAGIC Inference runs on the sample and is scored against gold labels:
# MAGIC
# MAGIC - **Per field, not aggregate** — a single accuracy number hides
# MAGIC   everything.
# MAGIC - **Per stratum, not pooled** — this is what catches clustered failure.
# MAGIC - **The all-strata figure is population-weighted, never pooled.**
# MAGIC   Stage 3 deliberately over-sampled `legacy_scan` by about 19×, so
# MAGIC   simply pooling the sample's rows would describe a population that
# MAGIC   does not exist — one that is 38% hard documents instead of 2%.
# MAGIC   Each stratum is therefore weighted back to its true share, and the
# MAGIC   interval is widened for the unequal sampling (a design effect):
# MAGIC   the weighted variance is converted to an *effective sample size*
# MAGIC   and the Wilson interval taken at that n. Reporting a raw pooled
# MAGIC   proportion from a stratified sample is a real and common error —
# MAGIC   it silently answers a question nobody asked.
# MAGIC - **Precision and recall separately** — extraction fails two different
# MAGIC   ways. Precision asks: of the values the model *asserted*, how many
# MAGIC   were right (hallucination control)? Recall asks: of the values that
# MAGIC   truly exist, how many were produced (miss control)? An abstention
# MAGIC   hurts recall but not precision — that asymmetry is exactly why the
# MAGIC   abstention path in stage 6 is so valuable.
# MAGIC - **Wilson score intervals, not bare proportions** — small per-stratum
# MAGIC   samples carry real uncertainty and the gate must see it.
# MAGIC - **Why not `mlflow.genai.evaluate()`?** That API and its scorers judge
# MAGIC   generative quality, often with LLM judges. This gate is deterministic
# MAGIC   frequentist statistics against human-adjudicated gold labels, so it
# MAGIC   uses plain MLflow 3 tracking (params, metrics, artifacts) and keeps
# MAGIC   the two evaluation concepts separate — a platform rule, not an
# MAGIC   accident. Add a calibrated judge later if you need one; it does not
# MAGIC   replace this gate.
# MAGIC
# MAGIC In `simulated` mode a deterministic extractor stands in for the model.
# MAGIC Its error rates are *constructed* so that the standard stratum looks
# MAGIC excellent while `legacy_scan` fails — the clustered-failure situation
# MAGIC this pattern exists to catch. (A real prompt-v2 improvement of that
# MAGIC size would take several iterations; the simulation is calibrated for
# MAGIC teaching, and says so.) In `live` mode this cell calls `ai_query` with
# MAGIC structured output on the sample instead.

# COMMAND ----------

# (correct, abstain) per (prompt_version, layout, field); the remainder is
# a wrong answer. Gold-absent rows are *hallucinated* at HALLUCINATE rate.
SIM_RATES = {
    ("1.0.0", "standard"): {
        "issuer_name": (0.995, 0.0),
        "tax_year": (0.998, 0.0),
        "box_amount": (0.990, 0.0),
        "account_id": (0.995, 0.0),
    },
    ("1.0.0", "legacy_scan"): {
        "issuer_name": (0.85, 0.0),
        "tax_year": (0.96, 0.0),
        "box_amount": (0.88, 0.0),
        "account_id": (0.90, 0.0),
    },
    ("2.0.0", "standard"): {
        "issuer_name": (0.997, 0.002),
        "tax_year": (0.998, 0.001),
        "box_amount": (0.995, 0.003),
        "account_id": (0.995, 0.002),
    },
    ("2.0.0", "legacy_scan"): {
        "issuer_name": (0.995, 0.004),
        "tax_year": (0.995, 0.003),
        "box_amount": (0.995, 0.004),
        "account_id": (0.985, 0.010),
    },
}
HALLUCINATE = {
    ("1.0.0", "standard"): 0.02,
    ("1.0.0", "legacy_scan"): 0.25,
    ("2.0.0", "standard"): 0.002,
    ("2.0.0", "legacy_scan"): 0.01,
}

PROMPT_V2 = (
    "Extract issuer_name, tax_year, box_amount, account_id from the tax "
    "document below, answering in the required JSON shape.\n"
    "Rules learned from legacy scanned copies: labels may be misspelled "
    "('Issu3r', 'lssuer') or replaced by synonyms ('Bx14', 'Acct #'); an "
    "issuer name may be split across two lines — rejoin it. Report "
    "amounts exactly as printed. Use null for a field that is absent. If "
    "you cannot read a value with at least 70% confidence, do NOT guess: "
    "add the field name to abstained_fields and explain in "
    "abstain_reason.\n\nDOCUMENT:\n"
)


def simulate_extraction(doc_id: str, layout: str, gold: dict, version: str):
    """Deterministic stand-in for the model, keyed on doc_id + version."""
    rates = SIM_RATES[(version, layout)]
    hallucinate = HALLUCINATE[(version, layout)]
    predicted: dict[str, str | None] = {}
    abstained: set[str] = set()
    for name in FIELD_NAMES:
        draw = doc_rng(doc_id, f"{name}|{version}").random()
        gold_value = gold[name]
        if gold_value is None:
            predicted[name] = "9999" if draw < hallucinate else None
            continue
        correct, abstain = rates[name]
        if draw < correct:
            predicted[name] = gold_value
        elif draw < correct + abstain:
            predicted[name] = None
            abstained.add(name)
        else:
            predicted[name] = gold_value + "X"  # a plausible-looking error
    return predicted, abstained


def live_extraction(spec, table: str) -> dict:
    """ai_query with structured output over `table`; returns {doc_id: row}."""
    request = f"concat({gbi.sql_string_literal(spec.prompt_template)}, doc_text)"
    rows = spark.sql(f"""
        SELECT doc_id,
               from_json(raw.response,
                         {gbi.sql_string_literal(gbi.response_struct_type(spec))}
               ) AS parsed,
               raw.errorMessage AS error_message
        FROM (
          SELECT doc_id,
                 ai_query(
                   {gbi.sql_string_literal(spec.endpoint)},
                   {request},
                   responseFormat => {gbi.response_format_sql_literal(spec)},
                   failOnError => false
                 ) AS raw
          FROM {table}
        )
        """).collect()
    return {row.doc_id: row for row in rows}


def evaluation_records(spec) -> list:
    sample = (
        spark.table(SAMPLE_TABLE)
        .join(spark.table(GOLD_TABLE).drop("layout"), "doc_id")
        .collect()
    )
    live = live_extraction(spec, SAMPLE_TABLE) if not SIMULATED else {}
    records = []
    failed = 0
    for row in sample:
        gold = {name: row[f"gold_{name}"] for name in FIELD_NAMES}
        declared: set[str] = set()
        if SIMULATED:
            predicted, declared = simulate_extraction(
                row.doc_id, row.layout, gold, spec.prompt_version
            )
            confidences = {
                name: (
                    None
                    if name in declared
                    else round(
                        doc_rng(row.doc_id, f"{name}|conf").uniform(0.72, 0.99), 3
                    )
                )
                for name in FIELD_NAMES
            }
        else:
            result = live[row.doc_id]
            parsed = result.parsed
            if parsed is None:
                # failOnError => false nulls the response for a failed row,
                # so `parsed` is null. One poisoned document must not abort
                # the whole gate: score it as producing no value — a miss
                # for recall, no assertion for precision — and report the
                # count. Deliberately NOT counted as an abstention: the
                # model did not decline, the call failed.
                failed += 1
                predicted = dict.fromkeys(FIELD_NAMES)
                confidences = dict.fromkeys(FIELD_NAMES)
            else:
                predicted = {name: parsed[name] for name in FIELD_NAMES}
                confidences = {
                    name: parsed[f"{name}_confidence"] for name in FIELD_NAMES
                }
                declared = set(parsed["abstained_fields"] or [])
        # Score what would actually land, not what the model returned: the
        # execute stage nulls anything abstained or under-confident, so
        # measuring the raw response would gate output that never ships
        # and miss output that does.
        permitted, abstained = gbi.apply_abstention_policy(
            spec, predicted, confidences, declared
        )
        records.append(
            gbi.EvaluationRecord(
                stratum=row.layout,
                # Stamped where the prediction was produced. Scoring
                # verifies it against the spec being gated, so v1 output
                # cannot be scored as v2 evidence — while a pure policy
                # change (tier, consumers) re-judges these same rows.
                inference=spec.inference,
                gold=gold,
                predicted=permitted,
                abstained=abstained,
            )
        )
    if failed:
        # A high rate here means the evidence is thin for infrastructure
        # reasons, not model reasons — investigate before trusting the gate.
        print(f"WARNING: {failed}/{len(sample)} sample rows failed inference")
    return records


def scores_frame(scores) -> pd.DataFrame:
    rows = []
    for score in scores:
        for metric_name, interval in (
            ("precision", score.precision),
            ("recall", score.recall),
        ):
            if interval is None:
                continue
            rows.append(
                {
                    "field": score.field,
                    "stratum": score.stratum,
                    "metric": metric_name,
                    "evidence": f"{interval.successes}/{interval.trials}",
                    "point": round(interval.point, 4),
                    "lower_bound": round(interval.lower, 4),
                    "abstention_rate": round(score.abstention_rate, 4),
                }
            )
    return pd.DataFrame(rows)


records_v1 = evaluation_records(spec_v1)
scores_v1 = gbi.score_extraction(records_v1, spec_v1, population)
print(
    "Tolerances in force (declared in stage 1, before any of these numbers "
    "existed):",
    {f.name: f.tolerable_error_rate for f in spec_v1.fields},
)
display(scores_frame(scores_v1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interlude — why the gate uses the interval's lower bound
# MAGIC
# MAGIC The most common way this pattern fails silently is gating on point
# MAGIC estimates. The demonstration the whole pipeline hinges on:

# COMMAND ----------

lesson = gbi.wilson_interval(97, 100, 0.95)
print(f"""A run scores 97/100 on the sample. Tolerance says ≥95% required.

    point estimate : {lesson.point:.4f}   (≥ 0.95 — LOOKS like a pass)
    95% Wilson interval : [{lesson.lower:.4f}, {lesson.upper:.4f}]
    lower bound    : {lesson.lower:.4f}   (< 0.95 — NOT a pass)

With only 100 labelled rows, "97%" is consistent with a true error rate
worse than 8%. The run has not passed — it has produced an encouraging
number with too little evidence behind it. The gate therefore compares
the LOWER BOUND of the interval to the declared tolerance, never the
point estimate.""")

# The same situation usually occurs naturally in this very evaluation:
for score in scores_v1:
    if score.stratum == gbi.WEIGHTED or score.precision is None:
        continue
    required = spec_v1.field_named(score.field).required_rate
    interval = score.precision
    if interval.point >= required > interval.lower:
        print(
            f"live example → {score.field} in {score.stratum}: point "
            f"{interval.point:.4f} clears {required:.2f}, lower bound "
            f"{interval.lower:.4f} does not ({interval.successes}/"
            f"{interval.trials})"
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interlude — what a naive random sample would have told you
# MAGIC
# MAGIC The central lesson. Score a **proportional** random sample of the same
# MAGIC size and look at the all-strata figure — the way this validation is
# MAGIC usually done when it is done at all. Every field sails past its
# MAGIC tolerance. Same model, same documents, same tolerances: "ship it."
# MAGIC
# MAGIC Note what is *not* wrong with that number: it is a perfectly good
# MAGIC estimate of the population error rate. That is the trap. A population
# MAGIC aggregate cannot fail on a segment holding 2% of the rows — the
# MAGIC arithmetic will not let it, no matter how broken that segment is. The
# MAGIC naive sample compounds this by containing only a handful of
# MAGIC `legacy_scan` rows, far too few to say anything about them even if
# MAGIC someone thought to look.
# MAGIC
# MAGIC The stratified per-stratum table above says the opposite for the one
# MAGIC population segment that matters. Aggregate accuracy — however
# MAGIC correctly computed — is the wrong question when the failures
# MAGIC concentrate in non-standard filings.

# COMMAND ----------

naive_ids = spark.sql(f"""
    SELECT doc_id FROM {PINNED_SOURCE}
    ORDER BY sha2(concat(doc_id, '|naive'), 256)
    LIMIT {LABELLING_BUDGET}
    """)
naive_rows = (
    pinned_source()
    .join(naive_ids, "doc_id")
    .join(spark.table(GOLD_TABLE).drop("layout"), "doc_id")
    .collect()
)

naive_records = []
for row in naive_rows:
    gold = {name: row[f"gold_{name}"] for name in FIELD_NAMES}
    predicted, abstained = simulate_extraction(row.doc_id, row.layout, gold, "1.0.0")
    naive_records.append(
        gbi.EvaluationRecord(
            stratum=row.layout,
            inference=spec_v1.inference,
            gold=gold,
            predicted=predicted,
            abstained=frozenset(abstained),
        )
    )

naive_scores = gbi.score_extraction(naive_records, spec_v1, population)
naive_layout_mix = pd.Series(
    [record.stratum for record in naive_records]
).value_counts()
print(f"naive sample stratum mix:\n{naive_layout_mix}\n")
naive_aggregate = scores_frame(
    [score for score in naive_scores if score.stratum == gbi.WEIGHTED]
)
print("naive population-level view — every lower bound clears its tolerance:")
display(naive_aggregate)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 5 — GATE: the load-bearing decision
# MAGIC
# MAGIC Per field: compare the Wilson **lower bound** against the tolerance
# MAGIC declared in stage 1. For `criticality: high` fields the gate uses the
# MAGIC **worst-performing stratum**, never any aggregate. Medium and low
# MAGIC fields are judged on the population-weighted estimate instead —
# MAGIC that is what their criticality means. Three outcomes are possible,
# MAGIC and they are different:
# MAGIC
# MAGIC - `adopt` — the interval's lower bound clears the declared bar;
# MAGIC - `reject` — the bar was not demonstrated. Decisively so when the
# MAGIC   interval's *upper* bound also sits below it: 0/30 and 30/30 are
# MAGIC   both too small to pass, but only 30/30 is uninformative. Telling
# MAGIC   someone to "label more rows" when the model got nothing right
# MAGIC   would waste a week to confirm what the first 30 rows established;
# MAGIC - `inconclusive` — the interval straddles the bar and the group is
# MAGIC   too small for any result to clear it. Label more rows; the model
# MAGIC   has not been shown to be bad.
# MAGIC
# MAGIC The outcome, evidence, and approver are logged as an MLflow run — the
# MAGIC gate artifact everything else joins back to.

# COMMAND ----------

mlflow.set_experiment("/Shared/governed-batch-inference-demo")

report_v1 = gbi.evaluate_gate(spec_v1, scores_v1, source_snapshot=SOURCE_SNAPSHOT)
with mlflow.start_run(run_name=f"{spec_v1.name}-prompt-1.0.0-gate"):
    gbi.log_gate_evidence(spec_v1, estimate_v1, allocation, scores_v1, report_v1)

print(f"gate decision for prompt 1.0.0: {report_v1.decision.value}\n")
for field_result in report_v1.fields:
    binding = (
        f"binding: {field_result.binding_stratum} / "
        f"{field_result.binding_metric} lower="
        f"{field_result.binding_lower_bound:.4f} "
        f"(point {field_result.binding_point_estimate:.4f})"
        if field_result.binding_stratum
        else "no evidence"
    )
    print(
        f"  {field_result.field:12s} [{field_result.criticality}] "
        f"required ≥{field_result.required_rate:.2f} → "
        f"{field_result.decision.value.upper():6s}  {binding}"
    )
    for reason in field_result.reasons:
        print(f"      · {reason}")

# COMMAND ----------

# The gate is not advisory. Execution refuses to build so much as a SQL
# statement without an adopting report for this exact spec revision.
try:
    gbi.require_executable(spec_v1, report_v1)
except gbi.GateNotPassed as refusal:
    print(f"GateNotPassed: {refusal}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The fix loop — prompt 2.0.0, evaluated on the *same* evidence terms
# MAGIC
# MAGIC The rejection names its cause: `legacy_scan`. Prompt 2.0.0 adds
# MAGIC targeted guidance for exactly the failure modes adjudicators saw there
# MAGIC (misspelled labels, split issuer lines, box-label synonyms) and — the
# MAGIC single highest-value change — an **abstention instruction**: below the
# MAGIC spec's confidence threshold, return null and a reason instead of
# MAGIC guessing. Silent wrong values become a visible queue.
# MAGIC
# MAGIC A changed prompt is a new release candidate: new `prompt_version`, new
# MAGIC spec digest, same declared tolerances, same sample, fresh gate run.
# MAGIC (In real work expect several iterations of this loop, not one.)

# COMMAND ----------

# Built explicitly rather than by string-replacing the YAML: the prompt
# template is multi-line, so a textual substitution cannot rewrite it
# safely. The new release gets the next sequence number, which is what
# stops a late-resuming v1 job writing its output back over v2's rows.
spec_v2 = gbi.BatchInferenceSpec.model_validate(
    {
        **spec_v1.model_dump(mode="json"),
        "prompt_version": "2.0.0",
        "prompt_template": PROMPT_V2,
        "release_sequence": 2,
    }
)
# Still a reviewable, committed artifact — that has not changed.
print(spec_v2.to_yaml())

# A new release is a new budget. Prompt v2 carries the legacy-scan rules
# and the abstention instruction, so it is materially longer than v1 —
# that difference is multiplied by every row in the table. Re-estimating
# is not ceremony: reusing v1's number would authorise this run against a
# cheaper release's assumptions, and `log_gate_evidence` refuses an
# estimate whose release does not match the spec.
estimate_v2 = gbi.require_within_ceiling(estimate_for(spec_v2))
print(
    f"prompt 1.0.0 projected {estimate_v1.projected_cost_cad:.2f} CAD "
    f"(~{estimate_v1.mean_input_tokens_per_row:.0f} input tokens/row)\n"
    f"prompt 2.0.0 projected {estimate_v2.projected_cost_cad:.2f} CAD "
    f"(~{estimate_v2.mean_input_tokens_per_row:.0f} input tokens/row) — "
    f"{estimate_v2.projected_cost_cad - estimate_v1.projected_cost_cad:+.2f} "
    "CAD for the longer instruction, still inside the declared ceiling"
)

records_v2 = evaluation_records(spec_v2)
scores_v2 = gbi.score_extraction(records_v2, spec_v2, population)
report_v2 = gbi.evaluate_gate(spec_v2, scores_v2, source_snapshot=SOURCE_SNAPSHOT)

gate_run = mlflow.start_run(run_name=f"{spec_v2.name}-prompt-2.0.0-gate")
RUN_ID = gate_run.info.run_id  # provenance key for everything downstream
gbi.log_gate_evidence(spec_v2, estimate_v2, allocation, scores_v2, report_v2)

print(f"gate decision for prompt 2.0.0: {report_v2.decision.value}")
display(
    scores_frame([s for s in scores_v2 if s.stratum != gbi.WEIGHTED]).sort_values(
        ["field", "stratum", "metric"]
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evidence belongs to the release that produced it
# MAGIC
# MAGIC The most dangerous shortcut available at this point is re-using the
# MAGIC last passing evaluation for a changed prompt: "we tested this, it was
# MAGIC fine." A prompt or model change is a new application release, so the
# MAGIC previous release's numbers say nothing about it. Scores therefore
# MAGIC carry a release stamp (spec digest, model version, prompt version)
# MAGIC and the gate refuses evidence that does not match the spec being
# MAGIC gated — otherwise the digest recorded on the report would describe a
# MAGIC release the numbers never measured, and `require_executable` would
# MAGIC wave it through.
# MAGIC
# MAGIC The same check catches a subtler mismatch: intervals computed at
# MAGIC one confidence level cannot satisfy a spec that declared another.
# MAGIC
# MAGIC Binding is to the exact spec revision, so editing *any* spec field
# MAGIC obliges a re-score. That is deliberate and cheap: scoring is
# MAGIC arithmetic over the labelled records you already hold — no new
# MAGIC inference, no new labelling — so the strict rule costs almost
# MAGIC nothing to satisfy and removes a whole class of "we tested
# MAGIC something like this" reasoning.

# COMMAND ----------

# Both refusals below pass the snapshot deliberately. Omitting it would
# also raise EvidenceMismatch — for the wrong reason — and these cells
# would still print a refusal while demonstrating nothing.
try:
    # v1's numbers, v2's spec
    gbi.evaluate_gate(spec_v2, scores_v1, source_snapshot=SOURCE_SNAPSHOT)
except gbi.EvidenceMismatch as refusal:
    print(f"EvidenceMismatch: {refusal}\n")

# Evidence that claims a confidence level the spec did not declare is
# refused on the same principle.
mislabelled = (scores_v2[0].model_copy(update={"confidence": 0.99}),) + scores_v2[1:]
try:
    gbi.evaluate_gate(spec_v2, mislabelled, source_snapshot=SOURCE_SNAPSHOT)
except gbi.EvidenceMismatch as refusal:
    print(f"EvidenceMismatch: {refusal}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### The tier 1 difference — measured accuracy cannot approve itself
# MAGIC
# MAGIC Had this output fed member-facing statements, valuation, financial
# MAGIC reporting, or a regulatory submission (`use_tier: 1`), the identical
# MAGIC passing evidence would **not** authorise execution. Tier 1 returns
# MAGIC `pending_approval`: a named human accepts the residual risk, every
# MAGIC abstained row gets human review, and a rollback path is on file before
# MAGIC the run. Unreviewed acceptance is not available at this tier regardless
# MAGIC of measured accuracy — and approval can never resurrect a rejection.

# COMMAND ----------

spec_tier1 = gbi.BatchInferenceSpec.model_validate(
    {
        **spec_v2.model_dump(mode="json"),
        "use_tier": 1,
        "consumed_by": ["member_annual_statements"],
        "rollback_plan": (
            "RESTORE the entities table to the pre-run version recorded in "
            "the run metadata table; re-point consumers; notify "
            "reconciliation owners."
        ),
    }
)
# Same labelled sample, same model outputs, different spec revision — so
# the scores are re-derived for the release actually being gated.
tier1_report = gbi.evaluate_gate(
    spec_tier1,
    gbi.score_extraction(records_v2, spec_tier1, population),
    source_snapshot=SOURCE_SNAPSHOT,
)
print(f"tier 1 decision with fully passing evidence: {tier1_report.decision.value}")
for obligation in tier1_report.human_review_obligations:
    print(f"  obligation: {obligation}")

# An accountable group, not an individual's email: the value is retained
# in the gate artifact and run metadata table, and never reaches a tag.
tier1_approved = gbi.approve_gate(tier1_report, "finance-data-governance-board")
print(
    f"after named sign-off: {tier1_approved.decision.value} "
    f"(approved_by={tier1_approved.approved_by})"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 6 — EXECUTE: structured output, abstention, restartable
# MAGIC
# MAGIC The full-table statement is generated from the spec:
# MAGIC
# MAGIC - **Structured output** via `responseFormat` (JSON schema, `strict`).
# MAGIC   Nullable fields use the `[type, "null"]` form — the only union the
# MAGIC   Foundation Model structured-output subset supports — which is
# MAGIC   exactly what the abstention path needs.
# MAGIC - **The abstention is enforced, not just recorded.** A model can
# MAGIC   contradict itself — return a value *and* list the field as
# MAGIC   abstained — or answer below the threshold it was told to respect.
# MAGIC   Either way the value is nulled rather than written, because the
# MAGIC   evaluation treated it as an abstention and so it has never been
# MAGIC   through the precision gate. Landing it would put unmeasured output
# MAGIC   in a consumer's table. Stage 4 applies the identical rule, so what
# MAGIC   was measured is exactly what lands.
# MAGIC - **Row-level metadata** lands beside every value: `ai_run_id`,
# MAGIC   `ai_spec_digest`, `ai_model_version`, `ai_prompt_version`,
# MAGIC   per-field confidence and abstention flags. Confidence is kept even
# MAGIC   when abstaining — it is diagnostic for whoever works the queue.
# MAGIC - **The target schema is migrated first.** `CREATE TABLE IF NOT
# MAGIC   EXISTS` does nothing to a table that already exists, so a release
# MAGIC   that added a field would fail `INSERT *` against the old schema,
# MAGIC   and one that removed a field would leave a column still serving
# MAGIC   the previous release's values. Added columns are applied
# MAGIC   automatically. A column this release no longer produces **blocks
# MAGIC   the run**: `UPDATE SET *` / `INSERT *` expand over the *target's*
# MAGIC   columns and need every one to resolve in the source, so it would
# MAGIC   fail at analysis anyway — better to stop here, naming the
# MAGIC   statements, than inside a SQL error that never mentions releases.
# MAGIC   Dropping a column destroys data and may itself be a governance
# MAGIC   event, so that decision stays with a human.
# MAGIC - **Idempotent restart**: an anti-join selects only rows this
# MAGIC   release has not landed yet, so re-running the same statement after
# MAGIC   a partial failure finishes the job instead of paying for inference
# MAGIC   twice. A million-row job *will* fail partway at some point.
# MAGIC   (Current Databricks guidance is to submit the pending set as
# MAGIC   **one** query — AI Functions manage parallelization and retries —
# MAGIC   rather than hand-chunking batches, so that is what the builder
# MAGIC   emits. This differs deliberately from older manual-batching
# MAGIC   advice.)
# MAGIC - **Release awareness**, which is the subtle half of that: the
# MAGIC   anti-join matches on the key *and* the model and prompt versions,
# MAGIC   and the write is a `MERGE`. A prompt change is a new application
# MAGIC   release, so rows carrying the older release must be reprocessed
# MAGIC   and replaced. Matching on the key alone would skip every
# MAGIC   previously landed row — the newly gated release would report
# MAGIC   success while the table still served the old release's values and
# MAGIC   provenance, which is precisely the silent staleness this pipeline
# MAGIC   exists to prevent.
# MAGIC - **`failOnError => false`**: a poisoned document records its error
# MAGIC   (the struct's `errorMessage` field) in `ai_error` and flows to the
# MAGIC   exception queue instead of killing the run.

# COMMAND ----------

gbi.require_executable(spec_v2, report_v2)  # raises unless the gate adopted

spark.sql(gbi.create_target_table_sql(spec_v2))

# `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
# exists, so a release that changed the field set has to migrate it.
# Additive changes are applied here; a column this release no longer
# produces raises, because `INSERT *` expands over the *target's* columns
# and cannot resolve one the source does not have. Dropping it destroys
# data and may itself be a governance event, so a human decides — and the
# refusal names the statements rather than failing later inside a SQL
# error that never mentions releases.
migration = gbi.require_migrated_target(
    spec_v2, [field.name for field in spark.table(TARGET_TABLE).schema.fields]
)
for statement in migration.statements:
    spark.sql(statement)
    print(f"migrated: {statement}")

# The statement is built from the report that authorised it, so the Delta
# version it reads is the one the evidence describes — not whatever the
# table has become while the gate was being reviewed. Rows that landed
# since are not lost; they are the next cycle's work, with evidence of
# their own.
execute_sql = gbi.build_execute_sql(spec_v2, run_id=RUN_ID, report=report_v2)
print(execute_sql[:1200] + "\n…")

# "Pending" is release-aware: a row is done if this release landed it
# from this content, or if a newer release — or a newer snapshot of this
# one — already did. Taken from the module rather than restated here, so
# the count always describes the set the run will actually process. A
# hand-written copy of this predicate drifted from the real one twice
# while these ordering rules were being settled.
release_predicate = gbi.restart_predicate_sql(spec_v2, SOURCE_SNAPSHOT)

# Pinned like everything else since the snapshot was captured: counting
# pending rows against a moved table would report work the run is not
# authorised to do.
pending_sql = f"""
    SELECT count(*) AS pending FROM {PINNED_SOURCE} AS source
    LEFT ANTI JOIN {TARGET_TABLE} AS done
      ON source.doc_id = done.doc_id{release_predicate}
"""
# Strata are row metadata, not model output. If a document's `layout` was
# corrected while its text stayed the same, the restart predicate rightly
# calls the row done — but the landed label would stay wrong forever, and
# monitoring groups by it. Fix the labels directly rather than paying an
# endpoint to regenerate identical values.
spark.sql(gbi.resync_strata_sql(spec_v2, SOURCE_SNAPSHOT))

print(f"pending before run: {spark.sql(pending_sql).first().pending:,}")

if not SIMULATED:
    spark.sql(execute_sql)
else:
    # Simulated mode writes the identical schema through the identical
    # discipline — release-aware anti-join, then MERGE — with the
    # deterministic extractor standing in for ai_query.
    pending = spark.sql(f"""
        SELECT source.doc_id, source.layout, source.doc_text
        FROM {PINNED_SOURCE} AS source
        LEFT ANTI JOIN {TARGET_TABLE} AS done
          ON source.doc_id = done.doc_id{release_predicate}
        """).collect()
    gold_by_id = {
        row.doc_id: {name: row[f"gold_{name}"] for name in FIELD_NAMES}
        for row in spark.table(GOLD_TABLE).collect()
    }
    output_rows = []
    for row in pending:
        predicted, abstained = simulate_extraction(
            row.doc_id, row.layout, gold_by_id[row.doc_id], "2.0.0"
        )
        confidences = {
            name: (
                None
                if name in abstained
                else round(doc_rng(row.doc_id, f"{name}|conf").uniform(0.72, 0.99), 3)
            )
            for name in FIELD_NAMES
        }
        # The same policy the generated SQL applies: nothing the gate would
        # have treated as an abstention is allowed to land as a value.
        permitted, effective = gbi.apply_abstention_policy(
            spec_v2, predicted, confidences, abstained
        )
        record: dict = {"doc_id": row.doc_id, "layout": row.layout}
        for name in FIELD_NAMES:
            record[gbi.ai_column(name)] = permitted[name]
            record[f"{gbi.ai_column(name)}_confidence"] = confidences[name]
            record[f"{gbi.ai_column(name)}_abstained"] = name in effective
        abstained = effective
        record["ai_abstained_fields"] = sorted(abstained)
        record["ai_abstain_reason"] = (
            "confidence below threshold on scanned copy" if abstained else None
        )
        record["ai_error"] = None
        record["ai_run_id"] = RUN_ID
        record["ai_spec_digest"] = spec_v2.spec_digest
        record["ai_model_version"] = spec_v2.model_version
        record["ai_prompt_version"] = spec_v2.prompt_version
        record["ai_release_sequence"] = spec_v2.release_sequence
        # Ordering is on the pair: the release says what ran, this says
        # over which rows. Two cycles of one release tie without it.
        record["ai_source_version"] = SOURCE_SNAPSHOT.version
        # Matches Spark's sha2(col, 256): both hash the UTF-8 bytes.
        record["ai_source_digest"] = hashlib.sha256(
            row.doc_text.encode("utf-8")
        ).hexdigest()
        output_rows.append(record)
    # Explicit DDL schema (from the module's single source of truth) so
    # Spark never has to infer types from rows with empty arrays/nulls.
    write_schema = ", ".join(
        f"{name} {sql_type}"
        for name, sql_type in gbi.target_columns(spec_v2)
        if name != "ai_executed_at"
    )
    (
        spark.createDataFrame(output_rows, schema=write_schema)
        .withColumn("ai_executed_at", F.current_timestamp())
        .createOrReplaceTempView("simulated_scored")
    )
    # Same MERGE the live path uses, including the guard that matters:
    # update rows carried over from an older release, insert keys never
    # processed, and never lower a row's release sequence. Without that
    # last predicate a job whose pending set was collected before a newer
    # release committed would write its older output over the newer rows —
    # the teaching path has to carry the discipline it teaches.
    spark.sql(f"""
        MERGE INTO {TARGET_TABLE} AS target
        USING simulated_scored AS source
        ON target.doc_id = source.doc_id
        WHEN MATCHED
          AND (
            coalesce(target.ai_release_sequence, -1)
                < source.ai_release_sequence
            OR (
              coalesce(target.ai_release_sequence, -1)
                  = source.ai_release_sequence
              AND coalesce(target.ai_source_version, -1)
                  <= source.ai_source_version
            )
          )
          THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """)

print(f"pending after run:  {spark.sql(pending_sql).first().pending:,}")
print("re-running the same statement now would be a no-op — that is the restart")
print(
    "a NEW prompt or model version, however, makes every row pending again — "
    "which is what keeps a gated release from serving stale values"
)

# COMMAND ----------

display(
    spark.table(TARGET_TABLE)
    .where("size(ai_abstained_fields) > 0 OR layout = 'legacy_scan'")
    .limit(8)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 7 — LAND: provenance that survives joins, three layers deep
# MAGIC
# MAGIC Any single provenance mechanism gets lost, so use three:
# MAGIC
# MAGIC 1. **Naming** — the `ai_` prefix travels with the column through every
# MAGIC    `SELECT *` into downstream tables.
# MAGIC 2. **Unity Catalog column tags** — queryable from the information
# MAGIC    schema, so "what in this estate is AI-derived?" is answerable
# MAGIC    across the whole catalog, whatever the columns were renamed to.
# MAGIC    (Requires `APPLY TAG`; ask the platform owner for that specific
# MAGIC    grant — rule: solve permission failures with the correct narrow
# MAGIC    privilege, never a broad one.)
# MAGIC 3. **The run metadata table** — joined by `ai_run_id`, holding the
# MAGIC    spec YAML, the gate decision, the approver, projected cost, and
# MAGIC    the exact Delta version the run wrote.

# COMMAND ----------

for statement in gbi.column_tag_statements(spec_v2, RUN_ID):
    try:
        spark.sql(statement)
        print(f"tagged: {statement.split(' SET TAGS')[0]}")
    except Exception as tag_error:  # surfaced, not masked: tags matter
        print("needs APPLY TAG grant → run manually:")
        print(f"  {statement}")
        print(f"  ({tag_error})")

target_version = spark.sql(f"DESCRIBE HISTORY {TARGET_TABLE} LIMIT 1").first()[
    "version"
]

spark.sql(gbi.create_run_metadata_table_sql(spec_v2))
spark.sql(
    gbi.run_metadata_upsert_sql(
        spec_v2,
        report_v2,
        run_id=RUN_ID,
        projected_cost_cad=estimate_v2.projected_cost_cad,
        target_table_version=int(target_version),
    )
)

mlflow.log_metric("target_table_version", int(target_version))
mlflow.log_metric("rows_landed", spark.table(TARGET_TABLE).count())
mlflow.end_run()

print(f"run {RUN_ID} recorded; target table version {target_version}")
print("\nestate-wide provenance query (needs information_schema access):")
print(f"""  SELECT table_name, column_name, tag_value
  FROM {CATALOG}.information_schema.column_tags
  WHERE tag_name = 'data_source' AND tag_value = 'ai_generated'""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 8 — MONITOR: abstention leads, re-sampling verifies
# MAGIC
# MAGIC Accuracy measured at build time is not accuracy today: source data
# MAGIC drifts, and so do models. Three instruments, cheapest first:
# MAGIC
# MAGIC - **Abstention rate is the leading indicator.** It needs no labels and
# MAGIC   moves before accuracy does — a rising rate means inputs are drifting
# MAGIC   away from what the prompt was validated on. Alert on it.
# MAGIC - **The exception queue is work, and someone owns clearing it.**
# MAGIC   Abstained and errored rows are the visible form of what used to be
# MAGIC   silent errors; an unowned queue silently converts back.
# MAGIC - **Scheduled re-sampling** re-runs stages 3–5 on fresh rows on a
# MAGIC   cadence, reusing and extending the versioned gold set. A failed
# MAGIC   re-sample gate is an incident for this table, not a curiosity.

# COMMAND ----------

display(spark.sql(gbi.abstention_trend_sql(spec_v2)))

queue_view = f"{CATALOG}.{SCHEMA}.tax_document_entities_exceptions"
spark.sql(gbi.exception_queue_view_sql(spec_v2, queue_view))
queue_depth = spark.table(queue_view).count()
print(f"exception queue {queue_view}: {queue_depth} rows awaiting an owner")

# COMMAND ----------

# MAGIC %md
# MAGIC **Re-sampling job skeleton.** Schedule this notebook's stages 3–5 (or a
# MAGIC thin job wrapper around them) via a bundle resource; keep the platform
# MAGIC tag set required by `docs/tagging-standard.md` on the job cluster, with
# MAGIC values supplied by bundle variables — never hardcoded:
# MAGIC
# MAGIC ```yaml
# MAGIC resources:
# MAGIC   jobs:
# MAGIC     governed_batch_resample:
# MAGIC       name: governed-batch-inference-resample
# MAGIC       schedule:
# MAGIC         quartz_cron_expression: "0 0 6 ? * MON"   # weekly, Monday 06:00
# MAGIC         timezone_id: America/Toronto
# MAGIC       tasks:
# MAGIC         - task_key: resample_and_gate
# MAGIC           notebook_task:
# MAGIC             notebook_path: ../examples/governed-batch-inference/example_notebook
# MAGIC             base_parameters: { inference_mode: live }
# MAGIC       # job cluster with the nine required platform tags via presets/vars
# MAGIC ```
# MAGIC
# MAGIC **Output-distribution drift** belongs to Lakehouse Monitoring on the
# MAGIC target table. The current API is `w.data_quality.create_monitor(...)`
# MAGIC from `databricks-sdk` — note this **replaced the deprecated
# MAGIC `quality_monitors` API** that older guides (and the brief for this
# MAGIC notebook) reference:
# MAGIC
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks.sdk.service.dataquality import (
# MAGIC     DataProfilingConfig, Monitor, SnapshotConfig,
# MAGIC )
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC table = w.tables.get(full_name=TARGET_TABLE)
# MAGIC schema_id = w.schemas.get(full_name=f"{CATALOG}.{SCHEMA}").schema_id
# MAGIC w.data_quality.create_monitor(
# MAGIC     monitor=Monitor(
# MAGIC         object_type="table",
# MAGIC         object_id=table.table_id,
# MAGIC         data_profiling_config=DataProfilingConfig(
# MAGIC             output_schema_id=schema_id,
# MAGIC             assets_dir=f"/Workspace/Shared/monitoring/{TARGET_TABLE}",
# MAGIC             snapshot=SnapshotConfig(),
# MAGIC         ),
# MAGIC     )
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC Spend stays observable in `system.billing.usage` (AI Functions bill
# MAGIC under `MODEL_SERVING`, offering type `BATCH_INFERENCE`) — query it
# MAGIC *before* tightening policy, to learn whether you have a real problem
# MAGIC or two people experimenting.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What just happened
# MAGIC
# MAGIC | Stage | Artifact |
# MAGIC |---|---|
# MAGIC | Declare | `BatchInferenceSpec` YAML, digested, tolerances fixed pre-results |
# MAGIC | Estimate | `CostEstimate` vs ceiling; refusal demonstrated |
# MAGIC | Sample | Stratified 400-row plan; rare stratum over-sampled ~19× |
# MAGIC | Evaluate | Per-field × per-stratum precision/recall with Wilson intervals |
# MAGIC | Gate | v1 **reject** (worst stratum), v2 **adopt**; both are MLflow runs |
# MAGIC | Execute | One restartable `ai_query` statement + row metadata |
# MAGIC | Land | `ai_` prefix + UC column tags + run metadata table |
# MAGIC | Monitor | Abstention trend, owned exception queue, re-sample skeleton |
# MAGIC
# MAGIC The naive pooled view passed everything; the stratified worst-stratum
# MAGIC gate caught the broken 2%. That difference is the entire reason this
# MAGIC pipeline exists.
# MAGIC
# MAGIC `README.md` in this directory covers adapting the pattern: choosing
# MAGIC strata, setting tolerances with consumers, tier placement, endpoint
# MAGIC governance, and what to change per use case.

# COMMAND ----------

summary = {
    "gate_v1": report_v1.decision.value,
    "gate_v2": report_v2.decision.value,
    "tier1_demo": tier1_approved.decision.value,
    "run_id": RUN_ID,
    "target_table": TARGET_TABLE,
    "target_table_version": int(target_version),
    "exception_queue_depth": queue_depth,
}
print(summary)
