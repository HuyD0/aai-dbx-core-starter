# MLflow 3 / AIMLOps production recipes

These recipes close the learning loop around the Email Support Agent without
giving an experiment permission to deploy itself. They compose the platform's
`aai_core.agentkit` scorer catalog, cost estimator, gate, application-release,
and decision contracts; there are no project-local scorer definitions.

The safe loop is:

1. configure bounded SDK-owned traces from environment bindings;
2. register **and** start `safety` and `guidelines` in a reviewed Databricks
   notebook at a 5% all-traffic floor;
3. deterministically oversample reviewed traffic by risk (critical 100%, high
   75%, medium 20%, low 5%) for scheduled full Agentkit evaluation;
4. merge only redacted, group-reviewed `edited`/`rejected` traces carrying an
   opaque DLP evidence reference into a governed regression dataset, preserving
   nested `inputs` and `expectations`;
5. optionally align the shared `guidelines` judge with SME labels whose schema
   name is also exactly `guidelines`; MemAlign returns an unregistered proposal;
6. optionally run bounded GEPA on a third, disjoint training dataset; it returns
   a proposal and never moves an alias;
7. compare the unchanged baseline and proposed release on the same held-out
   dataset, record a `result`, then require a human `decision`.

Every connected helper defaults to `dry_run`, resolves no environment values,
does not import MLflow, and touches no backend. Execution requires both:

```python
ExecutionPolicy(mode="execute", notebook_confirmed=True)
```

and the process-local acknowledgement:

```text
AAI_ENABLE_MLFLOW_MUTATIONS=I_UNDERSTAND_THIS_MUTATES_MLFLOW
```

All workspace, experiment, prompt, dataset, model, index, and warehouse values
come through the environment variable names declared in the strict plans.
There are no environment identifiers in Python source.

## Credential-free planning

From `examples/email-support-agent`:

```bash
PYTHONPATH=../../src:src python recipes/mlflow/plan.py
```

The output includes MLflow monitoring and curation plans plus Agentkit's real
judge fan-out estimate. Retrieval relevance is budgeted per retrieved chunk,
not incorrectly as one judge call per email.

## Promotion evidence

`FullReleaseLineage` records the exact model release and inference digest;
prompt name, immutable version and content digest; index, embedding and
chunking releases/config digests; every tool's schema and implementation
digest; dataset, gate and shared scorer versions; and measured model calls,
tokens, price evidence, coverage and cost.

`assess_promotion()` never returns `adopt`. It returns only readiness and
blockers. An explicit caller may construct an unpersisted `DecisionRecord` with
`create_manual_decision()`. An adopt is refused when any of these are absent:

- passing result-gate evidence over the same baseline/change dataset;
- enforced correctness, safety, guidelines, retrieval groundedness/relevance/
  sufficiency, zero false-auto-send, and full cost-coverage metrics;
- scorer-visible `RETRIEVER` spans and MLflow documents containing
  `page_content`, `doc_uri`, `chunk_id`, and `metadata`;
- shared retrieval scorer versions in release lineage;
- connected token/call measurements, immutable price evidence,
  `cost/coverage=1.0`, and a priced cost.

Persisting the decision and changing a controlled prompt/model alias remain
separate release-workflow actions.

## Native MLflow contracts used

- production monitoring: `Scorer.register(...).start(ScorerSamplingConfig)`;
- datasets: `mlflow.genai.datasets.get_dataset(...).merge_records(...)`;
- judge alignment: `base_judge.align(..., MemAlignOptimizer(...))` with matching
  label/judge names and an explicit embedding model;
- prompt optimization: `mlflow.genai.optimize_prompts(...,
  GepaPromptOptimizer(max_metric_calls=...))` with exact prompt versions and
  disjoint calibration/training/held-out datasets.

`databricks_monitoring_notebook.py` is notebook-source-safe and remains a dry
run until both opt-ins are deliberately changed in that reviewed notebook.
