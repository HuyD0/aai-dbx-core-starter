# Databricks notebook source
# Layer 3, context engineering: the knowledge router as a budget device.
# The published ablation behind this architecture found that handing the
# agent grep over thousands of SQL files moved accuracy by less than 1% —
# the bottleneck is mapping questions to entities, not access. So the
# system prompt carries only a thin index; full docs load on demand.

# COMMAND ----------

from pathlib import Path

from app.config import MAX_REFERENCE_DOC_CHARS, MAX_RESULT_ROWS_IN_CONTEXT
from app.knowledge import KnowledgeRouter
from app.semantics.models import load_semantic_model

ROOT = next(
    parent
    for parent in [Path.cwd(), *Path.cwd().parents]
    if (parent / "semantics" / "semantic_model.yml").exists()
)
model = load_semantic_model(ROOT / "semantics" / "semantic_model.yml")
router = KnowledgeRouter(ROOT / "knowledge")

# COMMAND ----------

# What the system prompt actually carries: the index summary plus the
# compact metric catalog — a few hundred characters, not the corpus.
index = router.index_summary()
catalog = model.metric_catalog()
print(index)
print()
print({"index_chars": len(index), "catalog_chars": len(catalog)})

# COMMAND ----------

# What loads on demand when the agent calls lookup_reference: one curated
# doc, capped at MAX_REFERENCE_DOC_CHARS. Compare the sizes.
full_corpus = sum(len(router.load(topic).body) for topic in router.topics)
one_doc = len(router.load("orders").body)
print(
    {
        "system_prompt_context": len(index) + len(catalog),
        "one_doc_on_demand": one_doc,
        "full_corpus_if_dumped": full_corpus,
        "doc_cap": MAX_REFERENCE_DOC_CHARS,
    }
)

# COMMAND ----------

# Result sets are budgeted the same way: at most MAX_RESULT_ROWS_IN_CONTEXT
# rows return to the model, with the true row_count stated so the agent can
# say "showing 20 of 3400" instead of silently truncating the analysis.
print({"row_budget": MAX_RESULT_ROWS_IN_CONTEXT})

# COMMAND ----------

# The anti-staleness contract: front-matter names tables and metrics, and
# offline CI fails when a doc references something the semantic model no
# longer defines. Docs, definitions, and data model move in one diff.
print({"cross_reference_issues": router.cross_reference_issues(model)})
