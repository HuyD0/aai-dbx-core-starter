# Databricks notebook source
# Layer 1-2 of the analytics architecture: canonical data + the semantic
# layer checked first. Credential-free by default — every cell runs against
# the versioned snapshot via FakeWarehouseExecutor. Exploration only;
# production logic lives in src/app and runs as jobs.

# COMMAND ----------

from pathlib import Path

from app.semantics.compiler import QueryFilter, SemanticQuery, TimeGrain
from app.semantics.executor import FakeWarehouseExecutor
from app.semantics.models import load_semantic_model

ROOT = next(
    parent
    for parent in [Path.cwd(), *Path.cwd().parents]
    if (parent / "semantics" / "semantic_model.yml").exists()
)
model = load_semantic_model(ROOT / "semantics" / "semantic_model.yml")
print(model.metric_catalog())

# COMMAND ----------

# Human-curated definitions are the product: the same question resolves to
# ONE governed answer instead of many plausible ones.
for name, source in model.sources.items():
    print({"source": name, "grain": source.grain, "owner": source.owner})
    for gotcha in source.gotchas:
        print(f"  gotcha: {gotcha}")

# COMMAND ----------

# The agent never writes SQL on this path — it emits a constrained plan and
# the compiler renders deterministic, dialect-portable SQL.
march = QueryFilter(dimension="order_date", value="2024-03", grain=TimeGrain.MONTH)
query = SemanticQuery(metrics=("revenue",), filters=(march,))
executor = FakeWarehouseExecutor(ROOT / "evals" / "data" / "seed_data.json")
result = executor.run_plan(model, query)
print(result.sql)
print({"columns": result.columns, "rows": result.rows})

# COMMAND ----------

# Grouped trends compile the same way; ORDER BY and LIMIT are structural.
trend = SemanticQuery(
    metrics=("revenue",),
    time_dimension="order_date",
    time_grain=TimeGrain.MONTH,
)
print(executor.run_plan(model, trend).rows)

# COMMAND ----------

# Live path (requires CAN USE on the warehouse and seeded tables; see
# README.md). The plan and SQL are identical — only the executor changes:
#
# from app.config import DEMO_CATALOG, DEMO_SCHEMA, resolve_warehouse_id
# from app.semantics.executor import DatabricksWarehouseExecutor
#
# live = DatabricksWarehouseExecutor(
#     warehouse_id=resolve_warehouse_id(None),
#     catalog=DEMO_CATALOG,
#     schema=DEMO_SCHEMA,
# )
# print(live.run_plan(model, query).rows)
