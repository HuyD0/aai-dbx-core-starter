"""Seed the demo star schema from the versioned snapshot.

Loads exactly the rows in evals/data/seed_data.json — including the fixed
_loaded_at stamps, never now() — so golden-case answers verified offline
against the FakeWarehouseExecutor hold on the live warehouse too.
Idempotent: CREATE TABLE IF NOT EXISTS plus INSERT OVERWRITE. This is
platform code, not an agent tool, which is why it uses the executor's
unguarded path; the read-only guard is an agent boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import DEMO_CATALOG, DEMO_SCHEMA, resolve_warehouse_id
from app.semantics.executor import DatabricksWarehouseExecutor

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "evals" / "data" / "seed_data.json"


def statements(seed: dict) -> list[str]:
    batch: list[str] = []
    for name, payload in seed["tables"].items():
        table = f"`{DEMO_CATALOG}`.`{DEMO_SCHEMA}`.`{name}`"
        columns = ", ".join(
            f"`{column['name']}` {column['type']}" for column in payload["columns"]
        )
        batch.append(f"CREATE TABLE IF NOT EXISTS {table} ({columns})")
        values = ", ".join(
            "(" + ", ".join(_literal(value) for value in row) + ")"
            for row in payload["rows"]
        )
        batch.append(f"INSERT OVERWRITE TABLE {table} VALUES {values}")
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse-id", default=None)
    args = parser.parse_args()

    executor = DatabricksWarehouseExecutor(
        warehouse_id=resolve_warehouse_id(args.warehouse_id),
        catalog=DEMO_CATALOG,
        schema=DEMO_SCHEMA,
    )
    seed = json.loads(SEED.read_text(encoding="utf-8"))
    for statement in statements(seed):
        executor.execute_unguarded(statement)
    for name, payload in seed["tables"].items():
        print({"table": name, "rows_loaded": len(payload["rows"])})


def _literal(value: str | None) -> str:
    if value is None:
        return "NULL"
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


if __name__ == "__main__":
    main()
