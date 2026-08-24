# Promotion-gate demo fixtures

Six fully synthetic method reports used by notebook
`08_mlflow_and_promotion.ipynb` and `tests/test_promotion_demo.py` to
demonstrate `decide_lora_promotion` accepting an intact change and rejecting a
schema-degraded copy.

Every value here is an invented classroom number. The intents are prefixed
`demo_` and no field is derived from the Bitext dataset or any other
third-party data, so nothing in this directory carries the CDLA-Sharing-1.0
obligations described in `DATA_LICENSE.md`.

The `evaluation_execution_contract_sha256`, `base_model`, and training-lineage
fields hold deterministic placeholders; `bind_promotion_demo_reports` replaces
them with the live machine's verified identity at run time so the real gates
execute. These files are teaching material only and must never be presented,
persisted, or logged as evaluation evidence.
