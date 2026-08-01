# Cost estimation

The platform console has a **Cost estimator** section (`/estimator`) where a
developer composes the workloads a project will run — jobs, all-purpose
clusters, SQL warehouses, Lakeflow/DLT pipelines, model serving, foundation
model tokens, vector search — and sees a transparent monthly estimate with a
per-line DBU and infrastructure breakdown, a global discount, CSV export, and a
shareable URL. Its capabilities are a lean, keyless port of
[databrickslabs/lakemeter-oss](https://github.com/databrickslabs/lakemeter-oss)
fitted to this console's constraints: server-rendered, no database, no runtime
network calls, no new dependencies.

## What it is not

- **Not observed billing.** Every figure is a forward-looking projection at
  public Azure pay-as-you-go list prices, stamped with the snapshot's `as_of`
  date. Observed, governed cost telemetry is the *Cost optimization* surface,
  which stays gated until the serving-view prerequisites in
  [`ai-platform-hub.md`](ai-platform-hub.md) exist. The estimator never blurs
  that line, and the page says so persistently.
- **Not a quote.** Negotiated, committed-use, and marketplace pricing differ.
  The global discount fields exist to sketch that, not to promise it.
- **Never zero for unknown.** Anything the snapshot cannot price fails as
  `unpriceable` with the offending request path — it is never rendered as $0.
  This is the repository's standing "unknown cost is not zero" rule.

## Where the section lives

| Piece | Path |
|---|---|
| Pricing snapshot (data) | `src/platform_app/aai_console/pricing_data/azure_prices.json` |
| Snapshot models + loader | `src/platform_app/aai_console/pricing.py` |
| Request models + calculators | `src/platform_app/aai_console/estimator.py` |
| Page, fragment, behaviour | `templates/estimator.html.j2`, `templates/fragments/estimate.html.j2`, `static/estimator.js` |
| Routes | `GET /estimator`, `POST /api/estimator/render`, `POST /api/estimator/export.csv` in `server.py` |
| VM-rate refresh script | `scripts/refresh_pricing_snapshot.py` |
| Tests | `tests/test_app_estimator.py` |

The estimate itself lives client-side and in the URL hash (`#e=<base64url
JSON>`); the server is stateless and does all arithmetic, so the math has
exactly one home. The snapshot ships inside the console only — per
[`genai-lifecycle.md`](genai-lifecycle.md), changeable vendor price tables never
enter `aai-core` or the published wheel.

## The arithmetic

All formulas produce a per-line SKU breakdown (the CSV mirrors it):

- **Classic clusters** (jobs, all-purpose, DLT):
  `DBU/h = (driver_DBU + worker_DBU × workers) × photon_multiplier`, costed at
  the SKU's regional $/DBU, plus Azure VM list rates per node (workers may use
  spot; the driver never does). Usage is `hours/month` or
  `runs/day × minutes/run × days/month`.
- **Serverless jobs / all-purpose / DLT:** serverless DBU emission is workload
  dependent, so the DBU/h is a *user-supplied estimate* (labelled as such);
  jobs support the ×2 performance mode. DLT serverless bills at the jobs
  serverless rate.
- **SQL warehouses:** size ladder (2X-Small 4 DBU/h … 4X-Large 528 DBU/h)
  × max clusters × hours. Classic and Pro additionally bill the documented
  driver/worker VM equivalents per size; Serverless is DBU-only.
- **Model serving:** workload-size DBU/h × scale-out units × hours (default
  720 h — the estimate assumes no scale-to-zero; `scale_out_units` defaults to
  1, a deliberate deviation from lakemeter's concurrency presets, which can
  overstate GPU endpoints).
- **Foundation models (FMAPI):** token volumes in millions × the model's
  DBU-per-1M-token rates, or provisioned throughput units × DBU/h × hours.
- **Vector search:** `ceil(vectors_millions / vectors_per_unit)` units
  × DBU/h × hours.
- **Custom DBU line:** escape hatch for anything the snapshot does not carry
  (e.g. Lakebase) — you supply the DBU quantity and $/DBU. A snapshot-absent
  SKU has no known discount eligibility, so custom lines are excluded from the
  cross-service DBU discount unless the requester explicitly ticks them as
  eligible.
- **Discounts:** the DBU % applies only to SKUs marked eligible for
  cross-service discounting — serving, foundation-model, and vector-search
  meters are excluded, mirroring Databricks' published SKU-group exclusions.
  The VM % applies to infrastructure rows.

## The pricing snapshot

`azure_prices.json` is a machine-written (`json.dumps(indent=2,
sort_keys=True)`), strictly validated document. Loading rejects any rate ≤ 0,
any region without a full SKU price table, any instance without a VM rate, and
any warehouse ladder that misses a size — so a half-edited snapshot fails tests
and fails the container at import, loudly.

`metadata.sources` records, per table, the public URL it was curated from and
the capture date. Azure Databricks sells the Premium tier only for new
workspaces, so the snapshot carries no tier axis. When the snapshot is older
than 180 days the page shows a staleness warning.

### Refreshing rates

- **VM rates** (`vm_rates`) are refreshed by a maintainer running
  `python scripts/refresh_pricing_snapshot.py`, which queries the public,
  unauthenticated Azure Retail Prices API for every region × instance in the
  snapshot, keeps Linux consumption meters (on-demand and spot), and rewrites
  only that section plus its `as_of`. It is deliberately not wired to CI, a
  workflow, or the app runtime.
- **DBU tables** (`dbu_prices`, `instances` DBU/h, warehouse ladders, serving
  sizes, FMAPI token rates, vector-search units) are hand-curated from the
  URLs cited in `metadata.sources`. Re-check them against those pages, edit the
  JSON, bump `metadata.as_of` and the per-section `as_of`, then run
  `pytest tests/test_app_estimator.py -q`.

### Adding a region

Append `{"id", "display"}` to `regions`, add a complete SKU table under
`dbu_prices.<id>`, and run the refresh script to fill `vm_rates.<id>`. The
loader's cross-reference validation makes a partial addition fail immediately.

### Adding an instance type or foundation model

Add the entry under `instances` (with its DBU/h from the Azure Databricks
pricing page) or `fmapi.models` (DBU per 1M tokens, and/or a provisioned DBU/h),
then refresh VM rates (instances only). Serving sizes and vector-search modes
follow the same pattern.

## Boundaries it keeps

- No new Python dependencies, no environment variables, no identifier literals
  under `src/platform_app`, and the snapshot is JSON so the repository's
  YAML-parse gate is untouched.
- The two POST endpoints return HTML and CSV, not JSON; request validation
  errors surface through the console's RFC 7807 problem path without echoing
  submitted values, and estimate labels render escaped.
- CSV cells that begin with a formula character are prefixed with `'` so a
  spreadsheet never executes user-entered text.
