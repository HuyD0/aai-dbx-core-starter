---
title: Orders domain reference
covers_tables: [analytics_orders]
covers_metrics: [revenue, order_count, average_order_value, active_customers]
keywords: [orders, revenue, sales, cancellations, amounts]
---

## Grain

One row per order in `analytics_orders`. There are no order lines in this
demo model; quantity questions are order counts.

## Scope

All orders ever received, including cancelled ones. The table is a daily
snapshot load — see the `_loaded_at` column and the 24h freshness SLA in the
semantic model.

## Exclusions

Nothing is deleted. Cancelled orders remain rows; metrics that should ignore
them (revenue, average_order_value) already encode that filter. Never
recompute revenue from raw amounts without excluding cancellations.

## Encodings

`status` uses single-letter codes: S = shipped, P = processing,
C = cancelled. Filters must compare against the letter code, never against
the word (`status = 'S'`, not `status = 'shipped'`).

## Gotchas

- `amount` is gross order value; the cancelled row keeps its amount, which is
  why raw sums overstate revenue.
- The largest single amount in the demo snapshot belongs to a cancelled
  order — a classic trap for "biggest order" questions; state the status.
- `order_count` deliberately includes cancelled orders (operations view);
  say so when reporting it next to revenue.

## Common patterns

- Monthly revenue: metric `revenue` with a month grain on `order_date`.
- Share of shipped orders: `order_count` filtered to `order_status = S`
  divided by unfiltered `order_count`.
- Customer activity: `active_customers` counts distinct customers with any
  order, regardless of status.

## Cross-references

Regional questions join through the customers domain (see the customers
reference). Metric definitions and ownership live in the semantic model and
the metrics definitions reference.
