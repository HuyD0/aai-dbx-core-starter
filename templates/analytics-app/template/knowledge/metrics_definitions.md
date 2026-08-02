---
title: Metric definitions and scope of this analytics surface
covers_tables: [analytics_orders, analytics_customers]
covers_metrics: [revenue, order_count, average_order_value, active_customers]
keywords: [definitions, scope, ambiguity, refusal, governance]
---

## Grain

Definitions here restate the governed semantic model in analyst language;
the semantic model file is the source of truth and is human-curated.

## Scope

This surface answers questions about orders, revenue, customers, and
regions from the governed sales model. It does not cover anything outside
those datasets: no finance ledger, no web traffic, no HR, no external or
real-world facts. Out-of-scope questions get a polite refusal that cites
this scope statement.

## Exclusions

- revenue and average_order_value exclude cancelled orders by definition.
- order_count and active_customers include cancelled orders by definition.

## Encodings

See the orders reference for status codes and the customers reference for
region values.

## Gotchas

- "Revenue" without a timeframe is ambiguous: ask whether the user means a
  specific month, a trend, or all-time, and mention that revenue excludes
  cancellations. Do not guess a timeframe.
- "Best" or "top" without a metric is ambiguous (revenue vs order count);
  clarify before querying.
- When a question needs row-level detail (individual orders), the semantic
  layer does not apply; use the guarded raw fallback and label the lower
  provenance tier honestly.

## Common patterns

- Clarifying question: restate the two or three most plausible readings and
  ask which one is meant.
- Freshness caveat: when the freshness check reports data outside its SLA,
  lead the answer with that caveat.

## Cross-references

Orders and customers references carry per-table detail. The runbook in the
system prompt defines the search order: semantic layer first, these
references second, raw SQL last.
