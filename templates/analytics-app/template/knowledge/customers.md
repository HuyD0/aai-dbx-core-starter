---
title: Customers domain reference
covers_tables: [analytics_customers]
covers_metrics: [active_customers]
keywords: [customers, regions, segmentation]
---

## Grain

One row per customer in `analytics_customers`.

## Scope

Every customer who has ever registered, whether or not they have ordered.
Weekly snapshot load (168h freshness SLA).

## Exclusions

No soft deletes in the demo model.

## Encodings

`region` holds lowercase compass names: north, south, east, west. There is
no "unknown" region; a missing join from orders means the order's customer
id has no customer row (data quality issue worth flagging).

## Gotchas

- Regional revenue questions must join orders to customers on
  `customer_id`; the semantic model's `region` dimension declares that join,
  so the semantic path handles it — avoid hand-written joins.
- A region with no orders in the period simply produces no row; say "no
  orders recorded" rather than "zero revenue" unless asked for a dense grid.

## Common patterns

- Revenue by region: metric `revenue` grouped by dimension `region`.
- Distinct buyers: `active_customers` (defined on orders, not this table).

## Cross-references

Order semantics, statuses, and amounts live in the orders reference.
