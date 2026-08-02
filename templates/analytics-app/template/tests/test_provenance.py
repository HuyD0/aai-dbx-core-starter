"""Footer round-trip: scorers parse exactly what the renderer writes."""

from app.provenance import (
    ProvenanceRecord,
    SourceTier,
    parse_footer,
    render_footer,
    strip_footer,
)

FULL = ProvenanceRecord(
    tier=SourceTier.SEMANTIC_LAYER,
    sources=("demo.sales.analytics_orders", "demo.sales.analytics_customers"),
    owner="group:test-owners",
    freshness="loaded 2024-05-01T06:00:00Z (OUTSIDE the 24h SLA)",
    rows=3,
    value="600.75",
    sql="SELECT SUM(`amount`) AS `revenue` FROM t LIMIT 100",
)
MINIMAL = ProvenanceRecord(
    tier=SourceTier.CURATED_REFERENCE, sources=("knowledge/orders.md",)
)


def test_round_trip_preserves_every_field():
    parsed = parse_footer("Answer prose.\n\n" + render_footer((FULL, MINIMAL)))
    assert parsed == (FULL, MINIMAL)


def test_multiline_values_flatten_but_still_parse():
    record = FULL.model_copy(update={"sql": "SELECT 1\n  FROM t\n  LIMIT 5"})
    parsed = parse_footer(render_footer((record,)))
    assert parsed[0].sql == "SELECT 1 FROM t LIMIT 5"


def test_text_without_footer_parses_to_nothing():
    assert parse_footer("Just an answer with no evidence.") == ()


def test_malformed_blocks_are_skipped():
    text = "[provenance]\ntier: semantic_layer\n[/provenance]"
    # No sources line -> the block is incomplete and must not parse.
    assert parse_footer(text) == ()


def test_strip_footer_returns_the_prose_only():
    answer = "The number is 42.\n\n" + render_footer((MINIMAL,))
    assert strip_footer(answer) == "The number is 42."
