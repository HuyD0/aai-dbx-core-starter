"""Knowledge router: thin index, on-demand load, anti-staleness check."""

from pathlib import Path

import pytest

from app.knowledge import KnowledgeRouter

ROOT = Path(__file__).resolve().parents[1]


def test_router_indexes_shipped_docs():
    router = KnowledgeRouter(ROOT / "knowledge")
    assert set(router.topics) == {"customers", "metrics_definitions", "orders"}
    summary = router.index_summary()
    assert "orders" in summary
    # The index is a routing surface, not the corpus.
    assert len(summary) < 600


def test_router_loads_full_docs_on_demand():
    router = KnowledgeRouter(ROOT / "knowledge")
    doc = router.load("orders")
    assert "Gotchas" in doc.body
    assert doc.covers_tables == ("analytics_orders",)
    with pytest.raises(KeyError, match="unknown knowledge topic"):
        router.load("finance")


def test_shipped_docs_do_not_drift_from_the_model(model):
    router = KnowledgeRouter(ROOT / "knowledge")
    assert router.cross_reference_issues(model) == []


def test_drifted_front_matter_is_reported(model, tmp_path):
    (tmp_path / "legacy.md").write_text(
        "---\ntitle: Legacy\ncovers_tables: [retired_orders]\n"
        "covers_metrics: [gmv]\n---\nBody.",
        encoding="utf-8",
    )
    issues = KnowledgeRouter(tmp_path).cross_reference_issues(model)
    assert len(issues) == 2
    assert any("retired_orders" in issue for issue in issues)
    assert any("gmv" in issue for issue in issues)


def test_docs_without_front_matter_are_rejected(tmp_path):
    (tmp_path / "raw.md").write_text("No front matter.", encoding="utf-8")
    with pytest.raises(ValueError, match="front-matter"):
        KnowledgeRouter(tmp_path)
