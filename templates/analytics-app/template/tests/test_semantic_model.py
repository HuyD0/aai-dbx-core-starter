"""The semantic contract rejects drift instead of absorbing it."""

import pytest
from pydantic import ValidationError

from app.semantics.models import SemanticModel


def test_demo_model_validates_and_summarizes(model):
    assert model.table_names() == {"analytics_orders", "analytics_customers"}
    catalog = model.metric_catalog()
    assert "revenue" in catalog
    assert "dimensions available" in catalog
    assert "yaml" not in catalog.lower()


def test_metric_referencing_unknown_source_is_rejected(model_payload):
    model_payload["metrics"]["revenue"]["source"] = "payments"
    with pytest.raises(ValidationError, match="unknown source"):
        SemanticModel.model_validate(model_payload)


def test_dimension_referencing_unknown_source_is_rejected(model_payload):
    model_payload["dimensions"]["region"]["source"] = "geo"
    with pytest.raises(ValidationError, match="unknown source"):
        SemanticModel.model_validate(model_payload)


def test_extra_keys_are_forbidden(model_payload):
    model_payload["metrics"]["revenue"]["sql"] = "SELECT 1"
    with pytest.raises(ValidationError):
        SemanticModel.model_validate(model_payload)


def test_two_part_table_names_are_rejected(model_payload):
    model_payload["sources"]["orders"]["table"] = "sales.analytics_orders"
    with pytest.raises(ValidationError, match="three-part"):
        SemanticModel.model_validate(model_payload)


def test_names_must_be_snake_case(model_payload):
    model_payload["metrics"]["Revenue-EU"] = model_payload["metrics"].pop("revenue")
    with pytest.raises(ValidationError, match="snake_case"):
        SemanticModel.model_validate(model_payload)


def test_free_sql_metric_filters_are_rejected(model_payload):
    model_payload["metrics"]["revenue"]["filter"] = "status <> 'C'"
    with pytest.raises(ValidationError):
        SemanticModel.model_validate(model_payload)
