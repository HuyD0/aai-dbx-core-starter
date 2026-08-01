"""Cost estimator: snapshot integrity, calculator arithmetic, and routes.

Calculator goldens run against a small fixture snapshot with round numbers so a
maintainer refreshing real list prices never breaks arithmetic tests. Tests
against the shipped snapshot assert structure and positivity, never exact
dollars. Identifier-literal hygiene for the snapshot file is already enforced
by test_app_content.py, which scans every file under src/platform_app.
"""

import json
from datetime import date
from pathlib import Path

import pytest
from asgi_client import ASGIClient
from pydantic import ValidationError

from aai_console.config import IDENTIFIER_KEYS, ConsoleConfig
from aai_console.estimator import (
    KIND_LABELS,
    EstimateError,
    EstimateRequest,
    estimate,
    estimate_csv,
    estimator_page_context,
)
from aai_console.pricing import PricingSnapshot, load_snapshot, usd
from aai_console.server import create_app

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIERS = json.loads((ROOT / "platform-identifiers.json").read_text())
SNAPSHOT_FILE = (
    ROOT / "src" / "platform_app" / "aai_console" / "pricing_data" / "azure_prices.json"
)

#: Every SKU key the calculators look up. The pydantic validator already forces
#: each of these to be priced in every region once it appears in `skus`.
REQUIRED_SKU_KEYS = {
    "JOBS_COMPUTE",
    "JOBS_COMPUTE_PHOTON",
    "JOBS_SERVERLESS",
    "ALL_PURPOSE_COMPUTE",
    "ALL_PURPOSE_COMPUTE_PHOTON",
    "ALL_PURPOSE_SERVERLESS",
    "DLT_CORE",
    "DLT_CORE_PHOTON",
    "DLT_PRO",
    "DLT_PRO_PHOTON",
    "DLT_ADVANCED",
    "DLT_ADVANCED_PHOTON",
    "DLT_SERVERLESS",
    "SQL_CLASSIC",
    "SQL_PRO",
    "SQL_SERVERLESS",
    "MODEL_SERVING",
    "FMAPI",
    "VECTOR_SEARCH",
}


def _fixture_snapshot() -> PricingSnapshot:
    source = {"url": "https://example.invalid/pricing", "as_of": "2026-01-01"}
    sku_price = {key: 0.5 for key in REQUIRED_SKU_KEYS}
    for key in ("MODEL_SERVING", "FMAPI", "VECTOR_SEARCH"):
        sku_price[key] = 0.1
    sku_price["SQL_PRO"] = 0.6
    return PricingSnapshot.model_validate(
        {
            "metadata": {
                "schema_version": 1,
                "cloud": "azure",
                "currency": "USD",
                "as_of": "2026-01-01",
                "disclaimer": "fixture",
                "sources": {
                    name: source
                    for name in (
                        "dbu_prices",
                        "instances",
                        "vm_rates",
                        "sql_warehouses",
                        "model_serving",
                        "fmapi",
                        "vector_search",
                    )
                },
            },
            "regions": [{"id": "r1", "display": "Region One"}],
            "skus": {
                key: {
                    "label": key.replace("_", " ").title(),
                    "cross_service_discount": key
                    not in {"MODEL_SERVING", "FMAPI", "VECTOR_SEARCH"},
                }
                for key in REQUIRED_SKU_KEYS
            },
            "dbu_prices": {"r1": sku_price},
            "instances": {
                "Standard_Test1": {
                    "vcpus": 4,
                    "memory_gb": 16,
                    "dbu_per_hour": 1.0,
                    "family": "General Purpose",
                },
                "Standard_Test2": {
                    "vcpus": 8,
                    "memory_gb": 32,
                    "dbu_per_hour": 2.0,
                    "family": "Memory Optimized",
                },
            },
            "vm_rates": {
                "r1": {
                    "Standard_Test1": {"on_demand": 1.0, "spot": 0.5},
                    "Standard_Test2": {"on_demand": 2.0},
                }
            },
            "multipliers": {
                "photon": {"ALL_PURPOSE": 2.0, "DLT": 2.0, "JOBS": 2.0},
                "jobs_serverless_performance": 2.0,
            },
            "sql_warehouses": {
                "sizes": ["Small", "Medium"],
                "dbu_per_hour": {
                    "classic": {"Small": 10, "Medium": 20},
                    "pro": {"Small": 10, "Medium": 20},
                    "serverless": {"Small": 10, "Medium": 20},
                },
                "compute_equivalents": {
                    warehouse_type: {
                        "Small": {
                            "driver_instance": "Standard_Test2",
                            "driver_count": 1,
                            "worker_instance": "Standard_Test1",
                            "worker_count": 4,
                        },
                        "Medium": {
                            "driver_instance": "Standard_Test2",
                            "driver_count": 1,
                            "worker_instance": "Standard_Test1",
                            "worker_count": 8,
                        },
                    }
                    for warehouse_type in ("classic", "pro")
                },
            },
            "model_serving": {
                "sizes": {
                    "cpu": {"label": "CPU", "dbu_per_hour": 1.0},
                    "gpu": {"label": "GPU", "dbu_per_hour": 10.0},
                }
            },
            "fmapi": {
                "models": {
                    "chat-model": {
                        "label": "Chat Model",
                        "per_million_tokens_dbu": {
                            "input_token": 2.0,
                            "output_token": 6.0,
                        },
                        "provisioned_dbu_per_hour": 50.0,
                    },
                    "embed-model": {
                        "label": "Embed Model",
                        "per_million_tokens_dbu": {"input_token": 1.0},
                    },
                }
            },
            "vector_search": {
                "modes": {
                    "standard": {
                        "label": "Standard",
                        "dbu_per_hour_per_unit": 4.0,
                        "vectors_millions_per_unit": 2,
                    }
                }
            },
        }
    )


FIXTURE = _fixture_snapshot()


def _request(lines: list[dict], **overrides) -> EstimateRequest:
    payload = {"region": "r1", "lines": lines, **overrides}
    return EstimateRequest.model_validate(payload)


def _jobs_classic(**overrides) -> dict:
    line = {
        "kind": "jobs_classic",
        "label": "nightly etl",
        "driver_instance": "Standard_Test1",
        "worker_instance": "Standard_Test1",
        "num_workers": 3,
        "usage": {"runs_per_day": 2, "avg_run_minutes": 90, "days_per_month": 20},
    }
    line.update(overrides)
    return line


# --- calculator goldens -----------------------------------------------------


def test_jobs_classic_runs_based_golden():
    result = estimate(_request([_jobs_classic()]), FIXTURE)
    line = result.lines[0]
    # 2 runs × 1.5 h × 20 days = 60 h; 4 DBU/h × 60 h × $0.50 = $120.
    assert line.dbu_per_month == pytest.approx(240.0)
    assert line.dbu_cost == pytest.approx(120.0)
    # Driver 60 h × $1 + workers 60 h × 3 × $1 = $240.
    assert line.infra_cost == pytest.approx(240.0)
    assert line.total == pytest.approx(360.0)
    assert line.sku_name == "Jobs Compute"
    assert result.monthly_total == pytest.approx(360.0)


def test_jobs_classic_photon_multiplies_dbu_and_switches_sku():
    result = estimate(_request([_jobs_classic(photon=True)]), FIXTURE)
    line = result.lines[0]
    assert line.dbu_per_month == pytest.approx(480.0)
    assert line.sku_name == "Jobs Compute Photon"
    assert line.infra_cost == pytest.approx(240.0)


def test_jobs_classic_spot_workers_use_spot_rate_driver_stays_on_demand():
    result = estimate(_request([_jobs_classic(spot_workers=True)]), FIXTURE)
    line = result.lines[0]
    # Driver 60 h × $1 stays on-demand; workers 180 h × $0.50 spot.
    assert line.infra_cost == pytest.approx(60.0 + 90.0)
    vm_rows = [row for row in line.breakdown if row.component == "vm"]
    assert any("spot" in row.sku for row in vm_rows)
    assert any("on-demand" in row.sku for row in vm_rows)


def test_spot_request_without_published_rate_is_unpriceable():
    line = _jobs_classic(worker_instance="Standard_Test2", spot_workers=True)
    with pytest.raises(EstimateError) as excinfo:
        estimate(_request([line]), FIXTURE)
    assert excinfo.value.path == "$.lines[0].worker_instance"


def test_all_purpose_hours_based_single_node():
    line = {
        "kind": "all_purpose_classic",
        "label": "dev box",
        "driver_instance": "Standard_Test2",
        "worker_instance": "Standard_Test2",
        "num_workers": 0,
        "photon": True,
        "usage": {"hours_per_month": 100},
    }
    result = estimate(_request([line]), FIXTURE)
    got = result.lines[0]
    # Photon ×2 on the 2-DBU driver; no worker rows for a single-node cluster.
    assert got.dbu_per_month == pytest.approx(400.0)
    assert got.infra_cost == pytest.approx(200.0)
    assert len([row for row in got.breakdown if row.component == "vm"]) == 1


def test_usage_requires_exactly_one_scheme():
    both = _jobs_classic(
        usage={
            "hours_per_month": 10,
            "runs_per_day": 1,
            "avg_run_minutes": 30,
            "days_per_month": 10,
        }
    )
    with pytest.raises(ValidationError):
        _request([both])
    with pytest.raises(ValidationError):
        _request([_jobs_classic(usage={})])
    with pytest.raises(ValidationError):
        _request([_jobs_classic(usage={"runs_per_day": 2})])


def test_jobs_serverless_performance_mode_doubles_the_estimate():
    line = {
        "kind": "jobs_serverless",
        "label": "serverless job",
        "estimated_dbu_per_hour": 5,
        "performance_mode": True,
        "usage": {"hours_per_month": 100},
    }
    result = estimate(_request([line]), FIXTURE)
    assert result.lines[0].dbu_per_month == pytest.approx(1000.0)
    assert result.lines[0].dbu_cost == pytest.approx(500.0)
    assert result.lines[0].infra_cost == 0
    assert "estimate" in result.lines[0].note


def test_sql_serverless_scales_with_clusters_and_has_no_vm_rows():
    line = {
        "kind": "sql_warehouse",
        "label": "bi warehouse",
        "warehouse_type": "serverless",
        "size": "Medium",
        "clusters": 2,
        "usage": {"hours_per_month": 10},
    }
    result = estimate(_request([line]), FIXTURE)
    got = result.lines[0]
    assert got.dbu_per_month == pytest.approx(400.0)
    assert got.dbu_cost == pytest.approx(200.0)
    assert got.infra_cost == 0


def test_sql_pro_adds_compute_equivalent_vm_cost():
    line = {
        "kind": "sql_warehouse",
        "label": "pro warehouse",
        "warehouse_type": "pro",
        "size": "Small",
        "usage": {"hours_per_month": 10},
    }
    result = estimate(_request([line]), FIXTURE)
    got = result.lines[0]
    # 10 DBU/h × 10 h × $0.60 pro rate.
    assert got.dbu_cost == pytest.approx(60.0)
    # Driver Test2 10 h × $2 + workers Test1 4 × 10 h × $1.
    assert got.infra_cost == pytest.approx(60.0)
    assert got.total == pytest.approx(120.0)


def test_dlt_edition_and_photon_select_the_sku():
    line = {
        "kind": "dlt_classic",
        "label": "pipeline",
        "edition": "advanced",
        "driver_instance": "Standard_Test1",
        "worker_instance": "Standard_Test1",
        "num_workers": 1,
        "photon": True,
        "usage": {"hours_per_month": 10},
    }
    result = estimate(_request([line]), FIXTURE)
    got = result.lines[0]
    assert got.sku_name == "Dlt Advanced Photon"
    # (1 + 1) DBU/h × photon 2 × 10 h = 40 DBU.
    assert got.dbu_per_month == pytest.approx(40.0)


def test_model_serving_defaults_to_always_on():
    line = {"kind": "model_serving", "label": "endpoint", "size": "gpu"}
    result = estimate(_request([line]), FIXTURE)
    got = result.lines[0]
    assert got.dbu_per_month == pytest.approx(7200.0)
    assert got.dbu_cost == pytest.approx(720.0)
    assert "scale to zero" in got.note


def test_fmapi_token_rows_sum_and_missing_rate_type_is_unpriceable():
    line = {
        "kind": "fmapi_tokens",
        "label": "rag chat",
        "model": "chat-model",
        "input_tokens_millions": 10,
        "output_tokens_millions": 5,
    }
    result = estimate(_request([line]), FIXTURE)
    got = result.lines[0]
    # 10 M × 2 DBU + 5 M × 6 DBU = 50 DBU × $0.10.
    assert got.dbu_per_month == pytest.approx(50.0)
    assert got.dbu_cost == pytest.approx(5.0)
    assert len(got.breakdown) == 2
    batch = dict(line, batch_tokens_millions=1)
    with pytest.raises(EstimateError) as excinfo:
        estimate(_request([batch]), FIXTURE)
    assert "batch_inference" in excinfo.value.message


def test_fmapi_provisioned_and_model_without_provisioned_rate():
    line = {
        "kind": "fmapi_provisioned",
        "label": "prov endpoint",
        "model": "chat-model",
        "provisioned_units": 2,
        "usage": {"hours_per_month": 10},
    }
    result = estimate(_request([line]), FIXTURE)
    assert result.lines[0].dbu_per_month == pytest.approx(1000.0)
    assert result.lines[0].dbu_cost == pytest.approx(100.0)
    with pytest.raises(EstimateError):
        estimate(_request([dict(line, model="embed-model")]), FIXTURE)


def test_vector_search_rounds_units_up():
    line = {
        "kind": "vector_search",
        "label": "index",
        "mode": "standard",
        "vectors_millions": 3,
        "usage": {"hours_per_month": 10},
    }
    result = estimate(_request([line]), FIXTURE)
    got = result.lines[0]
    # ceil(3 / 2) = 2 units × 4 DBU/h × 10 h = 80 DBU × $0.10.
    assert got.dbu_per_month == pytest.approx(80.0)
    assert got.dbu_cost == pytest.approx(8.0)
    assert "2 search units" in got.note


def test_custom_line_passes_rates_through():
    line = {
        "kind": "custom_dbu",
        "label": "lakebase",
        "sku_label": "Database Serverless",
        "dbu_per_month": 100,
        "price_per_dbu": 0.25,
        "infra_dollars_per_month": 10,
    }
    result = estimate(_request([line]), FIXTURE)
    got = result.lines[0]
    assert got.dbu_cost == pytest.approx(25.0)
    assert got.infra_cost == pytest.approx(10.0)
    assert got.sku_name == "Database Serverless"


def test_custom_line_discount_eligibility_is_declared_not_assumed():
    line = {
        "kind": "custom_dbu",
        "label": "mystery sku",
        "dbu_per_month": 100,
        "price_per_dbu": 1.0,
        "infra_dollars_per_month": 10,
    }
    # Default: a snapshot-absent SKU is NOT cross-service-discount eligible,
    # so only the VM discount touches its infrastructure amount.
    result = estimate(
        _request([line], discount_dbu_pct=10, discount_vm_pct=50), FIXTURE
    )
    assert result.lines[0].discount == pytest.approx(5.0)
    # Explicit opt-in applies the DBU discount too.
    opted_in = dict(line, discount_eligible=True)
    result = estimate(
        _request([opted_in], discount_dbu_pct=10, discount_vm_pct=50), FIXTURE
    )
    assert result.lines[0].discount == pytest.approx(10.0 + 5.0)


def test_discounts_split_by_sku_eligibility_and_component():
    serving = {"kind": "model_serving", "label": "endpoint", "size": "gpu"}
    request = _request(
        [_jobs_classic(), serving],
        discount_dbu_pct=10,
        discount_vm_pct=50,
    )
    result = estimate(request, FIXTURE)
    jobs, endpoint = result.lines
    # Jobs: 10% of $120 DBU + 50% of $240 VM.
    assert jobs.discount == pytest.approx(12.0 + 120.0)
    # Serving is excluded from the cross-service DBU discount and has no VM.
    assert endpoint.discount == 0
    assert result.discount_total == pytest.approx(132.0)
    assert result.monthly_total == pytest.approx(
        result.dbu_subtotal + result.infra_subtotal - result.discount_total
    )


def test_unknown_snapshot_keys_are_unpriceable_with_a_path():
    cases = [
        ({"region": "nowhere", "lines": [_jobs_classic()]}, "$.region"),
        (
            {"region": "r1", "lines": [_jobs_classic(driver_instance="Standard_X")]},
            "$.lines[0].driver_instance",
        ),
        (
            {
                "region": "r1",
                "lines": [
                    {
                        "kind": "sql_warehouse",
                        "label": "w",
                        "warehouse_type": "pro",
                        "size": "Giant",
                        "usage": {"hours_per_month": 1},
                    }
                ],
            },
            "$.lines[0].size",
        ),
    ]
    for payload, path in cases:
        with pytest.raises(EstimateError) as excinfo:
            estimate(EstimateRequest.model_validate(payload), FIXTURE)
        assert excinfo.value.path == path
        assert excinfo.value.as_problem_item()["code"] == "unpriceable"


def test_line_count_and_unknown_kind_are_schema_errors():
    with pytest.raises(ValidationError):
        _request([_jobs_classic()] * 51)
    with pytest.raises(ValidationError):
        _request([{"kind": "warp_drive", "label": "x"}])


def test_estimate_csv_guards_formulas_and_reconciles():
    request = _request([_jobs_classic(label="=SUM(A1:A9)")], discount_dbu_pct=10)
    result = estimate(request, FIXTURE)
    text = estimate_csv(result)
    assert '"\'=SUM(A1:A9)"' in text
    assert "=SUM(A1:A9)\n" not in text
    last = text.strip().splitlines()[-1]
    assert "monthly_total" in last
    assert str(round(result.monthly_total, 2)) in last
    assert "not observed billing" in text


# --- shipped snapshot integrity --------------------------------------------


def test_shipped_snapshot_loads_and_covers_every_calculator_sku():
    snapshot = load_snapshot()
    assert REQUIRED_SKU_KEYS <= set(snapshot.skus)
    assert snapshot.metadata.as_of <= date.today()
    sources = snapshot.metadata.sources
    for name in type(sources).model_fields:
        assert getattr(sources, name).as_of <= date.today(), name


def test_shipped_snapshot_is_normalised_machine_written_json():
    raw = SNAPSHOT_FILE.read_text(encoding="utf-8")
    assert raw == json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n"


def test_shipped_snapshot_prices_one_line_of_every_kind():
    snapshot = load_snapshot()
    region = snapshot.regions[0].id
    instance = next(iter(snapshot.instances))
    sql_size = snapshot.sql_warehouses.sizes[0]
    serving_size = next(iter(snapshot.model_serving.sizes))
    fmapi_model = next(
        key
        for key, model in snapshot.fmapi.models.items()
        if model.per_million_tokens_dbu and model.provisioned_dbu_per_hour
    )
    vector_mode = next(iter(snapshot.vector_search.modes))
    hours = {"usage": {"hours_per_month": 10}}
    cluster = {
        "driver_instance": instance,
        "worker_instance": instance,
        "num_workers": 1,
        **hours,
    }
    serverless = {"estimated_dbu_per_hour": 2, **hours}
    lines = [
        {"kind": "jobs_classic", "label": "a", **cluster},
        {"kind": "jobs_serverless", "label": "b", **serverless},
        {"kind": "all_purpose_classic", "label": "c", **cluster},
        {"kind": "all_purpose_serverless", "label": "d", **serverless},
        {
            "kind": "sql_warehouse",
            "label": "e",
            "warehouse_type": "pro",
            "size": sql_size,
            **hours,
        },
        {"kind": "dlt_classic", "label": "f", "edition": "core", **cluster},
        {"kind": "dlt_serverless", "label": "g", **serverless},
        {"kind": "model_serving", "label": "h", "size": serving_size, **hours},
        {
            "kind": "fmapi_tokens",
            "label": "i",
            "model": fmapi_model,
            "input_tokens_millions": 1,
        },
        {"kind": "fmapi_provisioned", "label": "j", "model": fmapi_model, **hours},
        {
            "kind": "vector_search",
            "label": "k",
            "mode": vector_mode,
            "vectors_millions": 1,
            **hours,
        },
        {
            "kind": "custom_dbu",
            "label": "l",
            "dbu_per_month": 1,
            "price_per_dbu": 0.5,
        },
    ]
    assert {line["kind"] for line in lines} == set(KIND_LABELS)
    result = estimate(
        EstimateRequest.model_validate({"region": region, "lines": lines}), snapshot
    )
    for line in result.lines:
        assert line.total > 0, line.kind


def test_page_context_carries_catalogs_and_freshness():
    snapshot = load_snapshot()
    context = estimator_page_context(snapshot, today=date(2026, 8, 1))
    assert context["regions"]
    assert context["instances"]
    assert {row["key"] for row in context["sku_rows"]} == set(snapshot.skus)
    assert context["pricing_as_of"] == snapshot.metadata.as_of
    far_future = estimator_page_context(snapshot, today=date(2030, 1, 1))
    assert far_future["pricing_stale"] is True


def test_usd_filter_formats_amounts():
    assert usd(1234.5) == "1,234.50"
    assert usd(0.0725, 4) == "0.0725"


# --- routes -----------------------------------------------------------------


@pytest.fixture
def client():
    config = ConsoleConfig(
        identifiers={key: IDENTIFIERS[key] for key in IDENTIFIER_KEYS},
        hosted=False,
        app_name=None,
    )
    return ASGIClient(create_app(config))


def _render_payload() -> dict:
    snapshot = load_snapshot()
    chat_model = next(
        key
        for key, model in snapshot.fmapi.models.items()
        if {"input_token", "output_token"} <= set(model.per_million_tokens_dbu)
    )
    return {
        "region": snapshot.regions[0].id,
        "discount_dbu_pct": 0,
        "discount_vm_pct": 0,
        "lines": [
            {
                "kind": "jobs_serverless",
                "label": "pipeline",
                "estimated_dbu_per_hour": 4,
                "usage": {"hours_per_month": 100},
            },
            {
                "kind": "fmapi_tokens",
                "label": "assistant",
                "model": chat_model,
                "input_tokens_millions": 10,
                "output_tokens_millions": 2,
            },
        ],
    }


def test_estimator_page_renders_with_stamp_and_boundaries(client):
    response = client.get("/estimator")
    assert response.status_code == 200
    assert "Cost estimator" in response.text
    assert "not observed billing" in response.text
    assert load_snapshot().metadata.as_of.isoformat() in response.text
    assert 'href="/optimization"' in response.text
    assert "Rates reference" in response.text


def test_render_fragment_matches_direct_estimate(client):
    payload = _render_payload()
    response = client.post("/api/estimator/render", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    expected = estimate(EstimateRequest.model_validate(payload), load_snapshot())
    assert usd(expected.monthly_total) in response.text
    assert response.text.count('data-remove-line="') == len(payload["lines"])


def test_render_with_no_lines_shows_empty_state(client):
    response = client.post(
        "/api/estimator/render", json={"region": "eastus", "lines": []}
    )
    assert response.status_code == 200
    assert "No workloads in the estimate yet" in response.text


def test_invalid_payloads_are_rfc7807_and_do_not_echo_labels(client):
    sentinel = "SENTINEL-DO-NOT-ECHO"
    bad = {
        "region": "eastus",
        "lines": [
            {
                "kind": "jobs_serverless",
                "label": sentinel,
                "estimated_dbu_per_hour": 4,
                "usage": {"hours_per_month": 100},
                "surprise": True,
            }
        ],
    }
    response = client.post("/api/estimator/render", json=bad)
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert sentinel not in response.text
    assert response.json()["errors"]

    unknown_region = dict(_render_payload(), region="atlantis")
    response = client.post("/api/estimator/render", json=unknown_region)
    assert response.status_code == 422
    body = response.json()
    assert body["errors"][0]["code"] == "unpriceable"
    assert "atlantis" not in response.text


def test_scalar_coercion_is_rejected_not_silently_applied(client):
    """Strict scalars: crafted payloads cannot shift pricing via coercion."""
    coerced_bool = {
        "region": load_snapshot().regions[0].id,
        "discount_dbu_pct": 10,
        "lines": [
            {
                "kind": "custom_dbu",
                "label": "probe",
                "dbu_per_month": 100,
                "price_per_dbu": 1,
                "discount_eligible": 1,
            }
        ],
    }
    assert client.post("/api/estimator/render", json=coerced_bool).status_code == 422
    stringly_number = _render_payload()
    stringly_number["lines"][0]["usage"] = {"hours_per_month": "100"}
    response = client.post("/api/estimator/render", json=stringly_number)
    assert response.status_code == 422
    assert response.json()["errors"]


def test_labels_render_escaped_in_the_fragment(client):
    payload = _render_payload()
    payload["lines"][0]["label"] = "<script>alert(1)</script>"
    response = client.post("/api/estimator/render", json=payload)
    assert response.status_code == 200
    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;" in response.text


def test_csv_export_streams_the_same_estimate(client):
    payload = _render_payload()
    response = client.post("/api/estimator/export.csv", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    expected = estimate(EstimateRequest.model_validate(payload), load_snapshot())
    assert str(round(expected.monthly_total, 2)) in response.text
    assert "monthly_total" in response.text


def test_too_many_lines_is_a_422(client):
    payload = _render_payload()
    payload["lines"] = [payload["lines"][0]] * 51
    assert client.post("/api/estimator/render", json=payload).status_code == 422
