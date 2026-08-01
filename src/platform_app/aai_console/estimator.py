"""Pure list-price estimation behind the cost estimator section.

Everything here is arithmetic over the bundled pricing snapshot: no I/O, no
provider calls, no stored state. Estimates are forward-looking list-price
projections and are labelled that way everywhere they render — they are not
observed billing, and the gated observed-cost surfaces stay separate.

Error messages never repeat free-form user input. They name request paths and
known snapshot vocabulary only, so a message can safely reach a page or log.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .pricing import (
    InstanceInfo,
    PricingSnapshot,
    SkuInfo,
    snapshot_age_days,
)

MAX_LINES = 50
STALE_AFTER_DAYS = 180
ALWAYS_ON_HOURS = 720.0

KIND_LABELS = {
    "jobs_classic": "Jobs compute (classic)",
    "jobs_serverless": "Jobs compute (serverless)",
    "all_purpose_classic": "All-Purpose compute (classic)",
    "all_purpose_serverless": "All-Purpose compute (serverless)",
    "sql_warehouse": "SQL warehouse",
    "dlt_classic": "Lakeflow / DLT pipeline (classic)",
    "dlt_serverless": "Lakeflow / DLT pipeline (serverless)",
    "model_serving": "Model serving endpoint",
    "fmapi_tokens": "Foundation model — pay per token",
    "fmapi_provisioned": "Foundation model — provisioned throughput",
    "vector_search": "Vector search",
    "custom_dbu": "Custom DBU workload",
}


class EstimatorModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Label = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)
]
Key = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Usage(EstimatorModel):
    """Exactly one usage scheme: direct hours, or runs × duration × days."""

    hours_per_month: float | None = Field(default=None, gt=0, le=10_000)
    runs_per_day: int | None = Field(default=None, ge=1, le=10_000)
    avg_run_minutes: float | None = Field(default=None, gt=0, le=1_440)
    days_per_month: int | None = Field(default=None, ge=1, le=31)

    @model_validator(mode="after")
    def _exactly_one_scheme(self) -> Usage:
        run_fields = (self.runs_per_day, self.avg_run_minutes, self.days_per_month)
        has_runs = any(value is not None for value in run_fields)
        has_hours = self.hours_per_month is not None
        if has_hours and has_runs:
            raise ValueError(
                "supply either hours per month or run-based usage, not both"
            )
        if has_runs and not all(value is not None for value in run_fields):
            raise ValueError(
                "run-based usage needs runs per day, minutes per run, and "
                "days per month"
            )
        if not has_hours and not has_runs:
            raise ValueError("supply hours per month or run-based usage")
        return self

    def monthly_hours(self) -> float:
        if self.hours_per_month is not None:
            return self.hours_per_month
        assert self.runs_per_day and self.avg_run_minutes and self.days_per_month
        return self.runs_per_day * (self.avg_run_minutes / 60.0) * self.days_per_month


_ALWAYS_ON = Usage(hours_per_month=ALWAYS_ON_HOURS)


class JobsClassicLine(EstimatorModel):
    kind: Literal["jobs_classic"]
    label: Label
    driver_instance: Key
    worker_instance: Key
    num_workers: int = Field(ge=0, le=1_000)
    photon: bool = False
    spot_workers: bool = False
    usage: Usage


class AllPurposeClassicLine(EstimatorModel):
    kind: Literal["all_purpose_classic"]
    label: Label
    driver_instance: Key
    worker_instance: Key
    num_workers: int = Field(ge=0, le=1_000)
    photon: bool = False
    spot_workers: bool = False
    usage: Usage


class DltClassicLine(EstimatorModel):
    kind: Literal["dlt_classic"]
    label: Label
    edition: Literal["core", "pro", "advanced"]
    driver_instance: Key
    worker_instance: Key
    num_workers: int = Field(ge=0, le=1_000)
    photon: bool = False
    spot_workers: bool = False
    usage: Usage


class JobsServerlessLine(EstimatorModel):
    kind: Literal["jobs_serverless"]
    label: Label
    estimated_dbu_per_hour: float = Field(gt=0, le=10_000)
    performance_mode: bool = False
    usage: Usage


class AllPurposeServerlessLine(EstimatorModel):
    kind: Literal["all_purpose_serverless"]
    label: Label
    estimated_dbu_per_hour: float = Field(gt=0, le=10_000)
    usage: Usage


class DltServerlessLine(EstimatorModel):
    kind: Literal["dlt_serverless"]
    label: Label
    estimated_dbu_per_hour: float = Field(gt=0, le=10_000)
    usage: Usage


class SqlWarehouseLine(EstimatorModel):
    kind: Literal["sql_warehouse"]
    label: Label
    warehouse_type: Literal["classic", "pro", "serverless"]
    size: Key
    clusters: int = Field(default=1, ge=1, le=40)
    usage: Usage


class ModelServingLine(EstimatorModel):
    kind: Literal["model_serving"]
    label: Label
    size: Key
    scale_out_units: int = Field(default=1, ge=1, le=64)
    usage: Usage = _ALWAYS_ON


class FmapiTokensLine(EstimatorModel):
    kind: Literal["fmapi_tokens"]
    label: Label
    model: Key
    input_tokens_millions: float = Field(default=0, ge=0, le=10_000_000)
    output_tokens_millions: float = Field(default=0, ge=0, le=10_000_000)
    batch_tokens_millions: float = Field(default=0, ge=0, le=10_000_000)

    @model_validator(mode="after")
    def _some_tokens(self) -> FmapiTokensLine:
        if not any(
            (
                self.input_tokens_millions,
                self.output_tokens_millions,
                self.batch_tokens_millions,
            )
        ):
            raise ValueError("supply at least one non-zero token volume")
        return self


class FmapiProvisionedLine(EstimatorModel):
    kind: Literal["fmapi_provisioned"]
    label: Label
    model: Key
    provisioned_units: int = Field(default=1, ge=1, le=16)
    usage: Usage = _ALWAYS_ON


class VectorSearchLine(EstimatorModel):
    kind: Literal["vector_search"]
    label: Label
    mode: Key
    vectors_millions: float = Field(gt=0, le=100_000)
    usage: Usage = _ALWAYS_ON


class CustomDbuLine(EstimatorModel):
    """Escape hatch for a SKU the snapshot does not carry."""

    kind: Literal["custom_dbu"]
    label: Label
    sku_label: Label = "Custom DBU workload"
    dbu_per_month: float | None = Field(default=None, gt=0, le=100_000_000)
    dbu_per_hour: float | None = Field(default=None, gt=0, le=100_000)
    usage: Usage | None = None
    price_per_dbu: float = Field(gt=0, le=1_000)
    infra_dollars_per_month: float = Field(default=0, ge=0, le=10_000_000)

    @model_validator(mode="after")
    def _one_quantity(self) -> CustomDbuLine:
        direct = self.dbu_per_month is not None
        hourly = self.dbu_per_hour is not None
        if direct == hourly:
            raise ValueError("supply either DBUs per month or DBUs per hour")
        if hourly and self.usage is None:
            raise ValueError("DBUs per hour needs usage hours")
        if direct and self.usage is not None:
            raise ValueError("DBUs per month does not take usage hours")
        return self


Line = Annotated[
    JobsClassicLine
    | JobsServerlessLine
    | AllPurposeClassicLine
    | AllPurposeServerlessLine
    | SqlWarehouseLine
    | DltClassicLine
    | DltServerlessLine
    | ModelServingLine
    | FmapiTokensLine
    | FmapiProvisionedLine
    | VectorSearchLine
    | CustomDbuLine,
    Field(discriminator="kind"),
]


class EstimateRequest(EstimatorModel):
    region: Key
    discount_dbu_pct: float = Field(default=0, ge=0, le=100)
    discount_vm_pct: float = Field(default=0, ge=0, le=100)
    lines: tuple[Line, ...] = Field(default=(), max_length=MAX_LINES)


class EstimateError(ValueError):
    """A structurally valid request that the snapshot cannot price."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")

    def as_problem_item(self) -> dict:
        return {"path": self.path, "message": self.message, "code": "unpriceable"}


@dataclass(frozen=True)
class BreakdownRow:
    component: Literal["dbu", "vm"]
    sku: str
    quantity: float
    unit: str
    unit_price: float
    cost: float


@dataclass(frozen=True)
class LineEstimate:
    label: str
    kind: str
    kind_label: str
    sku_name: str
    dbu_per_month: float
    dbu_cost: float
    infra_cost: float
    discount: float
    total: float
    breakdown: tuple[BreakdownRow, ...]
    note: str | None = None


@dataclass(frozen=True)
class EstimateResult:
    region_id: str
    region_display: str
    discount_dbu_pct: float
    discount_vm_pct: float
    lines: tuple[LineEstimate, ...]
    dbu_subtotal: float
    infra_subtotal: float
    discount_total: float
    monthly_total: float
    currency: str
    as_of: date


def _dbu_price(snapshot: PricingSnapshot, region_id: str, sku_key: str) -> float:
    return snapshot.dbu_prices[region_id][sku_key]


def _sku(snapshot: PricingSnapshot, sku_key: str) -> SkuInfo:
    return snapshot.skus[sku_key]


def _instance(snapshot: PricingSnapshot, name: str, path: str) -> InstanceInfo:
    try:
        return snapshot.instances[name]
    except KeyError:
        raise EstimateError(path, "unknown instance type") from None


def _vm_hour_rate(
    snapshot: PricingSnapshot,
    region_id: str,
    name: str,
    *,
    spot: bool,
    path: str,
) -> tuple[float, str]:
    rate = snapshot.vm_rates[region_id][name]
    if spot:
        if rate.spot is None:
            raise EstimateError(path, "no spot rate is published for this instance")
        return rate.spot, "spot"
    return rate.on_demand, "on-demand"


def _dbu_row(sku_label: str, dbu_month: float, price: float) -> BreakdownRow:
    return BreakdownRow(
        component="dbu",
        sku=f"{sku_label} DBU",
        quantity=dbu_month,
        unit="DBU",
        unit_price=price,
        cost=dbu_month * price,
    )


def _classic_cluster_rows(
    line: JobsClassicLine | AllPurposeClassicLine | DltClassicLine,
    sku_key: str,
    photon_family: str,
    region_id: str,
    snapshot: PricingSnapshot,
    path: str,
) -> tuple[BreakdownRow, ...]:
    hours = line.usage.monthly_hours()
    driver = _instance(snapshot, line.driver_instance, f"{path}.driver_instance")
    dbu_per_hour = driver.dbu_per_hour
    rows: list[BreakdownRow] = []
    worker_rows: list[BreakdownRow] = []
    if line.num_workers:
        worker = _instance(snapshot, line.worker_instance, f"{path}.worker_instance")
        dbu_per_hour += worker.dbu_per_hour * line.num_workers
        worker_rate, worker_pricing = _vm_hour_rate(
            snapshot,
            region_id,
            line.worker_instance,
            spot=line.spot_workers,
            path=f"{path}.worker_instance",
        )
        worker_rows.append(
            BreakdownRow(
                component="vm",
                sku=(
                    f"{line.worker_instance} × {line.num_workers} workers "
                    f"({worker_pricing})"
                ),
                quantity=hours * line.num_workers,
                unit="VM hours",
                unit_price=worker_rate,
                cost=hours * line.num_workers * worker_rate,
            )
        )
    if line.photon:
        dbu_per_hour *= snapshot.multipliers.photon[photon_family]
    price = _dbu_price(snapshot, region_id, sku_key)
    rows.append(_dbu_row(_sku(snapshot, sku_key).label, dbu_per_hour * hours, price))
    driver_rate, driver_pricing = _vm_hour_rate(
        snapshot, region_id, line.driver_instance, spot=False, path=path
    )
    rows.append(
        BreakdownRow(
            component="vm",
            sku=f"{line.driver_instance} driver ({driver_pricing})",
            quantity=hours,
            unit="VM hours",
            unit_price=driver_rate,
            cost=hours * driver_rate,
        )
    )
    rows.extend(worker_rows)
    return tuple(rows)


def _serverless_rows(
    estimated_dbu_per_hour: float,
    multiplier: float,
    usage: Usage,
    sku_key: str,
    region_id: str,
    snapshot: PricingSnapshot,
) -> tuple[BreakdownRow, ...]:
    dbu_month = estimated_dbu_per_hour * multiplier * usage.monthly_hours()
    price = _dbu_price(snapshot, region_id, sku_key)
    return (_dbu_row(_sku(snapshot, sku_key).label, dbu_month, price),)


def _line_rows(
    line: Line,
    path: str,
    region_id: str,
    snapshot: PricingSnapshot,
) -> tuple[str, tuple[BreakdownRow, ...], str | None]:
    """Return (sku_key, breakdown rows, note) for one request line."""

    if isinstance(line, JobsClassicLine):
        sku_key = "JOBS_COMPUTE_PHOTON" if line.photon else "JOBS_COMPUTE"
        return (
            sku_key,
            _classic_cluster_rows(line, sku_key, "JOBS", region_id, snapshot, path),
            None,
        )
    if isinstance(line, AllPurposeClassicLine):
        sku_key = "ALL_PURPOSE_COMPUTE_PHOTON" if line.photon else "ALL_PURPOSE_COMPUTE"
        return (
            sku_key,
            _classic_cluster_rows(
                line, sku_key, "ALL_PURPOSE", region_id, snapshot, path
            ),
            None,
        )
    if isinstance(line, DltClassicLine):
        sku_key = f"DLT_{line.edition.upper()}" + ("_PHOTON" if line.photon else "")
        return (
            sku_key,
            _classic_cluster_rows(line, sku_key, "DLT", region_id, snapshot, path),
            None,
        )
    if isinstance(line, JobsServerlessLine):
        multiplier = (
            snapshot.multipliers.jobs_serverless_performance
            if line.performance_mode
            else 1.0
        )
        note = "DBU/hour is a user-supplied estimate, not a measurement."
        return (
            "JOBS_SERVERLESS",
            _serverless_rows(
                line.estimated_dbu_per_hour,
                multiplier,
                line.usage,
                "JOBS_SERVERLESS",
                region_id,
                snapshot,
            ),
            note,
        )
    if isinstance(line, AllPurposeServerlessLine):
        note = "DBU/hour is a user-supplied estimate, not a measurement."
        return (
            "ALL_PURPOSE_SERVERLESS",
            _serverless_rows(
                line.estimated_dbu_per_hour,
                1.0,
                line.usage,
                "ALL_PURPOSE_SERVERLESS",
                region_id,
                snapshot,
            ),
            note,
        )
    if isinstance(line, DltServerlessLine):
        note = "DBU/hour is a user-supplied estimate, not a measurement."
        return (
            "DLT_SERVERLESS",
            _serverless_rows(
                line.estimated_dbu_per_hour,
                1.0,
                line.usage,
                "DLT_SERVERLESS",
                region_id,
                snapshot,
            ),
            note,
        )
    if isinstance(line, SqlWarehouseLine):
        sku_key = f"SQL_{line.warehouse_type.upper()}"
        ladder = snapshot.sql_warehouses.dbu_per_hour[line.warehouse_type]
        if line.size not in ladder:
            raise EstimateError(f"{path}.size", "unknown warehouse size")
        hours = line.usage.monthly_hours()
        dbu_month = ladder[line.size] * line.clusters * hours
        price = _dbu_price(snapshot, region_id, sku_key)
        rows = [_dbu_row(_sku(snapshot, sku_key).label, dbu_month, price)]
        if line.warehouse_type in snapshot.sql_warehouses.compute_equivalents:
            compute = snapshot.sql_warehouses.compute_equivalents[line.warehouse_type][
                line.size
            ]
            for instance_name, count, role in (
                (compute.driver_instance, compute.driver_count, "driver"),
                (compute.worker_instance, compute.worker_count, "workers"),
            ):
                rate, pricing = _vm_hour_rate(
                    snapshot, region_id, instance_name, spot=False, path=path
                )
                vm_hours = hours * count * line.clusters
                rows.append(
                    BreakdownRow(
                        component="vm",
                        sku=f"{instance_name} × {count} {role} ({pricing})",
                        quantity=vm_hours,
                        unit="VM hours",
                        unit_price=rate,
                        cost=vm_hours * rate,
                    )
                )
        return sku_key, tuple(rows), None
    if isinstance(line, ModelServingLine):
        try:
            size = snapshot.model_serving.sizes[line.size]
        except KeyError:
            raise EstimateError(f"{path}.size", "unknown serving size") from None
        hours = line.usage.monthly_hours()
        dbu_month = size.dbu_per_hour * line.scale_out_units * hours
        price = _dbu_price(snapshot, region_id, "MODEL_SERVING")
        note = "Assumes the endpoint does not scale to zero for the priced hours."
        return (
            "MODEL_SERVING",
            (_dbu_row(f"{size.label} serving", dbu_month, price),),
            note,
        )
    if isinstance(line, FmapiTokensLine):
        try:
            model = snapshot.fmapi.models[line.model]
        except KeyError:
            raise EstimateError(f"{path}.model", "unknown foundation model") from None
        price = _dbu_price(snapshot, region_id, "FMAPI")
        rows = []
        for rate_type, millions in (
            ("input_token", line.input_tokens_millions),
            ("output_token", line.output_tokens_millions),
            ("batch_inference", line.batch_tokens_millions),
        ):
            if not millions:
                continue
            if rate_type not in model.per_million_tokens_dbu:
                raise EstimateError(
                    f"{path}.model",
                    f"model has no {rate_type} rate",
                )
            dbu = model.per_million_tokens_dbu[rate_type] * millions
            rows.append(
                BreakdownRow(
                    component="dbu",
                    sku=f"{model.label} {rate_type.replace('_', ' ')} DBU",
                    quantity=dbu,
                    unit="DBU",
                    unit_price=price,
                    cost=dbu * price,
                )
            )
        return "FMAPI", tuple(rows), None
    if isinstance(line, FmapiProvisionedLine):
        try:
            model = snapshot.fmapi.models[line.model]
        except KeyError:
            raise EstimateError(f"{path}.model", "unknown foundation model") from None
        if model.provisioned_dbu_per_hour is None:
            raise EstimateError(
                f"{path}.model", "model has no provisioned throughput rate"
            )
        hours = line.usage.monthly_hours()
        dbu_month = model.provisioned_dbu_per_hour * line.provisioned_units * hours
        price = _dbu_price(snapshot, region_id, "FMAPI")
        return (
            "FMAPI",
            (_dbu_row(f"{model.label} provisioned throughput", dbu_month, price),),
            None,
        )
    if isinstance(line, VectorSearchLine):
        try:
            mode = snapshot.vector_search.modes[line.mode]
        except KeyError:
            raise EstimateError(f"{path}.mode", "unknown vector search mode") from None
        units = math.ceil(line.vectors_millions / mode.vectors_millions_per_unit)
        hours = line.usage.monthly_hours()
        dbu_month = mode.dbu_per_hour_per_unit * units * hours
        price = _dbu_price(snapshot, region_id, "VECTOR_SEARCH")
        note = f"Sized at {units} search unit{'s' if units != 1 else ''}."
        return (
            "VECTOR_SEARCH",
            (_dbu_row(f"{mode.label} vector search", dbu_month, price),),
            note,
        )
    assert isinstance(line, CustomDbuLine)
    if line.dbu_per_month is not None:
        dbu_month = line.dbu_per_month
    else:
        assert line.dbu_per_hour is not None and line.usage is not None
        dbu_month = line.dbu_per_hour * line.usage.monthly_hours()
    rows = [
        BreakdownRow(
            component="dbu",
            sku=f"{line.sku_label} DBU",
            quantity=dbu_month,
            unit="DBU",
            unit_price=line.price_per_dbu,
            cost=dbu_month * line.price_per_dbu,
        )
    ]
    if line.infra_dollars_per_month:
        rows.append(
            BreakdownRow(
                component="vm",
                sku=f"{line.sku_label} infrastructure",
                quantity=1,
                unit="month",
                unit_price=line.infra_dollars_per_month,
                cost=line.infra_dollars_per_month,
            )
        )
    return "CUSTOM", tuple(rows), "Rates supplied by the requester."


def estimate_line(
    line: Line,
    index: int,
    region_id: str,
    snapshot: PricingSnapshot,
    *,
    discount_dbu_pct: float,
    discount_vm_pct: float,
) -> LineEstimate:
    path = f"$.lines[{index}]"
    sku_key, rows, note = _line_rows(line, path, region_id, snapshot)
    if sku_key == "CUSTOM":
        sku_name = line.sku_label  # rendered escaped; never enters error messages
        discountable = True
    else:
        info = _sku(snapshot, sku_key)
        sku_name = info.label
        discountable = info.cross_service_discount
    dbu_cost = sum(row.cost for row in rows if row.component == "dbu")
    infra_cost = sum(row.cost for row in rows if row.component == "vm")
    dbu_month = sum(row.quantity for row in rows if row.component == "dbu")
    discount = infra_cost * discount_vm_pct / 100.0
    if discountable:
        discount += dbu_cost * discount_dbu_pct / 100.0
    return LineEstimate(
        label=line.label,
        kind=line.kind,
        kind_label=KIND_LABELS[line.kind],
        sku_name=sku_name,
        dbu_per_month=dbu_month,
        dbu_cost=dbu_cost,
        infra_cost=infra_cost,
        discount=discount,
        total=dbu_cost + infra_cost - discount,
        breakdown=rows,
        note=note,
    )


def estimate(request: EstimateRequest, snapshot: PricingSnapshot) -> EstimateResult:
    try:
        region = snapshot.region(request.region)
    except KeyError:
        raise EstimateError("$.region", "unknown region") from None
    lines = tuple(
        estimate_line(
            line,
            index,
            region.id,
            snapshot,
            discount_dbu_pct=request.discount_dbu_pct,
            discount_vm_pct=request.discount_vm_pct,
        )
        for index, line in enumerate(request.lines)
    )
    dbu_subtotal = sum(line.dbu_cost for line in lines)
    infra_subtotal = sum(line.infra_cost for line in lines)
    discount_total = sum(line.discount for line in lines)
    return EstimateResult(
        region_id=region.id,
        region_display=region.display,
        discount_dbu_pct=request.discount_dbu_pct,
        discount_vm_pct=request.discount_vm_pct,
        lines=lines,
        dbu_subtotal=dbu_subtotal,
        infra_subtotal=infra_subtotal,
        discount_total=discount_total,
        monthly_total=dbu_subtotal + infra_subtotal - discount_total,
        currency=snapshot.metadata.currency,
        as_of=snapshot.metadata.as_of,
    )


def _csv_text(value: str) -> str:
    """Neutralise spreadsheet formula interpretation of user-entered text."""

    return f"'{value}" if value.startswith(("=", "+", "-", "@")) else value


def estimate_csv(result: EstimateResult) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["List-price estimate", f"as of {result.as_of.isoformat()}"])
    writer.writerow(["Estimates are not observed billing and are not a quote.", ""])
    writer.writerow([])
    writer.writerow(
        [
            "line",
            "label",
            "workload",
            "component",
            "sku",
            "quantity",
            "unit",
            f"unit_price_{result.currency.lower()}",
            f"cost_{result.currency.lower()}",
        ]
    )
    for number, line in enumerate(result.lines, start=1):
        for row in line.breakdown:
            writer.writerow(
                [
                    number,
                    _csv_text(line.label),
                    line.kind_label,
                    row.component,
                    _csv_text(row.sku),
                    round(row.quantity, 4),
                    row.unit,
                    round(row.unit_price, 6),
                    round(row.cost, 2),
                ]
            )
        writer.writerow(
            [
                number,
                _csv_text(line.label),
                line.kind_label,
                "total",
                _csv_text(line.sku_name),
                "",
                "",
                "",
                round(line.total, 2),
            ]
        )
    writer.writerow([])
    blank = ["", "", ""]
    writer.writerow(
        blank + ["subtotal", "DBU", "", "", "", round(result.dbu_subtotal, 2)]
    )
    writer.writerow(
        blank
        + ["subtotal", "infrastructure", "", "", "", round(result.infra_subtotal, 2)]
    )
    writer.writerow(
        blank + ["discount", "", "", "", "", round(-result.discount_total, 2) + 0.0]
    )
    writer.writerow(
        blank + ["monthly_total", "", "", "", "", round(result.monthly_total, 2)]
    )
    return buffer.getvalue()


def estimator_page_context(snapshot: PricingSnapshot, *, today: date) -> dict:
    """Everything the estimator page template needs, computed once per request."""

    metadata = snapshot.metadata
    reference_rates = snapshot.vm_rates[snapshot.regions[0].id]
    instances = [
        {
            "name": name,
            "family": info.family,
            "vcpus": info.vcpus,
            "memory_gb": info.memory_gb,
            "dbu_per_hour": info.dbu_per_hour,
            "on_demand": reference_rates[name].on_demand,
            "spot": reference_rates[name].spot,
        }
        for name, info in sorted(
            snapshot.instances.items(), key=lambda item: (item[1].family, item[0])
        )
    ]
    sku_rows = [
        {
            "key": key,
            "label": info.label,
            "cross_service_discount": info.cross_service_discount,
            "prices": {
                region.id: snapshot.dbu_prices[region.id][key]
                for region in snapshot.regions
            },
        }
        for key, info in snapshot.skus.items()
    ]
    # Chat-capable models first: the select's default must be a model that can
    # price the common input+output token request, not an embeddings model.
    fmapi_models = [
        {
            "key": key,
            "label": model.label,
            "token_rates": dict(model.per_million_tokens_dbu),
            "provisioned_dbu_per_hour": model.provisioned_dbu_per_hour,
        }
        for key, model in sorted(
            snapshot.fmapi.models.items(),
            key=lambda item: (
                "output_token" not in item[1].per_million_tokens_dbu,
                item[0],
            ),
        )
    ]
    return {
        "pricing_as_of": metadata.as_of,
        "pricing_disclaimer": metadata.disclaimer,
        "pricing_sources": metadata.sources,
        "pricing_stale": snapshot_age_days(snapshot, today) > STALE_AFTER_DAYS,
        "regions": snapshot.regions,
        "kinds": [
            {"value": value, "label": label} for value, label in KIND_LABELS.items()
        ],
        "instances": instances,
        "sku_rows": sku_rows,
        "sql_sizes": snapshot.sql_warehouses.sizes,
        "sql_dbu_per_hour": snapshot.sql_warehouses.dbu_per_hour,
        "serving_sizes": [
            {"key": key, "label": size.label, "dbu_per_hour": size.dbu_per_hour}
            for key, size in snapshot.model_serving.sizes.items()
        ],
        "fmapi_models": fmapi_models,
        "vector_modes": [
            {
                "key": key,
                "label": mode.label,
                "dbu_per_hour_per_unit": mode.dbu_per_hour_per_unit,
                "vectors_millions_per_unit": mode.vectors_millions_per_unit,
            }
            for key, mode in snapshot.vector_search.modes.items()
        ],
        "multipliers": snapshot.multipliers,
        "max_lines": MAX_LINES,
    }
