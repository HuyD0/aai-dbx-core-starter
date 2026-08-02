"""Azure list-price snapshot backing the cost estimator.

The snapshot is a bundled JSON document — never a live pricing call. The console
runs without outbound network access, so estimates are computed from public list
prices captured at a known date and every rendered figure carries that date.
`scripts/refresh_pricing_snapshot.py` refreshes the VM rates from the public
Azure Retail Prices API on a maintainer's machine; the DBU tables are curated by
hand from the pages cited in ``metadata.sources``.

Rates are strictly positive by model contract, so a half-filled snapshot fails
at load — in the container that means at import, loud and early.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

PACKAGE_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH = PACKAGE_DIR / "pricing_data" / "azure_prices.json"

# Strictly positive: a placeholder zero can never ship as a price.
Rate = Annotated[float, Field(gt=0)]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
HttpsUrl = Annotated[str, StringConstraints(pattern=r"^https://")]
RegionId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9]*$")]
SkuKey = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]*$")]
InstanceName = Annotated[str, StringConstraints(pattern=r"^Standard_[A-Za-z0-9_]+$")]
SizeName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9-]+$")]
CatalogKey = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9._-]*$")]

PHOTON_FAMILIES = ("ALL_PURPOSE", "DLT", "JOBS")
WAREHOUSE_TYPES = ("classic", "pro", "serverless")
TOKEN_RATE_TYPES = ("batch_inference", "input_token", "output_token")


class PricingModel(BaseModel):
    """Persisted-data boundary: frozen, no unknown keys, strict scalars.

    Strictness is what makes "a bad snapshot fails at load" true for typos as
    well as structure: a hand-edited ``true`` in a rate field or ``1`` in
    ``cross_service_discount`` must stop the app, not price estimates.
    Validation happens in JSON mode (see ``load_snapshot``), where strict
    still accepts JSON ints for floats, ISO strings for dates, and arrays for
    tuples.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SourceRef(PricingModel):
    url: HttpsUrl
    as_of: date


class SnapshotSources(PricingModel):
    dbu_prices: SourceRef
    instances: SourceRef
    vm_rates: SourceRef
    sql_warehouses: SourceRef
    model_serving: SourceRef
    fmapi: SourceRef
    vector_search: SourceRef


class SnapshotMetadata(PricingModel):
    schema_version: Literal[1]
    cloud: Literal["azure"]
    currency: Literal["USD"]
    as_of: date
    disclaimer: NonEmptyStr
    sources: SnapshotSources


class Region(PricingModel):
    id: RegionId
    display: NonEmptyStr


class SkuInfo(PricingModel):
    label: NonEmptyStr
    cross_service_discount: bool


class InstanceInfo(PricingModel):
    vcpus: Annotated[int, Field(gt=0)]
    memory_gb: Rate
    dbu_per_hour: Rate
    family: NonEmptyStr


class VmRate(PricingModel):
    on_demand: Rate
    # None means Azure publishes no spot rate for the instance in that region.
    spot: Rate | None = None


class Multipliers(PricingModel):
    photon: dict[SkuKey, Rate]
    jobs_serverless_performance: Rate

    @model_validator(mode="after")
    def _photon_families(self) -> Multipliers:
        if tuple(sorted(self.photon)) != PHOTON_FAMILIES:
            raise ValueError(f"photon multipliers must cover {PHOTON_FAMILIES}")
        return self


class WarehouseCompute(PricingModel):
    """Azure classic/pro warehouses bill their VMs alongside the DBUs."""

    driver_instance: InstanceName
    driver_count: Annotated[int, Field(ge=1)]
    worker_instance: InstanceName
    worker_count: Annotated[int, Field(ge=1)]


class SqlWarehouses(PricingModel):
    sizes: tuple[SizeName, ...] = Field(min_length=1)
    dbu_per_hour: dict[str, dict[SizeName, Rate]]
    compute_equivalents: dict[str, dict[SizeName, WarehouseCompute]]

    @model_validator(mode="after")
    def _complete_ladders(self) -> SqlWarehouses:
        if tuple(sorted(self.dbu_per_hour)) != WAREHOUSE_TYPES:
            raise ValueError(f"dbu_per_hour must cover {WAREHOUSE_TYPES}")
        if tuple(sorted(self.compute_equivalents)) != ("classic", "pro"):
            raise ValueError("compute_equivalents must cover classic and pro")
        ladders: list[tuple[str, dict]] = [
            (f"dbu_per_hour.{name}", ladder)
            for name, ladder in self.dbu_per_hour.items()
        ]
        ladders += [
            (f"compute_equivalents.{name}", ladder)
            for name, ladder in self.compute_equivalents.items()
        ]
        # Set equality, not order: the snapshot file is normalised with
        # sort_keys, so display order always comes from `sizes`.
        for name, ladder in ladders:
            if set(ladder) != set(self.sizes):
                raise ValueError(f"{name} must cover exactly the listed sizes")
        return self


class ServingSize(PricingModel):
    label: NonEmptyStr
    dbu_per_hour: Rate


class ModelServing(PricingModel):
    sizes: dict[CatalogKey, ServingSize] = Field(min_length=1)


class FmapiModel(PricingModel):
    label: NonEmptyStr
    per_million_tokens_dbu: dict[str, Rate]
    provisioned_dbu_per_hour: Rate | None = None

    @model_validator(mode="after")
    def _known_rate_types(self) -> FmapiModel:
        unknown = sorted(set(self.per_million_tokens_dbu) - set(TOKEN_RATE_TYPES))
        if unknown:
            raise ValueError(f"unknown token rate types: {unknown}")
        if not self.per_million_tokens_dbu and self.provisioned_dbu_per_hour is None:
            raise ValueError("a model must carry token rates or a provisioned rate")
        return self


class Fmapi(PricingModel):
    models: dict[CatalogKey, FmapiModel] = Field(min_length=1)


class VectorSearchMode(PricingModel):
    label: NonEmptyStr
    dbu_per_hour_per_unit: Rate
    vectors_millions_per_unit: Annotated[int, Field(gt=0)]


class VectorSearch(PricingModel):
    modes: dict[CatalogKey, VectorSearchMode] = Field(min_length=1)


class PricingSnapshot(PricingModel):
    metadata: SnapshotMetadata
    regions: tuple[Region, ...] = Field(min_length=1)
    skus: dict[SkuKey, SkuInfo]
    dbu_prices: dict[RegionId, dict[SkuKey, Rate]]
    instances: dict[InstanceName, InstanceInfo]
    vm_rates: dict[RegionId, dict[InstanceName, VmRate]]
    multipliers: Multipliers
    sql_warehouses: SqlWarehouses
    model_serving: ModelServing
    fmapi: Fmapi
    vector_search: VectorSearch

    @model_validator(mode="after")
    def _cross_references(self) -> PricingSnapshot:
        region_ids = [region.id for region in self.regions]
        if len(set(region_ids)) != len(region_ids):
            raise ValueError("region ids must be unique")
        problems: list[str] = []
        sku_keys = set(self.skus)
        instance_names = set(self.instances)
        for table_name, table in ("dbu_prices", self.dbu_prices), (
            "vm_rates",
            self.vm_rates,
        ):
            if set(table) != set(region_ids):
                problems.append(f"{table_name} must cover exactly the listed regions")
        for region_id, prices in self.dbu_prices.items():
            if set(prices) != sku_keys:
                problems.append(f"dbu_prices[{region_id}] must price every SKU")
        for region_id, rates in self.vm_rates.items():
            if set(rates) != instance_names:
                problems.append(f"vm_rates[{region_id}] must rate every instance")
        for warehouse_type, ladder in self.sql_warehouses.compute_equivalents.items():
            for size, compute in ladder.items():
                for instance in (compute.driver_instance, compute.worker_instance):
                    if instance not in instance_names:
                        problems.append(
                            f"sql_warehouses {warehouse_type} {size} references "
                            f"unknown instance {instance}"
                        )
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def region(self, region_id: str) -> Region:
        for region in self.regions:
            if region.id == region_id:
                return region
        raise KeyError(region_id)


def load_snapshot(path: Path = SNAPSHOT_PATH) -> PricingSnapshot:
    return PricingSnapshot.model_validate_json(path.read_text(encoding="utf-8"))


def snapshot_age_days(snapshot: PricingSnapshot, today: date) -> int:
    return max((today - snapshot.metadata.as_of).days, 0)


def usd(value: float, precision: int = 2) -> str:
    """Jinja filter: a plain formatted amount; templates add the $ sign."""

    return f"{value:,.{precision}f}"
