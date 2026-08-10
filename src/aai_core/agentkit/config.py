"""``agentkit.yaml`` configuration and the project composition root.

Three lines are enough::

    version: 1
    agent: src/app/example_agent.py:respond
    dataset: evals/data/golden_cases.json

Everything else is inferred or defaulted; the optional keys are escape
hatches, not the normal path. The file holds logical names only — endpoint
deployments, hosts, and identifiers stay in ``aai-platform.yml`` and the
platform environment.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import yaml
from pydantic import Field, ValidationError, field_serializer, field_validator

from aai_core.agentkit.cost import DEFAULT_CHUNKS_PER_ROW
from aai_core.agentkit.errors import ConfigError, UnknownScorerError
from aai_core.contracts import ContractModel, freeze_value, thaw_value
from aai_core.evaluation import (
    MetricDirection,
    MetricRule,
)
from aai_core.evaluation import (
    judge_model_uri as resolve_judge_model_uri,
)
from aai_core.providers.types import ProviderConfigurationError
from aai_core.runtime import PlatformSettings, find_platform_config

if TYPE_CHECKING:
    from aai_core.experiments import ExperimentManager
    from aai_core.prompts import PromptManager


class _ServingEndpoints(Protocol):
    def get(self, endpoint: str) -> Any:
        raise NotImplementedError


class _WorkspaceClient(Protocol):
    serving_endpoints: _ServingEndpoints


CONFIG_FILENAME = "agentkit.yaml"
CONFIG_ENV = "AGENTKIT_CONFIG"
_THRESHOLD_OPERATORS = (">=", "<=", ">", "<")


def _as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _as_float(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return float(value)
    return value


class ScorersConfig(ContractModel):
    """Scorer selection — reference the shared catalog, never redefine it."""

    add: tuple[str, ...] = ()
    remove: tuple[str, ...] = ()
    judge_model: str = Field(default="judge-model", min_length=1)
    guidelines: tuple[str, ...] = ()

    @field_validator("add", "remove", "guidelines", mode="before")
    @classmethod
    def coerce_sequences(cls, value: Any) -> Any:
        return _as_tuple(value)


class BaselineConfig(ContractModel):
    file: str = Field(default="evals/baseline.json", min_length=1)
    run_id: str | None = None


class BudgetConfig(ContractModel):
    max_judge_calls: int | None = Field(default=None, ge=1)
    judge_price_per_1m_tokens: float | None = Field(default=None, gt=0.0)
    # MLflow judges retrieval relevance once per retrieved chunk. Before a
    # live run there are no traces to count, so the estimate needs the
    # retriever's `k` — the one number only the project knows.
    retrieved_chunks_per_row: int = Field(default=DEFAULT_CHUNKS_PER_ROW, ge=1, le=1000)

    @field_validator("judge_price_per_1m_tokens", mode="before")
    @classmethod
    def coerce_price(cls, value: Any) -> Any:
        return _as_float(value)


class SmokeConfig(ContractModel):
    rows: int = Field(default=20, ge=1, le=200)
    answer_sheet: str | None = None


class RequestMapping(ContractModel):
    """Field mapping for generic HTTP/JSON targets (Foundry included)."""

    request_field: str = Field(default="input", min_length=1)
    response_field: str = Field(default="output", min_length=1)
    extra_body: Mapping[str, Any] = Field(default_factory=dict)
    auth_env: str | None = Field(default=None, min_length=1)

    @field_validator("extra_body", mode="after")
    @classmethod
    def freeze_body(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], freeze_value(value))

    @field_serializer("extra_body")
    def serialize_body(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], thaw_value(value))


class AgentkitConfig(ContractModel):
    """The whole project configuration surface."""

    version: Literal[1]
    agent: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    scorers: ScorersConfig = Field(default_factory=ScorersConfig)
    thresholds: Mapping[str, str] = Field(default_factory=dict)
    regression_budget: Mapping[str, float] = Field(default_factory=dict)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    smoke: SmokeConfig = Field(default_factory=SmokeConfig)
    strata: tuple[str, ...] = ()
    request_mapping: RequestMapping = Field(default_factory=RequestMapping)
    concurrency: int = Field(default=8, ge=1, le=64)
    # The Unity Catalog model this project promotes into, if any. Evidence
    # reads its deployment-job approval tags to report who approved.
    registered_model: str | None = Field(default=None, min_length=1)
    # The deployment job's approval task names. Every one must carry an
    # `Approved` tag on the model version for evidence to report approval.
    # Without them, evidence can only report the tags that exist — which
    # cannot distinguish an approved gate from a stale tag left behind by a
    # renamed task, and says so rather than implying otherwise.
    approvals: tuple[str, ...] = ()

    @field_validator("strata", "approvals", mode="before")
    @classmethod
    def coerce_strata(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("regression_budget", mode="before")
    @classmethod
    def coerce_budget_values(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: _as_float(item) for key, item in value.items()}
        return value

    @field_validator("thresholds", "regression_budget", mode="after")
    @classmethod
    def freeze_mappings(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], freeze_value(value))

    @field_serializer("thresholds", "regression_budget")
    def serialize_mappings(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], thaw_value(value))


def parse_threshold(metric: str, expression: str) -> MetricRule:
    """Parse a ``">=0.7"``-style expression into a gate rule.

    ``>`` and ``<`` are exact: the bound moves one float ULP so the inclusive
    gate engine implements the strict comparison precisely.
    """

    text = str(expression).strip()
    for operator in _THRESHOLD_OPERATORS:
        if not text.startswith(operator):
            continue
        number_text = text[len(operator) :].strip()
        try:
            value = float(number_text)
        except ValueError:
            break
        if not math.isfinite(value):
            break
        if operator == ">=":
            return MetricRule(
                metric=metric, direction=MetricDirection.HIGHER, required=value
            )
        if operator == "<=":
            return MetricRule(
                metric=metric, direction=MetricDirection.LOWER, required=value
            )
        if operator == ">":
            return MetricRule(
                metric=metric,
                direction=MetricDirection.HIGHER,
                required=math.nextafter(value, math.inf),
            )
        return MetricRule(
            metric=metric,
            direction=MetricDirection.LOWER,
            required=math.nextafter(value, -math.inf),
        )
    raise ConfigError(
        f"threshold for {metric!r} must be a comparison like '>=0.7', "
        f"'<=2', '>0' or '<1'; got {expression!r}",
        remediation="Edit the thresholds block in agentkit.yaml.",
    )


def find_agentkit_config(
    start: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Locate ``agentkit.yaml`` like ``find_platform_config`` locates its file.

    Resolution order: the ``AGENTKIT_CONFIG`` environment variable, then an
    upward search from ``start`` (default: the working directory). Returns
    the conventional path in the start directory when nothing is found so
    missing-file errors stay clear.
    """

    env = environ if environ is not None else os.environ
    override = env.get(CONFIG_ENV)
    if override:
        return Path(override)
    base = Path(start) if start is not None else Path.cwd()
    for directory in (base, *base.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return base / CONFIG_FILENAME


def load_config(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> AgentkitConfig:
    """Load and fully validate an AgentKit project configuration."""

    config_path = (
        Path(path) if path is not None else find_agentkit_config(environ=environ)
    )
    if not config_path.is_file():
        raise ConfigError(
            f"no {CONFIG_FILENAME} found (looked at {config_path})",
            remediation=(
                "Run `agentkit init --name <project>` to scaffold a project, "
                "or pass --config with the file's path."
            ),
        )
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"{config_path} is not valid YAML: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping")
    try:
        config = AgentkitConfig(**loaded)
    except ValidationError as error:
        raise ConfigError(f"{config_path} is invalid: {error}") from error
    _validate_scorer_references(config)
    for metric, expression in config.thresholds.items():
        parse_threshold(metric, expression)
    for metric, value in config.regression_budget.items():
        if not math.isfinite(value) or value < 0:
            raise ConfigError(
                f"regression_budget for {metric!r} must be a finite number >= 0"
            )
    return config


class ProjectContext:
    """Joins ``agentkit.yaml`` with the governed platform settings.

    Transient composition state — deliberately a plain class, not a contract
    model. ``PlatformSettings`` supplies everything the config omits: the
    experiment name, resource tags, Unity Catalog locations, and the judge
    deployment behind the logical model name.
    """

    def __init__(
        self,
        *,
        config: AgentkitConfig,
        settings: PlatformSettings,
        root: Path,
    ) -> None:
        self.config = config
        self.settings = settings
        self.root = root

    @classmethod
    def load(
        cls,
        config_path: str | Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> ProjectContext:
        path = (
            Path(config_path)
            if config_path is not None
            else find_agentkit_config(environ=environ)
        )
        config = load_config(path, environ=environ)
        root = path.resolve().parent
        # A project's own aai-platform.yml wins over the AAI_PLATFORM_CONFIG
        # override. `agentkit.yaml` anchors the project, so an ambient
        # variable pointing at another project's configuration must not
        # silently redirect this one's judge endpoint or catalog.
        colocated = root / "aai-platform.yml"
        platform_path = (
            colocated
            if colocated.is_file()
            else find_platform_config(root, environ=environ)
        )
        environment = dict(environ) if environ is not None else None
        settings = PlatformSettings.load(platform_path, environ=environment)
        return cls(config=config, settings=settings, root=root)

    @property
    def baseline_path(self) -> Path:
        return self.root / self.config.baseline.file

    @property
    def results_dir(self) -> Path:
        return self.root / ".aai" / "agentkit" / "results"

    @property
    def evidence_dir(self) -> Path:
        return self.root / ".aai" / "agentkit" / "evidence"

    def experiment_manager(self, mlflow_module: Any | None = None) -> ExperimentManager:
        from aai_core.experiments import ExperimentManager

        return ExperimentManager(
            experiment_name=self.settings.effective_experiment_name,
            context=self.settings.resource,
            mlflow_module=mlflow_module,
        )

    def prompt_manager(self, mlflow_module: Any | None = None) -> PromptManager:
        from aai_core.prompts import PromptManager

        return PromptManager(
            context=self.settings.resource,
            catalog=self.settings.catalog,
            schema=self.settings.schema,
            mlflow_module=mlflow_module,
        )

    def judge_model_uri(self, logical_name: str | None = None) -> str:
        """Resolve the logical judge name to a governed serving endpoint."""

        name = logical_name or self.config.scorers.judge_model
        try:
            return resolve_judge_model_uri(self.settings, name)
        except ProviderConfigurationError as error:
            raise ConfigError(
                error.args[0] if error.args else str(error),
                remediation=error.remediation,
            ) from error

    def judge_model_identity(
        self,
        logical_name: str | None = None,
        *,
        client: _WorkspaceClient | None = None,
    ) -> str | None:
        """What the judge endpoint currently serves, if it can be read.

        ``endpoints:/pension-judge`` is a stable *name* pointing at a
        mutable thing: the platform team can repoint it at another model
        or promote a new version without the URI changing, and two runs
        would then look comparable while being judged by different models.
        This resolves the served entity so the comparison has something
        immutable to compare.

        Best effort by design. Reading an endpoint's configuration needs a
        permission (`CAN_VIEW`) that a least-privilege CI principal
        holding only `CAN_QUERY` may not have, and section 4 of AGENTS.md
        forbids widening a grant to make a check work. When it cannot be
        read the run says the judge could not be pinned rather than
        pretending it verified one.
        """

        endpoint = self.judge_model_uri(logical_name).removeprefix("endpoints:/")
        try:
            if client is None:
                from aai_core.identity import databricks_workspace_client

                client = cast(_WorkspaceClient, databricks_workspace_client())
            served = client.serving_endpoints.get(endpoint)
        except Exception:  # noqa: BLE001 - absent SDK, auth, or permission
            return None
        return _served_entity_identity(served)


def _validate_scorer_references(config: AgentkitConfig) -> None:
    contradictory = sorted(set(config.scorers.add) & set(config.scorers.remove))
    if contradictory:
        raise ConfigError(
            "the same scorer cannot appear in both scorers.add and "
            "scorers.remove: " + ", ".join(contradictory),
            remediation="Keep each scorer in only one selection list.",
        )

    from aai_core.agentkit.catalog import CATALOG

    known = {spec.name for spec in CATALOG}
    for field_name, names in (
        ("scorers.add", config.scorers.add),
        ("scorers.remove", config.scorers.remove),
    ):
        unknown = sorted(set(names) - known)
        if unknown:
            raise UnknownScorerError(
                f"{field_name} references unknown scorer(s): "
                f"{', '.join(unknown)}. Known scorers: "
                f"{', '.join(sorted(known))}",
                remediation=(
                    "Run `agentkit scorers ls` to browse the shared registry. "
                    "Projects reference scorers by name; they never define "
                    "new ones."
                ),
            )
    metric_by_name = {spec.name: spec.metric for spec in CATALOG}
    for name in config.scorers.remove:
        metric = metric_by_name.get(name)
        # A regression budget is a gate rule too: `build_policy` turns it
        # into one whether or not the scorer still runs, so a budget on a
        # removed scorer means paying for every judge and then failing on
        # the metric that was never going to appear.
        for field_name, keys in (
            ("thresholds", config.thresholds),
            ("regression_budget", config.regression_budget),
        ):
            for key in keys:
                if key in (name, metric):
                    raise ConfigError(
                        f"scorers.remove drops {name!r} but {field_name} "
                        f"still gates {key!r}",
                        remediation=(
                            f"Remove the {field_name} entry or keep the "
                            "scorer selected."
                        ),
                    )


def _served_entity_identity(served: Any) -> str | None:
    """A stable name for whatever a serving endpoint currently serves.

    Reads the endpoint's config rather than its name: the entity name plus
    version is what actually changes when the platform team repoints or
    promotes a judge.
    """

    config = getattr(served, "config", None) or getattr(served, "pending_config", None)
    entities = getattr(config, "served_entities", None) or getattr(
        config, "served_models", None
    )
    parts = []
    for entity in entities or ():
        name = (
            getattr(entity, "entity_name", None)
            or getattr(entity, "model_name", None)
            or getattr(entity, "name", None)
        )
        if not name:
            continue
        version = getattr(entity, "entity_version", None) or getattr(
            entity, "model_version", None
        )
        parts.append(f"{name}/{version}" if version else str(name))
    return ",".join(sorted(parts)) or None
