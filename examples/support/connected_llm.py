"""Connected async-streaming mechanics for the advanced prompt A/B lesson."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from time import monotonic
from typing import Any, NamedTuple

import pandas as pd

from aai_core.experiments import ExperimentManager, ExperimentRunMetadata, RunPurpose
from aai_core.prompts import PromptManager
from aai_core.tracing import (
    TraceCaptureMode,
    TraceIntegration,
    TracePolicy,
    configure_tracing,
    set_trace_resource_context,
)
from examples import lifecycle_support

JsonRecord = dict[str, Any]

TRACE_DISPLAY_COLUMNS = (
    "role",
    "case_id",
    "answer_preview",
    "fact_coverage",
    "exact_citation",
    "recommendation_compliance",
    "latency_ms",
    "total_tokens",
    "cost_usd",
    "trace_id",
)


class PromptPair(NamedTuple):
    """The exact registered and reloaded prompt versions used by one comparison."""

    manager: PromptManager
    versions: Mapping[str, Any]
    loaded: Mapping[str, Any]


def prepare_prompt_pair(ctx: Any) -> PromptPair:
    """Register idempotently, reload exactly, and verify content identity."""

    prompts = PromptManager(
        context=ctx.tags,
        catalog=ctx.settings.catalog,
        schema=ctx.settings.schema,
    )
    versions = {
        "baseline": lifecycle_support.ensure_prompt_version(
            prompts,
            role="baseline",
            template=lifecycle_support.BASELINE_PROMPT,
        ),
        "change": lifecycle_support.ensure_prompt_version(
            prompts,
            role="change",
            template=lifecycle_support.CHANGE_PROMPT,
        ),
    }
    loaded = {
        role: prompts.load(
            lifecycle_support.PROMPT_NAME,
            version=int(prompt_version.version),
        )
        for role, prompt_version in versions.items()
    }
    expected_templates = {
        "baseline": lifecycle_support.BASELINE_PROMPT,
        "change": lifecycle_support.CHANGE_PROMPT,
    }
    for role, prompt in loaded.items():
        if lifecycle_support.prompt_digest(prompt.template) != (
            lifecycle_support.prompt_digest(expected_templates[role])
        ):
            raise RuntimeError(f"Loaded {role} prompt digest does not match.")
    return PromptPair(manager=prompts, versions=versions, loaded=loaded)


def prompt_pair_summary(pair: PromptPair) -> JsonRecord:
    return {
        "prompt_name": pair.manager.qualify(lifecycle_support.PROMPT_NAME),
        "baseline_uri": pair.versions["baseline"].uri,
        "baseline_digest": lifecycle_support.prompt_digest(
            pair.loaded["baseline"].template
        ),
        "change_uri": pair.versions["change"].uri,
        "change_digest": lifecycle_support.prompt_digest(
            pair.loaded["change"].template
        ),
        "dataset": lifecycle_support.DATASET_NAME,
        "dataset_digest": lifecycle_support.dataset_digest(),
    }


def response_text(raw_content: Any) -> str:
    """Normalize provider text blocks without exposing telemetry as the answer."""

    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts = []
        for block in raw_content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif getattr(block, "type", None) == "text":
                parts.append(str(getattr(block, "text", "")))
        return "\n".join(part for part in parts if part)
    return str(raw_content or "")


async def invoke_prompt(
    *,
    native_async_client: Any,
    rendered_prompt: str,
    mlflow_module: Any,
    model: Any,
    resource_context: Any,
) -> JsonRecord:
    """Consume one native stream with bounded capture and guaranteed closure."""

    started = monotonic()
    with mlflow_module.start_span(
        name="earnings_summary.prompt_evaluation",
        span_type="CHAIN",
    ) as application_span:
        set_trace_resource_context(resource_context)
        application_span.set_attribute("mlflow.message.format", "openai")
        application_span.set_inputs(
            {"messages": [{"role": "user", "content": rendered_prompt}]}
        )
        mlflow_module.update_current_trace(request_preview=rendered_prompt)
        stream = None
        content_parts: list[str] = []
        usage: dict[str, int] = {}
        response_model = model.model
        try:
            stream = await native_async_client.chat.completions.create(
                model=model.model,
                messages=[{"role": "user", "content": rendered_prompt}],
                temperature=0.0,
                max_tokens=220,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for event in stream:
                response_model = str(getattr(event, "model", response_model))
                usage_object = getattr(event, "usage", None)
                usage_dumper = getattr(usage_object, "model_dump", None)
                if callable(usage_dumper):
                    raw_usage = usage_dumper()
                elif usage_object is not None:
                    raw_usage = dict(usage_object)
                else:
                    raw_usage = {}
                if raw_usage:
                    usage = {
                        str(name): int(value)
                        for name, value in raw_usage.items()
                        if isinstance(value, (int, float))
                    }
                choices = getattr(event, "choices", None) or ()
                if choices:
                    delta = getattr(choices[0].delta, "content", None)
                    if delta:
                        content_parts.append(response_text(delta))
        finally:
            if stream is not None:
                await stream.close()
        content = "".join(content_parts)
        latency_ms = (monotonic() - started) * 1000
        application_span.set_outputs({"content": content})
        application_span.set_attribute("aai.response_model", response_model)
        application_span.set_attribute("aai.latency_ms", latency_ms)
        application_span.set_attribute("aai.usage", usage)
        mlflow_module.update_current_trace(response_preview=content)
    return {
        "content": content,
        "latency_ms": latency_ms,
        "usage": usage,
        "model": response_model,
    }


def configure_comparison_tracing(ctx: Any, experiment_name: str) -> None:
    """Select one instrumentation owner for synthetic full-content capture."""

    configure_tracing(
        ctx.tags,
        experiment_name=experiment_name,
        integration=TraceIntegration.MLFLOW_OPENAI,
        policy=TracePolicy(capture_mode=TraceCaptureMode.FULL),
    )


def register_evaluation_dataset(
    *,
    environment: Any,
    connected: Any,
    setup_helpers: Any,
    mlflow_module: Any,
) -> Any | None:
    if environment.evidence_destination.value != "databricks":
        return None
    return setup_helpers.get_or_create_uc_evaluation_dataset(
        evidence=connected,
        dataset_name="fictional_earnings_summary_regression_v1",
        records=[case.evaluation_record() for case in lifecycle_support.CASES],
        mlflow_module=mlflow_module,
    )


def _translate_provider_error(error: Exception, model_name: str) -> RuntimeError | None:
    status = getattr(error, "status_code", None)
    if status == 403 or "PERMISSION_DENIED" in str(error):
        return RuntimeError(
            f"Your identity cannot query endpoint {model_name!r}; request CAN_QUERY "
            "from the platform team."
        )
    if status == 404 or "NotFound" in type(error).__name__:
        return RuntimeError(f"Endpoint {model_name!r} was not found; rerun preflight.")
    return None


async def run_prompt_comparison(
    *,
    ctx: Any,
    model: Any,
    experiment_name: str,
    pair: PromptPair,
    registered_dataset: Any | None,
    mlflow_module: Any,
) -> tuple[list[JsonRecord], dict[str, str], Any]:
    """Run the same ordered cases against the exact baseline and change prompts."""

    from mlflow import MlflowClient

    experiments = ExperimentManager(experiment_name=experiment_name, context=ctx.tags)
    client = MlflowClient()
    call_records: list[JsonRecord] = []
    run_ids: dict[str, str] = {}
    async with model.create_native_async_client() as native_async_client:
        for role, run_name in (
            ("baseline", lifecycle_support.BASELINE_NAME),
            ("change", lifecycle_support.CHANGE_NAME),
        ):
            prompt_version = pair.versions[role]
            loaded_prompt = pair.loaded[role]
            with experiments.run(
                run_name=f"connected-{run_name}",
                description=(
                    "Observed three-case baseline evidence for the immutable "
                    "earnings-summary prompt."
                    if role == "baseline"
                    else "Observed three-case change evidence for the cited "
                    "earnings-summary prompt against the governed baseline."
                ),
                parameters={
                    "hypothesis": lifecycle_support.HYPOTHESIS,
                    "prompt_uri": prompt_version.uri,
                    "prompt_digest_sha256": lifecycle_support.prompt_digest(
                        loaded_prompt.template
                    ),
                    "evaluation_dataset": lifecycle_support.DATASET_NAME,
                    "dataset_digest_sha256": lifecycle_support.dataset_digest(),
                    "case_count": len(lifecycle_support.CASES),
                    "evaluation_repetitions": 1,
                    "trace_mode": "mlflow_openai_autolog_native_async_stream",
                },
                tags={"experiment_role": role},
                metadata=ExperimentRunMetadata(
                    purpose=(
                        RunPurpose.BASELINE if role == "baseline" else RunPurpose.CHANGE
                    ),
                    change_id=lifecycle_support.CHANGE_ID,
                    change_summary=lifecycle_support.CHANGE_SUMMARY,
                    hypothesis=lifecycle_support.HYPOTHESIS,
                    baseline_run_id=run_ids.get("baseline"),
                ),
            ) as active_run:
                run_ids[role] = active_run.info.run_id
                if registered_dataset is not None:
                    mlflow_module.log_input(registered_dataset, context="evaluation")
                client.link_prompt_version_to_run(
                    active_run.info.run_id,
                    prompt_version,
                )
                for case in lifecycle_support.CASES:
                    rendered = loaded_prompt.format(
                        question=case.question,
                        earnings_excerpt=case.earnings_excerpt,
                        source_id=case.source_id,
                    )
                    try:
                        response = await invoke_prompt(
                            native_async_client=native_async_client,
                            rendered_prompt=rendered,
                            mlflow_module=mlflow_module,
                            model=model,
                            resource_context=ctx.tags,
                        )
                    except Exception as error:
                        translated = _translate_provider_error(error, model.model)
                        if translated is not None:
                            raise translated from error
                        raise
                    mlflow_module.flush_trace_async_logging()
                    trace_id = mlflow_module.get_last_active_trace_id()
                    if trace_id is None:
                        raise RuntimeError(
                            f"No trace was created for {role}/{case.case_id}."
                        )
                    client.link_prompt_versions_to_trace(
                        prompt_versions=[prompt_version],
                        trace_id=trace_id,
                    )
                    call_records.append(
                        {
                            "role": role,
                            "case": case,
                            "prompt_uri": prompt_version.uri,
                            "prompt_name": pair.manager.qualify(
                                lifecycle_support.PROMPT_NAME
                            ),
                            "prompt_version": str(prompt_version.version),
                            "run_id": active_run.info.run_id,
                            "trace_id": trace_id,
                            "response": response,
                        }
                    )
    return call_records, run_ids, client


def _trace_for_record(
    *,
    record: Mapping[str, Any],
    remote_mode: bool,
    remote_trace_infos: Mapping[str, Any],
    mlflow_module: Any,
) -> tuple[Any, list[str]]:
    if remote_mode:
        trace = remote_trace_infos.get(record["trace_id"])
        if trace is None:
            raise RuntimeError(
                f"Databricks did not return trace {record['trace_id']} for run "
                f"{record['run_id']}."
            )
        size_stats = json.loads(
            dict(trace.info.trace_metadata or {}).get("mlflow.trace.sizeStats", "{}")
        )
        if size_stats.get("num_spans") != 2:
            raise RuntimeError(
                f"Unexpected remote trace shape for {record['role']}/"
                f"{record['case'].case_id}: {size_stats!r}."
            )
        return trace, ["Databricks provider span (inspect in UI)"]
    trace = mlflow_module.get_trace(record["trace_id"])
    span_names = [span.name for span in trace.data.spans]
    application_span = "earnings_summary.prompt_evaluation"
    provider_spans = [name for name in span_names if name != application_span]
    if span_names.count(application_span) != 1 or len(provider_spans) != 1:
        raise RuntimeError(
            f"Unexpected trace shape for {record['role']}/"
            f"{record['case'].case_id}: {span_names!r}."
        )
    return trace, provider_spans


def _verify_trace_lineage(
    *,
    record: Mapping[str, Any],
    trace: Any,
    remote_mode: bool,
    client: Any,
    application: str,
) -> None:
    trace_metadata = dict(trace.info.trace_metadata or {})
    if trace_metadata.get("aai.application") != application:
        raise RuntimeError(f"Trace {record['trace_id']} lacks SDK metadata.")
    if trace_metadata.get("mlflow.sourceRun") != record["run_id"]:
        raise RuntimeError(
            f"Trace {record['trace_id']} is not associated with its comparison run."
        )
    lineage_tags = (
        dict(client.get_run(record["run_id"]).data.tags)
        if remote_mode
        else dict(trace.info.tags or {})
    )
    raw_links = lineage_tags.get("mlflow.linkedPrompts", "[]")
    linked_prompts = (
        json.loads(raw_links) if isinstance(raw_links, str) else list(raw_links)
    )
    expected = {
        "name": record["prompt_name"],
        "version": record["prompt_version"],
    }
    if linked_prompts != [expected]:
        raise RuntimeError(
            f"Trace {record['trace_id']} prompt links are {linked_prompts!r}; "
            f"expected {[expected]!r}."
        )


def build_trace_evaluation(
    *,
    call_records: Sequence[Mapping[str, Any]],
    run_ids: Mapping[str, str],
    environment: Any,
    experiment_name: str,
    ctx: Any,
    client: Any,
    mlflow_module: Any,
) -> pd.DataFrame:
    """Validate trace shape/lineage and return the unchanged evidence schema."""

    from examples.lifecycle_support import (
        citation_score,
        fact_coverage,
        quality_score,
        recommendation_policy_score,
    )

    remote_mode = environment.evidence_destination.value == "databricks"
    remote_trace_infos: dict[str, Any] = {}
    if remote_mode:
        experiment = client.get_experiment_by_name(experiment_name)
        for run_id in run_ids.values():
            traces = client.search_traces(
                locations=[experiment.experiment_id],
                run_id=run_id,
                include_spans=False,
                max_results=len(lifecycle_support.CASES),
                flush=True,
            )
            remote_trace_infos.update({trace.info.trace_id: trace for trace in traces})
    rows = []
    for record in call_records:
        trace, provider_spans = _trace_for_record(
            record=record,
            remote_mode=remote_mode,
            remote_trace_infos=remote_trace_infos,
            mlflow_module=mlflow_module,
        )
        _verify_trace_lineage(
            record=record,
            trace=trace,
            remote_mode=remote_mode,
            client=client,
            application=ctx.tags.application,
        )
        response = record["response"]
        answer = response["content"]
        if not answer:
            raise RuntimeError(
                f"Empty response for {record['role']}/{record['case'].case_id}."
            )
        expectations = record["case"].evaluation_record()["expectations"]
        trace_usage = dict(trace.info.token_usage or response["usage"])
        input_tokens = int(
            trace_usage.get("input_tokens", trace_usage.get("prompt_tokens", 0)) or 0
        )
        output_tokens = int(
            trace_usage.get("output_tokens", trace_usage.get("completion_tokens", 0))
            or 0
        )
        total_tokens = trace_usage.get("total_tokens", input_tokens + output_tokens)
        reported_cost = dict(trace.info.cost or {}).get("total_cost")
        output = {"answer": answer}
        rows.append(
            {
                "role": record["role"],
                "case_id": record["case"].case_id,
                "answer_preview": answer.replace("\n", " ")[:180],
                "fact_coverage": fact_coverage(output, expectations),
                "exact_citation": citation_score(output, expectations),
                "recommendation_compliance": recommendation_policy_score(
                    output, expectations
                ),
                "quality_score": quality_score(output, expectations),
                "latency_ms": response["latency_ms"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                # Missing cost is unknown, never zero.
                "cost_usd": reported_cost,
                "cost_available": float(reported_cost is not None),
                "trace_id": record["trace_id"],
                "prompt_uri": record["prompt_uri"],
                "provider_span": provider_spans[0],
                "answer": answer,
            }
        )
    return pd.DataFrame(rows)


def known_cost_total(series: pd.Series) -> float:
    """Preserve unknown aggregate cost instead of inventing zero."""

    return float(series.sum(min_count=1))


def record_exploratory_comparison(
    frame: pd.DataFrame,
    *,
    run_ids: Mapping[str, str],
    client: Any,
) -> tuple[pd.DataFrame, JsonRecord]:
    """Aggregate the same cases and persist an explicitly non-release decision."""

    comparison = frame.groupby("role", sort=False).agg(
        cases=("case_id", "count"),
        fact_coverage=("fact_coverage", "mean"),
        exact_citation=("exact_citation", "mean"),
        recommendation_compliance=("recommendation_compliance", "mean"),
        quality_score=("quality_score", "mean"),
        latency_ms_mean=("latency_ms", "mean"),
        total_tokens=("total_tokens", "sum"),
        cost_usd_total=("cost_usd", known_cost_total),
        cost_coverage=("cost_available", "mean"),
    )
    for role, metrics in comparison.iterrows():
        for name, value in metrics.items():
            if pd.notna(value):
                client.log_metric(run_ids[role], str(name), float(value))
    baseline = comparison.loc["baseline"]
    change = comparison.loc["change"]
    observed_preference = (
        "change"
        if (
            change["quality_score"] >= baseline["quality_score"]
            and change["exact_citation"] > baseline["exact_citation"]
            and change["recommendation_compliance"] == 1.0
        )
        else "baseline"
    )
    client.set_tag(run_ids["change"], "aai.result", "exploratory_comparison_recorded")
    client.set_tag(run_ids["change"], "aai.observed_preference", observed_preference)
    client.set_tag(run_ids["change"], "aai.decision", "inconclusive")
    client.set_tag(run_ids["change"], "aai.release", "blocked_until_evaluated")
    return comparison, {
        "observed_preference": observed_preference,
        "decision": "inconclusive",
        "next_action": "run_full_evaluation",
        "release": "blocked_until_evaluated",
    }


def publish_prompt_pair_to_databricks(
    *,
    ctx: Any,
    mlflow_module: Any,
) -> JsonRecord:
    """Copy prompt versions only, then restore the original registry routing."""

    original_registry_uri = mlflow_module.get_registry_uri()
    try:
        mlflow_module.set_registry_uri("databricks-uc")
        pair = prepare_prompt_pair(ctx)
        return {
            "baseline_uri": pair.versions["baseline"].uri,
            "baseline_digest": lifecycle_support.prompt_digest(
                pair.loaded["baseline"].template
            ),
            "change_uri": pair.versions["change"].uri,
            "change_digest": lifecycle_support.prompt_digest(
                pair.loaded["change"].template
            ),
            "local_traces_linked": False,
        }
    finally:
        mlflow_module.set_registry_uri(original_registry_uri)


__all__ = [
    "PromptPair",
    "TRACE_DISPLAY_COLUMNS",
    "build_trace_evaluation",
    "configure_comparison_tracing",
    "invoke_prompt",
    "known_cost_total",
    "prepare_prompt_pair",
    "prompt_pair_summary",
    "publish_prompt_pair_to_databricks",
    "record_exploratory_comparison",
    "register_evaluation_dataset",
    "response_text",
    "run_prompt_comparison",
]
