"""Credential-free tests for the Email Support Agent MLflow recipes."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aai_core.decisions import Decision
from aai_core.evaluation import (
    GatePolicy,
    MetricDirection,
    MetricRule,
    apply_gate,
)
from aai_core.experiments import RunPurpose

ROOT = Path(__file__).resolve().parents[1]
ACCELERATOR = ROOT / "examples" / "email-support-agent"
SOURCE = ACCELERATOR / "src"
_ADDED_SOURCE = str(SOURCE) not in sys.path
if _ADDED_SOURCE:
    sys.path.insert(0, str(SOURCE))

from email_support_agent.contracts import MeasurementSource, RiskTier  # noqa: E402
from email_support_agent.mlops import (  # noqa: E402
    ComparisonEvidence,
    EvaluationLineage,
    ExecutionMode,
    ExecutionPolicy,
    FullReleaseLineage,
    GepaOptimizationPlan,
    JudgeAlignmentPlan,
    ModelLineage,
    MonitoringPlan,
    MutationRefusedError,
    PromotionEvidence,
    PromotionEvidenceError,
    PromptLineage,
    RegisteredScorerRef,
    RetrievalLineage,
    RetrieverTraceEvidence,
    ReviewedTrace,
    RunEvidence,
    ToolLineage,
    TraceCurationPlan,
    TraceDestinationPlan,
    UsageCostEvidence,
    apply_trace_destination,
    assess_promotion,
    create_manual_decision,
    curate_reviewed_trace,
    estimate_agentkit_judge_budget,
    estimate_gepa_budget,
    merge_curated_records,
    monitoring_plan_from_agentkit,
    register_and_start_monitoring,
    require_promotion_ready,
    run_gepa_proposal,
    run_memalign_proposal,
    select_for_risk_batch,
)

if _ADDED_SOURCE:
    sys.path.remove(str(SOURCE))


class _CloudBomb:
    def __getattribute__(self, name: str):
        raise AssertionError(f"dry run touched a cloud dependency: {name}")


def _raise_if_called(*_args, **_kwargs):
    raise AssertionError("dry run called the tracing configurer")


def _shared_judge() -> RegisteredScorerRef:
    return RegisteredScorerRef(
        shared_name="guidelines",
        version_env="AAI_ALIGNED_GUIDELINES_JUDGE_VERSION",
    )


def _alignment_plan() -> JudgeAlignmentPlan:
    return JudgeAlignmentPlan(
        judge=_shared_judge(),
        label_schema_name="guidelines",
    )


def _gepa_plan() -> GepaOptimizationPlan:
    return GepaOptimizationPlan(scorers=(_shared_judge(),))


def test_all_connected_operations_are_cloud_free_in_default_dry_run():
    bomb = _CloudBomb()

    trace_receipt = apply_trace_destination(
        TraceDestinationPlan(),
        tracing_configurer=_raise_if_called,
    )
    monitoring_receipt = register_and_start_monitoring(
        MonitoringPlan(),
        mlflow_module=bomb,
    )
    curation_receipt = merge_curated_records(
        (),
        plan=TraceCurationPlan(),
        mlflow_module=bomb,
    )
    alignment_receipt, aligned = run_memalign_proposal(
        _alignment_plan(), mlflow_module=bomb
    )
    optimization_receipt, proposal = run_gepa_proposal(_gepa_plan(), mlflow_module=bomb)

    assert not trace_receipt.applied
    assert not monitoring_receipt.applied
    assert not curation_receipt.applied
    assert not alignment_receipt.applied
    assert not optimization_receipt.applied
    assert aligned is None
    assert proposal is None


def test_execute_requires_explicit_acknowledgement_before_backend_access():
    execution = ExecutionPolicy(
        mode=ExecutionMode.EXECUTE,
        notebook_confirmed=True,
    )
    with pytest.raises(MutationRefusedError, match="acknowledgement"):
        register_and_start_monitoring(
            MonitoringPlan(),
            execution=execution,
            environ={},
            mlflow_module=_CloudBomb(),
        )


def test_trace_configuration_contains_only_environment_references():
    plan = TraceDestinationPlan()
    assert plan.tracking_uri_env == "AAI_MLFLOW_TRACKING_URI"
    assert plan.experiment_name_env == "AAI_MLFLOW_EXPERIMENT_NAME"
    assert plan.trace_destination_env == "MLFLOW_TRACING_DESTINATION"
    assert plan.policy.capture_mode.value == "bounded"

    with pytest.raises(ValidationError, match="environment variable"):
        TraceDestinationPlan(tracking_uri_env="https://workspace.example")
    with pytest.raises(ValidationError, match="Extra inputs"):
        TraceDestinationPlan.model_validate({"workspace_id": "embedded-id"})


def test_monitoring_registers_and_starts_shared_names_at_native_floor():
    events: list[tuple[str, object]] = []

    class SamplingConfig:
        def __init__(self, *, sample_rate):
            self.sample_rate = sample_rate

    def scorer_class(kind: str):
        class NativeScorer:
            def __init__(self, **kwargs):
                events.append((f"construct:{kind}", kwargs))

            def register(self, *, name):
                events.append(("register", name))
                return self

            def start(self, *, sampling_config):
                events.append(("start", sampling_config.sample_rate))
                return self

        return NativeScorer

    fake = SimpleNamespace(
        set_experiment=lambda name: events.append(("experiment", name)),
        genai=SimpleNamespace(
            scorers=SimpleNamespace(
                Safety=scorer_class("safety"),
                Guidelines=scorer_class("guidelines"),
                ScorerSamplingConfig=SamplingConfig,
            )
        ),
    )
    plan = monitoring_plan_from_agentkit(ACCELERATOR)
    receipt = register_and_start_monitoring(
        plan,
        execution=ExecutionPolicy(
            mode=ExecutionMode.EXECUTE,
            notebook_confirmed=True,
        ),
        environ={
            "AAI_ENABLE_MLFLOW_MUTATIONS": ("I_UNDERSTAND_THIS_MUTATES_MLFLOW"),
            "AAI_MLFLOW_EXPERIMENT_NAME": "resolved-at-runtime",
            "AAI_JUDGE_MODEL_URI": "logical-model-resolution",
        },
        mlflow_module=fake,
    )

    assert receipt.applied
    assert ("register", "safety") in events
    assert ("register", "guidelines") in events
    assert events.count(("start", plan.risk_sampling.native_sample_rate)) == 2
    assert plan.scorer_versions == ("safety=1", "guidelines=1")


def test_risk_sampling_is_monotonic_and_critical_is_always_selected():
    policy = MonitoringPlan().risk_sampling
    assert policy.low <= policy.medium <= policy.high <= policy.critical
    assert select_for_risk_batch(
        trace_id="trace-critical-001",
        risk=RiskTier.CRITICAL,
        policy=policy,
    )
    first = select_for_risk_batch(
        trace_id="trace-low-001",
        risk=RiskTier.LOW,
        policy=policy,
    )
    second = select_for_risk_batch(
        trace_id="trace-low-001",
        risk=RiskTier.LOW,
        policy=policy,
    )
    assert first is second


def test_trace_curation_preserves_nested_inputs_expectations_and_hashes_ref():
    trace = ReviewedTrace(
        trace_id="trace-critical-001",
        risk=RiskTier.CRITICAL,
        inputs={
            "email": {
                "case_id": "case-001",
                "subject": "Cannot access the audit export",
            }
        },
        outputs={"response": "A specialist will review this case."},
        trace={
            "spans": [
                {
                    "name": "search_knowledge",
                    "span_type": "RETRIEVER",
                    "outputs": [
                        {
                            "page_content": "Audit export guidance.",
                            "metadata": {
                                "doc_uri": "synthetic://kb/audit-export",
                                "chunk_id": "chunk-001",
                            },
                        }
                    ],
                }
            ]
        },
        expectations={
            "expected_response": "A specialist will review this case.",
            "expected_intent": "question",
            "expected_urgency": "critical",
            "expected_route": "human_review",
            "requires_review": True,
        },
        feedback_name="support_review_outcome",
        feedback_value="edited",
        feedback_source="group:support-quality",
        rationale="SME corrected the commitment and kept human review.",
        dlp_evidence_ref="synthetic://dlp/trace-critical-001",
    )
    record = curate_reviewed_trace(trace, plan=TraceCurationPlan())

    assert record is not None
    native = record.as_mlflow_record()
    assert set(native) == {"inputs", "expectations", "outputs", "trace", "tags"}
    assert native["inputs"]["email"]["case_id"] == "case-001"
    assert native["expectations"]["expected_response"]
    assert native["trace"]["spans"][0]["span_type"] == "RETRIEVER"
    assert native["tags"]["source_trace_sha256"] != trace.trace_id
    assert native["tags"]["dlp_evidence_sha256"] != trace.dlp_evidence_ref
    assert trace.trace_id not in str(native)


def test_trace_curation_rejects_raw_email_and_sensitive_fragments():
    with pytest.raises(ValidationError, match="privacy boundary"):
        ReviewedTrace(
            trace_id="trace-raw-email-001",
            risk=RiskTier.CRITICAL,
            inputs={
                "raw_email": {
                    "from": "person@example.test",
                    "body": "unredacted message",
                }
            },
            outputs="Unsafe raw replay.",
            trace={"spans": [{"name": "root", "span_type": "CHAIN"}]},
            expectations={"expected_response": "A redacted response."},
            feedback_name="support_review_outcome",
            feedback_value="rejected",
            feedback_source="group:support-quality",
            rationale="Raw email material must never enter the dataset.",
            dlp_evidence_ref="synthetic://dlp/trace-raw-email-001",
        )


def test_memalign_label_schema_must_equal_shared_judge_name():
    with pytest.raises(ValidationError, match="exactly match"):
        JudgeAlignmentPlan(
            judge=_shared_judge(),
            label_schema_name="support_quality",
        )


def test_gepa_is_proposal_only_and_has_bounded_token_estimate():
    plan = _gepa_plan()
    budget = estimate_gepa_budget(plan)
    assert plan.proposal_only
    assert not plan.auto_promote
    assert budget.estimated_total_tokens == (
        plan.max_metric_calls * plan.estimated_tokens_per_metric_call
        + plan.max_reflection_calls * plan.estimated_tokens_per_reflection_call
    )
    with pytest.raises(ValidationError):
        GepaOptimizationPlan(
            scorers=(_shared_judge(),),
            auto_promote=True,
        )


def _usage() -> UsageCostEvidence:
    return UsageCostEvidence(
        model_calls=2,
        priced_model_calls=2,
        input_tokens=800,
        output_tokens=200,
        cost_usd=0.02,
        cost_coverage=1.0,
        measurement_source=MeasurementSource.CONNECTED,
        pricing_digest="9" * 64,
    )


def _lineage(*, release: str, prompt_digest: str, with_cost: bool = True):
    return FullReleaseLineage(
        application="email-support-agent",
        release=release,
        source_commit="a" * 40,
        core_sdk_version="1.0.0",
        environment="release-evaluation",
        model=ModelLineage(
            logical_name="response-model",
            provider="approved-provider",
            model_release="model-release-001",
            endpoint_binding_env="AAI_RESPONSE_MODEL_ENDPOINT",
            inference_config_digest="1" * 64,
        ),
        prompt=PromptLineage(
            qualified_name="catalog.schema.support_reply",
            version=2 if release.endswith("2") else 1,
            content_digest=prompt_digest,
        ),
        retrieval=RetrievalLineage(
            logical_index="support-knowledge",
            index_release="index-release-003",
            endpoint_binding_env="AAI_SUPPORT_INDEX_ENDPOINT",
            embedding_model_release="embedding-release-002",
            embedding_config_digest="2" * 64,
            chunking_release="chunking-release-004",
            chunking_config_digest="3" * 64,
        ),
        tools=(
            ToolLineage(
                name="ticket-upsert",
                version="2",
                input_schema_digest="4" * 64,
                implementation_digest="5" * 64,
                side_effecting=True,
            ),
            ToolLineage(
                name="reply-outbox",
                version="3",
                input_schema_digest="6" * 64,
                implementation_digest="7" * 64,
                side_effecting=True,
            ),
        ),
        usage_cost=_usage() if with_cost else None,
        evaluation=EvaluationLineage(
            dataset_digest="b" * 16,
            agentkit_config_digest="c" * 64,
            gate_policy_digest="d" * 64,
            scorer_versions=(
                "correctness=1",
                "safety=1",
                "guidelines=1",
                "retrieval_groundedness=1",
                "retrieval_relevance=1",
                "retrieval_sufficiency=1",
            ),
        ),
    )


def _promotion_evidence(
    *,
    with_retriever: bool = True,
    with_cost: bool = True,
    under_scoped_gate: bool = False,
) -> PromotionEvidence:
    baseline_lineage = _lineage(
        release="release-1",
        prompt_digest="e" * 64,
    )
    change_lineage = _lineage(
        release="release-2",
        prompt_digest="f" * 64,
        with_cost=with_cost,
    )
    metrics = {
        "correctness/mean": 0.95,
        "safety/mean": 1.0,
        "guidelines/mean": 1.0,
        "retrieval_groundedness/mean": 0.90,
        "retrieval_relevance/mean": 0.80,
        "retrieval_sufficiency/mean": 0.85,
        "safety/false_auto_send_rate": 0.0,
        "cost/coverage": 1.0,
    }
    rules = (
        (
            MetricRule(
                metric="correctness/mean",
                direction=MetricDirection.HIGHER,
                required=0.90,
            ),
        )
        if under_scoped_gate
        else (
            MetricRule(
                metric="correctness/mean",
                direction=MetricDirection.HIGHER,
                required=0.90,
            ),
            MetricRule(
                metric="safety/mean",
                direction=MetricDirection.HIGHER,
                required=1.0,
            ),
            MetricRule(
                metric="guidelines/mean",
                direction=MetricDirection.HIGHER,
                required=1.0,
            ),
            MetricRule(
                metric="retrieval_groundedness/mean",
                direction=MetricDirection.HIGHER,
                required=0.85,
            ),
            MetricRule(
                metric="retrieval_relevance/mean",
                direction=MetricDirection.HIGHER,
                required=0.75,
            ),
            MetricRule(
                metric="retrieval_sufficiency/mean",
                direction=MetricDirection.HIGHER,
                required=0.80,
            ),
            MetricRule(
                metric="safety/false_auto_send_rate",
                direction=MetricDirection.LOWER,
                required=0.0,
            ),
        )
    )
    gate = apply_gate(
        metrics,
        policy=GatePolicy(
            rules=rules,
            minimum_cost_coverage=1.0,
        ),
    )
    comparison = ComparisonEvidence(
        change_id="support-reply-v2",
        change_summary="Evaluate the proposed support reply prompt.",
        baseline=RunEvidence(
            run_id="baseline-run-001",
            purpose=RunPurpose.BASELINE,
            release_digest=baseline_lineage.digest,
            dataset_digest="b" * 16,
        ),
        change=RunEvidence(
            run_id="change-run-001",
            purpose=RunPurpose.CHANGE,
            release_digest=change_lineage.digest,
            dataset_digest="b" * 16,
        ),
        result=RunEvidence(
            run_id="result-run-001",
            purpose=RunPurpose.RESULT,
            release_digest=change_lineage.digest,
            dataset_digest="b" * 16,
        ),
        baseline_lineage=baseline_lineage,
        change_lineage=change_lineage,
        gate=gate,
    )
    retriever = (
        RetrieverTraceEvidence(
            eligible_trace_count=6,
            traces_with_retriever_span=6,
            retriever_span_count=6,
            retrieved_document_count=18,
            documents_with_required_fields=18,
        )
        if with_retriever
        else None
    )
    return PromotionEvidence(comparison=comparison, retriever=retriever)


def test_complete_lineage_projects_to_application_release():
    lineage = _lineage(release="release-2", prompt_digest="f" * 64)
    release = lineage.as_application_release()

    assert release.model["model_release"] == "model-release-001"
    assert release.prompt["version"] == 2
    assert release.retrieval["embedding_model_release"] == "embedding-release-002"
    assert len(release.evaluation["tools"]) == 2
    assert release.evaluation["usage_cost"]["cost_coverage"] == 1.0


def test_promotion_readiness_never_auto_adopts():
    evidence = _promotion_evidence()
    assessment = assess_promotion(evidence)

    assert assessment.ready_for_human_decision
    assert assessment.current_decision is Decision.INCONCLUSIVE
    assert assessment.automatic_decision == "none"

    explicit = create_manual_decision(
        evidence,
        decision=Decision.ADOPT,
        rationale="The release board reviewed the passing evidence.",
        decided_by="support-release-board",
    )
    assert explicit.decision is Decision.ADOPT


def test_under_scoped_passing_gate_cannot_authorize_adopt():
    evidence = _promotion_evidence(under_scoped_gate=True)
    assert evidence.comparison.gate.passed

    assessment = assess_promotion(evidence)

    assert not assessment.ready_for_human_decision
    assert any(
        blocker.startswith("promotion gate does not enforce safety/mean")
        for blocker in assessment.blockers
    )
    with pytest.raises(PromotionEvidenceError, match="does not enforce safety/mean"):
        create_manual_decision(
            evidence,
            decision=Decision.ADOPT,
            rationale="A passing but incomplete gate is not release evidence.",
            decided_by="support-release-board",
        )


@pytest.mark.parametrize(
    ("with_retriever", "with_cost", "message"),
    (
        (False, True, "RETRIEVER span evidence is missing"),
        (True, False, "usage and cost evidence is missing"),
    ),
)
def test_missing_retriever_or_cost_evidence_fails_promotion(
    with_retriever: bool,
    with_cost: bool,
    message: str,
):
    evidence = _promotion_evidence(
        with_retriever=with_retriever,
        with_cost=with_cost,
    )
    assessment = assess_promotion(evidence)

    assert not assessment.ready_for_human_decision
    assert message in assessment.blockers
    with pytest.raises(PromotionEvidenceError, match=message):
        require_promotion_ready(evidence)
    with pytest.raises(PromotionEvidenceError, match=message):
        create_manual_decision(
            evidence,
            decision=Decision.ADOPT,
            rationale="This must not bypass missing evidence.",
            decided_by="support-release-board",
        )


def test_agentkit_budget_uses_shared_scorers_and_retrieval_fanout():
    budget = estimate_agentkit_judge_budget(ACCELERATOR)

    assert budget.within_budget
    assert budget.maximum_judge_calls == 100
    assert budget.cost.rows == 11
    assert budget.cost.judge_calls == 99
    assert budget.cost.calls_by_scorer["retrieval_relevance"] == 44
    assert budget.cost.estimated_tokens > 0
    assert "retrieval_groundedness=1" in budget.scorer_versions
