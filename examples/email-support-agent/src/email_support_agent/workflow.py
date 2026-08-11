"""Production-shaped, infrastructure-neutral email preparation and commit flow."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from hashlib import sha256
from typing import Any

from pydantic import ValidationError

from aai_core.tracing import (
    GovernedSpan,
    TraceCaptureMode,
    current_trace_state,
    native_active_span,
    native_mlflow,
)
from email_support_agent.contracts import (
    AccessContext,
    ActionKind,
    ActionReceipt,
    Classification,
    Disposition,
    DraftResponse,
    EvidenceDocument,
    ExecutionResult,
    MeasurementSource,
    ModelUsage,
    PlannedAction,
    PolicyConfig,
    PreparedCase,
    RedactedEmail,
    ReviewAction,
    ReviewDecision,
    RuntimeBudget,
    VerifiedReviewerContext,
)
from email_support_agent.policy import (
    evaluate_draft,
    plan_actions,
    policy_draft,
    preflight_classification,
    requires_human_review,
    route_email,
    verify_evidence_scope,
)
from email_support_agent.ports import (
    AccessAuthorizer,
    Classifier,
    Drafter,
    KnowledgeRetriever,
    ReviewAuthorizer,
    TransactionalOutbox,
)


class BudgetExceededError(RuntimeError):
    """Raised before a request can exceed its configured inference budget."""


class EmailSupportWorkflow:
    """Prepare a response without writes, then commit through an idempotent outbox."""

    def __init__(
        self,
        *,
        classifier: Classifier,
        access_authorizer: AccessAuthorizer,
        review_authorizer: ReviewAuthorizer,
        retriever: KnowledgeRetriever,
        drafter: Drafter,
        outbox: TransactionalOutbox,
        policy: PolicyConfig | None = None,
        budget: RuntimeBudget | None = None,
    ) -> None:
        self.classifier = classifier
        self.access_authorizer = access_authorizer
        self.review_authorizer = review_authorizer
        self.retriever = retriever
        self.drafter = drafter
        self.outbox = outbox
        self.policy = policy or PolicyConfig()
        self.budget = budget or RuntimeBudget()

    async def prepare(self, email: RedactedEmail | Mapping[str, Any]) -> PreparedCase:
        """Create a checkpoint-safe proposal; this method performs no writes."""

        request = RedactedEmail.model_validate(email, strict=True)
        access = _admit_provider_contract(
            AccessContext,
            await self.access_authorizer.authorize(request),
            boundary="access authorization",
        )
        _validate_access_binding(request, access)
        ledger = _InferenceBudgetLedger(self.budget)
        trace_input = _request_trace_metadata(request)
        with _semantic_span("input.guardrail", "GUARDRAIL", inputs=trace_input) as span:
            classification = preflight_classification(request)
            _set_outputs(
                span,
                {
                    "short_circuited": classification is not None,
                    "reason": (
                        "known_security_or_injection_signal"
                        if classification is not None
                        else "admitted_for_classification"
                    ),
                },
            )
        if classification is None:
            output_limit = ledger.reserve_model_call(
                "classification",
                estimated_input_tokens=_estimate_tokens(
                    request.subject,
                    request.body,
                ),
            )
            with _semantic_span(
                "intent.classify",
                "CHAIN",
                inputs=trace_input,
            ) as span:
                classification = _admit_provider_contract(
                    Classification,
                    await self.classifier.classify(
                        request,
                        max_output_tokens=output_limit,
                    ),
                    boundary="classification",
                )
                ledger.observe(classification.usage, boundary="classification")
                _set_outputs(span, _classification_trace_output(classification))

        with _semantic_span(
            "route.select",
            "ROUTER",
            inputs={
                "intent": classification.intent.value,
                "urgency": classification.urgency.value,
                "risk": classification.risk.value,
            },
        ) as span:
            route, route_reasons = route_email(request, classification)
            _set_outputs(
                span,
                {"route": route.value, "reasons": list(route_reasons)},
            )

        evidence = ()
        if route.value in {"knowledge_reply", "bug_tracking"}:
            with _semantic_span(
                "knowledge.retrieve",
                "RETRIEVER",
                inputs={
                    "query_sha256": _digest_text(f"{request.subject}\n{request.body}"),
                    "access_context_digest": access_context_digest(access),
                    "knowledge_release": self.policy.knowledge_release,
                    "top_k": self.budget.max_retrieved_documents,
                },
            ) as span:
                returned = await self.retriever.retrieve(
                    f"{request.subject}\n{request.body}",
                    access=access,
                    release=self.policy.knowledge_release,
                    top_k=self.budget.max_retrieved_documents,
                )
                if len(returned) > self.budget.max_retrieved_documents:
                    raise BudgetExceededError(
                        "retriever returned more documents than the configured bound"
                    )
                evidence = tuple(
                    _admit_provider_contract(
                        EvidenceDocument,
                        document,
                        boundary="retrieval document",
                    )
                    for document in returned
                )
                evidence = verify_evidence_scope(
                    evidence,
                    access=access,
                    release=self.policy.knowledge_release,
                )
                _set_outputs(
                    span,
                    [document.as_mlflow_document() for document in evidence],
                )

        draft = policy_draft(classification, route)
        if draft is None:
            output_limit = ledger.reserve_model_call(
                "drafting",
                estimated_input_tokens=_estimate_tokens(
                    request.subject,
                    request.body,
                    *(document.page_content for document in evidence),
                ),
            )
            with _semantic_span(
                "response.draft",
                "CHAIN",
                inputs={
                    "case_sha256": _digest_text(request.case_id),
                    "intent": classification.intent.value,
                    "document_ids": [item.document_id for item in evidence],
                    "max_output_tokens": output_limit,
                },
            ) as span:
                try:
                    candidate = await self.drafter.draft(
                        request,
                        classification,
                        evidence,
                        max_output_tokens=output_limit,
                    )
                except ValidationError:
                    candidate = None
                draft = _admit_draft(candidate)
                ledger.observe(draft.usage, boundary="drafting")
                _set_outputs(span, _draft_trace_output(draft))
        else:
            with _semantic_span(
                "response.policy_template",
                "CHAIN",
                inputs={"route": route.value, "intent": classification.intent.value},
            ) as span:
                _set_outputs(span, _draft_trace_output(draft))

        usage = _combined_usage(classification.usage, draft.usage)
        self._enforce_budget(usage)
        with _semantic_span(
            "response.policy_gate",
            "GUARDRAIL",
            inputs={
                "route": route.value,
                "document_ids": [item.document_id for item in evidence],
            },
        ) as span:
            gates = evaluate_draft(draft, route=route, evidence=evidence)
            review = requires_human_review(
                classification=classification,
                route=route,
                draft=draft,
                findings=gates,
                policy=self.policy,
            )
            _set_outputs(
                span,
                {
                    "passed": all(item.passed for item in gates),
                    "failed": [item.name for item in gates if not item.passed],
                    "requires_review": review,
                },
            )

        actions = plan_actions(request, classification, route, draft)
        return PreparedCase(
            email=request,
            access_context_digest=access_context_digest(access),
            knowledge_release=self.policy.knowledge_release,
            classification=classification,
            route=route,
            route_reasons=route_reasons,
            evidence=evidence,
            draft=draft,
            gates=gates,
            planned_actions=actions,
            requires_review=review,
            disposition=(Disposition.PENDING_REVIEW if review else Disposition.READY),
            total_usage=usage,
            application_release=self.policy.application_release,
        )

    async def commit(
        self,
        prepared: PreparedCase | Mapping[str, Any],
        *,
        review: ReviewDecision | Mapping[str, Any] | None = None,
    ) -> ExecutionResult:
        """Commit approved actions to a transactional, idempotent outbox."""

        case = _restore_contract(PreparedCase, prepared)
        access = _admit_provider_contract(
            AccessContext,
            await self.access_authorizer.authorize(case.email),
            boundary="access authorization",
        )
        _validate_access_binding(case.email, access)
        self._validate_prepared_case(case, access=access)
        decision = None if review is None else _restore_review(review)
        reviewer: VerifiedReviewerContext | None = None
        if case.requires_review and decision is None:
            raise ValueError("human review is required before any action is committed")
        if not case.requires_review and decision is not None:
            raise ValueError("an automatic case must not accept an invented review")
        if decision is not None and decision.case_id != case.email.case_id:
            raise ValueError("review decision belongs to a different case")
        if (
            decision is not None
            and decision.application_release != case.application_release
        ):
            raise ValueError(
                "review decision belongs to a different application release"
            )
        if decision is not None and decision.proposal_digest != proposal_digest(case):
            raise ValueError("review decision belongs to a different proposal")
        if decision is not None:
            reviewer = _admit_provider_contract(
                VerifiedReviewerContext,
                await self.review_authorizer.authorize(decision),
                boundary="review authorization",
            )
            _validate_review_authorization(
                decision,
                reviewer,
                required_group=self.policy.required_reviewer_group,
            )
        if decision is not None and decision.action is ReviewAction.REJECT:
            return ExecutionResult(
                case_id=case.email.case_id,
                application_release=case.application_release,
                disposition=Disposition.HANDLED_BY_HUMAN,
                review_action=decision.action,
                review_reason=decision.reason,
                reviewer_group=reviewer.reviewer_group,
                reviewer_subject_ref=reviewer.reviewer_subject_ref,
                proposal_digest=decision.proposal_digest,
            )

        actions = case.planned_actions
        if decision is not None and decision.action is ReviewAction.EDIT:
            actions = self._reviewed_actions(case, decision.edited_response or "")
        elif not all(finding.passed for finding in case.gates):
            raise ValueError(
                "a failed policy gate cannot be approved unchanged; edit or reject"
            )
        if case.draft.abstained and not actions:
            raise ValueError("an abstention must be handled by a human, not approved")

        with _semantic_span(
            "outbox.commit_batch",
            "TOOL",
            inputs={
                "case_sha256": _digest_text(case.email.case_id),
                "actions": [
                    {
                        "kind": action.kind.value,
                        "idempotency_key_sha256": _trace_ref(
                            "outbox-idempotency", action.idempotency_key
                        ),
                    }
                    for action in actions
                ],
            },
        ) as span:
            returned = await self.outbox.enqueue_batch_once(actions)
            receipts = tuple(
                _admit_provider_contract(
                    ActionReceipt,
                    receipt,
                    boundary="outbox receipt",
                )
                for receipt in returned
            )
            _validate_outbox_receipts(actions, receipts)
            _set_outputs(
                span,
                [
                    {
                        "kind": receipt.kind.value,
                        "status": receipt.status.value,
                        "external_ref_sha256": _trace_ref(
                            f"outbox-{receipt.kind.value}", receipt.external_ref
                        ),
                        "duplicate": receipt.duplicate,
                    }
                    for receipt in receipts
                ],
            )
        reply = next(
            (
                action.reply_body
                for action in actions
                if action.kind is ActionKind.ENQUEUE_REPLY
            ),
            None,
        )
        return ExecutionResult(
            case_id=case.email.case_id,
            application_release=case.application_release,
            disposition=(
                Disposition.QUEUED if receipts else Disposition.HANDLED_BY_HUMAN
            ),
            receipts=receipts,
            final_response=reply,
            review_action=decision.action if decision is not None else None,
            review_reason=decision.reason if decision is not None else None,
            reviewer_group=reviewer.reviewer_group if reviewer is not None else None,
            reviewer_subject_ref=(
                reviewer.reviewer_subject_ref if reviewer is not None else None
            ),
            proposal_digest=(
                decision.proposal_digest if decision is not None else None
            ),
        )

    def _reviewed_actions(
        self,
        case: PreparedCase,
        edited_response: str,
    ) -> tuple[PlannedAction, ...]:
        edited = DraftResponse(
            body=edited_response,
            citations=case.draft.citations,
            abstained=False,
            confidence=1.0,
            prompt_version=case.draft.prompt_version,
            usage=ModelUsage(),
        )
        findings = evaluate_draft(edited, route=case.route, evidence=case.evidence)
        if not all(finding.passed for finding in findings):
            failures = ", ".join(
                finding.name for finding in findings if not finding.passed
            )
            raise ValueError(f"reviewed response still fails policy: {failures}")
        # Re-plan from trusted inputs rather than editing persisted actions.
        # This also lets a reviewer provide a safe response when the model
        # abstained and therefore proposed no reply action.
        return plan_actions(
            case.email,
            case.classification,
            case.route,
            edited,
        )

    def _validate_prepared_case(
        self,
        case: PreparedCase,
        *,
        access: AccessContext,
    ) -> None:
        """Re-derive every deterministic decision before an external write."""

        if case.application_release != self.policy.application_release:
            raise ValueError(
                "prepared checkpoint integrity failed: application release changed"
            )
        if case.knowledge_release != self.policy.knowledge_release:
            raise ValueError(
                "prepared checkpoint integrity failed: knowledge release changed"
            )
        if case.access_context_digest != access_context_digest(access):
            raise ValueError(
                "prepared checkpoint integrity failed: access authorization changed"
            )
        expected_route, expected_reasons = route_email(case.email, case.classification)
        if (case.route, case.route_reasons) != (expected_route, expected_reasons):
            raise ValueError(
                "prepared checkpoint integrity failed: route or reasons changed"
            )
        verify_evidence_scope(
            case.evidence,
            access=access,
            release=case.knowledge_release,
        )
        expected_gates = evaluate_draft(
            case.draft,
            route=case.route,
            evidence=case.evidence,
        )
        if case.gates != expected_gates:
            raise ValueError(
                "prepared checkpoint integrity failed: policy findings changed"
            )
        expected_review = requires_human_review(
            classification=case.classification,
            route=case.route,
            draft=case.draft,
            findings=expected_gates,
            policy=self.policy,
        )
        if case.requires_review is not expected_review:
            raise ValueError(
                "prepared checkpoint integrity failed: review policy changed"
            )
        expected_actions = plan_actions(
            case.email,
            case.classification,
            case.route,
            case.draft,
        )
        if case.planned_actions != expected_actions:
            raise ValueError(
                "prepared checkpoint integrity failed: planned actions changed"
            )
        expected_usage = _combined_usage(
            case.classification.usage,
            case.draft.usage,
        )
        if case.total_usage != expected_usage:
            raise ValueError(
                "prepared checkpoint integrity failed: usage evidence changed"
            )
        self._enforce_budget(expected_usage)

    def _enforce_budget(self, usage: ModelUsage) -> None:
        limits = (
            (usage.model_calls, self.budget.max_model_calls, "model calls"),
            (usage.input_tokens, self.budget.max_input_tokens, "input tokens"),
            (usage.output_tokens, self.budget.max_output_tokens, "output tokens"),
        )
        for observed, maximum, label in limits:
            if observed > maximum:
                raise BudgetExceededError(
                    f"request used {observed} {label}; configured maximum is {maximum}"
                )


def checkpoint_state(prepared: PreparedCase) -> dict[str, Any]:
    """Serialize only ordinary JSON-compatible values for durable checkpoints."""

    return prepared.model_dump(mode="json")


def proposal_digest(prepared: PreparedCase) -> str:
    """Bind a human decision to one exact, release-specific proposal."""

    canonical = json.dumps(
        prepared.model_dump(mode="json"),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + sha256(canonical.encode()).hexdigest()


def access_context_digest(access: AccessContext) -> str:
    """Persist only a stable entitlement digest, not authorization claims."""

    canonical = json.dumps(
        {
            "access_context_ref": access.access_context_ref,
            "tenant_id": access.tenant_id,
            "groups": sorted(access.groups),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + sha256(canonical.encode()).hexdigest()


def _restore_review(value: ReviewDecision | Mapping[str, Any]) -> ReviewDecision:
    try:
        return _admit_provider_contract(
            ReviewDecision,
            value,
            boundary="review decision",
        )
    except (TypeError, ValueError, ValidationError):
        raise ValueError("review decision failed strict admission policy") from None


def _admit_provider_contract(model: Any, value: Any, *, boundary: str) -> Any:
    """Revalidate even model instances returned by an untrusted adapter."""

    payload = value.model_dump(mode="json") if isinstance(value, model) else value
    try:
        return model.model_validate_json(json.dumps(payload), strict=True)
    except (TypeError, ValueError, ValidationError):
        raise ValueError(f"{boundary} returned an invalid strict contract") from None


def _admit_draft(candidate: Any) -> DraftResponse:
    if candidate is not None:
        payload = (
            candidate.model_dump(mode="json")
            if isinstance(candidate, DraftResponse)
            else candidate
        )
        try:
            return DraftResponse.model_validate_json(json.dumps(payload), strict=True)
        except (TypeError, ValueError, ValidationError):
            pass
    return DraftResponse(
        body=(
            "The generated draft failed durable-state admission. A support "
            "specialist must review this request."
        ),
        citations=(),
        abstained=True,
        confidence=1.0,
        prompt_version="policy-quarantine-v1",
        usage=ModelUsage(
            model_calls=1,
            measurement_source=MeasurementSource.CONNECTED,
        ),
        quarantined=True,
    )


def _validate_access_binding(email: RedactedEmail, access: AccessContext) -> None:
    if access.access_context_ref != email.access_context_ref:
        raise ValueError("access authorization belongs to a different ingress record")
    if access.tenant_id != email.tenant_id:
        raise ValueError("caller tenant does not match verified ingress authorization")


def _validate_review_authorization(
    decision: ReviewDecision,
    reviewer: VerifiedReviewerContext,
    *,
    required_group: str,
) -> None:
    bindings = (
        (reviewer.authorization_ref, decision.authorization_ref),
        (reviewer.authorized_case_id, decision.case_id),
        (reviewer.authorized_proposal_digest, decision.proposal_digest),
        (reviewer.application_release, decision.application_release),
    )
    if any(observed != expected for observed, expected in bindings):
        raise ValueError("review authorization is not bound to this decision")
    if reviewer.reviewer_group != required_group:
        raise ValueError("reviewer is not a member of the required review group")
    if decision.action not in reviewer.allowed_actions:
        raise ValueError("reviewer is not authorized for this review action")


def _validate_outbox_receipts(
    actions: tuple[PlannedAction, ...],
    receipts: tuple[ActionReceipt, ...],
) -> None:
    """Reject an adapter that acknowledges a different or partial action set."""

    expected = {(action.kind, action.idempotency_key) for action in actions}
    observed = {(receipt.kind, receipt.idempotency_key) for receipt in receipts}
    if len(receipts) != len(actions) or observed != expected:
        raise ValueError(
            "outbox receipt set is not bound to the committed action batch"
        )


class _InferenceBudgetLedger:
    """Reserve worst-case calls/context before invoking a provider."""

    def __init__(self, budget: RuntimeBudget) -> None:
        self.budget = budget
        self.reserved_calls = 0
        self.reported_calls = 0
        self.reserved_input_tokens = 0
        self.reported_input_tokens = 0
        self.reported_output_tokens = 0

    def reserve_model_call(
        self,
        boundary: str,
        *,
        estimated_input_tokens: int,
    ) -> int:
        calls = max(self.reserved_calls, self.reported_calls)
        if calls + 1 > self.budget.max_model_calls:
            raise BudgetExceededError(
                f"{boundary} would exceed the configured model-call budget"
            )
        inputs = max(self.reserved_input_tokens, self.reported_input_tokens)
        if inputs + estimated_input_tokens > self.budget.max_input_tokens:
            raise BudgetExceededError(
                f"{boundary} would exceed the configured input-token budget"
            )
        remaining_output = self.budget.max_output_tokens - self.reported_output_tokens
        if remaining_output <= 0:
            raise BudgetExceededError(
                f"{boundary} has no remaining configured output-token budget"
            )
        self.reserved_calls = calls + 1
        self.reserved_input_tokens = inputs + estimated_input_tokens
        return remaining_output

    def observe(self, usage: ModelUsage, *, boundary: str) -> None:
        self.reported_calls += usage.model_calls
        self.reported_input_tokens += usage.input_tokens
        self.reported_output_tokens += usage.output_tokens
        limits = (
            (self.reported_calls, self.budget.max_model_calls, "model calls"),
            (
                self.reported_input_tokens,
                self.budget.max_input_tokens,
                "input tokens",
            ),
            (
                self.reported_output_tokens,
                self.budget.max_output_tokens,
                "output tokens",
            ),
        )
        for observed, maximum, label in limits:
            if observed > maximum:
                raise BudgetExceededError(
                    f"{boundary} reported {observed} {label}; configured maximum "
                    f"is {maximum}"
                )


def _estimate_tokens(*values: str) -> int:
    characters = sum(len(value) for value in values)
    return max(1, (characters + 3) // 4)


def _classification_trace_output(classification: Classification) -> dict[str, object]:
    return {
        "intent": classification.intent.value,
        "urgency": classification.urgency.value,
        "risk": classification.risk.value,
        "confidence": classification.confidence,
        "usage": classification.usage.model_dump(mode="json"),
    }


def _draft_trace_output(draft: DraftResponse) -> dict[str, object]:
    return {
        "response_sha256": _digest_text(draft.body),
        "response_characters": len(draft.body),
        "citation_sha256": [
            _trace_ref("citation", citation) for citation in draft.citations
        ],
        "abstained": draft.abstained,
        "quarantined": draft.quarantined,
        "prompt_version": draft.prompt_version,
        "usage": draft.usage.model_dump(mode="json"),
    }


def _restore_contract(model: Any, value: Any) -> Any:
    """Strictly restore either an in-memory model or JSON-compatible state."""

    if isinstance(value, model):
        return value
    # Strict Python validation expects enum instances and tuples. A durable
    # checkpoint has JSON strings and arrays, for which Pydantic's strict JSON
    # contract performs the intended lossless decoding without broad coercion.
    return model.model_validate_json(json.dumps(value), strict=True)


def _combined_usage(*records: ModelUsage) -> ModelUsage:
    costs = [record.cost_usd for record in records]
    source = (
        MeasurementSource.CONNECTED
        if any(
            record.measurement_source is MeasurementSource.CONNECTED
            for record in records
        )
        else MeasurementSource.OFFLINE_FIXTURE
    )
    return ModelUsage(
        model_calls=sum(record.model_calls for record in records),
        input_tokens=sum(record.input_tokens for record in records),
        output_tokens=sum(record.output_tokens for record in records),
        cost_usd=sum(costs) if all(cost is not None for cost in costs) else None,
        measurement_source=source,
    )


def _request_trace_metadata(email: RedactedEmail) -> dict[str, object]:
    return {
        "case_sha256": _digest_text(email.case_id),
        "thread_sha256": _digest_text(email.thread_id),
        "tenant_sha256": _digest_text(email.tenant_id),
        "message_sha256": _digest_text(email.message_id),
        "subject_characters": len(email.subject),
        "body_characters": len(email.body),
        "sanitization_version": email.sanitization_version,
    }


def _digest_text(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _trace_ref(namespace: str, value: str) -> str:
    material = f"email-support-trace:v1:{namespace}:{value}"
    return "sha256:" + sha256(material.encode()).hexdigest()


@contextmanager
def _semantic_span(
    name: str,
    span_type: str,
    *,
    inputs: object,
) -> Iterator[Any | None]:
    """Add a non-duplicating child span only when a native trace is active.

    LangGraph autologging owns the root trace. These children add the domain
    taxonomy and the scorer-visible RETRIEVER output that a custom graph node
    would otherwise lack. Inputs are deliberately metadata-only.
    """

    state = current_trace_state()
    if state.policy.capture_mode is TraceCaptureMode.OFF:
        yield None
        return
    try:
        mlflow = native_mlflow()
        parent = native_active_span()
    except RuntimeError:
        yield None
        return
    if parent is None:
        yield None
        return
    with mlflow.start_span(name=name, span_type=span_type) as native_span:
        span = GovernedSpan(native_span, state.policy)
        span.set_inputs(inputs)
        yield span


def _set_outputs(span: Any | None, value: object) -> None:
    if span is not None:
        span.set_outputs(value)
