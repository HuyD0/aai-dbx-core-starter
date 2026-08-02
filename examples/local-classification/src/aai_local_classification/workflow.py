"""Repeatable baseline, comparison, frozen test gate, and registry workflow."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from aai_local_classification.contracts import (
    BaselineEvidence,
    CandidateResult,
    GateEvidence,
    ProjectSettings,
    PromotionDecision,
    PromotionEvidence,
    SelectionEvidence,
    SplitName,
)
from aai_local_classification.data import (
    load_manifest,
    load_split,
    prepare_dataset,
    validate_manifest_contract,
)
from aai_local_classification.evaluation import (
    evaluate_probabilities,
    maximum_recall_gap,
    metric_dict,
    promotion_checks,
    recall_slices,
    select_threshold,
)
from aai_local_classification.modeling import (
    build_baseline,
    build_candidate,
    candidate_specs,
    feature_frame,
)
from aai_local_classification.policy import (
    gate_policy_sha256,
    selection_policy_sha256,
)
from aai_local_classification.tracking import (
    configure_mlflow,
    local_paths,
    log_dataset,
    log_linked_metrics,
    log_reproducibility_artifacts,
    run_tags,
)


def _positive_probability(model, features):
    classes = list(model.classes_)
    return model.predict_proba(features)[:, classes.index(1)]


def _representative_example(features):
    complete = features.dropna()
    return (complete if len(complete) >= 5 else features).head(5)


def _write_state(paths, name: str, payload: dict[str, object]) -> Path:
    path = paths.state_root / name
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _selected_candidate(evidence: SelectionEvidence) -> CandidateResult:
    matches = [
        item for item in evidence.candidates if item.run_id == evidence.selected_run_id
    ]
    if len(matches) != 1:
        raise ValueError("Selection evidence must identify exactly one candidate run")
    selected = matches[0]
    if (
        selected.candidate_name != evidence.selected_candidate
        or selected.model_id != evidence.selected_model_id
        or selected.model_uri != evidence.selected_model_uri
        or selected.dataset_sha256 != evidence.dataset_sha256
    ):
        raise ValueError("Selection evidence has inconsistent candidate linkage")
    return selected


def _validate_selection_policy(
    evidence: SelectionEvidence,
    settings: ProjectSettings,
) -> None:
    if evidence.selection_policy_sha256 != selection_policy_sha256(settings):
        raise ValueError(
            "Selection evidence belongs to different code, dependencies, or policy; "
            "run candidate selection again"
        )
    _selected_candidate(evidence)


def _validate_decision_linkage(
    decision: GateEvidence,
    selection: SelectionEvidence,
    settings: ProjectSettings,
    dataset_sha256: str,
) -> None:
    _validate_selection_policy(selection, settings)
    chosen = _selected_candidate(selection)
    passed = all(decision.checks.model_dump(mode="python").values())
    expected_decision = PromotionDecision.ADOPT if passed else PromotionDecision.REJECT
    expected = {
        "gate outcome": (decision.decision, expected_decision),
        "selected candidate": (
            decision.selected_candidate,
            selection.selected_candidate,
        ),
        "selected run": (decision.selected_run_id, selection.selected_run_id),
        "selected model": (decision.selected_model_id, selection.selected_model_id),
        "dataset": (decision.dataset_sha256, dataset_sha256),
        "selection dataset": (selection.dataset_sha256, dataset_sha256),
        "selection policy": (
            decision.selection_policy_sha256,
            selection.selection_policy_sha256,
        ),
        "gate policy": (decision.gate_policy_sha256, gate_policy_sha256(settings)),
        "threshold": (decision.threshold, chosen.threshold_selection.threshold),
    }
    mismatches = [
        name for name, (actual, wanted) in expected.items() if actual != wanted
    ]
    if mismatches:
        raise ValueError(
            "Release decision is not bound to the current " + ", ".join(mismatches)
        )


def _log_sklearn_model(
    model,
    features,
    *,
    candidate_name: str,
    threshold: float,
    dataset_sha256: str,
):
    probabilities = model.predict_proba(features)
    signature = infer_signature(features, probabilities)
    return mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=_representative_example(features),
        signature=signature,
        serialization_format="skops",
        skops_trusted_types=["numpy.dtype"],
        pyfunc_predict_fn="predict_proba",
        metadata={
            "candidate_name": candidate_name,
            "decision_threshold": threshold,
            "positive_class": 1,
            "dataset_sha256": dataset_sha256,
        },
    )


def ensure_prepared(
    settings: ProjectSettings,
    project_root: Path | None = None,
):
    paths = local_paths(project_root)
    if not (paths.data_root / "manifest.json").is_file():
        return prepare_dataset(settings, paths.data_root)
    manifest = load_manifest(paths.data_root)
    validate_manifest_contract(manifest, settings)
    return manifest


def run_baseline(
    settings: ProjectSettings,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Fit and record a no-skill baseline before any candidate comparison."""

    paths = configure_mlflow(settings, project_root)
    manifest = ensure_prepared(settings, project_root)
    train = load_split(settings, SplitName.TRAIN, paths.data_root)
    validation = load_split(settings, SplitName.VALIDATION, paths.data_root)
    x_train = feature_frame(train, settings)
    y_train = train[settings.data.target_column]
    x_validation = feature_frame(validation, settings)
    y_validation = validation[settings.data.target_column]
    model = build_baseline(settings).fit(x_train, y_train)
    probabilities = _positive_probability(model, x_validation)
    metrics = evaluate_probabilities(
        y_validation,
        probabilities,
        0.5,
        false_negative_cost=settings.selection.false_negative_cost,
        false_positive_cost=settings.selection.false_positive_cost,
    )

    with mlflow.start_run(
        run_name="baseline-dummy-prior",
        tags=run_tags(
            settings,
            paths,
            lifecycle_role="baseline",
            dataset_sha256=manifest.dataset_sha256,
        ),
    ) as run:
        mlflow.log_params(
            {
                "model_family": "dummy_classifier",
                "strategy": "prior",
                "random_seed": settings.random_seed,
                "threshold": 0.5,
                "positive_label": 1,
            }
        )
        log_dataset(train, settings, paths, SplitName.TRAIN)
        validation_dataset = log_dataset(
            validation, settings, paths, SplitName.VALIDATION
        )
        model_info = _log_sklearn_model(
            model,
            x_train,
            candidate_name="dummy-prior",
            threshold=0.5,
            dataset_sha256=manifest.dataset_sha256,
        )
        log_linked_metrics(
            metric_dict(metrics, "validation_"),
            model_id=model_info.model_id,
            dataset=validation_dataset,
        )
        log_reproducibility_artifacts(paths)
        baseline_evidence = BaselineEvidence(
            schema_version=1,
            run_id=run.info.run_id,
            model_id=model_info.model_id,
            model_uri=model_info.model_uri,
            metrics=metrics,
            dataset_sha256=manifest.dataset_sha256,
            selection_policy_sha256=selection_policy_sha256(settings),
        )
        evidence = baseline_evidence.model_dump(mode="json")
        mlflow.log_dict(evidence, "evaluation/baseline.json")
    _write_state(paths, "baseline.json", evidence)
    return evidence


def run_candidate_selection(
    settings: ProjectSettings,
    project_root: Path | None = None,
) -> SelectionEvidence:
    """Compare declared candidates using training and validation data only."""

    paths = configure_mlflow(settings, project_root)
    manifest = ensure_prepared(settings, project_root)
    train = load_split(settings, SplitName.TRAIN, paths.data_root)
    validation = load_split(settings, SplitName.VALIDATION, paths.data_root)
    x_train = feature_frame(train, settings)
    y_train = train[settings.data.target_column]
    x_validation = feature_frame(validation, settings)
    y_validation = validation[settings.data.target_column]
    results: list[CandidateResult] = []

    with mlflow.start_run(
        run_name="change-model-selection",
        tags=run_tags(
            settings,
            paths,
            lifecycle_role="change",
            dataset_sha256=manifest.dataset_sha256,
        ),
    ) as selection_run:
        mlflow.log_param("primary_metric", settings.selection.primary_metric)
        mlflow.log_param("test_data_accessed", False)
        log_dataset(train, settings, paths, SplitName.TRAIN)
        log_dataset(validation, settings, paths, SplitName.VALIDATION)
        for spec in candidate_specs():
            with mlflow.start_run(
                run_name=f"change-{spec.name}",
                nested=True,
                tags=run_tags(
                    settings,
                    paths,
                    lifecycle_role="change-candidate",
                    dataset_sha256=manifest.dataset_sha256,
                ),
            ) as candidate_run:
                model = build_candidate(spec, settings).fit(x_train, y_train)
                probabilities = _positive_probability(model, x_validation)
                threshold = select_threshold(y_validation, probabilities, settings)
                mlflow.log_params(
                    {
                        "candidate_name": spec.name,
                        "model_family": spec.model_family,
                        "random_seed": settings.random_seed,
                        "positive_label": 1,
                        "threshold_selection_split": "validation",
                        "test_data_accessed": False,
                    }
                )
                mlflow.set_tag("rationale", spec.rationale)
                train_dataset = log_dataset(train, settings, paths, SplitName.TRAIN)
                validation_dataset = log_dataset(
                    validation, settings, paths, SplitName.VALIDATION
                )
                model_info = _log_sklearn_model(
                    model,
                    x_train,
                    candidate_name=spec.name,
                    threshold=threshold.threshold,
                    dataset_sha256=manifest.dataset_sha256,
                )
                log_linked_metrics(
                    metric_dict(threshold.validation_metrics, "validation_"),
                    model_id=model_info.model_id,
                    dataset=validation_dataset,
                )
                mlflow.log_metric(
                    "training_row_count",
                    float(len(train)),
                    model_id=model_info.model_id,
                    dataset=train_dataset,
                )
                mlflow.log_dict(
                    threshold.model_dump(mode="json"),
                    "evaluation/threshold-selection.json",
                )
                log_reproducibility_artifacts(paths)
                results.append(
                    CandidateResult(
                        schema_version=1,
                        candidate_name=spec.name,
                        run_id=candidate_run.info.run_id,
                        model_id=model_info.model_id,
                        model_uri=model_info.model_uri,
                        threshold_selection=threshold,
                        dataset_sha256=manifest.dataset_sha256,
                    )
                )

        best_primary = max(
            result.threshold_selection.validation_metrics.average_precision
            for result in results
        )
        eligible = [
            result
            for result in results
            if best_primary
            - result.threshold_selection.validation_metrics.average_precision
            <= settings.selection.simpler_model_tolerance
        ]
        complexity = {spec.name: spec.complexity_rank for spec in candidate_specs()}
        selected = min(
            eligible,
            key=lambda result: (
                complexity[result.candidate_name],
                result.threshold_selection.validation_metrics.cost_per_1000,
            ),
        )
        selection_rule = (
            "highest validation average precision, preferring the lower-complexity "
            f"candidate within {settings.selection.simpler_model_tolerance:.3f}"
        )
        evidence = SelectionEvidence(
            schema_version=1,
            selection_run_id=selection_run.info.run_id,
            primary_metric=settings.selection.primary_metric,
            selection_rule=selection_rule,
            selected_candidate=selected.candidate_name,
            selected_run_id=selected.run_id,
            selected_model_id=selected.model_id,
            selected_model_uri=selected.model_uri,
            dataset_sha256=manifest.dataset_sha256,
            selection_policy_sha256=selection_policy_sha256(settings),
            candidates=tuple(results),
        )
        mlflow.set_tags(
            {
                "selected_candidate": selected.candidate_name,
                "selected_run_id": selected.run_id,
                "selected_model_id": selected.model_id,
            }
        )
        mlflow.log_dict(evidence.model_dump(mode="json"), "evaluation/selection.json")

    _write_state(paths, "selection.json", evidence.model_dump(mode="json"))
    return evidence


def load_selection(project_root: Path | None = None) -> SelectionEvidence:
    path = local_paths(project_root).state_root / "selection.json"
    return SelectionEvidence.model_validate_json(path.read_text(encoding="utf-8"))


def load_decision(project_root: Path | None = None) -> GateEvidence:
    path = local_paths(project_root).state_root / "decision.json"
    return GateEvidence.model_validate_json(path.read_text(encoding="utf-8"))


def run_frozen_test_gate(
    settings: ProjectSettings,
    project_root: Path | None = None,
    selection: SelectionEvidence | None = None,
) -> GateEvidence:
    """Evaluate the exact selected artifact once on the frozen test split."""

    paths = configure_mlflow(settings, project_root)
    manifest = ensure_prepared(settings, project_root)
    selected = selection or load_selection(project_root)
    if selected.dataset_sha256 != manifest.dataset_sha256:
        raise ValueError("Selection evidence belongs to a different dataset")
    _validate_selection_policy(selected, settings)
    chosen = _selected_candidate(selected)
    decision_path = paths.state_root / "decision.json"
    if decision_path.is_file():
        existing = GateEvidence.model_validate_json(
            decision_path.read_text(encoding="utf-8")
        )
        if existing.dataset_sha256 == selected.dataset_sha256:
            try:
                _validate_decision_linkage(
                    existing,
                    selected,
                    settings,
                    manifest.dataset_sha256,
                )
            except ValueError as error:
                raise ValueError(
                    "This frozen-test dataset version already has a decision for "
                    "different code, dependencies, policy, or model. Create a new "
                    "frozen-test version instead of reusing consumed test evidence."
                ) from error
            return existing
    model = mlflow.sklearn.load_model(selected.selected_model_uri)
    test = load_split(settings, SplitName.TEST, paths.data_root)
    x_test = feature_frame(test, settings)
    y_test = test[settings.data.target_column]
    probabilities = _positive_probability(model, x_test)
    threshold = chosen.threshold_selection.threshold
    metrics = evaluate_probabilities(
        y_test,
        probabilities,
        threshold,
        false_negative_cost=settings.selection.false_negative_cost,
        false_positive_cost=settings.selection.false_positive_cost,
    )
    slices = recall_slices(test, y_test, probabilities, threshold)
    recall_gap = maximum_recall_gap(slices)
    checks = promotion_checks(metrics, recall_gap, settings)
    decision = (
        PromotionDecision.ADOPT if all(checks.values()) else PromotionDecision.REJECT
    )

    with mlflow.start_run(
        run_name="result-frozen-test-gate",
        tags=run_tags(
            settings,
            paths,
            lifecycle_role="result",
            dataset_sha256=manifest.dataset_sha256,
        ),
    ) as test_run:
        mlflow.set_tags(
            {
                "decision": decision.value,
                "selected_candidate": selected.selected_candidate,
                "selected_run_id": selected.selected_run_id,
                "selected_model_id": selected.selected_model_id,
                "selection_run_id": selected.selection_run_id,
                "selection_policy_sha256": selected.selection_policy_sha256,
                "gate_policy_sha256": gate_policy_sha256(settings),
            }
        )
        mlflow.log_params(
            {
                "decision_threshold": threshold,
                "threshold_selected_on": "validation",
                "test_policy": "frozen-one-time-release-evaluation",
                "false_negative_cost": settings.selection.false_negative_cost,
                "false_positive_cost": settings.selection.false_positive_cost,
            }
        )
        test_dataset = log_dataset(test, settings, paths, SplitName.TEST)
        linked_metrics = metric_dict(metrics, "test_") | {
            "test_maximum_slice_recall_gap": recall_gap
        }
        log_linked_metrics(
            linked_metrics,
            model_id=selected.selected_model_id,
            dataset=test_dataset,
        )
        # Native classic evaluation is diagnostic evidence. The explicit gate above
        # remains authoritative because it includes the selected business threshold.
        evaluation_frame = x_test.copy()
        evaluation_frame[settings.data.target_column] = y_test.to_numpy()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Hint: Inferred schema contains integer column.*",
                category=UserWarning,
            )
            mlflow.models.evaluate(
                model=lambda frame: (
                    _positive_probability(model, frame) >= threshold
                ).astype(int),
                data=evaluation_frame,
                targets=settings.data.target_column,
                model_type="classifier",
                evaluator_config={"log_model_explainability": False},
            )
        mlflow.log_table(slices, "evaluation/recall-slices.json")
        rationale = (
            "All predeclared frozen-test checks passed."
            if decision is PromotionDecision.ADOPT
            else "One or more predeclared frozen-test checks failed."
        )
        evidence = GateEvidence(
            schema_version=1,
            decision=decision,
            selected_candidate=selected.selected_candidate,
            selected_run_id=selected.selected_run_id,
            selected_model_id=selected.selected_model_id,
            test_run_id=test_run.info.run_id,
            threshold=threshold,
            dataset_sha256=manifest.dataset_sha256,
            selection_policy_sha256=selected.selection_policy_sha256,
            gate_policy_sha256=gate_policy_sha256(settings),
            metrics=linked_metrics,
            checks=checks,
            rationale=rationale,
        )
        mlflow.log_dict(
            evidence.model_dump(mode="json"), "evaluation/promotion-decision.json"
        )
        log_reproducibility_artifacts(paths)

    _write_state(paths, "decision.json", evidence.model_dump(mode="json"))
    return evidence


def promote_if_approved(
    settings: ProjectSettings,
    decision: GateEvidence,
    project_root: Path | None = None,
    selection: SelectionEvidence | None = None,
) -> dict[str, object]:
    """Register and alias only an exact model that passed the frozen gate."""

    paths = configure_mlflow(settings, project_root)
    manifest = ensure_prepared(settings, project_root)
    selected = selection or load_selection(project_root)
    _validate_decision_linkage(
        decision,
        selected,
        settings,
        manifest.dataset_sha256,
    )
    if decision.decision is not PromotionDecision.ADOPT:
        return {
            "registered": False,
            "decision": decision.decision.value,
            "reason": "The champion alias is unchanged because the gate did not pass.",
        }
    promotion_path = paths.state_root / "promotion.json"
    if promotion_path.is_file():
        existing = PromotionEvidence.model_validate_json(
            promotion_path.read_text(encoding="utf-8")
        )
        if existing.test_run_id == decision.test_run_id:
            if (
                existing.selected_model_id != selected.selected_model_id
                or existing.dataset_sha256 != manifest.dataset_sha256
                or existing.selection_policy_sha256 != selected.selection_policy_sha256
                or existing.gate_policy_sha256 != decision.gate_policy_sha256
            ):
                raise ValueError("Cached promotion evidence has inconsistent linkage")
            return existing.model_dump(mode="json")
    version = mlflow.register_model(
        selected.selected_model_uri,
        settings.registered_model_name,
    )
    client = mlflow.MlflowClient()
    tags = {
        "validation_status": "passed",
        "test_run_id": decision.test_run_id,
        "dataset_sha256": decision.dataset_sha256,
        "decision_threshold": str(decision.threshold),
        "selected_candidate": decision.selected_candidate,
        "selected_model_id": decision.selected_model_id,
        "selection_policy_sha256": decision.selection_policy_sha256,
        "gate_policy_sha256": decision.gate_policy_sha256,
    }
    for key, value in tags.items():
        client.set_model_version_tag(
            settings.registered_model_name,
            version.version,
            key,
            value,
        )
    client.set_registered_model_alias(
        settings.registered_model_name,
        "champion",
        version.version,
    )
    promotion = PromotionEvidence(
        schema_version=1,
        registered=True,
        model_name=settings.registered_model_name,
        model_version=int(version.version),
        alias="champion",
        model_uri=f"models:/{settings.registered_model_name}@champion",
        selected_model_id=decision.selected_model_id,
        test_run_id=decision.test_run_id,
        dataset_sha256=decision.dataset_sha256,
        selection_policy_sha256=decision.selection_policy_sha256,
        gate_policy_sha256=decision.gate_policy_sha256,
    )
    evidence = promotion.model_dump(mode="json")
    _write_state(paths, "promotion.json", evidence)
    return evidence


def run_full_workflow(
    settings: ProjectSettings,
    project_root: Path | None = None,
) -> dict[str, object]:
    manifest = ensure_prepared(settings, project_root)
    baseline = run_baseline(settings, project_root)
    try:
        selection = load_selection(project_root)
        _validate_selection_policy(selection, settings)
        if selection.dataset_sha256 != manifest.dataset_sha256:
            raise ValueError("Selection evidence belongs to a different dataset")
    except (OSError, ValueError):
        selection = run_candidate_selection(settings, project_root)
    decision = run_frozen_test_gate(settings, project_root, selection)
    promotion = promote_if_approved(settings, decision, project_root, selection)
    return {
        "dataset_sha256": manifest.dataset_sha256,
        "baseline": baseline,
        "selection": selection.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
        "promotion": promotion,
    }
