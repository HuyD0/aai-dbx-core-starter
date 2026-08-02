"""Command-line workflow for the offline local fine-tuning curriculum."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .capstone import (
    CAPSTONE_SYSTEM_PROMPT,
    CapstoneEvaluationReport,
    CapstonePrediction,
    CapstoneRecord,
    ReadinessStatus,
    build_hybrid_review,
    compact_expected,
    compact_review,
    deterministic_capstone_predictions,
    evaluate_capstone_predictions,
    generate_capstone_dataset,
    load_capstone_records,
    render_capstone_mlx_dataset,
    rule_catalog,
)
from .data import (
    DatasetIntegrityError,
    PreparationConfig,
    assert_no_leakage,
    check_split_files,
    prepare_dataset,
    require_valid_manifest,
)
from .evaluation import (
    BaselineEvaluation,
    EvaluationRecord,
    EvaluationReport,
    Evaluator,
    KeywordRuleBaseline,
    MajorityBaseline,
    Prediction,
    decide_lora_promotion,
    format_error_analysis,
    load_records_jsonl,
    write_predictions_jsonl,
    write_report_json,
)
from .modeling import LocalMLXPredictor, PromptStrategy, build_messages
from .offline import (
    OfflineAssetError,
    apple_silicon_status,
    deny_network,
    enable_offline_environment,
    prove_socket_denial,
    require_assets,
    verify_flight_manifest,
    write_flight_manifest,
)
from .settings import PROJECT_ROOT, ProjectSettings, load_settings
from .training import run_lora


class StudyCommandError(RuntimeError):
    """A study command could not satisfy its documented contract."""


def _preparation_config(settings: ProjectSettings) -> PreparationConfig:
    counts = settings.dataset.per_intent
    return PreparationConfig(
        seed=settings.dataset.split_seed,
        train_per_intent=counts.train,
        validation_per_intent=counts.validation,
        test_per_intent=counts.test,
        expected_intent_count=27,
        near_duplicate_threshold=0.9,
        source_dataset="bitext-customer-support",
        source_version=f"kaggle-v{settings.dataset.version}",
        source_provider="kaggle",
        source_owner=settings.dataset.owner,
        source_url=settings.dataset.url,
        source_license=settings.dataset.license,
        date_accessed=settings.dataset.accessed_on,
        processing_config_path="configs/project.yaml",
        output_version="1.0.0",
    )


def _prepare_data(settings: ProjectSettings) -> None:
    result = prepare_dataset(
        settings.archive_path,
        settings.processed_dir,
        _preparation_config(settings),
        related_raw_paths=(settings.csv_path,),
    )
    counts = {
        name: descriptor.record_count
        for name, descriptor in result.manifest.splits.items()
    }
    print(
        "Prepared leakage-safe Bitext splits: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )
    print(f"Dataset fingerprint: {result.manifest.dataset_fingerprint}")


def _load_splits(
    settings: ProjectSettings,
) -> tuple[list[EvaluationRecord], list[EvaluationRecord], list[EvaluationRecord]]:
    root = settings.processed_dir
    return (
        load_records_jsonl(root / "train.jsonl"),
        load_records_jsonl(root / "valid.jsonl"),
        load_records_jsonl(root / "test.jsonl"),
    )


def _require_prepared_split_integrity(processed_dir: Path) -> None:
    """Fail before model work when prepared dataset evidence no longer matches."""

    try:
        require_valid_manifest(processed_dir)
    except DatasetIntegrityError as error:
        raise StudyCommandError(
            "prepared dataset integrity check failed; rerun `make prepare-data`:\n"
            f"{error}"
        ) from error


def _write_evaluation(
    *,
    name: str,
    records: Sequence[EvaluationRecord],
    predictions: Sequence[Prediction],
    report: EvaluationReport,
) -> tuple[Path, Path]:
    output_dir = PROJECT_ROOT / "artifacts" / "evaluation"
    prediction_path = output_dir / f"{name}-predictions.jsonl"
    report_path = output_dir / f"{name}-report.json"
    write_predictions_jsonl(prediction_path, predictions)
    write_report_json(report_path, report)
    return prediction_path, report_path


def _track_report(
    settings: ProjectSettings,
    *,
    name: str,
    role: str,
    records: Sequence[EvaluationRecord],
    report: EvaluationReport,
) -> str:
    from .tracking import log_evaluation

    return log_evaluation(
        settings,
        run_name=name,
        role=role,
        method=name,
        metrics=report.flat_metrics(),
        report=report.model_dump(mode="json"),
        records=(record.model_dump(mode="json") for record in records),
        manifest_path=settings.processed_dir / "manifest.json",
        prediction_path=(
            PROJECT_ROOT / "artifacts" / "evaluation" / f"{name}-predictions.jsonl"
        ),
        model_based=name not in {"majority", "keyword-rule"},
    )


def _score_predictions(
    *,
    name: str,
    records: Sequence[EvaluationRecord],
    predictions: Sequence[Prediction],
    supported_intents: Sequence[str],
) -> EvaluationReport:
    report = Evaluator(supported_intents=supported_intents).evaluate(
        records, predictions
    )
    _write_evaluation(
        name=name,
        records=records,
        predictions=predictions,
        report=report,
    )
    return report


def _baseline_reports(
    settings: ProjectSettings,
    *,
    records: Sequence[EvaluationRecord] | None = None,
    track: bool = False,
) -> dict[str, EvaluationReport]:
    train, _, test = _load_splits(settings)
    evaluation_records = list(records or test)
    supported = tuple(sorted({record.target.intent for record in train}))
    methods = (
        ("majority", MajorityBaseline.fit(train)),
        ("keyword-rule", KeywordRuleBaseline.fit(train)),
    )
    reports: dict[str, EvaluationReport] = {}
    for name, method in methods:
        predictions = method.predict_many(evaluation_records)
        report = _score_predictions(
            name=name,
            records=evaluation_records,
            predictions=predictions,
            supported_intents=supported,
        )
        reports[name] = report
        if track:
            run_id = _track_report(
                settings,
                name=name,
                role="baseline",
                records=evaluation_records,
                report=report,
            )
            print(f"  local MLflow run: {run_id}")
        print(
            f"{name}: macro-F1={report.classification.macro_f1:.3f}, "
            f"schema={report.output_quality.json_schema_validity_rate:.3f}, "
            f"policy={report.output_quality.response_policy_compliance_rate:.3f}"
        )
    return reports


def _intent_categories(
    train: Sequence[EvaluationRecord],
) -> tuple[list[str], dict[str, str]]:
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    for record in train:
        categories[record.target.intent][record.target.category] += 1
    intents = sorted(categories)
    mapping = {
        intent: min(
            counts,
            key=lambda category: (-counts[category], category),
        )
        for intent, counts in categories.items()
    }
    return intents, mapping


def _few_shot_examples(
    train: Sequence[EvaluationRecord], count: int = 3
) -> list[tuple[str, dict[str, Any]]]:
    first_by_intent: dict[str, EvaluationRecord] = {}
    for record in train:
        first_by_intent.setdefault(record.target.intent, record)
    choices = [first_by_intent[key] for key in sorted(first_by_intent)]
    if len(choices) > count:
        positions = [
            round(index * (len(choices) - 1) / (count - 1)) for index in range(count)
        ]
        choices = [choices[position] for position in positions]
    return [
        (record.input_text, record.target.model_dump(mode="json")) for record in choices
    ]


def _model_predictions(
    predictor: LocalMLXPredictor,
    records: Sequence[EvaluationRecord],
    *,
    strategy: PromptStrategy,
    train: Sequence[EvaluationRecord],
    max_tokens: int,
) -> tuple[Prediction, ...]:
    intents, categories = _intent_categories(train)
    shots = _few_shot_examples(train) if strategy == "few_shot" else None
    predictions: list[Prediction] = []
    for index, record in enumerate(records, start=1):
        messages = build_messages(
            record.input_text,
            strategy=strategy,
            allowed_intents=intents,
            category_by_intent=categories,
            few_shot=shots,
        )
        generated = predictor.generate(messages, max_tokens=max_tokens)
        predictions.append(
            Prediction(
                example_id=record.example_id,
                raw_text=generated.text,
                latency_ms=generated.latency_ms,
                output_tokens=generated.output_tokens,
                peak_memory_mb=generated.peak_memory_mb,
            )
        )
        if index == len(records) or index % 10 == 0:
            print(f"  {strategy}: {index}/{len(records)}")
    return tuple(predictions)


def _balanced_subset(
    records: Sequence[EvaluationRecord], limit: int | None
) -> list[EvaluationRecord]:
    if limit is None or limit >= len(records):
        return list(records)
    if limit < 1:
        raise StudyCommandError("--limit must be positive")
    by_intent: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        by_intent[record.target.intent].append(record)
    selected: list[EvaluationRecord] = []
    round_number = 0
    while len(selected) < limit:
        added = False
        for intent in sorted(by_intent):
            values = by_intent[intent]
            if round_number < len(values):
                selected.append(values[round_number])
                added = True
                if len(selected) == limit:
                    return selected
        if not added:
            break
        round_number += 1
    return selected


def _run_local_probe(settings: ProjectSettings) -> None:
    train, _, _ = _load_splits(settings)
    intents, categories = _intent_categories(train)
    predictor = LocalMLXPredictor(settings.model_dir)
    messages = build_messages(
        "I forgot my password and need help signing in.",
        strategy="strong",
        allowed_intents=intents,
        category_by_intent=categories,
    )
    generated = predictor.generate(messages, max_tokens=96)
    if not generated.text.strip():
        raise StudyCommandError("the local model returned an empty readiness probe")
    print(
        "Local MLX inference passed: "
        f"{generated.latency_ms:.0f} ms, {generated.output_tokens} output tokens, "
        f"{generated.peak_memory_mb:.0f} MB process peak"
    )


def _check_notebook_runtime() -> None:
    required = {"jupyterlab": "4.6.2", "ipykernel": "7.3.0"}
    problems = []
    for package, expected in required.items():
        if importlib.util.find_spec(package) is None:
            problems.append(f"{package} is missing")
            continue
        actual = importlib.metadata.version(package)
        if actual != expected:
            problems.append(f"{package} is {actual}, expected {expected}")
    if problems:
        raise StudyCommandError(
            "notebook runtime is incomplete: " + "; ".join(problems)
        )
    print("Locked JupyterLab notebook runtime passed.")


def _generate_capstone() -> None:
    output_dir = PROJECT_ROOT / "data" / "processed" / "capstone-readiness-v1"
    manifest = generate_capstone_dataset(output_dir)
    counts = {
        artifact.split.value: artifact.record_count for artifact in manifest.artifacts
    }
    expected = {"train": 400, "validation": 100, "test": 150}
    if counts != expected or not manifest.frozen_test:
        raise StudyCommandError(
            "capstone generation violated its frozen split contract"
        )
    training_manifest = render_capstone_mlx_dataset(
        output_dir,
        PROJECT_ROOT / "data" / "processed" / "capstone-mlx-v1",
    )
    test_records = load_capstone_records(output_dir / "test.jsonl")
    policy_predictions = deterministic_capstone_predictions(test_records)
    policy_report = evaluate_capstone_predictions(test_records, policy_predictions)
    _write_capstone_evaluation(
        "deterministic-policy", test_records, policy_predictions, policy_report
    )
    hybrid_predictions = tuple(
        CapstonePrediction(
            example_id=record.example_id,
            raw_text=compact_review(
                build_hybrid_review(record.manifest).deterministic_review
            ).model_dump_json(),
            latency_ms=0.0,
            output_tokens=0,
            peak_memory_mb=0.0,
        )
        for record in test_records
    )
    hybrid_report = evaluate_capstone_predictions(test_records, hybrid_predictions)
    _write_capstone_evaluation(
        "hybrid-policy-text", test_records, hybrid_predictions, hybrid_report
    )
    if policy_report.aggregate.exact_review_rate != 1.0:
        raise StudyCommandError("deterministic capstone policy failed its own ceiling")
    print(
        "Generated policy-derived capstone: "
        f"train=400, validation=100, frozen-test=150; {manifest.dataset_sha256}"
    )
    print(
        "Rendered compact MLX capstone data: "
        f"{training_manifest.dataset_fingerprint}"
    )
    print("Policy and deterministic-hybrid frozen-test ceiling: exact=1.000")


def _write_capstone_evaluation(
    name: str,
    records: Sequence[CapstoneRecord],
    predictions: Sequence[CapstonePrediction],
    report: CapstoneEvaluationReport,
) -> tuple[Path, Path]:
    output = PROJECT_ROOT / "artifacts" / "capstone-evaluation"
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / f"{name}-predictions.jsonl"
    report_path = output / f"{name}-report.json"
    prediction_path.write_text(
        "".join(prediction.model_dump_json() + "\n" for prediction in predictions),
        encoding="utf-8",
    )
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return prediction_path, report_path


def _capstone_shots(
    train: Sequence[CapstoneRecord],
) -> list[tuple[str, str]]:
    selected: list[CapstoneRecord] = []
    for status in (
        ReadinessStatus.READY,
        ReadinessStatus.NOT_READY,
        ReadinessStatus.REVIEW_REQUIRED,
    ):
        match = next(
            (record for record in train if record.expected_output.status is status),
            None,
        )
        if match is not None:
            selected.append(match)
    return [
        (
            json.dumps(record.manifest, sort_keys=True, separators=(",", ":")),
            compact_expected(record).model_dump_json(),
        )
        for record in selected
    ]


def _capstone_messages(
    record: CapstoneRecord,
    *,
    strategy: str,
    shots: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    system = CAPSTONE_SYSTEM_PROMPT
    if strategy != "basic":
        rules = [
            {
                "name": rule.name,
                "kind": rule.kind.value,
                "failure_severity": rule.failure_severity.value,
                "remediation_id": rule.remediation_id,
            }
            for rule in rule_catalog()
        ]
        system += (
            " Valid statuses: ready, not_ready, review_required. Valid non-pass "
            "results: fail, review. Use only these rule definitions: "
            + json.dumps(rules, separators=(",", ":"), sort_keys=True)
        )
    messages = [{"role": "system", "content": system}]
    if strategy == "few-shot":
        for manifest, target in shots:
            messages.extend(
                [
                    {"role": "user", "content": manifest},
                    {"role": "assistant", "content": target},
                ]
            )
    messages.append(
        {
            "role": "user",
            "content": json.dumps(
                record.manifest,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    )
    return messages


def _capstone_model_predictions(
    predictor: LocalMLXPredictor,
    records: Sequence[CapstoneRecord],
    *,
    strategy: str,
    train: Sequence[CapstoneRecord],
    max_tokens: int,
    display_name: str | None = None,
) -> tuple[CapstonePrediction, ...]:
    shots = _capstone_shots(train)
    predictions: list[CapstonePrediction] = []
    for index, record in enumerate(records, start=1):
        generated = predictor.generate(
            _capstone_messages(record, strategy=strategy, shots=shots),
            max_tokens=max_tokens,
        )
        predictions.append(
            CapstonePrediction(
                example_id=record.example_id,
                raw_text=generated.text,
                latency_ms=generated.latency_ms,
                output_tokens=generated.output_tokens,
                peak_memory_mb=generated.peak_memory_mb,
            )
        )
        if index == len(records) or index % 10 == 0:
            print(f"  capstone {display_name or strategy}: {index}/{len(records)}")
    return tuple(predictions)


def _capstone_hybrid_predictions(
    predictor: LocalMLXPredictor,
    records: Sequence[CapstoneRecord],
) -> tuple[tuple[CapstonePrediction, ...], list[dict[str, Any]]]:
    cache: dict[tuple[str, ...], str] = {}
    evidence: list[dict[str, Any]] = []
    predictions: list[CapstonePrediction] = []
    for index, record in enumerate(records, start=1):
        generated_latency = 0.0
        generated_tokens = 0
        generated_peak = 0.0

        def renderer(check) -> str:
            nonlocal generated_latency, generated_tokens, generated_peak
            key = (
                check.name,
                check.result.value,
                check.severity.value,
                check.evidence,
                check.remediation_text or "",
            )
            if key not in cache:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the supplied deterministic check as one concise "
                            "sentence. Preserve every fact and never change the result."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "name": check.name,
                                "result": check.result.value,
                                "severity": check.severity.value,
                                "evidence": check.evidence,
                                "remediation": check.remediation_text,
                            },
                            separators=(",", ":"),
                        ),
                    },
                ]
                generated = predictor.generate(messages, max_tokens=80)
                cache[key] = generated.text.strip()
                generated_latency += generated.latency_ms
                generated_tokens += generated.output_tokens
                generated_peak = max(generated_peak, generated.peak_memory_mb)
            return cache[key]

        hybrid = build_hybrid_review(
            record.manifest,
            renderer=renderer,
            renderer_name="local-qwen-explanation",
        )
        evidence.append(
            {
                "example_id": record.example_id,
                "explanations": [
                    explanation.model_dump(mode="json")
                    for explanation in hybrid.explanations
                ],
            }
        )
        predictions.append(
            CapstonePrediction(
                example_id=record.example_id,
                raw_text=compact_review(hybrid.deterministic_review).model_dump_json(),
                latency_ms=generated_latency,
                output_tokens=generated_tokens,
                peak_memory_mb=generated_peak,
            )
        )
        if index == len(records) or index % 10 == 0:
            print(f"  capstone hybrid: {index}/{len(records)}")
    return tuple(predictions), evidence


def _cmd_prepare_flight(_args: argparse.Namespace, settings: ProjectSettings) -> None:
    from .acquisition import acquire_bitext, acquire_model

    print("Acquiring and verifying pinned public study assets...")
    acquire_bitext(settings)
    acquire_model(settings)
    _prepare_data(settings)
    _generate_capstone()
    _baseline_reports(settings)
    _check_notebook_runtime()

    enable_offline_environment()
    with deny_network():
        prove_socket_denial()
        _run_local_probe(settings)
        from .tracking import tracking_smoke

        run_id = tracking_smoke(settings)
        print(f"Local MLflow write passed: {run_id}")

    print("Running one real MLX-LM LoRA iteration...")
    evidence = run_lora(
        iterations=1,
        adapter_path=settings.preflight_adapter_dir,
        log_name="preflight-smoke",
    )
    if not (settings.preflight_adapter_dir / "adapters.safetensors").is_file():
        raise StudyCommandError("MLX-LM did not write the preflight adapter")
    print(
        f"MLX-LM training passed: return={evidence.return_code}, "
        f"peak={evidence.peak_memory_gb or 0.0:.2f} GB"
    )
    manifest_path = write_flight_manifest(settings)
    print(f"Flight manifest written: {manifest_path.relative_to(PROJECT_ROOT)}")
    print("Preparation complete. Run `make flight-check` before departure.")


def _cmd_flight_check(_args: argparse.Namespace, settings: ProjectSettings) -> None:
    machine = apple_silicon_status()
    if not machine.ready:
        raise StudyCommandError(f"MLX requires Apple silicon; found {machine.detail}")
    checks = require_assets(settings)
    verify_flight_manifest(settings)
    prove_socket_denial()
    leakage = check_split_files(settings.processed_dir)
    assert_no_leakage(leakage)
    _check_notebook_runtime()
    _generate_capstone()
    _run_local_probe(settings)
    from .tracking import tracking_smoke

    run_id = tracking_smoke(settings)
    verify_flight_manifest(settings)
    print(f"Verified {len(checks)} local asset groups and the locked manifest.")
    print(f"Local MLflow write passed: {run_id}")
    print("READY FOR OFFLINE STUDY")


def _cmd_smoke(_args: argparse.Namespace, settings: ProjectSettings) -> None:
    require_assets(settings)
    leakage = check_split_files(settings.processed_dir)
    assert_no_leakage(leakage)
    _baseline_reports(settings)
    _generate_capstone()
    print("Offline deterministic study smoke passed.")


def _cmd_prepare_data(_args: argparse.Namespace, settings: ProjectSettings) -> None:
    _prepare_data(settings)


def _cmd_baselines(args: argparse.Namespace, settings: ProjectSettings) -> None:
    _require_prepared_split_integrity(settings.processed_dir)
    _baseline_reports(settings, track=args.track)


def _cmd_train(args: argparse.Namespace, settings: ProjectSettings) -> None:
    _require_prepared_split_integrity(settings.processed_dir)
    require_assets(settings)
    evidence = run_lora(iterations=args.iterations)
    print(evidence.model_dump_json(indent=2))


def _cmd_evaluate(args: argparse.Namespace, settings: ProjectSettings) -> None:
    _require_prepared_split_integrity(settings.processed_dir)
    require_assets(settings)
    train, _, test = _load_splits(settings)
    records = _balanced_subset(test, args.limit)
    supported, _ = _intent_categories(train)
    deterministic = _baseline_reports(settings, records=records, track=args.track)

    requested = {"basic", "strong", "few-shot", "lora"}
    if args.methods != "all":
        requested = set(args.methods.split(","))
        unknown = requested.difference({"basic", "strong", "few-shot", "lora"})
        if unknown:
            raise StudyCommandError("unknown evaluation methods: " + ", ".join(unknown))

    reports: dict[str, EvaluationReport] = dict(deterministic)
    base_methods = [
        name for name in ("basic", "strong", "few-shot") if name in requested
    ]
    if base_methods:
        predictor = LocalMLXPredictor(settings.model_dir)
        for name in base_methods:
            strategy: PromptStrategy = "few_shot" if name == "few-shot" else name  # type: ignore[assignment]
            predictions = _model_predictions(
                predictor,
                records,
                strategy=strategy,
                train=train,
                max_tokens=args.max_tokens,
            )
            report = _score_predictions(
                name=name,
                records=records,
                predictions=predictions,
                supported_intents=supported,
            )
            reports[name] = report
            if args.track:
                run_id = _track_report(
                    settings,
                    name=name,
                    role="baseline",
                    records=records,
                    report=report,
                )
                print(f"  local MLflow run: {run_id}")
            print(
                f"{name}: macro-F1={report.classification.macro_f1:.3f}; "
                f"schema={report.output_quality.json_schema_validity_rate:.3f}"
            )

    lora_report: EvaluationReport | None = None
    if "lora" in requested:
        adapter_weights = settings.adapter_dir / "adapters.safetensors"
        if not adapter_weights.is_file():
            print("LoRA adapter is absent; run `make train` before LoRA evaluation.")
        else:
            predictor = LocalMLXPredictor(
                settings.model_dir,
                adapter_path=settings.adapter_dir,
            )
            predictions = _model_predictions(
                predictor,
                records,
                strategy="strong",
                train=train,
                max_tokens=args.max_tokens,
            )
            lora_report = _score_predictions(
                name="lora-change",
                records=records,
                predictions=predictions,
                supported_intents=supported,
            )
            reports["lora-change"] = lora_report
            if args.track:
                run_id = _track_report(
                    settings,
                    name="lora-change",
                    role="change",
                    records=records,
                    report=lora_report,
                )
                print(f"  local MLflow run: {run_id}")
            print(format_error_analysis(lora_report))

    complete_promotion_evidence = (
        lora_report is not None
        and args.limit is None
        and requested == {"basic", "strong", "few-shot", "lora"}
    )
    if complete_promotion_evidence and lora_report is not None:
        baselines = [
            BaselineEvaluation(
                name=name,
                report=report,
                meaningful=name != "majority",
            )
            for name, report in reports.items()
            if name != "lora-change"
        ]
        assessment = decide_lora_promotion(
            change_name="bitext-lora-v1",
            change_report=lora_report,
            baselines=baselines,
        )
        path = PROJECT_ROOT / "artifacts" / "evaluation" / "promotion.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(assessment.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"Decision: {assessment.decision.value}")
    elif lora_report is not None:
        print(
            "Decision: inconclusive (partial or debug evaluation is report-only; "
            "run the full frozen set with all methods for promotion evidence)"
        )


def _cmd_capstone(_args: argparse.Namespace, _settings: ProjectSettings) -> None:
    _generate_capstone()


def _cmd_capstone_train(args: argparse.Namespace, settings: ProjectSettings) -> None:
    require_assets(settings)
    _generate_capstone()
    evidence = run_lora(
        iterations=args.iterations,
        config_path=PROJECT_ROOT / "configs" / "training" / "capstone-lora.yaml",
        log_name="capstone-latest",
    )
    print(evidence.model_dump_json(indent=2))


def _cmd_capstone_evaluate(args: argparse.Namespace, settings: ProjectSettings) -> None:
    require_assets(settings)
    _generate_capstone()
    source = PROJECT_ROOT / "data" / "processed" / "capstone-readiness-v1"
    train = load_capstone_records(source / "train.jsonl")
    all_test = load_capstone_records(source / "test.jsonl")
    if args.limit is not None and args.limit < 1:
        raise StudyCommandError("--limit must be positive")
    records = all_test if args.limit is None else all_test[: args.limit]
    allowed = {"policy", "basic", "strong", "few-shot", "lora", "hybrid"}
    requested = allowed if args.methods == "all" else set(args.methods.split(","))
    unknown = requested - allowed
    if unknown:
        raise StudyCommandError(
            "unknown capstone methods: " + ", ".join(sorted(unknown))
        )

    reports: dict[str, CapstoneEvaluationReport] = {}

    def score(
        name: str,
        predictions: Sequence[CapstonePrediction],
    ) -> CapstoneEvaluationReport:
        report = evaluate_capstone_predictions(records, predictions)
        _write_capstone_evaluation(name, records, predictions, report)
        reports[name] = report
        print(
            f"{name}: exact={report.aggregate.exact_review_rate:.3f}, "
            f"status={report.aggregate.status_accuracy:.3f}, "
            f"checks={report.aggregate.check_result_accuracy:.3f}, "
            f"schema={report.aggregate.schema_validity_rate:.3f}"
        )
        return report

    if "policy" in requested:
        score("deterministic-policy", deterministic_capstone_predictions(records))

    needs_base = bool(requested & {"basic", "strong", "few-shot", "hybrid"})
    base_predictor = LocalMLXPredictor(settings.model_dir) if needs_base else None
    for method in ("basic", "strong", "few-shot"):
        if method in requested and base_predictor is not None:
            score(
                method,
                _capstone_model_predictions(
                    base_predictor,
                    records,
                    strategy=method,
                    train=train,
                    max_tokens=args.max_tokens,
                ),
            )

    if "hybrid" in requested and base_predictor is not None:
        predictions, explanation_evidence = _capstone_hybrid_predictions(
            base_predictor,
            records,
        )
        score("hybrid", predictions)
        evidence_path = (
            PROJECT_ROOT
            / "artifacts"
            / "capstone-evaluation"
            / "hybrid-explanations.json"
        )
        evidence_path.write_text(
            json.dumps(explanation_evidence, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "Hybrid decisions remain deterministic; generated explanations are "
            "report-only evidence for human review."
        )

    lora_report: CapstoneEvaluationReport | None = None
    if "lora" in requested:
        adapter = settings.capstone_adapter_dir / "adapters.safetensors"
        if not adapter.is_file():
            print("Capstone LoRA adapter is absent; run `make capstone-train` first.")
        else:
            predictor = LocalMLXPredictor(
                settings.model_dir,
                adapter_path=adapter.parent,
            )
            lora_report = score(
                "capstone-lora-change",
                _capstone_model_predictions(
                    predictor,
                    records,
                    strategy="basic",
                    train=train,
                    max_tokens=args.max_tokens,
                    display_name="lora",
                ),
            )

    decision_path = PROJECT_ROOT / "artifacts" / "capstone-evaluation" / "decision.json"
    required_baselines = {"basic", "strong", "few-shot"}
    complete = (
        lora_report is not None
        and args.limit is None
        and required_baselines <= requested
        and len(records) == 150
    )
    if complete and lora_report is not None:
        strongest_name = max(
            required_baselines,
            key=lambda name: (
                reports[name].aggregate.exact_review_rate,
                reports[name].aggregate.check_result_accuracy,
                name,
            ),
        )
        strongest = reports[strongest_name]
        change_score = lora_report.aggregate.exact_review_rate
        baseline_score = strongest.aggregate.exact_review_rate
        passed_gates = (
            lora_report.aggregate.schema_validity_rate >= 0.98
            and lora_report.aggregate.extra_check_rate == 0.0
            and lora_report.aggregate.status_accuracy
            >= strongest.aggregate.status_accuracy
        )
        decision = (
            "adopt" if change_score > baseline_score and passed_gates else "reject"
        )
        reason = (
            "the compact LoRA change beat the strongest untouched-model baseline "
            "and passed the absolute gates"
            if decision == "adopt"
            else "the compact LoRA change did not beat the strongest complete "
            "untouched-model evidence and every absolute gate"
        )
        payload: dict[str, Any] = {
            "baseline": {
                "name": strongest_name,
                "exact_review_rate": baseline_score,
            },
            "change": {
                "name": "capstone-lora-v1",
                "exact_review_rate": change_score,
            },
            "result": {"passed_absolute_gates": passed_gates, "reason": reason},
            "decision": decision,
        }
    else:
        payload = {
            "baseline": None,
            "change": {"name": "capstone-lora-v1"},
            "result": {
                "reason": (
                    "full frozen LoRA and basic/strong/few-shot evidence is required"
                )
            },
            "decision": "inconclusive",
        }
    payload["architecture_boundary"] = (
        "Deterministic checks remain authoritative; a model may only supply "
        "bounded normalization or explanation wording."
    )
    decision_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Capstone model-change decision: {payload['decision']}")


def _cmd_clean_study(_args: argparse.Namespace, settings: ProjectSettings) -> None:
    targets = (
        settings.processed_dir,
        PROJECT_ROOT / "data" / "processed" / "capstone-readiness-v1",
        PROJECT_ROOT / "data" / "processed" / "capstone-mlx-v1",
        PROJECT_ROOT / "artifacts",
        PROJECT_ROOT / ".aai",
    )
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            print(f"Removed generated output: {target.relative_to(PROJECT_ROOT)}")
    print("Raw downloads, the local model, and the locked environment were retained.")


Command = Callable[[argparse.Namespace, ProjectSettings], None]


def _parser() -> tuple[argparse.ArgumentParser, dict[str, Command]]:
    parser = argparse.ArgumentParser(
        prog="aai-finetune",
        description="Offline-first local fine-tuning curriculum",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="set library offline controls and deny Python socket connections",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    handlers: dict[str, Command] = {}

    def command(name: str, help_text: str, handler: Command) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name, help=help_text)
        handlers[name] = handler
        return child

    command(
        "prepare-flight", "acquire and verify all travel assets", _cmd_prepare_flight
    )
    command(
        "flight-check", "prove the prepared project works offline", _cmd_flight_check
    )
    command("smoke", "run fast deterministic offline checks", _cmd_smoke)
    command("prepare-data", "rebuild immutable portable splits", _cmd_prepare_data)
    baseline_parser = command(
        "baselines", "evaluate deterministic baselines", _cmd_baselines
    )
    baseline_parser.add_argument("--track", action="store_true")

    train_parser = command("train", "run the configured MLX-LM LoRA change", _cmd_train)
    train_parser.add_argument("--iterations", type=int)

    evaluate_parser = command(
        "evaluate", "score frozen local model predictions", _cmd_evaluate
    )
    evaluate_parser.add_argument("--track", action="store_true")
    evaluate_parser.add_argument(
        "--methods",
        default="all",
        help="all or a comma-separated subset of basic,strong,few-shot,lora",
    )
    evaluate_parser.add_argument(
        "--limit",
        type=int,
        help="balanced debug subset; omit for the complete frozen test set",
    )
    evaluate_parser.add_argument("--max-tokens", type=int, default=160)

    command("capstone", "generate the policy-derived capstone", _cmd_capstone)
    capstone_train_parser = command(
        "capstone-train",
        "train the compact capstone LoRA change",
        _cmd_capstone_train,
    )
    capstone_train_parser.add_argument("--iterations", type=int)
    capstone_evaluate_parser = command(
        "capstone-evaluate",
        "compare policy, prompt, LoRA, and hybrid capstone methods",
        _cmd_capstone_evaluate,
    )
    capstone_evaluate_parser.add_argument(
        "--methods",
        default="all",
        help="all or policy,basic,strong,few-shot,lora,hybrid",
    )
    capstone_evaluate_parser.add_argument("--limit", type=int)
    capstone_evaluate_parser.add_argument("--max-tokens", type=int, default=512)
    command(
        "clean-study", "remove generated outputs but retain assets", _cmd_clean_study
    )
    return parser, handlers


def run(argv: Sequence[str] | None = None) -> int:
    """Parse and execute one command, returning a process status."""

    parser, handlers = _parser()
    args = parser.parse_args(argv)
    settings = load_settings()
    strict_offline = args.offline or args.command in {"flight-check", "smoke"}
    if strict_offline:
        enable_offline_environment()
    context = deny_network() if strict_offline else nullcontext()
    try:
        with context:
            handlers[args.command](args, settings)
    except (OfflineAssetError, StudyCommandError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


def main() -> None:
    """Console-script entry point."""

    raise SystemExit(run())


if __name__ == "__main__":
    main()
