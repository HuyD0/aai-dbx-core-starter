# ruff: noqa: E501
"""Render the tracked narrative curriculum notebooks deterministically."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


@dataclass(frozen=True)
class Cell:
    kind: str
    source: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Notebook:
    order: int
    slug: str
    title: str
    stage: str
    duration: int
    prerequisites: tuple[str, ...]
    evidence: str
    cells: tuple[Cell, ...]

    @property
    def filename(self) -> str:
        return f"{self.order:02d}_{self.slug}.ipynb"


def md(source: str, *tags: str) -> Cell:
    return Cell("markdown", dedent(source).strip() + "\n", tuple(tags))


def code(source: str, *tags: str) -> Cell:
    return Cell("code", dedent(source).strip() + "\n", tuple(tags))


def intro(
    number: int,
    title: str,
    *,
    minutes: int,
    prerequisites: str,
    objectives: tuple[str, ...],
    evidence: str,
) -> Cell:
    objective_lines = "\n".join(f"- {item}" for item in objectives)
    return md(f"""
        # {number:02d} — {title}

        **Estimated time:** {minutes} minutes<br>
        **Prerequisites:** {prerequisites}<br>
        **Learner-produced evidence:** {evidence}

        ## Learning objectives

        {objective_lines}

        This notebook is a teaching interface over the reusable code in `src/`.
        It uses only prepared local files. Run `make prepare-flight` before the
        trip; no cell installs packages or downloads data.
        """)


OFFLINE_SETUP = code(
    """
    from aai_local_finetuning.offline import enable_offline_environment

    enable_offline_environment()
    """,
    "offline-setup",
)


NOTEBOOKS: tuple[Notebook, ...] = (
    Notebook(
        order=0,
        slug="start_here",
        title="Start here: question, contract, and offline workspace",
        stage="orientation",
        duration=20,
        prerequisites=(),
        evidence="a ready/not-ready asset table and a written learning goal",
        cells=(
            intro(
                0,
                "Start here: question, contract, and offline workspace",
                minutes=20,
                prerequisites="none",
                objectives=(
                    "Understand the experiment question and structured-output task.",
                    "Verify that the local study assets are present without using a network.",
                    "Distinguish a learning result from a production-readiness claim.",
                ),
                evidence="a ready/not-ready asset table and a written learning goal",
            ),
            md("""
                ## The experiment question

                Can a small LoRA adapter improve structured customer-support intent
                prediction over the strongest meaningful baseline while preserving
                strict JSON, known labels, and safe response wording?

                The evidence lifecycle is **baseline → change → result → decision**.
                Training success alone is not a decision.
                """),
            OFFLINE_SETUP,
            md(
                """
                ## Check this prepared machine

                The paths below stay local. A red row means preparation is incomplete;
                it does not trigger a download. The model, source archive, generated
                data, adapters, and MLflow database are deliberately ignored by Git.
                """,
                "what-to-notice",
            ),
            code("""
                from aai_local_finetuning.offline import (
                    apple_silicon_status,
                    asset_checks,
                    deny_network,
                    prove_socket_denial,
                )
                from aai_local_finetuning.settings import load_settings

                settings = load_settings()
                machine = apple_silicon_status()
                readiness = [
                    {
                        "asset": check.name,
                        "ready": check.ready,
                        "detail": check.detail,
                    }
                    for check in [machine, *asset_checks(settings)]
                ]
                readiness
                """),
            md(
                """
                ## Prove the Python guard

                This scoped check deliberately denies Python socket connections. The
                notebook also enables the supported offline flags before importing
                model or tracking libraries. Turning Wi-Fi off once before departure
                remains the strongest rehearsal because native libraries are outside
                Python's complete control.
                """,
                "what-to-notice",
            ),
            code("""
                with deny_network():
                    prove_socket_denial()
                "Python socket guard passed"
                """),
            md(
                """
                ## The output boundary

                The model must return exactly four typed fields. A plausible-looking
                sentence is not enough, and a high intent score cannot conceal invalid
                or unsafe generated output.
                """,
                "what-to-notice",
            ),
            code("""
                from aai_local_finetuning.evaluation import SupportOutput

                SupportOutput.model_json_schema()
                """),
            md(
                """
                ## Exercise — write your evidence question

                Replace the default sentence with the question you want the final
                decision to answer. Success means it names a baseline, a change, a
                frozen evaluation boundary, and at least one output-quality gate.
                """,
                "exercise",
            ),
            code(
                """
                my_evidence_question = (
                    "Does the LoRA change beat the strongest baseline on frozen macro-F1 "
                    "while meeting strict schema and response-policy gates?"
                )
                assert all(
                    term in my_evidence_question.lower()
                    for term in ("lora", "baseline", "frozen", "schema")
                )
                my_evidence_question
                """,
                "exercise",
            ),
            md(
                """
                **Hint:** describe the comparison and its evidence, not the result you
                hope to see. Keep production suitability outside this learning claim.
                """,
                "hint",
            ),
            md(
                """
                ## Checkpoint

                You can now explain why offline readiness, strict output validation,
                and a frozen comparison are three different concerns.

                **Next:** `01_dataset_provenance_and_license.ipynb` examines whether the
                source may be used for this curriculum and what remains unproven.
                """,
                "checkpoint",
            ),
        ),
    ),
    Notebook(
        order=1,
        slug="dataset_provenance_and_license",
        title="Dataset provenance, licensing, and immutable inputs",
        stage="provenance",
        duration=30,
        prerequisites=("00_start_here.ipynb",),
        evidence="a source-contract review and verified local SHA-256 values",
        cells=(
            intro(
                1,
                "Dataset provenance, licensing, and immutable inputs",
                minutes=30,
                prerequisites="00 — Start here",
                objectives=(
                    "Read the verified dataset identity and licensing record.",
                    "Explain why public access is not commercial or production clearance.",
                    "Verify immutable local inputs without printing source records.",
                ),
                evidence="a source-contract review and verified local SHA-256 values",
            ),
            OFFLINE_SETUP,
            md(
                """
                ## Read the source contract before the rows

                Provenance comes before modelling. The tracked card records title,
                owner, current source URL, version, license, intended curriculum use,
                redistribution constraints, source composition, quality findings, and
                limitations. Settings contain public identifiers and hashes—not secrets.
                """,
                "what-to-notice",
            ),
            code("""
                from aai_local_finetuning.settings import (
                    PROJECT_ROOT,
                    load_settings,
                    sha256_file,
                )

                settings = load_settings()
                card_path = PROJECT_ROOT / "dataset_cards" / "bitext-customer-support.md"
                source_contract = {
                    "title": settings.dataset.title,
                    "owner": settings.dataset.owner,
                    "source": settings.dataset.url,
                    "kaggle_version": settings.dataset.version,
                    "license_recorded": settings.dataset.license,
                    "accessed_on": settings.dataset.accessed_on,
                    "language": settings.dataset.language,
                    "dataset_card": str(card_path.relative_to(PROJECT_ROOT)),
                }
                source_contract
                """),
            md(
                """
                ## Verify bytes, not filenames

                A filename can be silently replaced. SHA-256 binds later results to the
                exact archive and CSV inspected for the course. Hashing is read-only and
                streams the files, so the raw directory remains immutable.
                """,
                "what-to-notice",
            ),
            code("""
                local_integrity = {
                    "archive_matches": (
                        sha256_file(settings.archive_path)
                        == settings.dataset.archive_sha256
                    ),
                    "csv_matches": (
                        sha256_file(settings.csv_path) == settings.dataset.csv_sha256
                    ),
                    "archive_sha256": settings.dataset.archive_sha256,
                    "csv_sha256": settings.dataset.csv_sha256,
                }
                local_integrity
                """),
            md("""
                ## Inspect schema without exposing content

                We need confirmed columns and file format, but not a screenful of raw
                customer-like text. Reading only the header keeps this check bounded.
                """),
            code("""
                import csv

                with settings.csv_path.open(encoding="utf-8", newline="") as stream:
                    columns = next(csv.reader(stream))
                {"format": "CSV", "columns": columns}
                """),
            md(
                """
                ## Exercise — make a scoped-use decision

                Fill in the rationale. The safe default is deliberately narrow: local
                learning is accepted under the recorded conditions; commercial,
                enterprise, and production use remain unassessed.
                """,
                "exercise",
            ),
            code(
                """
                use_review = {
                    "local_curriculum": "allowed_with_recorded_conditions",
                    "commercial_use": "not_assessed",
                    "production_suitability": "not_assessed",
                    "redistribution": "retain_license_and_attribution_obligations",
                    "rationale": (
                        "Public availability and a recorded data license do not prove "
                        "fitness, consent, accuracy, or policy acceptance for production."
                    ),
                }
                assert use_review["production_suitability"] == "not_assessed"
                use_review
                """,
                "exercise",
            ),
            md(
                """
                **Hint:** separate permission to use data from evidence that the data is
                accurate, representative, safe, and approved for a business purpose.
                """,
                "hint",
            ),
            md(
                """
                ## Checkpoint

                You have a reproducible source identity and a scoped use decision. A
                future source version requires a fresh card and fresh hashes.

                **Next:** `02_dataset_exploration_and_validation.ipynb` measures the
                current source instead of trusting its description.
                """,
                "checkpoint",
            ),
        ),
    ),
    Notebook(
        order=2,
        slug="dataset_exploration_and_validation",
        title="Dataset exploration and validation",
        stage="data_audit",
        duration=45,
        prerequisites=("01_dataset_provenance_and_license.ipynb",),
        evidence="a privacy-preserving quality report and dataset-card draft",
        cells=(
            intro(
                2,
                "Dataset exploration and validation",
                minutes=45,
                prerequisites="01 — Dataset provenance and license",
                objectives=(
                    "Measure missing, duplicate, label, length, and sensitive-pattern findings.",
                    "Use the exact local tokenizer for token-length analysis.",
                    "Interpret aggregate evidence without displaying unnecessary raw text.",
                ),
                evidence="a privacy-preserving quality report and dataset-card draft",
            ),
            OFFLINE_SETUP,
            md("""
                ## Load the inspected local source

                Reusable audit logic lives under `src/aai_local_finetuning/data`.
                The notebook asks questions of the resulting evidence rather than
                reimplementing the pipeline in cells.
                """),
            code("""
                import json

                from aai_local_finetuning.data import (
                    audit_dataset,
                    check_split_files,
                    summarize_instruction_tokens,
                )
                from aai_local_finetuning.settings import load_settings

                settings = load_settings()
                raw_csv = settings.csv_path
                processed_dir = settings.processed_dir
                if not raw_csv.is_file():
                    raise FileNotFoundError(
                        "The immutable Bitext CSV is missing. Prepare this machine online."
                    )
                """),
            md(
                """
                ## Quality audit

                The report contains counts and distributions, not raw samples. Exact
                duplicates are measured after canonicalization. Near duplicates and
                inferred templates are heuristic evidence and must be documented as such.
                """,
                "what-to-notice",
            ),
            code("""
                audit = audit_dataset(raw_csv)
                audit_payload = audit.model_dump(mode="json")
                core_quality = {
                    key: audit_payload[key]
                    for key in (
                        "source_records",
                        "valid_records",
                        "unique_records",
                        "curated_records",
                        "invalid_record_count",
                        "missing_by_field",
                        "exact_duplicate_count",
                        "exact_duplicate_rate",
                        "near_duplicate_pairs",
                        "near_duplicate_clusters",
                        "conflicting_group_count",
                        "excluded_conflicting_records",
                    )
                }
                core_quality
                """),
            md(
                """
                ## Labels, lengths, language, and sensitive-looking patterns

                Pattern matches are counts only. Email-, URL-, phone-like text and
                placeholders are masked before portable training records are written.
                Source flags remain explicit evaluation slices; difficulty is a separate
                versioned heuristic, not a human quality label.
                """,
                "what-to-notice",
            ),
            code("""
                token_lengths = summarize_instruction_tokens(raw_csv, settings.model_dir)
                distribution_report = {
                    "intents": audit_payload["intent_distribution"],
                    "categories": audit_payload["category_distribution"],
                    "languages": {settings.dataset.language: audit_payload["source_records"]},
                    "instruction_characters": audit_payload["instruction_characters"],
                    "instruction_word_proxy": audit_payload["instruction_words"],
                    "pinned_tokenizer_tokens": token_lengths.model_dump(mode="json"),
                    "source_flags": audit_payload["flag_distribution"],
                    "difficulty": audit_payload["difficulty_distribution"],
                    "sensitive_pattern_counts": audit_payload[
                        "sensitive_pattern_counts"
                    ],
                }
                distribution_report
                """),
            md("""
                ## Automated split-integrity gate

                This gate examines the prepared evidence boundaries without displaying
                frozen test content. It checks exact, inferred-template, and near-duplicate
                overlap plus target and demonstration leakage.
                """),
            code("""
                integrity = check_split_files(processed_dir)
                integrity.model_dump(mode="json")
                """),
            md("""
                ## Dataset-card draft inputs

                The tracked card remains reviewed prose. This draft makes measurements
                easy to revisit whenever source bytes, processing, or split policy change.
                """),
            code("""
                manifest = json.loads(
                    (processed_dir / "manifest.json").read_text(encoding="utf-8")
                )
                dataset_card_draft = {
                    "source": manifest["dataset"],
                    "fingerprint": manifest["dataset_fingerprint"],
                    "quality": core_quality,
                    "label_count": len(audit_payload["intent_distribution"]),
                    "sensitive_review": audit_payload["sensitive_pattern_counts"],
                    "split_strategy": manifest["split_strategy"],
                    "review_required": [
                        "near-duplicate threshold",
                        "sensitive-looking content",
                        "generated response policy",
                        "redistribution obligations",
                    ],
                }
                dataset_card_draft
                """),
            md(
                """
                ## Exercise — identify the highest-risk assumption

                Pick one measured finding and state what could go wrong if it were
                ignored. Success means your answer connects a finding to either leakage,
                fairness across labels, privacy, or misleading evaluation.
                """,
                "exercise",
            ),
            code(
                """
                finding = "near-duplicate clusters"
                risk = (
                    "Related templates crossing splits could make memorization look like "
                    "generalization, so groups must stay inside one evidence boundary."
                )
                assert finding and len(risk.split()) >= 10
                {"finding": finding, "risk": risk}
                """,
                "exercise",
            ),
            md(
                """
                **Hint:** counts are not conclusions. Ask how each observation could bias
                the final comparison or expose content unnecessarily.
                """,
                "hint",
            ),
            md(
                """
                ## Checkpoint

                The source has now been measured, not merely described. Keep the frozen
                test content out of prompt design and model development.

                **Next:** `03_leakage_safe_splits.ipynb` follows records from immutable
                source to portable train, validation, and test boundaries.
                """,
                "checkpoint",
            ),
        ),
    ),
    Notebook(
        order=3,
        slug="leakage_safe_splits",
        title="Portable records and leakage-safe splits",
        stage="data_preparation",
        duration=40,
        prerequisites=("02_dataset_exploration_and_validation.ipynb",),
        evidence="split counts, stable IDs, hashes, and a zero-leakage report",
        cells=(
            intro(
                3,
                "Portable records and leakage-safe splits",
                minutes=40,
                prerequisites="02 — Dataset exploration and validation",
                objectives=(
                    "Trace immutable source rows into framework-neutral chat records.",
                    "Explain group-aware balancing and stable record identifiers.",
                    "Verify split integrity without inspecting frozen-test examples.",
                ),
                evidence="split counts, stable IDs, hashes, and a zero-leakage report",
            ),
            OFFLINE_SETUP,
            md(
                """
                ## Evidence boundaries

                Train fits weights and train-derived rules. Validation selects prompts
                and settings. The frozen test is opened only after those choices stop.
                This notebook reads its manifest and runs automated leakage checks, but
                does not display or use test examples.
                """,
                "what-to-notice",
            ),
            code("""
                import json

                from aai_local_finetuning.data import check_split_files, text_similarity
                from aai_local_finetuning.evaluation import load_records_jsonl
                from aai_local_finetuning.settings import load_settings

                settings = load_settings()
                processed = settings.processed_dir
                manifest = json.loads(
                    (processed / "manifest.json").read_text(encoding="utf-8")
                )
                train = tuple(load_records_jsonl(processed / "train.jsonl"))
                validation = tuple(load_records_jsonl(processed / "valid.jsonl"))
                split_contract = {
                    name: {
                        "records": descriptor["record_count"],
                        "frozen": descriptor["frozen"],
                        "sha256": descriptor["sha256"],
                    }
                    for name, descriptor in manifest["splits"].items()
                }
                split_contract
                """),
            md(
                """
                ## What one portable record contains

                Framework-neutral JSONL preserves identity, source version, messages,
                labels, grouping evidence, flags, and difficulty. Source responses are
                not copied as training targets; a versioned response policy renders a
                short target independently. The preview is already masked and bounded.
                """,
                "what-to-notice",
            ),
            code("""
                example = train[0]
                safe_preview = {
                    "example_id": example.example_id,
                    "input_preview": example.input_text[:160],
                    "target": {
                        "intent": example.target.intent,
                        "category": example.target.category,
                        "requires_escalation": example.target.requires_escalation,
                        "response_preview": example.target.response[:100],
                    },
                    "flags": example.flags,
                    "difficulty": example.difficulty,
                    "split_group": example.metadata.get("split_group"),
                }
                safe_preview
                """),
            md("""
                ## Balance and separation

                The source has no reliable conversation, account, document, template,
                or timestamp identifier. The pipeline therefore keeps inferred exact,
                template, and near-duplicate groups together, excludes label-conflicting
                groups, and records this limitation instead of inventing source fields.
                """),
            code("""
                train_intents = {}
                validation_intents = {}
                for record in train:
                    train_intents[record.target.intent] = (
                        train_intents.get(record.target.intent, 0) + 1
                    )
                for record in validation:
                    validation_intents[record.target.intent] = (
                        validation_intents.get(record.target.intent, 0) + 1
                    )
                balance = {
                    "train_unique_counts": sorted(set(train_intents.values())),
                    "validation_unique_counts": sorted(
                        set(validation_intents.values())
                    ),
                    "train_validation_id_overlap": len(
                        {item.example_id for item in train}
                        & {item.example_id for item in validation}
                    ),
                    "dataset_fingerprint": manifest["dataset_fingerprint"],
                }
                balance
                """),
            md(
                """
                ## Run the automated leakage gate

                A passing gate means no configured relationship crossed the prepared
                boundaries. It does not prove that heuristic grouping discovered every
                semantic relationship in the world.
                """,
                "what-to-notice",
            ),
            code("""
                leakage = check_split_files(processed)
                {
                    "passed": leakage.passed,
                    "finding_count": len(leakage.findings),
                    "findings_by_kind": leakage.counts,
                }
                """),
            md(
                """
                ## Exercise — reason about near duplicates

                Change the second invented sentence and observe the similarity. Decide
                whether a numeric threshold is sufficient evidence of common origin.
                Success means you record both a decision and a limitation.
                """,
                "exercise",
            ),
            code(
                """
                sentence_a = "I forgot my password and cannot sign in"
                sentence_b = "I cannot sign in because I forgot the password"
                similarity = text_similarity(sentence_a, sentence_b)
                threshold = 0.88
                grouping_decision = similarity >= threshold
                limitation = (
                    "Similarity is a reproducible heuristic; it is not a verified "
                    "conversation or template identifier from the source."
                )
                {
                    "similarity": round(similarity, 3),
                    "same_group_at_threshold": grouping_decision,
                    "limitation": limitation,
                }
                """,
                "exercise",
            ),
            md(
                """
                **Hint:** a threshold makes behavior repeatable, not automatically true.
                Group conservatively and preserve the documented uncertainty.
                """,
                "hint",
            ),
            md(
                """
                ## Checkpoint

                You can identify what each split is allowed to influence and explain
                why a frozen flag, file hash, stable IDs, and leakage test work together.

                **Next:** `04_deterministic_baselines.ipynb` establishes a sanity floor
                and a transparent meaningful baseline using train and validation only.
                """,
                "checkpoint",
            ),
        ),
    ),
    Notebook(
        order=4,
        slug="deterministic_baselines",
        title="Deterministic baselines and the evaluation harness",
        stage="baselines",
        duration=45,
        prerequisites=("03_leakage_safe_splits.ipynb",),
        evidence="validation metrics for majority and train-derived keyword baselines",
        cells=(
            intro(
                4,
                "Deterministic baselines and the evaluation harness",
                minutes=45,
                prerequisites="03 — Portable records and leakage-safe splits",
                objectives=(
                    "Contrast a sanity-floor majority baseline with a meaningful rule baseline.",
                    "Interpret macro and weighted classification metrics.",
                    "Score structured output, response policy, performance, slices, and errors separately.",
                ),
                evidence="validation metrics for majority and train-derived keyword baselines",
            ),
            OFFLINE_SETUP,
            md(
                """
                ## Fit only on training evidence

                Both baselines learn from train. Validation estimates how well a method
                generalizes while prompts and settings may still change. The frozen test
                remains unopened.
                """,
                "what-to-notice",
            ),
            code("""
                import pandas as pd

                from aai_local_finetuning.evaluation import (
                    KeywordRuleBaseline,
                    MajorityBaseline,
                    evaluate_predictions,
                    format_error_analysis,
                )
                from aai_local_finetuning.learning import (
                    load_support_splits,
                    report_row,
                    support_contract,
                )

                splits = load_support_splits(include_test=False)
                allowed_intents, _ = support_contract(splits.train)
                majority = MajorityBaseline.fit(splits.train)
                keyword = KeywordRuleBaseline.fit(splits.train)
                """),
            md("""
                ## Score through one strict harness

                The same evaluator parses JSON, validates the schema, rejects unsupported
                labels, checks response policy, calculates classification metrics, and
                summarizes latency, output tokens, memory, slices, and bounded errors.
                """),
            code("""
                baseline_reports = {
                    "majority": evaluate_predictions(
                        splits.validation,
                        majority.predict_many(splits.validation),
                        supported_intents=allowed_intents,
                    ),
                    "keyword-rule": evaluate_predictions(
                        splits.validation,
                        keyword.predict_many(splits.validation),
                        supported_intents=allowed_intents,
                    ),
                }
                pd.DataFrame(
                    [report_row(name, report) for name, report in baseline_reports.items()]
                )
                """),
            md(
                """
                ## Why macro F1 matters

                Accuracy can be dominated by frequent labels. Macro F1 gives each intent
                equal weight; weighted F1 reflects observed support. Report both. The
                majority method is a sanity floor and is not considered meaningful for
                promotion, even when its JSON happens to be valid.
                """,
                "what-to-notice",
            ),
            code("""
                pd.DataFrame(
                    {
                        "intent": list(
                            baseline_reports["keyword-rule"].classification.per_intent_f1
                        ),
                        "f1": list(
                            baseline_reports[
                                "keyword-rule"
                            ].classification.per_intent_f1.values()
                        ),
                    }
                ).sort_values("f1").head(10)
                """),
            md(
                """
                ## Inspect what the transparent baseline learned

                Terms come only from training records. They are useful for debugging and
                also reveal brittleness: lexical shortcuts can fail on paraphrases,
                ambiguity, negation, or intents with overlapping vocabulary.
                """,
                "what-to-notice",
            ),
            code("""
                {
                    intent: keyword.keywords_by_intent[intent][:6]
                    for intent in list(allowed_intents)[:8]
                }
                """),
            md("""
                ## Bounded error evidence

                Error kinds remain separate: invalid JSON, schema mismatch, unsupported
                intent, intent/category/escalation errors, and response-policy failures.
                Only bounded masked previews are retained.
                """),
            code("""
                print(format_error_analysis(baseline_reports["keyword-rule"]))
                """),
            md(
                """
                ## Exercise — explain the meaningful baseline

                Choose one learned keyword group and predict a likely failure mode.
                Success means your explanation names the shortcut and a validation slice
                or example type that could expose it.
                """,
                "exercise",
            ),
            code(
                """
                inspected_intent = allowed_intents[0]
                likely_failure = (
                    "A paraphrase with none of the learned high-weight terms may fall back "
                    "to a different intent; inspect standard and hard validation examples."
                )
                {
                    "intent": inspected_intent,
                    "keywords": keyword.keywords_by_intent[inspected_intent][:6],
                    "hypothesis": likely_failure,
                }
                """,
                "exercise",
            ),
            md(
                """
                **Hint:** transparent rules are valuable because a failure hypothesis can
                be tied to visible features instead of model mythology.
                """,
                "hint",
            ),
            md(
                """
                ## Checkpoint

                Record which baseline is meaningful and why valid JSON alone is not a
                strong result.

                **Next:** `05_prompt_baselines.ipynb` holds weights fixed and changes only
                the prompt evidence on validation data.
                """,
                "checkpoint",
            ),
        ),
    ),
    Notebook(
        order=5,
        slug="prompt_baselines",
        title="Untouched-model prompt and few-shot baselines",
        stage="prompt_baselines",
        duration=50,
        prerequisites=("04_deterministic_baselines.ipynb",),
        evidence="a validation comparison of basic, strong, and few-shot prompts",
        cells=(
            intro(
                5,
                "Untouched-model prompt and few-shot baselines",
                minutes=50,
                prerequisites="04 — Deterministic baselines",
                objectives=(
                    "Hold model weights and validation examples constant across a prompt ladder.",
                    "Build few-shot demonstrations exclusively from training records.",
                    "Measure output quality and local resource use instead of eyeballing prose.",
                ),
                evidence="a validation comparison of basic, strong, and few-shot prompts",
            ),
            OFFLINE_SETUP,
            md(
                """
                ## Construct prompts before loading a model

                The basic prompt states the task. The strong prompt adds allowed labels,
                output shape, train-derived category mapping, escalation guidance, and
                safety constraints. Few-shot adds deterministic demonstrations from train.
                No prompt is selected using frozen-test failures.
                """,
                "what-to-notice",
            ),
            code("""
                import pandas as pd

                from aai_local_finetuning.evaluation import evaluate_predictions
                from aai_local_finetuning.learning import (
                    generate_support_predictions,
                    load_support_splits,
                    report_row,
                    select_few_shots,
                    support_contract,
                )
                from aai_local_finetuning.modeling import LocalMLXPredictor, build_messages
                from aai_local_finetuning.settings import load_settings

                settings = load_settings()
                splits = load_support_splits(settings, include_test=False)
                allowed_intents, categories = support_contract(splits.train)
                demonstrations = select_few_shots(splits.train, limit=4)
                validation_example = splits.validation[0]
                """),
            md("""
                ## Inspect the controlled change

                The final user message stays identical. Only the instruction/context
                changes. Demonstration IDs are not needed in the prompt, but selection is
                deterministic from training records and can be reconstructed.
                """),
            code("""
                prompt_ladder = {
                    strategy: build_messages(
                        validation_example.input_text,
                        strategy=strategy,
                        allowed_intents=list(allowed_intents),
                        category_by_intent=categories,
                        few_shot=demonstrations,
                    )
                    for strategy in ("basic", "strong", "few_shot")
                }
                {
                    strategy: {
                        "message_count": len(messages),
                        "system_preview": messages[0]["content"][:240],
                        "same_final_user_message": (
                            messages[-1]["content"] == validation_example.input_text
                        ),
                    }
                    for strategy, messages in prompt_ladder.items()
                }
                """),
            md(
                """
                ## Small measured validation experiment

                This default uses six validation examples so Run All remains practical on
                a MacBook Air. It teaches mechanics, not statistical certainty. Increase
                the limit only while prompts are still unlocked; record the final choice
                before opening the frozen evaluation notebook.
                """,
                "what-to-notice",
            ),
            code("""
                VALIDATION_LIMIT = 6
                validation_probe = splits.validation[:VALIDATION_LIMIT]
                predictor = LocalMLXPredictor(settings.model_dir)
                prompt_reports = {}
                prompt_predictions = {}
                for strategy in ("basic", "strong", "few_shot"):
                    predictions = generate_support_predictions(
                        predictor,
                        validation_probe,
                        strategy=strategy,
                        train_records=splits.train,
                        max_tokens=96,
                    )
                    prompt_predictions[strategy] = predictions
                    prompt_reports[strategy] = evaluate_predictions(
                        validation_probe,
                        predictions,
                        supported_intents=allowed_intents,
                    )
                pd.DataFrame(
                    [report_row(name, report) for name, report in prompt_reports.items()]
                )
                """),
            md(
                """
                ## Inspect output as evidence, not as a vibe

                A raw preview helps diagnose format errors. The strict report—not visual
                plausibility—determines JSON parse, schema validity, supported labels,
                classification, response policy, latency, tokens, and memory.
                """,
                "what-to-notice",
            ),
            code("""
                [
                    {
                        "strategy": strategy,
                        "example_id": predictions[0].example_id,
                        "output_preview": predictions[0].raw_text[:300],
                        "latency_ms": round(predictions[0].latency_ms, 1),
                        "output_tokens": predictions[0].output_tokens,
                    }
                    for strategy, predictions in prompt_predictions.items()
                ]
                """),
            md(
                """
                ## Exercise — lock a prompt strategy

                Choose using validation evidence. Success means the rationale mentions
                both classification and structured-output quality. Once notebook 07 is
                opened, do not revise this choice in response to frozen-test errors.
                """,
                "exercise",
            ),
            code(
                """
                chosen_strategy = "strong"
                choice_rationale = (
                    "Use the constrained label and schema contract as the default; the "
                    "small probe is insufficient to claim few-shot superiority."
                )
                assert chosen_strategy in prompt_reports
                {
                    "chosen_strategy": chosen_strategy,
                    "validation_metrics": report_row(
                        chosen_strategy, prompt_reports[chosen_strategy]
                    ),
                    "rationale": choice_rationale,
                }
                """,
                "exercise",
            ),
            md(
                """
                **Hint:** prefer a reproducible rule over choosing the nicest single
                output. A six-example probe can reject obvious failures, not establish a
                final winner.
                """,
                "hint",
            ),
            md(
                """
                ## Checkpoint

                You have three untouched-model baselines and a validation-locked prompt
                choice. The base weights have not changed.

                **Next:** `06_lora_finetuning.ipynb` inspects and optionally runs the
                adapter change without using frozen-test evidence.
                """,
                "checkpoint",
            ),
        ),
    ),
    Notebook(
        order=6,
        slug="lora_finetuning",
        title="LoRA fine-tuning as a controlled change",
        stage="training",
        duration=55,
        prerequisites=("05_prompt_baselines.ipynb",),
        evidence="a reviewed training configuration and measured adapter evidence",
        cells=(
            intro(
                6,
                "LoRA fine-tuning as a controlled change",
                minutes=55,
                prerequisites="05 — Prompt baselines",
                objectives=(
                    "Explain which weights LoRA changes and which base artifacts remain fixed.",
                    "Connect the portable chat records to the MLX-LM training configuration.",
                    "Interpret loss and memory as training evidence, not adoption evidence.",
                ),
                evidence="a reviewed training configuration and measured adapter evidence",
            ),
            OFFLINE_SETUP,
            md(
                """
                ## Inspect the change before running it

                The base model revision and 4-bit weights stay fixed. LoRA learns a small
                adapter over selected projections. The configuration is versioned so the
                change can be reproduced and hashed with its later evaluation evidence.
                """,
                "what-to-notice",
            ),
            code("""
                import json

                import yaml

                from aai_local_finetuning.settings import PROJECT_ROOT, load_settings

                settings = load_settings()
                config_path = PROJECT_ROOT / "configs" / "training" / "lora.yaml"
                training_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                training_config
                """),
            md(
                """
                ## Read one portable training example

                MLX-LM consumes the same framework-neutral messages a future trainer can
                consume. Metadata is evidence and slicing context; it is not appended to
                the user prompt as a shortcut to the target.
                """,
                "what-to-notice",
            ),
            code("""
                first_training_record = json.loads(
                    (settings.processed_dir / "train.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()[0]
                )
                {
                    "example_id": first_training_record["example_id"],
                    "message_roles": [
                        message["role"] for message in first_training_record["messages"]
                    ],
                    "user_preview": first_training_record["messages"][1]["content"][:140],
                    "assistant_target": json.loads(
                        first_training_record["messages"][2]["content"]
                    ),
                    "metadata_keys": sorted(first_training_record["metadata"]),
                }
                """),
            md(
                """
                ## What the conservative settings buy us

                Batch size 1, gradient accumulation, eight adapted layers, a bounded
                sequence length, prompt masking, and checkpointing reduce unified-memory
                pressure. They are design choices for the prepared 24 GB Apple-silicon
                machine, not universal optimal values.
                """,
                "what-to-notice",
            ),
            code("""
                training_anatomy = {
                    key: training_config.get(key)
                    for key in (
                        "model",
                        "train",
                        "fine_tune_type",
                        "num_layers",
                        "batch_size",
                        "grad_accumulation_steps",
                        "max_seq_length",
                        "mask_prompt",
                        "grad_checkpoint",
                        "iters",
                        "seed",
                    )
                }
                training_anatomy["lora_parameters"] = training_config.get(
                    "lora_parameters"
                )
                training_anatomy
                """),
            md("""
                ## Load measured preflight evidence

                Flight preparation already runs one real iteration to catch MLX compile,
                data-shape, and memory surprises. This is a readiness probe, not the final
                change. A missing file tells you preparation did not complete.
                """),
            code("""
                preflight_path = PROJECT_ROOT / "artifacts" / "training" / "preflight-smoke.json"
                preflight_evidence = (
                    json.loads(preflight_path.read_text(encoding="utf-8"))
                    if preflight_path.is_file()
                    else {"status": "missing; prepare this machine online"}
                )
                preflight_evidence
                """),
            md(
                """
                ## Optional live training cell

                The default is safe for Run All. Set `RUN_TRAINING = True` to run ten
                iterations into a notebook-specific adapter directory. This never
                overwrites the canonical `bitext-lora-v1` change. For the full configured
                run, set `TRAINING_ITERATIONS = None` only after the smoke evidence looks
                healthy and you have enough time and battery.
                """,
                "exercise",
            ),
            code(
                """
                from aai_local_finetuning.training import run_lora

                RUN_TRAINING = False
                TRAINING_ITERATIONS = 10
                notebook_adapter = (
                    PROJECT_ROOT / "artifacts" / "notebook" / "adapters" / "bitext-smoke"
                )
                if RUN_TRAINING:
                    training_evidence = run_lora(
                        iterations=TRAINING_ITERATIONS,
                        adapter_path=notebook_adapter,
                        log_name="notebook-bitext-smoke",
                    ).model_dump(mode="json")
                else:
                    training_evidence = {
                        "status": "skipped",
                        "how_to_run": "Set RUN_TRAINING = True",
                        "preflight": preflight_evidence,
                    }
                training_evidence
                """,
                "exercise",
            ),
            md(
                """
                ## Exercise — interpret optimization evidence

                After a run, compare training and validation losses and measured peak
                memory. Success means your conclusion avoids claiming that lower loss
                proves better frozen-test behavior or safer responses.
                """,
                "exercise",
            ),
            code(
                """
                optimization_conclusion = (
                    "The adapter optimization executed locally within the measured memory "
                    "profile. Only the frozen structured-output evaluation can determine "
                    "whether the change should be adopted."
                )
                assert "frozen" in optimization_conclusion.lower()
                optimization_conclusion
                """,
                "exercise",
            ),
            md(
                """
                **Hint:** loss is calculated on the training objective. Promotion asks a
                broader question about generalization, schema, labels, policy, latency,
                tokens, memory, and the strongest meaningful baseline.
                """,
                "hint",
            ),
            md(
                """
                ## Checkpoint

                You can now name the adapter as the change and explain why a successful
                training process is necessary but insufficient evidence.

                **Next:** `07_frozen_evaluation.ipynb` opens the frozen test once and
                applies the already locked methods.
                """,
                "checkpoint",
            ),
        ),
    ),
    Notebook(
        order=7,
        slug="frozen_evaluation",
        title="Frozen regression evaluation, slices, and bounded errors",
        stage="evaluation",
        duration=60,
        prerequisites=("06_lora_finetuning.ipynb",),
        evidence="comparable frozen reports, slice tables, and bounded error evidence",
        cells=(
            intro(
                7,
                "Frozen regression evaluation, slices, and bounded errors",
                minutes=60,
                prerequisites="06 — LoRA fine-tuning; prompts and settings are locked",
                objectives=(
                    "Score every method through the same framework-neutral evaluator.",
                    "Keep classification, structured-output, response-policy, and performance evidence separate.",
                    "Use slices and bounded errors without tuning against the frozen set.",
                ),
                evidence="comparable frozen reports, slice tables, and bounded error evidence",
            ),
            OFFLINE_SETUP,
            md(
                """
                ## Open the frozen boundary once choices are locked

                This is the first notebook that reads test examples for scoring. Do not
                revise prompts, demonstrations, thresholds, response policy, or training
                configuration after seeing these errors. A small default probe teaches
                the workflow; it is explicitly report-only and cannot support promotion.
                """,
                "what-to-notice",
            ),
            code("""
                import pandas as pd

                from aai_local_finetuning.evaluation import (
                    KeywordRuleBaseline,
                    MajorityBaseline,
                    evaluate_predictions,
                    format_error_analysis,
                    write_predictions_jsonl,
                    write_report_json,
                )
                from aai_local_finetuning.learning import (
                    generate_support_predictions,
                    load_support_splits,
                    report_row,
                    support_contract,
                )
                from aai_local_finetuning.modeling import LocalMLXPredictor
                from aai_local_finetuning.settings import PROJECT_ROOT, load_settings

                settings = load_settings()
                splits = load_support_splits(settings)
                allowed_intents, _ = support_contract(splits.train)
                FULL_FROZEN_COUNT = len(splits.test)
                EVALUATION_LIMIT = 9  # Set to None for complete promotion evidence.
                frozen_records = (
                    splits.test
                    if EVALUATION_LIMIT is None
                    else splits.test[:EVALUATION_LIMIT]
                )
                {
                    "scored_now": len(frozen_records),
                    "full_frozen_count": FULL_FROZEN_COUNT,
                    "promotion_eligible": len(frozen_records) == FULL_FROZEN_COUNT,
                }
                """),
            md("""
                ## Recompute deterministic methods on the same records

                Comparisons require the same record IDs, supported labels, and evaluator.
                The majority method is a sanity floor; the keyword/rule method is a
                meaningful transparent baseline.
                """),
            code("""
                methods = {
                    "majority": MajorityBaseline.fit(splits.train).predict_many(
                        frozen_records
                    ),
                    "keyword-rule": KeywordRuleBaseline.fit(
                        splits.train
                    ).predict_many(frozen_records),
                }
                reports = {
                    name: evaluate_predictions(
                        frozen_records,
                        predictions,
                        supported_intents=allowed_intents,
                    )
                    for name, predictions in methods.items()
                }
                pd.DataFrame(
                    [report_row(name, report) for name, report in reports.items()]
                )
                """),
            md(
                """
                ## Evaluate the three untouched-model prompts

                One local predictor keeps model weights fixed. Each method receives the
                same records and maximum output budget. The few-shot helper draws only
                from train. This can take several seconds on the prepared Mac.
                """,
                "what-to-notice",
            ),
            code("""
                predictor = LocalMLXPredictor(settings.model_dir)
                for strategy in ("basic", "strong", "few_shot"):
                    predictions = generate_support_predictions(
                        predictor,
                        frozen_records,
                        strategy=strategy,
                        train_records=splits.train,
                        max_tokens=96,
                    )
                    methods[strategy] = predictions
                    reports[strategy] = evaluate_predictions(
                        frozen_records,
                        predictions,
                        supported_intents=allowed_intents,
                    )
                pd.DataFrame(
                    [report_row(name, report) for name, report in reports.items()]
                )
                """),
            md("""
                ## Add the LoRA change when its adapter exists

                The canonical adapter is separate from notebook smoke adapters. Absence
                is evidence that the complete change has not been trained, so the later
                decision must remain inconclusive.
                """),
            code("""
                adapter_weights = settings.adapter_dir / "adapters.safetensors"
                if adapter_weights.is_file():
                    lora_predictor = LocalMLXPredictor(
                        settings.model_dir, adapter_path=settings.adapter_dir
                    )
                    methods["lora-change"] = generate_support_predictions(
                        lora_predictor,
                        frozen_records,
                        strategy="strong",
                        train_records=splits.train,
                        max_tokens=96,
                    )
                    reports["lora-change"] = evaluate_predictions(
                        frozen_records,
                        methods["lora-change"],
                        supported_intents=allowed_intents,
                    )
                else:
                    print("Canonical LoRA adapter absent; change evidence is incomplete.")
                pd.DataFrame(
                    [report_row(name, report) for name, report in reports.items()]
                )
                """),
            md("""
                ## Persist notebook evidence separately

                Notebook probes never overwrite official evaluation artifacts. Filenames
                include `partial` unless all frozen examples were scored. Reports carry
                the evaluation fingerprint used to prove comparability later.
                """),
            code("""
                evidence_dir = PROJECT_ROOT / "artifacts" / "notebook" / "evaluation"
                evidence_dir.mkdir(parents=True, exist_ok=True)
                scope = "full" if len(frozen_records) == FULL_FROZEN_COUNT else "partial"
                for name, report in reports.items():
                    write_predictions_jsonl(
                        evidence_dir / f"{scope}-{name}-predictions.jsonl",
                        methods[name],
                    )
                    write_report_json(
                        evidence_dir / f"{scope}-{name}-report.json",
                        report,
                    )
                sorted(path.name for path in evidence_dir.glob(f"{scope}-*"))
                """),
            md(
                """
                ## Slice and error analysis

                Counts accompany every slice so a perfect score on one example is not
                overinterpreted. Peak RSS is a process high-water mark, not precise
                per-request memory. Error previews are bounded and already masked.
                """,
                "what-to-notice",
            ),
            code("""
                inspected_method = "lora-change" if "lora-change" in reports else "strong"
                inspected_report = reports[inspected_method]
                lowest_intents = pd.DataFrame(
                    [
                        {"intent": intent, "f1": score}
                        for intent, score in inspected_report.classification.per_intent_f1.items()
                    ]
                ).sort_values("f1").head(10)
                difficulty_slices = pd.DataFrame(
                    [
                        {"difficulty": name, **metrics.model_dump(mode="json")}
                        for name, metrics in inspected_report.by_difficulty.items()
                    ]
                )
                lowest_intents, difficulty_slices
                """),
            code("""
                print(format_error_analysis(inspected_report))
                """),
            md(
                """
                ## Exercise — write a non-tuning error conclusion

                Select one error kind and explain what it means. Success means you do not
                propose changing the prompt or model based on frozen evidence; propose a
                future experiment with a newly versioned evaluation boundary instead.
                """,
                "exercise",
            ),
            code(
                """
                frozen_error_conclusion = (
                    "Record schema failures as a result of this locked experiment. Any "
                    "remediation becomes a new change evaluated on a new untouched test version."
                )
                assert "new" in frozen_error_conclusion.lower()
                frozen_error_conclusion
                """,
                "exercise",
            ),
            md(
                """
                **Hint:** frozen errors are evidence about this experiment, not free
                development feedback for the same test version.
                """,
                "hint",
            ),
            md(
                """
                ## Checkpoint

                Confirm that every compared report has the same fingerprint and record
                count. Partial evidence is useful for learning but cannot promote a change.

                **Next:** `08_mlflow_and_promotion.ipynb` records lineage and computes an
                adopt, reject, or inconclusive decision.
                """,
                "checkpoint",
            ),
        ),
    ),
    Notebook(
        order=8,
        slug="mlflow_and_promotion",
        title="Local MLflow evidence and promotion decisions",
        stage="decision",
        duration=40,
        prerequisites=("07_frozen_evaluation.ipynb",),
        evidence="a local run record and an adopt/reject/inconclusive assessment",
        cells=(
            intro(
                8,
                "Local MLflow evidence and promotion decisions",
                minutes=40,
                prerequisites="07 — Frozen regression evaluation",
                objectives=(
                    "Inspect local experiment lineage without a tracking server.",
                    "Require comparable complete reports before making a promotion decision.",
                    "Apply absolute output gates in addition to relative macro-F1 improvement.",
                ),
                evidence="a local run record and an adopt/reject/inconclusive assessment",
            ),
            OFFLINE_SETUP,
            md(
                """
                ## Local tracking, not cloud tracking

                MLflow uses a repository-local SQLite database and local artifact root.
                Runs can record dataset fingerprints, model revision, adapter/config
                hashes, predictions, reports, and measured metrics without credentials.
                """,
                "what-to-notice",
            ),
            code("""
                import mlflow

                from aai_local_finetuning.evaluation import (
                    BaselineEvaluation,
                    PromotionThresholds,
                    decide_lora_promotion,
                )
                from aai_local_finetuning.learning import load_report, load_support_splits
                from aai_local_finetuning.settings import PROJECT_ROOT, load_settings
                from aai_local_finetuning.tracking import configure_local_mlflow

                settings = load_settings()
                configure_local_mlflow(settings)
                runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=20)
                runs[
                    [
                        column
                        for column in (
                            "run_id",
                            "tags.mlflow.runName",
                            "tags.run_purpose",
                            "metrics.intent/macro_f1",
                            "metrics.output/json_schema_validity_rate",
                        )
                        if column in runs.columns
                    ]
                ]
                """),
            md("""
                ## Log a learner checkpoint

                This small run records that the notebook reached the decision stage. It
                does not masquerade as a frozen evaluation run and does not overwrite
                official evidence.
                """),
            code("""
                with mlflow.start_run(run_name="notebook-decision-checkpoint") as run:
                    mlflow.set_tags(
                        {
                            "run_purpose": "learner_checkpoint",
                            "execution_mode": "offline_local",
                        }
                    )
                    mlflow.log_metric("notebook_stage_complete", 1.0)
                    checkpoint_run_id = run.info.run_id
                checkpoint_run_id
                """),
            md(
                """
                ## Inventory full notebook reports

                Promotion requires all meaningful baselines and the LoRA change on the
                complete frozen set with identical fingerprints. Partial reports, missing
                methods, or mismatched fingerprints force `inconclusive`.
                """,
                "what-to-notice",
            ),
            code("""
                splits = load_support_splits(settings)
                report_dir = PROJECT_ROOT / "artifacts" / "notebook" / "evaluation"
                required_methods = (
                    "majority",
                    "keyword-rule",
                    "basic",
                    "strong",
                    "few_shot",
                    "lora-change",
                )
                report_paths = {
                    method: report_dir / f"full-{method}-report.json"
                    for method in required_methods
                }
                report_status = {
                    method: path.is_file() for method, path in report_paths.items()
                }
                report_status
                """),
            md(
                """
                ## Apply the decision contract

                Defaults require schema validity ≥ 0.98, response-policy compliance ≥
                0.95, unsupported-intent rate = 0, and macro F1 strictly above the
                strongest meaningful baseline. Majority is retained as a floor but is
                excluded from the meaningful-baseline competition.
                """,
                "what-to-notice",
            ),
            code("""
                thresholds = PromotionThresholds()
                if all(report_status.values()):
                    loaded_reports = {
                        method: load_report(path)
                        for method, path in report_paths.items()
                    }
                    fingerprints = {
                        report.evaluation_fingerprint
                        for report in loaded_reports.values()
                    }
                    counts = {
                        report.total_examples for report in loaded_reports.values()
                    }
                    complete_and_comparable = (
                        len(fingerprints) == 1
                        and counts == {len(splits.test)}
                    )
                else:
                    loaded_reports = {}
                    complete_and_comparable = False

                if complete_and_comparable:
                    assessment = decide_lora_promotion(
                        change_name="bitext-structured-output-lora-v1",
                        change_report=loaded_reports["lora-change"],
                        baselines=[
                            BaselineEvaluation(
                                name=name,
                                report=loaded_reports[name],
                                meaningful=name != "majority",
                            )
                            for name in required_methods
                            if name != "lora-change"
                        ],
                        thresholds=thresholds,
                    ).model_dump(mode="json")
                else:
                    assessment = {
                        "decision": "inconclusive",
                        "reasons": [
                            "all six methods must be scored on the complete frozen set",
                            "report counts and evaluation fingerprints must match",
                        ],
                        "available_reports": report_status,
                    }
                assessment
                """),
            md(
                """
                ## Exercise — defend the decision

                Write a short rationale that cites the strongest meaningful baseline,
                macro-F1 gain, schema gate, policy gate, and unsupported-intent gate—or
                names the missing evidence that makes the decision inconclusive.
                """,
                "exercise",
            ),
            code(
                """
                decision_rationale = (
                    "The current notebook run remains inconclusive unless all six full "
                    "reports share the frozen fingerprint; partial metrics are not promotion evidence."
                )
                assert any(
                    word in decision_rationale.lower()
                    for word in ("adopt", "reject", "inconclusive")
                )
                decision_rationale
                """,
                "exercise",
            ),
            md(
                """
                **Hint:** a decision is not a summary adjective. It is a reproducible
                function over baseline, change, result, thresholds, and comparability.
                """,
                "hint",
            ),
            md(
                """
                ## Checkpoint

                You have completed the Bitext lifecycle without treating training as
                success or using a partial run as promotion evidence.

                **Next:** `09_capstone_policy_dataset.ipynb` asks a different question:
                when should deterministic policy, not a language model, own the truth?
                """,
                "checkpoint",
            ),
        ),
    ),
    Notebook(
        order=9,
        slug="capstone_policy_dataset",
        title="Capstone: policy-derived ground truth",
        stage="capstone_data",
        duration=50,
        prerequisites=("08_mlflow_and_promotion.ipynb",),
        evidence="a versioned 400/100/150 policy dataset and deterministic ceiling",
        cells=(
            intro(
                9,
                "Capstone: policy-derived ground truth",
                minutes=50,
                prerequisites="08 — Local MLflow evidence and promotion decisions",
                objectives=(
                    "Separate deterministic, policy, external-lookup, and human-judgment rules.",
                    "Generate controlled ground truth without asking an LLM to invent critical labels.",
                    "Verify capstone split counts, slice coverage, hashes, and the deterministic ceiling.",
                ),
                evidence="a versioned 400/100/150 policy dataset and deterministic ceiling",
            ),
            OFFLINE_SETUP,
            md(
                """
                ## Why Kaggle is not the capstone ground truth

                Customer-support records do not represent Databricks application
                readiness. The capstone uses a small reviewed domain dataset generated
                from an explicit policy engine. Every expected check carries rule kind,
                source fields, facts origin, severity, and remediation provenance.
                """,
                "what-to-notice",
            ),
            code("""
                from collections import Counter

                import pandas as pd

                from aai_local_finetuning.capstone import (
                    REQUIRED_FROZEN_TEST_SLICES,
                    deterministic_capstone_predictions,
                    evaluate_capstone_predictions,
                    evaluate_manifest,
                    generate_capstone_dataset,
                    load_capstone_records,
                    render_capstone_mlx_dataset,
                    rule_catalog,
                )
                from aai_local_finetuning.settings import PROJECT_ROOT

                rules = rule_catalog()
                pd.DataFrame(
                    [
                        {
                            "rule": rule.rule_id,
                            "kind": rule.kind.value,
                            "severity": rule.failure_severity.value,
                            "source_fields": ", ".join(rule.source_fields),
                        }
                        for rule in rules
                    ]
                )
                """),
            md(
                """
                ## Rule kinds define authority

                Deterministic rules read manifest facts. Policy rules apply a versioned
                threshold or vocabulary. External lookups require an authorized system;
                human judgment requires a person. The latter two route to review rather
                than letting a tiny model invent facts.
                """,
                "what-to-notice",
            ),
            code("""
                Counter(rule.kind.value for rule in rules)
                """),
            md("""
                ## Generate reviewed combinations

                A fixed seed creates controlled rule violations and interacting failures.
                The frozen test includes required slices and unseen combinations of known
                failures. Generation writes portable records and immutable hashes.
                """),
            code("""
                source_dir = PROJECT_ROOT / "data" / "processed" / "capstone-readiness-v1"
                mlx_dir = PROJECT_ROOT / "data" / "processed" / "capstone-mlx-v1"
                split_manifest = generate_capstone_dataset(source_dir)
                training_manifest = render_capstone_mlx_dataset(source_dir, mlx_dir)
                {
                    "counts": {
                        artifact.split.value: artifact.record_count
                        for artifact in split_manifest.artifacts
                    },
                    "frozen_test": split_manifest.frozen_test,
                    "dataset_sha256": split_manifest.dataset_sha256,
                    "portable_training_fingerprint": training_manifest.dataset_fingerprint,
                }
                """),
            md("""
                ## Inspect one synthetic example and its provenance

                This is generated application metadata, not customer data. The expected
                review comes entirely from the policy engine, including review routing
                for facts that are unavailable locally.
                """),
            code("""
                train_records = load_capstone_records(source_dir / "train.jsonl")
                example = train_records[0]
                {
                    "example_id": example.example_id,
                    "slices": example.metadata.slices,
                    "manifest": example.manifest,
                    "expected_status": example.expected_output.status.value,
                    "non_pass_checks": [
                        {
                            "name": check.name,
                            "result": check.result.value,
                            "severity": check.severity.value,
                            "rule_kind": check.provenance.rule_kind.value,
                            "facts_origin": check.provenance.facts_origin,
                        }
                        for check in example.expected_output.checks
                        if check.result.value != "pass"
                    ],
                }
                """),
            md(
                """
                ## Verify the frozen slice contract and deterministic ceiling

                Every required slice must occur. The policy engine is the accuracy ceiling
                for rules it fully determines, so replacing those decisions with a model
                cannot improve correctness.
                """,
                "what-to-notice",
            ),
            code("""
                test_records = load_capstone_records(source_dir / "test.jsonl")
                test_slices = Counter(
                    slice_name
                    for record in test_records
                    for slice_name in record.metadata.slices
                )
                missing_slices = sorted(set(REQUIRED_FROZEN_TEST_SLICES) - set(test_slices))
                policy_report = evaluate_capstone_predictions(
                    test_records,
                    deterministic_capstone_predictions(test_records),
                )
                {
                    "missing_required_slices": missing_slices,
                    "exact_review_rate": policy_report.aggregate.exact_review_rate,
                    "schema_validity": policy_report.aggregate.schema_validity_rate,
                    "slice_counts": dict(sorted(test_slices.items())),
                }
                """),
            md(
                """
                ## Exercise — review a new manifest

                Change one field and inspect which check changes. Success means you can
                explain whether the outcome came from a manifest fact, platform policy,
                an external system, or human review.
                """,
                "exercise",
            ),
            code(
                """
                learner_manifest = dict(example.manifest)
                learner_manifest["owner"] = None
                learner_review = evaluate_manifest(learner_manifest)
                [
                    {
                        "name": check.name,
                        "result": check.result.value,
                        "kind": check.provenance.rule_kind.value,
                        "origin": check.provenance.facts_origin,
                        "evidence": check.evidence,
                    }
                    for check in learner_review.checks
                    if check.result.value != "pass"
                ]
                """,
                "exercise",
            ),
            md(
                """
                **Hint:** the engine may explain an absent local fact, but it must not
                claim that a registry lookup or human review occurred when it did not.
                """,
                "hint",
            ),
            md(
                """
                ## Checkpoint

                You now have deterministic, versioned ground truth and a measurable
                ceiling—not LLM-generated labels presented as facts.

                **Next:** `10_capstone_model_vs_hybrid.ipynb` tests where a tiny model may
                add value without owning authoritative readiness decisions.
                """,
                "checkpoint",
            ),
        ),
    ),
    Notebook(
        order=10,
        slug="capstone_model_vs_hybrid",
        title="Capstone: model versus deterministic and hybrid designs",
        stage="capstone_architecture",
        duration=55,
        prerequisites=("09_capstone_policy_dataset.ipynb",),
        evidence="a policy/model comparison and an explicit architecture choice",
        cells=(
            intro(
                10,
                "Capstone: model versus deterministic and hybrid designs",
                minutes=55,
                prerequisites="09 — Capstone policy-derived ground truth",
                objectives=(
                    "Compare a deterministic ceiling with an untouched tiny-model probe.",
                    "Prove that a hybrid renderer cannot alter authoritative decisions.",
                    "Decide which behavior, if any, justifies a language model.",
                ),
                evidence="a policy/model comparison and an explicit architecture choice",
            ),
            OFFLINE_SETUP,
            md(
                """
                ## Start with the accuracy ceiling

                The full deterministic frozen report should be exact for completely
                specified rules. The model must therefore justify itself through a
                different capability—such as bounded wording—rather than replacing
                correct policy decisions with probabilistic ones.
                """,
                "what-to-notice",
            ),
            code("""
                import json

                from aai_local_finetuning.capstone import (
                    CAPSTONE_SYSTEM_PROMPT,
                    CapstonePrediction,
                    build_hybrid_review,
                    deterministic_capstone_predictions,
                    evaluate_capstone_predictions,
                    load_capstone_records,
                )
                from aai_local_finetuning.modeling import LocalMLXPredictor
                from aai_local_finetuning.settings import PROJECT_ROOT, load_settings

                settings = load_settings()
                source_dir = PROJECT_ROOT / "data" / "processed" / "capstone-readiness-v1"
                test_records = load_capstone_records(source_dir / "test.jsonl")
                deterministic_report = evaluate_capstone_predictions(
                    test_records,
                    deterministic_capstone_predictions(test_records),
                )
                deterministic_report.aggregate.model_dump(mode="json")
                """),
            md("""
                ## One untouched-model probe

                This bounded example shows mechanics, not a winner. The compact model
                contract asks for status and non-pass checks. A full model comparison is
                expensive and should be run only after methods are locked.
                """),
            code("""
                probe_record = test_records[0]
                predictor = LocalMLXPredictor(settings.model_dir)
                generated = predictor.generate(
                    [
                        {"role": "system", "content": CAPSTONE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(
                                probe_record.manifest,
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                        },
                    ],
                    max_tokens=160,
                )
                model_prediction = CapstonePrediction(
                    example_id=probe_record.example_id,
                    raw_text=generated.text,
                    latency_ms=generated.latency_ms,
                    output_tokens=generated.output_tokens,
                    peak_memory_mb=generated.peak_memory_mb,
                )
                model_probe_report = evaluate_capstone_predictions(
                    (probe_record,), (model_prediction,)
                )
                {
                    "output_preview": generated.text[:500],
                    "metrics": model_probe_report.aggregate.model_dump(mode="json"),
                    "performance": model_probe_report.performance.model_dump(mode="json"),
                }
                """),
            md(
                """
                ## The hybrid authority boundary

                A renderer receives a frozen check and may return prose only. It has no
                channel for changing status, result, severity, rule ID, or remediation ID.
                Empty or failing renderers fall back to deterministic policy wording.
                """,
                "what-to-notice",
            ),
            code("""
                def learner_renderer(check):
                    return f"Review note: {check.evidence}"


                hybrid = build_hybrid_review(
                    probe_record.manifest,
                    renderer=learner_renderer,
                    renderer_name="notebook_demo",
                )
                {
                    "status_unchanged": (
                        hybrid.deterministic_review.status
                        == probe_record.expected_output.status
                    ),
                    "checks_unchanged": (
                        hybrid.deterministic_review.checks
                        == probe_record.expected_output.checks
                    ),
                    "explanation_preview": hybrid.explanations[0].model_dump(
                        mode="json"
                    ),
                }
                """),
            md(
                """
                ## Optional capstone LoRA smoke

                Keep this disabled for Run All. When enabled it writes a notebook-specific
                adapter and leaves the canonical capstone change untouched. Falling loss
                still does not beat the deterministic ceiling.
                """,
                "exercise",
            ),
            code(
                """
                from aai_local_finetuning.training import run_lora

                RUN_CAPSTONE_TRAINING = False
                if RUN_CAPSTONE_TRAINING:
                    capstone_training = run_lora(
                        iterations=10,
                        config_path=(
                            PROJECT_ROOT / "configs" / "training" / "capstone-lora.yaml"
                        ),
                        adapter_path=(
                            PROJECT_ROOT
                            / "artifacts"
                            / "notebook"
                            / "adapters"
                            / "capstone-smoke"
                        ),
                        log_name="notebook-capstone-smoke",
                    ).model_dump(mode="json")
                else:
                    capstone_training = {"status": "skipped"}
                capstone_training
                """,
                "exercise",
            ),
            md(
                """
                ## Exercise — choose the production shape

                Fill in one row per behavior. Success means deterministic checks retain
                authority, unavailable facts route outward, and the model is used only
                where probabilistic language actually helps.
                """,
                "exercise",
            ),
            code(
                """
                architecture_decision = [
                    {
                        "behavior": "readiness decision",
                        "owner": "deterministic policy engine",
                        "reason": "exact, auditable rules already define the answer",
                    },
                    {
                        "behavior": "external registry fact",
                        "owner": "authorized lookup",
                        "reason": "the fact is absent from the local manifest",
                    },
                    {
                        "behavior": "remediation wording",
                        "owner": "policy text or constrained tiny-model renderer",
                        "reason": "wording may vary without changing authority",
                    },
                ]
                architecture_decision
                """,
                "exercise",
            ),
            md(
                """
                **Hint:** ask what must be correct, what must be looked up, what needs a
                person, and what merely benefits from flexible wording.
                """,
                "hint",
            ),
            md(
                """
                ## Checkpoint

                The likely design is deterministic validation plus optional constrained
                language generation—not a model pretending to know every readiness fact.

                **Next:** `11_design_the_next_project.ipynb` turns the remaining dataset
                ideas into review plans without fabricating unverified schemas or rights.
                """,
                "checkpoint",
            ),
        ),
    ),
    Notebook(
        order=11,
        slug="design_the_next_project",
        title="Design the next dataset project safely",
        stage="extensions",
        duration=35,
        prerequisites=("10_capstone_model_vs_hybrid.ipynb",),
        evidence="a dataset-review checklist and task-specific evaluation plan",
        cells=(
            intro(
                11,
                "Design the next dataset project safely",
                minutes=35,
                prerequisites="10 — Capstone model versus hybrid",
                objectives=(
                    "Treat optional datasets as a review queue, not pre-approved inputs.",
                    "Narrow the task before adding multilingual, extraction, transformation, or code-generation complexity.",
                    "Match evaluation to the behavior instead of reusing intent metrics blindly.",
                ),
                evidence="a dataset-review checklist and task-specific evaluation plan",
            ),
            OFFLINE_SETUP,
            md(
                """
                ## Why these are plans, not executable labs yet

                The optional Kaggle datasets are not cached or verified in this project.
                Their current schema, license, access, source composition, sensitive-data
                risk, and redistribution terms must be reviewed before code or claims are
                added. Offline study should never fill those gaps with assumptions.
                """,
                "what-to-notice",
            ),
            code("""
                candidates = [
                    {
                        "project": "multilingual support tickets",
                        "first_task": "English-only queue/priority/type/review routing",
                        "evaluation": "per-field metrics, imbalance and queue/priority slices",
                        "status": "unsuitable until current source review is complete",
                    },
                    {
                        "project": "invoice extraction",
                        "first_task": "text-only fields only if reliable OCR text exists",
                        "evaluation": "field exactness/F1, normalization, hallucinated fields",
                        "status": "unsuitable until modality, schema, and rights are verified",
                    },
                    {
                        "project": "prompt transformation",
                        "first_task": "required prompt components and structured output",
                        "evaluation": "rubric components plus calibrated human review",
                        "status": "optional; open-ended scoring is less deterministic",
                    },
                    {
                        "project": "MiniZinc generation",
                        "first_task": "small natural-language-to-program problems",
                        "evaluation": "parse, compile, execute, constraints, objective, runtime",
                        "status": "advanced; may exceed tiny-model capability",
                    },
                ]
                candidates
                """),
            md(
                """
                ## Required source review

                A current review must record title, owner, URL, license, permitted use,
                redistribution, size, formats, confirmed columns, languages, labels,
                missingness, duplicates, sensitive information, human/synthetic origin,
                accessibility, and access date. An unclear license means unsuitable until
                the learner verifies it directly.
                """,
                "what-to-notice",
            ),
            code("""
                required_review_fields = (
                    "title",
                    "owner",
                    "current_url",
                    "license",
                    "permitted_use",
                    "redistribution",
                    "size",
                    "formats",
                    "columns",
                    "languages",
                    "label_quality",
                    "missing_values",
                    "duplicate_rate",
                    "sensitive_information",
                    "record_origin",
                    "accessible",
                    "accessed_on",
                )
                required_review_fields
                """),
            md(
                """
                ## Exercise — draft, but do not invent, a review

                Choose one project. Fill only facts you have verified later while online;
                leave unknowns as `None`. The suitability gate must remain false whenever
                the license, schema, access, or modality is unknown.
                """,
                "exercise",
            ),
            code(
                """
                dataset_review = {field: None for field in required_review_fields}
                dataset_review["title"] = "<verify current title online>"
                dataset_review["accessible"] = None
                blocking_fields = ("license", "columns", "accessible")
                suitable_for_lab = all(dataset_review[field] for field in blocking_fields)
                {
                    "review": dataset_review,
                    "suitable_for_lab": suitable_for_lab,
                    "decision": (
                        "continue to adapter design"
                        if suitable_for_lab
                        else "unsuitable until direct verification"
                    ),
                }
                """,
                "exercise",
            ),
            md(
                """
                ## Exercise — choose task-shaped metrics

                Select one task and add a metric that catches a failure ordinary exact
                match would miss. For images, remember that a text-only model cannot see
                pixels without a separate OCR or multimodal stage.
                """,
                "exercise",
            ),
            code(
                """
                evaluation_plan = {
                    "task": "invoice text extraction",
                    "principal_metrics": [
                        "field-level precision/recall/F1",
                        "normalized date and currency exactness",
                        "hallucinated-field rate",
                        "schema validity",
                    ],
                    "human_review_rule": (
                        "route missing, conflicting, or low-confidence critical fields"
                    ),
                    "modality_boundary": (
                        "requires reliable OCR text; a tiny text model cannot read images"
                    ),
                }
                evaluation_plan
                """,
                "exercise",
            ),
            md(
                """
                **Hint:** the output contract determines the evaluator. Classification,
                extraction, open-ended transformation, and executable code require
                different evidence.
                """,
                "hint",
            ),
            md(
                """
                ## Final checkpoint

                You have completed a full local lifecycle and can now design the next
                project without assuming that a public dataset is licensed, suitable,
                text-only, balanced, clean, or evaluable in the same way.

                Revisit `00_start_here.ipynb` with a new evidence question, version the
                source contract, and create a new untouched evaluation boundary.
                """,
                "checkpoint",
            ),
        ),
    ),
)


def _cell_payload(notebook: Notebook, index: int, cell: Cell) -> dict[str, object]:
    digest = hashlib.sha1(
        f"{notebook.filename}:{index}:{cell.kind}".encode(), usedforsecurity=False
    ).hexdigest()[:12]
    payload: dict[str, object] = {
        "cell_type": cell.kind,
        "id": digest,
        "metadata": {"tags": list(cell.tags)} if cell.tags else {},
        "source": cell.source.splitlines(keepends=True),
    }
    if cell.kind == "code":
        payload.update({"execution_count": None, "outputs": []})
    return payload


def render(notebook: Notebook) -> None:
    payload = {
        "cells": [
            _cell_payload(notebook, index, cell)
            for index, cell in enumerate(notebook.cells)
        ],
        "metadata": {
            "aai_curriculum": {
                "order": notebook.order,
                "stage": notebook.stage,
                "duration_minutes": notebook.duration,
                "prerequisites": list(notebook.prerequisites),
                "learner_evidence": notebook.evidence,
            },
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = NOTEBOOK_DIR / notebook.filename
    destination.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    expected = {notebook.filename for notebook in NOTEBOOKS}
    for path in NOTEBOOK_DIR.glob("[0-9][0-9]_*.ipynb"):
        if path.name not in expected:
            path.unlink()
    for notebook in NOTEBOOKS:
        render(notebook)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "black",
            "--quiet",
            *(str(NOTEBOOK_DIR / notebook.filename) for notebook in NOTEBOOKS),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
