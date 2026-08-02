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

from notebook_pedagogy import (
    PRACTICE_REVIEWED_ON,
    PRIMERS,
    REFERENCE_URLS,
    RUNNING_EXAMPLES,
)

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
    return md(
        "\n".join(
            [
                f"# {number:02d} — {title}",
                "",
                f"**Estimated time:** {minutes} minutes<br>",
                f"**Prerequisites:** {prerequisites}<br>",
                f"**Learner-produced evidence:** {evidence}",
                "",
                "## Learning objectives",
                "",
                *(f"- {item}" for item in objectives),
                "",
                (
                    "This notebook is a teaching interface over the reusable code "
                    "in `src/`."
                ),
                (
                    "It uses only prepared local files. Run `make prepare-flight` "
                    "before the trip;"
                ),
                "no cell installs packages or downloads data.",
            ]
        )
    )


def _reference_line(reference: str) -> str:
    """Render a categorized primary reference without nested Markdown brackets."""

    category, title = reference.removeprefix("[").split("] ", maxsplit=1)
    return f"- **{category}:** [{title}]({REFERENCE_URLS[reference]})"


def pedagogy_cells(notebook: Notebook) -> tuple[Cell, ...]:
    """Build the beginner-first conceptual layer that precedes execution."""

    primer = PRIMERS[notebook.order]
    terms = [f"- **{term}:** {meaning}." for term, meaning in primer.terms]
    decisions = [f"- {question}" for question in primer.decision_questions]
    practices = [f"- {practice}" for practice in primer.practices]
    mistakes = [f"- {mistake}" for mistake in primer.mistakes]
    references = [_reference_line(reference) for reference in primer.references]

    mechanics: tuple[Cell, ...] = ()
    if notebook.order == 0:
        mechanics = (
            md(
                """
                ## How to use this course

                A **Markdown cell** explains an idea; a **code cell** performs a
                small local experiment. Read the explanation first, select the
                `AAI Local Fine-Tuning (offline)` kernel, and press **Shift+Enter**
                to run one cell at a time. Cells headed **Setup — run, do not edit**
                are plumbing rather than lesson exercises.

                When a cell returns a table or dictionary, interpret it in three
                steps:

                1. **What does it say?** Describe the measured value without judgment.
                2. **What would concern me?** Connect the value to a failure risk.
                3. **What would I do next?** Name a check, mitigation, or stop condition.

                The notebooks contain worked examples, then exercises and checkpoints.
                Generated `.ipynb` files are outputs: maintainers change the narrative
                in `scripts/render_notebooks.py` or `scripts/notebook_pedagogy.py` and
                regenerate them.
                """,
                "course-mechanics",
                "concepts",
            ),
        )

    return (
        *mechanics,
        md(
            "\n".join(
                [
                    "## Why this matters",
                    "",
                    primer.why,
                    "",
                    "## Key terms in plain language",
                    "",
                    *terms,
                ]
            ),
            "concepts",
        ),
        md(
            "\n".join(
                [
                    "## Mental model — how to think about this",
                    "",
                    primer.mental_model,
                    "",
                    "### Running example",
                    "",
                    RUNNING_EXAMPLES[notebook.order],
                    "",
                    "### Questions to ask before continuing",
                    "",
                    *decisions,
                ]
            ),
            "mental-model",
        ),
        md(
            "\n".join(
                [
                    "## Current best practices",
                    "",
                    (
                        f"**Guidance reviewed:** {PRACTICE_REVIEWED_ON}. "
                        "These are reasons to inspect future tool changes, not a "
                        "claim that practice stops evolving."
                    ),
                    "",
                    *practices,
                    "",
                    "## Common mistakes and why they fail",
                    "",
                    *mistakes,
                    "",
                    "### What kind of guidance is this?",
                    "",
                    (
                        "A **specification** defines a technical contract; **tool "
                        "guidance** describes current official library behavior; "
                        "**risk guidance** is voluntary governance guidance; and a "
                        "**course rule** is this project's deliberately conservative "
                        "choice. Do not call all four a formal standard. The lesson is "
                        "complete offline; these primary links are optional follow-up reading."
                    ),
                    "",
                    *references,
                ]
            ),
            "best-practices",
            "standards-reference",
        ),
    )


OFFLINE_SETUP = code(
    """
    import sys
    from importlib import import_module
    from pathlib import Path

    current = Path.cwd().resolve()
    project_root = None
    for candidate in (current, *current.parents):
        direct = candidate
        nested = candidate / "examples" / "local-finetuning"
        if (direct / "src" / "aai_local_finetuning").is_dir():
            project_root = direct
            break
        if (nested / "src" / "aai_local_finetuning").is_dir():
            project_root = nested
            break
    if project_root is None:
        raise RuntimeError(
            "Cannot locate examples/local-finetuning. Open this notebook from the "
            "repository, or run `make notebook` from the repository root."
        )

    expected_python = (project_root / ".venv" / "bin" / "python").resolve()
    active_python = Path(sys.executable).resolve()
    if not expected_python.is_file() or active_python != expected_python:
        raise RuntimeError(
            "Wrong notebook kernel. Run `make notebook` from the repository root, "
            "then select 'AAI Local Fine-Tuning (offline)'. "
            f"Active Python: {active_python}; expected: {expected_python}"
        )

    source_root = str(project_root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    enable_offline_environment = import_module(
        "aai_local_finetuning.offline"
    ).enable_offline_environment
    enable_offline_environment()

    {
        "setup": "ready",
        "kernel": "AAI Local Fine-Tuning (offline)",
        "python": str(active_python),
        "network_library_flags": "enabled",
        "note": "Continue to the lesson; this cell is setup, not an exercise.",
    }
    """,
    "offline-setup",
    "setup-run-do-not-edit",
)


SETUP_GUIDANCE = md(
    """
    ## Setup — run, do not edit

    Run the next cell once. It verifies the dedicated local Python kernel, finds
    this sample project, and enables supported offline flags **before** model or
    tracking libraries are imported. A successful cell ends with `setup: ready`.

    This is one defense layer, not proof that every native library is physically
    incapable of networking. The flight-preparation manifest, cached assets,
    socket-denial checks, and a Wi-Fi-off rehearsal provide the other layers.
    """,
    "setup-guidance",
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

                The paths below stay local. A row with `ready: false` means preparation
                is incomplete; it does not trigger a download. The cell stops rather
                than letting a missing asset become an in-flight surprise. The model,
                source archive, generated data, adapters, and MLflow database are
                deliberately ignored by Git.
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
                not_ready = [item for item in readiness if not item["ready"]]
                if not_ready:
                    failed = ", ".join(item["asset"] for item in not_ready)
                    raise RuntimeError(
                        "Offline study is not ready. Re-run `make prepare-flight` while "
                        f"online, then rehearse with Wi-Fi off. Failed checks: {failed}"
                    )
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
                or policy-breaking generated output. In this course, “response-policy
                compliant” means only that the output passed a small versioned set of
                wording checks; it is not a broad production-safety claim.

                A valid example has all four fields and the correct types:

                ```json
                {"intent":"recover_password","category":"account","requires_escalation":false,"response":"I can help you reset your password."}
                ```

                `{"intent":"recover_password"}` is valid JSON but fails the schema
                because fields are missing. `intent=recover_password` is not JSON at all.
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
                streams the files, so the raw directory remains unchanged. “Immutable”
                here is a process rule: the hash detects changed bytes but cannot prevent
                a person or program from replacing them.
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
                if not all(
                    local_integrity[key]
                    for key in ("archive_matches", "csv_matches")
                ):
                    raise RuntimeError(
                        "Local data bytes differ from the reviewed course snapshot. "
                        "Stop; do not explore, train, or reuse the old license review."
                    )
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
                    PreparationConfig,
                    audit_dataset,
                    check_split_files,
                    processing_source_sha256,
                    sha256_file,
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
                manifest_path = processed_dir / "manifest.json"
                quality_report_path = processed_dir / "quality_report.json"
                """),
            md(
                """
                ## Quality audit

                The record funnel explains why counts can shrink:

                | Stage | Meaning |
                |---|---|
                | Source | Every parsed CSV row |
                | Valid | Rows satisfying the required field and type contract |
                | Unique | Repeated normalized learning content counted once |
                | Non-conflicting | Duplicate groups whose labels do not disagree |
                | Curated | Eligible records selected under the versioned balance policy |
                | Split | Curated records assigned to train, validation, or frozen test |

                The report contains counts and distributions, not raw samples. Exact
                duplicates are measured after canonicalization. Near duplicates and
                inferred templates are heuristic evidence and must be documented as such.

                **Modern evidence practice:** preparation computes this expensive audit
                once and puts its bytes under the dataset manifest. A lesson should verify
                and read that evidence by default rather than silently repeat minutes of
                work. Recompute when the source, policy, or pipeline changes, then compare
                the new evidence before replacing the approved artifact. Set the switch
                below to `True` when you deliberately want that full local recomputation.
                """,
                "what-to-notice",
            ),
            code("""
                RECOMPUTE_FULL_AUDIT = False

                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                processing_config = PreparationConfig.model_validate(
                    manifest["processing"]
                )
                quality_evidence = manifest["artifacts"]["quality_report"]
                source_evidence = next(
                    item
                    for item in manifest["raw_files"]
                    if item["path"] == raw_csv.name
                )
                evidence_mismatches = []
                if quality_report_path.stat().st_size != quality_evidence["size_bytes"]:
                    evidence_mismatches.append("quality report size")
                if sha256_file(quality_report_path) != quality_evidence["sha256"]:
                    evidence_mismatches.append("quality report SHA-256")
                if raw_csv.stat().st_size != source_evidence["size_bytes"]:
                    evidence_mismatches.append("source size")
                if sha256_file(raw_csv) != source_evidence["sha256"]:
                    evidence_mismatches.append("source SHA-256")
                processing_is_current = (
                    processing_source_sha256()
                    == manifest["processing_source_sha256"]
                )
                if not processing_is_current and not RECOMPUTE_FULL_AUDIT:
                    evidence_mismatches.append("processing source SHA-256")
                if evidence_mismatches:
                    raise RuntimeError(
                        "Prepared audit evidence is stale or damaged. Re-run flight "
                        "preparation before studying. Failed checks: "
                        + ", ".join(evidence_mismatches)
                    )

                prepared_audit_payload = json.loads(
                    quality_report_path.read_text(encoding="utf-8")
                )
                measured_fields = (
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
                if RECOMPUTE_FULL_AUDIT:
                    audit_payload = audit_dataset(
                        raw_csv,
                        config=processing_config,
                    ).model_dump(mode="json")
                    audit_source = "recomputed now from immutable source bytes"
                    changes_from_prepared = {
                        key: {
                            "prepared": prepared_audit_payload[key],
                            "recomputed": audit_payload[key],
                        }
                        for key in measured_fields
                        if audit_payload[key] != prepared_audit_payload[key]
                    }
                else:
                    audit_payload = prepared_audit_payload
                    audit_source = "prepared once and SHA-256 verified from the manifest"
                    changes_from_prepared = {}
                core_quality = {
                    "evidence_source": audit_source,
                    **{key: audit_payload[key] for key in measured_fields},
                }
                {
                    "quality": core_quality,
                    "changes_from_prepared": changes_from_prepared,
                }
                """),
            md(
                """
                ## Labels, lengths, language, and sensitive-looking patterns

                Pattern matches are counts only. Email-, URL-, phone-like text and
                placeholders are masked before portable training records are written.
                Source flags remain explicit evaluation slices; difficulty is a separate
                versioned heuristic, not a human quality label. Language below is the
                source's declared coverage, not language detected independently in every row.
                """,
                "what-to-notice",
            ),
            code("""
                token_lengths = summarize_instruction_tokens(raw_csv, settings.model_dir)
                distribution_report = {
                    "intents": audit_payload["intent_distribution"],
                    "categories": audit_payload["category_distribution"],
                    "declared_language_coverage": {
                        settings.dataset.language: audit_payload["source_records"]
                    },
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
                ## Preview the prepared split-integrity gate

                Notebook 03 teaches how the split is constructed. For now, this is a
                preview of already-prepared evidence: the gate examines boundaries
                without displaying frozen test content. It checks exact, inferred-template,
                and near-duplicate overlap plus target and demonstration leakage.
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

                | Split | May influence | Must not influence |
                |---|---|---|
                | Train | Learned weights, train-derived statistics, demonstrations | Final generalization claim |
                | Validation | Prompt, checkpoint, threshold, and configuration choice | Learned examples or final claim |
                | Frozen test | Final measurement of already-locked methods | Any earlier design choice |

                “Frozen” is a governance promise, not a file permission. A hash reveals
                changed bytes but does not stop a person from reading them. This notebook
                reads manifest metadata and runs automated checks, but does not display or
                use test examples. If a test result later causes a prompt change, that test
                has become development data and a fresh untouched boundary is required.
                """,
                "what-to-notice",
            ),
            code("""
                import json

                from aai_local_finetuning.data import (
                    check_split_files,
                    text_similarity,
                    verify_manifest,
                )
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
                verification = verify_manifest(processed / "manifest.json")
                {
                    "splits": split_contract,
                    "artifact_hashes_valid": verification.valid,
                    "checked_files": verification.checked_files,
                    "mismatches": verification.mismatches,
                }
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

                The gate distinguishes exact overlap, inferred-template overlap, near
                duplicates, shared source groups, target leakage, and demonstration
                leakage. Development contamination—changing a method after test feedback—is
                a process failure and cannot be detected from files alone.

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
                threshold = manifest["processing"]["near_duplicate_threshold"]
                grouping_decision = similarity >= threshold
                limitation = (
                    "Similarity is a reproducible heuristic; it is not a verified "
                    "conversation or template identifier from the source."
                )
                {
                    "similarity": round(similarity, 3),
                    "configured_threshold": threshold,
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
        evidence="validation metrics for majority and locked keyword/rule baselines",
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
                evidence="validation metrics for majority and locked keyword/rule baselines",
            ),
            OFFLINE_SETUP,
            md(
                """
                ## Fit only on training evidence

                The majority baseline learns its fixed label from train. The keyword/rule
                baseline learns label counts, token weights, category mappings, and
                escalation defaults from train **and** includes human-authored phrase and
                escalation rules locked in source code before validation. It is therefore
                transparent and input-aware, but not wholly train-derived. Validation
                estimates behavior while methods may still change; test remains unopened.
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

                Read the evaluation ladder from basic usability toward task quality:

                `parseable JSON → schema-valid fields → allowed label → correct task answer →`
                `course-policy-compliant response → acceptable latency/tokens/memory`

                One record can fail several layers, so error-kind counts can overlap and
                must not be added together to infer a number of failed records.
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
                ## How to read the score table

                | Field | Better direction | Question it answers |
                |---|---:|---|
                | Intent accuracy | Higher | How often was the exact intent correct? |
                | Macro precision / recall / F1 | Higher | How well does each intent perform when every intent has equal influence? |
                | Weighted F1 | Higher | How well does the observed label mix perform, giving common intents more influence? |
                | Category / escalation accuracy | Higher | Were the other authoritative target fields correct? |
                | JSON parse / schema validity | Higher | Can code read the output, and does it satisfy the exact contract? |
                | Unsupported-intent rate | Lower | How often did the method invent a label outside the vocabulary? |
                | Response-policy compliance | Higher | Did wording pass this course's narrow lexical policy? |
                | Latency, tokens, memory | Context or lower | What local resource cost accompanied the result? |

                “Support” in a per-intent table means the number of true evaluation
                examples for that intent; it does not mean customer-support quality.
                Passing the response policy does not establish truthfulness, helpfulness,
                privacy, or general safety.
                """,
                "interpretation",
                "what-to-notice",
            ),
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
                        "support": [
                            baseline_reports["keyword-rule"].by_intent[intent].count
                            for intent in baseline_reports[
                                "keyword-rule"
                            ].classification.per_intent_f1
                        ],
                    }
                ).sort_values("f1").head(10)
                """),
            md(
                """
                ## Inspect what the transparent baseline learned

                The displayed weighted terms come only from training records. Separate
                phrase rules and escalation triggers are human-authored course rules.
                Both are useful for debugging and reveal brittleness: lexical shortcuts
                can fail on paraphrases, ambiguity, negation, or overlapping vocabulary.
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

                This default selects one record from each of six distinct intents so Run
                All remains practical on a MacBook Air. It is a stratified smoke probe,
                not a representative validation estimate. Macro F1 is still calculated
                over all 27 supported intents, so the 21 absent intents receive zero and
                the absolute value is deliberately not a prompt-quality claim. Use probe
                coverage plus obvious contract failures here; a broad validation run is
                required before locking a winner.
                """,
                "what-to-notice",
            ),
            code("""
                VALIDATION_LIMIT = 6
                first_by_intent = {}
                for record in splits.validation:
                    first_by_intent.setdefault(record.target.intent, record)
                    if len(first_by_intent) == VALIDATION_LIMIT:
                        break
                validation_probe = tuple(first_by_intent.values())
                probe_context = {
                    "selection": "one validation record per distinct intent",
                    "records": len(validation_probe),
                    "covered_intents": sorted(first_by_intent),
                    "supported_intents": len(allowed_intents),
                    "macro_f1_warning": (
                        "Absent supported intents score zero; do not interpret this "
                        "small-probe macro F1 as an absolute quality estimate."
                    ),
                }
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
                display(pd.DataFrame(
                    [report_row(name, report) for name, report in prompt_reports.items()]
                ))
                probe_context
                """),
            md(
                """
                ## Inspect output as evidence, not as a vibe

                A raw preview helps diagnose format errors. The strict report—not visual
                plausibility—determines JSON parse, schema validity, supported labels,
                classification, response policy, latency, tokens, and memory. This course
                currently records output tokens only; few-shot prompts also consume more
                **input** tokens, so the displayed token field is not a total-cost comparison.
                The first local generation can pay compilation/warm-up cost, which also
                makes a six-record latency comparison diagnostic rather than definitive.
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
                    "This is a conservative course default, not the computed winner. Use "
                    "the constrained contract until a broad validation run compares all "
                    "strategies; the small probe cannot establish few-shot superiority."
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
                MLX-LM calls LoRA training over this quantized base **QLoRA**. The YAML's
                `fine_tune_type: lora` names the adapter type; the local model's 4-bit
                weights determine that quantized training path.
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
                machine, not universal optimal values. `grad_checkpoint` recomputes
                intermediate activations to save memory; it is different from a saved
                adapter checkpoint. With batch size 1 and four accumulation steps, the
                approximate effective batch is four sequences per optimizer update.
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
                        "optimizer",
                        "num_layers",
                        "batch_size",
                        "grad_accumulation_steps",
                        "learning_rate",
                        "max_seq_length",
                        "mask_prompt",
                        "grad_checkpoint",
                        "iters",
                        "steps_per_eval",
                        "save_every",
                        "seed",
                    )
                }
                training_anatomy["method"] = "QLoRA (LoRA over a 4-bit base in MLX-LM)"
                training_anatomy["approximate_effective_batch_sequences"] = (
                    training_config["batch_size"]
                    * training_config["grad_accumulation_steps"]
                )
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
                healthy and you have enough time and battery. Immediately before MLX-LM
                starts, the cell rechecks the content-addressed preparation manifest. A
                split that changed since preparation fails closed instead of becoming an
                unrecorded training-data change. The run also pins the expected base-model
                revision and every required model and dataset file—not merely paths from
                the YAML. Training writes into a fresh staging directory; only a zero-exit
                run with valid adapter outputs and durable evidence is published, with
                `training-manifest.json` acting as the success token. An exclusive
                per-adapter lock serializes training and publication, so two terminals
                cannot splice different generations together.
                """,
                "exercise",
            ),
            code(
                """
                from aai_local_finetuning.data import require_valid_manifest
                from aai_local_finetuning.training import run_lora

                RUN_TRAINING = False
                TRAINING_ITERATIONS = 10
                notebook_adapter = (
                    PROJECT_ROOT / "artifacts" / "notebook" / "adapters" / "bitext-smoke"
                )
                if RUN_TRAINING:
                    require_valid_manifest(
                        PROJECT_ROOT / "data" / "processed" / "bitext-v1"
                    )
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
                proves better frozen-test behavior or safer responses. One or ten
                iterations prove plumbing, not a trustworthy loss trend. A formal run
                should also record the selected checkpoint, adapter hash, base revision,
                configuration hash, seed, and whether examples were truncated at 512 tokens.
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
                ## Lock the decision contract before opening test

                These versioned **course defaults** require at least a 0.01 absolute
                macro-F1 gain over the strongest meaningful baseline, schema validity
                ≥ 0.98, response-policy compliance ≥ 0.95, and zero unsupported labels.
                They are fixed now, before any test row or result is loaded.

                Category accuracy, escalation accuracy, latency, input/output tokens,
                peak memory, and adapter size are still reported but are **observational,
                not gating**, because this learning project has not invented business
                budgets for them. A production owner must set risk-based non-regression
                and resource gates. This compact course also uses point estimates; a
                higher-stakes decision should add paired uncertainty analysis and, for
                training variance, repeated seeds.
                """,
                "decision-contract",
                "best-practices",
            ),
            code(
                """
                from aai_local_finetuning.evaluation import PromotionThresholds

                locked_thresholds = PromotionThresholds()
                locked_thresholds.model_dump(mode="json")
                """,
                "decision-contract",
            ),
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
                from aai_local_finetuning.training import (
                    TrainingManifestError,
                    recheck_training_snapshot,
                    require_valid_training_snapshot,
                    shared_adapter_lock,
                )

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
                ## Add the LoRA change and persist its evidence atomically

                Notebook probes never overwrite official evaluation artifacts. Filenames
                include `partial` unless all frozen examples were scored. Reports carry
                the evaluation fingerprint used to prove comparability later.

                The canonical adapter is separate from notebook smoke adapters. A weight
                file alone is not training lineage: the success manifest must also match
                the current adapter bytes, adapter configuration, full training YAML,
                effective settings, expected base-model revision and files, and every
                prepared training-data file. One shared adapter lock now covers validation,
                prediction, scoring, report writes, and the exact manifest copy. Training
                cannot replace the adapter during any part of that evidence chain. Missing
                or stale evidence keeps the later decision inconclusive.
                """),
            code("""
                evidence_dir = PROJECT_ROOT / "artifacts" / "notebook" / "evaluation"
                evidence_dir.mkdir(parents=True, exist_ok=True)
                scope = "full" if len(frozen_records) == FULL_FROZEN_COUNT else "partial"
                lineage_copy = (
                    evidence_dir
                    / f"{scope}-lora-change-training-manifest.json"
                )
                lora_prediction_path = (
                    evidence_dir / f"{scope}-lora-change-predictions.jsonl"
                )
                lora_report_path = evidence_dir / f"{scope}-lora-change-report.json"
                for stale_path in (
                    lineage_copy,
                    lora_prediction_path,
                    lora_report_path,
                ):
                    stale_path.unlink(missing_ok=True)

                for name, report in reports.items():
                    write_predictions_jsonl(
                        evidence_dir / f"{scope}-{name}-predictions.jsonl",
                        methods[name],
                    )
                    write_report_json(
                        evidence_dir / f"{scope}-{name}-report.json",
                        report,
                    )

                adapter_weights = settings.adapter_dir / "adapters.safetensors"
                adapter_snapshot = None
                if adapter_weights.is_file():
                    try:
                        with shared_adapter_lock(settings.adapter_dir):
                            adapter_snapshot = require_valid_training_snapshot(
                                settings.adapter_dir,
                                config_path=(
                                    PROJECT_ROOT
                                    / "configs"
                                    / "training"
                                    / "lora.yaml"
                                ),
                            )
                            lora_predictor = LocalMLXPredictor(
                                settings.model_dir,
                                adapter_path=settings.adapter_dir,
                            )
                            methods["lora-change"] = generate_support_predictions(
                                lora_predictor,
                                frozen_records,
                                strategy="strong",
                                train_records=splits.train,
                                max_tokens=96,
                            )
                            recheck_training_snapshot(adapter_snapshot)
                            reports["lora-change"] = evaluate_predictions(
                                frozen_records,
                                methods["lora-change"],
                                supported_intents=allowed_intents,
                            ).model_copy(
                                update={
                                    "training_manifest_sha256": (
                                        adapter_snapshot.manifest_sha256
                                    )
                                }
                            )
                            write_predictions_jsonl(
                                lora_prediction_path,
                                methods["lora-change"],
                            )
                            write_report_json(
                                lora_report_path,
                                reports["lora-change"],
                            )
                            lineage_copy.write_bytes(
                                adapter_snapshot.raw_manifest_bytes
                            )
                            recheck_training_snapshot(adapter_snapshot)
                    except (OSError, ValueError, TrainingManifestError) as error:
                        adapter_snapshot = None
                        methods.pop("lora-change", None)
                        reports.pop("lora-change", None)
                        for incomplete_path in (
                            lineage_copy,
                            lora_prediction_path,
                            lora_report_path,
                        ):
                            incomplete_path.unlink(missing_ok=True)
                        print(f"Canonical LoRA adapter ignored: {error}")
                else:
                    print("Canonical LoRA adapter absent; change evidence is incomplete.")

                display(pd.DataFrame(
                    [report_row(name, report) for name, report in reports.items()]
                ))
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
                from aai_local_finetuning.settings import (
                    PROJECT_ROOT,
                    load_settings,
                    sha256_file,
                )
                from aai_local_finetuning.tracking import configure_local_mlflow
                from aai_local_finetuning.training import (
                    TrainingManifest,
                    TrainingManifestError,
                    recheck_training_snapshot,
                    require_valid_training_snapshot,
                    shared_adapter_lock,
                )

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
                ## Understand a run before writing one

                An MLflow **experiment** groups related work. A **run** is one evidence
                envelope: parameters describe inputs and configuration, metrics record
                numeric results, tags make purpose searchable, and artifacts preserve
                files such as reports and decisions. The local SQLite store is the index;
                the artifact directory holds evidence files. This notebook writes one
                decision run only after it has assembled the assessment below.
                """),
            code("""
                tracking_contract = {
                    "experiment": settings.tracking.experiment,
                    "backend_store": settings.tracking.uri,
                    "artifact_root": settings.tracking.artifact_root,
                    "planned_run_purpose": "promotion_assessment",
                    "required_lineage": [
                        "evaluation fingerprint",
                        "model revision",
                        "adapter and configuration hashes",
                        "evaluator and policy versions",
                        "locked thresholds",
                        "reports and final decision",
                    ],
                }
                tracking_contract
                """),
            md(
                """
                ## Inventory full notebook reports

                Promotion requires all meaningful baselines and the LoRA change on the
                complete frozen set with identical evaluation fingerprints. The LoRA
                report must also carry the same training-manifest fingerprint that was
                validated before inference and is still current now. Partial reports,
                missing methods, or either kind of mismatch force `inconclusive`.
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
                lineage_path = report_dir / "full-lora-change-training-manifest.json"
                report_status
                """),
            md(
                """
                ## Apply the decision contract

                Defaults require schema validity ≥ 0.98, response-policy compliance ≥
                0.95, unsupported-intent rate = 0, and an absolute macro-F1 gain ≥ 0.01
                over the strongest meaningful baseline. These course defaults were shown
                and locked before notebook 07 opened test. Majority is retained as a floor
                but excluded from the meaningful-baseline competition. The point-estimate
                gate is suitable for this lab, not a substitute for risk-based thresholds
                and paired uncertainty in a higher-stakes decision.
                """,
                "what-to-notice",
            ),
            code("""
                thresholds = PromotionThresholds()
                thresholds.model_dump(mode="json")
                """),
            md(
                """
                ## Persist the assessment with its contract

                Now the notebook writes one MLflow run whose purpose is explicit. The
                assessment artifact is useful whether the decision is adopt, reject, or
                inconclusive. Re-running creates a new attempt with a new run ID instead
                of silently replacing prior evidence. A shared adapter lock covers report
                loading, lineage validation, the promotion calculation, and the MLflow
                decision artifact commit, so one assessment cannot mix adapter generations.
                """,
                "what-to-notice",
            ),
            code("""
                lineage_matches = False
                lineage_error = None
                current_snapshot = None
                current_manifest_sha256 = None
                loaded_reports = {}
                fingerprints = set()
                counts = set()
                complete_and_comparable = False

                with shared_adapter_lock(settings.adapter_dir):
                    try:
                        current_snapshot = require_valid_training_snapshot(
                            settings.adapter_dir,
                            config_path=(
                                PROJECT_ROOT
                                / "configs"
                                / "training"
                                / "lora.yaml"
                            ),
                        )
                        current_manifest = current_snapshot.manifest
                        recorded_manifest = TrainingManifest.model_validate_json(
                            lineage_path.read_text(encoding="utf-8")
                        )
                        current_manifest_sha256 = current_snapshot.manifest_sha256
                        lineage_matches = (
                            recorded_manifest == current_manifest
                            and sha256_file(lineage_path) == current_manifest_sha256
                        )
                        report_status["lora-training-lineage"] = lineage_matches
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
                                report.total_examples
                                for report in loaded_reports.values()
                            }
                            complete_and_comparable = (
                                len(fingerprints) == 1
                                and counts == {len(splits.test)}
                                and loaded_reports[
                                    "lora-change"
                                ].training_manifest_sha256
                                == current_manifest_sha256
                            )
                    except (OSError, ValueError, TrainingManifestError) as error:
                        report_status["lora-training-lineage"] = False
                        lineage_error = str(error)

                    if complete_and_comparable:
                        recheck_training_snapshot(current_snapshot)
                        assessment = decide_lora_promotion(
                            change_name="bitext-structured-output-lora-v1",
                            training_manifest_sha256=current_manifest_sha256,
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
                                "the LoRA report must match the current success manifest",
                            ],
                            "available_reports": report_status,
                            "lineage_error": lineage_error,
                        }

                    if current_snapshot is not None:
                        recheck_training_snapshot(current_snapshot)
                    with mlflow.start_run(
                        run_name="notebook-promotion-assessment"
                    ) as run:
                        mlflow.set_tags(
                            {
                                "run_purpose": "promotion_assessment",
                                "execution_mode": "offline_local",
                                "decision": str(assessment["decision"]),
                            }
                        )
                        mlflow.log_params(
                            {
                                f"threshold.{name}": value
                                for name, value in thresholds.model_dump(
                                    mode="json"
                                ).items()
                            }
                        )
                        mlflow.log_dict(assessment, "decision/assessment.json")
                        if complete_and_comparable:
                            mlflow.log_param(
                                "evaluation_fingerprint",
                                next(iter(fingerprints)),
                            )
                            mlflow.log_param(
                                "training_manifest_sha256",
                                current_manifest_sha256,
                            )
                            for name, value in loaded_reports[
                                "lora-change"
                            ].flat_metrics().items():
                                mlflow.log_metric(f"change.{name}", value)
                        decision_run_id = run.info.run_id
                    if current_snapshot is not None:
                        recheck_training_snapshot(current_snapshot)
                {"decision_run_id": decision_run_id, "assessment": assessment}
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
                Application-context families are assigned to one split *before* variants
                are expanded, so a new application name cannot disguise a duplicated
                scenario across train and test. The frozen test includes required slices
                and unseen combinations of known failures. Generation writes portable
                records and immutable hashes.
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
                ## Lock the frozen contract without inspecting test rows

                The portable-data renderer has already loaded and validated every split,
                including the frozen test, so it can serialize the model-ready files and
                bind their hashes. That mechanical validation is different from looking
                at examples or labels: here we inspect only the committed count, hashes,
                and required slice vocabulary. The deterministic engine is the *declared*
                correctness ceiling for rules it fully determines; notebook 10 measures
                that claim only after its model probe and optional training are finished.
                """,
                "what-to-notice",
            ),
            code("""
                test_artifact = next(
                    artifact
                    for artifact in split_manifest.artifacts
                    if artifact.split.value == "test"
                )
                {
                    "test_rows_inspected_or_displayed": False,
                    "frozen": split_manifest.frozen_test,
                    "record_count": test_artifact.record_count,
                    "sha256": test_artifact.sha256,
                    "example_ids_sha256": test_artifact.example_ids_sha256,
                    "required_slice_contract": list(REQUIRED_FROZEN_TEST_SLICES),
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

                You now have deterministic, versioned ground truth and a declared
                correctness ceiling ready for a locked measurement—not LLM-generated
                labels presented as facts. Test rows have not been opened.

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
        evidence="a validation mechanics probe, frozen ceiling check, and explicit architecture choice",
        cells=(
            intro(
                10,
                "Capstone: model versus deterministic and hybrid designs",
                minutes=55,
                prerequisites="09 — Capstone policy-derived ground truth",
                objectives=(
                    "Distinguish a one-record mechanics probe from comparative evidence.",
                    "Prove that a hybrid renderer cannot alter authoritative decisions.",
                    "Decide which behavior, if any, justifies a language model.",
                ),
                evidence="a validation mechanics probe, frozen ceiling check, and explicit architecture choice",
            ),
            OFFLINE_SETUP,
            md(
                """
                ## Rehearse the deterministic method on validation

                Test is still closed. The deterministic method should be exact on
                validation for fully specified rules because the same versioned policy
                engine generated the labels. This rehearses the evaluator; it is not yet
                the frozen ceiling measurement. A model must justify a different
                capability—such as bounded wording—rather than probabilistically
                duplicating already-computable policy decisions.
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
                validation_records = load_capstone_records(
                    source_dir / "validation.jsonl"
                )
                deterministic_validation_report = evaluate_capstone_predictions(
                    validation_records,
                    deterministic_capstone_predictions(validation_records),
                )
                deterministic_validation_report.aggregate.model_dump(mode="json")
                """),
            md("""
                ## One untouched-model validation probe — demo, not evidence

                This bounded **validation** example shows mechanics, not a winner and not
                a model-versus-policy comparison. The compact model contract asks for
                status and non-pass checks. One output has no useful uncertainty or slice
                coverage; a claimed comparison must score locked methods on identical
                records and fingerprints.
                """),
            code(
                """
                probe_record = validation_records[0]
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
                    "evidence_status": "mechanics demo only; do not rank methods",
                }
                """,
                "demo-not-evidence",
            ),
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
                ## Prove fallback—and understand what the type boundary cannot prove

                A renderer exception cannot weaken the policy decision: deterministic
                wording replaces it. The typed boundary also prevents the renderer from
                changing status or severity. It does **not** prove that arbitrary prose is
                truthful, so generated explanations still need their own groundedness and
                human-review evaluation.
                """,
                "what-to-notice",
            ),
            code("""
                def failing_renderer(_check):
                    raise RuntimeError("simulated renderer outage")


                fallback_review = build_hybrid_review(
                    probe_record.manifest,
                    renderer=failing_renderer,
                    renderer_name="simulated_failure",
                )
                {
                    "authoritative_review_unchanged": (
                        fallback_review.deterministic_review
                        == hybrid.deterministic_review
                    ),
                    "fallback_is_nonempty": all(
                        explanation.text for explanation in fallback_review.explanations
                    ),
                    "remaining_risk": (
                        "A nonempty generated explanation may still be misleading; "
                        "evaluate wording separately."
                    ),
                }
                """),
            md(
                """
                ## Optional capstone LoRA smoke

                Keep this disabled for Run All. When enabled it writes a notebook-specific
                adapter and leaves the canonical capstone change untouched. Its success
                evidence binds the expected base model and exact capstone training files;
                a smoke adapter cannot qualify as the canonical change. Falling loss still
                does not beat the deterministic ceiling.
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
                ## Lock methods, then open the capstone test once

                All model probing and optional training now precede this boundary. The
                deterministic ceiling is measured on the complete test. The expensive
                base-model comparison is opt-in, but when enabled it uses every identical
                test record so the two reports share a defensible scope. Leaving it off
                produces **missing model evidence**, not permission to infer a winner.
                """,
                "frozen-boundary",
                "what-to-notice",
            ),
            code(
                """
                test_records = load_capstone_records(source_dir / "test.jsonl")
                deterministic_report = evaluate_capstone_predictions(
                    test_records,
                    deterministic_capstone_predictions(test_records),
                )

                RUN_FROZEN_MODEL_COMPARISON = False
                model_frozen_report = None
                if RUN_FROZEN_MODEL_COMPARISON:
                    model_predictions = []
                    for record in test_records:
                        generated = predictor.generate(
                            [
                                {"role": "system", "content": CAPSTONE_SYSTEM_PROMPT},
                                {
                                    "role": "user",
                                    "content": json.dumps(
                                        record.manifest,
                                        separators=(",", ":"),
                                        sort_keys=True,
                                    ),
                                },
                            ],
                            max_tokens=160,
                        )
                        model_predictions.append(
                            CapstonePrediction(
                                example_id=record.example_id,
                                raw_text=generated.text,
                                latency_ms=generated.latency_ms,
                                output_tokens=generated.output_tokens,
                                peak_memory_mb=generated.peak_memory_mb,
                            )
                        )
                    model_frozen_report = evaluate_capstone_predictions(
                        test_records,
                        tuple(model_predictions),
                    )

                frozen_comparison = {
                    "records": len(test_records),
                    "deterministic_policy": (
                        deterministic_report.aggregate.model_dump(mode="json")
                    ),
                    "untouched_model": (
                        model_frozen_report.aggregate.model_dump(mode="json")
                        if model_frozen_report is not None
                        else "not run; comparative model evidence is absent"
                    ),
                    "comparison_complete": model_frozen_report is not None,
                }
                frozen_comparison
                """,
                "frozen-boundary",
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
                        "failure_handling": "fail closed to not-ready or review",
                        "evidence": "policy tests and frozen deterministic report",
                    },
                    {
                        "behavior": "external registry fact",
                        "owner": "authorized lookup",
                        "reason": "the fact is absent from the local manifest",
                        "failure_handling": "route to review; never assume false",
                        "evidence": "lookup provenance and authorization record",
                    },
                    {
                        "behavior": "residual risk acceptance",
                        "owner": "qualified human",
                        "reason": "risk appetite is a governed judgment, not a text prediction",
                        "failure_handling": "await an explicit recorded decision",
                        "evidence": "reviewer identity, rationale, and timestamp",
                    },
                    {
                        "behavior": "remediation wording",
                        "owner": "policy text or constrained tiny-model renderer",
                        "reason": "wording may vary without changing authority",
                        "failure_handling": "deterministic wording fallback",
                        "evidence": "groundedness, policy, latency, and fallback tests",
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
                    "source_modalities",
                    "model_input_modality",
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

                Choose one project. Every field carries an explicit `verified`, `unknown`,
                or `blocked` state. A plausible placeholder is not verified evidence. The
                gate stays false whenever rights, permitted use, redistribution, access,
                schema, or modality is unknown or blocked.
                """,
                "exercise",
            ),
            code(
                """
                selected_project = "invoice extraction"
                dataset_review = {
                    field: {
                        "state": "unknown",
                        "value": None,
                        "evidence": None,
                    }
                    for field in required_review_fields
                }
                dataset_review["title"]["value"] = "verify current title online"
                blocking_fields = (
                    "license",
                    "permitted_use",
                    "redistribution",
                    "formats",
                    "source_modalities",
                    "model_input_modality",
                    "columns",
                    "accessible",
                )
                suitable_for_lab = all(
                    dataset_review[field]["state"] == "verified"
                    and dataset_review[field]["value"] not in (None, "", [], {})
                    and dataset_review[field]["evidence"]
                    for field in blocking_fields
                )
                {
                    "project": selected_project,
                    "review": dataset_review,
                    "blocking_fields": blocking_fields,
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
                    "project": selected_project,
                    "task_contract": {
                        "input": "verified OCR text, not invoice image pixels",
                        "output": "strict optional invoice fields with normalized values",
                        "authority": "human-reviewed annotations from the verified source",
                    },
                    "split_risks": [
                        "same vendor template crossing splits",
                        "duplicate invoice or OCR variants crossing splits",
                    ],
                    "baselines": [
                        "null/empty-field sanity baseline",
                        "deterministic pattern-and-normalization extractor",
                        "untouched prompted text model",
                    ],
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
                    "resource_constraints": (
                        "verify context-length distribution, latency, and peak memory on "
                        "the prepared 24 GB laptop before training"
                    ),
                }
                assert evaluation_plan["project"] == selected_project
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


def _expanded_cells(notebook: Notebook) -> tuple[Cell, ...]:
    """Place explanation before execution and label shared setup consistently."""

    cells: list[Cell] = [notebook.cells[0], *pedagogy_cells(notebook)]
    for cell in notebook.cells[1:]:
        if cell is OFFLINE_SETUP:
            cells.append(SETUP_GUIDANCE)
        cells.append(cell)
    return tuple(cells)


def render(notebook: Notebook) -> None:
    primer = PRIMERS[notebook.order]
    cells = _expanded_cells(notebook)
    payload = {
        "cells": [
            _cell_payload(notebook, index, cell) for index, cell in enumerate(cells)
        ],
        "metadata": {
            "aai_curriculum": {
                "order": notebook.order,
                "stage": notebook.stage,
                "duration_minutes": notebook.duration,
                "prerequisites": list(notebook.prerequisites),
                "learner_evidence": notebook.evidence,
                "concepts_introduced": [term for term, _ in primer.terms],
                "practice_guidance_reviewed_on": PRACTICE_REVIEWED_ON,
                "pedagogical_structure": [
                    "why",
                    "terms",
                    "mental_model",
                    "running_example",
                    "decision_questions",
                    "best_practices",
                    "common_mistakes",
                    "practice",
                    "checkpoint",
                ],
            },
            "kernelspec": {
                "display_name": "AAI Local Fine-Tuning (offline)",
                "language": "python",
                "name": "aai-local-finetuning",
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
