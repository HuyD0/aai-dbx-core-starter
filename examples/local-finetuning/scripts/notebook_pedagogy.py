# ruff: noqa: E501
"""Beginner-first primers injected into the generated notebook curriculum.

The executable lessons live in ``render_notebooks.py``.  This module keeps the
conceptual layer in one reviewable place so that every notebook teaches the
vocabulary and decision process before asking a learner to run code.
"""

from __future__ import annotations

from dataclasses import dataclass

PRACTICE_REVIEWED_ON = "2026-08-01"


@dataclass(frozen=True)
class Primer:
    """The teaching context that must appear before a notebook's first code cell."""

    why: str
    terms: tuple[tuple[str, str], ...]
    mental_model: str
    decision_questions: tuple[str, ...]
    practices: tuple[str, ...]
    mistakes: tuple[str, ...]
    references: tuple[str, ...]


PRIMERS: dict[int, Primer] = {
    0: Primer(
        why=(
            "Fine-tuning can make a model look better in a demo while making the "
            "overall system less reliable. Before touching data or weights, you need "
            "a precise question, a comparison point, and a definition of acceptable "
            "behavior. This notebook establishes those boundaries and verifies that "
            "the plane-ready workspace actually contains the evidence-producing assets."
        ),
        terms=(
            (
                "large language model (LLM)",
                "a model trained to predict tokens; useful language behavior emerges "
                "from that prediction task, but its output is not automatically a fact "
                "or a policy decision",
            ),
            (
                "base model",
                "the unchanged model checkpoint on which prompting or an adapter is built",
            ),
            (
                "inference",
                "running a trained model to produce an output; no weights are learned",
            ),
            (
                "fine-tuning",
                "continuing training on task-specific examples so some learned parameters "
                "change",
            ),
            (
                "LoRA adapter",
                "a small set of learned low-rank updates used with a frozen base model; it "
                "is not a complete model by itself",
            ),
            (
                "structured output",
                "an answer that must obey a machine-checkable contract such as known JSON "
                "fields and allowed labels",
            ),
            (
                "baseline",
                "a reproducible existing approach that a proposed change must beat",
            ),
            (
                "train / validation / test",
                "examples used respectively to learn, choose among changes, and estimate "
                "the final chosen system's behavior on untouched data",
            ),
        ),
        mental_model=(
            "Treat model development as a controlled scientific comparison, not a talent "
            "show. Write the question, measure a baseline, make one named change, evaluate "
            "both under the same contract, and then decide. The lifecycle is **baseline → "
            "change → result → decision**. A successful training command proves only that "
            "training ran; it does not prove that the change helped."
        ),
        decision_questions=(
            "What user or system decision will this model output support?",
            "Which behavior must improve, and which safety or format properties may not regress?",
            "What unchanged baseline and untouched examples make the comparison credible?",
            "What artifact would let another person reproduce or challenge the conclusion?",
        ),
        practices=(
            "**State the contract before implementation.** Name the input, allowed output, "
            "quality metrics, safety constraints, and promotion thresholds up front.",
            "**Change one explainable thing at a time.** Prompt changes, data changes, base-model "
            "changes, and adapter training are separate experimental changes.",
            "**Measure layers separately.** Classification quality, JSON validity, response-policy "
            "compliance, latency, and memory answer different questions and should not be collapsed.",
            "**Keep lineage.** Record exact data, model, code, configuration, and evaluation "
            "fingerprints so that a score has meaning later.",
            "**Start with the smallest useful experiment.** A smoke run catches broken plumbing; "
            "a full run is justified only after the evidence path works end to end.",
        ),
        mistakes=(
            "**Equating lower training loss with product improvement.** Loss measures the training "
            "objective, not generalization, output validity, safety, or user value.",
            "**Judging a few attractive examples by eye.** Hand-picked outputs hide base rates and "
            "failure slices; use a fixed evaluation set and bounded error review.",
            "**Using the test set while designing.** That turns the test into development data and "
            "makes the reported result optimistic.",
            "**Assuming a public dataset is production-ready.** Availability says nothing by itself "
            "about license scope, consent, representativeness, or fitness for the intended use.",
        ),
        references=(
            "[Risk guidance] NIST AI RMF Generative AI Profile",
            "[Tool guidance] MLflow evaluation-driven development overview",
        ),
    ),
    1: Primer(
        why=(
            "A model result is unusable if nobody can establish where its data came from or "
            "whether that use was permitted. Provenance and license review happen before "
            "exploration because inspection, redistribution, training, and commercial use can "
            "have different permissions. This is a governance gate, not paperwork added after "
            "a model succeeds."
        ),
        terms=(
            ("dataset", "a defined collection of records used for a stated purpose"),
            (
                "provenance",
                "the chain of custody describing the source, acquisition method, version, and "
                "transformations of data",
            ),
            (
                "license",
                "the legal terms under which a rights holder permits specified uses; it is not "
                "a quality or ethics certificate",
            ),
            (
                "permitted use",
                "a concrete activity—such as local study, modification, redistribution, or model "
                "training—that the relevant terms and policy allow",
            ),
            (
                "dataset card",
                "documentation of a dataset's contents, intended uses, origin, limitations, "
                "license metadata, and known risks",
            ),
            (
                "immutable raw input",
                "the acquired bytes preserved without hand edits so later transformations remain auditable",
            ),
            (
                "SHA-256 fingerprint",
                "a deterministic digest of bytes used to detect change; it identifies content but "
                "does not prove that the content is safe, correct, or lawful",
            ),
        ),
        mental_model=(
            "Think like an evidence custodian receiving a sealed package. First record who supplied "
            "it, under what terms, on what date, and with what fingerprint. Then keep the package "
            "unchanged and derive working copies through repeatable transformations. Apply two "
            "independent gates: **may we use it?** and **is it fit for this purpose?** Passing one "
            "never implies the other."
        ),
        decision_questions=(
            "Can the exact bytes and source version be identified again?",
            "Do the verified terms cover this activity and this audience, including redistribution?",
            "Which statements are source facts, which are interpretations, and who approved them?",
            "If a term is missing or ambiguous, should the workflow stop or escalate to an owner?",
        ),
        practices=(
            "**Verify the current primary source.** Save the source URL or identifier, observed date, "
            "version, and license text or authoritative reference before use.",
            "**Hash the acquired bytes.** A content fingerprint makes silent replacement visible and "
            "connects every derivative artifact to a specific input.",
            "**Never edit raw data in place.** Normalize and filter into a separate interim or processed "
            "layer with recorded code and configuration.",
            "**Document intended and out-of-scope uses.** A useful dataset card covers limitations, "
            "sensitive content, collection context, and known representational gaps—not just columns.",
            "**Fail closed on unclear rights.** Record `not verified` rather than inferring permission "
            "from public visibility or from a repository filename.",
        ),
        mistakes=(
            "**Public means free for every use.** Public access and legal permission are different facts.",
            "**A license field settles all rights.** Hosted content can include third-party material, "
            "privacy obligations, or terms that require separate review.",
            "**The filename is a version.** Mutable files can keep the same name; preserve a digest and "
            "source revision.",
            "**License review proves responsible use.** It does not establish consent, fairness, data "
            "quality, security, or suitability for deployment.",
        ),
        references=(
            "[Tool guidance] Hugging Face Dataset Cards documentation",
            "[Risk guidance] NIST AI RMF Generative AI Profile",
        ),
    ),
    2: Primer(
        why=(
            "Training code cannot repair an unknown data contract. Missing fields, label imbalance, "
            "duplicates, long outliers, and sensitive strings all shape what a model can learn and "
            "what an evaluation can honestly claim. Exploration should therefore produce aggregate, "
            "repeatable evidence before it exposes raw examples or launches training."
        ),
        terms=(
            (
                "schema",
                "the explicit contract for required fields, types, allowed values, and constraints",
            ),
            (
                "label",
                "the expected answer or category attached to a supervised example",
            ),
            (
                "label distribution",
                "the count or proportion of examples in each class; severe imbalance can make accuracy misleading",
            ),
            (
                "duplicate",
                "a record whose normalized learning content is identical to another record",
            ),
            (
                "near duplicate",
                "records that differ superficially but carry substantially the same learning signal",
            ),
            (
                "token",
                "a unit produced by the model's tokenizer; it is not necessarily a word or character",
            ),
            (
                "sensitive-data indicator",
                "a conservative pattern or classifier that flags possible sensitive content for review; "
                "it is not proof that content is or is not sensitive",
            ),
            (
                "aggregate",
                "a count, distribution, or summary that reduces unnecessary exposure of individual rows",
            ),
        ),
        mental_model=(
            "Treat the dataset as the first model: it already encodes assumptions about what inputs "
            "exist, which answers count as correct, and which groups are common. Audit it as a system "
            "with inputs, invariants, and failure modes. Begin with schemas and aggregates, drill into "
            "a small masked sample only when a summary reveals something that needs explanation."
        ),
        decision_questions=(
            "What properties are measured directly, and what properties are only heuristic signals?",
            "Which rare labels, lengths, languages, or sources could disappear in a single average?",
            "Could two records carry the same learning signal despite different punctuation or IDs?",
            "What is the minimum row-level content needed to investigate a problem safely?",
        ),
        practices=(
            "**Validate the schema before statistics.** Invalid records should be counted and quarantined, "
            "not silently coerced into plausible values.",
            "**Inspect distributions and slices.** Report counts by label, length, language/source where "
            "verified, and policy-relevant subgroup rather than relying on a global row count.",
            "**Use the exact model tokenizer for final sizing.** Character or whitespace counts are useful "
            "proxies, but only the selected tokenizer determines context and training token lengths.",
            "**Detect duplicates before splitting.** Normalize with a versioned rule and group related "
            "records so they cannot leak across evidence boundaries.",
            "**Prefer aggregate-first privacy.** Mask bounded previews, never print whole datasets, and "
            "treat zero pattern matches as `none detected by this rule`, not a privacy guarantee.",
        ),
        mistakes=(
            "**Printing random raw rows as exploration.** It increases exposure without first establishing "
            "what question the preview answers.",
            "**Treating a heuristic as ground truth.** Language, difficulty, toxicity, or sensitive-data "
            "rules have known blind spots and need explicit labels such as `detected` or `estimated`.",
            "**Deduplicating after the split.** A duplicate can already have crossed into validation or test.",
            "**Reporting only the average.** A healthy average can hide a missing class, an extreme tail, or "
            "a severe minority-slice failure.",
        ),
        references=(
            "[Tool guidance] Hugging Face Dataset Cards documentation",
            "[Risk guidance] NIST AI RMF Generative AI Profile",
        ),
    ),
    3: Primer(
        why=(
            "Evaluation is credible only when development examples and final-exam examples are genuinely "
            "independent. Support datasets often contain templates, paraphrases, repeated conversations, "
            "or records from the same source. A random row split can place siblings on both sides and reward "
            "memorization. Stable IDs, grouping, manifests, and leakage gates create evidence boundaries."
        ),
        terms=(
            (
                "example ID",
                "a stable identifier derived from learning-relevant content or a governed source key",
            ),
            (
                "split",
                "an explicit assignment of records to train, validation, or test",
            ),
            (
                "group",
                "records that must travel together because they share a source, template, subject, or near-duplicate signal",
            ),
            (
                "stratification",
                "preserving important class proportions across splits when the grouping constraints permit it",
            ),
            (
                "data leakage",
                "information crossing an evidence boundary in a way that makes measured performance unrealistically easy",
            ),
            (
                "frozen test set",
                "a held-out split whose examples and results are not used to design the same candidate change",
            ),
            (
                "manifest",
                "a machine-readable record of split membership, counts, configuration, and fingerprints",
            ),
            (
                "fingerprint",
                "a deterministic digest used to verify that an artifact or evaluation input has not silently changed",
            ),
        ),
        mental_model=(
            "Picture three locked rooms. The training room may teach. The validation room may help choose "
            "between already-defined options. The test room opens only after the choice is locked. Related "
            "examples are a family and must enter the same room. A manifest is the signed seating chart; a "
            "fingerprint is the tamper seal."
        ),
        decision_questions=(
            "What makes two records related enough that seeing one helps predict the other?",
            "Which grouping key should take priority over perfect label balance?",
            "Has any test content, label, error, or result influenced prompts, examples, thresholds, or training?",
            "Can the exact split be reconstructed and checked without trusting filenames?",
        ),
        practices=(
            "**Group before assigning splits.** Exact duplicates, near duplicates, shared templates, users, "
            "documents, or time windows should remain together when they can transmit learning signal.",
            "**Use stable IDs and versioned configuration.** Record normalization rules, group logic, seed, "
            "target proportions, and the code/data version that produced the manifest.",
            "**Automate leakage gates.** Assert disjoint IDs and group keys, verify file fingerprints, and fail "
            "the pipeline when a protected boundary is violated.",
            "**Use validation for iteration and test for the final estimate.** After looking at test errors, "
            "the next fix needs a newly defined change and a fresh untouched evaluation version.",
            "**Prefer honest imbalance to broken independence.** Group constraints can prevent exact class "
            "ratios; document the resulting slices instead of splitting a related group.",
        ),
        mistakes=(
            "**Randomly splitting rows with a fixed seed.** Reproducibility does not prevent related content "
            "from leaking across splits.",
            "**Using test examples as few-shot demonstrations.** Their labels have now entered the system being evaluated.",
            "**Tuning after reading test errors.** The original test result becomes development feedback, not an untouched estimate.",
            "**Claiming `no leakage` from one detector.** A passed check means no violation was detected under "
            "the implemented definition; undocumented semantic relationships may remain.",
        ),
        references=(
            "[Tool guidance] scikit-learn common pitfalls: data leakage",
            "[Tool guidance] scikit-learn grouped splitting documentation",
        ),
    ),
    4: Primer(
        why=(
            "A fine-tuned model is not useful merely because it produces non-random answers. It must beat "
            "simple approaches that are cheaper, faster, easier to inspect, and often surprisingly strong. "
            "Deterministic baselines also exercise the exact output and evaluation contracts before model "
            "inference adds cost or uncertainty."
        ),
        terms=(
            ("prediction", "the answer emitted by a method for one input"),
            (
                "majority baseline",
                "a sanity-floor classifier that always predicts the most common training label",
            ),
            (
                "deterministic baseline",
                "a repeatable method such as rules or retrieval whose output is fixed for the same input",
            ),
            (
                "precision",
                "among items predicted as a class, the fraction whose label is actually that class",
            ),
            (
                "recall",
                "among items that truly belong to a class, the fraction the method finds",
            ),
            (
                "F1",
                "the harmonic mean of precision and recall; it is high only when both are reasonably high",
            ),
            (
                "macro F1",
                "the unweighted mean of per-class F1, giving each class equal influence",
            ),
            (
                "weighted F1",
                "the mean of per-class F1 weighted by class frequency, so common classes influence it more",
            ),
            (
                "schema validity",
                "whether the output parses and satisfies all required types, fields, and allowed values",
            ),
        ),
        mental_model=(
            "Build a minimum-bar ladder. The majority baseline is the floor: it catches broken metrics and "
            "class imbalance. A task-aware deterministic baseline is the meaningful rung: it represents what "
            "an inexpensive inspectable system can already do. Prompted and fine-tuned models must be compared "
            "against the strongest relevant rung with the same records and evaluator."
        ),
        decision_questions=(
            "What simple method could solve a large fraction of this task without a generative model?",
            "Which error is more costly for each class: a false positive or a false negative?",
            "Would an output with the right label but invalid JSON be usable by the downstream system?",
            "Do global metrics conceal a rare but important intent or policy failure?",
        ),
        practices=(
            "**Fit or configure baselines from training data only.** Validation measures choices; it is not a "
            "source for silently extending rules after seeing results.",
            "**Use the same evaluation harness for every method.** Identical IDs, labels, parsers, policy "
            "checks, and timing boundaries make comparisons meaningful.",
            "**Report complementary metrics.** Pair accuracy with macro and weighted F1, per-label counts, "
            "schema validity, response-policy compliance, and bounded errors.",
            "**Keep unsupported inputs explicit.** A known abstain or unsupported intent is preferable to "
            "inventing a confident in-scope answer.",
            "**Retain the simple winner.** If a rule system meets the contract, a model must justify added "
            "latency, cost, nondeterminism, and governance burden—not merely tie its score.",
        ),
        mistakes=(
            "**Reporting only accuracy.** A majority class can dominate it while minority classes fail completely.",
            "**Counting invalid JSON as only a wrong class.** Format validity is an independent operational contract.",
            "**Choosing an intentionally weak baseline.** Beating a straw man provides little evidence for adoption.",
            "**Editing rules after each validation result without versioning.** That is still model selection and "
            "must be recorded as a new change before final test evaluation.",
        ),
        references=(
            "[Tool guidance] scikit-learn metrics and scoring guide",
            "[Tool guidance] scikit-learn dummy estimators as baseline values",
        ),
    ),
    5: Primer(
        why=(
            "Prompting changes model behavior without learning new weights, so it is usually cheaper and "
            "faster to test than fine-tuning. A strong prompt baseline tells you whether the base model already "
            "has the needed capability and whether the remaining gap is about instructions, examples, domain "
            "knowledge, or consistency. Prompt text and demonstrations are versioned system components."
        ),
        terms=(
            (
                "prompt",
                "the instructions and context supplied to a model for one inference",
            ),
            (
                "system message",
                "high-level behavior instructions supplied separately from user content in chat-style models",
            ),
            (
                "zero-shot",
                "asking for a task without including solved examples in that request",
            ),
            (
                "few-shot",
                "including a small number of solved demonstrations in the request; no weights are updated",
            ),
            (
                "demonstration",
                "an example input and desired output placed in prompt context",
            ),
            (
                "prompt baseline",
                "a versioned prompt strategy evaluated with unchanged model weights",
            ),
            (
                "decoding",
                "the procedure and settings used to select output tokens from model scores",
            ),
            (
                "output budget",
                "the maximum number of tokens the model is allowed to generate",
            ),
        ),
        mental_model=(
            "A prompt is a temporary program interpreted by a probabilistic runtime. Compare prompt variants "
            "like an A/B test: same base checkpoint, same inputs, same decoding settings, same parser and scorer; "
            "only the named prompt strategy changes. Basic, constrained, and few-shot prompts form another "
            "minimum-bar ladder before training weights."
        ),
        decision_questions=(
            "Is the failure caused by missing task instructions, missing examples, missing knowledge, or unstable generation?",
            "Can the output space be made smaller and machine-checkable?",
            "Were all demonstrations selected without looking at frozen test examples or labels?",
            "Does a prompt improvement survive every relevant slice rather than a few showcased inputs?",
        ),
        practices=(
            "**Version the complete prompt contract.** Record system/user templates, label vocabulary, examples, "
            "decoding values, parser, base revision, and output budget.",
            "**Constrain outputs explicitly.** State allowed labels and schema, request only required fields, and "
            "validate rather than trusting prose assurances.",
            "**Source demonstrations from training data.** Select them with a documented rule and keep validation "
            "for choosing among prompt variants.",
            "**Make controlled comparisons.** Reuse inputs and evaluator; record any model, quantization, prompt, "
            "or inference-setting change separately.",
            "**Prefer the simplest prompt that meets the gate.** Longer context consumes memory and latency and can "
            "introduce contradictory instructions or accidental leakage.",
        ),
        mistakes=(
            "**Cherry-picking impressive outputs.** A qualitative example illustrates behavior but cannot estimate a rate.",
            "**Using test examples as demonstrations.** That teaches the system from the exam and invalidates the comparison.",
            "**Changing several variables at once.** A new model plus new prompt plus different decoding cannot reveal "
            "which change caused the result.",
            "**Assuming few-shot must be better.** Poor, unrepresentative, or conflicting examples can reduce quality; measure it.",
        ),
        references=(
            "[Tool guidance] MLflow prompt evaluation documentation",
            "[Tool guidance] MLflow evaluation-driven development overview",
        ),
    ),
    6: Primer(
        why=(
            "Full fine-tuning changes every selected model weight and is often unnecessary for a narrow local "
            "task. LoRA learns a much smaller update while keeping the base model frozen, which reduces trainable "
            "parameters and makes adapters portable. It still creates a new model behavior that needs exact "
            "lineage, validation, resource measurement, and frozen evaluation."
        ),
        terms=(
            (
                "parameter / weight",
                "a learned numeric value used by the neural network to transform inputs",
            ),
            (
                "gradient",
                "the direction and magnitude by which optimization proposes changing trainable parameters",
            ),
            (
                "loss",
                "the numerical training objective minimized over examples; it is a proxy, not the product decision",
            ),
            (
                "iteration",
                "one optimizer update; it may consume one batch or an accumulated set of micro-batches",
            ),
            (
                "batch",
                "examples processed together before an optimizer update or gradient-accumulation step",
            ),
            (
                "quantization",
                "representing model values with lower precision to reduce memory and sometimes compute cost",
            ),
            (
                "LoRA rank",
                "the size of the low-dimensional learned update; larger is more expressive but trains more parameters",
            ),
            (
                "adapter",
                "the learned LoRA configuration and weights that must be paired with the compatible base checkpoint",
            ),
            (
                "checkpoint",
                "a saved training state or adapter snapshot from a particular step",
            ),
            (
                "overfitting",
                "improving on training examples while generalization to independent examples stagnates or worsens",
            ),
        ),
        mental_model=(
            "Imagine the frozen base model as a large reference book and LoRA as a small transparent correction "
            "sheet placed over selected pages. Training writes only the correction sheet. Inference combines both. "
            "The sheet is useful only with the exact compatible book, and a neater-looking sheet (lower loss) still "
            "has to pass the same independent exam as every other change."
        ),
        decision_questions=(
            "What measured gap remains after the strongest deterministic and prompt baselines?",
            "Which base revision, quantization, layers, rank, and training data define this adapter?",
            "Do validation loss and task metrics suggest useful learning or overfitting?",
            "Is the added adapter operational burden justified by a frozen-evaluation improvement?",
        ),
        practices=(
            "**Pin the exact base revision and training configuration.** An adapter is not reproducible or safely "
            "loadable without its model, tokenizer, quantization, data, seed, and library versions.",
            "**Run a smoke train first.** Prove parsing, batching, checkpoint paths, validation, and artifact capture "
            "with a few iterations before committing time and battery to a full run.",
            "**Save adapters separately and immutably.** Use run-specific paths and never overwrite the canonical "
            "candidate from an exploratory notebook cell.",
            "**Track training and validation evidence.** Log losses, steps, effective batch configuration, duration, "
            "memory context, and checkpoints; then use task evaluation for the actual decision.",
            "**Name quantized methods precisely.** In MLX-LM, training a quantized model uses its QLoRA path; do not "
            "use `QLoRA` as a generic label for every low-bit adapter workflow in other tools.",
        ),
        mistakes=(
            "**Calling lower training loss a win.** It does not establish generalization, parseability, safety, or usefulness.",
            "**Silently changing the base model.** Adapters are coupled to architecture and revision; mismatches can fail or mislead.",
            "**Full-tuning by default on constrained hardware.** It adds memory, storage, and catastrophic-forgetting risk without "
            "first proving that a parameter-efficient change is insufficient.",
            "**Evaluating on training examples.** That measures memorization opportunity, not behavior on independent inputs.",
        ),
        references=(
            "[Tool guidance] Apple MLX-LM fine-tuning with LoRA or QLoRA",
            "[Tool guidance] Hugging Face PEFT LoRA conceptual guide",
        ),
    ),
    7: Primer(
        why=(
            "This is the first notebook allowed to open the frozen test set. Its job is not to rescue the adapter; "
            "it is to estimate how the already-locked methods behave on independent examples. Comparable evaluation "
            "requires the same records, contracts, inference settings, and evaluator, plus enough slice and error "
            "evidence to explain what a global score hides."
        ),
        terms=(
            (
                "frozen evaluation",
                "a measurement run whose data, methods, parser, scorers, and thresholds were locked before results",
            ),
            (
                "evaluation fingerprint",
                "a digest of the exact records and evaluation contract used for a result",
            ),
            (
                "slice",
                "a meaningful subset such as an intent, length band, source, or risk category",
            ),
            (
                "error taxonomy",
                "a versioned set of failure categories used to summarize errors consistently",
            ),
            (
                "latency",
                "elapsed time for a defined inference boundary; hardware, warm-up, and batch context affect it",
            ),
            (
                "p95 latency",
                "a value at or above approximately 95 percent of measured latencies, highlighting the slow tail",
            ),
            (
                "peak RSS",
                "the process resident-memory high-water mark; useful context, not a universal GPU or system-memory measure",
            ),
            (
                "comparability",
                "the condition that differences in results can reasonably be attributed to the named method change",
            ),
        ),
        mental_model=(
            "Treat frozen evaluation as the final exam after the student and grading rubric are locked. The test result "
            "may support adoption, rejection, or an inconclusive decision. It may also inspire the *next* experiment, "
            "but changing the current prompt, adapter, policy, or threshold in response means the old test has become "
            "development feedback and a fresh exam is required."
        ),
        decision_questions=(
            "Are every method and evaluator consuming the identical ordered example IDs?",
            "Which metric maps to each real failure cost, and which risks need hard gates?",
            "Do slice counts support the interpretation, or is an apparently perfect slice tiny?",
            "Were timing, token, and memory boundaries measured consistently on documented hardware?",
        ),
        practices=(
            "**Lock methods and contracts before opening test.** Freeze adapter path, prompts, response policy, "
            "parsers, label vocabulary, decoding, metrics, and thresholds.",
            "**Report a metric portfolio.** Include classification metrics, per-label results and counts, schema "
            "validity, policy compliance, latency, output tokens, and carefully scoped memory measurements.",
            "**Pair aggregates with bounded errors.** Use a documented taxonomy and masked samples so reviewers can "
            "understand failures without turning evaluation into an uncontrolled data dump.",
            "**Verify fingerprints before comparison.** Partial smoke results and full frozen results are different "
            "evidence and must not share a comparison table without explicit qualification.",
            "**Preserve uncertainty.** Small slices, missing artifacts, incomparable runs, or unstable execution should "
            "lead to an inconclusive decision rather than an overstated win.",
        ),
        mistakes=(
            "**Fixing the candidate after seeing test errors and rerunning the same test.** This optimizes to the exam.",
            "**Letting one average hide failures.** Macro/weighted metrics and slice tables answer different questions.",
            "**Interpreting perfect performance on a tiny slice as certainty.** Always display the denominator and limitations.",
            "**Quoting resource numbers without scope.** Peak RSS is not GPU peak memory, and timings from different "
            "machines or warm-up states are not directly comparable.",
        ),
        references=(
            "[Tool guidance] MLflow evaluation datasets and regression-test guidance",
            "[Tool guidance] scikit-learn metrics and scoring guide",
        ),
    ),
    8: Primer(
        why=(
            "Metrics printed in a notebook are easy to lose, mislabel, or compare against the wrong data. Experiment "
            "tracking connects parameters, metrics, datasets, and artifacts into reviewable runs. Promotion then applies "
            "predeclared gates to comparable evidence. MLflow stores evidence; it does not decide whether the evidence is "
            "complete or whether the risk trade-off is acceptable."
        ),
        terms=(
            (
                "experiment",
                "an organized collection of related runs answering a development question",
            ),
            (
                "run",
                "one recorded execution with a defined method, inputs, parameters, metrics, tags, and artifacts",
            ),
            (
                "parameter",
                "a configuration value such as prompt version or LoRA rank, usually treated as an input to the run",
            ),
            (
                "metric",
                "a numeric measurement such as macro F1, validity rate, or p95 latency",
            ),
            (
                "artifact",
                "a file attached to evidence, such as a manifest, error table, configuration, or adapter metadata",
            ),
            (
                "lineage",
                "the traceable relationship among source data, processed data, code, configuration, model, and result",
            ),
            (
                "promotion gate",
                "a rule declared before evaluation that maps complete evidence to an allowed decision",
            ),
            (
                "adopt / reject / inconclusive",
                "use the change, decline it, or state that the available evidence cannot support either conclusion",
            ),
        ),
        mental_model=(
            "An experiment run is a lab notebook entry, not a trophy cabinet. A valid decision record joins four things: "
            "the baseline, the precisely named change, a comparable result, and the rule used to decide. Gates should work "
            "like compiled policy: the same evidence always produces the same decision, and missing evidence fails closed "
            "to `inconclusive`."
        ),
        decision_questions=(
            "Can a reviewer trace every result to exact data, method, code, and configuration fingerprints?",
            "Were gates declared before the frozen result was visible?",
            "Is the comparator the strongest meaningful baseline on the same evaluation fingerprint?",
            "Which required artifact or slice would make the decision inconclusive if absent?",
        ),
        practices=(
            "**Log inputs as lineage, not only filenames.** Record dataset source, schema/profile where appropriate, "
            "digest, split manifest, and evaluation fingerprint.",
            "**Keep run roles explicit.** Separate smoke, baseline, candidate, frozen evaluation, and remediation runs so "
            "partial evidence cannot masquerade as final evidence.",
            "**Predeclare multi-dimensional gates.** Require quality improvement while protecting schema validity, "
            "policy compliance, and operational limits; do not optimize one metric at all costs.",
            "**Separate practical significance from statistical uncertainty.** Declare a minimum useful gain, and "
            "for higher-stakes decisions estimate paired uncertainty on the same records plus training variance across seeds.",
            "**Store decision evidence as immutable artifacts.** Include the comparison table, gate configuration, "
            "failure reasons, and the selected decision vocabulary.",
            "**Use local tracking honestly.** A local SQLite-backed store is excellent offline evidence, but team access, "
            "durability, access control, and retention require a governed shared deployment later.",
        ),
        mistakes=(
            "**Metric shopping.** Selecting whichever score improved after results defeats predeclared evaluation intent.",
            "**Moving thresholds after the run.** That converts a gate into post-hoc justification and requires a new experiment.",
            "**Treating MLflow as proof by itself.** Tracking preserves what was logged; it cannot verify omitted inputs or bad methodology.",
            "**Comparing smoke and full runs as peers.** Different record sets, methods, or fingerprints are intentionally incomparable.",
        ),
        references=(
            "[Tool guidance] MLflow experiment tracking documentation",
            "[Tool guidance] MLflow dataset tracking and lineage documentation",
        ),
    ),
    9: Primer(
        why=(
            "The capstone models a policy-driven document review. Here the correct answer depends on an explicit "
            "policy and known synthetic facts, so ground truth can be generated deterministically. That is different "
            "from asking an LLM to invent labels or assuming a convenient public dataset represents the domain. The "
            "quality ceiling is set by the authority of the labels, not by the number of rows."
        ),
        terms=(
            (
                "ground truth",
                "the authoritative expected value used for evaluation, with a documented source and limitations",
            ),
            (
                "policy engine",
                "deterministic code that applies versioned policy rules to validated facts",
            ),
            (
                "external lookup",
                "a governed source outside the document, such as an approved vendor or jurisdiction registry",
            ),
            (
                "human judgment",
                "a decision supplied by a qualified reviewer when rules or available facts do not determine the answer",
            ),
            (
                "synthetic data",
                "artificially generated examples whose construction process and assumptions are known",
            ),
            (
                "controlled violation",
                "a deliberately introduced, labeled rule breach used to test coverage",
            ),
            (
                "provenance per label",
                "metadata stating which rule, fact source, generator version, and seed produced an expected answer",
            ),
            (
                "accuracy ceiling",
                "the maximum defensible performance given the available inputs and authority boundaries",
            ),
        ),
        mental_model=(
            "Draw an authority map before drawing a model architecture. Some facts are present in the document and can "
            "be validated. Some come only from an external registry. Some decisions are deterministic policy. Others "
            "require human judgment. Generate or label only what the chosen authority can actually know; mark everything "
            "else as unavailable or requiring review."
        ),
        decision_questions=(
            "For each target field, what is the authoritative source of truth?",
            "Can the answer be derived from provided facts and versioned policy without subjective interpretation?",
            "Which interactions and rare policy violations must the generated splits cover?",
            "What uncertainty or missing external fact should route to a human instead of receiving an invented label?",
        ),
        practices=(
            "**Generate critical labels from deterministic rules.** The LLM may help with non-authoritative language, "
            "but it must not create the policy truth used to grade itself.",
            "**Version the generator, policy, schema, seed, and rule catalog.** Reproducibility requires more than saving "
            "the final JSONL files.",
            "**Design coverage intentionally.** Include compliant cases, each rule violation, boundary values, missing "
            "facts, and interactions; publish counts by rule and split.",
            "**Hold out combinations, not merely rows.** Keep related templates and selected rule interactions out of "
            "training so evaluation tests composition rather than memorized wording.",
            "**Retain a human escalation class.** Unknown or subjective situations should remain explicit instead of being "
            "forced into confident binary labels.",
        ),
        mistakes=(
            "**Asking an LLM to generate both examples and authoritative labels.** Correlated mistakes can create a convincing but circular benchmark.",
            "**Treating a nearby public dataset as domain ground truth.** Similar columns do not establish the same policy, jurisdiction, or intended use.",
            "**Mistaking synthetic volume for validity.** Thousands of rows from one narrow generator can repeat the same assumptions.",
            "**Skipping rule-coverage review.** Deterministic generation can still encode an incomplete or incorrect policy model.",
        ),
        references=(
            "[Risk guidance] NIST AI RMF Generative AI Profile",
            "[Tool guidance] Hugging Face Dataset Cards documentation",
        ),
    ),
    10: Primer(
        why=(
            "Not every task that contains text should be delegated end to end to a generative model. In policy review, "
            "rules can preserve authoritative decisions while a model improves wording or extracts bounded candidates. "
            "A hybrid design makes that authority boundary explicit and gives uncertain or failed model output a "
            "deterministic fallback. Architecture is part of the evaluation question."
        ),
        terms=(
            (
                "deterministic system",
                "a system that produces the same result for the same validated input and versioned rules",
            ),
            (
                "generative model",
                "a probabilistic model that produces sequences and can vary or create unsupported content",
            ),
            (
                "hybrid architecture",
                "a system that assigns different responsibilities to rules, models, external systems, and humans",
            ),
            (
                "authority boundary",
                "the explicit line defining which component is allowed to decide or modify each field",
            ),
            (
                "renderer",
                "a component that converts authoritative structured findings into user-facing wording",
            ),
            (
                "normalization",
                "mapping varied model text into a validated canonical form without changing authoritative meaning",
            ),
            (
                "fallback",
                "a safe alternate behavior used when the model is unavailable, invalid, uncertain, or outside scope",
            ),
            (
                "fail closed",
                "defaulting to a safe non-approval or human-review state when required evidence is missing",
            ),
        ),
        mental_model=(
            "Use the rule **rules decide; models explain** whenever truth is computable from governed facts. Think of the "
            "model as an untrusted assistant behind a typed interface: it may propose wording or bounded extractions, but "
            "validated code owns authoritative fields. Any model output that crosses that boundary is rejected, normalized, "
            "or routed to review."
        ),
        decision_questions=(
            "Which fields are authoritative, and which component has permission to set each one?",
            "What capability does the model add that deterministic code cannot provide adequately?",
            "Can invalid, unavailable, or contradictory model output fall back without changing the decision?",
            "Does the measured quality gain justify latency, cost, nondeterminism, privacy, and monitoring obligations?",
        ),
        practices=(
            "**Use the least-necessary model role.** Keep computable policy in rules; use a model only for measured language "
            "or extraction capabilities that benefit from it.",
            "**Constrain and validate the interface.** Allow-list fields and values, forbid model mutation of authoritative "
            "decisions, and test adversarial or malformed outputs.",
            "**Provide a deterministic fallback.** A model outage, timeout, parse failure, or low-confidence result should "
            "have a documented safe path.",
            "**Evaluate each layer and the whole system.** Score rule correctness, model contribution, end-to-end contract, "
            "fallback rate, latency, and human-escalation quality separately.",
            "**Keep provenance visible.** Reviewers should know which findings came from rules, model output, external facts, "
            "or human decisions.",
        ),
        mistakes=(
            "**Putting the LLM in the authoritative path because the input is text.** This adds uncertainty where validated rules may suffice.",
            "**Letting generated prose change the decision.** A renderer should express canonical findings, not reinterpret them.",
            "**Having no failure mode.** Retries are not a safety design; define timeout, invalid-output, outage, and uncertainty behavior.",
            "**Ignoring the model's operational tax.** A small score gain may not justify new latency, cost, access controls, and monitoring.",
        ),
        references=(
            "[Risk guidance] NIST AI RMF Generative AI Profile",
            "[Specification] JSON Schema Draft 2020-12",
        ),
    ),
    11: Primer(
        why=(
            "The workflow transfers to new domains, but the dataset contract and evaluator do not transfer blindly. "
            "A support-intent metric is wrong for image documents, multi-label findings, code generation, or answers with "
            "many valid phrasings. This final notebook teaches how to identify the task before selecting data, models, and "
            "scores—and when to stop because the required authority or modality is missing."
        ),
        terms=(
            (
                "task formulation",
                "a precise statement of the input, output, constraints, user decision, and failure costs",
            ),
            (
                "input/output contract",
                "the machine- and reviewer-checkable shape and meaning of accepted inputs and outputs",
            ),
            (
                "modality",
                "the type of signal required, such as text, image, audio, tabular data, or combinations",
            ),
            (
                "OCR",
                "optical character recognition: converting pixels that depict text into machine-readable text, with possible errors",
            ),
            (
                "multi-label task",
                "a task where more than one label can be correct for the same example",
            ),
            (
                "exact match",
                "a strict metric requiring predicted and expected representations to be identical after defined normalization",
            ),
            (
                "field-level F1",
                "precision/recall-based scoring over predicted versus expected fields or items",
            ),
            (
                "executable evaluation",
                "checking generated code or structured plans by parsing, compiling, running tests, or enforcing invariants",
            ),
            (
                "human calibration",
                "measuring and reconciling reviewer agreement so subjective labels or judges have a known interpretation",
            ),
        ),
        mental_model=(
            "Choose the evaluator from the output contract and failure cost, not from the previous notebook. Work backward: "
            "what decision will consume the output, what makes that output correct or safe, who can know the truth, what "
            "modality carries the evidence, and which automatic plus human checks approximate those facts? Only then choose "
            "a dataset, baseline, prompt, or tuning method."
        ),
        decision_questions=(
            "Does the proposed input actually contain the information and modality needed for the target answer?",
            "Can multiple outputs be correct, and if so what semantic or executable check replaces naive exact match?",
            "Who supplies ground truth, and how will disagreement or uncertainty be represented?",
            "Do license, privacy, safety, or missing-domain facts block the project before model selection?",
        ),
        practices=(
            "**Verify a real sample and its documentation first.** Confirm fields, modalities, license, label authority, and "
            "known limitations before designing an adapter around a dataset name.",
            "**Start with one narrow task.** A small explicit contract makes leakage analysis, baselines, error taxonomies, "
            "and promotion gates possible.",
            "**Match metrics to output semantics.** Use per-field or set metrics for structured/multi-label output, executable "
            "checks for code, retrieval measures for search, and calibrated human review for subjective quality.",
            "**Respect modality boundaries.** A text-only model does not inspect pixels; OCR or a vision-capable component "
            "must be named, versioned, and evaluated as part of the system.",
            "**Block unsuitable projects explicitly.** `Not verified`, `requires external authority`, and `not enough evidence` "
            "are successful design outcomes when they prevent an invalid experiment.",
        ),
        mistakes=(
            "**Assuming columns from a dataset title or screenshot.** Inspect the actual versioned bytes and dataset card.",
            "**Expecting a text model to read scanned images.** Extracted text is a separate fallible component and needs its own evaluation.",
            "**Using exact match where several answers are valid.** Formatting variation can be penalized while substantive errors pass unnoticed.",
            "**Using an LLM judge without calibration.** Judges can be biased, inconsistent, or correlated with the model under test; compare with human labels.",
            "**Combining many tasks in the first experiment.** Mixed targets obscure which capability, data, and evaluator caused success or failure.",
        ),
        references=(
            "[Tool guidance] MLflow evaluation datasets and systematic evaluation",
            "[Specification] JSON Schema Draft 2020-12",
            "[Risk guidance] NIST AI RMF Generative AI Profile",
        ),
    ),
}


RUNNING_EXAMPLES: dict[int, str] = {
    0: (
        'The input `"I forgot my password"` belongs to the known intent '
        "`recover_password`. A usable answer is not free-form prose; it is a validated "
        'object such as `{"intent": "recover_password", "category": "account", '
        '"requires_escalation": false, "response": "I can help you reset your '
        'password."}`. A right intent in broken JSON still fails the system contract.'
    ),
    1: (
        "Follow that password-reset row through its chain of custody: a named Kaggle "
        "dataset revision contains a downloaded archive; the archive contains a CSV; "
        "the preserved bytes have a recorded SHA-256; repeatable code later derives a "
        "validated record. The hash answers `same bytes?`, not `safe and lawful?`."
    ),
    2: (
        'Suppose one row says `"I forgot my password"` and another says '
        '`"  I FORGOT my password  "`. A versioned normalization rule may group them '
        "as duplicate learning content. If the two rows carry different intents, the "
        "conflict is evidence to quarantine—not permission to pick the convenient label."
    ),
    3: (
        "All password-reset paraphrases assigned to one duplicate/template group travel "
        "together into train, validation, or test. If one appears in train and a sibling "
        "appears in test, the score may reward memorized wording. If you inspect a test "
        "mistake and rewrite the prompt, that test has become development data."
    ),
    4: (
        "A majority baseline may call every message the most frequent intent and miss "
        "`recover_password`. A keyword rule may notice `forgot` plus `password` and emit "
        "the right typed object. The future model must justify why its extra complexity is "
        "better than that task-aware rule, not merely better than majority guessing."
    ),
    5: (
        "The same frozen base model sees the password message three ways: a basic request, "
        "a constrained request naming the JSON contract and allowed labels, and a few-shot "
        "request with training demonstrations. Only prompt context changes. Compare these "
        "as method versions, not as three anecdotes."
    ),
    6: (
        "During LoRA training, the base model remains fixed while small adapter matrices "
        "learn from records such as the password example. The adapter may learn the mapping "
        "more consistently, but decreasing loss only shows progress on its objective. The "
        "locked adapter still has to beat the prompt and rule baselines on independent data."
    ),
    7: (
        "A frozen password-related example is now opened for the first formal comparison. "
        "Every locked method receives the identical input and is scored by the same parser, "
        "label set, response policy, and timing definition. An error can be both `wrong intent` "
        "and `invalid schema`; those categories overlap and should not be summed as records."
    ),
    8: (
        "The score for the LoRA answer is meaningful only beside its data fingerprint, base "
        "revision, adapter hash, prompt/configuration, evaluator version, and comparator. MLflow "
        "indexes that evidence. A predeclared gate then decides `adopt`, `reject`, or "
        "`inconclusive`; the tracking tool does not invent the decision."
    ),
    9: (
        "For a deployment-readiness manifest, `owner_group` is either present or absent in "
        "validated input, and a versioned rule determines the corresponding check. Whether an "
        "external deployment really exists requires an authorized registry lookup. Whether its "
        "remaining business risk is acceptable belongs to a human—not to generated ground truth."
    ),
    10: (
        "Think of the policy engine as a calculator and the optional model as a copywriter. The "
        "calculator owns readiness status, check IDs, and severity. The copywriter can explain "
        "those immutable findings. If it times out, emits invalid output, or contradicts a finding, "
        "a deterministic renderer takes over without changing the decision."
    ),
    11: (
        "An invoice project may contain image files, OCR text, token labels, boxes, or only some "
        "of them. A text-only model cannot see pixels. First verify the actual modality and ground "
        "truth; then choose field-level extraction metrics, normalization checks, and human review. "
        "Reusing intent-classification macro F1 would answer the wrong question."
    ),
}


REFERENCE_URLS: dict[str, str] = {
    "[Risk guidance] NIST AI RMF Generative AI Profile": (
        "https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence"
    ),
    "[Tool guidance] MLflow evaluation-driven development overview": (
        "https://mlflow.org/docs/latest/genai/eval-monitor"
    ),
    "[Tool guidance] Hugging Face Dataset Cards documentation": (
        "https://huggingface.co/docs/hub/en/datasets-cards"
    ),
    "[Tool guidance] scikit-learn common pitfalls: data leakage": (
        "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage"
    ),
    "[Tool guidance] scikit-learn grouped splitting documentation": (
        "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html"
    ),
    "[Tool guidance] scikit-learn metrics and scoring guide": (
        "https://scikit-learn.org/stable/modules/model_evaluation.html"
    ),
    "[Tool guidance] scikit-learn dummy estimators as baseline values": (
        "https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators"
    ),
    "[Tool guidance] MLflow prompt evaluation documentation": (
        "https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/prompts/"
    ),
    "[Tool guidance] Apple MLX-LM fine-tuning with LoRA or QLoRA": (
        "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md"
    ),
    "[Tool guidance] Hugging Face PEFT LoRA conceptual guide": (
        "https://huggingface.co/docs/peft/main/conceptual_guides/lora"
    ),
    "[Tool guidance] MLflow evaluation datasets and regression-test guidance": (
        "https://mlflow.org/docs/latest/genai/datasets/"
    ),
    "[Tool guidance] MLflow experiment tracking documentation": (
        "https://mlflow.org/docs/latest/ml/tracking"
    ),
    "[Tool guidance] MLflow dataset tracking and lineage documentation": (
        "https://mlflow.org/docs/latest/dataset/"
    ),
    "[Specification] JSON Schema Draft 2020-12": (
        "https://json-schema.org/draft/2020-12"
    ),
    "[Tool guidance] MLflow evaluation datasets and systematic evaluation": (
        "https://mlflow.org/docs/latest/genai/datasets/"
    ),
}
