# Offline local fine-tuning sample project

This is a self-contained study project for an Apple M3 MacBook Air with 24 GB
of unified memory. It teaches one evidence-driven change—LoRA fine-tuning a
small instruct model for structured customer-support classification—and then
applies the same discipline to a deterministic application-readiness capstone.

The project has two deliberately separate phases:

1. `make prepare-flight` is online. It installs the exact locked environment,
   downloads pinned public assets, verifies them, creates leakage-safe splits,
   records governed source plus interpreter/platform and package evidence,
   writes local MLflow evidence, and runs a small real MLX check.
2. `make flight-check` is offline. It performs no installation or download,
   enables library offline controls, denies Python sockets, rechecks every
   digest, writes to local MLflow, and runs the local model.

Run both before leaving:

```bash
cd examples/local-finetuning
make prepare-flight
make flight-check
```

The final line must be:

```text
READY FOR OFFLINE STUDY
```

For the strongest practical check, turn Wi-Fi off once and rerun
`make flight-check` before the trip. See [OFFLINE_STUDY.md](OFFLINE_STUDY.md)
for what is and is not included in that promise.

## What you will build

The primary task maps one support utterance to strict JSON:

```json
{
  "intent": "recover_password",
  "category": "account",
  "requires_escalation": false,
  "response": "I can help you reset your password."
}
```

The ordered workflow is:

```text
inspect data
  -> record a dataset card
  -> analyze duplicates and inferred templates
  -> make leakage-safe balanced splits
  -> run majority and deterministic baselines
  -> compare basic, strong, and few-shot untouched-model prompts
  -> run an MLX-LM smoke test
  -> train a LoRA change
  -> evaluate the frozen test set and slices
  -> inspect errors
  -> adopt, reject, or mark the change inconclusive
```

Classification and structured-output validity remain the principal measures.
The long source responses are inspected only in aggregate and are not copied
into training targets. A versioned curriculum policy renders brief target
responses, whose safety compliance is scored separately from classification;
a high intent score never hides weak structured or response output.

## Notebook course

The main learning experience is a 12-notebook narrative course. It begins with
the experiment question and data rights, shows every intermediate data and
evaluation object, and ends with the production-readiness capstone and a safe
extension-design exercise.

From the repository root:

```bash
make notebook
```

Start with [`00_start_here.ipynb`](notebooks/00_start_here.ipynb). The complete
ordered index, learning questions, and expected study time are in the
[`notebooks/README.md`](notebooks/README.md). Each notebook contains:

- learning objectives, prerequisites, and expected evidence;
- plain-language definitions before a term is used in code;
- a mental model, one running example, and decision questions;
- current best practices, common failure modes, and categorized primary guidance;
- small executable Python cells over local artifacts;
- interpretation guidance, exercises, hints, and checkpoints;
- a clear handoff to the next notebook.

The course distinguishes formal specifications, official tool guidance,
voluntary risk guidance, and conservative course rules. The explanations are
self-contained offline; links are optional reading after the flight. Maintainers
edit the generated narrative in `scripts/render_notebooks.py` and
`scripts/notebook_pedagogy.py`, then regenerate the tracked notebooks.

The CLI is not required for the lessons. It remains available for repeatable
preflight checks and longer unattended runs.

## Optional automation commands

All commands below are local after `prepare-flight`:

```bash
make study-smoke       # deterministic data/evaluation/policy path
make prepare-data      # reproduce the curated portable JSONL splits
make baselines         # majority and deterministic baseline evidence
make train-smoke       # ten local MLX-LM LoRA iterations
make train             # configured 200-iteration LoRA change
make evaluate          # frozen test metrics and error slices
make capstone          # deterministic readiness dataset and policy ceiling
make capstone-train-smoke # one real compact capstone LoRA iteration
make capstone-train    # configured compact capstone LoRA change
make capstone-evaluate # all six capstone methods on the frozen test set
make notebook          # open the numbered notebook course
make test              # fast curriculum contract tests
```

To inspect runs after returning to a connected development workflow:

```bash
uv run --offline --frozen --no-sync mlflow ui \
  --backend-store-uri sqlite:///.aai/mlflow.db \
  --default-artifact-root .aai/mlruns
```

No command needs Databricks, Azure, Kaggle, Hugging Face, or another remote
service during the offline phase.

## Baseline and decision contract

The project compares:

1. Majority-class output.
2. A train-only deterministic keyword/rule baseline.
3. The untouched model with a basic prompt.
4. The untouched model with a constrained prompt.
5. The untouched model with training-only few-shot examples.
6. The LoRA fine-tuned change.

The change is not adopted merely because it trained successfully. It must beat
the strongest meaningful baseline on macro F1 while meeting absolute JSON,
schema, unsupported-label, and response-policy gates. Evaluation uses the
repository vocabulary `baseline -> change -> result -> decision`; decisions
are `adopt`, `reject`, or `inconclusive`.

A completed training process is not sufficient evidence by itself. The
canonical adapter is eligible only when its success manifest matches the exact
base-model revision and runtime files, every training-data file, the effective
configuration, both adapter outputs, governed Python/notebook source, the exact
Python implementation and platform, and every installed distribution version.
Every baseline and change report also records its evaluation-time execution
contract. Promotion revalidates the preparation manifest and requires all
reports, the current runtime, and the LoRA training evidence to agree. A failed
retrain, source/package drift, or a mid-evaluation adapter change therefore
fails closed instead of reusing stale evidence under the same change name. A
per-adapter shared/exclusive lock keeps training publication and evaluation
from overlapping, and tracked change runs retain both the adapter weights and
`adapter_config.json` needed to reload them.

Runtime capture supports ordinary path-based editables and the standard
setuptools PEP 660 finder form. Other executable `.pth` files fail closed except
for exact, syntax-validated bootstrap forms emitted by the locked setuptools,
coverage, and virtualenv toolchain; virtualenv's distribution-less bootstrap is
content-bound as environment evidence. Effective `sys.path` precedence and
covered module origins are bound portably; unmatched roots, overlapping import
names, and shadowed origins fail closed. When the project directory itself is
the interpreter entry, its Python import surface is content-bound as separate
environment evidence. Runtime-significant distribution metadata includes
canonical `RECORD` and entry-point evidence. Generated launchers remain
portable only after their traversal target is proven to stay inside the Python
environment, with their machine-specific hash/size fields normalized.

Active source-equivalent `.pyc` caches are validated in an isolated no-site
Python process and tracked transiently; stale caches that Python will ignore
remain identity-tracked. Recognized active pytest-instrumented caches are bound
by a relocation-stable semantic digest; unknown sourceless or active
source-mismatched bytecode fails closed. Every lexical path component must be a
physical directory or file: environments reached through an ancestor symlink are
unsupported, so invoke the physical path instead. The setuptools `strict`
editable mode is also deliberately unsupported because its generated
symlink/hardlink tree cannot be represented as portable strict evidence; use
the standard finder or path editable mode when producing governed runs.
Python modules loaded after the evidence guard initializes are also checked
against the top-level code object observed when they execute; modules already
present at initialization require a normal, active, source-equivalent bytecode
cache as proof of the code Python loaded. A preloaded source module with no
cache, an instrumenting loader, or a stale cache fails closed. If runtime source
changes in a long-lived notebook or REPL, restart the Python process before
creating new evidence: a snapshot will not bind replacement bytes to stale code
that is still resident in memory. Arbitrary spec-less and originless modules are
rejected, and both importer/hook activation order and governed module
import/reload/removal activity are checked across the operation. Native or
other non-source modules loaded later fail closed unless their pre-load file
identity was established.

MLX-LM training launches the captured Python executable in isolated mode and
passes a fixed child environment with every `PYTHON*` override removed. This
means a caller's mutable `PYTHONPATH`, current project directory, or user site
cannot substitute a different `mlx_lm` implementation during training.

The governed source boundary is the reusable `src/aai_local_finetuning` package
plus `scripts/render_notebooks.py` and `scripts/notebook_pedagogy.py`. Generated
notebooks are outputs of those canonical sources; saving cell outputs does not
silently redefine evaluator or promotion logic.

## Project map

```text
configs/                  pinned source, model, split, and LoRA settings
data/                     ignored raw/generated data with tracked contracts
dataset_cards/            reviewed source and redistribution findings
docs/                     curriculum, optional dataset review, capstone policy
notebooks/                numbered narrative course with exercises/checkpoints
scripts/                  deterministic notebook renderer
src/aai_local_finetuning/ data, evaluation, MLX, tracking, and policy code
tests/                    fast cross-platform and offline guard tests
```

The generated chat JSONL is framework-neutral. MLX-LM consumes it today; a
future TRL/PEFT implementation can consume the same logical records without
changing IDs, labels, or frozen split membership.

## Dataset and model boundaries

The Bitext dataset is a public Kaggle learning input, not proof that its content
is suitable for commercial, enterprise, or production use. Its Kaggle metadata
currently identifies `CDLA-Sharing-1.0`; locally generated Data or Enhanced Data
retains that agreement. Review [the dataset card](dataset_cards/bitext-customer-support.md)
and [data notice](DATA_LICENSE.md) before redistributing anything.

Model weights, Kaggle files, adapters, MLflow stores, predictions, and learner
inputs are ignored by Git. Credentials are neither required for the public
downloads nor accepted as project files.

## Apple memory profile

The pinned 4-bit Qwen2.5 0.5B instruct model is intentionally small. The LoRA
configuration uses batch size 1, eight adapted layers, 512-token sequences,
gradient accumulation, and checkpointing. These are conservative settings for
24 GB unified memory; measured peak memory and latency are recorded locally,
not claimed in advance.
