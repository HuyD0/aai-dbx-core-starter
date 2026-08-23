"""Lesson sources for the fine-tuning course notebooks.

This module must remain stdlib-only plain data: the repository-level guard
test imports it from the root SDK environment (which has no torch or
transformers) to derive the expected lesson list.

Each lesson is an ordered list of ("markdown" | "code", text) cells rendered
by scripts/render_notebooks.py. A code cell whose source starts with
"# Preflight" is tagged as the lesson's environment check; one starting with
"# Reference solution" is tagged as a solution cell.
"""

from __future__ import annotations

# Anchor asserted by the tests: removing or adding a lesson is always a
# deliberate, named edit here, never a silent side effect.
EXPECTED_LESSON_COUNT = 1

_LESSON_00 = [
    (
        "markdown",
        """
# Lesson 00 — Start here: when prompting is not enough

This course teaches the third way to change what a language model does:
changing its weights. The repository's numbered GenAI curriculum already
teaches the first two — writing better prompts, and retrieving better
context — and its earnings-summary assistant is exactly the kind of
application that eventually needs the third: after two prompt versions, the
assistant still sometimes forgets to cite its source identifier in the exact
required format.

Everything in this course runs on your machine, on CPU, offline after
installation. The models are tiny and built from configuration objects, so
nothing is downloaded and no GPU, cloud credential, or Databricks workspace
is needed. What you learn transfers directly to real models because the
mathematics is identical — only the sizes change.

This lesson establishes two things: **when** fine-tuning is the right tool,
and **why** doing it naively is so expensive that the rest of this course
exists.
""",
    ),
    (
        "code",
        """
# Preflight: confirm the exact course environment before any other cell.
import importlib.util
import os
import sys

required = ("mlflow", "peft", "torch", "transformers", "aai_fine_tuning")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing packages: "
        + ", ".join(missing)
        + ". Run `make install`, then select the 'AAI Fine-Tuning' kernel."
    )
os.environ.setdefault("HF_HUB_OFFLINE", "1")
print("Environment OK on Python", sys.version.split()[0])
print("Offline guard active: HF_HUB_OFFLINE =", os.environ["HF_HUB_OFFLINE"])
""",
    ),
    (
        "markdown",
        """
## Words introduced

| Word | Meaning in this course |
|---|---|
| Parameter (weight) | One learned number inside the model. Modern models have billions. |
| Fine-tuning | Continuing training on your own examples so the weights change. |
| Supervised fine-tuning (SFT) | Fine-tuning on input → desired-output pairs. |
| Gradient | The training signal stored for every trainable parameter on each step. |
| Optimizer state | Extra numbers the optimizer keeps per trainable parameter (AdamW keeps three). |
| Activation | Intermediate results kept during the forward pass so gradients can be computed. |
| Precision | How many bytes store one number: FP32 uses 4, BF16 uses 2, NF4 uses about half a byte. |
| Quantization | Storing weights in fewer bits (lesson 01). |
| LoRA | Training a small low-rank adapter instead of every weight (lesson 02). |
| Adapter | The small set of new weights LoRA trains and ships. |
""",
    ),
    (
        "markdown",
        """
## The three levers

There are exactly three places to change what a model-backed application
does, and they form an escalation ladder, not a menu of equals:

1. **Prompting** changes the instructions. The model and its context are
   untouched. Cheapest, fastest to evaluate, first choice always.
2. **Retrieval (RAG)** changes what the model reads before answering. This
   is how an application knows facts the model was never trained on.
3. **Fine-tuning** changes the weights. This is how a model learns to
   *behave* differently: follow a format every time, keep a tone, emit
   exactly one citation, produce parseable JSON without coaxing.

The rule that keeps teams out of trouble: **fine-tuning changes behavior,
not knowledge.** If the application answers incorrectly because it lacks
facts, fix retrieval. If it answers in the wrong shape no matter how the
prompt begs — that is behavioral, and behavioral problems are what
supervised fine-tuning solves. Escalate to this course's techniques only
after honest prompt and retrieval attempts have failed a real evaluation
gate, because a fine-tuned model is a new application release that must be
evaluated, versioned, and paid for like any other
(see `docs/genai-lifecycle.md`, "Model customization").
""",
    ),
    (
        "markdown",
        """
## The real cost of changing the weights

### Before you run this

The next cell prices the naive approach: fully fine-tuning every weight of
an 8-billion-parameter model in BF16 with the standard AdamW optimizer.

Predict before running: the weights themselves occupy 16 GB
(8 billion × 2 bytes). Will the whole training bill be closer to 20 GB,
50 GB, or 150 GB? Write your guess down — the gap between guesses and the
answer is the entire motivation for lessons 01 and 02.
""",
    ),
    (
        "code",
        """
from aai_fine_tuning.memory import Precision, full_fine_tune_estimate

estimate = full_fine_tune_estimate(
    parameters_billions=8.0,
    weight_precision=Precision.BF16,
    micro_batch=2,
    sequence_length=2048,
    hidden_size=4096,
    num_layers=32,
)

rows = (
    ("weights (BF16)", estimate.weights_gb),
    ("gradients (BF16)", estimate.gradients_gb),
    ("optimizer (AdamW, FP32)", estimate.optimizer_gb),
    ("activations (approximate)", estimate.activations_gb),
)
for label, gigabytes in rows:
    print(f"{label:<28}{gigabytes:9.1f} GB")
print("-" * 40)
print(f"{'total':<28}{estimate.total_gb:9.1f} GB")
print(f"weights share of the bill:  {estimate.weights_share:9.1%}")
""",
    ),
    (
        "markdown",
        """
### What you should see

A four-line bill totalling about **146 GB**: 16 GB of weights, 16 GB of
gradients, 96 GB of AdamW optimizer state (an FP32 master copy of the
weights plus two FP32 moment tensors — 12 bytes per parameter), and around
18 GB of activations for this batch shape. The final line reports that the
weights are only about **11%** of the total.
""",
    ),
    (
        "markdown",
        """
### How to interpret

The model itself is the smallest problem. Roughly 89% of the bill is
*training overhead*, and every line of it scales with the number of
**trainable** parameters — gradients and optimizer state exist only for
parameters that learn. That observation hands us exactly two levers, and
they are the next two lessons:

- **Store the frozen weights in fewer bits** — quantization (lesson 01).
  This shrinks the weights line without touching what the model computes
  at full precision.
- **Train far fewer parameters** — LoRA (lesson 02). If only a small
  adapter learns, the gradient and optimizer lines nearly vanish.

Stacked together (QLoRA, lesson 03), the same 8B fine-tune drops from
146 GB toward the memory of a single commodity accelerator — which is why
these techniques, not bigger hardware, are the industry default.
""",
    ),
    (
        "markdown",
        """
## Guided exercise

Estimate the bill for a 70-billion-parameter model
(`hidden_size=8192`, `num_layers=80`, same batch shape). Then answer from
the printed lines, not intuition:

1. Which line items would shrink if the weights were frozen?
2. Which line would quantization shrink?
3. Could even a single 80 GB accelerator hold just the BF16 weights and
   optimizer state of the 70B model?
""",
    ),
    (
        "code",
        """
# Edit the marked values, run, and answer the three questions above.
from aai_fine_tuning.memory import Precision, full_fine_tune_estimate

my_estimate = full_fine_tune_estimate(
    parameters_billions=8.0,  # <- edit me
    weight_precision=Precision.BF16,
    micro_batch=2,
    sequence_length=2048,
    hidden_size=4096,  # <- edit me
    num_layers=32,  # <- edit me
)
print(f"total: {my_estimate.total_gb:,.1f} GB")
print(f"weights: {my_estimate.weights_gb:,.1f} GB")
print(f"gradients + optimizer: {my_estimate.gradients_gb + my_estimate.optimizer_gb:,.1f} GB")
""",
    ),
    (
        "markdown",
        """
### Solution
""",
    ),
    (
        "code",
        """
# Reference solution
from aai_fine_tuning.memory import Precision, full_fine_tune_estimate

seventy_b = full_fine_tune_estimate(
    parameters_billions=70.0,
    weight_precision=Precision.BF16,
    micro_batch=2,
    sequence_length=2048,
    hidden_size=8192,
    num_layers=80,
)
print(f"70B total: {seventy_b.total_gb:,.1f} GB")
print(f"70B weights alone: {seventy_b.weights_gb:,.1f} GB")
print(f"70B gradients + optimizer: {seventy_b.gradients_gb + seventy_b.optimizer_gb:,.1f} GB")

# 1. Freezing the weights removes the gradient line and the optimizer line
#    for every frozen parameter — together far larger than the weights.
# 2. Quantization shrinks the weights line (16 -> ~4 GB at 4-bit for 8B).
# 3. No: the 70B BF16 weights alone are 140 GB, and optimizer state adds
#    840 GB more. Full fine-tuning at this scale is a multi-node problem,
#    which is why lesson 02's answer — train fewer parameters — matters.
""",
    ),
    (
        "markdown",
        """
## Recap

- Prompting, retrieval, and fine-tuning are an escalation ladder; fine-tune
  for **behavior**, never to inject facts.
- A full fine-tune pays for four things — weights, gradients, optimizer
  state, activations — and the weights are only ~11% of the bill.
- Gradients and optimizer state exist only for *trainable* parameters,
  which is why quantization (lesson 01) and LoRA (lesson 02) attack the
  bill from opposite sides.
- In lifecycle terms nothing is released yet: this lesson produced a
  hypothesis about cost, not evidence about quality. The
  `baseline -> change -> result -> decision` contract arrives with the
  training lessons.

Next: **lesson 01, quantization from first principles** — storing the same
numbers in fewer bits, and measuring exactly what that costs in accuracy.
""",
    ),
]

LESSONS: dict[str, list[tuple[str, str]]] = {
    "00_start_here.ipynb": _LESSON_00,
}

assert len(LESSONS) == EXPECTED_LESSON_COUNT
