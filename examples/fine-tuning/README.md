# Learn fine-tuning from first principles, locally

This course teaches how a language model's weights are actually changed:
what full fine-tuning really costs, how quantization stores the same numbers
in fewer bits, how LoRA trains a small adapter instead of every weight, and
how the two combine as QLoRA. It ends where the platform begins — treating a
fine-tuned adapter as a governed application release with an evaluation gate,
as defined in [`docs/genai-lifecycle.md`](../../docs/genai-lifecycle.md)
("Model customization").

Everything runs locally on CPU with tiny models built from configuration
objects, so after installation the course needs **no download, no GPU, no
cloud credential, and no Databricks workspace**. The mathematics is identical
to what runs on real accelerators; only the sizes change.

## Who this is for

You should be comfortable with basic Python and have met the ideas in the
repository's numbered GenAI curriculum (prompts, traces, evaluation gates).
You do **not** need prior deep-learning, PyTorch, or fine-tuning experience:
the lessons build every concept before using it, starting from "what is a
parameter".

## Supported machines

- **macOS on Apple Silicon** (M-series). Intel Macs are not supported:
  current PyTorch releases publish no macOS x86_64 wheels.
- **Linux x86_64**, which is also what CI executes. Linux installs use the
  official CPU-only PyTorch wheel index, so no multi-gigabyte CUDA packages
  are ever downloaded.

## One-time setup

The course uses [`uv`](https://docs.astral.sh/uv/) with its own exact
`uv.lock`, so it never touches the repository's SDK environment or your
system Python. Install `uv` (the repository standardises on 0.8.23), then:

```bash
make install
make doctor
```

`make doctor` explains anything that is missing and names the Jupyter kernel
to select (**AAI Fine-Tuning**).

## Run the course

```bash
make notebook        # opens JupyterLab at lesson 00
make check           # formatting, unit + contract tests, execute every lesson twice
make mlflow-ui       # experiment UI for the training lessons' local store
make course-reset    # recoverably archive saved course state
```

| Lesson | What it teaches |
|---|---|
| `00_start_here` | When fine-tuning is the right lever, and the four-line memory bill that makes naive fine-tuning impractical. |

Later lessons extend this course through quantization from first principles,
LoRA built from scratch, QLoRA and the PEFT library, the SFT training-data
contract, a real CPU LoRA training run tracked in MLflow, the evaluation
gate, and finally the guarded, connected Databricks serverless-GPU lab. They
land as numbered notebooks here; the roadmap lives in the repository plan and
`docs/genai-lifecycle.md` defines the lifecycle stage they implement.

## The offline promise

Lessons construct models with `transformers` configuration objects rather
than downloading pretrained weights, and the Makefile exports
`HF_HUB_OFFLINE=1` so any accidental download attempt fails loudly instead
of succeeding quietly. Course state — the MLflow store and any caches —
lives under the ignored `.aai/course-v1/` directory and is archived, never
deleted, by `make course-reset`.

## Notebooks are generated

The `.ipynb` files are rendered from `scripts/notebook_content.py`. Edit the
lesson sources there and run
`.venv/bin/python scripts/render_notebooks.py`; `make check` fails if the
rendered notebooks are stale. This keeps lessons reviewable in diffs and
deterministic to execute.

## Relationship to the platform

The course deliberately uses the same lifecycle vocabulary as the rest of
the repository — `baseline -> change -> result -> decision`, with `adopt`,
`reject`, or `inconclusive` — and keeps torch, transformers, and peft out of
the SDK's certified dependency locks by owning its own environment. Serving
and training on Databricks stay keyless; nothing in this course asks for a
token or API key.
