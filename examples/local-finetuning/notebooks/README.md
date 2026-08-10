# Offline fine-tuning notebook course

From the repository root, run `make notebook`, then work from `00` through `11`.
Every notebook starts by enabling local-only library controls. The course uses
the prepared dataset, model, Python environment, and MLflow store; it does not
install or download anything.

This is a beginner-first sequence. Before the first code cell, every module
answers **why the topic matters**, defines its vocabulary, provides a mental
model and running example, asks decision questions, and names both current best
practices and common mistakes. Primary links are categorized as specification,
tool guidance, or risk guidance; they are optional because the teaching itself
is complete offline.

| Notebook | Main question | Typical live work |
|---|---|---:|
| `00_start_here.ipynb` | What are we trying to prove, and is this machine ready? | 20 min |
| `01_dataset_provenance_and_license.ipynb` | What source and usage rights were actually verified? | 30 min |
| `02_dataset_exploration_and_validation.ipynb` | What quality risks are present in the current bytes? | 45 min |
| `03_leakage_safe_splits.ipynb` | How do stable records become trustworthy evidence boundaries? | 40 min |
| `04_deterministic_baselines.ipynb` | What must a model beat? | 45 min |
| `05_prompt_baselines.ipynb` | How much can untouched weights do with better context? | 50 min |
| `06_lora_finetuning.ipynb` | What exactly changes during LoRA training? | 55 min plus optional training |
| `07_frozen_evaluation.ipynb` | Does the locked change generalize, and where does it fail? | 60 min plus inference |
| `08_mlflow_and_promotion.ipynb` | Is the evidence complete enough to adopt, reject, or stay inconclusive? | 40 min |
| `09_capstone_policy_dataset.ipynb` | How should domain ground truth be generated and governed? | 50 min |
| `10_capstone_model_vs_hybrid.ipynb` | Does a model belong in the authoritative decision path? | 55 min |
| `11_design_the_next_project.ipynb` | How should evaluation change for another task? | 35 min |

The full course is about 8¾ hours. It is easier to retain as four resumable
sessions:

| Session | Notebooks | Theme |
|---|---|---|
| 1 | `00`–`03` | Setup and trustworthy data |
| 2 | `04`–`06` | Baselines, prompting, and adapter training |
| 3 | `07`–`08` | Frozen evaluation and an evidence-backed decision |
| 4 | `09`–`11` | Policy-grounded capstone, hybrid design, and transfer |

The first executable cell in every notebook verifies the exact nested Python
kernel and enables the offline environment. The launch target registers that
kernel as `AAI Local Fine-Tuning (offline)`. If you open a notebook directly in
another editor, select that kernel before running cells; a wrong selection now
reports both the active and expected Python paths.

The notebooks display bounded, masked previews and aggregate findings instead
of unnecessary raw content. Optional training cells are disabled by default so
`Run All` remains practical; their instructions write to notebook-specific
artifact paths and cannot overwrite the canonical change.

Notebook `07` is the frozen-test boundary. Do not use its errors to revise the
same prompt, adapter, response policy, or thresholds. A follow-up remediation is
a new change and needs a new untouched evaluation version.

The generated notebooks are tracked. Maintainers can reproduce them after
editing the executable narrative in `scripts/render_notebooks.py` or the
beginner primers in `scripts/notebook_pedagogy.py` with:

```bash
.venv/bin/python scripts/render_notebooks.py
```
