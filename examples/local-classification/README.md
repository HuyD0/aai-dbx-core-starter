# Learn classification and MLOps locally

This course teaches you how to build a small model that estimates whether a
fictional subscription will be cancelled in the next 30 days. It starts with
the meaning of a classification prediction, then adds data checks, evaluation,
MLflow tracking, release decisions, inference, and monitoring.

Everything runs locally on a Mac. The dataset is generated for the course, so
it contains no customer data and needs no Databricks account, cloud credential,
GPU, or data download. Installing the tools and Python packages does require an
internet connection the first time.

## Who this is for

You should already be comfortable with basic Python and pandas: variables,
functions, lists and dictionaries, selecting DataFrame columns, and reading a
small table. You do **not** need prior machine-learning, scikit-learn, MLflow,
MLOps, or Databricks knowledge. The notebooks define those concepts before
using them.

You also need to be willing to use the macOS Terminal for a few setup commands.
The notebooks do the learning work after setup.

## One-time Mac setup

The project supports Python 3.11 and 3.12; Python 3.12 is recommended. `uv`
creates an isolated environment and installs the exact package versions in
`uv.lock`, so the course does not change your system Python packages.

1. Open **Terminal**.
2. Install Apple's command-line tools if `make --version` does not work:

   ```bash
   xcode-select --install
   ```

3. Install `uv` using one of its official installation methods. With Homebrew:

   ```bash
   brew install uv
   ```

   Or with the official installer:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Close and reopen Terminal if the installer asks you to refresh your shell.

4. Install the recommended Python version and confirm the tools are visible:

   ```bash
   uv python install 3.12
   uv --version
   uv python find 3.12
   make --version
   ```

See the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
if your organization manages developer tools differently.

## Quick start

Change into this directory, then run exactly these commands:

```bash
cd examples/local-classification
make install
make doctor
make notebook
```

What success looks like:

- `make install` creates `.venv` and finishes without a dependency error. The
  first install can take several minutes.
- `make doctor` prints a short set of passing checks for Python, installed
  packages, the notebook kernel, writable course state, and all ten lessons. It
  should finish in well under a minute.
- `make notebook` opens JupyterLab at `00_start_here.ipynb`. The selected kernel
  should be **AAI Local Classification**.

From the repository root, the equivalent commands are:

```bash
make classification-install
make classification-doctor
make classification-notebook
```

Do not start with `make check`; that is the slower contributor verification
suite described near the end of this README.

## How to use the notebooks

Start at notebook 00 and continue in numeric order. In JupyterLab:

1. Read the explanation above a code cell.
2. Predict what the cell will show when the notebook asks you to.
3. Press **Shift+Enter** to run the cell and move to the next one.
4. Read the output interpretation before continuing.
5. Complete the practice cell and its self-check.

An execution number such as `[3]` means the cell finished. `[*]` means it is
still running. Some MLflow operations take a little longer on their first use.
If cells were run out of order, choose **Kernel → Restart Kernel and Run All
Cells**.

The notebooks are generated without saved outputs. Each lesson tells you the
expected value or range, so a newly opened notebook looking unexecuted is
normal.

## Where your work goes

Course commands keep generated data, MLflow runs, registered models, and
lesson evidence under:

```text
.aai/course-v2/
```

That directory is ignored by Git. It lets lessons reuse earlier work without
mixing it with another version of the course.

To start again, stop Jupyter with `Ctrl-C` in its Terminal and run:

```bash
make course-reset
```

The reset is recoverable: it moves the current course state aside and prints
where the backup was placed. From the repository root, use
`make classification-reset`.

## The ten lessons

| Lesson | Plain-language question | Typical time |
|---:|---|---:|
| 00 | Is my environment working, and what will this model do? | 25–35 min |
| 01 | What is a classification prediction, and what decision will it support? | 35–45 min |
| 02 | What rows and columns do we have, and can we trust them? | 45–55 min |
| 03 | Why do we split by time, and how can future information leak? | 40–50 min |
| 04 | What does “better than guessing” mean, why can accuracy mislead us, and how is a baseline recorded? | 50–60 min |
| 05 | How do preprocessing and model training work as one Pipeline? | 60–75 min |
| 06 | How do we choose a model separately from choosing an action threshold? | 60–75 min |
| 07 | Did the fixed model pass a genuinely untouched final test? | 50–65 min |
| 08 | How do we register, reload, and use the approved model safely? | 45–60 min |
| 09 | What should we monitor, and how do these local ideas map to Databricks? | 60–75 min |

The first pass is about eight to ten hours. It is fine to stop after any lesson;
the notebook tells you what evidence it saved and what the next lesson needs.
See [the full curriculum](docs/curriculum.md) and the
[beginner glossary](docs/glossary.md).

## View your MLflow work

Starting in lesson 04, the notebooks create MLflow experiments and runs. Leave
Jupyter running, open a **second Terminal**, return to this directory, and run:

```bash
make mlflow-ui
```

Open <http://127.0.0.1:5000>. The UI shows the same local runs and registered
model versions used by the notebooks. Stop only the MLflow UI with `Ctrl-C` in
the second Terminal.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `uv: command not found` | Reopen Terminal, then follow the official uv installation guide linked above. |
| `make` asks for developer tools | Run `xcode-select --install`, finish the Apple installer, and retry. |
| `make install` cannot download packages | Confirm the Mac has internet access and that any company proxy permits Python package downloads. Notebooks are offline after installation. |
| `ModuleNotFoundError: aai_local_classification` | Stop the notebook, rerun `make install`, then launch it with `make notebook`; do not use an unrelated Jupyter installation. |
| Kernel is not **AAI Local Classification** | Choose **Kernel → Change Kernel → AAI Local Classification**. If it is absent, stop Jupyter and rerun `make doctor`. |
| A cell stays at `[*]` | Wait for the first MLflow initialization. If it does not finish, restart the kernel and run all cells. |
| A lesson reports incompatible or consumed evidence | Run `make course-reset`, relaunch the notebook, and run the lessons in order. The old state is backed up. |
| Port 8888 or 5000 is already in use | Stop the older Jupyter/MLflow process with `Ctrl-C`, or close its Terminal, then retry. |
| The browser did not open | Copy the local `http://127.0.0.1:...` URL printed by the command into your browser. Never share a URL containing a Jupyter token. |

If a cell still fails, copy the **first error message**, the notebook number,
and the output of `make doctor`. Later traceback lines are often consequences
of the first failure.

## What the project deliberately practices

By the end, you will have built one evidence chain:

```text
question and action
  → checked data
  → time-based train / validation / test split
  → simple baseline
  → preprocessing and two trained models
  → validation-only model and threshold choice
  → untouched test decision
  → conditional model registration
  → representative label-free inference and monitoring
```

The notebooks first show the important operations directly. Reusable versions
live under `src/aai_local_classification/`, because production jobs should run
tested Python modules rather than depend on a notebook's hidden execution state.

## Project map

```text
configs/project.yaml               learning assumptions and release rules
data/README.md                     generated-data contract
docs/curriculum.md                 lesson outcomes and completion checks
docs/glossary.md                   plain-language course vocabulary
docs/resources.md                  official reading path by lesson
docs/databricks-handoff.md         local concepts mapped to Databricks
notebooks/                         ten ordered lessons
src/aai_local_classification/      reusable and tested workflow code
  model-requirements.lock          exact MLflow model restore closure
tests/                             code and lifecycle checks
uv.lock                            exact dependency versions
```

## Contributor verification

Learners do not need this before opening the course. Run it after changing the
course code, configuration, or notebook source:

```bash
make check
```

It checks formatting, runs the unit/workflow tests, executes all ten notebooks
from clean temporary state, and reruns state-sensitive lessons.

## Important limits

This is a learning system, not a production architecture or a real retention
model. Synthetic data cannot establish business value, fairness, privacy, or
representativeness. Local SQLite is suitable for one learner, not a concurrent
team. A model tested on macOS must still be tested in its target Linux runtime.
Only after completing the local lifecycle should you use the
[Databricks handoff](docs/databricks-handoff.md) to map these concepts to a
governed workspace.
