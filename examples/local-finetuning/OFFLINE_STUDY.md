# Offline-study contract

## The promise

After `make prepare-flight` and a successful `make flight-check`, this project
can perform its study workflow without internet access on the prepared Mac.
Offline commands:

- use the nested locked environment without syncing it;
- reference a real local model directory, never a Hugging Face repository ID;
- reference immutable local Kaggle files and generated split files;
- set Hugging Face, Transformers, Datasets, MLflow, and uv offline controls;
- remove proxy variables and deny Python socket connections;
- use a repository-local SQLite MLflow store and local artifacts;
- fail with one complete missing/changed-assets report.

The check verifies the dataset archive and CSV SHA-256 hashes, exact model
revision and every required weight/config/tokenizer hash, processed data,
frozen split evidence, dependency lock, bundled JupyterLab kernel, local MLflow
write, and real local inference.
Preparation also runs a minimal LoRA training check so compilation surprises
happen before travel. Use `make notebook` to open the complete numbered
notebook course without extension downloads.

## What the promise does not mean

- A fresh clone with no preparation does not contain third-party model weights
  or Kaggle data.
- The code cannot stop every native library from opening a socket. It combines
  supported offline controls, local-only paths, a Python socket guard, and the
  recommended Wi-Fi-off rehearsal.
- Offline readiness does not make the public dataset suitable for production.
- The deterministic smoke path does not substitute for measured MLX results.
- Local MLflow evidence is private to this working copy until deliberately
  moved through an approved workflow.

## Recovery checklist before departure

If `flight-check` fails while you still have internet:

1. Run `make prepare-flight` again; it is idempotent and verifies cached files.
2. Confirm the final line says `READY FOR OFFLINE STUDY`.
3. Turn Wi-Fi off and run `make flight-check` once more.
4. Open the notebook course and run the first notebook.
5. Do not delete `data/raw`, `data/processed`, `models`, `.venv`, `.aai`, or
   `artifacts` until the trip is over.

No Kaggle token, Hugging Face token, cloud credential, or `.env` file belongs in
this directory.
