# Notebook course

Run `make notebook` and open the lessons in order:

1. `00_start_here.ipynb` — lifecycle map, environment, and evidence question.
2. `01_problem_and_data_contract.ipynb` — prediction time, target, action,
   positive label, error costs, feature contract, and release gate.
3. `02_data_quality_and_eda.ipynb` — provenance, schema, prevalence,
   missingness, feature distributions, and cohort drift.
4. `03_leakage_safe_splits.ipynb` — time partitions, frozen manifest, leakage
   demonstration, and train-fitted transformations.
5. `04_baseline.ipynb` — no-skill baseline and why accuracy is misleading.
6. `05_pipeline_and_training.ipynb` — reusable preprocessing, two controlled
   candidates, explicit MLflow runs, dataset inputs, signature, and lock.
7. `06_model_selection_and_threshold.ipynb` — ranking selection versus
   validation-only operating-threshold choice.
8. `07_frozen_test_gate.ipynb` — one exact-artifact test, classic MLflow
   evaluation, operational slices, and release decision.
9. `08_registry_and_inference.ipynb` — conditional registration, version tags,
   `champion`, model reload, and inference parity.
10. `09_monitoring_and_databricks.ipynb` — simulated input drift, delayed-label
    boundary, and the local-to-Databricks map.

Every notebook has stable cell IDs, no stored outputs, and the same pattern:
objectives, prerequisites, executable evidence, interpretation, an exercise with
a hint, and a checkpoint. The notebook checker executes each lesson against a
fresh temporary project root, so the tracked course does not depend on state in
your `.aai/` folder.
