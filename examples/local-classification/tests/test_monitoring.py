from __future__ import annotations

from aai_local_classification.contracts import SplitName
from aai_local_classification.data import load_split, prepare_dataset
from aai_local_classification.monitoring import compare_batches, shifted_batch


def test_shift_report_is_diagnostic_not_performance_claim(settings, tmp_path):
    root = tmp_path / "processed"
    prepare_dataset(settings, root)
    reference = load_split(settings, SplitName.VALIDATION, root)
    current = shifted_batch(reference, settings.random_seed + 1)
    report = compare_batches(reference, current, settings)

    assert report.maximum_numeric_psi > 0
    assert report.maximum_categorical_total_variation >= 0
    assert set(report.numeric_psi) == set(settings.features.numeric)
    assert "not proof" in report.interpretation
