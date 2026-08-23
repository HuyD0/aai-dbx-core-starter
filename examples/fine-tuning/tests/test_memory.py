import pydantic
import pytest

from aai_fine_tuning.memory import (
    Precision,
    activation_gb,
    full_fine_tune_estimate,
    optimizer_gb,
    parameter_gb,
)


def test_default_estimate_reproduces_the_8b_teaching_numbers():
    estimate = full_fine_tune_estimate()
    assert estimate.weights_gb == pytest.approx(16.0)
    assert estimate.gradients_gb == pytest.approx(16.0)
    assert estimate.optimizer_gb == pytest.approx(96.0)
    assert estimate.activations_gb == pytest.approx(18.253611008)
    assert estimate.total_gb == pytest.approx(146.253611008)
    # The lesson's headline: weights are only ~11% of the bill.
    assert estimate.weights_share == pytest.approx(0.1094, abs=1e-4)


def test_precision_table_scales_parameter_storage():
    assert parameter_gb(8.0, Precision.FP32) == pytest.approx(32.0)
    assert parameter_gb(8.0, Precision.BF16) == pytest.approx(16.0)
    assert parameter_gb(8.0, Precision.INT8) == pytest.approx(8.0)
    assert parameter_gb(8.0, Precision.NF4) == pytest.approx(4.0)


def test_optimizer_state_is_twelve_bytes_per_parameter():
    assert optimizer_gb(1.0) == pytest.approx(12.0)


def test_activation_estimate_scales_linearly_with_batch():
    one = activation_gb(1, 2048, 4096, 32)
    two = activation_gb(2, 2048, 4096, 32)
    assert two == pytest.approx(2 * one)


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_shapes_are_rejected(value):
    with pytest.raises(ValueError):
        parameter_gb(value, Precision.BF16)
    with pytest.raises(ValueError):
        optimizer_gb(value)
    with pytest.raises(ValueError):
        activation_gb(value, 2048, 4096, 32)


def test_estimate_is_immutable_evidence():
    estimate = full_fine_tune_estimate()
    with pytest.raises(pydantic.ValidationError):
        estimate.weights_gb = 0.0
    with pytest.raises(pydantic.ValidationError):
        type(estimate)(**estimate.model_dump(), surprise=1.0)
