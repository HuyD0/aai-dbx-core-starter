"""Training-memory accounting for full fine-tuning.

Lesson 00 uses these functions to show why fully fine-tuning a modern model
needs roughly nine times the memory of its weights: the weights are one small
line item next to gradients, optimizer states, and activations.

All estimates use decimal gigabytes (1 GB = 1e9 bytes) and standard
mixed-precision training assumptions: BF16 weights and gradients, an AdamW
optimizer that keeps an FP32 master copy of the weights plus two FP32 moment
tensors, and half-precision activations without recomputation.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

BYTES_PER_GB = 1_000_000_000
# FP32 master weights + FP32 first moment + FP32 second moment.
ADAMW_BYTES_PER_PARAMETER = 12
# Half-precision activation bytes per (layer x token x hidden unit) without
# activation recomputation, dropping the attention-score term of the
# published estimate (Korthikanti et al., 2022). Good to teach magnitudes,
# not to plan a real cluster.
ACTIVATION_BYTES_PER_UNIT = 34


class Precision(StrEnum):
    """Storage formats the course discusses, with bytes per parameter."""

    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    INT8 = "int8"
    NF4 = "nf4"


PRECISION_BYTES: dict[Precision, float] = {
    Precision.FP32: 4.0,
    Precision.BF16: 2.0,
    Precision.FP16: 2.0,
    Precision.INT8: 1.0,
    Precision.NF4: 0.5,
}


class FullFineTuneEstimate(BaseModel, frozen=True, extra="forbid"):
    """One full fine-tuning memory bill, split into its four line items."""

    parameters_billions: float
    weight_precision: Precision
    weights_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float

    @property
    def total_gb(self) -> float:
        return sum(
            (
                self.weights_gb,
                self.gradients_gb,
                self.optimizer_gb,
                self.activations_gb,
            )
        )

    @property
    def weights_share(self) -> float:
        """The fraction of the bill that is the weights themselves."""
        return self.weights_gb / self.total_gb


def parameter_gb(parameters_billions: float, precision: Precision) -> float:
    """Storage for one copy of every parameter at the given precision."""
    if parameters_billions <= 0:
        raise ValueError("parameters_billions must be positive")
    bytes_total = parameters_billions * 1e9 * PRECISION_BYTES[precision]
    return bytes_total / BYTES_PER_GB


def optimizer_gb(parameters_billions: float) -> float:
    """AdamW state: FP32 master weights plus two FP32 moments per parameter."""
    if parameters_billions <= 0:
        raise ValueError("parameters_billions must be positive")
    return parameters_billions * 1e9 * ADAMW_BYTES_PER_PARAMETER / BYTES_PER_GB


def activation_gb(
    micro_batch: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
) -> float:
    """Approximate half-precision activation memory without recomputation."""
    for name, value in (
        ("micro_batch", micro_batch),
        ("sequence_length", sequence_length),
        ("hidden_size", hidden_size),
        ("num_layers", num_layers),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    units = micro_batch * sequence_length * hidden_size * num_layers
    return units * ACTIVATION_BYTES_PER_UNIT / BYTES_PER_GB


def full_fine_tune_estimate(
    parameters_billions: float = 8.0,
    weight_precision: Precision = Precision.BF16,
    micro_batch: int = 2,
    sequence_length: int = 2048,
    hidden_size: int = 4096,
    num_layers: int = 32,
) -> FullFineTuneEstimate:
    """The complete bill for fully fine-tuning a model of this shape.

    The defaults describe a Llama-style 8B model and reproduce the familiar
    result: about 146 GB, of which the weights are only about 11 percent.
    """
    return FullFineTuneEstimate(
        parameters_billions=parameters_billions,
        weight_precision=weight_precision,
        weights_gb=parameter_gb(parameters_billions, weight_precision),
        gradients_gb=parameter_gb(parameters_billions, weight_precision),
        optimizer_gb=optimizer_gb(parameters_billions),
        activations_gb=activation_gb(
            micro_batch=micro_batch,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
        ),
    )
