"""Small command-line entry points backing the notebook course."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from aai_fine_tuning.memory import Precision, full_fine_tune_estimate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    memory = subcommands.add_parser(
        "memory",
        help="Estimate the memory bill for fully fine-tuning a model.",
    )
    memory.add_argument("--parameters-billions", type=float, default=8.0)
    memory.add_argument(
        "--weight-precision",
        type=Precision,
        choices=list(Precision),
        default=Precision.BF16,
    )
    memory.add_argument("--micro-batch", type=int, default=2)
    memory.add_argument("--sequence-length", type=int, default=2048)
    memory.add_argument("--hidden-size", type=int, default=4096)
    memory.add_argument("--num-layers", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    estimate = full_fine_tune_estimate(
        parameters_billions=args.parameters_billions,
        weight_precision=args.weight_precision,
        micro_batch=args.micro_batch,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
    record = estimate.model_dump(mode="json")
    record["total_gb"] = estimate.total_gb
    record["weights_share"] = estimate.weights_share
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
