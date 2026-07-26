"""Lifecycle step 6: make one governed call through the stable synchronous adapter.

This connected teaching track uses a real configured model. Run it outside an
event loop with:

    make workspace-example EXAMPLE=connected_first_call

Async servers and streaming applications should use the advanced native-client
notebook instead.
"""

from __future__ import annotations

import hashlib

import mlflow

from aai_core import bootstrap
from aai_core.experiments import (
    ExperimentRunMetadata,
    RunPurpose,
    record_reproducibility,
)
from aai_core.tracing import TraceIntegration, TracePolicy

SYSTEM_PROMPT = (
    "Summarize only the supplied fictional company excerpt. Do not give "
    "investment advice. State when the excerpt lacks an answer."
)
QUESTION = (
    "Fictional Aster Ridge Systems reported quarterly revenue of $84.2 million. "
    "What revenue did it report?"
)


def main() -> None:
    ctx = bootstrap()
    ctx.configure_tracing(
        integration=TraceIntegration.SDK,
        policy=TracePolicy(),
    )
    model = ctx.providers.model("general-chat")
    prompt_digest = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()

    with ctx.experiments.run(
        run_name="connected-general-chat-grounded-summary-baseline",
        parameters={
            "logical_model": model.logical_name,
            "provider": model.provider,
            "deployment": model.model,
            "prompt_digest_sha256": prompt_digest,
            "temperature": 0.0,
            "cost_status": "unknown_until_trace_pricing_is_available",
        },
        metadata=ExperimentRunMetadata(
            purpose=RunPurpose.BASELINE,
            change_id="connected-grounded-summary-v1",
            change_summary="Establish a real-model grounded-summary baseline.",
            hypothesis=(
                "The configured model will report the supplied revenue without "
                "adding unsupported facts or investment advice."
            ),
        ),
    ):
        response = model.generate(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": QUESTION},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        metrics = {"latency_ms": response.latency_ms}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = response.usage.get(name)
            if isinstance(value, int):
                metrics[name] = float(value)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "aai.cost_status": "unknown",
                "aai.result": "connected_baseline_observed",
                "aai.decision": "evaluate_before_release",
                "aai.release": "blocked_until_evaluated",
            }
        )
        record_reproducibility(
            seed=0,
            extra={"prompt_digest_sha256": prompt_digest},
        )

    print(
        {
            "content": response.content,
            "provider": response.provider,
            "model": response.model,
            "usage": dict(response.usage),
            "latency_ms": round(response.latency_ms, 2),
            "cost_usd": None,
            "decision": "evaluate_before_release",
        }
    )


if __name__ == "__main__":
    main()
