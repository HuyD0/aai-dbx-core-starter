"""Hello world with a fictional earnings summary and zero cloud access.

Runs entirely on in-memory fakes — no Azure login, no Databricks workspace,
no endpoints. It demonstrates the SDK contracts every example and generated
project builds on: logical resource names, the normalized model response,
explicitly unknown cost, and secret redaction. It also introduces the
baseline/change/result/decision/release vocabulary used by the remaining
examples. Run it before any cloud onboarding:

    python examples/00_offline_hello_world.py
"""

from __future__ import annotations

from types import SimpleNamespace

from lifecycle_support import (
    BASELINE_NAME,
    CASES,
    CHANGE_NAME,
    HYPOTHESIS,
    dataset_digest,
    emit_result,
    quality_score,
)

from aai_core import PlatformContext, PlatformSettings
from aai_core.providers import ModelCapabilities, OpenAICompatibleChatModel
from aai_core.secrets import SecretResolver, SecretValue
from aai_core.tags import ResourceContext


class FakeCompletions:
    """Stands in for any OpenAI-compatible endpoint (Databricks serving,
    Foundry, or an APIM gateway) — the adapter neither knows nor cares."""

    def create(self, **request):
        del request
        return SimpleNamespace(
            model="fake-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=CASES[0].earnings_excerpt,
                        tool_calls=None,
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=8, completion_tokens=9),
        )


class StaticSecretProvider:
    def resolve(self, reference):
        return "not-a-real-secret"


def _disable_unconfigured_optional_tracing() -> None:
    """Avoid an unmanaged local trace when MLflow happens to be installed."""

    try:
        import mlflow
    except ImportError:
        return
    mlflow.tracing.disable()


def main() -> None:
    _disable_unconfigured_optional_tracing()
    settings = PlatformSettings(
        resource=ResourceContext(
            application="offline-hello",
            project="learning",
            environment="dev",
            team="my-team",
            owner_group="group:my-team-owners",
            cost_center="CC-0000",
            data_classification="internal",
            lifecycle="experimental",
            repository="local/checkout",
            release="dev",
        )
    )
    context = PlatformContext(settings)

    # Application code asks for a LOGICAL name; configuration decides what
    # serves it. Here we inject a fake exactly like the unit tests do.
    context.providers.register_model(
        "general-chat",
        OpenAICompatibleChatModel(
            logical_name="general-chat",
            provider="fake",
            model="fake-model",
            client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
            capabilities=ModelCapabilities(),
        ),
    )
    model = context.providers.model("general-chat")
    case = CASES[0]
    response = model.generate([{"role": "user", "content": case.question}])
    print(response.content)
    print(
        {
            "provider": response.provider,
            "model": response.model,
            "usage": dict(response.usage),
            "platform_tags": context.tags.for_mlflow(),
        }
    )

    # Secrets are references; verify their safe representations in memory.
    # Output only the result of that check, never data derived from the value.
    secrets = SecretResolver()
    secrets.register("fake", StaticSecretProvider())
    secret: SecretValue = secrets.resolve("fake://vault/example-key")
    assert repr(secret) == "SecretValue('[REDACTED]')"
    assert str(secret) == "[REDACTED]"
    print({"secret_redaction_verified": True})

    print("offline hello world completed with zero credentials")
    input_tokens = response.usage.get(
        "input_tokens",
        response.usage.get("prompt_tokens", 0),
    )
    output_tokens = response.usage.get(
        "output_tokens",
        response.usage.get("completion_tokens", 0),
    )
    result = {
        "quality_score": quality_score(
            {"answer": response.content},
            case.evaluation_record()["expectations"],
        ),
        "latency_ms": round(response.latency_ms, 3),
        "total_tokens": float(input_tokens + output_tokens),
        "cost_usd": None,
        "cost_coverage": 0.0,
    }
    emit_result(
        {
            "stage": "offline_contract",
            "hypothesis": HYPOTHESIS,
            "baseline": BASELINE_NAME,
            "change": CHANGE_NAME,
            "result": result,
            "decision": "start_governed_tracing",
            "release": "blocked_until_evaluated",
            "dataset_digest_sha256": dataset_digest(),
        }
    )


if __name__ == "__main__":
    main()
