from aai_core import bootstrap
from aai_core.tracing import configure_tracing, traced


def main() -> None:
    ctx = bootstrap()
    configure_tracing(
        ctx.tags,
        experiment_name=ctx.settings.effective_experiment_name,
        tracking_uri="databricks",
    )

    @traced(span_type="CHAIN")
    def answer(question: str) -> str:
        return f"Replace this example with a model call for: {question}"

    print(answer("What should this application measure?"))
    print(
        "Trace sent to the Databricks experiment "
        f"{ctx.settings.effective_experiment_name!r}; view it in the workspace UI."
    )


if __name__ == "__main__":
    main()
