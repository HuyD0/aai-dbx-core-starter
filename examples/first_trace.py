from aai_core import bootstrap
from aai_core.tracing import configure_tracing, traced

ctx = bootstrap()
configure_tracing(ctx.tags, experiment_name=ctx.settings.experiment_name)


@traced(span_type="CHAIN")
def answer(question: str) -> str:
    return f"Replace this example with a model call for: {question}"


print(answer("What should this application measure?"))
