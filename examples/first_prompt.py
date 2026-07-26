from aai_core import bootstrap
from aai_core.prompts import PromptManager


def main() -> None:
    ctx = bootstrap()
    prompts = PromptManager(
        context=ctx.tags,
        catalog=ctx.settings.catalog,
        schema=ctx.settings.schema,
    )
    registered = prompts.register(
        "first-prompt",
        "Answer this question using approved evidence: {{question}}",
        commit_message="Initial learning example",
    )
    loaded = prompts.load("first-prompt", version=registered.version)
    print({"name": loaded.name, "version": loaded.version})


if __name__ == "__main__":
    main()
