from pathlib import Path

from aai_core import bootstrap
from aai_core.prompts import PromptManager

ROOT = Path(__file__).resolve().parents[1]

context = bootstrap(ROOT / "aai-platform.yml")
prompts = PromptManager(
    context=context.tags,
    catalog=context.settings.catalog,
    schema=context.settings.schema,
)
registered = prompts.register(
    "agent-system",
    [
        {
            "role": "system",
            "content": (
                "Answer only from supplied context. Cite sources and explicitly "
                "state when evidence is insufficient."
            ),
        },
        {"role": "user", "content": "{{question}}\n\n{{context}}"},
    ],
    commit_message="Initial governed Agentic RAG prompt",
)
prompts.set_alias(
    "agent-system",
    alias="development",
    version=registered.version,
)
print({"name": registered.name, "version": registered.version})
