"""Register the prompt source as a new immutable version.

Reads prompts/system_prompt.json, registers it in the Unity Catalog prompt
registry, and points the `development` alias at the new version. Promotion to
validation/production is a separate, gated step (scripts/promote_prompt.py).
"""

import json
from pathlib import Path

from aai_core import bootstrap
from aai_core.prompts import PromptManager
from app.config import PROMPT_NAME

ROOT = Path(__file__).resolve().parents[1]

source = json.loads((ROOT / "prompts" / "system_prompt.json").read_text("utf-8"))
context = bootstrap(ROOT / "aai-platform.yml")
prompts = PromptManager(
    context=context.tags,
    catalog=context.settings.catalog,
    schema=context.settings.schema,
)
registered = prompts.register(
    PROMPT_NAME,
    source["messages"],
    commit_message=source.get("commit_message", "Update prompt"),
)
prompts.set_alias(PROMPT_NAME, alias="development", version=registered.version)
print({"name": registered.name, "version": registered.version})
