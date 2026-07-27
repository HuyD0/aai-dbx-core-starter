"""Build the copy-pasteable commands the developer runs on their own machine.

This is the console's one genuinely additive capability. The generated project's
preflight verifies access far better than a hosted app can, but nothing else pre-fills
the wizard invocation with the chosen template and this workspace's identifiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import ConsoleConfig
from .content import Block

# `bundle init --output-dir` becomes a directory name and is echoed into a shell
# command, so constrain it rather than quoting defensively at every call site.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

TEMPLATE_IDS = (
    "experiment-starter",
    "prompt-app",
    "evaluation-project",
    "rag-app",
    "agent-app",
)


class GenerateError(ValueError):
    """Raised when the requested generation inputs are not usable."""


@dataclass(frozen=True)
class GenerateRequest:
    template: str
    project_name: str = "my-project"


def _validate(request: GenerateRequest) -> None:
    if request.template not in TEMPLATE_IDS:
        raise GenerateError(f"unknown template {request.template!r}")
    if not _SAFE_NAME.match(request.project_name):
        raise GenerateError(
            "project name must start with a letter or digit and contain only letters, "
            "digits, dots, hyphens or underscores"
        )


def bundle_init(request: GenerateRequest, config: ConsoleConfig) -> list[Block]:
    """Render the keyless-auth + `bundle init` sequence for the chosen template."""
    _validate(request)

    if config.hosted and not config.template_repo:
        # The in-checkout relative form is a sensible default for `make app-run`,
        # but a hosted viewer has no checkout, so emitting it would hand every
        # developer a command that cannot work. Refuse instead of guessing — and
        # deliberately do not fall back to a baked-in URL: an identifier literal
        # here is exactly what makes a clone silently point at another tenant's
        # repository (tests/test_app_content.py forbids one).
        raise GenerateError(
            "this console has no template repository configured, so it cannot "
            "generate a working `bundle init`. The bundle must set "
            "AAI_CONSOLE_TEMPLATE_REPO from the `template_repo` variable, whose "
            "value comes from platform-identifiers.json."
        )

    source = config.template_repo or f"./templates/{request.template}"
    if config.template_repo:
        init = (
            f"databricks bundle init {source} \\\n"
            f"  --template-dir templates/{request.template} "
            f"--output-dir {request.project_name}"
        )
    else:
        init = f"databricks bundle init {source} --output-dir {request.project_name}"

    return [
        Block(lang="bash", code="az login"),
        Block(
            lang="bash",
            code=f"export DATABRICKS_HOST={config.identifier('databricks_host')}",
        ),
        Block(lang="bash", code="export DATABRICKS_AUTH_TYPE=azure-cli"),
        Block(lang="bash", code=init),
        Block(
            lang="bash",
            code=f"cd {request.project_name}\npython3.12 scripts/setup_dev.py",
        ),
    ]
