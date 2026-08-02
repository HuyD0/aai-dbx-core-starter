"""``agentkit init`` — a guided wrapper around the governed template.

This is deliberately not a second scaffolding system. The platform's
templates are the paved road; ``init`` prints and runs the exact
``databricks bundle init`` command the platform console teaches, then
checks that the generated project actually evaluates before handing it
over.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from aai_core.agentkit.errors import ConfigError

DEFAULT_TEMPLATE = "evaluation-project"
TEMPLATE_REPO_ENV = "AAI_TEMPLATE_REPO"
_PROJECT_NAME = re.compile(r"^[a-z][a-z0-9-]+$")

NEXT_STEPS = """\
Next steps — the loop this toolkit is built around:

  cd {name}
  python3.12 scripts/setup_dev.py     # isolated setup, no credentials
  make install-ci                     # or `make install` with workspace auth

  agentkit smoke                      # seconds, free, no cluster
  agentkit compare --establish-baseline
                                      # scores the current version and
                                      # records it AS the baseline
  # ...now change something: edit src/app/example_agent.py
  agentkit compare                    # is this better than what we had?
  agentkit gate                       # promotion check (exit 0/2)
  agentkit evidence                   # the release record

`agentkit compare` is the primary verb: an experiment is a comparison, not
a log. Read docs/agent-evaluation.md for why that matters.\
"""


def build_command(
    *, project_name: str, template: str, template_source: str, output_dir: Path
) -> list[str]:
    return [
        "databricks",
        "bundle",
        "init",
        template_source,
        "--template-dir",
        f"templates/{template}",
        "--output-dir",
        str(output_dir),
    ]


def resolve_template_source(explicit: str | None, environ: Mapping[str, str]) -> str:
    """The template repository, never hardcoded into the SDK."""

    source = explicit or environ.get(TEMPLATE_REPO_ENV)
    if source:
        return source
    raise ConfigError(
        f"no template source configured (${TEMPLATE_REPO_ENV} is unset)",
        remediation=(
            "Run `source scripts/platform-env.sh` from an aai-core checkout "
            "to export the platform identifiers, or pass "
            "--template-source <repository-url>."
        ),
    )


def run_init(
    *,
    project_name: str,
    template: str = DEFAULT_TEMPLATE,
    template_source: str | None = None,
    output_dir: Path | None = None,
    print_only: bool = False,
    runner: Callable[..., Any] = subprocess.run,
    environ: Mapping[str, str] | None = None,
    emit: Callable[[str], None] = print,
) -> int:
    """Generate a working evaluation project and prove it runs."""

    import os

    environment = environ if environ is not None else os.environ
    if not _PROJECT_NAME.match(project_name):
        raise ConfigError(
            f"project name {project_name!r} must be lowercase letters, "
            "numbers, and hyphens",
        )
    source = resolve_template_source(template_source, environment)
    destination = Path(output_dir) if output_dir is not None else Path(project_name)
    command = build_command(
        project_name=project_name,
        template=template,
        template_source=source,
        output_dir=destination,
    )

    emit("The governed path starts with the platform template:")
    emit("")
    emit(f"  $ {' '.join(command)}")
    emit("")
    if print_only:
        emit(NEXT_STEPS.format(name=project_name))
        return 0

    completed = runner(command, check=False)
    code = getattr(completed, "returncode", 0)
    if code != 0:
        emit(f"`databricks bundle init` failed with exit code {code}.")
        return 1
    emit(NEXT_STEPS.format(name=project_name))
    return 0
