"""``agentkit init`` — a guided wrapper around the governed template.

This is deliberately not a second scaffolding system. The platform's
templates are the paved road; ``init`` prints and runs the exact
``databricks bundle init`` command the platform console teaches, then
checks that the generated project actually evaluates before handing it
over.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
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
    *,
    project_name: str,
    template: str,
    template_source: str,
    output_dir: Path,
    config_file: Path | None = None,
) -> list[str]:
    command = [
        "databricks",
        "bundle",
        "init",
        template_source,
        "--template-dir",
        f"templates/{template}",
        "--output-dir",
        str(output_dir),
    ]
    if config_file is not None:
        command += ["--config-file", str(config_file)]
    return command


def parse_settings(values: Sequence[str] | None) -> dict[str, str]:
    """``--set key=value`` pairs for the template's own prompts."""

    settings: dict[str, str] = {}
    for item in values or ():
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise ConfigError(
                f"--set expects key=value, got {item!r}",
                remediation=(
                    "Pass template answers as `--set repository_url=https://"
                    "example.com/org/repo`."
                ),
            )
        settings[key.strip()] = value
    return settings


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
    settings: Mapping[str, str] | None = None,
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

    # `databricks bundle init --config-file` answers the template's prompts
    # from a file instead of asking, and it is all-or-nothing: it does not
    # fall back to prompting for what the file omits. So the file is written
    # only when the caller supplied template answers with --set, and it
    # carries the project name so `--name` reaches the generated bundle
    # rather than only naming the directory.
    answers = dict(settings or {})
    config_file: Path | None = None
    if answers:
        answers.setdefault("project_name", project_name)
        config_file = destination.parent / f".{destination.name}-init.json"
    command = build_command(
        project_name=project_name,
        template=template,
        template_source=source,
        output_dir=destination,
        config_file=config_file,
    )

    emit("The governed path starts with the platform template:")
    emit("")
    emit(f"  $ {' '.join(command)}")
    emit("")
    if config_file is None:
        emit(
            f"The template will prompt for its values; answer "
            f"'Project and bundle name' with `{project_name}` so the bundle "
            "matches this directory (or pass `--set key=value` answers to "
            "skip the prompts)."
        )
        emit("")
    if print_only:
        emit(NEXT_STEPS.format(name=project_name))
        return 0

    if config_file is not None:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(
            json.dumps(answers, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    try:
        completed = runner(command, check=False)
    finally:
        if config_file is not None:
            config_file.unlink(missing_ok=True)
    code = getattr(completed, "returncode", 0)
    if code != 0:
        emit(f"`databricks bundle init` failed with exit code {code}.")
        if config_file is not None:
            emit(
                "A --config-file answers every prompt at once, so the "
                "template fails on any value it needs that was not supplied. "
                "Add it with `--set key=value`, or drop --set to be prompted."
            )
        return 1
    _warn_on_name_mismatch(destination, project_name, emit)
    emit(NEXT_STEPS.format(name=project_name))
    return 0


def _warn_on_name_mismatch(
    destination: Path, project_name: str, emit: Callable[[str], None]
) -> None:
    """Say so when the generated bundle is not named what was asked for."""

    bundle = destination / "databricks.yml"
    try:
        text = bundle.read_text(encoding="utf-8")
    except OSError:
        return
    match = re.search(r"^\s{2}name:\s*(\S+)\s*$", text, re.MULTILINE)
    if match is None or match.group(1).strip("\"'") == project_name:
        return
    emit(
        f"Note: the generated bundle is named {match.group(1)}, not "
        f"{project_name}. Edit `bundle.name` in databricks.yml if that was "
        "not deliberate."
    )
