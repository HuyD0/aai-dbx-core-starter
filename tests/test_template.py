"""Template catalog tests.

Discovery-driven: a template is any templates/<dir>/ containing
databricks_template_schema.json (templates/_shared/ and the tombstone are
excluded automatically). Static checks run without the Databricks CLI; the
render matrix and generated-project quality tier are skipped when it is
absent. All of it is credential-free.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from template_matrix import COMBOS

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / "templates"
IDENTIFIERS = json.loads((ROOT / "platform-identifiers.json").read_text())
COMPATIBILITY = json.loads((ROOT / "compatibility.json").read_text())
ACTION_PIN = re.compile(r"^\s*(?:-\s+)?uses:\s*[^@\s]+@([0-9a-f]{40})", re.MULTILINE)
ACTION_REF = re.compile(r"^\s*(?:-\s+)?uses:\s*[^@\s]+@([^\s]+)", re.MULTILINE)
REQUIRED_TAGS = {
    "application",
    "project",
    "environment",
    "team",
    "owner_group",
    "cost_center",
    "data_classification",
    "lifecycle",
    "tag_schema_version",
}


def discover_templates() -> list[Path]:
    return sorted(
        entry
        for entry in TEMPLATES_DIR.iterdir()
        if entry.is_dir() and (entry / "databricks_template_schema.json").is_file()
    )


TEMPLATES = discover_templates()
requires_cli = pytest.mark.skipif(
    shutil.which("databricks") is None, reason="Databricks CLI absent"
)


@pytest.fixture(scope="session")
def packaged_sdk_python(tmp_path_factory) -> Path:
    """Build and install the candidate wheel so template tests cannot silently
    import the SDK checkout through the repository's pytest pythonpath."""

    scratch = tmp_path_factory.mktemp("packaged-sdk")
    wheelhouse = scratch / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheelhouse),
        ],
        cwd=ROOT,
        check=True,
    )
    wheels = list(wheelhouse.glob("aai_core-*.whl"))
    assert len(wheels) == 1, wheels

    environment = scratch / "venv"
    subprocess.run(
        [
            "uv",
            "venv",
            "--python",
            sys.executable,
            str(environment),
        ],
        check=True,
        cwd=scratch,
        stdin=subprocess.DEVNULL,
    )
    python = (
        environment / "Scripts" / "python.exe"
        if os.name == "nt"
        else environment / "bin" / "python"
    )
    development_requirements = scratch / "development-requirements.txt"
    subprocess.run(
        [
            "uv",
            "export",
            "--locked",
            "--extra",
            "dev",
            "--extra",
            "all",
            "--no-emit-project",
            "--output-file",
            str(development_requirements),
        ],
        check=True,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--requirements",
            str(development_requirements),
            str(wheels[0]),
        ],
        check=True,
        cwd=scratch,
        stdin=subprocess.DEVNULL,
    )
    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import aai_core, json; "
                "print(json.dumps({'file': aai_core.__file__, "
                "'version': aai_core.__version__}))"
            ),
        ],
        check=True,
        cwd=scratch,
        capture_output=True,
        text=True,
    )
    installed = json.loads(probe.stdout)
    assert str(ROOT / "src") not in installed["file"]
    assert installed["version"] == COMPATIBILITY["sdk"]["version"]
    return python


def template_ids(template: Path) -> str:
    return template.name


def schema_for(template: Path) -> dict:
    return json.loads((template / "databricks_template_schema.json").read_text())


def combos_for(template: Path) -> list[dict]:
    combos = COMBOS.get(template.name)
    assert combos, (
        f"templates/{template.name} has no entry in tests/template_matrix.py — "
        "every template must declare its render combinations"
    )
    return combos


def all_combo_params() -> list:
    return [
        pytest.param(template, combo, id=f"{template.name}-{combo['name']}")
        for template in TEMPLATES
        for combo in combos_for(template)
    ]


def build_config(template: Path, overrides: dict) -> dict:
    """Identifier + combo overrides only; omitted properties resolve from
    schema defaults inside the CLI (verified: no prompting, and Go-templated
    defaults render correctly only when left to the CLI)."""

    schema_properties = schema_for(template)["properties"]
    config = {
        key: value
        for key, value in {
            "workspace_host": IDENTIFIERS["databricks_host"],
            "compute_policy_id": IDENTIFIERS["job_compute_policy_id"],
            "aai_core_volume": IDENTIFIERS["sdk_artifact_volume"],
            "aai_core_pip_source": (
                "git+https://github.com/HuyD0/aai-dbx-core-starter"
            ),
        }.items()
        if key in schema_properties
    }
    config.update(overrides)
    unknown = set(config) - set(schema_properties)
    assert not unknown, f"combo overrides unknown to schema: {sorted(unknown)}"
    return config


_RENDER_CACHE: dict[tuple[str, str], Path] = {}


def render(template: Path, combo: dict, tmp_path_factory) -> Path:
    key = (template.name, combo["name"])
    if key in _RENDER_CACHE:
        return _RENDER_CACHE[key]
    workdir = tmp_path_factory.mktemp(f"render-{template.name}-{combo['name']}")
    config_path = workdir / "config.json"
    config_path.write_text(
        json.dumps(build_config(template, combo["overrides"])), encoding="utf-8"
    )
    output = workdir / "generated"
    environment = dict(os.environ)
    # Local rendering must stay credential-free: a deliberately non-working
    # host + token keep the CLI from resolving real auth.
    environment["DATABRICKS_HOST"] = "https://localhost.invalid"
    environment["DATABRICKS_TOKEN"] = "x"
    environment.pop("DATABRICKS_CONFIG_PROFILE", None)
    environment.pop("DATABRICKS_AUTH_TYPE", None)
    subprocess.run(
        [
            "databricks",
            "bundle",
            "init",
            str(template),
            "--output-dir",
            str(output),
            "--config-file",
            str(config_path),
        ],
        check=True,
        cwd=workdir,
        env=environment,
        stdin=subprocess.DEVNULL,
    )
    _RENDER_CACHE[key] = output
    return output


# ---------------------------------------------------------------- static tier


@pytest.mark.parametrize("template", TEMPLATES, ids=template_ids)
def test_template_source_has_no_secret_references(template: Path):
    offenders = [
        path
        for path in template.rglob("*")
        if path.is_file() and "${{ secrets." in path.read_text(errors="ignore")
    ]
    assert not offenders, f"secret references in {offenders}"


@pytest.mark.parametrize("template", TEMPLATES, ids=template_ids)
def test_template_prompt_promotion_uses_validation_alias(template: Path):
    promotion = template / "template" / "scripts" / "promote_prompt.py"
    if promotion.is_file():
        text = promotion.read_text()
        assert '"candidate"' not in text
        assert '"validation"' in text
        assert '"production"' in text


@pytest.mark.parametrize("template", TEMPLATES, ids=template_ids)
def test_template_deploy_workflow_is_pinned_and_environment_free(template: Path):
    deploy = template / "template" / ".github" / "workflows" / "deploy.yml"
    text = deploy.read_text()
    assert len(ACTION_REF.findall(text)) == len(ACTION_PIN.findall(text))
    assert "id-token: write" in text
    assert "environment:" not in text
    release_gate = "databricks bundle run release_gate -t dev"
    app_deploy = "databricks bundle run agent_app -t dev"
    assert text.index(release_gate) < text.index(app_deploy)
    assert "hashFiles('resources/agent_app.yml')" in text
    app_preflight = 'databricks apps get "$APP_NAME"'
    assert text.index(app_preflight) < text.index("databricks bundle deploy -t dev")


@pytest.mark.parametrize("template", TEMPLATES, ids=template_ids)
def test_template_makefile_uses_generated_virtual_environment(template: Path):
    makefile = template / "template" / "Makefile"
    assert "PYTHON ?= .venv/bin/python" in makefile.read_text()


@pytest.mark.parametrize("template", TEMPLATES, ids=template_ids)
def test_template_schema_shared_contract(template: Path):
    schema = schema_for(template)
    properties = schema["properties"]
    for name in (
        "project_name",
        "application_name",
        "team",
        "owner_group",
        "cost_center",
        "catalog",
        "schema",
        "aai_core_version",
        "aai_core_volume",
        "workspace_host",
        "compute_policy_id",
        "aai_core_pip_source",
    ):
        assert name in properties, f"{template.name} schema missing {name}"
    for name, prop in properties.items():
        assert "default" in prop, f"{template.name}.{name} needs a default"
        assert "description" in prop, f"{template.name}.{name} needs a description"
    assert properties["project_name"]["pattern"] == "^[a-z][a-z0-9-]+$"
    if "model_provider" in properties:
        assert properties["model_provider"]["enum"] == ["databricks", "foundry"]
    if template.name == "agent-app":
        application_name = properties["application_name"]
        assert application_name["maxLength"] == 26
        assert application_name["pattern"] == "^[a-z](?:[a-z0-9-]*[a-z0-9])?$"
        prompt_version = properties["prompt_version"]
        assert prompt_version["default"] == "1"
        assert prompt_version["pattern"] == "^[1-9][0-9]*$"
        for app_only in (
            "experiment_id",
            "app_usage_policy_id",
        ):
            assert "skip_prompt_if" in properties[app_only]
        assert "{{if and" in schema["success_message"]
    if "retrieval_provider" in properties:
        assert properties["retrieval_provider"]["enum"] == [
            "azure_ai_search",
            "databricks_ai_search",
        ]


# ---------------------------------------------------------------- render tier


@requires_cli
@pytest.mark.parametrize("template,combo", all_combo_params())
def test_render_matrix(template: Path, combo: dict, tmp_path_factory):
    output = render(template, combo, tmp_path_factory)

    # T1: no unrendered template markers anywhere.
    for path in output.rglob("*"):
        if path.is_file():
            assert "{{." not in path.read_text(errors="ignore"), path

    # T2: bundle configuration parses.
    bundle = yaml.safe_load((output / "databricks.yml").read_text())
    for resource_file in sorted((output / "resources").glob("*.yml")):
        yaml.safe_load(resource_file.read_text())
    if (output / "aai-platform.yml").is_file():
        platform_config = yaml.safe_load((output / "aai-platform.yml").read_text())
        platform = platform_config["platform"]
        experiment_name = platform["experiment_name"]
        assert platform["application"] in experiment_name
        assert any(
            purpose in experiment_name
            for purpose in ("quality", "cost", "release", "evaluation")
        )
        assert not re.search(
            r"(?:^|[-_/])(first|test-\d+|experiment-\d+)(?:$|[-_/])",
            experiment_name.lower(),
        )

    # T3: generated CI stays credential-free; deploy keeps GitHub expressions.
    generated_ci = (output / ".github" / "workflows" / "ci.yml").read_text()
    ci_workflow = yaml.safe_load(generated_ci)
    assert "pull_request" in ci_workflow[True]
    assert ci_workflow["permissions"] == {"contents": "read"}
    ci_actions = ACTION_REF.findall(generated_ci)
    assert ci_actions and len(ci_actions) == len(ACTION_PIN.findall(generated_ci))
    assert "pip install -r requirements.lock" in generated_ci
    assert "pip check" in generated_ci
    generated_deploy = (output / ".github" / "workflows" / "deploy.yml").read_text()
    assert "${{ vars.AZURE_CLIENT_ID }}" in generated_deploy

    # T4: mandatory cost tags on the bundle preset AND every job cluster.
    preset_tags = bundle["targets"]["dev"]["presets"]["tags"]
    assert REQUIRED_TAGS.issubset(preset_tags)
    for resource_file in sorted((output / "resources").glob("*.yml")):
        resources = yaml.safe_load(resource_file.read_text())
        for job in resources.get("resources", {}).get("jobs", {}).values():
            for cluster in job.get("job_clusters", []):
                custom_tags = cluster["new_cluster"]["custom_tags"]
                assert REQUIRED_TAGS.issubset(custom_tags), resource_file

    # T5: provenance stamp identifies the template.
    stamp = json.loads((output / ".aai-template.json").read_text())
    assert stamp["template"] == template.name
    assert (
        stamp["template_version"]
        == COMPATIBILITY["templates"][template.name]["version"]
    )
    assert (
        stamp["generated_with"]["aai_core_version"]
        == COMPATIBILITY["templates"][template.name]["aai_core"]
    )

    # T6: generated setup is rendered, parseable, and exposes safe preflight.
    setup = output / "scripts" / "setup_dev.py"
    setup_text = setup.read_text()
    assert IDENTIFIERS["databricks_host"] in setup_text
    assert IDENTIFIERS["sdk_artifact_volume"] in setup_text
    subprocess.run(
        [sys.executable, str(setup), "--help"],
        check=True,
        cwd=output,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )

    # T7: combo file toggles (doubles as the dead-skip-glob guard).
    for expected in combo["expect_present"]:
        assert (output / expected).is_file(), f"missing {expected}"
    for unexpected in combo["expect_absent"]:
        assert not (output / unexpected).exists(), f"unexpected {unexpected}"
    assert not (output / "__preamble").exists()

    if template.name == "agent-app":
        readme = (output / "README.md").read_text()
        assert bundle["variables"]["prompt_version"]["default"] == "1"
        assert stamp["generated_with"]["prompt_version"] == "1"
        release_gate = yaml.safe_load(
            (output / "resources" / "agent_job.yml").read_text()
        )["resources"]["jobs"]["release_gate"]
        assert release_gate["tasks"][0]["spark_python_task"]["parameters"] == [
            "--prompt-version",
            "${var.prompt_version}",
        ]
        app_resource = output / "resources" / "agent_app.yml"
        if app_resource.is_file():
            application = yaml.safe_load(app_resource.read_text())["resources"]["apps"][
                "agent_app"
            ]
            environment = {item["name"]: item for item in application["config"]["env"]}
            assert environment["AAI_PROMPT_VERSION"] == {
                "name": "AAI_PROMPT_VERSION",
                "value": "${var.prompt_version}",
            }
            assert environment["MLFLOW_EXPERIMENT_ID"] == {
                "name": "MLFLOW_EXPERIMENT_ID",
                "value": "replace-with-mlflow-experiment-id",
            }
            assert "resources" not in application
            assert "one-time binding" in readme
            assert "deliberately contains no permission-bearing" in readme
            assert "databricks bundle run agent_app -t dev" in readme
        else:
            assert "databricks bundle run agent_app -t dev" not in readme
            assert "Serving resources were intentionally omitted" in readme


@requires_cli
@pytest.mark.parametrize("template", TEMPLATES, ids=template_ids)
def test_generated_project_quality(
    template: Path, tmp_path_factory, packaged_sdk_python: Path
):
    """Deep tier on each template's first combo: lint, unit tests, and the
    offline release-gate checks of the generated project."""

    combo = combos_for(template)[0]
    output = render(template, combo, tmp_path_factory)
    environment = dict(os.environ)
    # Only application source is added. aai-core must come from the candidate
    # wheel installed by packaged_sdk_python, never ROOT/src.
    environment["PYTHONPATH"] = str(output / "src")
    for command in (
        [str(packaged_sdk_python), "-m", "ruff", "check", "."],
        [str(packaged_sdk_python), "-m", "black", "--check", "."],
        [
            str(packaged_sdk_python),
            "-m",
            "pytest",
            "-q",
            str(output / "tests"),
        ],
        [str(packaged_sdk_python), str(output / "evals" / "offline_checks.py")],
    ):
        subprocess.run(command, check=True, cwd=output, env=environment)


# ------------------------------------------------------------- shared assets


def test_requirements_ci_uses_pip_source():
    for template in TEMPLATES:
        content = (template / "template" / "requirements-ci.txt.tmpl").read_text()
        assert "aai-core @ {{.aai_core_pip_source}}" in content
