"""Template catalog tests.

Discovery-driven: a template is any templates/<dir>/ containing
databricks_template_schema.json (templates/_shared/ and the tombstone are
excluded automatically). Static checks run without the Databricks CLI; the
render matrix and generated-project quality tier are skipped when it is
absent. All of it is credential-free.
"""

import configparser
import json
import os
import re
import shutil
import subprocess
import sys
import types
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
CURRENT_TEMPLATE_VERSIONS = {
    "agent-app": "1.5.0",
    "analytics-app": "1.3.0",
    "evaluation-project": "2.3.0",
    "experiment-starter": "1.4.0",
    "prompt-app": "1.4.0",
    "rag-app": "1.4.0",
}
STRICT_JUDGE_CALLS = (
    pytest.param(
        "agent-app/template/evals/evaluate.py",
        "agent = ToolAgent(",
        id="agent-evaluation",
    ),
    pytest.param(
        "prompt-app/template/evals/evaluate.py",
        "version = resolve_version(",
        id="prompt-evaluation",
    ),
    pytest.param(
        "rag-app/template/evals/evaluate.py",
        "version = resolve_version(",
        id="rag-evaluation",
    ),
    pytest.param(
        "analytics-app/template/evals/evaluate.py",
        "warehouse_id = resolve_warehouse_id(",
        id="analytics-evaluation",
    ),
    pytest.param(
        "agent-app/template/notebooks/02_enable_monitoring.py",
        "mlflow.set_experiment(",
        id="agent-monitoring",
    ),
)


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
    """Build and install the reviewed wheel so template tests cannot silently
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
    """Required answers plus identifier + combo overrides; other properties
    resolve from schema defaults inside the CLI (verified: no prompting, and
    Go-templated defaults render correctly only when left to the CLI)."""

    schema_properties = schema_for(template)["properties"]
    config = {
        key: value
        for key, value in {
            "repository_url": (
                "https://github.com/aai-template-tests/"
                f"{overrides.get('project_name', template.name)}"
            ),
            "workspace_host": IDENTIFIERS["databricks_host"],
            "compute_policy_id": IDENTIFIERS["job_compute_policy_id"],
            "aai_core_volume": IDENTIFIERS["sdk_artifact_volume"],
            # The SDK source fields are deliberately NOT supplied: the pip-source
            # default embeds `{{.aai_core_source_ref}}`, which the CLI expands only
            # for a schema default, never for an answer passed in. Leaving both out
            # exercises the release-metadata-stamped defaults.
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


@pytest.mark.parametrize("relative_path,target_work", STRICT_JUDGE_CALLS)
def test_generated_llmops_resolves_judges_canonically_before_target_work(
    relative_path: str, target_work: str
):
    source = (TEMPLATES_DIR / relative_path).read_text()
    resolution = "judge_model = judge_model_uri(context.settings)"

    assert "from aai_core.evaluation import" in source
    assert source.count(resolution) == 1
    assert "def _judge_model_uri" not in source
    assert "def judge_model_uri" not in source
    assert "ProviderConfigurationError" not in source
    assert source.index("context = bootstrap") < source.index(resolution)
    assert source.index(resolution) < source.index(target_work)


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
        assert "context.prompts.promote(" in text
        assert ".set_alias(" not in text
        assert '"--decision-run-id"' in text

        evaluation = (template / "template" / "evals" / "evaluate.py").read_text()
        assert "record_decision(" in evaluation
        assert "change_run_id=evaluation_run_id" in evaluation
        assert "prompt_version=version" in evaluation
        assert "prompt_digest=prompt_digest(" in evaluation


@pytest.mark.parametrize("template", TEMPLATES, ids=template_ids)
def test_template_deploy_workflow_is_pinned_and_has_manual_branch_ref_uat(
    template: Path,
):
    deploy = template / "template" / ".github" / "workflows" / "deploy.yml"
    text = deploy.read_text()
    workflow = yaml.safe_load(text)
    assert len(ACTION_REF.findall(text)) == len(ACTION_PIN.findall(text))
    assert "id-token: write" in text
    assert workflow[True]["workflow_dispatch"]["inputs"]["target"]["options"] == [
        "dev",
        "uat",
    ]
    assert all("environment" not in job for job in workflow["jobs"].values())
    assert "UAT_DEPLOYMENT_ENABLED" in text
    assert "DATABRICKS_UAT_HOST" in text
    assert "refs/heads/main" in text
    assert "scripts/release_artifact.py verify" in text
    assert "BUNDLE_VAR_deployment_release=$GITHUB_SHA" in text
    release_gate = "databricks bundle run release_gate -t dev"
    app_deploy = "databricks bundle run agent_app -t dev"
    assert text.index(release_gate) < text.index(app_deploy)
    assert "databricks bundle run release_gate -t uat" in text
    assert "databricks bundle run agent_app -t uat" in text
    assert "hashFiles('resources/agent_app.yml')" in text
    app_preflight = 'databricks apps get "$APP_NAME"'
    assert text.index(app_preflight) < text.index("databricks bundle deploy -t dev")
    # The deployed app is verified live AFTER it starts, on both targets: a
    # red smoke fails the workflow, so an unverified deployment cannot reach
    # UAT and the baseline is never moved onto it.
    dev_smoke = "python scripts/smoke_deployment.py --target dev"
    uat_smoke = "python scripts/smoke_deployment.py --target uat"
    assert text.index(app_deploy) < text.index(dev_smoke)
    assert text.index("databricks bundle run agent_app -t uat") < text.index(uat_smoke)
    smoke_step = text[text.index("Post-deploy smoke (dev app)") :]
    assert "hashFiles('resources/agent_app.yml')" in smoke_step


@pytest.mark.parametrize("template", TEMPLATES, ids=template_ids)
def test_template_makefile_uses_generated_virtual_environment(template: Path):
    makefile = template / "template" / "Makefile"
    assert "PYTHON ?= .venv/bin/python" in makefile.read_text()


@pytest.mark.parametrize("template", TEMPLATES, ids=template_ids)
def test_template_coverage_ratchet_is_branch_aware(template: Path):
    policy = configparser.ConfigParser()
    loaded = policy.read(template / "template" / ".coveragerc")

    assert loaded
    assert policy.getboolean("run", "branch")
    assert policy.get("run", "source") == "app"
    assert 80 <= policy.getint("report", "fail_under") <= 100


@pytest.mark.parametrize("template", TEMPLATES, ids=template_ids)
def test_template_schema_shared_contract(template: Path):
    schema = schema_for(template)
    properties = schema["properties"]
    for name in (
        "project_name",
        "application_name",
        "repository_url",
        "team",
        "owner_group",
        "cost_center",
        "catalog",
        "schema",
        "aai_core_version",
        "aai_core_source_ref",
        "aai_core_volume",
        "workspace_host",
        "uat_workspace_host",
        "compute_policy_id",
        "aai_core_pip_source",
    ):
        assert name in properties, f"{template.name} schema missing {name}"
    assert properties["repository_url"]["type"] == "string"
    assert "default" not in properties["repository_url"]
    for name, prop in properties.items():
        if name != "repository_url":
            assert "default" in prop, f"{template.name}.{name} needs a default"
        assert "description" in prop, f"{template.name}.{name} needs a description"
    orders = [prop["order"] for prop in properties.values()]
    assert len(orders) == len(
        set(orders)
    ), f"{template.name} has duplicate prompt orders"
    assert properties["project_name"]["pattern"] == "^[a-z][a-z0-9-]+$"
    if "model_provider" in properties:
        assert properties["model_provider"]["enum"] == ["databricks"]
    generated_sdk = COMPATIBILITY["sdk"]["generated_project_default"]
    assert properties["aai_core_version"]["default"] == generated_sdk["version"]
    assert (
        properties["aai_core_source_ref"]["default"] == generated_sdk["source"]["ref"]
    )
    assert (
        COMPATIBILITY["templates"][template.name]["aai_core"]
        == generated_sdk["version"]
    )
    if template.name in CURRENT_TEMPLATE_VERSIONS:
        assert (
            COMPATIBILITY["templates"][template.name]["version"]
            == CURRENT_TEMPLATE_VERSIONS[template.name]
        )
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
    if template.name == "rag-app":
        assert properties["prompt_version"]["pattern"] == "^[1-9][0-9]*$"
        assert properties["knowledge_version"]["maxLength"] == 128


# ---------------------------------------------------------------- render tier


def _assert_delivery_contract(bundle: dict) -> None:
    assert set(bundle["targets"]) == {"dev", "uat"}
    assert bundle["targets"]["dev"]["mode"] == "development"
    assert bundle["targets"]["uat"]["mode"] == "production"
    assert (
        bundle["targets"]["uat"]["workspace"]["host"]
        == IDENTIFIERS["databricks_uat_host"]
    )
    assert bundle["targets"]["uat"]["variables"] == {
        "deployment_environment": "uat",
        "deployment_lifecycle": "validation",
    }
    for target_name in ("dev", "uat"):
        preset_tags = bundle["targets"][target_name]["presets"]["tags"]
        assert REQUIRED_TAGS.issubset(preset_tags)
        assert preset_tags["environment"] == "${bundle.target}"
        assert preset_tags["lifecycle"] == "${var.deployment_lifecycle}"


@requires_cli
@pytest.mark.parametrize("template,combo", all_combo_params())
def test_render_matrix(template: Path, combo: dict, tmp_path_factory):
    output = render(template, combo, tmp_path_factory)

    assert (output / ".coveragerc").is_file()

    # T1: no unrendered template markers anywhere.
    for path in output.rglob("*"):
        if path.is_file():
            assert "{{." not in path.read_text(errors="ignore"), path

    # T2: bundle configuration parses.
    bundle = yaml.safe_load((output / "databricks.yml").read_text())
    _assert_delivery_contract(bundle)
    manifest = yaml.safe_load((output / "ai-app.yaml").read_text())
    assert manifest["spec"]["environments"]["uat"]["tags"] == {
        "environment": "uat",
        "lifecycle": "validation",
    }
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
        validation_environment = dict(os.environ)
        validation_environment["PYTHONPATH"] = str(ROOT / "src")
        subprocess.run(
            [sys.executable, str(output / "scripts" / "validate_project.py")],
            check=True,
            cwd=output,
            env=validation_environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
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
    assert (
        stamp["generated_with"]["aai_core_source_ref"]
        == COMPATIBILITY["sdk"]["generated_project_default"]["source"]["ref"]
    )
    assert (
        stamp["generated_with"]["uat_workspace_host"]
        == IDENTIFIERS["databricks_uat_host"]
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
        makefile = (output / "Makefile").read_text()
        assert bundle["variables"]["prompt_version"]["default"] == "1"
        assert bundle["variables"]["source_commit"]["default"] == "local-dev"
        assert bundle["variables"]["source_state"]["default"] == "unknown"
        assert stamp["generated_with"]["prompt_version"] == "1"
        assert "PROMPT_VERSION ?=" in makefile
        assert '--prompt-version "$(PROMPT_VERSION)"' in makefile
        help_result = subprocess.run(
            ["make", "-s", "help"],
            cwd=output,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "PROMPT_VERSION=<version>" in help_result.stdout
        release_gate = yaml.safe_load(
            (output / "resources" / "agent_job.yml").read_text()
        )["resources"]["jobs"]["release_gate"]
        assert release_gate["tasks"][0]["spark_python_task"]["parameters"] == [
            "--prompt-version",
            "${var.prompt_version}",
        ]
        assert release_gate["job_clusters"][0]["new_cluster"]["spark_env_vars"] == {
            "GIT_COMMIT": "${var.source_commit}",
            "GIT_DIRTY": "${var.source_state}",
            "AAI_ENVIRONMENT": "${var.deployment_environment}",
            "AAI_LIFECYCLE": "${var.deployment_lifecycle}",
            "AAI_RELEASE": "${var.deployment_release}",
        }
        deploy_workflow = (output / ".github/workflows/deploy.yml").read_text()
        assert "Verify immutable release evidence" in deploy_workflow
        assert "git status --porcelain --untracked-files=normal" in deploy_workflow
        assert "BUNDLE_VAR_source_commit=$GITHUB_SHA" in deploy_workflow
        assert "BUNDLE_VAR_source_state=false" in deploy_workflow
        assert (
            "${bundle.git.commit}"
            not in (output / "resources" / "agent_job.yml").read_text()
        )
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
            assert environment["AAI_ENVIRONMENT"]["value"] == (
                "${var.deployment_environment}"
            )
            assert environment["AAI_LIFECYCLE"]["value"] == (
                "${var.deployment_lifecycle}"
            )
            assert environment["AAI_RELEASE"]["value"] == ("${var.deployment_release}")
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

    _assert_evaluation_release_contract(template, output)
    _assert_rag_release_contract(template, output, bundle, stamp)


def _assert_evaluation_release_contract(template: Path, output: Path) -> None:
    if template.name != "evaluation-project":
        return
    release_gate = yaml.safe_load(
        (output / "resources" / "evaluation_job.yml").read_text()
    )["resources"]["jobs"]["release_gate"]
    assert release_gate["tasks"][0]["spark_python_task"]["parameters"] == ["--yes"]
    deployment_gate = yaml.safe_load(
        (output / "resources" / "optional" / "deployment_job.yml").read_text()
    )["resources"]["jobs"]["agent_deployment_gate"]
    assert deployment_gate["tasks"][0]["spark_python_task"]["parameters"][:2] == [
        "--yes",
        "--model-name",
    ]


def _assert_rag_release_contract(
    template: Path, output: Path, bundle: dict, stamp: dict
) -> None:
    if template.name != "rag-app":
        return
    readme = (output / "README.md").read_text()
    makefile = (output / "Makefile").read_text()
    assert bundle["variables"]["prompt_version"]["default"] == "1"
    assert bundle["variables"]["knowledge_version"]["default"] == (
        "replace-with-knowledge-version"
    )
    assert bundle["variables"]["source_commit"]["default"] == "local-dev"
    assert bundle["variables"]["source_state"]["default"] == "unknown"
    assert stamp["generated_with"]["prompt_version"] == "1"
    assert stamp["generated_with"]["knowledge_version"] == (
        "replace-with-knowledge-version"
    )
    assert "PROMPT_VERSION ?=" in makefile
    assert "KNOWLEDGE_VERSION ?=" in makefile
    help_result = subprocess.run(
        ["make", "-s", "help"],
        cwd=output,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PROMPT_VERSION=<n>" in help_result.stdout
    assert "KNOWLEDGE_VERSION=<id>" in help_result.stdout
    release_gate = yaml.safe_load((output / "resources" / "rag_job.yml").read_text())[
        "resources"
    ]["jobs"]["release_gate"]
    evaluate_task = next(
        task for task in release_gate["tasks"] if task["task_key"] == "evaluate"
    )
    assert evaluate_task["spark_python_task"]["parameters"] == [
        "--prompt-version",
        "${var.prompt_version}",
        "--knowledge-version",
        "${var.knowledge_version}",
    ]
    assert release_gate["job_clusters"][0]["new_cluster"]["spark_env_vars"] == {
        "GIT_COMMIT": "${var.source_commit}",
        "GIT_DIRTY": "${var.source_state}",
        "AAI_ENVIRONMENT": "${var.deployment_environment}",
        "AAI_LIFECYCLE": "${var.deployment_lifecycle}",
        "AAI_RELEASE": "${var.deployment_release}",
    }
    deploy_workflow = (output / ".github/workflows/deploy.yml").read_text()
    assert "Verify source checkout provenance" in deploy_workflow
    assert "git status --porcelain --untracked-files=normal" in deploy_workflow
    assert "BUNDLE_VAR_source_commit=$GITHUB_SHA" in deploy_workflow
    assert "BUNDLE_VAR_source_state=false" in deploy_workflow
    assert "--knowledge-version <knowledge-version>" in readme
    assert "--evaluation-run <run-id>" in readme


@requires_cli
@pytest.mark.generated_project
@pytest.mark.parametrize("template,combo", all_combo_params())
def test_generated_project_quality(
    template: Path,
    combo: dict,
    tmp_path_factory,
    packaged_sdk_python: Path,
):
    """Run the deep credential-free quality tier on every semantic render."""

    output = render(template, combo, tmp_path_factory)
    environment = dict(os.environ)
    # Only application source is added. aai-core must come from the reviewed
    # wheel installed by packaged_sdk_python, never ROOT/src.
    environment["PYTHONPATH"] = str(output / "src")
    for command in (
        [str(packaged_sdk_python), str(output / "scripts" / "validate_project.py")],
        [str(packaged_sdk_python), "-m", "ruff", "check", "."],
        [str(packaged_sdk_python), "-m", "black", "--check", "."],
        [
            str(packaged_sdk_python),
            "-m",
            "mypy",
            "--ignore-missing-imports",
            "--check-untyped-defs",
            "--disallow-incomplete-defs",
            "--disallow-untyped-defs",
            "src/app",
        ],
        [
            str(packaged_sdk_python),
            "-m",
            "pytest",
            "-q",
            "--cov=app",
            "--cov-report=term-missing",
            str(output / "tests"),
        ],
        [str(packaged_sdk_python), str(output / "evals" / "offline_checks.py")],
        [
            str(packaged_sdk_python),
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
        ],
    ):
        subprocess.run(command, check=True, cwd=output, env=environment)


# ------------------------------------------------------------- shared assets


def test_requirements_ci_uses_pip_source():
    for template in TEMPLATES:
        content = (template / "template" / "requirements-ci.txt.tmpl").read_text()
        assert "aai-core @ {{.aai_core_pip_source}}" in content
        schema = schema_for(template)
        source = schema["properties"]["aai_core_pip_source"]["default"]
        assert "@{{.aai_core_source_ref}}" in source
        assert "@v{{.aai_core_version}}" not in source


def _template_script(name):
    """Load one of the evaluation-project scripts as a module.

    They are plain .py files (not .tmpl), so they can be exercised without
    rendering a project — and the deployment gate's correctness is worth
    testing directly rather than only through a render.
    """

    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "templates"
        / "evaluation-project"
        / "template"
        / name
    )
    spec = importlib.util.spec_from_file_location(f"_tmpl_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_approval_prefers_the_run_the_evaluation_task_handed_over(capsys):
    """A concurrent evaluation of the same version must not win.

    The search is by model version and start time, so another manual or
    automated run finishing first would send the reviewer to evidence from
    a different dataset, config, or gate.
    """

    script = _template_script("scripts/link_deployment_job.py")
    searched = []
    script._evaluation_run_id = lambda *args: searched.append(args) or "wrong-run"

    script.await_approval("main.eval.agent", "7", "the-exact-run")

    output = capsys.readouterr().out
    assert "agentkit evidence --run the-exact-run" in output
    assert "wrong-run" not in output
    assert searched == []


def test_approval_falls_back_to_the_search_and_says_so(capsys):
    script = _template_script("scripts/link_deployment_job.py")
    script._evaluation_run_id = lambda *args: "found-by-search"

    script.await_approval("main.eval.agent", "7", "   ")

    output = capsys.readouterr().out
    assert "agentkit evidence --run found-by-search" in output
    assert "not handed over by the evaluation task" in output


def test_the_evaluation_shim_publishes_the_run_it_recorded(tmp_path, monkeypatch):
    script = _template_script("evals/evaluate.py")
    results = tmp_path / ".aai" / "agentkit" / "results"
    results.mkdir(parents=True)
    (results / "20260803T000000Z-eval.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "command": "eval",
                "recorded_at": "2026-08-03T00:00:00Z",
                "run_id": "run-from-this-job",
                "agent": "models:/main.eval.agent/7",
                "mode": "live",
                "dataset": {"ref": "d.json", "digest": "abc123", "rows": 12},
                "scope": {"mode": "full", "rows": 12},
                "metrics": {},
                "versions": {"agent": "a", "aai_core": "0.4.0"},
                "decision": "inconclusive",
                "change_id": "abc",
                "gate_passed": True,
            }
        )
    )

    runtime_imports = []
    original_import = __import__

    def track_runtime_import(name, *args, **kwargs):
        if name == "databricks.sdk.runtime":
            runtime_imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(script, "find_spec", lambda name: None)
    monkeypatch.setattr("builtins.__import__", track_runtime_import)

    # No dbutils outside a Databricks job: importantly, even importing the
    # local SDK fallback can authenticate and retry, so the shim must avoid
    # that import rather than merely catch its eventual error.
    assert script.publish_evidence_run_id(tmp_path) == "run-from-this-job"
    assert script.publish_evidence_run_id(tmp_path / "empty") is None
    assert runtime_imports == []

    writes = []
    runtime = types.ModuleType("databricks.sdk.runtime")
    runtime.dbutils = types.SimpleNamespace(
        jobs=types.SimpleNamespace(
            taskValues=types.SimpleNamespace(
                set=lambda **values: writes.append(values),
            )
        )
    )
    monkeypatch.setitem(sys.modules, "databricks.sdk.runtime", runtime)
    monkeypatch.setattr(script, "find_spec", lambda name: object())

    assert script.publish_evidence_run_id(tmp_path) == "run-from-this-job"
    assert writes == [{"key": "evidence_run_id", "value": "run-from-this-job"}]
    assert runtime_imports == ["databricks.sdk.runtime"]


def test_the_evaluation_shim_requires_explicit_spend_confirmation():
    script = _template_script("evals/evaluate.py")

    assert script.build_arguments([]) == ["eval"]
    assert script.build_arguments(["--yes"]) == ["eval", "--yes"]
    assert script.build_arguments(
        [
            "--yes",
            "--model-name",
            "main.eval.agent",
            "--model-version",
            "7",
        ]
    ) == [
        "eval",
        "--yes",
        "--agent",
        "models:/main.eval.agent/7",
        "--mode",
        "live",
    ]


# ------------------------------------------------- post-deploy smoke script


def _smoke_module(tmp_path, monkeypatch, target="dev"):
    module = _template_script("scripts/smoke_deployment.py")
    (tmp_path / ".aai-template.json").write_text(
        json.dumps({"generated_with": {"application_name": "test-agent"}})
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["smoke_deployment.py", "--target", target, "--poll-seconds", "1"]
    )
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    return module


def test_smoke_deployment_waits_for_running_then_passes_status_only(
    tmp_path, monkeypatch, capsys
):
    module = _smoke_module(tmp_path, monkeypatch)
    states = iter(
        [
            {"app_status": {"state": "DEPLOYING"}},
            {"app_status": {"state": "RUNNING"}, "url": "https://app.example"},
        ]
    )
    seen_names = []

    def fake_apps_get(app_name):
        seen_names.append(app_name)
        return next(states)

    monkeypatch.setattr(module, "_apps_get", fake_apps_get)

    code = module.main()

    assert code == 0
    assert seen_names == ["test-agent-dev", "test-agent-dev"]
    output = capsys.readouterr().out
    assert "RUNNING" in output
    assert "status only" in output


def test_smoke_deployment_fails_fast_on_a_terminal_state(tmp_path, monkeypatch, capsys):
    module = _smoke_module(tmp_path, monkeypatch, target="uat")
    monkeypatch.setattr(
        module,
        "_apps_get",
        lambda app_name: {"app_status": {"state": "ERROR", "message": "boom"}},
    )

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert excinfo.value.code == 1
    assert "test-agent-uat is ERROR" in capsys.readouterr().err


class _FakeResponse:
    def __init__(self, body, status=200):
        self._body = body
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _probed_module(tmp_path, monkeypatch, responder):
    module = _smoke_module(tmp_path, monkeypatch)
    (tmp_path / "evals" / "data").mkdir(parents=True)
    (tmp_path / "evals" / "data" / "live_probes.json").write_text(
        json.dumps(
            [
                {
                    "id": "gp-001",
                    "path": "/invocations",
                    "request": {"input": [{"role": "user", "content": "hi"}]},
                    "must_contain": ["A-1001"],
                    "max_seconds": 30,
                }
            ]
        )
    )
    monkeypatch.setattr(
        module,
        "_wait_until_running",
        lambda name, **kwargs: {
            "app_status": {"state": "RUNNING"},
            "url": "https://app.example",
        },
    )
    monkeypatch.setattr(module, "_bearer_token", lambda: "redacted-token")
    monkeypatch.setattr(module.urllib.request, "urlopen", responder)
    return module


def test_smoke_deployment_probes_log_verdicts_never_bodies(
    tmp_path, monkeypatch, capsys
):
    secret_body = b'{"output": "Order A-1001 has shipped; ssn 000-00-0000"}'
    requests = []

    def fake_urlopen(request, timeout=None):
        requests.append(request)
        return _FakeResponse(secret_body)

    module = _probed_module(tmp_path, monkeypatch, fake_urlopen)

    code = module.main()

    assert code == 0
    assert requests[0].full_url == "https://app.example/invocations"
    output = capsys.readouterr().out
    assert "gp-001: ok" in output
    assert "shipped" not in output
    assert "ssn" not in output


def test_smoke_deployment_names_the_grant_on_forbidden(tmp_path, monkeypatch, capsys):
    import io
    import urllib.error

    def forbidden(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 403, "Forbidden", hdrs=None, fp=io.BytesIO(b"")
        )

    module = _probed_module(tmp_path, monkeypatch, forbidden)

    code = module.main()

    assert code == 1
    captured = capsys.readouterr()
    assert "CAN_USE" in captured.err
    assert "docs/uat-promotion.md" in captured.err


def test_smoke_deployment_fails_on_missing_expected_content(
    tmp_path, monkeypatch, capsys
):
    module = _probed_module(
        tmp_path,
        monkeypatch,
        lambda request, timeout=None: _FakeResponse(b'{"output": "unrelated"}'),
    )

    code = module.main()

    assert code == 1
    captured = capsys.readouterr()
    assert "missing expected content 'A-1001'" in captured.err
    assert "unrelated" not in captured.err
