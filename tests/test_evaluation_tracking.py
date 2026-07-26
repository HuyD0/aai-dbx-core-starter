"""Config discovery, experiment naming convention, and tracked eval runs."""

from types import SimpleNamespace

from aai_core.evaluation import EvaluationSuite, QualityThreshold, workspace_run_url
from aai_core.experiments import ExperimentManager
from aai_core.runtime import find_platform_config
from aai_core.testing import dev_settings


def test_find_platform_config_prefers_env_then_walks_upward(tmp_path, monkeypatch):
    project = tmp_path / "repo"
    nested = project / "notebooks" / "deep"
    nested.mkdir(parents=True)
    config = project / "aai-platform.yml"
    config.write_text("platform: {}\n")

    assert find_platform_config(nested) == config

    override = tmp_path / "elsewhere.yml"
    monkeypatch.setenv("AAI_PLATFORM_CONFIG", str(override))
    assert find_platform_config(nested) == override


def test_find_platform_config_falls_back_to_conventional_path(tmp_path):
    lonely = tmp_path / "empty"
    lonely.mkdir()

    assert find_platform_config(lonely) == lonely / "aai-platform.yml"


def test_experiment_name_convention_derives_from_governed_tags():
    conventional = dev_settings(team="ml-team", application="order-agent")
    explicit = dev_settings(experiment_name="/Shared/custom")

    assert conventional.effective_experiment_name == "/Shared/ml-team-order-agent-dev"
    assert explicit.effective_experiment_name == "/Shared/custom"


class FakeMlflow:
    def __init__(self):
        self.params: dict = {}
        self.metrics: dict = {}
        self.tags: dict = {}
        self.genai = SimpleNamespace(
            evaluate=lambda **kwargs: SimpleNamespace(
                metrics={"safety/mean": 1.0, "correctness/mean": 0.8}
            )
        )

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None, nested=False):
        class _Run:
            def __enter__(self):
                return SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

            def __exit__(self, *args):
                return False

        return _Run()

    def set_tags(self, tags):
        self.tags.update(tags)

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics, step=None):
        self.metrics.update(metrics)


def test_run_tracked_links_prompt_dataset_and_verdict():
    fake = FakeMlflow()
    experiments = ExperimentManager(
        experiment_name="/Shared/test",
        context=dev_settings().resource,
        mlflow_module=fake,
    )
    suite = EvaluationSuite(
        scorers=[],
        thresholds=[
            QualityThreshold(metric="safety/mean", direction="higher", required=1.0)
        ],
        mlflow_module=fake,
    )

    report, run_id = suite.run_tracked(
        experiments=experiments,
        run_name="gate-v3",
        data=[{"inputs": {"q": "a"}}, {"inputs": {"q": "b"}}],
        prompt_uri="prompts:/main.apps.assistant/3",
        dataset_name="main.eval.golden",
        parameters={"prompt_version": 3},
    )

    assert report.passed
    assert run_id == "run-123"
    assert fake.params["prompt_uri"] == "prompts:/main.apps.assistant/3"
    assert fake.params["evaluation_dataset"] == "main.eval.golden"
    assert fake.params["case_count"] == 2
    assert fake.params["prompt_version"] == 3
    assert fake.metrics["correctness/mean"] == 0.8
    assert fake.tags["aai.gate_passed"] == "true"
    assert fake.tags["aai.team"] == "test-team"  # governed tags on the run


def test_workspace_run_url_derives_from_environment(monkeypatch):
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    assert workspace_run_url("run-1") is None

    monkeypatch.setenv("DATABRICKS_HOST", "https://adb-1.azuredatabricks.net/")
    assert (
        workspace_run_url("run-1", "exp-9")
        == "https://adb-1.azuredatabricks.net/ml/experiments/exp-9/runs/run-1"
    )
