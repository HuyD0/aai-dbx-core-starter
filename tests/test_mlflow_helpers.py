from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from aai_core.evaluation import EvaluationDatasetManager
from aai_core.experiments import ExperimentManager
from aai_core.prompts import PromptManager
from aai_core.secrets import SecretValue
from aai_core.tags import ResourceContext


def context():
    return ResourceContext(
        application="claims-agent",
        project="claims",
        environment="dev",
        team="claims-ai",
        owner_group="group:claims-ai-owners",
        cost_center="CC-1042",
        data_classification="internal",
        lifecycle="experimental",
        repository="org/claims",
        release="1.0.0",
    )


class FakeMlflow:
    def __init__(self):
        self.experiment = None
        self.tags = {}
        self.parameters = {}
        self.genai = FakeGenAI()

    def set_experiment(self, name):
        self.experiment = name

    @contextmanager
    def start_run(self, **kwargs):
        yield SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

    def set_tags(self, tags):
        self.tags = tags

    def log_params(self, parameters):
        self.parameters = parameters


class FakeGenAI:
    def __init__(self):
        self.alias = None
        self.datasets = FakeDatasets()

    def register_prompt(self, **kwargs):
        return SimpleNamespace(name=kwargs["name"], version=1, kwargs=kwargs)

    def load_prompt(self, uri, **kwargs):
        return SimpleNamespace(uri=uri, kwargs=kwargs)

    def set_prompt_alias(self, **kwargs):
        self.alias = kwargs


class FakeDataset:
    def __init__(self, name, tags):
        self.name = name
        self.tags = tags
        self.records = []

    def merge_records(self, records):
        self.records.extend(records)
        return self


class FakeDatasets:
    def __init__(self):
        self.created = None

    def create_dataset(self, **kwargs):
        self.created = FakeDataset(kwargs["name"], kwargs["tags"])
        return self.created

    def get_dataset(self, **kwargs):
        return kwargs


def test_experiment_manager_attaches_platform_tags_and_parameters():
    mlflow = FakeMlflow()
    manager = ExperimentManager(
        experiment_name="/Shared/claims",
        context=context(),
        mlflow_module=mlflow,
    )

    with manager.run(run_name="candidate", parameters={"temperature": 0.1}):
        pass

    assert mlflow.experiment == "/Shared/claims"
    assert mlflow.tags["aai.application"] == "claims-agent"
    assert mlflow.parameters == {"temperature": 0.1}


def test_experiment_manager_refuses_secret_parameters():
    manager = ExperimentManager(
        experiment_name="/Shared/claims",
        context=context(),
        mlflow_module=FakeMlflow(),
    )

    with pytest.raises(ValueError, match="sensitive"):
        with manager.run(
            run_name="unsafe",
            parameters={"vendor_api_key": SecretValue("do-not-log")},
        ):
            pass


def test_prompt_manager_qualifies_registers_and_loads_versions():
    mlflow = FakeMlflow()
    manager = PromptManager(
        context=context(),
        catalog="main",
        schema="claims",
        mlflow_module=mlflow,
    )

    registered = manager.register(
        "system",
        "Question: {{question}}",
        commit_message="initial",
    )
    loaded = manager.load("system", version=registered.version)
    manager.set_alias("system", alias="candidate", version=registered.version)

    assert registered.name == "main.claims.system"
    assert loaded.uri == "prompts:/main.claims.system/1"
    assert mlflow.genai.alias["alias"] == "candidate"


def test_prompt_manager_rejects_uncontrolled_aliases():
    manager = PromptManager(
        context=context(),
        catalog="main",
        schema="claims",
        mlflow_module=FakeMlflow(),
    )

    with pytest.raises(ValueError, match="Unsupported"):
        manager.set_alias("system", alias="latest-prod", version=1)


def test_evaluation_dataset_is_tagged_and_populated():
    mlflow = FakeMlflow()
    manager = EvaluationDatasetManager(
        context=context(),
        experiment_id="experiment-1",
        mlflow_module=mlflow,
    )

    dataset = manager.create(
        "release-suite",
        records=[{"inputs": {"question": "hello"}}],
    )

    assert dataset.tags["aai.application"] == "claims-agent"
    assert dataset.records[0]["inputs"]["question"] == "hello"
