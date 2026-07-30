import traceback
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aai_console.hub.jobs import (
    DatabricksJobRunner,
    JobIdempotencyConflictError,
    JobLaunchError,
    JobLaunchRequest,
    JobRunnerMode,
    JobRunnerUnavailableError,
    JobRunState,
    RecordingJobRunner,
    UnavailableJobRunner,
)


def request(**changes):
    values = {
        "job_id": "123456",
        "idempotency_token": "evaluation:claims:v1",
        "parameters": {
            "application_id": "claims_assistant",
            "application_version": "1.2.3",
            "dataset_version": "sha256:abc123",
            "environment": "dev",
        },
    }
    values.update(changes)
    return JobLaunchRequest(**values)


def test_launch_request_is_strict_canonical_and_immutable():
    launch = request(
        job_id=" 123456 ",
        idempotency_token=" evaluation:claims:v1 ",
        parameters={"zeta": "last", "alpha": "first"},
    )

    assert launch.job_id == "123456"
    assert launch.idempotency_token == "evaluation:claims:v1"
    assert list(launch.parameters) == ["alpha", "zeta"]
    assert launch.model_dump()["parameters"] == {
        "alpha": "first",
        "zeta": "last",
    }
    with pytest.raises(TypeError):
        launch.parameters["another"] = "value"
    with pytest.raises(ValidationError):
        JobLaunchRequest(
            job_id="1",
            idempotency_token="safe",
            parameters={},
            unexpected=True,
        )
    with pytest.raises(ValidationError):
        JobLaunchRequest(
            job_id=123,
            idempotency_token="safe",
            parameters={},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_id", "0"),
        ("job_id", "-1"),
        ("job_id", "job-key"),
        ("job_id", "9223372036854775808"),
        ("idempotency_token", ""),
        ("idempotency_token", "contains spaces"),
        ("idempotency_token", "x" * 65),
    ],
)
def test_launch_request_rejects_invalid_ids(field, value):
    with pytest.raises(ValidationError):
        request(**{field: value})


@pytest.mark.parametrize(
    "parameters",
    [
        {"client_secret": "not-even-a-real-secret"},
        {"access-token": "not-even-a-real-secret"},
        {"payload": "Authorization=Bearer redacted"},
        {"payload": "https://user:password@example.test/path"},
        {"payload": "-----BEGIN PRIVATE KEY----- redacted"},
        {"bad key": "value"},
        {"ok": ""},
        {"ok": "line1\nline2"},
        {"ok": "\N{SNOWMAN}"},
        {"ok": "x" * 2_049},
        {f"k{index}": "value" for index in range(65)},
    ],
)
def test_launch_request_rejects_credential_bearing_or_unsafe_parameters(parameters):
    with pytest.raises(ValidationError) as error:
        request(parameters=parameters)

    assert "not-even-a-real-secret" not in str(error.value)
    assert "password@example.test" not in str(error.value)


def test_launch_request_enforces_databricks_payload_size():
    parameters = {f"key{index}": "x" * 2_000 for index in range(6)}

    with pytest.raises(ValidationError, match="10000"):
        request(parameters=parameters)


def test_unavailable_runner_fails_closed():
    runner = UnavailableJobRunner()

    assert runner.capability.mode is JobRunnerMode.UNAVAILABLE
    assert runner.capability.enabled is False
    assert runner.capability.remote_execution is False
    with pytest.raises(
        JobRunnerUnavailableError,
        match="not configured",
    ):
        runner.launch(request())


def test_recording_runner_is_deterministic_and_idempotent():
    runner = RecordingJobRunner()
    launch = request()

    first = runner.launch(launch)
    duplicate = runner.launch(launch)
    second = runner.launch(
        request(
            idempotency_token="evaluation:claims:v2",
            parameters={"application_version": "2.0.0"},
        )
    )

    assert runner.capability.mode is JobRunnerMode.PREVIEW
    assert runner.capability.remote_execution is False
    assert first == duplicate
    assert first.run_id == "preview-run-000001"
    assert first.run_page_url == (
        "https://local.invalid/jobs/123456/runs/preview-run-000001"
    )
    assert first.state is JobRunState.QUEUED
    assert first.preview is True
    assert second.run_id == "preview-run-000002"
    assert runner.requests[0] == launch
    assert len(runner.requests) == 2


def test_recording_runner_rejects_conflicting_idempotency_reuse():
    runner = RecordingJobRunner()
    runner.launch(request())

    with pytest.raises(JobIdempotencyConflictError, match="already used"):
        runner.launch(request(job_id="999999"))

    assert len(runner.requests) == 1


class FakeJobs:
    def __init__(self):
        self.run_now_arguments = None
        self.get_run_arguments = None

    def run_now(self, **arguments):
        self.run_now_arguments = arguments
        return SimpleNamespace(run_id=987654)

    def get_run(self, **arguments):
        self.get_run_arguments = arguments
        return SimpleNamespace(
            run_page_url="https://workspace.example/jobs/123456/runs/987654",
            state=SimpleNamespace(life_cycle_state=SimpleNamespace(value="PENDING")),
        )


def test_databricks_runner_is_lazy_and_launches_with_safe_job_parameters():
    jobs = FakeJobs()
    factory_calls = []

    def client_factory():
        factory_calls.append(True)
        return SimpleNamespace(jobs=jobs)

    runner = DatabricksJobRunner(client_factory)
    assert factory_calls == []
    assert runner.capability.mode is JobRunnerMode.DATABRICKS
    assert runner.capability.remote_execution is True

    result = runner.launch(request())

    assert factory_calls == [True]
    assert jobs.run_now_arguments == {
        "job_id": 123456,
        "idempotency_token": "evaluation:claims:v1",
        "job_parameters": {
            "application_id": "claims_assistant",
            "application_version": "1.2.3",
            "dataset_version": "sha256:abc123",
            "environment": "dev",
        },
    }
    assert jobs.get_run_arguments == {"run_id": 987654}
    assert result.run_id == "987654"
    assert result.run_page_url == ("https://workspace.example/jobs/123456/runs/987654")
    assert result.state is JobRunState.PENDING
    assert result.preview is False


def test_databricks_runner_does_not_wait_and_accepts_response_run_id():
    class ResponseOnlyJobs(FakeJobs):
        def run_now(self, **arguments):
            self.run_now_arguments = arguments
            return SimpleNamespace(response=SimpleNamespace(run_id=42))

    jobs = ResponseOnlyJobs()
    runner = DatabricksJobRunner(lambda: SimpleNamespace(jobs=jobs))

    result = runner.launch(request())

    assert result.run_id == "42"


def test_databricks_runner_returns_run_id_when_initial_status_read_fails():
    class EventuallyConsistentJobs(FakeJobs):
        def get_run(self, **arguments):
            raise RuntimeError("credential-sentinel-that-must-not-escape")

    runner = DatabricksJobRunner(
        lambda: SimpleNamespace(jobs=EventuallyConsistentJobs())
    )

    result = runner.launch(request())

    assert result.run_id == "987654"
    assert result.run_page_url is None
    assert result.state is JobRunState.QUEUED


def test_databricks_runner_sanitizes_authentication_and_launch_errors():
    sentinel = "credential-sentinel-that-must-not-escape"

    def failed_factory():
        raise RuntimeError(sentinel)

    with pytest.raises(JobRunnerUnavailableError) as auth_error:
        DatabricksJobRunner(failed_factory).launch(request())
    assert sentinel not in str(auth_error.value)
    assert sentinel not in "".join(traceback.format_exception(auth_error.value))

    class FailedJobs:
        def run_now(self, **arguments):
            raise RuntimeError(sentinel)

    with pytest.raises(JobLaunchError) as launch_error:
        DatabricksJobRunner(lambda: SimpleNamespace(jobs=FailedJobs())).launch(
            request()
        )
    assert sentinel not in str(launch_error.value)
    assert sentinel not in "".join(traceback.format_exception(launch_error.value))


def test_databricks_runner_rejects_missing_run_id_without_fake_success():
    class MissingRunIdJobs:
        def run_now(self, **arguments):
            return SimpleNamespace()

    runner = DatabricksJobRunner(lambda: SimpleNamespace(jobs=MissingRunIdJobs()))

    with pytest.raises(JobLaunchError, match="without returning a run ID"):
        runner.launch(request())
