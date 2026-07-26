import asyncio
import inspect
import sys
from functools import wraps
from types import SimpleNamespace

from aai_core import tracing
from aai_core.tags import ResourceContext


class FakeMlflow:
    def __init__(self):
        self.active = False
        self.experiment = None
        self.trace_metadata = None
        self.trace_session = None
        self.trace_updates = []
        self.openai_autologged = False
        self.langchain_autologged = False
        self.openai = SimpleNamespace(autolog=self._autolog_openai)
        self.langchain = SimpleNamespace(autolog=self._autolog_langchain)

    def _autolog_openai(self):
        self.openai_autologged = True

    def _autolog_langchain(self):
        self.langchain_autologged = True

    def set_experiment(self, name):
        self.experiment = name

    def trace(self, **trace_options):
        def decorate(target):
            if inspect.iscoroutinefunction(target):

                @wraps(target)
                async def invoke_async(*args, **kwargs):
                    self.active = True
                    try:
                        return await target(*args, **kwargs)
                    finally:
                        self.active = False

                return invoke_async

            @wraps(target)
            def invoke(*args, **kwargs):
                self.active = True
                try:
                    return target(*args, **kwargs)
                finally:
                    self.active = False

            return invoke

        return decorate

    def update_current_trace(self, **options):
        assert self.active, "trace metadata must be applied inside an active trace"
        self.trace_updates.append(options)
        if "metadata" in options:
            self.trace_metadata = options["metadata"]
        if "session_id" in options:
            self.trace_session = options["session_id"]


def test_configured_metadata_is_applied_inside_traced_call(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(tracing, "_TRACE_METADATA", {})
    context = ResourceContext(
        application="example-assistant",
        project="example-ai",
        environment="dev",
        team="data-platform",
        owner_group="group:data-platform-owners",
        cost_center="CC-1234",
        data_classification="internal",
        lifecycle="experimental",
        repository="HuyD0/aai-dbx-core-starter",
        release="dev",
    )

    tracing.configure_tracing(
        context,
        experiment_name="/Shared/example-ai",
        enable_openai_autolog=False,
    )

    assert fake_mlflow.trace_metadata is None

    @tracing.traced(span_type="CHAIN")
    def answer() -> str:
        return "ready"

    assert answer() == "ready"
    assert fake_mlflow.experiment == "/Shared/example-ai"
    assert fake_mlflow.trace_metadata == context.for_trace()


def test_async_traced_call_stays_active_until_awaited_body_finishes(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(tracing, "_TRACE_METADATA", {"release": "test"})

    @tracing.traced(span_type="CHAIN")
    async def answer() -> str:
        assert fake_mlflow.active
        await asyncio.sleep(0)
        assert fake_mlflow.active
        return "ready"

    assert inspect.iscoroutinefunction(answer)
    assert asyncio.run(answer()) == "ready"
    assert fake_mlflow.active is False
    assert fake_mlflow.trace_metadata == {"release": "test"}


def test_langchain_autolog_is_opt_in(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(tracing, "_TRACE_METADATA", {})
    context = ResourceContext(
        application="example-assistant",
        project="example-ai",
        environment="dev",
        team="data-platform",
        owner_group="group:data-platform-owners",
        cost_center="CC-1234",
        data_classification="internal",
        lifecycle="experimental",
        repository="HuyD0/aai-dbx-core-starter",
        release="dev",
    )

    tracing.configure_tracing(
        context,
        experiment_name="/Shared/example-ai",
        enable_langchain_autolog=True,
    )

    assert not fake_mlflow.openai_autologged
    assert fake_mlflow.langchain_autologged


def test_openai_autolog_is_opt_in(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(tracing, "_TRACE_METADATA", {})
    context = ResourceContext(
        application="example-assistant",
        project="example-ai",
        environment="dev",
        team="data-platform",
        owner_group="group:data-platform-owners",
        cost_center="CC-1234",
        data_classification="internal",
        lifecycle="experimental",
        repository="HuyD0/aai-dbx-core-starter",
        release="dev",
    )

    tracing.configure_tracing(
        context,
        experiment_name="/Shared/example-ai",
        enable_openai_autolog=True,
    )

    assert fake_mlflow.openai_autologged
    assert not fake_mlflow.langchain_autologged


def test_trace_session_uses_dedicated_mlflow_field(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(tracing, "_TRACE_METADATA", {})

    @tracing.traced(span_type="CHAIN")
    def answer() -> str:
        tracing.set_trace_session("conversation-123")
        return "ready"

    assert answer() == "ready"
    assert fake_mlflow.trace_session == "conversation-123"
    assert fake_mlflow.trace_updates == [{"session_id": "conversation-123"}]
