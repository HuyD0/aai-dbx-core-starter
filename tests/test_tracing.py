import sys
from functools import wraps

from aai_core import tracing
from aai_core.tags import ResourceContext


class FakeMlflow:
    def __init__(self):
        self.active = False
        self.experiment = None
        self.trace_metadata = None

    def set_experiment(self, name):
        self.experiment = name

    def trace(self, **trace_options):
        def decorate(target):
            @wraps(target)
            def invoke(*args, **kwargs):
                self.active = True
                try:
                    return target(*args, **kwargs)
                finally:
                    self.active = False

            return invoke

        return decorate

    def update_current_trace(self, *, metadata):
        assert self.active, "trace metadata must be applied inside an active trace"
        self.trace_metadata = metadata


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
