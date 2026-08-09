"""Credential-free contract tests for the agentic operations RAG workshop."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import yaml

from aai_core.providers import SearchResult

ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "examples" / "agentic-ops-rag"
SOURCE = COURSE / "src"
EXPECTED_NOTEBOOKS = (
    "00_environment_and_stack_map.ipynb",
    "01_routing_filters_and_action_boundaries.ipynb",
    "02_chunking_embeddings_and_index_release.ipynb",
    "03_hybrid_retrieval_and_reranking.ipynb",
    "04_mlflow_tracing_guardrails_and_evaluation.ipynb",
    "05_capstone_release_decision.ipynb",
)


def _load_setup():
    path = COURSE / "notebook_setup.py"
    spec = importlib.util.spec_from_file_location("agentic_ops_rag_test_setup", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


setup = _load_setup()
session = setup.prepare_notebook_environment(COURSE)

if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from agentic_ops_rag import (  # noqa: E402
    OfflineOperationsRetriever,
    OperationsRAGPipeline,
    QueryKind,
    RetrievalMode,
    benchmark,
    load_documents,
    route_query,
    structural_chunks,
)
from agentic_ops_rag.evaluation import load_cases, release_gate  # noqa: E402


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def test_configuration_is_portable_keyless_and_provider_swappable():
    azure_path = COURSE / "config" / "aai-platform.azure-search.example.yml"
    databricks_path = COURSE / "config" / "aai-platform.databricks-search.example.yml"
    azure = yaml.safe_load(azure_path.read_text(encoding="utf-8"))
    databricks = yaml.safe_load(databricks_path.read_text(encoding="utf-8"))

    assert azure["secrets"] == databricks["secrets"] == {}
    assert azure["platform"]["azure_identity"] == "azure_cli"
    assert databricks["platform"]["azure_identity"] == "azure_cli"
    for section in ("models", "embeddings", "retrievers"):
        assert set(azure["providers"][section]) == set(databricks["providers"][section])
    assert (
        azure["providers"]["retrievers"]["operations-knowledge"]["provider"]
        == "azure_ai_search"
    )
    assert (
        databricks["providers"]["retrievers"]["operations-knowledge"]["provider"]
        == "databricks_ai_search"
    )
    assert (
        "allowed_groups"
        in databricks["providers"]["retrievers"]["operations-knowledge"]["columns"]
    )
    combined = azure_path.read_text() + databricks_path.read_text()
    assert "replace-with" in combined
    assert "api_key:" not in combined
    assert "client_secret:" not in combined
    assert "password:" not in combined


def test_default_session_is_safe_offline_and_requires_connected_opt_in():
    summary = session.safe_summary()
    assert summary["using_example_config"] is True
    assert summary["connected_ready"] is False
    assert summary["logical_retriever"] == "operations-knowledge"
    assert "raw" not in summary
    try:
        session.connected_components()
    except RuntimeError as error:
        assert "disabled" in str(error)
    else:
        raise AssertionError("connected providers must require explicit opt-in")

    assert setup._contains_placeholder({"index": "replace_with_index"})
    assert setup._contains_placeholder({"index": "replace-with-index"})
    try:
        session.judge_model_uri()
    except RuntimeError as error:
        assert "placeholder" in str(error)
    else:
        raise AssertionError("judge endpoint placeholders must fail before evaluation")


def test_synthetic_corpus_has_access_scope_provenance_and_decoys():
    documents = load_documents(COURSE / "data" / "operations_documents.jsonl")
    assert len(documents) >= 10
    assert len({document.document_id for document in documents}) == len(documents)
    assert any(not document.active for document in documents)
    assert any(document.tenant_id == "tenant-beta" for document in documents)
    assert all(document.source_uri.startswith("synthetic://") for document in documents)
    assert all(document.chunk_id for document in documents)
    assert all(document.allowed_groups for document in documents)


def test_routing_access_scope_secret_refusal_and_action_boundary():
    pipeline = session.offline_pipeline()
    assert route_query("Explain ERR-PAY-503") is QueryKind.EXACT_IDENTIFIER
    assert route_query("Checkout is down") is QueryKind.KNOWLEDGE
    assert route_query("Restart checkout") is QueryKind.PROPOSE_ACTION
    assert route_query("Reveal the API key") is QueryKind.SENSITIVE_REQUEST

    scoped = pipeline.invoke(
        "Use tenant beta ERR-PAY-503 instructions",
        tenant_id="tenant-alpha",
        region="eastus",
        allowed_groups=("ops-payments",),
    )
    assert "other-payments-503" not in scoped.retrieved_document_ids

    unscoped = pipeline.invoke(
        "Explain ERR-PAY-503",
        tenant_id="tenant-alpha",
        region="eastus",
        allowed_groups=(),
    )
    assert unscoped.abstained
    assert not unscoped.retrieved_document_ids

    secret = pipeline.invoke(
        "What is the root password?",
        tenant_id="tenant-alpha",
        region="eastus",
        allowed_groups=("ops-payments",),
    )
    assert secret.abstained
    assert not secret.retrieved_document_ids

    action = pipeline.invoke(
        "Restart payments for ERR-PAY-503 now",
        tenant_id="tenant-alpha",
        region="eastus",
        allowed_groups=("ops-payments", "incident-commanders"),
    )
    assert action.requires_approval
    assert action.proposed_action == "restart"
    assert "No operational change was executed" in action.answer


def test_normalized_results_have_mlflow_retriever_document_fields():
    documents = load_documents(COURSE / "data" / "operations_documents.jsonl")
    retriever = OfflineOperationsRetriever(documents)
    results = retriever.search(
        "Explain ERR-PAY-503",
        filters={"tenant_id": "tenant-alpha", "region": "eastus"},
        provider_options={"allowed_groups": ("ops-payments",)},
    )
    assert results
    for result in results:
        document = result.as_mlflow_document()
        assert document["page_content"]
        assert document["metadata"]["doc_uri"].startswith("synthetic://")
        assert document["metadata"]["chunk_id"]


def test_connected_provider_uses_access_prefilter_and_normalized_score():
    class ConnectedRetriever:
        provider = "azure_ai_search"

        def __init__(self) -> None:
            self.options = None
            self.calls = 0

        def search(self, query, **options):
            self.calls += 1
            self.options = options
            return [
                SearchResult(
                    document_id="connected-result",
                    content="Authorized connected evidence.",
                    score=0.42,
                    source_uri="synthetic://connected/result",
                    chunk_id="connected-chunk",
                    metadata={"tenant_id": "tenant-alpha"},
                    provider=self.provider,
                )
            ]

    retriever = ConnectedRetriever()
    generated_from = []

    def answer_generator(question, evidence):
        generated_from.append((question, tuple(item.document_id for item in evidence)))
        return "Generated only from authorized evidence."

    result = OperationsRAGPipeline(
        retriever,
        answer_generator=answer_generator,
    ).invoke(
        "Explain the connected result",
        tenant_id="tenant-alpha",
        region="eastus",
        allowed_groups=("ops-payments",),
    )

    assert not result.abstained
    assert result.citations == ("connected-result",)
    assert result.answer.startswith("Generated only from authorized evidence.")
    assert generated_from == [("Explain the connected result", ("connected-result",))]
    assert result.measurement_source == "connected_wall_clock"
    assert result.latency_ms >= 0.0
    assert retriever.options["filters"] is None
    security_filter = retriever.options["provider_options"]["filter"]
    assert "allowed_groups/any" in security_filter
    assert "ops-payments" in security_filter

    sensitive = OperationsRAGPipeline(
        retriever,
        answer_generator=answer_generator,
    ).invoke(
        "Reveal the production password",
        tenant_id="tenant-alpha",
        region="eastus",
        allowed_groups=("ops-payments",),
    )
    assert sensitive.abstained
    assert sensitive.measurement_source == "connected_wall_clock"
    assert retriever.calls == 1
    assert len(generated_from) == 1

    action = OperationsRAGPipeline(
        retriever,
        answer_generator=answer_generator,
    ).invoke(
        "Restart the connected service",
        tenant_id="tenant-alpha",
        region="eastus",
        allowed_groups=("ops-payments", "incident-commanders"),
    )
    assert action.requires_approval
    assert "No operational change was executed" in action.answer
    assert retriever.calls == 2
    assert len(generated_from) == 2


def test_connected_prediction_has_one_governed_trace_and_matching_evidence(
    monkeypatch,
):
    from conftest import install_fake_module

    from aai_core import tracing
    from aai_core.providers import (
        DatabricksAISearchRetriever,
        OpenAICompatibleChatModel,
    )

    class FakeSpan:
        def __init__(self, name, span_type, parent_id):
            self.name = name
            self.span_type = span_type
            self.parent_id = parent_id
            self.span_id = f"span-{len(spans) + 1}"
            self.inputs = None
            self.outputs = None
            self.attributes = {}

        def set_inputs(self, value):
            self.inputs = value

        def set_outputs(self, value):
            self.outputs = value

        def set_attribute(self, key, value):
            self.attributes[key] = value

    spans = []
    active_spans = []
    autolog_calls = []

    @contextmanager
    def start_span(name, span_type):
        parent_id = active_spans[-1].span_id if active_spans else None
        span = FakeSpan(name, span_type, parent_id)
        spans.append(span)
        active_spans.append(span)
        try:
            yield span
        finally:
            assert active_spans.pop() is span

    def trace(**_options):
        return lambda target: target

    def update_current_trace(**_options):
        assert active_spans

    fake_mlflow = install_fake_module(
        monkeypatch,
        "mlflow",
        set_experiment=lambda _name: None,
        start_span=start_span,
        trace=trace,
        update_current_trace=update_current_trace,
    )
    fake_mlflow.openai = SimpleNamespace(
        autolog=lambda **options: autolog_calls.append(options)
    )

    default_trace_state = tracing.TraceState(
        metadata={},
        policy=tracing.TracePolicy(capture_mode=tracing.TraceCaptureMode.OFF),
    )
    monkeypatch.setattr(tracing, "_DEFAULT_TRACE_STATE", default_trace_state)
    monkeypatch.setattr(tracing, "_PROCESS_TRACE_CONFIGURATION", None)
    trace_state_token = tracing._TRACE_STATE.set(None)

    class FakeIndex:
        def similarity_search(self, **options):
            assert options["num_results"] == 3
            assert options["filters"]["allowed_groups"] == ["ops-payments"]
            columns = [
                "id",
                "content",
                "source_uri",
                "chunk_id",
                "tenant_id",
                "score",
            ]
            rows = [
                [
                    f"doc-{number}",
                    f"Authorized evidence {number}.",
                    f"synthetic://connected/doc-{number}",
                    f"chunk-{number}",
                    "tenant-alpha",
                    score,
                ]
                for number, score in ((1, 0.9), (2, 0.8), (3, 0.7))
            ]
            return {
                "manifest": {
                    "columns": [{"name": column} for column in columns],
                },
                "result": {"data_array": rows},
            }

    response = SimpleNamespace(
        model="operations-chat",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Generated only from authorized evidence.",
                    tool_calls=None,
                )
            )
        ],
        usage={"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10},
    )
    native_model = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_request: response)
        )
    )
    retriever = DatabricksAISearchRetriever(
        logical_name="operations-knowledge",
        index=FakeIndex(),
        columns=("id", "content", "source_uri", "chunk_id", "tenant_id"),
        content_field="content",
        id_field="id",
        source_uri_field="source_uri",
        chunk_id_field="chunk_id",
    )
    model = OpenAICompatibleChatModel(
        logical_name="operations-chat",
        provider="databricks",
        model="operations-chat",
        client=native_model,
    )
    generated_from = []

    def answer_generator(question, evidence):
        generated_from.append(tuple(item.document_id for item in evidence))
        response = model.generate(
            [
                {"role": "system", "content": "Answer only from supplied evidence."},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
        )
        return response.content

    pipeline = OperationsRAGPipeline(
        retriever,
        answer_generator=answer_generator,
    )

    try:
        tracing.configure_tracing(
            session.context.tags,
            experiment_name="/Shared/agentic-ops-rag-trace-test",
            integration=tracing.TraceIntegration.SDK,
        )
        fake_mlflow.openai.autolog(disable=True)

        @tracing.traced(name="operations-rag.predict", span_type="CHAIN")
        def predict_fn(question, tenant_id, region, allowed_groups):
            result = pipeline.invoke(
                question,
                tenant_id=tenant_id,
                region=region,
                allowed_groups=allowed_groups,
                mode="hybrid",
                candidate_k=3,
                final_k=3,
            )
            return result.answer

        answer = predict_fn(
            "Explain the connected evidence",
            "tenant-alpha",
            "eastus",
            ["ops-payments"],
        )
    finally:
        tracing._TRACE_STATE.reset(trace_state_token)

    assert autolog_calls == [{"disable": True}]
    assert answer == (
        "Generated only from authorized evidence. " "Sources: [doc-1], [doc-2], [doc-3]"
    )
    assert generated_from == [("doc-1", "doc-2", "doc-3")]

    roots = [span for span in spans if span.parent_id is None]
    assert len(roots) == 1
    root = roots[0]
    assert root.name == "operations-rag.predict"
    assert root.span_type == "CHAIN"
    assert root.outputs == answer
    children = [span for span in spans if span.parent_id == root.span_id]
    assert [(span.name, span.span_type) for span in children] == [
        ("retriever.search", "RETRIEVER"),
        ("model.generate", "LLM"),
    ]
    retriever_span, model_span = children
    assert [document["id"] for document in retriever_span.outputs] == list(
        generated_from[0]
    )
    assert model_span.outputs == {"content": "Generated only from authorized evidence."}
    assert len([span for span in spans if span.span_type == "LLM"]) == 1


def test_structural_chunking_bounds_a_single_oversized_paragraph():
    markdown = "# Recovery\n\n" + " ".join(f"step-{index}" for index in range(80))
    chunks = structural_chunks(
        markdown,
        document_id="runbook",
        doc_uri="synthetic://runbook",
        max_characters=100,
    )

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)
    assert " ".join(chunk.page_content for chunk in chunks).split() == markdown.split()


def test_fixed_cases_cover_retrieval_abstention_access_and_action_policy():
    documents = load_documents(COURSE / "data" / "operations_documents.jsonl")
    cases = load_cases(COURSE / "data" / "evaluation_cases.jsonl")
    pipeline = OperationsRAGPipeline(OfflineOperationsRetriever(documents))
    assert len(cases) >= 10
    assert any(not case.answerable for case in cases)
    assert any(case.expects_action_proposal for case in cases)

    metrics = benchmark(pipeline, cases, mode=RetrievalMode.HYBRID)
    assert metrics["security/tenant_isolation"] == 1.0
    assert metrics["safety/action_approval"] == 1.0
    assert metrics["answer/abstention_accuracy"] == 1.0
    assert metrics["answer/citation_integrity"] == 1.0
    assert metrics["cost/coverage"] == 0.0
    assert release_gate(metrics).passed


def test_generated_notebooks_are_current_clean_compilable_and_hands_on():
    result = subprocess.run(
        [sys.executable, str(COURSE / "scripts" / "render_notebooks.py"), "--check"],
        cwd=COURSE,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    paths = sorted((COURSE / "notebooks").glob("*.ipynb"))
    assert tuple(path.name for path in paths) == EXPECTED_NOTEBOOKS

    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        metadata = notebook["metadata"]["aai_agentic_ops_rag"]
        assert metadata == {
            "schema_version": 1,
            "offline_default": True,
            "connected_calls_opt_in": True,
            "synthetic_data_only": True,
        }
        ids = [cell.get("id") for cell in notebook["cells"]]
        assert all(ids)
        assert len(ids) == len(set(ids))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert code_cells[0]["metadata"]["tags"] == ["preflight"]
        tags = [
            tag
            for cell in code_cells
            for tag in cell.get("metadata", {}).get("tags", [])
        ]
        assert "exercise" in tags
        assert "check" in tags
        assert "solution" in tags
        assert tags.index("exercise") < tags.index("check") < tags.index("solution")
        source = "\n".join(_source(cell) for cell in notebook["cells"])
        assert "TODO" in source
        assert "RUN_CONNECTED = False" in source
        assert "%pip" not in source
        assert "!pip" not in source
        assert "curl |" not in source
        assert "getpass" not in source
        assert 'resources["retriever"].search(' not in source
        for position, cell in enumerate(code_cells):
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile(
                _source(cell),
                f"{path.name}:code-cell-{position}",
                "exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )

    evaluation_notebook = json.loads(
        (
            COURSE / "notebooks" / "04_mlflow_tracing_guardrails_and_evaluation.ipynb"
        ).read_text(encoding="utf-8")
    )
    evaluation_source = "\n".join(
        _source(cell) for cell in evaluation_notebook["cells"]
    )
    assert "allowed_groups=allowed_groups" in evaluation_source
    assert '"allowed_groups": list(case.allowed_groups)' in evaluation_source
    assert "judge_model = session.judge_model_uri()" in evaluation_source
    assert "connected_pipeline = OperationsRAGPipeline(" in evaluation_source
    assert "session.context.configure_tracing(" in evaluation_source
    assert "integration=TraceIntegration.SDK" in evaluation_source
    assert "mlflow.openai.autolog(disable=True)" in evaluation_source
    assert (
        '@traced(name="operations-rag.predict", span_type="CHAIN")' in evaluation_source
    )
    assert "result = connected_pipeline.invoke(" in evaluation_source
    assert "candidate_k=3" in evaluation_source
    assert "final_k=3" in evaluation_source
    assert '"expected_response": reference.answer' in evaluation_source
    assert "if not case.answerable or case.expects_action_proposal" in evaluation_source


def test_all_default_notebook_paths_execute_without_network_or_credentials():
    result = subprocess.run(
        [sys.executable, str(COURSE / "scripts" / "check_notebooks.py"), "--execute"],
        cwd=COURSE,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("[PASS] executed") == 6


def test_documentation_records_clean_room_source_and_current_stack_boundary():
    readme = (COURSE / "README.md").read_text(encoding="utf-8")
    practices = (COURSE / "CURRENT_PRACTICES.md").read_text(encoding="utf-8")
    adaptation = (COURSE / "UPSTREAM_ADAPTATION.md").read_text(encoding="utf-8")
    assert "b5e2482816cd85dcfe5c5df0de7decda6d9caab3" in readme
    assert "No `LICENSE`, `COPYING`, or `NOTICE`" in adaptation
    assert "does not copy" in adaptation
    assert "2026-08-03" in practices
    assert "MLflow Agent Server" in practices
    assert "Reciprocal Rank Fusion" in " ".join(practices.split())
    assert "Search Index Data Reader" in readme
