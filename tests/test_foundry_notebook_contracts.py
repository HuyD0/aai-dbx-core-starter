"""Focused API and resource-lifecycle contracts for Foundry lessons 11-12."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "examples" / "foundry-curriculum" / "notebooks"


def code_cell_source(notebook_name, cell_id):
    notebook = json.loads((NOTEBOOKS / notebook_name).read_text(encoding="utf-8"))
    return next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("id") == cell_id
    )


def test_mlflow_retriever_output_uses_the_document_315_contract():
    source = code_cell_source(
        "11_mlflow_tracing_and_genai_evaluation.ipynb",
        "mlflow-manual-trace",
    )
    tree = ast.parse(source)
    document_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Document"
    )
    keywords = {keyword.arg: keyword.value for keyword in document_call.keywords}
    assert set(keywords) == {"page_content", "metadata", "id"}
    assert isinstance(keywords["metadata"], ast.Dict)
    metadata_keys = {
        node.value
        for node in keywords["metadata"].keys
        if isinstance(node, ast.Constant)
    }
    assert {"doc_uri", "chunk_id", "classification", "freshness"} <= metadata_keys
    assert "from mlflow.entities import Document, SpanType" in source


def test_connected_dual_export_always_releases_owned_telemetry_resources():
    configuration_source = code_cell_source(
        "12_dual_otel_export_foundry_and_mlflow.ipynb",
        "dual-connected-configuration",
    )
    trace_source = code_cell_source(
        "12_dual_otel_export_foundry_and_mlflow.ipynb",
        "dual-connected-trace",
    )
    assert "RUN_DUAL_EXPORT = False" in configuration_source
    assert "instrumentor = AIProjectInstrumentor()" in configuration_source
    assert "return instrumentor, provider" in configuration_source

    trace_tree = ast.parse(trace_source)
    cleanup = next(
        node
        for node in ast.walk(trace_tree)
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.Attribute) and child.attr == "uninstrument"
            for final_node in node.finalbody
            for child in ast.walk(final_node)
        )
    )
    nested_cleanup = next(
        node for node in cleanup.finalbody if isinstance(node, ast.Try)
    )
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "uninstrument"
        for statement in nested_cleanup.body
        for node in ast.walk(statement)
    )
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "shutdown"
        for statement in nested_cleanup.finalbody
        for node in ast.walk(statement)
    )
    assert trace_source.index("configure_connected_dual_export()") < trace_source.index(
        "try:"
    )
    compile(configuration_source, "dual-connected-configuration", "exec")
    compile(trace_source, "dual-connected-trace", "exec")
