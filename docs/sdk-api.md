# SDK public API

`aai-core` keeps its root namespace deliberately small. Most applications begin
with the four supported top-level exports:

```python
from aai_core import PlatformContext, PlatformSettings, __version__, bootstrap
```

`bootstrap()` loads `aai-platform.yml` and returns a `PlatformContext`. Use the
context as a synchronous context manager so SDK-created clients are closed:

```python
from aai_core import bootstrap

with bootstrap() as context:
    model = context.providers.model("general-chat")
    response = model.generate([{"role": "user", "content": "Hello"}])
```

The domain modules hold the rest of the public API. A name is supported when it
is listed in that module's `__all__`; its docstring is the symbol-level
reference. Feature maturity and required dependency extras remain defined by
[`compatibility.json`](../compatibility.json), and compatibility rules are in
the [versioning policy](versioning.md).

| Domain module | Purpose | Main public entry points |
|---|---|---|
| `aai_core.context`, `aai_core.runtime` | Configuration and the process composition root | `bootstrap`, `PlatformContext`, `PlatformSettings` |
| `aai_core.tags` | Validated ownership, cost, environment, lifecycle, and request-tag context | `ResourceContext`, `LifecycleStage`, `DataClassification` |
| `aai_core.providers` | Logical model, embedding, and retrieval capabilities with native-client access | `ProviderResolver`, `ChatModel`, `EmbeddingProvider`, `Retriever`, `RetrievalMode`, `AzureSemanticRankOptions`, `DatabricksRerankOptions`, `SearchResult` |
| `aai_core.structured` | Strict structured generation over the stable chat contract | `generate_structured`, `generate_typed` |
| `aai_core.secrets`, `aai_core.identity` | Non-secret references and provider-native keyless identity | `SecretResolver`, `SecretRef`, `SecretValue`, `azure_credential` |
| `aai_core.tracing`, `aai_core.logging` | Governed MLflow tracing, resource projection, structured logging, and redaction | `configure_tracing`, `TracePolicy`, `traced`, `configure_logging` |
| `aai_core.experiments`, `aai_core.prompts` | Reproducible MLflow runs and immutable prompt references | `ExperimentManager`, `ExperimentRunMetadata`, `record_reproducibility`, `PromptManager` |
| `aai_core.evaluation` | Deterministic absolute and regression gates | `GatePolicy`, `MetricRule`, `apply_gate` |
| `aai_core.rag` | Versioned embedding/chunking contracts and MLflow document evidence | `EmbeddingProfile`, `ChunkingProfile`, `RAGDocument`, `mlflow_documents` |
| `aai_core.agents` | Bounded, observable agent request, response, and decision evidence | `AgentRequest`, `AgentResponse`, `AgentDecision` |
| `aai_core.deployment`, `aai_core.manifest` | Persisted release evidence and validated application manifests | `ApplicationRelease`, `AIApplicationManifest`, `load_manifest` |
| `aai_core.diagnostics`, `aai_core.testing` | Safe preflight checks and credential-free test doubles | `run_doctor`, `dev_context`, `FakeChatModel`, `FakeRetriever` |
| `aai_core.contracts`, `aai_core.exceptions` | Strict boundary-model primitives and the stable SDK error base | `ContractModel`, `AaiCoreError` |

## Lifecycle and evidence

Use one `ResourceContext` across providers, MLflow, traces, and logs. Experiments
follow `baseline -> change -> result -> decision`; the decision is `adopt`,
`reject`, or `inconclusive`. An `ApplicationRelease` records immutable evidence
about the selected code, model, prompt, retrieval configuration, evaluation,
tools, and controls. It is not a deployment engine.

Provider response objects stay native rather than being mirrored throughout the
SDK. Stable normalized results retain the original object through `raw`, and
provider adapters expose `native_client` for provider-specific functionality.

## Retrieval and ranking

`Retriever.search()` supports the portable `text`, `vector`, and `hybrid`
algorithms. Databricks text retrieval maps to its `FULL_TEXT` query type;
hybrid remains the general-purpose default. Query-time second-stage ranking is
explicit and provider-native:

```python
from aai_core.providers import DatabricksRerankOptions, RetrievalMode

documents = retriever.search(
    question,
    mode=RetrievalMode.HYBRID,
    ranking=DatabricksRerankOptions(
        columns_to_rerank=("content", "parent_summary"),
    ),
)
```

Use `AzureSemanticRankOptions` to select an externally provisioned Azure AI
Search semantic configuration. The adapters reject a ranking option for the
wrong provider, and Databricks reranker columns must already be present in the
retriever's governed column list. Ranking configuration, candidate count,
latency, and retrieval-quality evidence should be benchmarked together before
adoption; changing any of them is an application release.

## Ownership, async, and streaming

- `PlatformContext` owns and closes only resources it creates. Models,
  retrievers, and clients registered by the application remain caller-owned.
- The stable `ChatModel`, `EmbeddingProvider`, and `Retriever` contracts are
  synchronous. This keeps batch jobs and common application paths small.
- Use `ChatModel.create_native_async_client()` for provider-native async or
  streaming calls. The returned client is caller-owned and must be closed by
  the application; native timeout and cancellation behavior still applies.
- Streaming transports and event shapes remain provider-native. The SDK does
  not add a second universal event protocol. Generated serving templates add
  their own bounded async handlers at the application boundary.
- `SecretValue` never reveals its resolved value through `str`, `repr`, or
  serialization. Do not place resolved secret material in exceptions, traces,
  tags, parameters, or persisted evidence.

For lifecycle examples, see the [developer guide](developer-guide.md) and the
[progressive examples](../examples/README.md).
