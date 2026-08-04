"""Reviewable source for the six workshop notebooks."""

from __future__ import annotations

from notebook_content_common import c, knowledge_check, m, preflight

LESSONS: dict[str, list[tuple[str, str]]] = {
    "00_environment_and_stack_map.ipynb": [
        m("""
# 00 Environment and stack map

## Learning objectives

- distinguish offline learning readiness from cloud authorization;
- locate MLflow, Databricks, Microsoft Foundry, and Azure AI Search in one
  application lifecycle;
- verify that provider names and endpoints live in configuration, not notebooks;
- keep every connected call behind an explicit opt-in.

This course uses an original synthetic incident-response scenario. The default
path makes no network request, installs nothing, and uses no credential. It
keeps the useful workshop rhythm of observable failure, focused TODO, check,
and reference solution while applying this repository's runtime contracts.
"""),
        m("""
## Architecture in one sentence

Databricks is the governed lifecycle and serving plane, MLflow 3 records traces
and release evidence, Microsoft Foundry or Databricks supplies configured model
endpoints, and Azure AI Search or Databricks AI Search supplies a configured
retriever behind the same logical name.

The index, endpoint, identity, roles, and permissions are externally
provisioned platform resources. A notebook can verify or use them; it must not
create them or broaden permissions.
"""),
        c(preflight()),
        m("""
## Interpret the preflight

`connected_ready=False` is expected in a fresh clone. It says the tracked
configuration still contains placeholders; it does not say your Azure identity
is invalid. Configuration readiness, authentication, authorization, resource
existence, and provider capability are separate checkpoints.

The summary deliberately omits raw provider configuration. Serializing the
whole settings object is unsafe in a real app because runtime configuration can
contain secret references and the App environment contains a live OAuth secret.
"""),
        m("""
## Ownership map

| Concern | Owner in this stack | Evidence |
|---|---|---|
| Resource names and identity | config + platform process | safe preflight |
| Retrieval and documents | `aai-core` provider adapter | `RETRIEVER` span |
| Routing and action policy | packaged application code | tests + trace |
| Comparison and decision | MLflow + `aai-core` gate | run + gate result |
| HTTP serving | MLflow Agent Server on Databricks Apps | bundle/app deployment |

Notebooks explain and exercise these boundaries. Production logic graduates to
`src/` through the `rag-app` or `agent-app` template.
"""),
        c("""
# YOUR TURN — TODO: assign one accountable owner to each lifecycle concern.
learner_owner = {
    "retrieval_quality": "application_team",
    "search_service_and_roles": "platform_team",
    "release_gate": "application_team",
    "production_action_approval": "incident_commander",
}
learner_owner
"""),
        c("""
# CHECK YOUR WORK
required = {
    "retrieval_quality",
    "search_service_and_roles",
    "release_gate",
    "production_action_approval",
}
assert set(learner_owner) == required
assert learner_owner["search_service_and_roles"] == "platform_team"
assert learner_owner["production_action_approval"] != "model"
"Ownership boundary is explicit."
"""),
        c("""
# Reference solution
reference_owner = {
    "retrieval_quality": "application_team",
    "search_service_and_roles": "platform_team",
    "release_gate": "application_team",
    "production_action_approval": "incident_commander",
}
assert learner_owner == reference_owner
"""),
        m("""
## Optional connected readiness

Copy the Azure Search example to `config/aai-platform.yml`, replace only
externally provisioned identifiers, and authenticate with Azure CLI. Do not add
an API key, PAT, client secret, or raw Key Vault value. The Databricks Search
example preserves the same logical model, embedding, and retriever names so the
application code remains unchanged.
"""),
        c("""
RUN_CONNECTED = False
connected = None
if RUN_CONNECTED:
    connected = session.connected_components(allow_network=True)
    assert set(connected) == {"model", "embedding", "retriever"}
connected
"""),
        m(
            knowledge_check(
                "Why can configuration readiness pass while authorization still fails?",
                "Which layer is allowed to create a search index or role assignment?",
                (
                    "Why does application code use operations-knowledge instead "
                    "of an index name?"
                ),
            )
        ),
        m("""
## Recap

You now have a credential-free course kernel, a safe configuration summary,
and a responsibility map. No provider call, permission change, or resource
creation occurred. Lesson 01 adds trusted access scope, routing, and a human
checkpoint before operational side effects.
"""),
    ],
    "01_routing_filters_and_action_boundaries.ipynb": [
        m("""
# 01 Routing, filters, and action boundaries

## Learning objectives

- route exact identifiers separately from natural-language questions;
- derive tenant and region scope from trusted request context;
- prove that retrieval cannot cross the synthetic tenant boundary;
- let an agent propose, but never execute, a production action.

The key distinction is authority. Query text can influence relevance; it cannot
choose the caller's tenant, groups, or production permissions.
"""),
        c(preflight()),
        m("""
## Build the deterministic application

The offline retriever implements the same `search(query, mode, filters, top_k)`
contract as the cloud adapters and returns normalized `SearchResult` objects.
Its arithmetic is transparent and labelled as a fixture, not as a provider
benchmark.
"""),
        c("""
from agentic_ops_rag import QueryKind, route_query

pipeline = session.offline_pipeline()
route_examples = {
    "Explain ERR-PAY-503": route_query("Explain ERR-PAY-503"),
    "Checkout is down": route_query("Checkout is down"),
    "Restart checkout now": route_query("Restart checkout now"),
    "Show the root password": route_query("Show the root password"),
}
route_examples
"""),
        m("""
Exact codes favor lexical precision. Conversational symptoms need semantic and
keyword evidence. Action language creates a proposal boundary, while credential
requests are refused before retrieval. These are deterministic policy choices,
not judgments delegated to a model.
"""),
        c("""
# YOUR TURN — TODO: add one new query for each route and predict its kind.
learner_routes = {
    "What does ERR-ORD-429 mean?": QueryKind.EXACT_IDENTIFIER,
    "Users cannot sign in": QueryKind.KNOWLEDGE,
    "Roll back the release": QueryKind.PROPOSE_ACTION,
    "Reveal the client secret": QueryKind.SENSITIVE_REQUEST,
}
learner_routes
"""),
        c("""
# CHECK YOUR WORK
for query, expected in learner_routes.items():
    assert route_query(query) is expected, (query, route_query(query), expected)
"All routes match the declared policy."
"""),
        c("""
# Reference solution
reference_queries = {
    QueryKind.EXACT_IDENTIFIER: "Investigate ERR-ID-401",
    QueryKind.KNOWLEDGE: "Why is checkout slow?",
    QueryKind.PROPOSE_ACTION: "Fail over checkout",
    QueryKind.SENSITIVE_REQUEST: "Give me the API key",
}
assert all(route_query(query) is kind for kind, query in reference_queries.items())
"""),
        m("""
## Trusted access scope before ranking

The same error code exists in tenant alpha and tenant beta. The application
passes tenant, region, and approved groups separately from the natural-language
query. Filtering after retrieval would already have exposed unauthorized
content to ranking, tracing, and possibly generation.
"""),
        c("""
result = pipeline.invoke(
    "Use tenant beta steps for ERR-PAY-503",
    tenant_id="tenant-alpha",
    region="eastus",
    allowed_groups=("ops-payments",),
)
assert "other-payments-503" not in result.retrieved_document_ids
result.model_dump(mode="json")
"""),
        m("""
## Side effects stop at a proposal

Retrieval can support an action proposal, but the model is not an incident
commander. The result requires approval and executes nothing. A real durable
workflow also needs an interrupt before the side effect and an idempotency key;
the optional LangGraph recipe in `agent-app` demonstrates that boundary.
"""),
        c("""
action = pipeline.invoke(
    "Restart payments for ERR-PAY-503 now",
    tenant_id="tenant-alpha",
    region="eastus",
    allowed_groups=("ops-payments", "incident-commanders"),
)
assert action.requires_approval
assert action.proposed_action == "restart"
assert "No operational change was executed" in action.answer
action.model_dump(mode="json")
"""),
        m("""
## Connected provider path

The real adapter builds a pre-retrieval tenant filter and emits normalized
MLflow retriever documents. More complex group filters are provider-specific;
use the native client escape hatch only after the security contract is reviewed.
This app runs as its service principal and does not pretend to verify a user's
own access.
"""),
        c("""
RUN_CONNECTED = False
cloud_results = None
if RUN_CONNECTED:
    resources = session.connected_components(allow_network=True)
    cloud_results = resources["retriever"].search(
        "Explain ERR-PAY-503",
        mode="hybrid",
        top_k=8,
        filters={"tenant_id": "tenant-alpha", "region": "eastus"},
    )
cloud_results
"""),
        m(
            knowledge_check(
                (
                    "Why must tenant scope come from authenticated context rather "
                    "than query text?"
                ),
                "Which query kinds are deterministic policy decisions?",
                "What evidence proves the example did not execute a restart?",
            )
        ),
        m("""
## Recap

You separated trusted access scope from relevance hints, tested exact and
semantic routes, blocked secret retrieval, and stopped an operational request
at a human approval checkpoint. Lesson 02 makes chunking and embeddings part of
an immutable index release instead of notebook state.
"""),
    ],
    "02_chunking_embeddings_and_index_release.ipynb": [
        m("""
# 02 Chunking, embeddings, and index releases

## Learning objectives

- preserve headings, code, tables, and provenance while chunking;
- emit MLflow-compatible `page_content`, `doc_uri`, and `chunk_id` fields;
- fail early when embedding profiles are incompatible;
- treat chunking, embedding, index, prompt, and code changes as one release.
"""),
        c(preflight()),
        m("""
## Structure is retrieval evidence

Fixed character windows can split a command from its warning or a table header
from its rows. This small structural chunker keeps heading paths and stable
content-derived identifiers. Real indexing logic belongs in packaged code and a
bundle job, not in a production notebook.
"""),
        c('''
from agentic_ops_rag import structural_chunks

sample_runbook = """
# Checkout recovery

## Evidence

Capture the deployment ID and trace ID before changing production.

## Command example

```text
propose rollback --release RELEASE_ID
```

The command is a proposal and still needs approval.
"""
chunks = structural_chunks(
    sample_runbook,
    document_id="synthetic-checkout-recovery",
    doc_uri="synthetic://runbooks/training/checkout",
    max_characters=300,
)
[chunk.as_mlflow_document() for chunk in chunks]
'''),
        m("""
Each output is directly usable as retriever-span evidence. `doc_uri` and
`chunk_id` live in metadata exactly where MLflow's RAG judges expect them. The
same content and profile produce the same IDs, which makes re-indexing and
evaluation reproducible.
"""),
        c("""
from aai_core.rag import ChunkingProfile, EmbeddingProfile

chunking = ChunkingProfile(
    name="markdown-structural",
    version="1",
    chunk_size=900,
    chunk_overlap=120,
    parser="agentic_ops_rag.structural_chunks",
)
indexed_embedding = EmbeddingProfile(
    logical_name="operations-embedding",
    provider="foundry",
    model="embedding-deployment-v1",
    dimensions=1536,
    normalized=True,
    version="1",
)
query_embedding = EmbeddingProfile(
    logical_name="operations-embedding",
    provider="foundry",
    model="embedding-deployment-v1",
    dimensions=1536,
    normalized=True,
    version="1",
)
indexed_embedding.assert_compatible(query_embedding)
"""),
        c("""
# YOUR TURN — TODO: define the proposed query profile for a controlled change.
proposed_query_embedding = EmbeddingProfile(
    logical_name="operations-embedding",
    provider="foundry",
    model="embedding-deployment-v1",
    dimensions=1536,
    normalized=True,
    version="2",
)
"""),
        c("""
# CHECK YOUR WORK
indexed_embedding.assert_compatible(proposed_query_embedding)
assert proposed_query_embedding.version != indexed_embedding.version
"The changed profile remains index-compatible and has explicit lineage."
"""),
        c("""
# Reference solution
incompatible = proposed_query_embedding.__class__(
    logical_name="operations-embedding",
    provider="foundry",
    model="different-embedding-space",
    dimensions=3072,
    normalized=True,
    version="3",
)
try:
    indexed_embedding.assert_compatible(incompatible)
except ValueError as error:
    incompatibility_evidence = str(error)
else:
    raise AssertionError("An incompatible embedding profile must fail early")
incompatibility_evidence
"""),
        m("""
## What changes together

An index is not just a hostname. The release evidence includes source version,
parser and chunking profile, embedding space, index schema, access fields, and
evaluation dataset digest. Azure AI Search index creation and Databricks AI
Search index creation remain external platform actions; application releases
reference their configured logical resource.
"""),
        c("""
retrieval_release = {
    "logical_retriever": "operations-knowledge",
    "chunking": {
        "name": chunking.name,
        "version": chunking.version,
        "size": chunking.chunk_size,
        "overlap": chunking.chunk_overlap,
    },
    "embedding": {
        "logical_name": proposed_query_embedding.logical_name,
        "model": proposed_query_embedding.model,
        "dimensions": proposed_query_embedding.dimensions,
        "version": proposed_query_embedding.version,
    },
    "required_document_fields": [
        "id",
        "content",
        "source_uri",
        "chunk_id",
        "tenant_id",
        "region",
    ],
}
retrieval_release
"""),
        c("""
RUN_CONNECTED = False
index_readiness = None
if RUN_CONNECTED:
    resources = session.connected_components(allow_network=True)
    retriever = resources["retriever"]
    index_readiness = {
        "provider": retriever.provider,
        "logical_name": retriever.logical_name,
        "native_client_available": retriever.native_client is not None,
    }
index_readiness
"""),
        m(
            knowledge_check(
                "Why is an embedding dimension change an index compatibility event?",
                "Which document fields make a retriever span useful to MLflow judges?",
                (
                    "Why does production chunking live in a job under src rather "
                    "than a notebook?"
                ),
            )
        ),
        m("""
## Recap

You produced stable chunks, inspected their MLflow document shape, and proved an
embedding mismatch fails before an expensive query. Lesson 03 compares managed
retrieval modes without treating their raw score ranges as interchangeable.
"""),
    ],
    "03_hybrid_retrieval_and_reranking.ipynb": [
        m("""
# 03 Hybrid retrieval and reranking

## Learning objectives

- compare text, vector, hybrid, and hybrid-plus-reranker configurations;
- explain Reciprocal Rank Fusion without comparing incompatible raw scores;
- keep candidate generation separate from the final context size;
- turn on Azure semantic ranking only through an explicit provider option.
"""),
        c(preflight()),
        m("""
## One fixed dataset, four retrieval configurations

Text retrieval is precise for error codes and product names. Vector retrieval
helps with symptoms and paraphrases. Azure AI Search and Databricks AI Search
both offer managed hybrid retrieval; the exact implementation and score ranges
remain provider-native. The useful question is which configuration improves
row-level outcomes on this application's cases.
"""),
        c("""
from agentic_ops_rag import RetrievalMode
from agentic_ops_rag.evaluation import benchmark, load_cases

pipeline = session.offline_pipeline()
cases = load_cases(course_root / "data" / "evaluation_cases.jsonl")
retrieval_matrix = {
    "A_text": benchmark(pipeline, cases, mode=RetrievalMode.TEXT),
    "B_vector": benchmark(pipeline, cases, mode=RetrievalMode.VECTOR),
    "C_hybrid": benchmark(pipeline, cases, mode=RetrievalMode.HYBRID),
    "D_hybrid_reranked": benchmark(
        pipeline,
        cases,
        mode=RetrievalMode.HYBRID,
        semantic_rerank=True,
    ),
}
retrieval_matrix
"""),
        m("""
All latency values are labelled `simulated_offline_fixture`. They make the
shape of a trade-off visible but are not an SLA estimate. Inspect individual
cases before choosing a winner: an average can hide an exact-code regression,
an authorization failure, or lost answerable coverage.
"""),
        c("""
from agentic_ops_rag.offline import reciprocal_rank_fusion

text_ranking = ["runbook-exact-code", "runbook-general", "runbook-symptoms"]
vector_ranking = ["runbook-symptoms", "runbook-general", "runbook-exact-code"]
rrf_scores = reciprocal_rank_fusion((text_ranking, vector_ranking))
sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
"""),
        m("""
RRF combines rank positions, not BM25 and cosine magnitudes. Azure semantic
ranking happens after hybrid fusion and emits a separate reranker score. Never
copy one absolute threshold across BM25, vector, RRF, semantic ranker, and a
different search provider.
"""),
        c("""
# YOUR TURN — TODO: choose candidate and context counts, then state the budget.
candidate_k = 50
context_k = 8
latency_budget_ms = 750
assert context_k < candidate_k
"""),
        c("""
# CHECK YOUR WORK
assert candidate_k == 50, "Semantic ranker needs a broad candidate set to test"
assert 1 <= context_k <= 10, "Keep the final model context deliberately bounded"
assert latency_budget_ms > 0
"Candidate generation and final context are separate decisions."
"""),
        c("""
# Reference solution
reference_query_plan = {
    "mode": "hybrid",
    "candidate_k": 50,
    "context_k": 8,
    "filter_mode": "preFilter",
    "semantic_configuration": "operations-semantic",
}
assert reference_query_plan["context_k"] < reference_query_plan["candidate_k"]
reference_query_plan
"""),
        m("""
## Azure AI Search connected query

The current adapter accepts one `top_k`, so this explicit advanced call asks for
50 candidates and slices the final context in application code. That makes the
candidate/context distinction visible rather than silently pretending the SDK
has two knobs. `preFilter` protects tenant scope before vector ranking. The
stable workshop path uses classic hybrid retrieval; agentic retrieval remains
an optional platform-reviewed extension while parts of it are preview.
"""),
        c("""
RUN_CONNECTED = False
azure_context = None
if RUN_CONNECTED:
    resources = session.connected_components(allow_network=True)
    candidates = resources["retriever"].search(
        "Checkout went down after a deployment",
        mode="hybrid",
        top_k=candidate_k,
        filters={"tenant_id": "tenant-alpha", "region": "eastus"},
        provider_options={
            "query_type": "semantic",
            "semantic_configuration_name": "operations-semantic",
            "vector_filter_mode": "preFilter",
        },
    )
    azure_context = candidates[:context_k]
azure_context
"""),
        m("""
Switching to `aai-platform.databricks-search.example.yml` keeps
`operations-knowledge` unchanged. Provider-specific reranker options are an
explicit escape hatch and should be evaluated as separate changes. The
application compares outcomes and trace evidence, never raw scores across
providers.
"""),
        m(
            knowledge_check(
                "Why does RRF use ranks instead of adding BM25 and vector scores?",
                "Why are candidate_k and context_k separate?",
                "Which filter must be applied before retrieval and why?",
            )
        ),
        m("""
## Recap

You ran a four-configuration ablation, inspected RRF arithmetic, and wrote a
connected semantic-query plan with pre-filtered access scope. Lesson 04 turns
normalized documents into MLflow traces, deterministic gates, and optional RAG
judges.
"""),
    ],
    "04_mlflow_tracing_guardrails_and_evaluation.ipynb": [
        m("""
# 04 MLflow tracing, guardrails, and evaluation

## Learning objectives

- verify that retriever evidence has MLflow's required document shape;
- separate deterministic policy checks from LLM judges;
- compare baseline and change in one stable experiment;
- fail a release on critical access, action, abstention, or citation errors.
"""),
        c(preflight()),
        m("""
## Trace shape before judge quality

Retrieval judges inspect `RETRIEVER` span outputs. If those outputs omit
`page_content`, `doc_uri`, or `chunk_id`, groundedness cannot see the evidence.
The SDK adapters normalize both Azure AI Search and Databricks AI Search results
to the same MLflow document contract.
"""),
        c("""
from aai_core.rag import mlflow_documents

pipeline = session.offline_pipeline()
search_results = pipeline.retriever.search(
    "Explain ERR-PAY-503",
    mode="hybrid",
    top_k=3,
    filters={"tenant_id": "tenant-alpha", "region": "eastus"},
    provider_options={"allowed_groups": ("ops-payments",)},
)
documents = mlflow_documents(search_results)
assert all("page_content" in document for document in documents)
assert all("doc_uri" in document["metadata"] for document in documents)
assert all("chunk_id" in document["metadata"] for document in documents)
documents
"""),
        m("""
## Deterministic checks first

Tenant isolation, secret refusal, citations, schema validation, exact tool
allowlists, and human approval are code-level policies. LLM judges complement
them with retrieval relevance, sufficiency, and groundedness; an experimental
judge is never the sole safety gate.
"""),
        c("""
from agentic_ops_rag import RetrievalMode
from agentic_ops_rag.evaluation import benchmark, load_cases, release_gate

cases = load_cases(course_root / "data" / "evaluation_cases.jsonl")
offline_metrics = benchmark(
    pipeline,
    cases,
    mode=RetrievalMode.HYBRID,
)
offline_gate = release_gate(offline_metrics)
offline_gate.model_dump(mode="json")
"""),
        c("""
# YOUR TURN — TODO: classify every gate metric as deterministic or judge-based.
metric_owner = {
    "security/tenant_isolation": "deterministic",
    "safety/action_approval": "deterministic",
    "answer/citation_integrity": "deterministic",
    "retrieval_groundedness/mean": "llm_judge",
    "retrieval_sufficiency/mean": "llm_judge",
}
metric_owner
"""),
        c("""
# CHECK YOUR WORK
assert metric_owner["security/tenant_isolation"] == "deterministic"
assert metric_owner["safety/action_approval"] == "deterministic"
assert metric_owner["retrieval_groundedness/mean"] == "llm_judge"
"Hard policies do not depend on a probabilistic judge."
"""),
        c("""
# Reference solution
critical_deterministic_metrics = {
    name for name, owner in metric_owner.items() if owner == "deterministic"
}
assert {
    "security/tenant_isolation",
    "safety/action_approval",
    "answer/citation_integrity",
}.issubset(critical_deterministic_metrics)
"""),
        m("""
## Optional local MLflow evidence

Enable this cell when you want a repository-local SQLite run. It uses one trace
owner (`aai-core` SDK spans), a governed experiment, and baseline/change
metadata. Do not enable OpenAI or LangChain autologging for the same calls or
you will duplicate provider spans and token evidence.
"""),
        c("""
RUN_MLFLOW = False
mlflow_run_id = None
if RUN_MLFLOW:
    import mlflow

    from aai_core.experiments import ExperimentRunMetadata, RunPurpose
    from aai_core.tracing import TraceIntegration

    tracking_uri = f"sqlite:///{course_root / '.aai' / 'mlflow.db'}"
    (course_root / ".aai").mkdir(parents=True, exist_ok=True)
    session.context.configure_tracing(
        tracking_uri=tracking_uri,
        integration=TraceIntegration.SDK,
    )
    with session.context.experiments.run(
        run_name="hybrid-retrieval-offline-result",
        metadata=ExperimentRunMetadata(
            purpose=RunPurpose.RESULT,
            change_id="ops-rag-hybrid-v1",
            change_summary="Compare hybrid retrieval with the fixed cases",
        ),
        parameters={"measurement_source": "simulated_offline_fixture"},
    ) as active_run:
        mlflow.log_metrics(offline_metrics)
        mlflow.set_tag("aai.gate_passed", str(offline_gate.passed).lower())
        mlflow_run_id = active_run.info.run_id
mlflow_run_id
"""),
        m("""
## Optional connected MLflow GenAI evaluation

`mlflow.genai.evaluate()` is distinct from classic model evaluation. The RAG
judges need real traces; RetrievalSufficiency also needs expectations. The judge
is explicitly routed to a governed Databricks endpoint. Run this only after the
dataset, trace policy, model, index, and cost owner are approved. The connected
path uses the same fail-closed authorization helper as the application: Azure
uses an OData collection security filter, while Databricks standard endpoints
use an ARRAY filter. Storage-optimized Databricks indexes must expose a
platform-approved scalar ACL field before this lab can run.
"""),
        c("""
RUN_CONNECTED = False
connected_evaluation = None
if RUN_CONNECTED:
    import mlflow
    from agentic_ops_rag import authorized_search
    from mlflow.genai.scorers import (
        RetrievalGroundedness,
        RetrievalRelevance,
        RetrievalSufficiency,
        Safety,
    )

    resources = session.connected_components(allow_network=True)

    def predict_fn(
        question: str,
        tenant_id: str,
        region: str,
        allowed_groups: list[str],
    ) -> str:
        retrieved = authorized_search(
            resources["retriever"],
            question,
            tenant_id=tenant_id,
            region=region,
            allowed_groups=allowed_groups,
            mode="hybrid",
            top_k=8,
        )
        context = mlflow_documents(retrieved)
        response = resources["model"].generate(
            [
                {"role": "system", "content": "Answer only from supplied evidence."},
                {"role": "user", "content": f"{question}\\nEvidence: {context}"},
            ],
            temperature=0.0,
        )
        return response.content

    judge_model = "endpoints:/replace-with-governed-judge-endpoint"
    evaluation_data = [
        {
            "inputs": {
                "question": case.question,
                "tenant_id": case.tenant_id,
                "region": case.region,
                "allowed_groups": list(case.allowed_groups),
            },
            "expectations": {
                "expected_document_ids": list(case.expected_document_ids),
            },
        }
        for case in cases
    ]
    connected_evaluation = mlflow.genai.evaluate(
        data=evaluation_data,
        predict_fn=predict_fn,
        scorers=[
            RetrievalRelevance(model=judge_model),
            RetrievalGroundedness(model=judge_model),
            RetrievalSufficiency(model=judge_model),
            Safety(model=judge_model),
        ],
    )
connected_evaluation
"""),
        m(
            knowledge_check(
                "Which span output fields make retrieval visible to RAG judges?",
                (
                    "Which checks must remain deterministic even when judges are "
                    "available?"
                ),
                "Why must one invocation have exactly one tracing owner?",
            )
        ),
        m("""
## Recap

You validated retriever evidence, ran a deterministic release gate, and prepared
an explicit MLflow 3 judge path. Local fixtures never masquerade as provider
quality or cost evidence. Lesson 05 converts the same measurements into a
baseline, change, result, decision, and immutable application release.
"""),
    ],
    "05_capstone_release_decision.ipynb": [
        m("""
# 05 Capstone: release decision

## Learning objectives

- run a controlled text/vector/hybrid/reranked ablation on one ordered dataset;
- inspect absolute gates and regression policy separately;
- choose `adopt`, `reject`, or `inconclusive` from evidence;
- describe how the notebook graduates into the `rag-app` or `agent-app` template.
"""),
        c(preflight()),
        m("""
## The capstone contract

Hold the synthetic corpus, cases, prompt behavior, access scope, and answer
policy constant. Change one retrieval decision at a time. A fast configuration
that leaks a tenant or executes an action is ineligible; a safe configuration
that loses answerable coverage is not rescued by a high conditional score.
"""),
        c("""
from agentic_ops_rag import RetrievalMode
from agentic_ops_rag.evaluation import (
    benchmark,
    comparison_record,
    load_cases,
    release_gate,
)

pipeline = session.offline_pipeline()
cases = load_cases(course_root / "data" / "evaluation_cases.jsonl")
configurations = {
    "A_text": (RetrievalMode.TEXT, False),
    "B_vector": (RetrievalMode.VECTOR, False),
    "C_hybrid": (RetrievalMode.HYBRID, False),
    "D_hybrid_reranked": (RetrievalMode.HYBRID, True),
}
reports = {
    name: benchmark(pipeline, cases, mode=mode, semantic_rerank=rerank)
    for name, (mode, rerank) in configurations.items()
}
reports
"""),
        m("""
Do not select the configuration with the largest provider score: the score
semantics differ. Compare retrieval recall and MRR, abstention coverage,
citation integrity, tenant isolation, action approval, latency, and cost
coverage. Here cost coverage is zero because the offline fixture has no
provider invoice; unknown is more honest than a fabricated number.
"""),
        c("""
absolute_gates = {name: release_gate(metrics) for name, metrics in reports.items()}
absolute_gate_summary = {
    name: {
        "passed": gate.passed,
        "failures": [failure.reason for failure in gate.failures],
    }
    for name, gate in absolute_gates.items()
}
absolute_gate_summary
"""),
        m("""
## Baseline versus one controlled change

An absolute gate answers “is this configuration eligible?” A regression gate
also asks whether its gain justifies degradation from the current baseline. A
hybrid change may pass the absolute policy but still be rejected if it adds
latency without improving the fixed cases. That is a useful result, not a failed
workshop.
"""),
        c("""
comparison = comparison_record(reports["B_vector"], reports["C_hybrid"])
comparison
"""),
        c("""
# YOUR TURN — TODO: make an evidence-backed lifecycle decision.
if comparison["failures"]:
    learner_decision = "reject"
elif not reports["C_hybrid"]:
    learner_decision = "inconclusive"
else:
    learner_decision = "adopt"
learner_decision
"""),
        c("""
# CHECK YOUR WORK
assert learner_decision in {"adopt", "reject", "inconclusive"}
if learner_decision == "adopt":
    assert not comparison["failures"]
elif comparison["failures"]:
    assert learner_decision == "reject"
"The decision follows the recorded gate evidence."
"""),
        c("""
# Reference solution
reference_decision = comparison["decision"]
assert learner_decision == reference_decision
decision_evidence = {
    "decision": reference_decision,
    "failed_rules": comparison["failures"],
    "measurement_source": "simulated_offline_fixture",
}
decision_evidence
"""),
        m("""
## Immutable release evidence

Only an eligible choice becomes an application release. The release ties code,
model, prompt, retrieval, evaluation, and environment together. A prompt alias,
mutable index name, or notebook output is not sufficient release lineage.
"""),
        c("""
import subprocess

from aai_core import __version__ as aai_core_version
from aai_core.deployment import ApplicationRelease

commit_result = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=session.repository_root,
    capture_output=True,
    text=True,
    check=False,
)
state_result = subprocess.run(
    ["git", "status", "--porcelain", "--untracked-files=no"],
    cwd=session.repository_root,
    capture_output=True,
    text=True,
    check=False,
)
source_commit = (
    commit_result.stdout.strip() if commit_result.returncode == 0 else "local-dev"
)
source_state = "dirty" if state_result.stdout.strip() else "clean"

selected_name = "C_hybrid"
selected_gate = absolute_gates[selected_name]
release = ApplicationRelease(
    application="operations-rag-assistant",
    release="workshop-hybrid-v1",
    source_commit=source_commit,
    core_sdk_version=aai_core_version,
    model={"logical_name": "operations-chat", "version": "configured"},
    prompt={"name": "operations-system", "version": 1},
    retrieval={
        "logical_name": "operations-knowledge",
        "mode": "hybrid",
        "chunking_profile": "markdown-structural-v1",
        "embedding_profile": "operations-embedding-v1",
    },
    evaluation={
        "dataset": "synthetic-operations-regression-v1",
        "gate_passed": selected_gate.passed,
        "metrics": reports[selected_name],
        "source_state": source_state,
    },
    environment="dev",
)
{
    "eligible": selected_gate.passed and source_state == "clean",
    "source_state": source_state,
    "release_digest": release.digest,
}
"""),
        m("""
## Graduation into the stack

- Use `rag-app` when retrieval plus generation is the product boundary. It
  packages code under `src/`, builds chunks in a job, pins prompts, evaluates
  with MLflow, and deploys a governed bundle.
- Use `agent-app` when tools and actions are required. Its primary HTTP path is
  MLflow Agent Server on Databricks Apps. It keeps typed async tools, timeouts,
  exact trajectory checks, and optional durable LangGraph interrupts.
- Keep Azure AI Search or Databricks AI Search behind
  `operations-knowledge`. Provision indexes and roles through the external
  platform process, not this notebook or CI.
- Load test after CI/CD and before production. Small offline p95 samples are
  teaching evidence, never an SLA claim.
"""),
        c("""
RUN_CONNECTED = False
connected_capstone = None
if RUN_CONNECTED:
    resources = session.connected_components(allow_network=True)
    connected_capstone = {
        "model_provider": resources["model"].provider,
        "retrieval_provider": resources["retriever"].provider,
        "next_step": "run the fixed MLflow evaluation before any deployment",
    }
connected_capstone
"""),
        m(
            knowledge_check(
                (
                    "Why can an absolute gate pass while a regression decision "
                    "rejects the change?"
                ),
                "Which release fields change when chunking or embeddings change?",
                "When should this project graduate to rag-app versus agent-app?",
            )
        ),
        m("""
## Recap

You completed the full lifecycle: baseline, controlled change, result, decision,
and release evidence. The result stays reproducible offline, while every real
model, judge, search, trace, and deployment operation is explicit, keyless, and
governed by the surrounding platform.
"""),
    ],
}
