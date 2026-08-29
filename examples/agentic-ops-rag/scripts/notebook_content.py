"""Reviewable source for the seven workshop notebooks."""

from __future__ import annotations

from notebook_content_common import c, knowledge_check, m, preflight

LESSONS: dict[str, list[tuple[str, str]]] = {
    "00_environment_and_stack_map.ipynb": [
        m("""
# 00 Environment and stack map

## Learning objectives

- distinguish offline learning readiness from cloud authorization;
- locate MLflow, Databricks, and Azure AI Search in one application
  lifecycle;
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
and release evidence, Databricks supplies configured model endpoints, and
Azure AI Search or Databricks AI Search supplies a configured retriever
behind the same logical name.

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

First look at what the index itself would hand to a caller with no scope
filter. This raw probe simulates a missing security filter; it is exactly the
call shape `authorized_search` exists to prevent.
"""),
        c("""
beta_query = "Use tenant beta steps for ERR-PAY-503"
unscoped_candidates = pipeline.retriever.search(
    beta_query,
    top_k=8,
    filters=None,
    mode="hybrid",
    provider_options={"allowed_groups": ("ops-payments", "beta-ops-payments")},
)
unscoped_ids = [result.document_id for result in unscoped_candidates]
assert "other-payments-503" in unscoped_ids
print("without a tenant filter, candidates:", unscoped_ids)
print("tenant-beta evidence exposed:", "other-payments-503" in unscoped_ids)
"""),
        m("""
The beta runbook ranks first: relevance ranking is not an access control. Now
run the same question through the governed path. The tenant and region come
from authenticated request context, are applied before ranking, and no query
phrasing can widen them.
"""),
        c("""
from agentic_ops_rag import authorized_search

scoped_candidates = authorized_search(
    pipeline.retriever,
    beta_query,
    tenant_id="tenant-alpha",
    region="eastus",
    allowed_groups=("ops-payments",),
    mode="hybrid",
    top_k=8,
)
scoped_ids = [result.document_id for result in scoped_candidates]
result = pipeline.invoke(
    beta_query,
    tenant_id="tenant-alpha",
    region="eastus",
    allowed_groups=("ops-payments",),
)
assert "other-payments-503" not in scoped_ids
assert "other-payments-503" not in result.retrieved_document_ids
print("with trusted scope, candidates:  ", scoped_ids)
print("tenant-beta evidence excluded from candidates and from the answer.")
"""),
        m("""
## Layered defense: the retired revision

`alpha-payments-503-retired` is an inactive, superseded revision of the same
runbook code. The provider filter already refuses inactive documents, so the
decoy never even reaches ranking. But defense does not stop at the provider:
suppose an index were mislabelled and the retired revision came back marked
active. The application rerank keeps only the newest `effective_at` revision
per `runbook_code`, so the stale procedure still cannot reach generation.
"""),
        c("""
from agentic_ops_rag import OfflineOperationsRetriever, OperationsRAGPipeline

mislabelled = [
    document.model_copy(update={"active": True})
    if document.document_id == "alpha-payments-503-retired"
    else document
    for document in pipeline.retriever.documents
]
mislabelled_pipeline = OperationsRAGPipeline(OfflineOperationsRetriever(mislabelled))
provider_candidates = mislabelled_pipeline.retriever.search(
    "Explain ERR-PAY-503",
    top_k=8,
    filters={"tenant_id": "tenant-alpha", "region": "eastus"},
    provider_options={"allowed_groups": ("ops-payments",)},
)
before_ids = [result.document_id for result in provider_candidates]
recovered = mislabelled_pipeline.invoke(
    "Explain ERR-PAY-503",
    tenant_id="tenant-alpha",
    region="eastus",
    allowed_groups=("ops-payments",),
)
after_ids = list(recovered.retrieved_document_ids)
assert "alpha-payments-503-retired" in before_ids
assert "alpha-payments-503-retired" not in after_ids
assert "alpha-payments-503-current" in after_ids
print("provider candidates (mislabelled index):", before_ids)
print("after application rerank:               ", after_ids)
print("newest effective_at per runbook_code wins; the 2024 revision is dropped.")
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

The connected helper builds a pre-retrieval tenant, region, and group filter and
emits normalized MLflow retriever documents. It maps the common authorization
contract to each supported provider and fails closed for unsupported filter
shapes. This app runs as its service principal and does not pretend to verify a
developer's own access.
"""),
        c("""
RUN_CONNECTED = False
cloud_results = None
if RUN_CONNECTED:
    from agentic_ops_rag import authorized_search

    resources = session.connected_components(allow_network=True)
    cloud_results = authorized_search(
        resources["retriever"],
        "Explain ERR-PAY-503",
        tenant_id="tenant-alpha",
        region="eastus",
        allowed_groups=("ops-payments",),
        mode="hybrid",
        top_k=8,
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

You separated trusted access scope from relevance hints, watched an unscoped
index probe expose tenant-beta evidence that the governed path excludes, saw
the application rerank retire a stale runbook revision, blocked secret
retrieval, and stopped an operational request at a human approval checkpoint.
Lesson 02 makes chunking and embeddings part of an immutable index release
instead of notebook state.
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

Capture the deployment ID and trace ID before changing production. Attach the
regional health signal, the dependency status, and the incident channel link
so reviewers can replay the decision later.

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
`chunk_id` live in metadata exactly where MLflow's RAG judges expect them.

Now put the naive alternative beside it. A fixed 300-character window has no
idea a code fence exists: on this runbook it slices between the rollback
command and the sentence that says the command still needs approval. A
retriever serving that first window would hand a model the command with its
safety context amputated. The structural chunk keeps them together.
"""),
        c("""
naive_windows = [
    sample_runbook[index : index + 300]
    for index in range(0, len(sample_runbook), 300)
]
severed_window = next(
    window
    for window in naive_windows
    if "propose rollback" in window and "needs approval" not in window
)
intact_chunk = next(
    chunk.page_content
    for chunk in chunks
    if "propose rollback" in chunk.page_content
    and "needs approval" in chunk.page_content
)
print(f"naive fixed windows: {len(naive_windows)}")
print("--- naive window: command severed from its warning ---")
print(severed_window[-120:])
print("--- structural chunk: command and warning stay together ---")
print(intact_chunk)
"""),
        m("""
Chunk identifiers are content-derived, which makes indexing a pure function of
the source: the same content and profile always produce the same IDs, and any
edit — even one character — produces a new ID for exactly the affected chunk.
Re-indexing, cache invalidation, and evaluation lineage all hang off that
property.
"""),
        c("""
rechunked = structural_chunks(
    sample_runbook,
    document_id="synthetic-checkout-recovery",
    doc_uri="synthetic://runbooks/training/checkout",
    max_characters=300,
)
assert [chunk.chunk_id for chunk in chunks] == [
    chunk.chunk_id for chunk in rechunked
]
mutated = structural_chunks(
    sample_runbook.replace("RELEASE_ID", "RELEASE_1D"),
    document_id="synthetic-checkout-recovery",
    doc_uri="synthetic://runbooks/training/checkout",
    max_characters=300,
)
changed_ids = [
    (original.chunk_id, edited.chunk_id)
    for original, edited in zip(chunks, mutated, strict=True)
    if original.chunk_id != edited.chunk_id
]
assert len(changed_ids) == 1
print("chunking twice is byte-stable: identical chunk_ids")
print(f"one-character edit changed exactly one chunk_id: {changed_ids[0]}")
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
    provider="databricks",
    model="embedding-deployment-v1",
    dimensions=1536,
    normalized=True,
    version="1",
)
query_embedding = EmbeddingProfile(
    logical_name="operations-embedding",
    provider="databricks",
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
    provider="databricks",
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
    provider="databricks",
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

You produced stable chunks, watched a naive fixed window sever a command from
its safety warning while the structural chunk kept them together, proved
chunk identifiers are deterministic and content-derived, and proved an
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
configurations = {
    "A_text": (RetrievalMode.TEXT, False),
    "B_vector": (RetrievalMode.VECTOR, False),
    "C_hybrid": (RetrievalMode.HYBRID, False),
    "D_hybrid_reranked": (RetrievalMode.HYBRID, True),
}
retrieval_matrix = {
    name: benchmark(pipeline, cases, mode=mode, semantic_rerank=rerank)
    for name, (mode, rerank) in configurations.items()
}
for metric in ("retrieval/recall_at_3", "retrieval/mrr"):
    print(metric)
    for name, report in retrieval_matrix.items():
        print(f"  {name:18s} {report[metric]:.4f}")
retrieval_matrix
"""),
        m("""
The four configurations now genuinely disagree. Quality metrics are computed
from real row-level retrieval outcomes on this corpus. The latency column is
different in kind: it is labelled `simulated_offline_fixture` and only sketches
the shape of a cost trade-off — real latency evidence comes from measured
traces on the connected path, never from this fixture. Inspect individual
cases before choosing a winner: an average can hide an exact-code regression,
an authorization failure, or lost answerable coverage.
"""),
        c("""
import pandas as pd

def case_row(case):
    row = {"expected": ", ".join(case.expected_document_ids) or "(abstain)"}
    for name, (mode, rerank) in configurations.items():
        result = pipeline.invoke(
            case.question,
            tenant_id=case.tenant_id,
            region=case.region,
            allowed_groups=case.allowed_groups,
            mode=mode,
            semantic_rerank=rerank,
        )
        retrieved = list(result.retrieved_document_ids)
        expected = set(case.expected_document_ids)
        hits = len(expected.intersection(retrieved))
        marker = f"{hits}/{len(expected)}" if expected else "-"
        row[name] = f"[{marker}] " + (", ".join(retrieved) or "(abstained)")
    return row

per_case = pd.DataFrame(
    {case.case_id: case_row(case) for case in cases}
).transpose()
with pd.option_context("display.max_colwidth", 90, "display.width", 400):
    print(per_case.to_string())

divergent = per_case[per_case["A_text"] != per_case["B_vector"]]
assert per_case.loc["paraphrase-checkout-outage", "A_text"].startswith("[0/1]")
assert per_case.loc["paraphrase-checkout-outage", "B_vector"].startswith("[1/1]")
assert per_case.loc["restart-needs-approval", "B_vector"].startswith("[1/2]")
assert per_case.loc["restart-needs-approval", "A_text"].startswith("[2/2]")
for case_id in ("paraphrase-checkout-outage", "restart-needs-approval"):
    assert per_case.loc[case_id, "C_hybrid"].split("]")[0].lstrip("[") in {
        "1/1",
        "2/2",
    }
print()
print("text misses the paraphrase case; vector loses the approval policy on")
print("the exact-code case; hybrid fusion is the only configuration that")
print("recovers both:", sorted(divergent.index))
"""),
        m("""
## Why fusion wins: real ranks, one real query

Take the deployment-outage question from the fixed cases (case
`semantic-checkout-outage`) and compute the two orderings the retriever
actually fuses: the lexical ranking and the embedding ranking over the
authorized corpus. No mock document IDs — this is the corpus you just
benchmarked.
"""),
        c("""
from agentic_ops_rag.offline import (
    cosine,
    deterministic_embedding,
    lexical_score,
    reciprocal_rank_fusion,
)

fusion_query = (
    "Checkout went down immediately after a deployment. "
    "How should on-call recover?"
)
scope_groups = {"ops-payments", "incident-commanders"}
scoped_documents = [
    document
    for document in pipeline.retriever.documents
    if document.active
    and document.tenant_id == "tenant-alpha"
    and document.region == "eastus"
    and scope_groups.intersection(document.allowed_groups)
]
lexical = {
    document.document_id: lexical_score(fusion_query, document)
    for document in scoped_documents
}
query_vector = deterministic_embedding(fusion_query)
semantic = {
    document.document_id: cosine(
        query_vector,
        deterministic_embedding(f"{document.title} {document.content}"),
    )
    for document in scoped_documents
}
lexical_order = sorted(lexical, key=lambda key: (-lexical[key], key))
semantic_order = sorted(semantic, key=lambda key: (-semantic[key], key))
fused = reciprocal_rank_fusion((lexical_order, semantic_order))
fused_order = sorted(fused, key=lambda key: (-fused[key], key))

def show(label, order, scores):
    print(label)
    for rank, document_id in enumerate(order[:4], start=1):
        print(f"  {rank}. {document_id:32s} {scores[document_id]: .3f}")

show("lexical ranking", lexical_order, lexical)
show("embedding ranking", semantic_order, semantic)
show("fused (RRF) ranking", fused_order, fused)

winner = fused_order[0]
assert winner == "alpha-payments-503-current"
assert lexical_order[0] != winner and semantic_order[0] != winner
print()
print(
    f"fused #1 {winner}: lexical #{lexical_order.index(winner) + 1}, "
    f"embedding #{semantic_order.index(winner) + 1} - first on neither list"
)
print(
    f"lexical #1 {lexical_order[0]} sits at embedding "
    f"#{semantic_order.index(lexical_order[0]) + 1} "
    f"and falls to fused #{fused_order.index(lexical_order[0]) + 1}"
)
print(
    f"embedding #1 {semantic_order[0]} sits at lexical "
    f"#{lexical_order.index(semantic_order[0]) + 1} "
    f"and falls to fused #{fused_order.index(semantic_order[0]) + 1}"
)
"""),
        m("""
The runbook that answers the case is first on neither list — the lexical list
is topped by an incidental `on-call` identifier match and the embedding list by
the status-page mirror — yet it wins the fusion because it is strong on both.
RRF combines rank positions, not BM25 and cosine magnitudes, which is what
makes that outcome stable. Azure semantic ranking happens after hybrid fusion
and emits a separate reranker score. Never copy one absolute threshold across
BM25, vector, RRF, semantic ranker, and a different search provider.
"""),
        c("""
# YOUR TURN — TODO: choose candidate and context counts, then run the wide plan.
candidate_k = 50
context_k = 3
action_query = "Restart payments for ERR-PAY-503 now."
wide_plan = pipeline.invoke(
    action_query,
    tenant_id="tenant-alpha",
    region="eastus",
    allowed_groups=("ops-payments", "incident-commanders"),
    mode="hybrid",
    candidate_k=candidate_k,
    final_k=context_k,
)
print("wide plan retrieved:", list(wide_plan.retrieved_document_ids))
print(
    f"latency {wide_plan.latency_ms:.1f}ms "
    f"({wide_plan.measurement_source})"
)
"""),
        c("""
# CHECK YOUR WORK
assert candidate_k == 50, "Semantic ranker needs a broad candidate set to test"
assert 1 <= context_k <= 10, "Keep the final model context deliberately bounded"
assert "alpha-action-approval" in wide_plan.retrieved_document_ids
"Candidate generation and final context are separate decisions."
"""),
        c("""
# Reference solution
narrow_plan = pipeline.invoke(
    action_query,
    tenant_id="tenant-alpha",
    region="eastus",
    allowed_groups=("ops-payments", "incident-commanders"),
    mode="hybrid",
    candidate_k=2,
    final_k=2,
)
print("narrow plan retrieved:", list(narrow_plan.retrieved_document_ids))
print(f"latency {narrow_plan.latency_ms:.1f}ms ({narrow_plan.measurement_source})")
assert narrow_plan.latency_ms < wide_plan.latency_ms
assert "alpha-payments-503-current" in wide_plan.retrieved_document_ids
assert "alpha-payments-503-current" not in narrow_plan.retrieved_document_ids
print()
print("narrowing candidate_k to 2 was cheaper on fixture latency - and it")
print("silently dropped the ERR-PAY-503 runbook itself from the context.")
print("Candidate breadth is a quality decision, not only a cost knob.")
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
    from agentic_ops_rag import authorized_search

    resources = session.connected_components(allow_network=True)
    candidates = authorized_search(
        resources["retriever"],
        "Checkout went down after a deployment",
        tenant_id="tenant-alpha",
        region="eastus",
        allowed_groups=("ops-payments",),
        mode="hybrid",
        top_k=candidate_k,
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
providers. A positive provider score only orders candidates; it cannot make an
unrelated result answerable. The deterministic shell requires identifier or
query/evidence support and abstains when that support is uncertain.
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

You ran a four-configuration ablation whose configurations genuinely disagree
at the case level, watched RRF fuse two real rankings so the runbook that tops
neither list wins, and saw candidate breadth silently decide which evidence
reaches the model. Lesson 04 turns normalized documents into MLflow traces,
deterministic gates, and optional RAG judges.
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
allowlists, region and group authorization, current-evidence selection, and
human approval are code-level policies. LLM judges complement them with
retrieval relevance, sufficiency, and groundedness; an experimental judge is
never the sole safety gate.
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
        m("""
## Plant a defect, watch the gate catch it

A gate that has never failed proves nothing. Break the retriever on purpose:
this subclass ignores the caller's tenant scope the way a mis-built index or a
missing security filter would, letting tenant-beta evidence compete for
tenant-alpha answers. The benchmark is unchanged — only the system under test
is broken — and the deterministic security metrics must catch it.
"""),
        c("""
from agentic_ops_rag import OfflineOperationsRetriever, OperationsRAGPipeline

class TenantBlindRetriever(OfflineOperationsRetriever):
    \"\"\"Deliberately broken: drops the tenant dimension of authorization.\"\"\"

    @staticmethod
    def _eligible(document, filters, allowed_groups):
        # The planted defect: no tenant filter and no group check, as if the
        # index shipped without its security filter. Region and active-state
        # filtering still work, so only the tenant boundary is broken.
        return bool(document.active and document.region == filters.get("region"))

broken_pipeline = OperationsRAGPipeline(
    TenantBlindRetriever(pipeline.retriever.documents)
)
broken_metrics = benchmark(broken_pipeline, cases, mode=RetrievalMode.HYBRID)
broken_gate = release_gate(broken_metrics)
assert broken_metrics["security/tenant_isolation"] < 1.0
assert not broken_gate.passed
print(
    "security/tenant_isolation =",
    f"{broken_metrics['security/tenant_isolation']:.4f} (must be 1.0)",
)
print("gate failures:")
for failure in broken_gate.failures:
    print(f"  {failure.metric}: {failure.reason}")
"""),
        m("""
The tenant-beta ERR-PAY-503 runbook is newer than tenant alpha's, so once the
tenant boundary is gone it even outranks the right answer — recall and MRR
collapse alongside the security metrics. Now confirm the exact same gate
passes for the correct pipeline: the failure above is evidence about the
defect, not gate noise.
"""),
        c("""
assert offline_gate.passed
assert offline_metrics["security/tenant_isolation"] == 1.0
print("correct pipeline: gate passed =", offline_gate.passed)
print(
    "correct pipeline: security/tenant_isolation =",
    offline_metrics["security/tenant_isolation"],
)
"""),
        c("""
# YOUR TURN — TODO: classify every gate metric as deterministic or judge-based.
metric_owner = {
    "security/tenant_isolation": "deterministic",
    "security/region_isolation": "deterministic",
    "security/group_access": "deterministic",
    "security/current_evidence": "deterministic",
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
assert metric_owner["security/region_isolation"] == "deterministic"
assert metric_owner["security/group_access"] == "deterministic"
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
    "security/region_isolation",
    "security/group_access",
    "security/current_evidence",
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
platform-approved scalar ACL field before this lab can run. This evaluation
records two truthful evidence stages even when `candidate_k` equals `final_k`:
the scorer-visible top-level `retriever.final_context` `RETRIEVER` span contains
only current, supported documents supplied to the answer model after
deduplication, while the SDK's raw provider-candidate `retriever.search` span is
nested beneath it. The governed `predict_fn` span owns the complete invocation.
MLflow's evaluation harness can
otherwise enable OpenAI autologging temporarily, so this SDK-owned path disables
that second tracing owner before either evaluation begins.
"""),
        c("""
RUN_CONNECTED = False
connected_evaluation = None
if RUN_CONNECTED:
    import mlflow
    from agentic_ops_rag import OperationsRAGPipeline
    from mlflow.genai.scorers import (
        RetrievalGroundedness,
        RetrievalRelevance,
        RetrievalSufficiency,
        Safety,
    )

    from aai_core.tracing import TraceIntegration, traced

    if RUN_MLFLOW:
        raise RuntimeError(
            "Restart the kernel before connected evaluation: local SQLite tracing "
            "and connected tracing cannot share one process."
        )
    session.context.configure_tracing(integration=TraceIntegration.SDK)
    mlflow.openai.autolog(disable=True)
    resources = session.connected_components(allow_network=True)

    def generate_answer(question: str, retrieved) -> str:
        context = mlflow_documents(retrieved)
        response = resources["model"].generate(
            [
                {"role": "system", "content": "Answer only from supplied evidence."},
                {"role": "user", "content": f"{question}\\nEvidence: {context}"},
            ],
            temperature=0.0,
        )
        return response.content

    connected_pipeline = OperationsRAGPipeline(
        resources["retriever"],
        answer_generator=generate_answer,
    )

    @traced(name="operations-rag.predict", span_type="CHAIN")
    def predict_fn(
        question: str,
        tenant_id: str,
        region: str,
        allowed_groups: list[str],
    ) -> str:
        result = connected_pipeline.invoke(
            question,
            tenant_id=tenant_id,
            region=region,
            allowed_groups=allowed_groups,
            mode="hybrid",
            candidate_k=3,
            final_k=3,
        )
        return result.answer

    judge_model = session.judge_model_uri()
    def evaluation_row(case):
        reference = pipeline.invoke(
            case.question,
            tenant_id=case.tenant_id,
            region=case.region,
            allowed_groups=case.allowed_groups,
            mode="hybrid",
        )
        return {
            "inputs": {
                "question": case.question,
                "tenant_id": case.tenant_id,
                "region": case.region,
                "allowed_groups": list(case.allowed_groups),
            },
            "expectations": {
                "expected_response": reference.answer,
                "expected_document_ids": list(case.expected_document_ids),
            },
        }

    policy_data = [
        evaluation_row(case)
        for case in cases
        if not case.answerable or case.expects_action_proposal
    ]
    rag_data = [
        evaluation_row(case)
        for case in cases
        if case.answerable and not case.expects_action_proposal
    ]
    connected_evaluation = {
        "policy": mlflow.genai.evaluate(
            data=policy_data,
            predict_fn=predict_fn,
            scorers=[Safety(model=judge_model)],
        ),
        "rag": mlflow.genai.evaluate(
            data=rag_data,
            predict_fn=predict_fn,
            scorers=[
                RetrievalRelevance(model=judge_model),
                RetrievalGroundedness(model=judge_model),
                RetrievalSufficiency(model=judge_model),
                Safety(model=judge_model),
            ],
        ),
    }
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

You validated retriever evidence, ran a deterministic release gate, planted a
tenant-scope defect and watched the gate catch it, and prepared an explicit
MLflow 3 judge path. Local fixtures never masquerade as provider quality or
cost evidence. Lesson 05 converts the same measurements into a baseline,
change, result, decision, and immutable application release.
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
    is_release_eligible,
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
also asks whether its gain justifies degradation from the current baseline.
Here the hybrid change genuinely improves recall and MRR on the fixed cases,
so the vector-to-hybrid comparison should come back `adopt`. Note the latency
rule's role: the offline latencies are labelled `simulated_offline_fixture`
and only sketch a trade-off shape, so the policy grants them a wide regression
allowance and a real quality gain is not vetoed by an invented number. In a
connected deployment the latency budget comes from measured traces, and a
change that regressed it would be rejected on real evidence.
"""),
        c("""
comparison = comparison_record(
    reports["B_vector"],
    reports["C_hybrid"],
    baseline_configuration="B_vector",
    change_configuration="C_hybrid",
)
comparison.model_dump(mode="json")
"""),
        c("""
# YOUR TURN — TODO: make an evidence-backed lifecycle decision.
if comparison.failures:
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
    assert not comparison.failures
elif comparison.failures:
    assert learner_decision == "reject"
"The decision follows the recorded gate evidence."
"""),
        c("""
# Reference solution
reference_decision = comparison.decision
assert learner_decision == reference_decision
decision_evidence = {
    "baseline_configuration": comparison.baseline_configuration,
    "change_configuration": comparison.change_configuration,
    "decision": reference_decision,
    "failures": [failure.model_dump(mode="json") for failure in comparison.failures],
    "measurement_source": "simulated_offline_fixture",
}
decision_evidence
"""),
        m("""
## A forged decision does not survive the gate

The decision field is evidence, not authority. Record a genuinely rejected
comparison — moving from hybrid back to text loses recall beyond the
regression allowance — then forge its decision to `adopt` and watch
`is_release_eligible` refuse it anyway: eligibility recomputes the gate from
the recorded metrics and requires the recomputed result to agree with the
record.
"""),
        c("""
downgrade = comparison_record(
    reports["C_hybrid"],
    reports["A_text"],
    baseline_configuration="C_hybrid",
    change_configuration="A_text",
)
assert downgrade.decision == "reject"
forged = downgrade.model_copy(update={"decision": "adopt", "failures": ()})
forged_eligible = is_release_eligible(
    "A_text",
    absolute_gate=release_gate(reports["A_text"]),
    baseline_metrics=reports["C_hybrid"],
    comparison=forged,
    source_state="clean",
)
assert not forged_eligible
print("honest decision:", downgrade.decision)
print("recorded failures:", [failure.metric for failure in downgrade.failures])
print("forged decision:  ", forged.decision)
print("is_release_eligible(forged):", forged_eligible)
print("editing the record does not edit the evidence: the gate is recomputed.")
"""),
        m("""
## Immutable release evidence

Only an eligible choice becomes an application release. The release ties code,
model, prompt, retrieval, evaluation, and environment together. A prompt alias,
mutable index name, or notebook output is not sufficient release lineage.

Eligibility is a conjunction of independently checkable preconditions, so print
them as a checklist rather than a single boolean: a dirty working tree and a
rejected comparison both make `is_release_eligible` return `False`, and an
operator has to be able to tell those situations apart.
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
source_commit = commit_result.stdout.strip()
git_provenance_available = (
    commit_result.returncode == 0
    and state_result.returncode == 0
    and bool(source_commit)
)
source_state = (
    "clean"
    if git_provenance_available and not state_result.stdout.strip()
    else "dirty"
)
print("live source_state:", source_state)
"""),
        c("""
from agentic_ops_rag import ComparisonRecord

selected_name = comparison.change_configuration
selected_gate = absolute_gates[selected_name]
trusted_baseline = reports[comparison.baseline_configuration]
recomputed_gate = release_gate(
    dict(comparison.change),
    baseline_metrics=dict(trusted_baseline),
)
gate_metrics = dict(selected_gate.metrics)
eligibility_checklist = {
    "comparison_is_a_strict_record": type(comparison) is ComparisonRecord,
    "source_tree_is_clean": source_state == "clean",
    "absolute_gate_passed": selected_gate.passed,
    "recorded_baseline_matches_trusted_baseline": (
        dict(comparison.baseline) == dict(trusted_baseline)
    ),
    "selected_configuration_is_the_recorded_change": (
        comparison.change_configuration == selected_name
    ),
    "recorded_change_and_result_match_gate_metrics": (
        dict(comparison.change) == gate_metrics
        and dict(comparison.result) == gate_metrics
    ),
    "recomputed_gate_agrees_with_the_record": (
        dict(recomputed_gate.metrics) == gate_metrics
        and recomputed_gate.failures == comparison.failures
        and recomputed_gate.passed
    ),
    "decision_is_adopt_with_no_recorded_failures": (
        comparison.decision == "adopt" and not comparison.failures
    ),
}
release_eligible = is_release_eligible(
    selected_name,
    absolute_gate=selected_gate,
    baseline_metrics=reports[comparison.baseline_configuration],
    comparison=comparison,
    source_state=source_state,
)
for check_name, check_passed in eligibility_checklist.items():
    print(f"[{'PASS' if check_passed else 'FAIL'}] {check_name}")
print("is_release_eligible:", release_eligible)
assert release_eligible == all(eligibility_checklist.values())
"""),
        m("""
## The digest demonstration and the honest checklist

This notebook also runs in automated verification, where the working tree
legitimately contains the notebook execution itself — so `source_tree_is_clean`
can honestly be `FAIL` while every evidence check passes. The checklist above
always reports the live tree. To keep the mechanism teachable in both worlds,
the cell below first re-verifies that the recorded comparison is adopt-grade
under an explicit `demonstration_source_state`, then builds the release record
and prints its digest. The record itself carries both values: the
demonstration state it was built under and the live state observed at render
time. A publishable release is only ever cut when the live checklist passes
end-to-end from a committed tree.
"""),
        c("""
if release_eligible:
    provenance_note = (
        "live tree is clean: this digest is real, publishable release evidence"
    )
else:
    provenance_note = (
        "live tree is dirty (normal while editing or during automated "
        "notebook execution): the digest below demonstrates the mechanism "
        "under an explicit demonstration source_state and must not ship"
    )
demonstration_source_state = "clean"
adopt_grade_evidence = is_release_eligible(
    selected_name,
    absolute_gate=selected_gate,
    baseline_metrics=reports[comparison.baseline_configuration],
    comparison=comparison,
    source_state=demonstration_source_state,
)
assert adopt_grade_evidence, "never demonstrate a digest from rejected evidence"
release = ApplicationRelease(
    application="operations-rag-assistant",
    release="workshop-hybrid-v1",
    source_commit=source_commit if git_provenance_available else "unavailable",
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
        "comparison": comparison.model_dump(mode="json"),
        "metrics": reports[selected_name],
        "source_state": demonstration_source_state,
        "source_state_at_render": source_state,
    },
    environment="dev",
)
print(provenance_note)
print("decision:", comparison.decision)
print("release digest:", release.digest)
"""),
        m("""
The digest is a canonical hash over the whole record, which is what makes the
release immutable in practice: any change to any field — a different release
name, one edited metric, a swapped comparison — produces a different digest,
so evidence cannot drift silently after the fact.
"""),
        c("""
release_variant = release.model_copy(update={"release": "workshop-hybrid-v2"})
assert release.digest != release_variant.digest
print("workshop-hybrid-v1 digest:", release.digest)
print("workshop-hybrid-v2 digest:", release_variant.digest)
print("one changed field, a different digest: evidence cannot drift silently.")
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

You completed the full lifecycle: baseline, controlled change, result, an
`adopt` decision earned on row-level evidence, and a digest-sealed release
record that refuses forged decisions. The result stays reproducible offline,
while every real model, judge, search, trace, and deployment operation is
explicit, keyless, and governed by the surrounding platform. Lesson 06 asks the
question every one of those numbers deserves: how sure are we?
"""),
    ],
    "06_confidence_intervals_for_release_gates.ipynb": [
        m("""
# 06 Confidence intervals for release gates

## Learning objectives

- read a scorer mean as an estimate with an interval, never as a fact;
- choose between normal and bootstrap intervals for bounded scores;
- diagnose a failing retrieval slice from interval width;
- gate promotion on lower confidence bounds and paired improvements.
"""),
        c(preflight()),
        m("""
## Per-row scores are the raw evidence

`benchmark` collapses every case into one mean per metric. That mean hides how
the number would move if the next ten cases arrived. `benchmark_samples` keeps
the per-case scores in dataset order — `None` where a case is out of scope for
a metric — which is exactly the shape `aai_core.agentkit.statistics` consumes.
The same module computes the uncertainty section of a real AgentKit project's
`agentkit compare` report; this lesson applies it to the workshop's fixed
cases.
"""),
        c("""
from agentic_ops_rag import RetrievalMode
from agentic_ops_rag.evaluation import benchmark_samples, load_cases

pipeline = session.offline_pipeline()
cases = load_cases(course_root / "data" / "evaluation_cases.jsonl")
samples = benchmark_samples(pipeline, cases, mode=RetrievalMode.HYBRID)
{
    "retrieval/recall_at_3": samples["retrieval/recall_at_3"],
    "retrieval/mrr": samples["retrieval/mrr"],
}
"""),
        m("""
## Two interval methods, one bounded scale

Retrieval recall lives on a 0..1 scale and piles up near the ceiling. A normal
approximation draws a symmetric interval around the mean, so on nine scored
rows it promises recall above 100% — a bound no future case can reach. The
percentile bootstrap resamples the recorded rows with replacement and reads
the bounds off the resampled means, so it cannot leave the observed range.
Draws come from a generator seeded per metric: the same rows and configuration
always reproduce the same interval.
"""),
        c("""
from aai_core.agentkit.statistics import StatisticsConfig, build_statistical_evidence
from aai_core.evaluation import MetricDirection, MetricRule

retrieval_rules = (
    MetricRule(
        metric="retrieval/recall_at_3",
        direction=MetricDirection.HIGHER,
        required=0.8,
    ),
    MetricRule(
        metric="retrieval/mrr",
        direction=MetricDirection.HIGHER,
        required=0.75,
    ),
)
normal_evidence, _ = build_statistical_evidence(
    samples, {}, retrieval_rules, StatisticsConfig()
)
bootstrap_evidence, _ = build_statistical_evidence(
    samples, {}, retrieval_rules, StatisticsConfig(method="bootstrap")
)
interval_comparison = {
    normal.metric: {
        "rows_scored": normal.sample_size,
        "mean": normal.mean,
        "normal": [normal.lower, normal.upper],
        "bootstrap": [bootstrap.lower, bootstrap.upper],
    }
    for normal, bootstrap in zip(
        normal_evidence.estimates, bootstrap_evidence.estimates, strict=True
    )
    if normal.metric.startswith("retrieval/")
}
interval_comparison
"""),
        m("""
Both methods agree the recall mean is 0.9375; they disagree about what the
nine scored rows can promise. The normal upper bound exceeds 1.0 — an
impossible recall — while the bootstrap bounds stay inside the scale. Note the
metrics that scored 1.0 on every row, such as abstention accuracy: their
intervals collapse to a point. Perfection on eleven cases is still evidence
from only eleven cases, which is why enforcement pairs every bound with a
minimum
sample size instead of trusting a zero-width interval.
"""),
        m("""
## Interval width is a retrieval diagnostic

A wide interval on a retrieval metric usually means a slice of the dataset is
failing while the rest is fine — not that every answer is uniformly mediocre.
Simulate the classic cause: an index build that silently missed one runbook.
The corpus, cases, and pipeline code stay identical; one document never made
it into the index.
"""),
        c("""
from agentic_ops_rag import (
    OfflineOperationsRetriever,
    OperationsRAGPipeline,
    load_documents,
)

documents = load_documents(course_root / "data" / "operations_documents.jsonl")
missing_runbook = "alpha-payments-latency"
degraded_pipeline = OperationsRAGPipeline(
    OfflineOperationsRetriever(
        tuple(
            document
            for document in documents
            if document.document_id != missing_runbook
        )
    )
)
degraded_samples = benchmark_samples(
    degraded_pipeline, cases, mode=RetrievalMode.HYBRID
)
degraded_evidence, _ = build_statistical_evidence(
    degraded_samples, {}, retrieval_rules, StatisticsConfig(method="bootstrap")
)
width_report = {
    degraded.metric: {
        "full_index": [full.lower, full.upper],
        "degraded_index": [degraded.lower, degraded.upper],
        "width_ratio": round(
            (degraded.upper - degraded.lower) / (full.upper - full.lower), 2
        ),
    }
    for full, degraded in zip(
        bootstrap_evidence.estimates, degraded_evidence.estimates, strict=True
    )
    if degraded.metric == "retrieval/recall_at_3"
}
width_report
"""),
        m("""
The recall interval tripled in width because every case that expected the
missing runbook lost ground — two rows fell to zero, and the approval case
kept only its second expected document — while every other row held its
score. Read `degraded_samples["retrieval/recall_at_3"]` to see exactly which
rows moved. Abstention accuracy and citation integrity stayed perfect on both
indexes — the damage is invisible to them. Width localized the failure to
retrieval before anyone read a single transcript.
"""),
        m("""
## Paired improvement beats interval overlap

Compare the full index against the degraded one and the two recall intervals
overlap — by the folklore rule the difference "is not significant". The rule
is wrong here: both runs scored the same ordered rows, so their noise is
correlated, and the honest test is the per-row paired difference. Passing the
degraded run as the baseline samples pairs each row with itself.
"""),
        c("""
paired_evidence, _ = build_statistical_evidence(
    samples,
    degraded_samples,
    retrieval_rules,
    StatisticsConfig(method="bootstrap"),
)
{
    paired.metric: {
        "pairs": paired.pair_count,
        "mean_improvement": paired.mean_improvement,
        "improvement_interval": [
            paired.lower_improvement,
            paired.upper_improvement,
        ],
    }
    for paired in paired_evidence.paired
}
"""),
        m("""
## Gate on the bound, not the mean

Enforcement turns these intervals into promotion policy. With
`enforce_confidence`, the statistics module adds synthetic rules next to each
threshold: the lower confidence bound must clear the same bar, and the scored
sample must reach `minimum_cases`. In an AgentKit project the identical policy
runs as `agentkit gate`, whose exit codes are a CI contract — `0` pass, `2`
threshold failed. Here the same rules are applied object-shaped.
"""),
        c("""
from agentic_ops_rag.evaluation import benchmark

from aai_core.agentkit.statistics import extend_rules_with_statistics
from aai_core.evaluation import GatePolicy, apply_gate

recall_rule = (retrieval_rules[0],)
strict_config = StatisticsConfig(method="bootstrap", enforce_confidence=True)
_, strict_synthetic = build_statistical_evidence(
    samples, {}, recall_rule, strict_config
)
strict_rules = extend_rules_with_statistics(
    recall_rule, strict_config, allow_missing_regression_baseline=True
)
aggregates = benchmark(pipeline, cases, mode=RetrievalMode.HYBRID)
strict_gate = apply_gate(
    {"retrieval/recall_at_3": aggregates["retrieval/recall_at_3"], **strict_synthetic},
    policy=GatePolicy(rules=strict_rules, allow_missing_regression_baseline=True),
)
{
    "passed": strict_gate.passed,
    "failures": [failure.metric for failure in strict_gate.failures],
}
"""),
        m("""
The lower bound clears 0.8, and the gate still refuses: nine scored rows are
below the default `minimum_cases` of 30. That refusal is the correct verdict
for this dataset, not an obstacle. The cell below lowers the minimum to the
nine rows the recall metric actually scores, purely as a teaching allowance —
a production suite grows to the minimum before enforcing, it never lowers the
bar to its own size.
"""),
        c("""
teaching_config = StatisticsConfig(
    method="bootstrap", enforce_confidence=True, minimum_cases=9
)
teaching_rules = extend_rules_with_statistics(
    recall_rule, teaching_config, allow_missing_regression_baseline=True
)
gate_outcomes = {}
for label, row_samples, target in (
    ("full_index", samples, pipeline),
    ("degraded_index", degraded_samples, degraded_pipeline),
):
    _, synthetic = build_statistical_evidence(
        row_samples, {}, recall_rule, teaching_config
    )
    observed = benchmark(target, cases, mode=RetrievalMode.HYBRID)
    verdict = apply_gate(
        {"retrieval/recall_at_3": observed["retrieval/recall_at_3"], **synthetic},
        policy=GatePolicy(
            rules=teaching_rules, allow_missing_regression_baseline=True
        ),
    )
    gate_outcomes[label] = {
        "passed": verdict.passed,
        "failures": [failure.metric for failure in verdict.failures],
    }
gate_outcomes
"""),
        c("""
# YOUR TURN — TODO: check whether the paired recall gain survives reseeding.
seed_lower_bounds = {}
for seed in (0, 7, 20260823):
    reseeded, _ = build_statistical_evidence(
        samples,
        degraded_samples,
        retrieval_rules,
        StatisticsConfig(method="bootstrap", bootstrap_seed=seed),
    )
    by_metric = {paired.metric: paired for paired in reseeded.paired}
    seed_lower_bounds[seed] = by_metric["retrieval/recall_at_3"].lower_improvement
seed_lower_bounds
"""),
        c("""
# CHECK YOUR WORK
assert set(seed_lower_bounds) == {0, 7, 20260823}
assert all(lower >= 0.0 for lower in seed_lower_bounds.values())
assert set(seed_lower_bounds.values()) == {0.0}
"A lower bound pinned at zero under every seed is a sample-size finding."
"""),
        c("""
# Reference solution
reference_bounds = {}
for seed in sorted(seed_lower_bounds):
    evidence, _ = build_statistical_evidence(
        samples,
        degraded_samples,
        retrieval_rules,
        StatisticsConfig(method="bootstrap", bootstrap_seed=seed),
    )
    lower_by_metric = {
        paired.metric: paired.lower_improvement for paired in evidence.paired
    }
    reference_bounds[seed] = lower_by_metric["retrieval/recall_at_3"]
assert reference_bounds == seed_lower_bounds
{
    "lower_bounds_by_seed": reference_bounds,
    "verdict": "nine pairs cannot certify this margin; grow the suite",
}
"""),
        m("""
## Optional: the same statistics on judge scores

Nothing above is specific to deterministic scorers. In a connected project the
per-row scores come from governed LLM judges, and the identical configuration
lives in `agentkit.yaml` under `statistics:` — the `evaluation-project`
template ships the block commented out, including `method: bootstrap`.
"""),
        c("""
RUN_CONNECTED = False
connected_confidence = None
if RUN_CONNECTED:
    resources = session.connected_components(allow_network=True)
    connected_confidence = {
        "model_provider": resources["model"].provider,
        "next_step": (
            "score the fixed cases through the governed judge, then feed the "
            "per-row scores to build_statistical_evidence unchanged"
        ),
    }
connected_confidence
"""),
        m(
            knowledge_check(
                (
                    "Why does the normal recall interval exceed 1.0 while the "
                    "bootstrap interval cannot?"
                ),
                (
                    "The degraded index widened the recall interval without "
                    "touching abstention accuracy. What does that combination "
                    "localize?"
                ),
                (
                    "The paired lower bound touched zero for one seed. What is "
                    "the honest remediation, and what would be the dishonest "
                    "one?"
                ),
            )
        ),
        m("""
## Recap

You turned scorer means into interval evidence: the bootstrap kept bounds
inside the metric's feasible range where the normal approximation escaped it,
interval width localized a missing runbook that abstention and citation
metrics could not see, and the paired improvement decided what overlapping
intervals could not. The gate refused what eleven cases cannot certify — first on
sample size, then on the lower bound once the index degraded. A real project
enables the same policy in `agentkit.yaml` and reads the verdict from
`agentkit gate` exit codes.
"""),
    ],
}
