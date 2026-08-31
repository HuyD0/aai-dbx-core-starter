# Embedding tokens stay out of the chat-token aggregate
Status: adopted

## Context

`OpenAICompatibleEmbeddingProvider.embed_documents` opened an `EMBEDDING`
span that recorded nothing but timing and identifiers: the provider's usage
was read and discarded. A RAG query embeds before it retrieves, so the one
call every retrieval row makes carried no token evidence at all.

The obvious fix — set `mlflow.chat.tokenUsage`, the key the `LLM` span
already uses — is wrong, and the reason is not visible from the call site.
MLflow's `aggregate_usage_from_span_nodes` walks the whole span tree and
sums that attribute **regardless of span type**, writing the result to the
authoritative trace-level `mlflow.trace.tokenUsage`. `agentkit`'s economics
evidence reads that total first, and when a project configures the
`economics.price_per_1m_input_tokens`/`..._output_tokens` pair it prices the
total at that pair — a rate for the *agent's chat model*. A query embedding
of a few thousand tokens would then be billed at chat rates on every row.
Measured against the pinned MLflow, an `EMBEDDING` span carrying 5,000 input
tokens beside an `LLM` span carrying 100 aggregates to 5,100: a 51x
over-count of the priced input on that row.

## Decision

Record embedding usage on the OpenTelemetry GenAI attribute
`gen_ai.usage.input_tokens`, which MLflow does not aggregate, and never on
`mlflow.chat.tokenUsage`. The governed-attribute allowlist in
`aai_core.tracing` already admits that key with integer validation, so this
adds no attribute surface. Embeddings generate no tokens, so no output side
is recorded.

`economics._span_usage` — whose contract was already documented as "a sum
over the LLM spans" while its implementation summed every span — now skips
spans typed `EMBEDDING`. A span whose type cannot be read still counts:
dropping unlabelled spans would trade an over-count for a silent
under-count.

Alternatives rejected:

- **`mlflow.chat.tokenUsage` on the embedding span.** Corrupts the priced
  total upstream of this repository, in MLflow's own aggregation, where no
  downstream filter can separate the two models again.
- **Filtering `_span_usage` to `LLM`/`CHAT_MODEL` only.** Matches the
  docstring, but drops usage from spans that arrive without a readable type
  in a serialized envelope.
- **Pricing embeddings from a second configured rate pair.** A price table
  by another name; the economics module deliberately ships none.

## Consequences

Embedding spend is visible on the span and in the trace UI, and a future
per-model pricing path can read it, but it does not reach the configured
price estimate. Cost figures for RAG projects are unchanged by this change,
which is the point: they were correct because the tokens were missing, and
they stay correct now that the tokens are recorded.

A later change that "harmonises" the two spans onto one usage key would
silently reintroduce the over-count, with no test failing in MLflow and no
number looking obviously wrong. `test_embedding_span_records_billed_tokens_outside_the_chat_aggregate`
asserts the absence of `mlflow.chat.tokenUsage` for that reason.
