# Changelog

All notable changes to `aai-core` are documented here.

## Unreleased

- Added session-aware traces, secure opt-in MLflow OpenAI/LangChain
  autologging (including compatible LangGraph agents), bounded LLM/tool span
  inputs, outputs, and token usage, complete tool-loop response/usage
  aggregation, and assessment provenance passthrough.
- Added approved logical judge-model resolution and made every generated
  GenAI gate route judges explicitly. Gated row-level scorer errors now fail
  releases instead of disappearing from partial aggregates.
- Upgraded the agent template to preserve governed multi-turn history and gate
  exact tool names, arguments, empty trajectories, and duplicate multiplicity
  with an MLflow `ToolCallCorrectness` compatibility scorer. Responses API
  text parts are normalized while unsupported multimodal input and raw user
  identifiers are rejected or omitted from traced application requests.
- Added an executable domain-policy judge, explicit judge-metric gates,
  human-alignment guidance, and bounded failure-rationale triage to the
  evaluation template.
- Expanded the evaluation lifecycle guidance for judge alignment, held-out
  prompt optimization, conversational evaluation, and cost-quality decisions.

## 0.2.0

- Added the five-template catalog (experiment-starter, prompt-app,
  evaluation-project, rag-app, agent-app) with a shared synced scaffold,
  discovery-driven render-matrix tests, and per-template two-tier release
  gates.
- Retired `agentic-rag` into `rag-app` + `agent-app` (last renderable at tag
  `v0.2.0-agentic-rag-final`).
- Added platform conventions for experiments and evaluation runs:
  `effective_experiment_name` (`/Shared/<team>-<application>-<environment>`
  unless configured), portable `bootstrap()` config discovery
  (`AAI_PLATFORM_CONFIG` or upward search), and
  `EvaluationSuite.run_tracked` — every template gate is now a governed
  MLflow run linking the pinned prompt URI, the registered UC dataset,
  traces, params/metrics, and an `aai.gate_passed` verdict, deep-linked
  from the published report.
- Registered ground truth with the catalog across templates
  (scripts/sync_dataset.py) and added agent register-as-code
  (`deploy_serving.py --register-only`).
- Added SDK experiment logging/reproducibility helpers, `apply_thresholds`,
  `publish_report`, the tool-execution loop (`ToolRegistry`/`run_tool_loop`),
  `structured.generate_structured`, the serving adapter
  (`agent_resources`/`deploy_agent`), and scripted tool calls in
  `aai_core.testing`.

## 0.1.0

- Added the installable AI/ML platform SDK.
- Added native identity, secret-reference, redaction, and tagging foundations.
- Added Databricks and Foundry model adapters.
- Added Azure AI Search and Databricks AI Search retrievers.
- Added MLflow experiment, prompt, tracing, feedback, and evaluation helpers.
- Added RAG schemas and immutable application release manifests.
- Added the Agentic RAG Databricks bundle template.
- Added immutable Unity Catalog volume wheel publication.
