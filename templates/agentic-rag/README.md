# agentic-rag — retired

This template was split into two focused successors:

| If you wanted | Use instead |
|---|---|
| Retrieval-centric apps (chunking, indexing, grounded generation, groundedness gate) | [`templates/rag-app`](../rag-app) |
| Tool-centric agents (tool loop, structured outputs, serving, monitoring) | [`templates/agent-app`](../agent-app) |

Both successors carry all of agentic-rag's hardening — the multi-judge
release gate, generated credential-free CI, prompt registration/promotion,
and the provenance stamp (their `.aai-template.json` records
`"supersedes": ["agentic-rag"]`).

```bash
databricks bundle init https://github.com/HuyD0/aai-dbx-core-starter \
  --template-dir templates/rag-app --output-dir my-rag-app
# or --template-dir templates/agent-app
```

**Reproducing an old render** (for projects generated from agentic-rag):
the last commit containing this template is tagged; pin it explicitly:

```bash
databricks bundle init https://github.com/HuyD0/aai-dbx-core-starter \
  --tag v0.2.0-agentic-rag-final \
  --template-dir templates/agentic-rag --output-dir my-agent
```

Generated projects identify themselves via `.aai-template.json`
(`"template": "agentic-rag"`); migrate them by regenerating from a
successor template and porting `src/app` code — the SDK surface they use is
unchanged.
