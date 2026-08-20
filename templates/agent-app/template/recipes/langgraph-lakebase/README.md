# Lakebase persistence for the LangGraph recipe

The sibling `recipes/langgraph/` recipe requires production deployments to
inject a durable async `BaseCheckpointSaver` and a persistent `BaseStore`.
This recipe is that production wiring: Lakebase is managed PostgreSQL, so the
native LangGraph Postgres saver and store (`langgraph-checkpoint-postgres`)
are the implementation, and the only Databricks-specific concern — OAuth
credential minting — is isolated in a fail-closed provider. It also builds
the user-scoped long-term memory tools that give the agent durable context
and decision lineage across conversations.

Install the certified recipe dependencies, then the project's optional
`langgraph-lakebase` dependency group, and copy the recipe into `src/app`:

```bash
python -m pip install -r recipes/langgraph-lakebase/requirements.lock
python -m pip install -e '.[langgraph-lakebase]'
LANGGRAPH_STRICT_MSGPACK=true \
  python -m pytest -q recipes/langgraph-lakebase/
```

The default test tier is credential-free. To run the integration tier
against any PostgreSQL server (a local scratch database is enough), set
`AAI_LAKEBASE_TEST_DSN` to its DSN first; those tests prove interrupt →
resume durability across freshly constructed saver instances, duplicate
delivery safety, decision-reason survival through a real checkpoint round
trip, and that setup places every LangGraph table in the dedicated schema.
The tier creates and drops a scratch schema in that database, so reruns
start clean.

## What is provisioned externally

The Lakebase instance, database, role, and grants are provisioned through
the approved external platform process, like every other platform resource.
This recipe never creates, scales, or deletes Lakebase objects — it only
connects to an existing endpoint. At runtime a Databricks Apps `postgres`
resource binding supplies the non-secret coordinates as environment
variables: `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGSSLMODE`, and the
`LAKEBASE_ENDPOINT` autoscaling endpoint resource path. Note the binding's
`value_from: "database"` yields a hostname, while credential minting needs
the endpoint *resource path* — supply `LAKEBASE_ENDPOINT` explicitly. The
binding does not name the application's schema either — supply
`LAKEBASE_SCHEMA`, the lowercase PostgreSQL schema the app role owns.

`LakebaseSettings.from_environment(os.environ)` validates that binding and
refuses unsafe contracts: a malformed hostname or endpoint path, control
characters in identifiers, a schema name that is not a lowercase PostgreSQL
identifier, or any non-TLS `sslmode` for a non-loopback host.

Every pooled connection pins `search_path` to the validated schema when it
is created — as a session `SET`, not the libpq `options` startup parameter,
which pooled endpoints can strip. The LangGraph saver and store issue only
unqualified statements, so this is what places their DDL and every later
lookup in the application's schema instead of a shared `public`. The search
path is the schema alone; if you later enable extension-backed store
features (for example vector indexes), append the extension's schema.

## Credential lifecycle

`LakebaseCredentialProvider` wraps an injected minting call — in production:

```python
from databricks.sdk import WorkspaceClient

workspace_client = WorkspaceClient()
generate = lambda: workspace_client.postgres.generate_database_credential(
    endpoint=settings.endpoint
)
```

The provider caches the OAuth token, refreshes it ahead of expiry, and fails
closed when minting returns no token or one that expires inside the refresh
skew. The pool refreshes the password inside the connect path, so a pooled
connection created long after startup still authenticates with a live
token. The token exists only in connection keyword arguments for the
duration of the connect call — never in a DSN string, environment variable,
log field, exception, or `repr`.

## Construction

```python
import os

from persistence import LakebaseSettings, build_lakebase_persistence
from graph import build_graph  # the sibling langgraph recipe

settings = LakebaseSettings.from_environment(os.environ)
async with build_lakebase_persistence(
    settings, generate, run_setup=False
) as (checkpointer, store):
    graph = build_graph(dependencies, checkpointer=checkpointer, store=store)
    ...
```

The pair satisfies the sibling recipe's async construction check. The
savers' one-time DDL (`run_setup=True`) is the application's own startup
decision; it first ensures the configured schema exists and is owned by the
connected role — creating it when absent, which is exactly what the
binding's `CAN_CONNECT_AND_CREATE` permission covers — and fails closed
when another principal owns it. Run it once per environment, not on every
request path. Everything from the base recipe's contract still applies:
`thread_id` joins durable state to traces, resume payloads are strict
`ApprovalDecision` values, and the graph interrupts before its irreversible
action.

## User-scoped memory and decision lineage

`build_user_memory_tools(store, user_id=...)` returns `get_user_memory`,
`save_user_memory`, and `delete_user_memory` specs whose handlers close over
one user's namespace — the model can only ever touch that user's memories.
Handlers are defensive: a missing memory is a structured not-found result
and deletion is idempotent, so degraded memory never crashes the agent loop.

Memories carry a `kind`. A `preference` is durable user context. A
`decision` records a reviewed approval or rejection and must carry its
`reason_code` and originating `request_id`, so a later session can retrieve
why something was rejected and a review can trace which signal changed which
behavior. When a resume completes with a rejection worth remembering, save
it as a decision memory alongside recording the trace assessment described
in the sibling recipe's README.

The `user_id` is an opaque identifier the serving layer resolves — for a
Databricks App, from the forwarded identity of the authenticated user; for
direct API calls, from an explicitly validated request field. Resolving and
authorizing identity is a deployment decision made outside this recipe. (The
platform console's rejection of on-behalf-of authorization is a
console-specific policy and does not constrain agent applications.) Never
place user identifiers in resource tags, trace inputs, or tool output
metadata.
