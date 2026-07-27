# Platform audit — July 2026

A full-repository audit across three lenses (maintainer, first-time developer,
platform team), plus an upstream/downstream sync review added when the goal of
cloning this repository into an enterprise Git organisation was confirmed.

**Status.** The "Tier −1" fork-proofing items are implemented; see the
CHANGELOG entry for this change. Everything else is recorded as a
prioritised backlog, not as work in progress. Line references are accurate as
of the commit that introduced this file and will age.

> This document quotes identifier values it found drifted, so it is the one
> file exempted from
> `tests/test_smoke.py::test_markdown_does_not_restate_environment_identifiers`.

---


> **Primary goal (stated after the audit):** this is a personal repo that will
> be cloned into an enterprise Git org, with updates continuing to flow
> personal → enterprise. **Section "Lens 0" below is now the top priority** and
> re-orders the roadmap; the rest of the audit stands as written.

## Context

This repo is a keyless Azure Databricks AI/ML platform: an installable SDK
(`aai-core`), five `bundle init` project templates, a 13-part MLflow curriculum,
a guided onboarding Databricks App, and credential-free OIDC CI/CD.

It is unusually well-built for its age. The security model is coherent and
actually enforced by tests (no secrets, credential-free PR CI, SHA-pinned
actions, no `environment:` on credentialed jobs, immutable releases). The SDK is
small and disciplined. The template scaffold, dependency policy, and drift
guards are more rigorous than most production platforms.

The problems are not correctness problems. They are **scale, onboarding, and
carrying-cost** problems: the repo is engineered for one workspace, one
environment, one maintainer, and one tenant, and it pays a large recurring
maintenance tax to stay internally consistent.

This document records the audit across three lenses and a sequenced remediation
roadmap. No code changes are proposed for immediate execution.

---

## Scale of the thing (measured)

| Area | Lines |
|---|---|
| `src/aai_core` (the actual product, 23 files) | 4,012 |
| `src/platform_app` | 2,460 (826 of it CSS) |
| `tests/` (30 files, 216 test functions) | 6,508 |
| `templates/` | 10,813 |
| `examples/` | 5,117 |
| `scripts/` | 1,820 |
| docs surface (`docs/` + README + AGENTS + examples/README) | ~2,700 |

**~33,000 lines to support a 4,000-line SDK.** That ratio is the headline
finding — everything below is a consequence of it.

Two things deserve saying up front because they are genuinely rare: there are
**zero `TODO`/`FIXME`/`NotImplementedError` markers in `src/` or `examples/`**
— the code is finished, not scaffolded — and `tests/test_public_api.py:21`
subprocess-asserts that `import aai_core` pulls in no mlflow/openai/databricks/
azure. The small-surface discipline is enforced, not merely documented.

---

# Lens 0 — Upstream/downstream sync (personal → enterprise)

**Verdict: the repo is currently structured in a way that makes this painful,
but the fix is small, mechanical, and the repo already contains the machinery
to do it.**

## F1. The conflict surface is 19 files / 81 occurrences
Every environment-specific identifier (workspace host, tenant, subscription,
compute policy, artifact volume, GitHub org) is physically duplicated across
tracked source. Measured:

| Occurrences | File |
|---|---|
| 15 | `docs/cloud-setup.md` |
| 7 | `AGENTS.md` |
| 6 | `platform-identifiers.json` ← *the intended single source* |
| 5 | `databricks.yml`, `docs/platform-operations.md`, `tests/test_tracing.py` |
| 4 | `README.md`, `docs/developer-onboarding.md`, **each of 5 `databricks_template_schema.json`** |
| 3 | `docs/developer-guide.md` |
| 2 | `docs/platform-console.md`, `templates/agentic-rag/README.md` |
| 1 | `.github/CODEOWNERS`, `aai-platform.example.yml`, `tests/test_template.py` |

Because the enterprise clone **must** change all of these, and upstream
(personal) will also keep editing them, **every single `git merge upstream/main`
will conflict** — and it will conflict in prose files like `docs/cloud-setup.md`
and `AGENTS.md`, which are the most annoying kind to resolve and the easiest to
resolve *wrongly* (silently reverting the enterprise host back to the personal
one).

This is the single thing standing between the current repo and the seamless
flow you want.

## F2. The five template schemas are byte-identical duplication
Verified: `workspace_host`, `compute_policy_id`, and `aai_core_volume` have
**exactly the same value in all five** `templates/*/databricks_template_schema.json`.
There is zero per-template variation. They are pure copies of
`platform-identifiers.json`.

The repo already has the pattern to eliminate this:
`scripts/sync_template_shared.py` generates files into every template from a
canonical source and runs in `--check` mode in CI. Identifier defaults simply
were never put through it. Extending that script to stamp the three defaults
from the fixture removes 5 of the 19 conflict files with no new concepts.

## F3. Generated enterprise projects would pull the SDK from your personal GitHub
This is the sharpest risk found in this pass.

Every template defaults `aai_core_pip_source` to
`git+https://github.com/HuyD0/aai-dbx-core-starter@v{{.aai_core_version}}`, and
`templates/_shared/files/requirements-ci.txt.tmpl:6` renders it into every
generated project as `aai-core @ {{.aai_core_pip_source}}`.

So unless the enterprise clone changes it, **every generated enterprise
project's CI installs the platform SDK from a personal GitHub account over the
public internet.** That is a supply-chain dependency on an individual's
account, an availability dependency (rename/delete/rate-limit breaks all
downstream CI), and almost certainly a violation of enterprise egress policy.

**And the existing guard does not catch it.**
`tests/test_smoke.py:428` — the test the runbook tells you to trust ("the
identifier cross-checks report each remaining value that must agree") — checks
only `workspace_host`, `compute_policy_id`, and `aai_core_volume`.
`aai_core_pip_source` is validated for *format* only
(`scripts/validate_release.py:267` checks it resolves to an exact tag) and is
**never** cross-checked against `platform-identifiers.json.template_repo`.
`template_repo` itself is cross-checked, but only for `databricks.yml`
(`tests/test_app_resource.py:330`).

Net: a clone can correctly update `template_repo`, pass `pytest -q`, pass
`cloud-verify.sh`, deploy green — and still ship five templates pointing at the
upstream personal repo. The most security-sensitive default is the one the
guard misses.

## F4. Test fixtures hardcode the real repo identity
`tests/test_tracing.py` uses `repository="HuyD0/aai-dbx-core-starter"` in five
places and `aai-platform.example.yml` uses it as a sample value. These are
*test data*, not configuration — they have no reason to name a real repo, and
each one is a needless merge conflict. They should be `example-org/example-repo`.

## F5. One-way flow is a security boundary, not just a convention
Merging **enterprise → personal** would pull the enterprise tenant id,
subscription id, workspace hostname, compute policy id, and volume paths into
a personal (possibly public) repository. None of those are secrets by this
repo's own classification (`AGENTS.md` §3), but they are enterprise
infrastructure reconnaissance and most security teams treat them as
non-public.

The sync must be **strictly one-way**. That needs to be written down and
ideally enforced, because the natural git workflow (`git pull upstream`,
`git push`) makes the wrong direction easy to do by accident.

---

## Recommended target state

**Reduce the enterprise clone's divergence from 19 tracked files to 2.**

After the changes below, the enterprise repo differs from personal in exactly:

1. `platform-identifiers.json` — the environment fixture (by design)
2. `.github/CODEOWNERS` — different reviewers (by design)

Everything else merges clean, forever. Both divergent files are settings files
that upstream rarely touches, and both can be given an automatic merge
strategy so that even *they* never prompt.

### The changes that get you there

| Step | Removes | How |
|---|---|---|
| **A.** Generate the three identifier defaults in all 5 template schemas from `platform-identifiers.json` | 5 files | Extend `scripts/sync_template_shared.py` (+ its existing `--check` CI mode). Pattern already exists. |
| **B.** Add `aai_core_pip_source` to `platform-identifiers.json` and generate it too; cross-check it in `test_smoke.py:428` | closes F3 | Same script. Highest-value item here. |
| **C.** Purge identifier literals from all 7 docs; replace with a pointer to the fixture, or generate the snippet | 7 files | `docs/cloud-setup.md` (15 occurrences) is the worst offender. |
| **D.** Neutralize test/example fixtures to `example-org/example-repo` | 3 files | Pure find/replace. |
| **E.** Generate `databricks.yml`'s identifier lines from the fixture | 1 file | Caveat: `workspace.host` must stay a literal — the Databricks CLI forbids variable interpolation in auth fields. So generate the literal rather than interpolate it. |
| **F.** Delete `templates/agentic-rag/` | 1 file | Already in Tier 0 for other reasons. |

Steps A–D are the high-value 80% and are mechanical. E is fiddly. F is free.

### Git mechanics

**Do not use a GitHub fork.** Cross-org forks frequently break under enterprise
SSO/EMU, and a fork relationship implies a PR path back upstream — the exact
direction that must not exist (F5). Use an **independent enterprise repo with
the personal repo as a read-only `upstream` remote**:

```bash
# in the enterprise clone, once
git remote add upstream https://github.com/HuyD0/aai-dbx-core-starter
git remote set-url --push upstream DISABLED    # makes the wrong direction fail loudly
```

**Sync on tags, not `main`.** `git merge upstream/main` inherits whatever is
mid-flight. Because this repo already has an immutable, gated release story
(`publish-sdk` from `main`, `compatibility.json`, exact version pinning), the
enterprise should consume the same way:

```bash
git fetch upstream --tags
git merge v0.4.0          # a reviewed, tested, known-good point
```

**Merge, never rebase.** Rebasing the enterprise's divergent commits re-applies
the identifier conflict on every sync. A merge resolves it once per release.

**Make the two divergent files auto-resolve.** Add to the enterprise clone's
`.gitattributes`:

```gitattributes
platform-identifiers.json  merge=keepours
.github/CODEOWNERS         merge=keepours
```

with a one-time local config (this is *not* inherited by clone, so it belongs
in the enterprise setup script or a `make` target):

```bash
git config merge.keepours.driver 'true'   # keep %A (ours), discard theirs
git config merge.keepours.name  'always keep the enterprise value'
```

The caveat worth knowing: this silently discards upstream changes to those
files — including a *new key* added to `platform-identifiers.json`. Guard that
by having CI assert the enterprise fixture has the same key set as upstream's,
so a new identifier surfaces as a test failure rather than a silent omission.

**Verify after every sync** — the repo's own gate is exactly right for this:

```bash
make verify                  # credential-free: lock check, lint, tests, build, zizmor
pytest -q tests/test_smoke.py   # the identifier cross-checks (once B lands)
gh workflow run auth-smoke.yml --ref main
```

### Also update the runbook
`docs/enterprise-clone-runbook.md` §3 currently says to hand-edit `AGENTS.md`,
`databricks.yml`, and all five template schemas, and promises the smoke tests
"report each remaining value that must agree." After steps A–D that section
collapses to "edit `platform-identifiers.json`, run `make sync-templates`" —
and the promise becomes true, which it currently is not (F3).

---

# Lens 0b — The platform console as an onboarding tool

**Verdict: this is the best-engineered component in the repo, it already solves
the fork problem for itself, and it is currently under-used — it explains the
lifecycle but does not remove the one blocker that actually stops a developer.**

## C1. It already implements the Lens 0 fix — everywhere else should copy it
`src/platform_app/aai_console/config.py` is the only place in the repo that
treats environment identifiers as **injected data rather than source**:

- `IDENTIFIER_KEYS` is a closed set of exactly three keys — a typo in content
  fails a test instead of rendering an empty string into a pasted command.
- Values arrive as `AAI_CONSOLE_*` env vars from the bundle.
- A test forbids *any* identifier literal under `src/platform_app`.
- Local `make app-run` falls back to reading `platform-identifiers.json` from
  the checkout — and a hosted app never takes that branch, so a missing value
  stays a loud failure in production.

That is precisely the pattern Lens 0 recommends for the other 19 files. You
already built the answer, in the one place the deployment model forced you to.
**Steps A–E of Lens 0 are "make the rest of the repo behave like `config.py`."**

Two other things worth preserving as-is: `assert_platform_state` in `checks.py`
enforces the platform-state/personal-access distinction structurally rather than
by convention, and `_safe_detail` scrubs `DATABRICKS_CLIENT_ID` from error text
because the Databricks SDK's own auth errors interpolate it verbatim. Both are
unusually disciplined. Don't regress them.

## C2. The highest-value enhancement: let the console answer the wizard
This is the one that matters. Finding **D2** is that the wizard asks 21
questions and five of them default to `replace-with-*`:
`model_deployment`, `judge_deployment`, `experiment_id`, `app_usage_policy_id`,
`foundry_endpoint`. A developer cannot answer any of them, so they cannot
self-serve a working project.

**The console is the only component that can answer them**, because it holds a
workspace identity and the developer does not. `checks.py` already has the seam
(`WorkspaceProbe`, a thin testable wrapper). Extending it with list operations —
serving endpoints, MLflow experiments, catalogs/schemas, usage policies — turns
five unanswerable free-text fields into pickers of things that actually exist.

Then extend `generate.py` to emit a populated **`bundle init --config-file`**
JSON alongside the command, instead of only the bare invocation. The developer's
experience goes from *"answer 21 prompts, five of which you must file a ticket
for"* to *"pick a template, pick from dropdowns, paste two commands."*

`generate.py` is currently 75 lines and already the console's "one genuinely
additive capability" (its own docstring). This makes that claim much stronger.

**Two constraints that must be respected, not glossed:**

1. **This is still platform state, not the viewer's access.** The app SP being
   able to list an endpoint does *not* mean the developer can invoke it. The
   pickers must be labelled as platform inventory, and the generated project's
   `scripts/setup_dev.py --check-only` must remain the authority on personal
   access. `assert_platform_state` exists to prevent exactly this conflation —
   extend the guard to cover inventory rows rather than routing around it.
2. **It is an information-disclosure widening.** Anyone with `CAN VIEW` on the
   app would see the workspace's endpoint and experiment inventory. That is
   probably acceptable for a platform-team-run console, but it is a deliberate
   decision, and it should be recorded in `docs/platform-console.md` next to
   the existing on-behalf-of-user rationale.

## C3. Bug: unset `template_repo` produces a broken command in a hosted app
`generate.py:52` — `source = config.template_repo or f"./templates/{request.template}"`.

`ConsoleConfig.template_repo` is optional, and when unset the console emits
`databricks bundle init ./templates/agent-app`, a relative path that only works
inside a checkout. In a **hosted** app that is never correct.

This is **finding F3 surfacing through the console**: an enterprise clone that
forgets `AAI_CONSOLE_TEMPLATE_REPO` gets a console that either hands every
developer a broken command, or — if it is set but left at the upstream value —
confidently instructs the whole enterprise to generate projects from a personal
GitHub repo. `config.py`'s own docstring calls the unset case "keeps a clone
working before its own repository URL is configured," which is right for
`make app-run` and wrong for a deployed app.

**Fix:** when `hosted` is true, treat a missing `template_repo` as a `fail`
check with remediation naming the bundle variable, and add a check that warns
when it matches the known-upstream URL. Cheap, and it closes the enterprise
failure mode at the point a developer would actually hit it.

## C4. `volume_entries` throws away the data it just fetched
`checks.py` — `volume_entries()` does `len(list(...))` and returns a count, so
the console reports "readable: 7 entries." It already has the listing. Returning
the SDK versions instead would let the console show *which* `aai-core` versions
are published and available to pin — directly useful during generation, and
roughly a one-line change.

## C5. No template decision support
`generate.py`'s `TEMPLATE_IDS` is a flat tuple. The README has a good
"use when you want" table for choosing among the five templates; the console
does not surface it. Since the console is the thing a developer is looking at
when they choose, that table belongs there. Cheap, and `content/onboarding.yml`
already has a `Choice` type for it.

## C6. Standing cost vs. availability — split the two capabilities
The console is `MEDIUM` compute with no scale-to-zero and is stopped by default
so a forgotten console cannot bill. That is the right default, but it means
**the onboarding tool is off exactly when a new developer needs it**, and for an
enterprise rollout to N teams that tension gets worse, not better.

The resolution is already latent in the design: **`generate.py` has no workspace
dependency at all.** It is pure string templating over three identifiers plus
`template_repo`. Only `checks.py` needs a live app identity. So:

- Expose the generation half as something with zero standing cost — a
  `make generate TEMPLATE=agent-app` target, or a static page — so onboarding
  never depends on a billable app being awake.
- Keep the console running only where the *checks* half earns its cost.

For the enterprise clone specifically, this also means the console stays
genuinely optional (as `resources/optional/` already intends) without the
onboarding story degrading.

---

# Lens 1 — Maintainer

## M1. Sub-minor dependency ceilings create a permanent bump treadmill
`pyproject.toml` pins `databricks-sdk>=0.122,<0.123`, `databricks-ai-search>=0.77,<0.78`,
`databricks-openai>=0.17,<0.18`, `mlflow>=3.14,<3.15`, `langgraph>=1.2,<1.3`.

Every upstream **patch** release requires a coordinated edit across:
`pyproject.toml` → `dependency-policy.toml` (`certified`) → `uv.lock` →
`templates/_shared/versions.json` → 5 × `templates/*/template/requirements.lock`
→ 5 × `pyproject.toml.tmpl` → `compatibility.json` →
`src/platform_app/requirements.txt`.

Renovate + the `--check` scripts automate most of it, but every bump is still a
reviewed PR on a repo with **one CODEOWNER**. `databricks-sdk` alone ships
roughly monthly. This is the single largest recurring cost in the repo.

**Recommendation:** widen ceilings to the next *minor* (`<0.123` → `<1.0` for
0.x clients is too loose; use `<0.130`-style windows or `~=0.122.0` compatible
release ranges) and let `uv.lock` + the weekly canary carry the reproducibility
guarantee, which is what they exist for. Keep the tight pin only for `mlflow`,
where GenAI APIs genuinely move.

## M2. Documentation is a fourth copy of the source of truth
`platform-identifiers.json` is declared the single source, and
`tests/test_smoke.py:428` enforces it — but **only against template schemas and
`databricks.yml`**. Markdown is uncovered, and it has already drifted:

- `AGENTS.md:62` and `AGENTS.md:320` say the SDK artifact volume is
  `/Volumes/platform/artifacts/python_packages`.
- `platform-identifiers.json`, `databricks.yml`, all five template schemas,
  `README.md:244`, and `docs/platform-operations.md` say
  `/Volumes/dbx_dev/dbx_platform/python_packages`.
- `docs/cloud-setup.md:71` still tells you to `gh variable set SDK_ARTIFACT_VOLUME`
  to the stale path — i.e. following the docs configures CI wrongly.

The same values are physically duplicated across ~13 files.
`docs/enterprise-clone-runbook.md` §3 exists *only* to walk a cloner through this
manual fan-out — the runbook is a symptom, not a solution.

**Recommendation:** extend the identifier smoke test to scan `*.md` for the
identifier literals, and replace the `AGENTS.md` §3 table with a pointer to the
JSON file. `AGENTS.md` also references `docs/archive/`, which does not exist.

## M3. Tests assert on prose, so formatting a doc breaks CI
`tests/test_app_content.py:47-64` asserts every command block in
`content/onboarding.yml` appears **verbatim** in the doc it cites. Reflowing a
fenced code block in `docs/developer-guide.md` or reformatting the `Makefile`
fails the build. `tests/test_app_content.py:67-71` breaks on any doc rename.
`AGENTS.md` *rule numbers* are cited by number in at least four files, so
renumbering a section silently invalidates assertion messages.

The intent (console can't lie about commands) is right; the mechanism is too
brittle. **Recommendation:** have the console *derive* its commands from the
Makefile/identifier fixture rather than assert string equality against prose.

## M4. Test-to-source ratio and meta-test weight
6,508 test lines vs 4,012 SDK lines, and a large share test the *repository*
rather than the SDK: `test_smoke.py` (580), `test_examples_runner.py` (697),
`test_template.py` (485), `test_app_*.py` (930), `test_notebook_setup.py` (328).
This is defensible for a governance repo but it means most CI time and most
review effort is spent on scaffolding, not product.

## M5. Two scripts carry 63% of the script surface
`scripts/validate_release.py` (587) validates SDK + template + dependency +
wheel + release-tag invariants in one file. `scripts/examples.py` (567) is the
curriculum runner with its own 697-line test. Both are prime split candidates.

## M6. Four independent redaction implementations, two drifting key lists
Redaction — the thing the security model most depends on — is written four
times:

- `logging.Redactor.redact` (`logging.py:27-40`) — matches on *values*
- `tracing._redact_payload` (`tracing.py:607-682`) — matches on *keys*
- `tracing._sanitize_trace_metadata` (`tracing.py:567-604`)
- `experiments._safe_parameters` (`experiments.py:246-257`)

`experiments._SENSITIVE_NAMES` (`experiments.py:21-29`) and
`TracePolicy.redacted_keys` (`tracing.py:56-65`) are near-identical lists with
no mechanism keeping them in sync, and `tracing._is_redacted_key` duplicates the
normalization inside `_safe_parameters`. Adding a key to one list and not the
other is a silent leak. This is the most security-relevant duplication in the
repo. Related: `as_mlflow_document()` is implemented identically in
`providers/types.py:68-78` and `rag.py:57-67`; the "MLflow requires the `genai`
extra" `RuntimeError` is copy-pasted verbatim in three modules; and the
freeze/thaw validator+serializer pair is copy-pasted into four models
(`runtime.py`, `agents.py` ×2, `deployment.py`, `evaluation.py`).

## M7. `tracing.py` is 2.4× the next-largest module
749 lines carrying five capture modes, four integration modes, a `GovernedSpan`
proxy, process-wide config-signature enforcement, and sync+async × full+bounded
— **four distinct code paths inside one `traced()` decorator**
(`tracing.py:234-336`). It also has the largest test file (931 lines). One of
those five capture modes, `TraceCaptureMode.REDACTED` (`tracing.py:32`), has no
distinct branch in `sanitize_trace_payload`, is absent from `TracePolicy`'s
docstring, and appears in **no test** — it is a dead enum member that a
developer can select and silently get FULL-with-key-redaction behavior.

## M8. Small drift and dead weight
- `templates/agentic-rag/` is a tombstone README only — still discovered by
  humans browsing `templates/`.
- `AGENTS.md` §3 retains "Legacy" Entra app/client/object ids inline alongside
  the dedicated ones.
- `docs/mlflow-cookbook-assessment.md` (159 lines) is a dated point-in-time
  review that will rot; it is not a guide.
- `dependency-canary.yml:63-83` re-implements `ci.yml`'s provider-import
  heredoc almost verbatim.
- `src/aai_core/__init__.py` hardcodes a `0.3.0` version fallback — a fifth
  place the version lives.
- `logging.configure_logging`'s docstring says "Configure the root logger once"
  but it clears and reinstalls handlers on every call (`logging.py:78`) —
  contrast `tracing.configure_tracing`, which genuinely enforces once-per-process.
- `PlatformContext.configure_logging` is annotated `-> None` (`context.py:91`)
  and silently discards the `Redactor` the underlying function returns.

---

# Lens 2 — First-time developer (never used Databricks, MLflow, or LangGraph)

## D1. The zero-credential path is genuinely good — and hard to find
`git clone && make quickstart && make local-lifecycle && make local-ui` works
with no cloud account at all, writing to a local SQLite MLflow store. That is
the single best thing in this repo for a newcomer.

But there is **no `docs/quickstart.md`**. The reader must assemble day 1 from
README §"Start locally", `docs/developer-onboarding.md` (which is mostly a
*platform-team* checklist), and `docs/developer-guide.md` §0-2. The README's
"Learning paths" section is a flat 25-item list mixing 13 examples with 8 docs
in no pedagogical order — `docs/genai-lifecycle.md`, the most important
conceptual doc, is second-to-last.

**Recommendation:** one `docs/quickstart.md` that is the only thing linked from
the top of the README, ending in "you now understand X, go here next."

## D2. The template wizard asks 21 questions, 5 of which you cannot answer
`templates/agent-app/databricks_template_schema.json` has 21 properties.
Five carry `replace-with-*` defaults that require the platform team:
`model_deployment`, `judge_deployment`, `experiment_id`, `app_usage_policy_id`,
`foundry_endpoint`.

A first-time developer therefore **cannot generate a working project alone**.
They generate one, it renders, and it fails at the first cloud call with an
error about an endpoint they've never heard of. There is no "give me something
that runs against fakes so I can see the shape" mode, despite `aai_core.testing`
already providing exactly the fakes that would make it possible.

**Recommendation:** a `--sandbox`-style wizard answer that renders the project
wired to `aai_core.testing` fakes, so `make test` and `evals/offline_checks.py`
pass on generation with zero platform tickets. This is the highest-leverage
onboarding change available.

## D3. `aai-core doctor` says "configuration is valid" when there is no config
Verified. `run_doctor` defaults `config_path` to the **literal relative string**
`"aai-platform.yml"` (`diagnostics.py:25`), bypassing `find_platform_config()`,
and `PlatformSettings.load` treats a missing file as an empty document
(`runtime.py:143`). So running `aai-core doctor` from a subdirectory, or in a
project with no config at all, prints `configuration: pass`.

This is the first diagnostic a stuck newcomer runs, and it confidently tells
them the wrong thing. There is no test for the missing-config case —
`tests/test_diagnostics.py:21` writes a valid file, `:38` covers only the
prod-placeholder failure. **This is a small, unambiguous bug and the cheapest
high-value fix in the audit.**

Related: configuration is only meaningfully *validated* in strict environments.
In `dev` you can typo `catalog`, `schema`, or `team` and nothing complains until
an MLflow call fails much later with a Databricks-side error.

## D4. Errors are excellent where they exist and absent where they don't
The design is right: `AaiCoreError` carries a stable machine-readable `code` and
a human `remediation`, and the docstring calls the error message "the platform's
first support channel." The provider layer delivers on it — capability preflight
before the HTTP call (`openai_compatible.py:131-138`), an HTTP-status →
remediation table for 401/403/404/429 (`:38-51`), an event-loop preflight that
names `create_native_async_client()`, and resolver errors that point at the exact
YAML key (`search.py:60-66`).

But the coverage is thin: **10 `remediation=` sites against 74 `raise`
statements** (verified). Forty-seven raises are plain `ValueError`/`TypeError`/
`RuntimeError` with no code and no remediation, and they sit exactly where a
newcomer lands — `PlatformSettings.validate` (`runtime.py:104-119`),
`ResourceContext.validate` (`tags.py:55-76`), and every `contracts` validation
error. So the beginner's most common failure (misconfigured YAML) produces the
*least* helpful error class, while the expert's failure (a 429 from a serving
endpoint) produces the best one. That is backwards.

Three concrete sub-issues:
- **Inconsistent import guarding.** Only `databricks_openai` gets a friendly
  error (`resolver.py:119-132`). `foundry`/`azure_apim` (`resolver.py:140,159`),
  `azure.search.documents` (`:218`), `databricks.ai_search` (`:239`), and both
  `identity.py` factories raise a raw `ModuleNotFoundError` for the identical
  class of problem — "you didn't install the extra."
- **`ProviderRequestError` is not exported** from
  `providers/__init__.py:24-39`, while its three siblings are. The one error a
  caller most wants to catch requires reaching into `aai_core.providers.types`.
- **SDK errors give repo-local instructions.** "run `make examples-install` and
  use `.venv/bin/python`" appears in three `RuntimeError`s (`tracing.py:745`,
  `experiments.py:148`, `prompts.py:109`) and one remediation
  (`resolver.py:129-131`). Someone who ran `pip install aai-core` is told to run
  a Makefile target they do not have.

## D5. The curriculum front-loads governance vocabulary
Examples 00-04 are excellent and deterministic. But the framing
("baseline → change → result → decision", "cost coverage", "governed trace",
"evidence chain") is platform vocabulary, not developer vocabulary. Someone who
has never traced an LLM call meets `ResourceContext` with nine mandatory fields
before they have made a single model call. `examples/00` requires the developer
to supply `application`, `project`, `environment`, `team`, `owner_group`,
`cost_center`, `data_classification`, `lifecycle`, `repository`, `release`
just to say hello.

**Recommendation:** a `ResourceContext.for_learning()` (or `dev_context()` from
`aai_core.testing`, which already exists — just use it in example 00) so the
first example is five lines.

## D6. The stable adapter is sync and non-streaming
Every real chat/agent UI needs token streaming. The SDK's paved road doesn't
have it — you must drop to `native_client` / `create_native_async_client()`.
That is a defensible platform decision, but it means the *first real product
requirement* pushes a developer off the paved road, which undermines the SDK's
value proposition on day one. Worth stating explicitly and early in the docs
rather than in `providers/openai_compatible.py:159`.

## D7. LangGraph is optional and unwrapped — good, but under-explained
`recipes/langgraph/` is opt-in with a durability contract test. For someone who
has never used LangGraph, there is no "why would I reach for this" guidance in
the main docs — only the recipe README.

---

# Lens 3 — AI/ML platform team offering this to developers

## P1. There is no path to production. Anywhere.
This is the most serious structural finding.

- `databricks.yml` has exactly one target: `dev`. Prod is a commented-out block.
- **All five templates** render exactly one target: `dev`.
- The federated credential subject is
  `repo:…:ref:refs/heads/main` — one branch, one workspace.
- `AGENTS.md` §4 rule 4 forbids GitHub `environment:` gates without a new FIC.
- `docs/cloud-setup.md` "Adding a prod target" is guidance, not scaffolding.

So the platform teaches a rigorous evaluation-gated lifecycle and then stops at
a dev workspace. The first team that ships will invent its own promotion
mechanism, and the governance model will fork immediately.

**Recommendation:** this must be designed before wider rollout — multi-target
bundles, per-environment FICs, and an approval gate are a coherent unit of work,
not three separate tickets.

## P2. `data_classification` is required, validated, and does nothing
It is a required tag (`tags.py:36`), but it is a free-form `str` — not a
`StrEnum`, unlike `lifecycle`, and contrary to `AGENTS.md` §5's own rule about
platform-owned policy vocabularies. More importantly it has **zero runtime
effect**: it never influences trace capture, payload redaction, evaluation
storage, or retention (`grep data_classification src/aai_core` → 4 hits, all
plumbing).

`TracePolicy.redacted_keys` covers credentials only (`access_token`, `api_key`,
`password`, …). It does **not** cover PII. A RAG or chat app on this platform
writes raw user prompts into MLflow traces in Unity Catalog by default, at
`BOUNDED` capture with a 4,096-char string limit. For anything handling customer
data that is a compliance incident waiting to happen, and the docstring's answer
("install a native MLflow processor") pushes the hardest problem onto every
application team independently.

**Recommendation:** make `data_classification` a `StrEnum` and have it *select*
the default `TracePolicy` — e.g. `confidential`/`restricted` defaults to
capture-metadata-only. This is the single highest-value SDK change in the audit.

## P3. Silent downgrade of production guardrails on an environment typo
`PlatformSettings.strict` (`runtime.py:72`) keys off a hardcoded set:
`{test, staging, uat, prod, production}`. An environment named `prd`, `prod-eu`,
or `production-uk` is **not strict** — placeholder tags, `azure_identity=auto`,
and missing catalog/experiment all pass silently. Failure mode: a real
production deployment with `owner_group: group:my-team-owners` and no identity
selection, and nothing complains.

**Recommendation:** invert the default — unknown environments are strict, or
`environment` becomes a closed vocabulary.

## P4. Bus factor of one
`.github/CODEOWNERS` is `* @HuyD0`. Combined with M1's bump treadmill, protected
`main`, and required code-owner review, every dependency PR, every template
change, and every clone question funnels through one person. This does not
survive contact with 10 consuming teams.

## P5. Single-tenant by construction; cloning is a manual fan-out
The FIC subject embeds immutable repo/owner ids, so a clone must re-mint
identity — correctly documented. But cloning also requires hand-editing
identifiers across ~13 files, of which only the schema defaults and
`databricks.yml` are test-covered. `docs/enterprise-clone-runbook.md` is 115
lines of manual steps.

## P6. No runtime cost or rate guardrails
`cost_center` tagging attributes spend *after the fact*. There is no per-app
token budget, no request-rate ceiling, no circuit breaker. `resolver.py` gives
`max_retries` and `timeout_seconds` per model. An agent loop with a bad prompt
can burn a serving endpoint's quota, and the platform's only response is a
billing report next month. For a platform whose stated concern is cost
attribution, the absence of *cost control* is notable.

## P7. Support surface is undefined
There is no `SUPPORT.md`, no issue templates, no SLA, no versioned deprecation
window beyond `docs/versioning.md`'s prose. `AaiCoreError` carries a stable
`code` and `remediation` — genuinely good, and the right foundation — but
nothing maps codes to runbook entries.

## P8. Observability of the platform itself
The console reports platform state as the app SP, and is stopped by default. But
there is no telemetry answering the questions a platform team lives on: how many
projects were generated, from which template, on which SDK version, and which
ones are still passing their gates. `.aai-template.json` provenance stamps exist
in every generated project — that is the hook, unused.

---

# Remediation roadmap

Re-ordered around the stated primary goal: get the enterprise clone to a state
where upstream syncs are boring. Everything in the original audit is retained
below it.

### Tier −1 — DONE (implemented in the change that added this document)

Doing these first was worth far more than doing them later: once the enterprise
repo is cut, every one of them becomes a change you have to *merge* across the
divergence you were trying to remove.

The clone's divergence is now two files — `platform-identifiers.json` and
`.github/CODEOWNERS` — both marked `merge=keepours` in `.gitattributes`.

| # | Item | Removes |
|---|---|---|
| 0a | Add `aai_core_pip_source` to `platform-identifiers.json`; generate it into all 5 schemas; cross-check it in `test_smoke.py:428` | **F3** — the supply-chain risk |
| 0b | Generate `workspace_host` / `compute_policy_id` / `aai_core_volume` into all 5 schemas via `scripts/sync_template_shared.py` | 5 conflict files |
| 0c | Purge identifier literals from the 7 docs (`docs/cloud-setup.md` first — 15 occurrences) | 7 conflict files |
| 0d | Neutralize `tests/test_tracing.py`, `tests/test_template.py`, `aai-platform.example.yml` to `example-org/example-repo` | 3 conflict files |
| 0e | Generate `databricks.yml`'s identifier lines (literal, not interpolated — CLI forbids interpolation in auth fields) | 1 conflict file |
| 0f | Rewrite `docs/enterprise-clone-runbook.md` §3 to "edit the fixture, run `make sync-templates`" | makes the runbook true |
| 0g | Add the `.gitattributes` + `merge=keepours` recipe and the one-way-remote setup to the runbook, plus a CI check that the fixture's *key set* matches upstream | makes syncs non-interactive |

| 0h | Make hosted+unset `template_repo` a `fail` check (`generate.py`, `checks.py`) | **C3** — F3 at the point of use |
| 0i | `make resolve-upstream` for the one conflict class that survives: upstream changing its *own* identifiers | residual F1 |

> Deviation from the plan: 0h originally also proposed warning when
> `template_repo` equals the known-upstream URL. That would require an
> identifier literal under `src/platform_app`, which
> `tests/test_app_content.py` forbids precisely because such a literal is what
> makes a clone silently wrong. Fail-on-unset is the real failure mode and is
> what shipped.

**Exit criterion:** `grep -rE '<host>|<tenant>|<sub>|<policy>|<org>' . --exclude-dir=.git --exclude=uv.lock` returns hits in exactly two files: `platform-identifiers.json` and `.github/CODEOWNERS`.

### Tier 1b — Console (pairs with Tier 1; same goal, different surface)

| # | Item | Closes |
|---|---|---|
| 1a | `volume_entries` returns versions, not a count | C4 |
| 1b | Surface the README template-decision table via the existing `Choice` type | C5 |
| 1c | **Workspace inventory pickers** — extend `WorkspaceProbe` with endpoint / experiment / catalog / usage-policy listing; extend `assert_platform_state` to cover inventory rows | **C2**, and most of **D2** |
| 1d | `generate.py` emits a populated `bundle init --config-file` JSON | completes C2 |
| 1e | Split generation (no workspace dependency) from checks so onboarding survives a stopped console | C6 |

1c/1d and the Tier 1 item 11 sandbox mode attack **D2** from opposite ends —
the sandbox removes the *need* for the five values, the console *supplies*
them. Either alone is a large improvement; both is the complete answer. If only
one gets built, build the sandbox first: it works with no app running and no
enterprise provisioning.

### Tier 0 — Do now (hours, no risk, no design decisions)
1. **Fix `aai-core doctor`** — route `config_path=None` through
   `find_platform_config()` and report `fail`/`warn` when no config is found.
   Add the missing-config test to `tests/test_diagnostics.py`. (D3.)
2. Fix the SDK artifact volume drift in `AGENTS.md:62`, `AGENTS.md:320`, and
   `docs/cloud-setup.md:71`; extend `tests/test_smoke.py:428` to scan `*.md`
   for identifier literals so markdown can no longer drift.
3. Export `ProviderRequestError` from `providers/__init__.py`. (D4.)
4. Reword the four "run `make examples-install`" messages to lead with the
   `pip install aai-core[...]` form. (D4.)
5. Replace the `AGENTS.md` §3 identifier table with a pointer to
   `platform-identifiers.json`; move the legacy Entra app rows to a clearly
   marked historical note.
6. Create `docs/archive/` or remove the `AGENTS.md` §11 reference.
7. Delete `templates/agentic-rag/`; tag `v0.2.0-agentic-rag-final` and the
   CHANGELOG already carry the history.
8. Either implement or remove `TraceCaptureMode.REDACTED`. (M7.)
9. Use `aai_core.testing.dev_context()` in `examples/00` so hello world is
   five lines instead of a ten-field `ResourceContext`.

### Tier 1 — Onboarding (days, high developer value)
10. `docs/quickstart.md` as the single day-1 entry point; restructure the
    README "Learning paths" list into a sequenced path rather than a flat
    25-item mix of examples and reference docs.
11. **Wizard "sandbox" mode**: render every template wired to
    `aai_core.testing` fakes so a generated project passes `make test` and
    `evals/offline_checks.py` on generation, with zero platform tickets.
    (D2 — the single best onboarding change available.)
12. Give `PlatformSettings.validate` / `ResourceContext.validate` /
    `contracts` errors the `AaiCoreError` code+remediation treatment the
    provider layer already has, and guard the five unguarded provider imports
    the way `databricks_openai` is guarded. (D4.)
13. State the sync/non-streaming adapter boundary prominently in the README
    and developer guide, with the async escape hatch shown once, early.

### Tier 2 — Governance correctness (weeks, highest platform value)
14. `data_classification` → `StrEnum`, wired to *select* the default
    `TracePolicy`; add a PII-aware redaction tier. (P2 — highest-value SDK
    change in the audit.)
15. Consolidate the four redaction implementations behind one key list and one
    matcher. Do this **before** 14, since 14 adds a fifth caller. (M6.)
16. Make `PlatformSettings.strict` fail-safe for unknown environments. (P3.)
17. Design the dev → prod promotion path as one unit: multi-target bundles,
    per-environment FICs, approval gate, and template scaffolding for all five
    templates. (P1 — the largest single piece of work in this audit, and the
    one most likely to be forked by the first team that ships without it.)

### Tier 3 — Carrying cost (weeks, compounding)
18. Widen dependency ceilings from sub-minor to minor windows; let `uv.lock`
    and the weekly canary carry reproducibility. (M1.)
19. Replace verbatim doc-string assertions in `tests/test_app_content.py` with
    commands derived from the Makefile and identifier fixture. (M3.)
20. Split `scripts/validate_release.py` along its four concerns; de-duplicate
    the canary/CI provider-import heredoc; factor the freeze/thaw pair out of
    its four copies. (M5, M6, M8.)
21. Reduce `traced()`'s four code paths in `tracing.py`. (M7.)
22. Add a second code owner. (P4.)

### Tier 4 — Platform maturity
23. Runtime cost/rate guardrails in the model adapter. (P6.)
24. `SUPPORT.md`, issue templates, and an error-code → runbook map — the
    `AaiCoreError.code` field is already the right hook. (P7.)
25. Fleet telemetry from `.aai-template.json` provenance stamps. (P8.)

---

## Verification

Nothing above changes behavior yet. When items are executed, the existing gates
are sufficient — this repo's verification story is one of its strengths and
needs no new machinery. Note the suite is unrunnable from a bare checkout:
`make install` (which requires `uv` **exactly** 0.8.23) must come first.

```bash
make install
make check          # template sync, format, tests, build
make verify         # ./scripts/cloud-verify.sh — the full credential-free gate
make quickstart     # proves a fresh clone still works with no credentials
make local-lifecycle
python scripts/validate_release.py --wheel dist
```

Per-item acceptance:

- **Tier −1 (0a–0e)**: the exit criterion grep above must return exactly two
  files. Then `make check` (which already runs `sync_template_shared.py --check`)
  must pass, and `scripts/validate_release.py --wheel dist` must still accept
  the generated `aai_core_pip_source` as an exact-tag reference.
- **Tier −1 (0a)** specifically: change `template_repo` in
  `platform-identifiers.json` to a dummy value and confirm
  `pytest -q tests/test_smoke.py` **fails**. Today it passes — that is the bug.
- **Tier −1 (0g)**: rehearse the whole loop before it matters. Create a scratch
  clone, point `upstream` at the personal repo, change every value in
  `platform-identifiers.json`, then `git merge` an upstream tag and confirm zero
  conflict prompts and that `make verify` passes with the enterprise values
  intact.
- **Tier 1b**: the console has a strong existing harness — `tests/test_app_server.py`
  (15 tests), `test_app_resource.py` (23), `test_app_redaction.py` (10, including
  a real uvicorn spawn). Extend those rather than adding files. For 1c
  specifically, `WorkspaceProbe`'s constructor already takes an injected client,
  so inventory listing is testable with no cloud identity — and the
  `assert_platform_state` guard must have a test proving inventory rows cannot
  be rendered under a personal-access heading.
- **Tier 0 item 1** (`doctor`): new test in `tests/test_diagnostics.py` running
  `run_doctor()` from a `tmp_path` with no `aai-platform.yml`, asserting the
  `configuration` check is not `pass`. Also verify by hand:
  `cd /tmp && aai-core doctor` must not report a valid configuration.
- **Tier 0 item 2** (identifier drift): extend
  `tests/test_smoke.py::test_identifier_fixture_is_the_single_source_of_truth`
  to glob `**/*.md` and assert no stale volume/host/policy literal appears.
  It should fail before the fix and pass after.
- **Tier 1 item 11** (sandbox render): add one combo per template to
  `tests/template_matrix.py`. The deep-tier mechanism documented at
  `tests/template_matrix.py:1-11` already renders, lints, and runs the
  generated `pytest` — the sandbox combo is exactly that, with the assertion
  that it needs no `replace-with-*` value.
- **Tier 2 items 14-16**: extend `tests/test_tracing.py` (931 lines, already
  covers `TracePolicy` and redaction) and `tests/test_runtime.py` rather than
  adding new files. For item 15, the regression test worth writing is a single
  parametrized case asserting all redaction paths agree on the same key list.
- **Tier 2 item 17** (prod path): `databricks bundle validate -t prod` against
  a real workspace, plus `.github/workflows/auth-smoke.yml` re-run for the new
  FIC subject. This one genuinely cannot be verified credential-free.
