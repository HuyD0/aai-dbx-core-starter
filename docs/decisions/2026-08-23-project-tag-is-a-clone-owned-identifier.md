# The project cost-attribution tag is a clone-owned identifier

Status: adopted

## Context

The `project` tag was a literal, repeated in `databricks.yml`'s two target presets
and in both jobs under `resources/`. `aai_core.billing` buckets observed spend by
`custom_tags['project']`, so that tag is the key the cost anomaly watch groups by.

A clone into another tenant therefore inherited a silent failure: nothing stamped
the tag, no test compared it to anything, and the four copies were easy to miss.
The deploy stayed green, the workspace ran, and every Azure VM and
`system.billing.usage` row attributed the clone's own spend to this repository's
name — indefinitely, because nothing ever fails.

Two alternatives were considered.

A **bundle variable with a literal default** collapses four copies into one, which
is most of the benefit for none of the disruption. It was rejected because one
unguarded copy is still an unguarded copy: a clone that never opens
`databricks.yml` still ships upstream's name, and the drift is still invisible.

A **repository variable**, matching how `cost_center`, `team`, and `owner_group`
reach `deploy.yml`, was rejected for the opposite reason. Those three describe who
pays for a *deployment* and can reasonably differ per environment. `project`
identifies the repository itself; it is constant for a clone's whole life. Routing
it through a variable would add a fifth value that defaults silently when unset —
precisely the failure being removed.

## Decision

`project` is a key in `platform-identifiers.json`, registered in
`BUNDLE_VARIABLE_DEFAULTS` and in the fixture-key guard.

That single placement buys three properties from machinery that already exists:
`make sync-templates` stamps the bundle default, `bundle_identifier_drift()`
reports any copy that disagrees, and `test_identifier_fixture_carries_every_required_key`
fails a clone that merges this release without setting its own value. Resource
files reference `${var.project}` and hold no literal.

The same reasoning applies to `node_type_id`, changed in the same pass to a bundle
variable: an Azure VM size is region- and policy-dependent, so a clone must be able
to override it without editing resource files. It is not a fixture key, because it
is a deployment choice rather than an identity.

## Consequences

Adding the key to the required set is a deliberate breaking change. An existing
clone that merges this release fails its next test run until it sets its own value.
That is the point: the guard exists so a new key cannot be silently dropped by the
`merge=keepours` driver that clones use to keep their fixture, and a loud failure
naming the missing key is the cheapest possible migration signal.

Cost attribution now has one place to be wrong instead of four, and being wrong
there fails a test rather than a quarterly invoice.

The rule this establishes, and that future work must not quietly undo: a value that
identifies the clone belongs to the fixture. A value that varies per deployment
belongs to a variable. Anything reaching a billing surface needs a drift check,
because billing is the one output nobody reads until the money is already spent.
