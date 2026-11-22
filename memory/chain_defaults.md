# Layer defaults that follow the chain (task 7.10)

**Works:** the defaults live in the plan, not in the bindings' logic. `NodeState` carries
`chain_layer` (set by every node whose `writes_layer()` is `Some`, and by
`with_default_layer`: the nearest wins) and `chain_embedding` (set by
`Transform::writes_representation`, only `pca`). `Plan::input_layer` resolves
`layer=None` to `chain_layer`, so every matrix step (and `rank_genes_groups`) follows
the chain; expressions (`feature`, `pull`, QC leaves) still read the *default* layer.
Default output names come from `ChainStep::default_into` through
`Plan::into_or_default`, called by the thin kyto-py methods before the builder: the
node stores the full name, so the plan JSON of a bare chain equals the explicit one
and `from_json` replays it without the defaults (R6, R7 as written: user, Nov 22, 2026).

**Gotcha:** the Rust builders still take `into: &str` (300 call sites); only
`Option` at the Python boundary. A new step that writes a layer gets the chain for
free through `writes_layer`; a new step with a default name needs a `ChainStep` arm.
A default name the plan already has (ignoring case; `pca` also its `_loadings`) is
`KY-E220`; an explicit `into=` still replaces. `into_or_default` checks `KY-E216`
(sink) first, so a closed plan reports the sink, not a clash.

**Gotcha:** `collect()` returns a plan over a temporary store with the default layer
set again, so the chain restarts at that default layer, not at the layer the collected
plan wrote last.
