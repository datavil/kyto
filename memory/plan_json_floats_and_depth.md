# Exact JSON floats and deep expression trees (task 4.3)

**Floats:** `serde_json` without the `float_roundtrip` feature may parse a float one
ULP off, so a JSON round trip is not exact. kyto-plan enables `float_roundtrip` (it
unifies across the workspace). JSON numbers cannot hold `NaN`/`inf`: `serde_json` writes
them as `null`. Float literals in expressions go through `ExprJson`'s `float_text`
(`"NaN"`, `"inf"`, `"-inf"`); plan parameters are finite because the builders refuse
others. A `NaN` literal never compares equal, so round-trip tests compare the text then.

**Depth:** expressions may be 1000 levels deep (`MAX_EXPR_DEPTH`). `serde_json` refuses
nesting beyond 128 by default, and recursive (de)serialization risks small thread
stacks. `ExprJson` is a flat list (operands by position, root last), written and read
with loops, which also writes a shared `Arc` subtree once instead of once per use.

**Replay, not trust:** `Plan::from_json` rebuilds each node through its public builder,
so every 4.2 check runs again; the resolver is idempotent on resolved expressions, which
is what makes the round trip exact. Keep it so when adding a node or a resolver rule:
add the node to `plan_json.rs` (`Step`) and to the corpus in
`crates/kyto-plan/tests/it/plan_json.rs`.
