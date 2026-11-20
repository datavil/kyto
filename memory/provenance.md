# Provenance: what varies, and where it goes (task 8.5)

- `provenance.json`'s `plan` is the plan **without its sink step**. With the sink, the
  target path landed in the store, so the same plan written to two paths differed in
  bytes (5.20's thread test had to skip the whole file). Readers drop a trailing sink
  step, so stores written before 8.5 still rebuild (`kyto_exec::provenance_plan`).
- Everything that varies between identical runs goes in `run` (timestamps, per-node
  timings). Never put timings in `Statistics`: its `PartialEq` is what the R11 tests
  compare. They ride on the `BlockStream` instead.
- Timings are keyed by **logical** plan node ID (`n<ID>`); stages carry optimized-node
  indices, so `Pipeline::stage_nodes` maps each stage to its optimized node's first
  origin. Per-cell maps and cell steps have no time of their own: they run fused in
  the sink's stream, timed under the sink node.
- Seeds are found generically (any step JSON with an integer `seed`), so a new seeded
  step needs no list update.
- The scan path stays as given: a relative path rebuilds only from the same working
  directory, and relative vs absolute scans of one input differ in bytes (8.7: write
  both runs from the same path).
