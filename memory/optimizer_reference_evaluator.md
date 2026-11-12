# Checking optimizer rules before an executor exists

Task 4.4 had to show "optimized results equal unoptimized results" with no executor
(5.x). What worked:

- **A test-only reference evaluator** (`crates/kyto-plan/tests/it/support.rs`) runs both
  the logical chain and the optimized arena over a small seeded in-memory dataset, with
  **one semantics function per node shared by both runs**, so any difference comes from
  the rewrites. Per-cell nodes are computed for real; cross-cell ones (HVG, PCA,
  neighbours, clustering) are stand-ins that really depend on every cell (statistics,
  pairwise distances, ranks), so a filter moved past one changes the result. A member the
  optimized scan did not read is an error, which catches projection bugs.
- **Mutation-check the property test.** Break each rule condition on purpose (scope,
  reads vs writes, recompute vs gene axis, single reader, ...) and confirm a test fails
  without a saved `.proptest-regressions` seed (delete it first: a seed saved during a
  mutation run makes the next run fail instantly and proves nothing). 512 cases missed
  "filter reads a layer the node below writes"; 2048 with written layers among the leaf
  choices catch it every time.
- **Rule order matters:** a filter between two maps blocks fusion; fusing first keeps
  `normalize_total(median)` + `log1p` in one pass with the filter above, and fusing again
  after predicate pushdown merges maps the moved filters separated.
- When the 5.x executor lands, rerun the same property against it on the real fixtures.
