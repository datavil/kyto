# scanpy neighbours and `hnsw_rs` (task 6.6)

**scanpy's exact path** (1.12.5, fewer than 4096 cells or euclidean below 8192):
`KNeighborsTransformer(algorithm="brute", n_neighbors)` then `fit_transform(X)`: the
graph has `n_neighbors + 1` entries a row including the cell (distance mode adds one),
scanpy keeps the first `n_neighbors` columns and drops column 0 as "the cell itself",
so `distances` holds `n_neighbors - 1` a row. The stored distances are `float64` but
only `float32`-precise (8.6e-8 relative to the exact `f64` distance of the `float32`
`X_pca` on pbmc3k). The reference CSR has columns sorted per row.

**Breaks (scanpy):** with duplicate cells, column 0 may be a duplicate, not the cell:
scanpy then keeps the cell as its own neighbour (a self loop at 0) and loses the
duplicate. The synthetic fixture has duplicates (cells 38, 39, 125): 10 of 500 rows
differ from a by-position exclusion, all explained by that or by tie order at the last
neighbour. pbmc3k has none: every row equal. Compare rows with those two explanations
allowed, never as plain set equality (deviations.md).

**Works:** the scanpy reference `neighbors.npz` searched the same `X_pca` stored in
`pca.npz` (same `make_reference` run), so kyto's search can be checked on that
embedding exactly, independent of kyto's PCA. On kyto's own PCA, compare as overlap.

**`hnsw_rs` 0.3.5 and R11:** its level generator is `StdRng` seeded with a fixed 397
behind a mutex, so `parallel_insert` gives a different index per run; insert one row
at a time (fixed order, shuffled by kyto's seed) and the index is the same on every
run; `search` is read-only, so queries can run in parallel. `extend_candidates` (off by
default) iterates a randomly seeded `hashbrown` map: keep it off. Distances inside the
index are its own `f32` ones: re-rank the candidates with kyto's kernels so every
written distance is exact and ties follow kyto's rule. Its dependencies (env_logger,
mmap-rs, cpu-time, rand 0.10) come with it.
