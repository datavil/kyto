# scanpy neighbours and kyto's HNSW (tasks 6.6, 6.17)

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

**`hnsw_rs` 0.3.5 broke at 50k (6.17), so the HNSW is kyto's own now
(`neighbors_hnsw.rs`).** On 10.8's 50k atlas `hnsw_rs` gave 986 cells recall 0
(neighbours in another cluster), Leiden ARI 0.983. Two library faults: its query moves
at most ONE link per upper layer (no greedy loop), so queries reach layer 0 in the wrong
cluster; and it trims an overfull reverse-link list by dropping the farthest link
(not the heuristic), which removes the links between clusters, so rows inserted from
the wrong region keep 2-7 links into another cluster and none back. Fixing the descent
alone left 87 lost; the paper's heuristic on reverse links too, still a few. What fixed
it: every query is a row of the index, so start the layer-0 search at the row's own
node (no descent), on an index that applies the heuristic to reverse links. Even a
correct top-down descent loses ~16 of 50k on this data: keep self-entry.
**Reproducing:** the failure needs scale (maxlevel 3, ~30k+ cells): a Gaussian
mixture of 8 clusters, sd 0.97 in 50 components, centres 15.4 apart reproduces it at
30k-50k (1.1-1.3% lost) but hardly at 20k (0.045%) and not at 10k. The suite test runs
3k (a regression guard); the ignored 50k test in release is the real check.
**R11:** insertion is sequential (levels and order from kyto's seed, `f64` kernel
distances ordered by (distance, row)), queries are read-only and parallel. The `uv
run` debug test of 8k cells takes ~2.5 min (exact search and insertion unoptimised).
