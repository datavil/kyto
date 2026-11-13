# UMAP layout: what umap-learn really runs under scanpy (task 6.9)

**Works:** scanpy's `tl.umap` calls `simplicial_set_embedding` with `parallel=False`, so
umap-learn's SGD is sequential: kyto's sequential SGD is the same algorithm, not a
compromise for R11. Its layout is checked by trustworthiness (scikit-learn's, against
`X_pca`, at 5 and 15 neighbours) inside umap-learn's own range over 5 seeds; layouts
are never compared point by point (umap's seeds differ that much from each other).
The reference stores the range (`trustworthiness/<k>` in `umap.npz`); the Rust test
recomputes scikit-learn's score on scanpy's layout to check its own helper (1e-6).

**Breaks:**
- `umap.utils.tau_rand_int` is `numba.njit("i4(i8[:])")`: the `int64` xor is truncated
  to `int32`, so draws are negative half the time; `% n_vertices` is Python's
  (non-negative). Replicating it on `i64` gave the wrong range.
- Thresholds and epochs per sample are `float32` (NumPy 2 keeps `float32` for
  `float32_array / python_float` and `int * float32_array`).
- A restarted block Krylov eigensolver on a long path graph (eigengap ~ (pi/n)^2) did
  not converge in 500 restarts: unit tests use circulant expanders; kNN graphs
  converge fast (pbmc3k layout 2.6 s in release, spectral start included).
- `make_reference` regenerates every reference; on this machine the old ones came
  back with identical digests (check `git diff` of the manifest for `-` lines).
