# umap connectivities and numba's types (task 6.7)

**Breaks:** umap-learn 0.5.12's `smooth_knn_dist` declares `psum`, `lo`, `mid`, `hi`
as `float32` in `numba.njit(locals=...)`, but the loop is a `prange` body under
`parallel=True`, and numba types them `f64` there. A `float32` replica of the
bisection missed 4 of the first 300 pbmc3k sigmas, and so 1-2% of the weights by a
few ulps (2.1e-6). Only `f64` `lo`/`mid`/`hi`/`psum` (with the `float32` difference
`distances[i, j] - rho[i]`) reproduces every sigma. The same goes for `np.mean` (the sigma floors): numba's source rounds it back to
`float32`, but under `parallel=True` it becomes an `f64` reduction that is not
rounded (bug-hunter, 6.7: 1 ulp of sigma at rows of duplicates otherwise).
`compute_membership_strengths`
(no `prange`) does keep its `float32` `val`, and scipy's fuzzy union runs in
`float32` (`(w + w^T) - w o w^T`).

**Works:** feed scanpy's own `distances` back as umap's input (the cell in column 0 at
distance 0, then the row sorted by (distance, column)) and the stored
`connectivities` come back bit for bit on both fixtures, even at the synthetic
duplicates where scanpy keeps a self loop. So `NeighborGraph::from_csr` plus the
reference `distances` is the exact Done-when check. kyto's own search moves weights by
up to 1.1e-5 (pbmc3k), only through the distances' last bits.

**Check numba's real types** by replicating in Python and comparing to the numba
function on real rows, not by reading `locals`.
