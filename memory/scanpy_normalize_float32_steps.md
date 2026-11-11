# scanpy's `normalize_total` bit for bit, and aliased references (task 6.2)

**Works:** scanpy 1.12 (`_normalize_csr`, `_normalize_total_helper`) on `float32` CSR:
numba sums each row in `float64` and stores the total in a `float32` array; the divisor
is `float32(total) / float32(target_sum)` (numpy 2 casts the Python float to the
array's `float32`); a divisor of 0 becomes 1 (`+ (counts == 0)`); each value is divided
by it in `float32`. The median target is `np.median` of the positive `float32`
totals, so an even count takes `(a + b) / 2` in `float32`. kyto follows these
steps (`normalize_divisor`, `Kernels::divide_rows`, `cross_cell.rs` `median`) and is
bit-identical on both fixtures. Multiplying by `target / total` in `float64` (5.2's
stand-in) differed on a quarter of the values by 1 ulp: divide, never multiply by the
reciprocal. Integer counts are cast to `float32` by scanpy first.

**Breaks:** `normalized = adata.X; sc.pp.log1p(adata)` (and `csr_matrix(adata.X)`,
which does not copy a CSR) aliases the matrix that `log1p` then changes in place: the
1.5 reference `normalize_log1p` stores `log1p` values under `*/normalized/*` (plan 6.2
Open). Copy (`adata.X.copy()`, `csr_matrix(x, copy=True)`) or use `inplace=False`
before a later in-place step. The 6.2 parity test computes `normalized` with scanpy
live until the reference is regenerated.

**numpy's `float32` `log1p`** is 1 ulp off kyto's correctly rounded one on 620 of 2.29M
pbmc3k values (macOS arm64, numpy 2.5.4): compare `log1p` within 1e-6, not by bits
(deviations.md "`log1p` rounding").
