# Scrublet: what scanpy 1.12.5 really runs (task 6.12)

**Works:** given scanpy's simulated doublets (`uns["scrublet"]["doublet_parents"]`),
kyto's scores equal scanpy's exactly (worst difference 0) on pbmc3k and on every
synthetic batch. The route: per batch, `filter_genes(min_cells=3)`,
`filter_cells(min_genes=3)`, `normalize_total` to the median in `float32` steps
([scanpy_normalize_float32_steps.md](scanpy_normalize_float32_steps.md)), `log1p`,
seurat HVG, subset the raw counts to the HVGs, sum parent rows, `normalize_total(1e6)`
on cells and doublets, z-score with fast_array_utils' `mean_var` (`f64` sums,
`var = sumsq/n - mean^2` times `n/(n-1)`, `z = (x - mean) * (1/std)`), sklearn PCA
fit on the cells (the Gram matrix's eigenvectors, `svd_flip`, give arpack's axes), then
brute-force kNN and Scrublet's Bayesian score. `k = round(0.5 * sqrt(n_obs))` with
Python's half-to-even rounding; `k_adj = round(k * (1 + n_sim / n_obs))`, lowered to
`1 + n_total // 2` when above the rows.

**Breaks:**
- `_has_self_column`: scanpy asks the kNN for `k_adj` neighbours of every row and drops
  column 0 only if some row's first neighbour is itself at distance 0. Any two
  identical embedded rows (common: two simulated doublets with the same parents, or
  identical cells) make the check see a duplicate instead of self, so every row keeps
  `k_adj - 1` others and the doublet's own row counts as a doublet neighbour: +1 for
  every simulated row. Without duplicates (synthetic batch_0), the column is dropped,
  no self count. Check both on every batch (`dups.py` style) before trusting a match.
- An `f32` embedding for the search flipped neighbours at near-ties (1e-8 relative) on
  synthetic batch_2: sims 6 and 282 off by one count. Search in `f64`.
- skimage's `threshold_minimum` keeps a plateau's *last* index as its maximum; flat
  (uniform) modes give a threshold at the first mode's edge, not the valley
  (0.0923 for two uniform blocks at 0.1 and 0.7). Tests: take the expected value from
  skimage, never from intuition.
- The seeded-calls Done-when (95%) does not hold on batches of ~165 cells for any
  seed: scanpy agrees with *itself* across seeds 1-7 on 86-94% of the synthetic
  by-batch cells (the automatic threshold moves with the draws). Measure scanpy's own
  seed spread before chasing an agreement number.
- The fixture's `.npz` stores NaN scores for filtered cells; scanpy's call there is
  `False`, kyto's null.
