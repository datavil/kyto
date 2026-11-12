# Streamed PCA: block Krylov on the genes' side (task 6.5)

- **Breaks:** the plan's cell-side Halko (`A Omega`, then `A^T Q` / `A Z` per power
  iteration, `Q^T A` last: 16 passes, 10 oversamples) was 4.6e-2 off scanpy's variances
  on pbmc3k's 50 components (2,000 HVGs); 1e-3 needed ~100 oversamples and 22+ passes,
  and the cell-side blocks hold `n_cells x l` twice.
- **Works:** iterate on `C = Xc^T Xc` from the genes' side. A pass computes
  `X^T (X Q)` one sparse row at a time (`w = x Q`, then `x^T w`), so nothing per cell is
  kept; each pass's product grows a Krylov basis (project off twice, thin SVD, drop
  directions below 1e-10 of the block); Rayleigh-Ritz on the whole basis at the end. 9
  products (11 passes): variances to 1.3e-6, the `float32` reference's own precision.
  Exact when the basis spans every gene (synthetic's 100 genes).
- Centring stays implicit but per row: `w = x Q - mu^T Q`, then `C Q = X^T W - mu (sum w)`.
  `X^T X Q - n mu mu^T Q` cancels catastrophically (1.7e-2 off at means of 1e7,
  bug-hunter). Identical cells: total variance 0, report variance 0 and ratio NaN. Sketch `X^T Omega - mu (1^T
  Omega)` in pass 0 needs no means yet (sum `Omega`'s rows alongside).
- `Omega` rows: SplitMix64 of (seed, store table position), uniform `[-1, 1)`: no
  libm, independent of threads, blocks and filters (R11).
- scanpy 1.12.5 + scikit-learn >= 1.5 on sparse `arpack` goes through `sklearn.PCA`
  (not `_pca_compat_sparse`): `svd_flip(u_based_decision=False)` (largest absolute
  loading positive), `explained_variance_ = S^2 / (n - 1)`, ratio over the genes'
  `ddof=1` variances. Compute the total variance from centred squares
  (`sum_stored (x - mu)^2 + (n - stored) mu^2`), not `sum x^2 - n mu^2`.
- Testing: the seed changes nothing when the basis spans every gene; check it only on
  a sketched fixture. Compare trailing components by subspace angle only (flat spectrum).
