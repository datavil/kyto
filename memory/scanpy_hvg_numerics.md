# scanpy's highly variable genes, number for number (task 6.3)

**Works:** follow scanpy 1.12.5's steps exactly, then every flavour's flags match on
the 1.5 fixtures (`crates/kyto-exec/tests/hvg_parity.rs`):
- `mean_var` (fast_array_utils) squares `float32` values in `float32` and sums in
  `float64`; variance `sq/n - mean^2`, times `n/(n-1)`. Not Welford: kyto sums too.
- seurat: `expm1` of the `float32` log data (kyto rounds the `f64` result once, numpy
  is 1 ulp off on 0.5%: deviations.md), `pd.cut` with numpy `linspace` edges and the
  0.1% widening, pandas' Kahan group mean and Welford std; a one-gene bin gets norm 1.
- cell_ranger: `np.percentile` (numpy `_lerp`) edges at 10..100 plus +-inf, median and
  statsmodels `mad` (/0.6744897501960817).
- Cutoff and `n_top_genes` flags go through `nan_to_num` (inf -> MAX) unbatched, but
  batched cutoffs `fillna(0)` and skip it; batched seurat/cell_ranger filter genes
  without a positive value per batch (zeros, not NaN).
- seurat_v3: skmisc loess (netlib `dloess`, interpolate surface) ported in `loess.rs`,
  to about 1e-14.

**Trap: the interpolate loess is discontinuous in tied x.** Its kd-tree splits at a
data point's x; two genes with the same mean in exact arithmetic but 1 ulp apart move
the vertices, and the fit at many genes changes by percent. Batched `seurat_v3` means
come from scanpy's `aggregate` (sequential Welford over stored values, then a zero
merge), and its 1-ulp differences from sums/n changed `variances_norm` by up to 4%.
Fix: the same Welford in the same order, as one sequential leaf of the statistics
pass (`HvgAccumulator::sequential`, `reduction.rs` puts every block in leaf 0). Any
change to how means are computed for loess input must be checked bit for bit.

**Ties:** numpy's `argsort` is unstable and platform dependent for equal normalised
variances (identical genes); kyto ranks in gene order. The parity test resolves them
from scanpy's data (overall ties, and the widest per-batch tie a gene sits in).
