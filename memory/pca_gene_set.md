# PCA over a gene set (task 6.14)

**Works:** scanpy's `mask_var` as a step argument, not a filter: `CrossCellOp::Pca`
holds a resolved, metadata-only boolean `GeneExpr` (as a QC metric's gene set), the
executor evaluates it with `GeneSet::evaluate` on the gene table of the genes present
at the node (so a column written by `highly_variable_genes` or `with_genes` earlier in
the plan works), and `PcaAccumulator` skips the other genes as it reads each row. The
sums then run in the order a `filter_genes` before the node gives them: the embedding,
variances and loadings are the same bits (asserted in Rust on a sketched store and in
Python on pbmc3k). Loadings get a zero row per gene outside the set (scanpy's
`varm['PCs']`).

**Watch:** `CrossCellOp` derives serde and a test round-trips it, but `GeneExpr` has no
serde: the field goes through `ExprJson` (`with = "gene_set_json"`,
`skip_serializing_if` so a plain pca's JSON is unchanged). A new optional `Step` key in
the plan JSON needs no version bump (absent = `None`), but its schema entry must allow
`null` (serde writes it).

**Watch:** a copied `target` still rebuilt HDF5 and Polars in release for the first
`uv run` (about 6 minutes with another worktree building); delete both
`hdf5-metno-src-*/out` folders before, not after, the first failure.
