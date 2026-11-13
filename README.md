# kyto

Lazy, streaming single-cell analysis, written in Rust, used from Python. Results match
scanpy on the reference fixtures; the matrix is read in blocks, within a stated memory
limit.

Chained methods build a plan and read nothing. Only the terminals (`collect`,
`sink_*`, `pull`, `select`, `to_*`) run it. Every method returns a new plan; layers,
seeds, devices and memory limits are arguments, never global settings. The rules are
in [AGENTS.md](AGENTS.md); the roadmap is [Implementation Plan.md](Implementation%20Plan.md).

## Pipeline

The first steps of scanpy's PBMC 3k tutorial. `kyto.convert` turns an `.h5ad`, 10x
`.h5` or Matrix Market file into a kyto store first:
`ky.convert("pbmc3k.h5ad", "pbmc3k.kyto", layer="counts")`.

```python
import kyto as ky
from kyto import cell_column, gene_column, qc

atlas = ky.scan("pbmc3k.kyto").with_default_layer("counts")
mito = gene_column("symbol").str.starts_with("MT-")
hvg = (
    atlas.with_cells(
        n_genes_by_counts=qc.genes_detected(),
        total_counts=qc.total_counts(),
        pct_counts_mt=qc.percent_counts_in(mito),
    )
    .filter(cell_column("n_genes_by_counts") >= 200)
    .filter(cell_column("n_genes_by_counts") < 2500)
    .filter(cell_column("pct_counts_mt") < 5)
    .normalize_total(1e4, into="normalized")
    .log1p(layer="normalized", into="log1p")
    .highly_variable_genes(layer="log1p", n_top_genes=2000)
    .filter(gene_column("highly_variable"))
)
print(hvg.explain())  # the logical and optimized plans; nothing is read yet
print(hvg.estimate())  # bytes read, peak memory and passes, before running
hvg.sink_zarr("pbmc3k_hvg.kyto", memory_limit="1GB")  # runs the plan

result = ky.scan("pbmc3k_hvg.kyto").with_default_layer("log1p")
markers = (
    result.with_cells(high_counts=cell_column("total_counts") > 2500)
    .rank_genes_groups("high_counts", method="wilcoxon", n_genes=5)
    .collect()  # a polars.DataFrame
)
print(markers)

# PCA and neighbours build lazily too (sinks write their outputs; reading them from Python arrives with 7.3).
graph = result.pca(into="pca", n_components=50, seed=0).neighbors("pca")
print(graph.estimate())
```
