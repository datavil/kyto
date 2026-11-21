# From AnnData and scanpy to kyto

The common AnnData and scanpy calls and their kyto counterparts. Reading single
values, columns and genes (`adata.obs[...]`, `sc.get.obs_df`) has its own page,
[Reading features](anndata_migration.md), where every row is run on pbmc3k on both
sides and compared value for value.

Three differences explain most rows:

- **Nothing runs until a terminal** (R1). The scanpy calls below change `adata` at
  once; the kyto ones add a step to a plan. `collect`, `sink_*`, `pull`, `select` and
  `to_*` run it.
- **No in-place changes** (R2, R4). Every method returns a new plan, so there is no
  `adata.copy()`. A step that makes a matrix names it with `into=` and reads one with
  `layer=` (or the plan's default layer, `with_default_layer`), instead of overwriting
  `.X`.
- **No hidden settings** (R7). Seeds, threads, devices and memory limits are arguments
  of the call that uses them: `seed=` has no default where the result is random, and
  there is no `sc.settings`.

`tests/test_docs_site.py` checks every kyto cell of the tables: each name exists in
the package, on the class the step before it returns (as the stubs type it), and every
call binds to the signature of the function or method it names. It also builds the
plan at the end of this page.

## Reading and writing

| AnnData / scanpy | kyto | Notes |
| --- | --- | --- |
| `sc.read_h5ad("data.h5ad")` | `ky.convert("data.h5ad", "data.kyto")`, then `ky.scan("data.kyto")` | Import once into a kyto store, then scan it lazily; `scan` reads only the schema. |
| `sc.read_10x_h5("sample.h5")`, `sc.read_10x_mtx("folder")` | `ky.convert("sample.h5", "sample.kyto")` | `convert` reads `.h5ad`, 10x `.h5` and Matrix Market folders. |
| `ad.concat(adatas, label="sample")` | `ky.concat(["a.h5ad", "b.h5ad"], "all.kyto", label="sample")` | Writes one store with a `sample` cell column; `on_duplicate_ids="suffix"` makes repeated barcodes unique. |
| `adata.write_h5ad("out.h5ad")` | `atlas.sink_h5ad("out.h5ad")` | Runs the plan and writes its output. |
| `adata.write_zarr("out.zarr")` | `atlas.sink_zarr("out.kyto")` | Writes a kyto store ([store format](format.md)), not AnnData's Zarr layout. |
| `adata` (for plotting or other tools) | `atlas.to_anndata(var_names="symbol")` | Runs the plan and returns a new `AnnData` ([plotting](plotting.md)). |
| `adata.copy()` | `atlas` | Plans are immutable (R2); there is nothing to copy. |

## Quality control and filtering

| AnnData / scanpy | kyto | Notes |
| --- | --- | --- |
| `sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"])` | `atlas.with_cells(total_counts=qc.total_counts(), n_genes_by_counts=qc.genes_detected(), pct_counts_mt=qc.percent_counts_in(gene_column("symbol").str.starts_with("MT-")))` | Each metric is an expression you name; the gene set is a gene predicate, not a `var` column. |
| `adata.var["n_cells_by_counts"]` from `calculate_qc_metrics` | `atlas.with_genes(n_cells_by_counts=qc.cells_detected())` | The gene metrics: `qc.cells_detected`, `qc.gene_total_counts`, `qc.gene_mean_counts`, `qc.percent_dropout`. |
| `sc.pp.filter_cells(adata, min_genes=200)` | `atlas.filter(qc.genes_detected() >= 200)` | A cell predicate keeps cells. |
| `sc.pp.filter_genes(adata, min_cells=3)` | `atlas.filter(qc.cells_detected() >= 3)` | A gene predicate keeps genes. |
| `adata[adata.obs["pct_counts_mt"] < 5]` | `atlas.filter(cell_column("pct_counts_mt") < 5)` | No `__getitem__` (R4). |
| `adata[:, ["ENSG01", "ENSG02"]]` | `atlas.select_genes(["ENSG01", "ENSG02"])` | By gene ID; `take_cells` does the same for cell IDs. |
| `sc.pp.scrublet(adata, batch_key="sample")` | `atlas.scrublet(batch_key="sample", seed=0)` | Adds the doublet score and call as cell columns. |

## Preprocessing

| AnnData / scanpy | kyto | Notes |
| --- | --- | --- |
| `adata.layers["counts"] = adata.X.copy()` | `atlas.with_default_layer("counts")` | Layers are never overwritten; the default layer is the one steps read when given no `layer=`. |
| `sc.pp.normalize_total(adata, target_sum=1e4)` | `atlas.normalize_total(1e4, into="normalized")` | Writes a new layer. |
| `sc.pp.log1p(adata)` | `atlas.log1p(layer="normalized", into="log1p")` | |
| `sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample")` | `atlas.highly_variable_genes(n_top_genes=2000, batch_key="sample")` | Adds `highly_variable` and the flavour's statistics as gene columns. |
| `adata[:, adata.var["highly_variable"]]` | `atlas.filter(gene_column("highly_variable"))` | |
| `sc.pp.scale(adata, max_value=10)` | `atlas.scale(into="scaled", max_value=10)` | |

## Embeddings, graphs and clustering

| AnnData / scanpy | kyto | Notes |
| --- | --- | --- |
| `sc.pp.pca(adata, n_comps=50)` | `atlas.pca(into="pca", n_components=50, seed=0)` | scanpy's default `mask_var="highly_variable"` is `genes=gene_column("highly_variable")`, never implicit. |
| `sc.pp.neighbors(adata, n_neighbors=15)` | `atlas.neighbors("pca", n_neighbors=15)` | Names the embedding to read; writes the `distances` and `connectivities` graphs. |
| `sc.tl.umap(adata)` | `atlas.umap(into="umap", seed=0)` | |
| `sc.tl.leiden(adata, resolution=1.0)` | `atlas.leiden(resolution=1.0, seed=0, into="leiden")` | |
| `sc.tl.louvain(adata)` | `atlas.louvain(seed=0, into="louvain")` | |
| `sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")` | `atlas.rank_genes_groups("leiden", method="wilcoxon")` | Returns a `pl.LazyFrame` (one row per group and gene) instead of filling `adata.uns`; `sc.get.rank_genes_groups_df` is its `collect()`. |

## Reading results

| AnnData / scanpy | kyto | Notes |
| --- | --- | --- |
| `adata.obs` | `atlas.cells.to_polars()` | A `pl.DataFrame`; `adata.var` is `atlas.genes.to_polars()`. |
| `adata.layers["counts"]` | `atlas.layer("counts").to_scipy()` | A new SciPy CSR matrix. |
| `adata.X.toarray()` | `atlas.layer("counts").to_dense(max_bytes=2**30)` | Dense only when asked, within a size you state (R8). |
| `adata.obsm["X_pca"]` | `atlas.embedding("pca").to_numpy()` | |
| `adata.obsp["connectivities"]` | `atlas.graph("connectivities").to_scipy()` | |
| `adata.n_obs`, `adata.n_vars` | `atlas.count_cells()`, `atlas.count_genes()` | More in [Reading features](anndata_migration.md). |

## Running

| AnnData / scanpy | kyto | Notes |
| --- | --- | --- |
| `sc.settings.n_jobs = 8` | `atlas.collect(threads=8)` | Every terminal takes `threads=`, `memory_limit=` and `progress=`; results do not depend on the thread count (R11). |
| no counterpart | `atlas.collect(memory_limit="4GB")` | Runs the plan within a memory limit, or fails before it starts (`KY-E4xx`). |
| `rsc.get.anndata_to_GPU(adata)` (rapids-singlecell) | `atlas.on(device="auto")` | Places the following steps on a device; `explain()` shows where each runs. |
| no counterpart | `atlas.explain()`, `atlas.estimate()` | Show the plan, and its cost and peak memory, without reading the matrix. |

## A whole workflow

scanpy's preprocessing and clustering as one kyto plan. Nothing is read until
`collect`; [the clustering tutorial](tutorials/clustering.md) runs the full version on
two samples and compares it with scanpy.

```python
import kyto as ky
from kyto import cell_column, gene_column, qc

mito = gene_column("symbol").str.starts_with("MT-")
plan = (
    ky.scan("pbmc3k.kyto")
    .with_default_layer("counts")
    .with_cells(
        n_genes_by_counts=qc.genes_detected(), pct_counts_mt=qc.percent_counts_in(mito)
    )
    .filter(cell_column("n_genes_by_counts") >= 1)
    .filter(qc.cells_detected() >= 1)
    .normalize_total(1e4, into="normalized")
    .log1p(layer="normalized", into="log1p")
    .highly_variable_genes(layer="log1p", n_top_genes=2)
    .pca(
        layer="log1p",
        into="pca",
        n_components=1,
        seed=0,
        genes=gene_column("highly_variable"),
    )
    .neighbors("pca", n_neighbors=2)
    .umap(into="umap", seed=0)
    .leiden(into="leiden", seed=0)
)
print(plan.explain())
```

The thresholds and sizes here fit the tiny store the test builds the plan on; on real
data use scanpy's (`min_genes=200`, `min_cells=3`, `n_top_genes=2000`,
`n_components=50`, `n_neighbors=15`).
