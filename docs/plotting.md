# Plotting kyto results with cellestial

kyto draws no plots (decided by the user, Nov 13, 2026). Results go to
[cellestial](https://pypi.org/project/cellestial/), which reads an `anndata.AnnData`.
`Atlas.to_anndata()` runs a plan and returns its output as one, under scanpy's names:

- cell embeddings in `obsm` as `X_<name>` (`pca` becomes `X_pca`, `umap` becomes
  `X_umap`), where scanpy and cellestial look for them;
- `var_names="symbol"` names the genes by symbol, as scanpy's `read_10x_h5` does
  (repeated symbols made unique, gene IDs kept in the `gene_id` column), so genes are
  plotted by symbol;
- `Enum` columns (`leiden` labels, `concat`'s `sample`) are categoricals in their
  category order (`"0"`, `"1"`, ..., `"10"` for clusters);
- `uns` is empty: kyto stores have none (format v0), so plots that read it
  (`cl.elbow`, `dotplot(markers=True)`) need it filled first.

The `Atlas` itself keeps no `obs`/`var` (R4): the `AnnData` is a copy the caller
owns. Install the plotting extra with `pip install "kyto[plot]"` (anndata and
cellestial).

The examples below are one per plot of scanpy's
[clustering tutorial](https://scanpy.scverse.org/en/stable/tutorials/basics/clustering.html):
QC violins and scatter, UMAP coloured by sample, cluster and gene, and a marker
dotplot. `tests/test_plotting.py` runs them on two halves of the pbmc3k fixture and
builds every plot. Each plot is a ggrust plot: `.show()` draws it in a notebook,
`.to_svg(path)` or `.to_png(path)` saves it.

## Setup: two samples and their QC metrics

```python
>>> import cellestial as cl
>>> import kyto as ky
>>> from ggrust import aes
>>> from kyto import cell_column, gene_column, qc
>>> ky.concat(["s1d1.kyto", "s1d3.kyto"], "samples.kyto", keys=["s1d1", "s1d3"])
>>> symbol = gene_column("symbol")
>>> samples = (
...     ky.scan("samples.kyto")
...     .with_default_layer("counts")
...     .with_cells(
...         n_genes_by_counts=qc.genes_detected(),
...         total_counts=qc.total_counts(),
...         pct_counts_mt=qc.percent_counts_in(symbol.str.starts_with("MT-")),
...         pct_counts_ribo=qc.percent_counts_in(
...             symbol.str.starts_with("RPS") | symbol.str.starts_with("RPL")
...         ),
...     )
... )
>>> qc_data = samples.to_anndata()
>>> qc_data.obs["sample"].cat.categories.tolist()
['s1d1', 's1d3']

```

## QC violins

scanpy: `sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
jitter=0.4, multi_panel=True)`.

```python
>>> qc_violins = cl.violins(
...     qc_data, ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
... )

```

## QC scatter

scanpy: `sc.pl.scatter(adata, "total_counts", "n_genes_by_counts",
color="pct_counts_mt")`.

```python
>>> qc_scatter = cl.xyplot(
...     qc_data,
...     "total_counts",
...     "n_genes_by_counts",
...     mapping=aes(color="pct_counts_mt"),
... )

```

## Clustering

Filter, normalize, embed and cluster; then hand the result over once, with genes
named by symbol.

```python
>>> result = (
...     samples.filter(cell_column("n_genes_by_counts") >= 100)
...     .normalize_total(into="normalized")
...     .log1p(layer="normalized", into="log1p")
...     .with_default_layer("log1p")
...     .pca(into="pca", n_components=30, seed=0)
...     .neighbors("pca")
...     .umap(into="umap", seed=0)
...     .leiden(into="leiden", resolution=0.5, seed=0)
... )
>>> adata = result.to_anndata(var_names="symbol")
>>> sorted(adata.obsm)
['X_pca', 'X_umap']

```

## UMAP coloured by sample

scanpy: `sc.pl.umap(adata, color="sample")`.

```python
>>> umap_sample = cl.umap(adata, key="sample")

```

## UMAP coloured by cluster

scanpy: `sc.pl.umap(adata, color="leiden")`.

```python
>>> umap_cluster = cl.umap(adata, key="leiden")

```

## UMAP coloured by gene

scanpy: `sc.pl.umap(adata, color="CD14")`. The gene is named by symbol because of
`var_names="symbol"`; its values are `X`, the plan's default layer (`log1p`).

```python
>>> umap_gene = cl.umap(adata, key="CD14")

```

## Marker dotplot

scanpy: `sc.pl.dotplot(adata, marker_genes, groupby="leiden")`.

```python
>>> marker_genes = {
...     "CD14+ Mono": ["FCN1", "CD14"],
...     "CD16+ Mono": ["FCGR3A"],
...     "B cells": ["MS4A1", "CD79A"],
...     "NK": ["GNLY", "NKG7"],
...     "T cells": ["CD3E", "IL7R"],
... }
>>> markers_dotplot = cl.dotplot(adata, marker_genes, group_by="leiden")

```
