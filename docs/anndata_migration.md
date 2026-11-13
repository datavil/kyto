# From AnnData to kyto: reading features

How the AnnData (and `scanpy.get`) ways of reading cells, genes and values map to
kyto's feature access (task 7.4). Task 10.5 folds this page into the docs site's
migration table. Every row of the table below is run by
`tests/test_feature_access.py` on the pbmc3k fixture, both sides, and the results are
compared value for value (`test_migration_row`): a row that stops matching fails the
local gate.

kyto has no `obs`, `var`, `.X` or `__getitem__` (R4). Values come out through
terminals that run the plan and return Polars objects or ints (R1, R3), and every name
is resolved against the plan when the call is made (R6): a typo fails before anything
is read, with close matches (`KY-E210`).

## Setup

```python
import kyto as ky
import scanpy as sc
from kyto import cell_column, gene_column, gene_expression, qc

adata = sc.read_h5ad("pbmc3k.h5ad")
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

atlas = (
    ky.scan("pbmc3k.kyto")
    .with_default_layer("counts")
    .with_cells(total_counts=qc.total_counts(), n_genes_by_counts=qc.genes_detected())
)
```

`pbmc3k.kyto` is `ky.convert("pbmc3k.h5ad", "pbmc3k.kyto", layer="counts")`. Its gene
IDs are Ensembl IDs, as `adata.var_names`, and the symbols are the gene table's `symbol`
column.

## Table

| AnnData | kyto | kyto returns |
| --- | --- | --- |
| `adata.n_obs` | `atlas.count_cells()` | `int` |
| `adata.n_vars` | `atlas.count_genes()` | `int` |
| `adata.obs_names` | `atlas.cell_ids()` | `pl.Series` (`cell_id`) |
| `adata.var_names` | `atlas.gene_ids()` | `pl.Series` (`gene_id`) |
| `adata.obs["n_genes_by_counts"]` | `atlas.pull("n_genes_by_counts")` | `pl.Series` |
| `adata.obs[["total_counts", "n_genes_by_counts"]]` | `atlas.select("total_counts", "n_genes_by_counts")` | `pl.DataFrame`, `cell_id` first |
| `adata.var["symbol"]` | `atlas.pull(gene_column("symbol"))` | `pl.Series` |
| `adata[:, "ENSG00000105374"].X.toarray().ravel()` | `atlas.pull("ENSG00000105374")` | `pl.Series` |
| `sc.get.obs_df(adata, keys=["ENSG00000105374"])` | `atlas.pull_genes(["ENSG00000105374"])` | `pl.DataFrame`, `cell_id` first |
| `sc.get.obs_df(adata, keys=["CD3E", "MS4A1"], gene_symbols="symbol")` | `atlas.pull_genes(["CD3E", "MS4A1"], by="symbol")` | `pl.DataFrame`, `cell_id` first |
| `sc.get.obs_df(adata, keys=["CD3E"], gene_symbols="symbol")["CD3E"]` | `atlas.pull(gene_expression("CD3E", by="symbol"))` | `pl.Series` |
| `adata[adata.obs["total_counts"] > 5000].obs_names` | `atlas.filter(cell_column("total_counts") > 5000).cell_ids()` | `pl.Series` |
| `adata[:, adata.var["symbol"].str.startswith("MT-")].n_vars` | `atlas.filter(gene_column("symbol").str.starts_with("MT-")).count_genes()` | `int` |

## Notes

- **One return type.** `pull` always gives a `pl.Series`, `select` and `pull_genes`
  always a `pl.DataFrame` with `cell_id` first, even for one feature (R3). Rows are in
  `cell_ids()` order, as `adata.obs_names`.
- **Symbols.** A gene is keyed by its ID. Read one by symbol with `pull_genes(...,
  by="symbol")` or `gene_expression(symbol, by="symbol")`: the symbol is resolved to
  its gene ID when the call is made, and a symbol carried by several genes raises
  `KY-E204` with their IDs (scanpy's `obs_df` raises too). A bare short name such as
  `pull("CD3E")` matches cell columns and gene IDs only (docs/feature_names.md); its
  error suggests `CD3E (gene symbol)`.
- **Dtypes.** Values keep the layer's dtype (`Float32` here; a store converted from 10x
  keeps `UInt32` counts), and QC columns are `Float64` (`total_counts`) and `UInt32`
  (`n_genes_by_counts`); scanpy writes `float32` and `int32` for these.
- **No equivalent.** `adata.uns` is not stored (format.md section 12, a v0 gap).
  `atlas.metadata()` returns the store's `provenance.json` instead (how the store was
  made). `atlas.features()` lists every full feature name the plan resolves, with no
  AnnData counterpart.
