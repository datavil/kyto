# %% [markdown]
# # Preprocessing and clustering, in kyto
#
# scanpy's basic clustering tutorial
# (https://scanpy.scverse.org/en/stable/tutorials/basics/clustering.html), step for
# step, in kyto (task 10.7). Every plot goes through cellestial (`pip install
# "kyto[plot]"`). Where kyto cannot do a step yet, the cell says so and what stands in;
# [clustering.md](clustering.md) lists those gaps, the comparison with scanpy's own run
# of the tutorial, and the timings.
#
# This file is a notebook in the percent format (`# %%` cells, for VS Code or Jupyter
# with jupytext) and a script:
#
# ```bash
# uv run python docs/tutorials/clustering.py s1d1.h5 s1d3.h5 <output folder>
# ```
#
# The output folder (replaced) receives the combined store, `kyto.h5ad` (the clustered
# cells), `ranked.parquet` (the marker ranking), `run.json` (timings, peak memory, the genes kept) and one PNG per plot.
# `tests/test_clustering_tutorial.py` runs it in the gate. The only change from the
# tutorial's loading is where the two files come from: the test fetches them into
# kyto's verified fixture cache instead of `pooch`'s.

# %%
import json
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cellestial as cl
import kyto as ky
import numpy as np
from ggrust import aes
from kyto import cell_column, gene_column, qc

# anndata, cellestial and ggrust ship without complete type information.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false

if len(sys.argv) != 4:
    raise SystemExit("usage: clustering.py <s1d1.h5> <s1d3.h5> <output folder>")
samples = {"s1d1": Path(sys.argv[1]), "s1d3": Path(sys.argv[2])}
output = Path(sys.argv[3])
# The folder is replaced: refuse one that is not empty and not an earlier run's.
if output.exists() and any(output.iterdir()) and not (output / "samples.kyto").exists():
    raise SystemExit(f"{output} is not empty and not an earlier run's output folder")
shutil.rmtree(output, ignore_errors=True)
output.mkdir(parents=True)

seconds: dict[str, float] = {}
peak_rss_mb: dict[str, float] = {}
started = time.perf_counter()
lap_start = started


def _peak_rss_mb() -> float:
    """The process's peak resident memory so far (``ru_maxrss``), in MiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    return peak / (2**20 if sys.platform == "darwin" else 2**10)


def lap(stage: str) -> None:
    """Record the wall-clock seconds since the last stage and the peak memory."""
    global lap_start
    now = time.perf_counter()
    seconds[stage] = now - lap_start
    peak_rss_mb[stage] = _peak_rss_mb()
    lap_start = now


# %% [markdown]
# ## Reading and combining the samples
#
# `ky.concat` writes both samples into one store with a `sample` column (scanpy:
# `ad.concat(adatas, label="sample")`). Repeated barcodes get `-1` suffixes, as
# `obs_names_make_unique()` gives. kyto imports every 10x feature and keys genes by ID
# (no `var_names_make_unique` needed); the tutorial's `read_10x_h5` keeps only "Gene
# Expression" features, so the plan filters on `feature_type`.

# %%
store = output / "samples.kyto"
ky.concat(list(samples.values()), store, keys=list(samples), on_duplicate_ids="suffix")
atlas = (
    ky.scan(store)
    .filter(gene_column("feature_type") == "Gene Expression")
    .with_default_layer("counts")
)
lap("concat")

# %% [markdown]
# ## Quality control
#
# QC metrics are expressions computed on read (scanpy: `calculate_qc_metrics(qc_vars=
# ["mt", "ribo", "hb"])`). The haemoglobin pattern `^HB[^(P)]` is "starts with HB, not
# with HBP": kyto's string expressions have no regular expressions, and no symbol is
# `HB` alone. kyto has no `log1p_` variants and no per-gene QC columns yet (proposed
# task 6.13).

# %%
symbol = gene_column("symbol")
mito = symbol.str.starts_with("MT-")
ribo = symbol.str.starts_with("RPS") | symbol.str.starts_with("RPL")
hb = symbol.str.starts_with("HB") & ~symbol.str.starts_with("HBP")
samples_qc = atlas.with_cells(
    n_genes_by_counts=qc.genes_detected(),
    total_counts=qc.total_counts(),
    total_counts_mt=qc.total_counts_in(mito),
    pct_counts_mt=qc.percent_counts_in(mito),
    total_counts_ribo=qc.total_counts_in(ribo),
    pct_counts_ribo=qc.percent_counts_in(ribo),
    total_counts_hb=qc.total_counts_in(hb),
    pct_counts_hb=qc.percent_counts_in(hb),
)
qc_data = samples_qc.to_anndata(var_names="symbol")
lap("qc metrics")

# %%
qc_violins = cl.violins(qc_data, ["n_genes_by_counts", "total_counts", "pct_counts_mt"])
qc_scatter = cl.xyplot(
    qc_data, "total_counts", "n_genes_by_counts", mapping=aes(color="pct_counts_mt")
)

# %% [markdown]
# ## Filtering cells and genes
#
# `filter_cells(min_genes=100)` is a filter on the QC column. kyto cannot express
# `filter_genes(min_cells=3)` yet (no per-gene count of detected cells, proposed task
# 6.13). As a stand-in, the counts of the kept cells are read once (`to_scipy`), the
# cells per gene counted in NumPy, and the genes kept by ID (`select_genes`).

# %%
cells = samples_qc.filter(cell_column("n_genes_by_counts") >= 100)
gene_ids = cells.gene_ids().to_list()
counts = cells.layer("counts").to_scipy()
cells_per_gene = np.bincount(counts.indices, minlength=len(gene_ids))
kept_genes = [
    gene_id
    for gene_id, n_cells in zip(gene_ids, cells_per_gene, strict=True)
    if n_cells >= 3
]
del counts
filtered = cells.select_genes(kept_genes)
lap("filters")

# %% [markdown]
# ## Doublets, normalisation, highly variable genes
#
# Scrublet scores each sample on its own (`batch_key="sample"`); `seed` is required
# (R7). The counts stay in their layer (scanpy copies `X` to `layers["counts"]`).
# `normalize_total` without `target_sum` scales to the median total, as scanpy's
# default. The plan is collected once here so the two branches below do not run it
# again.

# %%
preprocessed = (
    filtered.scrublet(batch_key="sample", seed=0)
    .normalize_total(into="normalized")
    .log1p(layer="normalized", into="log1p")
    .with_default_layer("log1p")
    .highly_variable_genes(n_top_genes=2000, batch_key="sample")
    .collect()
)
lap("scrublet, normalize, log1p, hvg")

# %% [markdown]
# ## PCA, neighbours, UMAP, Leiden
#
# scanpy's `pca` uses the highly variable genes and keeps every gene in the data
# (`mask_var`). kyto's `pca` reads every gene of the plan, so this branch keeps only
# the highly variable genes (the marker plots go back to `preprocessed`; a gene mask
# for `pca` is proposed task 6.14). The tutorial's first Leiden runs igraph's flavour
# with `n_iterations=2`; kyto's runs until no move improves (deviations.md "Leiden and
# Louvain parameters not provided"). `pca_variance_ratio` cannot be drawn: PCA's
# variances do not reach Python (7.8 Open (2)).

# %%
resolutions = (0.02, 0.5, 2.0)
plan = (
    preprocessed.filter(gene_column("highly_variable"))
    .pca(into="pca", n_components=50, seed=0)
    .neighbors("pca")
    .umap(into="umap", seed=0)
    .leiden(into="leiden", seed=0)
)
for resolution in resolutions:
    plan = plan.leiden(
        into=f"leiden_res_{resolution:4.2f}", resolution=resolution, seed=0
    )
clustered = plan.collect()
lap("pca, neighbors, umap, leiden")

# %% [markdown]
# ## Marker genes
#
# The Wilcoxon test runs on the clustered plan, so on its highly variable genes only
# (scanpy tests every gene; task 6.14 would close this). The marker dotplots read every
# gene: `preprocessed` is handed over and the Leiden columns are copied into its
# `AnnData` by cell (kyto cannot add another plan's cell column to a plan).

# %%
ranked = clustered.rank_genes_groups("leiden_res_0.50", method="wilcoxon").collect()
lap("rank genes groups")

adata = clustered.to_anndata(var_names="symbol")
markers = preprocessed.to_anndata(var_names="symbol")
if not markers.obs_names.equals(adata.obs_names):
    raise ValueError("the clustered and preprocessed plans hold other cells")
leiden_columns = ["leiden", *(f"leiden_res_{r:4.2f}" for r in resolutions)]
for column in leiden_columns:
    markers.obs[column] = adata.obs[column]
lap("to_anndata")

# %%
marker_genes = {
    "CD14+ Mono": ["FCN1", "CD14"],
    "CD16+ Mono": ["TCF7L2", "FCGR3A", "LYN"],
    # Note: DMXL2 should be negative
    "cDC2": ["CST3", "COTL1", "LYZ", "DMXL2", "CLEC10A", "FCER1A"],
    "Erythroblast": ["MKI67", "HBA1", "HBB"],
    # Note HBM and GYPA are negative markers
    "Proerythroblast": ["CDK6", "SYNGR1", "HBM", "GYPA"],
    "NK": ["GNLY", "NKG7", "CD247", "FCER1G", "TYROBP", "KLRG1", "FCGR3A"],
    "ILC": ["ID2", "PLCG2", "GNLY", "SYNE1"],
    "Naive CD20+ B": ["MS4A1", "IL4R", "IGHD", "FCRL1", "IGHM"],
    # Note IGHD and IGHM are negative markers
    "B cells": [
        "MS4A1",
        "ITGB1",
        "COL4A4",
        "PRDM1",
        "IRF4",
        "PAX5",
        "BCL11A",
        "BLK",
        "IGHD",
        "IGHM",
    ],
    "Plasma cells": ["MZB1", "HSP90B1", "FNDC3B", "PRDM1", "IGKC", "JCHAIN"],
    # Note PAX5 is a negative marker
    "Plasmablast": ["XBP1", "PRDM1", "PAX5"],
    "CD4+ T": ["CD4", "IL7R", "TRBC2"],
    "CD8+ T": ["CD8A", "CD8B", "GZMK", "GZMA", "CCL5", "GZMB", "GZMH", "GZMA"],
    "T naive": ["LEF1", "CCR7", "TCF7"],
    "pDC": ["GZMB", "IL3RA", "COBLL1", "TCF4"],
}


def one_group_each(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """Keep each gene in the first group that lists it.

    cellestial's dotplot takes a gene in one group only (scanpy draws it once per group
    that lists it).
    """
    seen: set[str] = set()
    result: dict[str, list[str]] = {}
    for group, genes in groups.items():
        unique = [gene for gene in dict.fromkeys(genes) if gene not in seen]
        seen.update(unique)
        if unique:
            result[group] = unique
    return result


# The first five ranked genes of each 0.5 cluster, named as `adata` names them.
names = dict(zip(adata.var["gene_id"], adata.var_names, strict=True))
top_genes: dict[str, list[str]] = {}
for group, gene_id in (
    ranked.group_by("group", maintain_order=True)
    .head(5)
    .select("group", "gene_id")
    .iter_rows()
):
    top_genes.setdefault(group, []).append(names[gene_id])

# %% [markdown]
# ## Plots
#
# Highly variable genes (`sc.pl.highly_variable_genes`: normalised dispersion against
# mean, one point per gene), PCA by sample and mitochondrial share on components 1-2
# and 3-4, UMAP by sample, cluster, doublet calls and QC (kyto has no `log1p_` QC
# columns, so the counts are shown as they are), the three resolutions, the marker
# dotplots at 0.02 and 0.5 (cellestial has no `standard_scale="var"`), the top five
# ranked genes of each 0.5 cluster (`sc.pl.rank_genes_groups_dotplot`) and the UMAP of
# cluster 7's top genes (cluster labels are arbitrary: kyto's 7 is not scanpy's).

# %%
plots: dict[str, Any] = {
    "qc_violins": qc_violins,
    "qc_scatter": qc_scatter,
    "highly_variable_genes": cl.xyplot(
        markers,
        "means",
        "dispersions_norm",
        axis=1,
        mapping=aes(color="highly_variable"),
    ),
    "pca": cl.pcas(adata, ["sample", "pct_counts_mt"], size=0.4, ncol=2),
    "pca_3_4": cl.pcas(adata, ["sample", "pct_counts_mt"], xy=(3, 4), size=0.4, ncol=2),
    "umap_sample": cl.umap(adata, key="sample", size=0.4),
    "umap_leiden": cl.umap(adata, key="leiden", size=0.4),
    "umap_doublets": cl.umaps(
        adata, ["leiden", "predicted_doublet", "doublet_score"], size=0.4, ncol=3
    ),
    "umap_qc": cl.umaps(
        adata,
        ["leiden", "total_counts", "pct_counts_mt", "n_genes_by_counts"],
        size=0.4,
        ncol=2,
    ),
    "umap_resolutions": cl.umaps(
        adata, leiden_columns[1:], size=0.4, ncol=3, legend_ondata=True
    ),
    "markers_res_0.02": cl.dotplot(
        markers, one_group_each(marker_genes), group_by="leiden_res_0.02"
    ),
    "markers_res_0.50": cl.dotplot(
        markers, one_group_each(marker_genes), group_by="leiden_res_0.50"
    ),
    "ranked_genes": cl.dotplot(
        adata, one_group_each(top_genes), group_by="leiden_res_0.50"
    ),
    "umap_cluster_7_genes": cl.umaps(
        adata, [*top_genes.get("7", []), "leiden_res_0.50"], size=0.4, ncol=3
    ),
}
for name, plot in plots.items():
    plot.to_png(str(output / f"{name}.png"))
lap("plots")

# %% [markdown]
# ## Outputs
#
# What the gate compares with scanpy's run: the clustered cells, the ranking and the
# timings (peak memory is the process's `ru_maxrss`).

# %%
adata.write_h5ad(output / "kyto.h5ad")
ranked.write_parquet(output / "ranked.parquet")
(output / "run.json").write_text(
    json.dumps(
        {
            "stages": seconds,
            "total": time.perf_counter() - started,
            "peak_rss_mb": _peak_rss_mb(),
            "peak_rss_mb_after": peak_rss_mb,
            "kept_genes": kept_genes,
        },
        indent=2,
    )
)
print(json.dumps(seconds, indent=2))
