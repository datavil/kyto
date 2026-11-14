# scanpy's clustering tutorial in kyto

scanpy's [basic clustering tutorial](https://scanpy.scverse.org/en/stable/tutorials/basics/clustering.html)
written in kyto (task 10.7), compared with scanpy's own run of it.

| File | What it is |
|---|---|
| [clustering.py](clustering.py) | the tutorial in kyto: a notebook in the percent format (`# %%` cells) that also runs as a script; every plot drawn by cellestial |
| [clustering_scanpy.py](clustering_scanpy.py) | scanpy's tutorial, step for step, as the reference (no plots) |
| [clustering_compare.py](clustering_compare.py) | the comparison below |
| [`tests/test_clustering_tutorial.py`](../../tests/test_clustering_tutorial.py) | runs both in the gate and holds the comparison to its bounds |

```bash
uv run python docs/tutorials/clustering.py s1d1.h5 s1d3.h5 <output folder>
uv run python docs/tutorials/clustering_scanpy.py s1d1.h5 s1d3.h5 scanpy.h5ad
uv run python docs/tutorials/clustering_compare.py <output folder> scanpy.h5ad s1d1.h5
```

## Data

The tutorial's two samples, `s1d1` and `s1d3` (10x Cell Ranger `.h5`, 22.8 and 20.4 MB,
figshare DOI 10.6084/m9.figshare.22716739.v1, the files the tutorial's `pooch` registry
fetches). `tests/fixtures/tutorial_samples.py` downloads them once into `.cache/fixtures`
and checks their SHA-256, as the pbmc3k fixture does. 17 125 cells (8785 + 8340), 36 601
genes plus 140 antibody features that kyto imports and the plan filters out.

## Results against scanpy

Measured Nov 14, 2026, macOS 27 arm64 (Apple M4, 10 cores, 16 GB), scanpy 1.12.5,
anndata 0.13.4, umap-learn 0.5.12, pynndescent 0.7.0, igraph 0.11.9; kyto's column
measured again Nov 15, 2026 at the 6.14 commit (the gene filter as a plan step, 6.13;
one plan over every kept gene with `pca(genes=gene_column("highly_variable"))`, 6.14).
Same seeds as the tutorial (scanpy's defaults, kyto `seed=0`).

| Check | kyto | scanpy | Agreement |
|---|---|---|---|
| Cells kept by `n_genes_by_counts >= 100` | 17 041 | 17 041 | same IDs, same samples |
| `n_genes_by_counts`, `total_counts`, `total_counts_mt`/`_ribo`/`_hb` | | | equal on every cell |
| `pct_counts_mt`, `pct_counts_ribo`, `pct_counts_hb` | `Float64` | `float32` | largest relative difference 1.0e-7 (scanpy's `float32` rounding) |
| Genes kept by `min_cells=3` (`filter(qc.cells_detected() >= 3)`, a plan step) | 23 427 | 23 427 | same IDs |
| Highly variable genes (`batch_key="sample"`) | 2000 | 2000 | 2000 in common |
| PCA variances (50 components, on the 2000 HVGs of 23 427 genes: `genes=` against `mask_var`) | | | largest relative difference 1.6e-6 |
| `leiden` (resolution 1) | 27 clusters | 24 clusters | ARI 0.819 |
| `leiden_res_0.02` | 5 | 5 | ARI 0.979 |
| `leiden_res_0.50` | 17 | 13 | ARI 0.920 |
| `leiden_res_2.00` | 36 | 35 | ARI 0.878 |
| UMAP trustworthiness, 5 / 15 neighbours (3000 seeded cells, against each tool's `X_pca`) | 0.9634 / 0.9616 | 0.9627 / 0.9602 | kyto's layout as trustworthy |
| Doublets called | 432 | 213 | 211 by both; scores correlate 0.987 |
| Top 10 Wilcoxon genes per `leiden_res_0.50` cluster (both rank all 23 427 genes; the scanpy cluster sharing most cells) | | | 6.6 of 10 in common on average; 7 to 10 for the 11 largest clusters (six of them 10), 0 to 6 for small ones that have no close scanpy cluster |

**Doublets.** The two samples run Scrublet with 25 000+ rows (cells and simulated
doublets) each, where scanpy switches to pynndescent's approximate search; kyto's search
is exact. scanpy's automatic threshold then flips between two minima from seed to seed:
over seeds 0 to 5 it calls 1 to 221 doublets in `s1d1` and 5 to 167 in `s1d3`. With
`use_approx_neighbors=False` scanpy calls 252 to 255 in `s1d1` (threshold 0.20) and 180 to
198 in `s1d3` (one seed of three flips to 20); kyto calls 249 to 255 (threshold 0.21) and
181 to 207 over seeds 0 to 5. So kyto agrees with scanpy's exact search; the gap is
pynndescent's (deviations.md "Doublet detection search").

**Leiden.** The tutorial's first Leiden is igraph's flavour with `n_iterations=2`; kyto's
runs until no move improves, so it finds more, smaller clusters at resolution 1 (lowest
ARI). The neighbour graphs also differ: scanpy's `neighbors` uses pynndescent above 4096
cells, kyto searches exactly.

## Time and memory

Wall clock of the whole tutorial in one process, both with every core (kyto's default;
scanpy's numba and BLAS threads), each run twice, the second given:

| | kyto | scanpy (first run) | scanpy (second run) |
|---|---|---|---|
| Read, combine, QC | 2.9 s | 6.7 s | 1.5 s |
| Filters | 0.0 s (plan steps: they run with the next stage) | 0.3 s | 0.2 s |
| Scrublet, normalize, log1p, HVG | 24.9 s | 69.1 s | 32.7 s |
| PCA, neighbours, UMAP, 4 Leiden | 10.8 s | 9.3 s | 8.2 s |
| Wilcoxon ranking (23 427 genes) | 1.0 s | 10.1 s | 8.5 s |
| Hand-off to `AnnData`, plots | 2.4 s | (not plotted) | |
| **Total** | **42.3 s** | **95.5 s** | **51.0 s** |
| **Peak RSS** | **4.5 GB** | **6.2 GB** | **5.9 GB** |

scanpy's first run includes numba compiling (scrublet's and UMAP's kernels); its second
reuses numba's cache. kyto's peak comes from the Scrublet stage (the process held 1.3 GB
before it: the QC `AnnData` of every gene; the gene filter is one more pass over the
kept cells inside that stage, 6.13);
Scrublet holds the layer of each batch and its simulated doublets in memory, as scanpy's
does (6.12 Open). Both Wilcoxon rankings test all 23 427 genes since task 6.14 (kyto's
tested the 2000 HVGs before, in 0.1 s). kyto's column was measured again on Nov 15 (one
run after a warm-up run); scanpy's is the Nov 14 measurement. kyto is the gate's build
(maturin builds the extension in release for `uv sync`).

## Where kyto differs, and the gaps

Each is in `Implementation Plan - Done.md` under 10.7 Open, with a proposed task where
it is real work:

1. **`filter_genes(min_cells=3)`**: closed by task 6.13. It is
   `filter(qc.cells_detected() >= 3)`, a plan step computed over the cells the plan
   kept (the notebook used to count cells per gene in NumPy outside the plan).
2. ~~**PCA on the highly variable genes keeps only those genes.**~~ Closed by task
   6.14: `pca(genes=gene_column("highly_variable"))` computes the components over the
   HVGs and keeps every gene, so clustering, the Wilcoxon ranking and the marker plots
   are one plan over all 23 427 genes. Unlike scanpy, kyto uses the HVGs only when
   asked (`genes=`; no implicit default, deviations.md "`pca` gene mask").
3. **`log1p_` QC columns** (`log1p_total_counts`, `log1p_n_genes_by_counts`): closed by
   task 6.13. They are `.log1p()` on the QC expressions, and the UMAP QC panels show
   them as the tutorial does.
4. **No `pca_variance_ratio` plot**: PCA's variances do not reach Python (`uns` is
   empty, 7.8 Open (2)).
5. **Leiden's `n_iterations`** is not provided (deviations.md).
6. **The haemoglobin pattern** `^HB[^(P)]` is written as "starts with HB and not with
   HBP": kyto's string expressions have no regular expressions. It selects the same
   genes here (no symbol is `HB` alone).
7. **Cell type annotation** (`obs["leiden_res_0.02"].map({...})`) has no kyto
   expression (no `replace` on cell columns); the tutorial only names the clusters,
   so the notebook leaves it out.
8. **cellestial (upstream):** `dotplot` takes each gene in one group only (scanpy draws
   `FCGR3A` under both CD16+ Mono and NK), so the notebook keeps each gene in its first
   group; it has no `standard_scale="var"`; `pca` takes one pair of components per
   call.
