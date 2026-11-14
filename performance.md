# Performance

kyto against scanpy on a basic clustering workflow. Measured Nov 14, 2026, at commit
`3dfe804` (kyto built with `maturin develop --release`), scanpy 1.13.0 with
`flavor="igraph"` Leiden.

## Setup

- **Machine:** cloud Linux container, 2 CPU cores, 7 GB RAM. Absolute times are slow; the
  ratio between the tools is the useful number.
- **Data:** synthetic, because public datasets could not be downloaded there. 20,000 genes,
  8 clusters, clustered negative-binomial counts (gamma–Poisson, dispersion 2), about 7%
  nonzero (~1,400 genes per cell), 13 `MT-` genes, seed 0. 10,000 and 50,000 cells.
- **Workflow, both tools:** QC (genes detected, percent mitochondrial), filters
  `200 <= n_genes < 2500` and `pct_mt < 5`, `normalize_total(1e4)`, `log1p`, 2,000 HVGs
  (`seurat`), keep HVGs, PCA (50, seed 0), neighbours (15), Leiden (seed 0), UMAP (seed 0),
  write the result.
- **scanpy:** read `.h5ad` → … → write `.h5ad`, timed per step. `sc.settings.n_jobs = 2`.
  Timings are from a warm second run; the first run spent 149 s in neighbours compiling
  numba code.
- **kyto:** `ky.convert` to a store, then one lazy plan run by `sink_zarr`, exact neighbour
  search, no `memory_limit`. Per-step times come from running the plan cut after each step
  and taking differences, so they are approximate.
- **Peak memory:** `ru_maxrss` of the process.
- Each configuration ran once.

## Results

| | 10k cells | 50k cells |
|---|---|---|
| scanpy, total (read → write) | 65 s | 72 s |
| kyto, total (convert + run) | 72 s (1.8 + 70.2) | 181 s (5.4 + 175.6) |
| Peak memory, scanpy / kyto | 0.77 / 0.93 GB | 2.02 / 1.70 GB |
| Cells kept / Leiden clusters | 9,977 / 8 (both) | 49,888 / 8 (both) |

### Per step, 50k cells

| Step | scanpy | kyto |
|---|---|---|
| Read (or convert), QC, normalize, log1p, HVG | 9.4 s | 6.0 s run (+ 5.4 s convert) |
| PCA | 2.6 s | ~26 s |
| Neighbours | 31.0 s (mostly numba compilation) | ~28 s (exact) |
| Leiden | 1.6 s | ~3 s |
| UMAP | 26.7 s | ~112 s |

### Per step, 10k cells

| Step | scanpy | kyto |
|---|---|---|
| Read through Leiden | 49.1 s (40.5 s neighbours) | ~8.7 s |
| UMAP | 15.5 s | ~56 s |

kyto's `search="approximate"` at 10k gave the same total (69.5 s), so neighbour search is not
where kyto loses time.

## Findings

1. **UMAP is the main gap:** about 4× slower than scanpy. Both run UMAP's SGD on one
   thread (scanpy uses `parallel=False`), so the gap is in kyto's implementation (6.9).
2. **PCA is about 10× slower** at 50k cells (6.5): the streaming randomized SVD against
   scanpy's in-memory PCA on the 2,000 HVGs.
3. **Preprocessing is faster** than scanpy, and peak memory is lower at 50k cells
   (1.70 vs 2.02 GB) even without a memory limit.
4. scanpy pays numba compilation on fresh processes (149 s cold, 31–40 s warm for
   neighbours here); a long-lived session would narrow kyto's neighbour advantage.

## Reproducing

The scripts lived outside the repo; in short: generate counts with
`numpy.random.default_rng(0)` as above and write `.h5ad` (no `obs`/`var` columns that repeat
the index name, or `ky.convert` raises `KY-E121`), then run the chain from the README
extended with `.pca(layer="log1p", into="X_pca", n_components=50, seed=0)`,
`.neighbors("X_pca", n_neighbors=15, seed=0)`, `.leiden(seed=0, into="leiden")`,
`.umap(into="X_umap", seed=0)` and `sink_zarr`.
