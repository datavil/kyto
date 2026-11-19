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

## PCA with its input held in memory (task 5.12)

Measured Nov 19, 2026, on the task 5.12 tree (base `c08894b`; release extension built by
`uv sync`), scanpy 1.13.0, numpy 2.5.5, scikit-learn 1.9.1.

- **Machine:** Apple Mac, 10 cores (4 performance + 6 efficiency), 16 GB RAM, macOS 27.
- **Data:** the same synthetic generator, 50,000 cells x 20,000 genes (71.2M entries).
  QC filters keep 49,888 cells; 2,000 HVGs leave a `float32` matrix of 8.74M entries.
- **Work, both tools:** PCA of that HVG `log1p` matrix, 50 components, seed 0, implicit
  centring, `float32` embedding out. scanpy: `sc.pp.pca(n_comps=50, random_state=0)`
  (sparse `arpack`) on the matrix in memory, threads capped with threadpoolctl and
  `sc.settings.n_jobs`; second run reported (the first, cold, took 5.2 s). kyto:
  `scan(hvg_store).pca(...)` to the embedding, so it also reads and decodes the stored
  HVG layer once (`isolated`); and the whole plan from the counts store, QC through PCA
  (`workflow`, the HVG pass included). kyto sums in `f64` (6.5); scanpy runs ARPACK on
  the `float32` matrix.
- **Held vs streamed:** held is the default. Streamed is forced with a `memory_limit`
  that leaves no room to hold the input (180 MB isolated, 1 GB workflow); that limit
  also shrinks the blocks, so the streamed times are an upper bound on streaming at full
  blocks. Times are the median of 3 runs.
- **Peak memory:** `ru_maxrss` of a fresh process running one configuration at 10
  threads (bytes on macOS).

| Seconds | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| scanpy `pp.pca` (matrix in memory) | 1.96 | 1.96 | 1.95 |
| kyto PCA, isolated, held | 2.82 (1.44x) | 1.31 (0.67x) | 1.13 (0.58x) |
| kyto PCA, isolated, streamed | 6.07 | 4.06 | 3.95 |
| kyto QC through PCA (workflow), held | 6.62 | 3.03 | 2.67 |
| kyto QC through PCA (workflow), streamed | 25.37 | 16.60 | 15.30 |

| Peak memory (process) | |
|---|---|
| scanpy read through PCA | 2,853 MB (2,802 MB before PCA: reading the `.h5ad`) |
| kyto PCA isolated, held / streamed | 408 MB / 304 MB |
| kyto QC through PCA, held (no limit) / streamed (1 GB limit) | 1,315 MB / 485 MB |

- **Bit identity (R11):** the embedding is bit-identical held and streamed, at 1, 4 and 10
  threads, in both cases (`pca_timing.py`).
- **Parity:** kyto's first 10 components against scanpy's: worst column difference
  2.9e-5 relative to the column, largest principal angle 1.7e-5 rad.
- **Where kyto loses:** on one thread PCA alone is 1.44x scanpy's time (it decodes the
  stored layer, which scanpy has in memory; inside the 1.5x target). Holding costs
  memory: about 100 MB more than streaming at isolated PCA (71 MB of entries plus
  allocator slack). The estimate holds the layer's schema size, an upper bound: 570 MB
  for the workflow, where 71 MB is held, because the HVG filter's genes are not known
  before the run; under a tight `memory_limit` that over-count makes PCA stream when
  holding would fit.
- The workflow's held run is 5.7x faster than streaming at 10 threads: the 10 passes it
  saves each re-read the store, re-ran QC, normalisation, `log1p` and the gene subset.

## One block in flight per thread (task 5.13)

Measured Nov 19, 2026, on the task 5.13 tree (base `d0d471d`; release extension built by
`uv sync`), same machine, data and scanpy as 5.12 above.

- **Change:** a run without `memory_limit` streams one cell block per thread at once
  (was 2). With a limit, the fit starts there and tries fewer until the plan fits.
- **Work, kyto:** from the converted counts store. `preprocessing` is QC, the three
  filters, `normalize_total(1e4)`, `log1p`, 2,000 `seurat` HVGs and the HVG filter,
  streamed to nowhere (`_execute`); `QC through PCA` adds PCA (50 components, seed 0) to the
  embedding; `PCA alone` is the 5.12 isolated case. Each run reads and decodes the store.
- **Work, scanpy:** the same steps on the `.h5ad` read into memory (read 0.09 s, not
  included), threads capped with threadpoolctl and `sc.settings.n_jobs`, second run.
- **Times:** median of 5 runs after a warm-up (`thread_scaling.py --repeats 5`, final tree). **Peak memory:**
  `ru_maxrss` of a fresh process running one configuration once (kyto); scanpy's is 5.12's
  read-through-PCA run.
- **Before** is the same tree with the old default (2 blocks in flight), rebuilt; its
  times are medians of 3.
- **Memory-limit fit:** with a limit, the fit now tries the thread count, then every
  power of two below it (10, 8, 4, 2, 1), not its halvings (10, 5, 2, 1), so a limit that
  fits 2 full blocks still gets them from 3 threads.

| Seconds (speedup over 1 thread) | 1 thread | 2 | 4 | 10 | 10, before |
|---|---|---|---|---|---|
| kyto preprocessing | 2.90 | 1.56 | 1.05 | 0.80 (3.6x) | 1.15 (2.5x) |
| scanpy preprocessing (in memory) | 0.99 | | | 0.96 | |
| kyto QC through PCA | 6.50 | 3.71 | 2.64 | 2.15 (3.02x) | 2.67 (2.5x) |
| kyto PCA alone (held input) | 2.72 | 1.67 | 1.30 | 1.12 (2.4x) | 1.13 (2.5x) |
| scanpy `pp.pca` (in memory) | 2.02 | | | 1.93 | |

| Peak memory, MB | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| kyto preprocessing, before / after | 1,001 / 958 | 1,268 / 1,622 | 1,265 / **2,404** |
| kyto QC through PCA, before / after | 1,126 / 1,096 | 1,362 / 1,756 | 1,364 / **2,616** |
| kyto PCA alone, before / after | | | 408 / 570 |
| scanpy read through PCA | | | 2,853 |

- **Run to run:** an earlier run (median of 3, before the memory-limit fit change, which
  does not touch runs without a limit) gave 2.91 / 0.81 s (3.6x) for preprocessing, 6.56 /
  2.21 s (**2.97x**) for QC through PCA and 2.73 / 1.14 s (2.4x) for PCA alone: QC through
  PCA sits at the 3x bar, within the noise between runs.
- **Bit identity (R11):** the HVG gene table, the cell table and both embeddings are
  bit-identical at 1, 2, 4 and 10 threads.
- **Why not 5x:** the work costs more CPU on more threads: preprocessing uses 2.95 s of
  CPU on 1 thread and 5.05 s on 10 (6 of the 10 cores are efficiency cores, memory
  bandwidth is shared, idle rayon threads spin). Even with every core busy that caps it at
  about 5.8x; on average 6.2 cores are busy (5.4 for QC through PCA, 4.5 for PCA alone).
  It is not block granularity: 2,048- and 1,024-cell blocks (26 and 49 per pass instead of
  13) took 0.83 s and 0.90 s against 0.82 s. PCA alone does not change: its passes over
  the held input fold leaves in chunks of the thread count (13 leaves: 10, then 3) with
  serial dense steps between passes (task 5.15).
- **Where kyto loses:** peak memory nearly doubles at 10 threads (QC through PCA 1,364 to
  2,616 MB, close to scanpy's 2,853 MB): every thread holds a 4,096-cell block of every
  layer. Smaller blocks undo most of it at the same speed (preprocessing at 10 threads:
  1,785 MB with 2,048-cell blocks, 1,205 MB with 1,024; task 5.14). On 1 thread kyto's
  preprocessing is 2.9x scanpy's time, which has the matrix in memory (kyto decodes the
  store); kyto matches scanpy at about 4 threads (1.05 s vs 0.96 s) and is 1.2x faster
  at 10.

## Reproducing

Scripts and commands: [benchmarks/scanpy_workflow](benchmarks/scanpy_workflow/README.md).
