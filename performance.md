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

## Smaller blocks on many threads, cut from whole reads (task 5.14)

Measured Nov 19, 2026, on the task 5.14 tree (base `c186467`; release extension built
by `uv sync`), same machine, data and scripts as 5.13 above.

- **Change:** a run without `memory_limit` on more than 2 threads still holds one block
  in flight per thread, but each holds `ceil(2 x C / threads)` cells (820 at 10 threads,
  2,048 at 4), so the cells in flight are those of 2 store cell blocks, as before 5.13.
  The store is still read in whole cell blocks: a window reads `window x B / C` of them
  (2 at 10 threads), decodes their tiles once, and the reader joins each block's parts
  from the decoded tiles in parallel. 1 and 2 threads are unchanged.
- **Times:** median of 5 runs after a warm-up (`thread_scaling.py --threads 1,2,4,10
  --repeats 5`). **Peak memory:** `ru_maxrss` of a fresh process running one
  configuration once. **5.13** is the same tree with the old rule (whole blocks),
  measured twice in the same session through a temporary switch, since removed.
- **scanpy** is not re-run: its numbers are 5.13's and 5.12's (same machine and data).

| Seconds, 10 threads | 5.13 (whole blocks) | split reads (first try) | 5.14 (cut from whole reads) |
|---|---|---|---|
| kyto preprocessing | 0.815 / 0.799 | 0.961 to 0.976 | **0.805** |
| kyto QC through PCA | 2.170 / 2.182 | 2.385 to 2.406 | **2.161** |
| QC through PCA peak, MB | 2,606 / 2,547 | 1,269 to 1,308 | **1,336** |

- **Split reads** (the first try, not kept) read each 820-cell part from the store on
  its own: every part reads all 20 gene tiles of its cell block, with each tile's `indptr`
  whole and the zstd chunks (65,536 entries) around its rows, so a cell block costs five
  times the tile reads and about twice the decoding. A first cut-in-memory version read
  whole blocks and copied the parts out (2.30 s, 0.905 s): the reader joined each whole
  block's rows on one thread, so only 2 of 10 threads joined. The kept version joins the
  parts in parallel from the decoded tiles.

| Seconds (speedup over 1 thread) | 1 thread | 2 | 4 | 10 |
|---|---|---|---|---|
| kyto preprocessing | 2.90 | 1.56 | 1.06 | 0.81 (3.6x) |
| scanpy preprocessing (in memory, 5.13) | 0.99 | | | 0.96 |
| kyto QC through PCA | 6.49 | 3.71 | 2.69 | 2.16 (3.0x) |
| kyto PCA alone (held input) | 2.72 | 1.68 | 1.29 | 1.11 (2.4x) |
| scanpy `pp.pca` (in memory, 5.13) | 2.02 | | | 1.93 |

| Peak memory, MB | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| kyto preprocessing, 5.13 / 5.14 | 958 / 959 | 1,622 / 1,178 | 2,404 / **1,158** |
| kyto QC through PCA, 5.13 / 5.14 | 1,096 / 1,096 | 1,756 / 1,304 | 2,616 / **1,336** |
| kyto PCA alone, 5.13 / 5.14 | | not measured / 450 | 570 / 478 |
| scanpy read through PCA (5.12) | | | 2,853 |

- **Bit identity (R11):** the HVG gene table, the cell table and both embeddings are
  bit-identical at 1, 2, 4 and 10 threads.
- **Estimate:** `estimate(threads=10)` gives the same bytes read as 1 thread (whole
  reads). Its peak is high for cut parts: 1,498 MB modelled against 1,158 MB measured for
  preprocessing at 10 threads. On the fixtures it runs up to 70% high and is never low
  (`estimate_fixtures.rs`), because the reading term assumes every tile decodes at once.
- **Where kyto loses:** on 1 thread kyto's preprocessing is 2.9x scanpy's time, which
  has the matrix in memory (kyto decodes the store); kyto catches up at about 4 threads
  and is 1.2x faster at 10. PCA alone is 1.35x scanpy's time on 1 thread (inside 5.12's
  1.5x target) and 0.58x at 10. QC through PCA now peaks at 47% of scanpy's read-through-PCA
  peak at 10 threads (it was 92% after 5.13).

## Parallel PCA passes over a held input (task 5.15)

Measured Nov 19, 2026, on the task 5.15 tree (base `0b30f09`; release extension built by
`uv sync`), same machine, data, scanpy and scripts as 5.12 to 5.14 above. A game launcher
kept about 0.7 of a core busy through every run (load average 1.5 when idle), so each
comparison below ran its builds back to back in the same session; the final timings ran at
load 1.5.

- **Change:** each PCA pass over the held input folds a leaf's run of blocks as one group,
  and splits it inside: row work (the sketch's random rows, each row's projection on the
  basis, the embedding rows) runs in parallel over tiles of 64 rows, which walk the genes
  128 at a time so the basis rows they read stay in cache; the per-gene sums run in parallel
  over fixed parts of 256 genes, each gene adding its rows in cell order. Neither split
  depends on the thread count, and each sum adds in the old order, so the bits are those of
  the row-by-row fold (R11). Between passes, the Rayleigh-Ritz step's two products run as two
  fixed halves on the pool; the basis's QR stays on one thread.
- **Before 5.15** (10 threads): of PCA alone's 1.11 s, the held read took 0.07 s, the 11
  passes' folds about 0.53 s, the dense steps between them 0.13 s, and the handle
  terminal's temporary store (writing the 2,000-gene `log1p` layer, `materialize.rs`) about
  0.35 s, which barely scales (0.42 s on 1 thread; task 5.17).
- **Stage:** PCA alone run with no sink, `_execute` (`pca_timing.py --cases stage`): the
  held read, the passes and the output's stream, without the temporary store. Median of 5
  after a warm-up. **End to end:** `pca_timing.py` and `thread_scaling.py`, median of 5.
  **Peak memory:** `ru_maxrss` of a fresh process running one configuration once, median
  of 3 alternating runs per build.

**Dense products against peak memory.** Every thread that runs a gemm keeps its own
packing buffers, so the more of the dense products between passes run on the pool, the
faster the stage and the higher its peak. Three builds, same session, stage at 1 and 10
threads, peak at 10 threads against base:

| Build | 1 thread | 10 threads | Speedup | Peak at 10 threads |
|---|---|---|---|---|
| base (5.14) | 2.57 s | 0.83 s | 3.1x | 369 to 384 MB stage, 446 to 477 MB PCA alone |
| A: every dense product in 32-row groups on the pool | 2.34 s | 0.626 s | 3.74x | 25 to 35 MB above base |
| B: QR on one thread, Rayleigh-Ritz in 32-row groups | 2.31 s | 0.652 s | 3.55x | 13 to 30 MB above base |
| **C, kept: QR on one thread, Rayleigh-Ritz in 2 halves** | 2.32 s | 0.66 s | 3.43 to 3.52x | equal to base |

The user took C and set the target at 3.4x (A breaks "peak not above 5.14's"). The final
run of C gave:

| Seconds (speedup over 1 thread) | 1 thread | 2 | 4 | 10 |
|---|---|---|---|---|
| kyto PCA stage (`_execute`) | 2.311 | 1.251 | 0.903 | **0.661 (3.50x)** |
| kyto PCA alone, end to end, base (5.14) | 2.70 | 1.69 | 1.28 | 1.11 (2.43x) |
| kyto PCA alone, end to end, 5.15 | 2.466 | 1.487 | 1.163 | 0.937 (2.63x) |
| kyto PCA alone, streamed (`memory_limit` 180MB) | 5.72 | 3.51 | 2.89 | 2.55 |
| kyto QC through PCA, base (5.14) | 6.47 | 3.70 | 2.69 | 2.16 (3.0x) |
| kyto QC through PCA, 5.15 | 6.25 | 3.53 | 2.60 | 2.00 (3.13x) |
| scanpy `pp.pca` (matrix in memory) | 1.95 | | 1.94 | 1.94 |

| Peak memory at 10 threads, MB (median of 3) | base (5.14) | 5.15 |
|---|---|---|
| kyto PCA stage | 380 | 372 |
| kyto PCA alone, end to end | 461 | 464 |
| kyto QC through PCA | 1,332 | 1,297 |
| scanpy read through PCA (5.12) | | 2,853 |

At 1 thread the stage peaks at 274 MB (base 276 MB).

- **Bit identity (R11):** the embedding is bit-identical at 1, 2, 4 and 10 threads, held
  and streamed (`pca_timing.py`), and in QC through PCA (`thread_scaling.py`). It differs
  from 5.14's by at most 1.2e-7: the Rayleigh-Ritz products are cut differently.
- **Parity with scanpy:** first 10 components, worst column difference 2.9e-5 relative to
  the column, largest principal angle 1.7e-5 rad (as in 5.12).
- **Where kyto loses:** on 1 thread PCA alone takes 1.27x scanpy's time (scanpy has the
  matrix in memory; kyto decodes the stored layer and writes the temporary store). From 2
  threads kyto is faster, 0.48x scanpy's time at 10. End to end, PCA alone scales only
  2.63x: the temporary store's 0.35 s is the largest serial part left (task 5.17). The
  stage reaches 3.5x of a 5.8x hardware ceiling: the QR and the eigen solve between passes
  stay serial, and the folds are bound by L2 bandwidth (each stored value reads a 480-byte
  row of the basis).

## Reproducing

Scripts and commands: [benchmarks/scanpy_workflow](benchmarks/scanpy_workflow/README.md).
