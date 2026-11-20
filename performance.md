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

## Free a block's layers once nothing reads them (task 5.16)

Measured Nov 20, 2026, on the task 5.16 tree (base `4d512d6`, 5.15; release extensions built
by `uv sync`), same machine, data and scripts as 5.12 to 5.15 above. **Not a quiet machine:**
a game client used 1.3 to 1.5 cores throughout, and the 1-minute load average went from 2.3
to 7.9 during the runs (each line below states it). So the two builds ran interleaved, base
then 5.16, twice per configuration, each a fresh process with a warm-up run and the median of
5 (`bench.py`, equivalent to `thread_scaling.py`'s cases). **Peak memory:** `ru_maxrss` of a
fresh process running one configuration once, 5 alternating runs per build.

- **Change:** each block gives a layer back to the pool as soon as no later operator of its
  stream and not the output reads it, instead of at the stream's end; inside a fused
  `normalize_total` + `log1p` pass, `counts` goes back before `log1p` takes its buffers when
  nothing later reads it. `estimate()` counts a block's largest step (the layers it carries
  into a node plus what the node writes) and its most buffers at once.
- **What it does not change:** this workflow's output (`embedding().to_numpy()` writes the
  whole plan to a temporary store; `_execute` streams it) reads `counts`, `normalized` and
  `log1p`, so the output's last stream carries all three at 20,000 genes until the HVG gene
  filter, and nothing there is dead. The streams that gain are the HVG statistics pass (2
  layer buffers a block instead of 3 at once) and the held PCA read (the gene filter copies
  `log1p` only, not `counts` and `normalized` too).
- **The Done-when's peak was already met by 5.15:** "lower than 5.14's 1,336 MB" holds for
  5.15 too (median 1,318 MB here, 1,297 MB in 5.15's run); 5.16 itself lowers it by about 2%.

| Seconds (median of 5), base 5.15 / 5.16, two interleaved rounds | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| QC through PCA (`embedding().to_numpy()`) | 6.165, 6.163 / **5.958, 5.961** | 2.605, 2.755 / **2.532, 2.638** | 2.125, 2.091 / **2.098, 2.029** |
| preprocessing (`_execute`) | 2.870, 2.871 / 2.872, 2.873 | 1.039, 1.042 / 1.045, 1.036 | 0.812, 0.798 / 0.809, 0.800 |
| load average during the pair | 2.3 to 2.8 | 2.3 to 4.6 | 4.5 to 7.9 |

QC through PCA is 3.3% faster on 1 thread, 3 to 4% on 4 and 1 to 3% on 10 (the held read's
gene filter no longer copies two full-width layers); preprocessing is unchanged (its stream's
last operator is the maps, where the old code dropped the same layers).

| Peak memory at 10 threads, MB (5 runs, median) | base 5.15 | 5.16 |
|---|---|---|
| kyto QC through PCA | 1,294 to 1,342 (**1,318**) | 1,274 to 1,310 (**1,290**) |
| kyto preprocessing (`_execute`) | 1,082 to 1,126 (1,120) | 1,059 to 1,157 (1,131) |
| scanpy read through PCA (5.12, not re-run) | | 2,853 |

- **Where the peak is:** RSS traced by progress (QC through PCA, 10 threads): the first pass
  (HVG statistics) climbs from about 445 MB after its first window to about 1,000 MB (5.16:
  about 950 MB), about 100 MB every two windows, although a window's blocks and reads hold
  about 400 MB; then the output's last stream lifts it to about 1,300 MB. That growth is task
  5.18.
- **Bit identity (R11):** the QC-through-PCA embedding is bit-identical between 5.15 and 5.16
  at 1, 4 and 10 threads, and across thread counts; `thread_scaling.py` gave bit-identical HVG
  gene tables, cell tables and embeddings at 1, 4 and 10 threads on both builds.
- **Estimate:** on the fixtures (`estimate_fixtures.rs`) whole blocks are within -11.6% to
  +7.7% of the measured heap (the lowest, pbmc3k `maps` at 1,024-cell blocks and 1 in flight,
  was +0.9% before the buffer count changed: `log1p` now reuses `counts`' buffers); split and
  cut blocks stay high, never low by more than 25%.
- **scanpy** is not re-run (5.12's and 5.13's numbers: preprocessing 0.99 s on 1 thread and
  0.96 s on 10, `pp.pca` 1.94 s, read through PCA peak 2,853 MB).
- **Where kyto loses:** on 1 thread preprocessing takes 2.9x scanpy's time (scanpy has the
  matrix in memory, kyto decodes the store); kyto is faster from about 4 threads. The peak at
  10 threads is 45% of scanpy's, but this task did not lower it much: the first pass's RSS
  growth and the output's three full-width layers set it.

## Parallel encoding in the output sink (task 5.17)

Measured Nov 20, 2026, on the task 5.17 tree (base `fe194a8`, 5.16; release extensions built
by `uv sync`), same machine, data and scripts as 5.12 to 5.16 above. **Not a quiet machine:**
a game client ran throughout; the 1-minute load average was 6 to 10 during the timed runs.
Base and 5.17 ran alternating, each a fresh process running PCA alone (the `isolated` plan of
`pca_timing.py`) with a warm-up per thread count, then `embedding("X_pca").to_numpy()` and
`_execute` (`pca_timing.py`'s `stage`) once each; the median of 5 such processes per build.
"After the PCA" is end to end minus `_execute`: the handle terminal's temporary store, the
final write and reading the embedding back. **Peak memory:** `ru_maxrss` of a fresh process
running PCA alone end to end once at 10 threads, 3 alternating runs per build.

- **Change:** each shard of a layer array (1M stored values of `indices` or `data`, or a cell
  block's `indptr`) is now written by kyto: its 16 inner chunks are compressed with zstd in
  parallel and placed in chunk order, then the index, the same bytes zarrs writes on one
  thread. `indices` and `data` shards are written at once.
- **Finding 1, three layers:** the temporary store behind `embedding()` holds every layer the
  output carries: `counts`, `normalized` and `log1p` (8.7M stored values each), 54 shards,
  about 0.4 to 0.5 s of zstd CPU time, although `embedding()` reads none of them (task 5.19).
- **Finding 2, the sink is not on the run's pool:** `StoreWriter::push` runs on the sink's own
  thread, so its parallel work goes to rayon's global pool (one thread per core) whatever
  `threads` is. At `threads=1` the writes already overlap the stream on other cores: the time
  after the PCA is about 0.11 to 0.13 s, not the 0.42 s the task assumed, and the task's first
  target (2.5x from 1 to 10 threads, about 0.05 s at 10) lay below the final write's floor
  (about 55 ms: tables, embedding, sync, rename). The user changed the target to "1.5x less
  than 5.16's at 10 threads" (Nov 20, 2026). Running the writes on the run's pool (R7) is task
  5.20; it will make 1 thread about 0.4 s slower.

| Seconds (median of 5), base 5.16 / 5.17 | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| kyto PCA alone, end to end | 2.408 / 2.391 | 1.150 / **1.054** | 0.932 / **0.833** |
| kyto PCA stage (`_execute`) | 2.285 / 2.276 | 0.877 / 0.881 | 0.650 / 0.652 |
| after the PCA (end to end minus stage) | 0.131 / 0.109 | 0.273 / **0.176** | 0.281 / **0.181 (1.55x)** |
| scanpy `pp.pca` (5.15, not re-run) | 1.95 | 1.94 | 1.94 |

| Peak memory at 10 threads, MB (3 runs) | base 5.16 | 5.17 |
|---|---|---|
| kyto PCA alone, end to end | 471, 479, 450 | 451, 453, 446 |
| kyto PCA alone, 5.15 (its run) | | 464 |

- **Byte identity (R11):** `sink_zarr` of PCA alone at 1, 4 and 10 threads, base and 5.17: all
  136 files identical except `provenance.json` (target path, timestamps). Unit tests compare
  every file with zarrs' one-thread encoder on pools of 1, 3 and 8 threads.
- **scanpy** has no step that writes a store in the middle of PCA; its `pp.pca` (matrix in
  memory) is 5.15's number.
- **Where kyto loses:** on 1 thread PCA alone takes 1.23x scanpy's `pp.pca` (kyto decodes the
  stored layer and writes three layers it does not need to its temporary store); from 4
  threads kyto is faster (0.43x scanpy's time at 10). At 10 threads the time after the PCA is
  still 22% of the end-to-end time, mostly zstd on layers nobody reads (5.19).

## Handle terminals write only what they read (task 5.19)

Measured Nov 20, 2026, on the task 5.19 tree (base `bc43f8d`, 5.17; release extensions built
by `uv sync`), same machine, data and scripts as 5.12 to 5.17 above. **Not a quiet machine:**
a game client ran throughout; the 1-minute load average was 4 to 8 during the timed runs
(stated per table). Base and 5.19 ran alternating, as in 5.17: each a fresh process running
PCA alone (the `isolated` plan of `pca_timing.py`) with a warm-up per thread count, then
`embedding("X_pca").to_numpy()` and `_execute` (`pca_timing.py`'s `stage`) once each; the
median of 5 such processes per build. **Peak memory:** `ru_maxrss` of a fresh process running
PCA alone end to end once, 5 alternating runs per build and thread count.

- **Change:** a handle terminal's temporary store holds only the member it reads (an
  embedding, a graph, a layer, or one table's columns) and the ID columns. Before, it held
  the plan's whole output. The optimizer then leaves out every step that feeds nothing the
  terminal reads, so after the PCA's passes, PCA alone's output stream reads no layer.
  `sink_zarr`, `sink_h5ad`, `collect()` and `to_anndata` still write everything.
- **Layers written** to `embedding("X_pca").to_numpy()`'s temporary store on the 50k data
  (found by walking `TMPDIR` from the progress callback): base `counts`, `normalized`, `log1p`
  (54 zstd shards); 5.19 none.
- **"After the PCA" is now below zero.** End to end minus `_execute` was the time the handle
  terminal added after the PCA's work. `_execute` still runs the whole output, and its
  stream after the PCA decodes the store's three layers. The terminal's run no longer does
  that work, so its end-to-end time is now shorter than `_execute`. The end-to-end time is
  the honest comparison.

| Seconds (median of 5), base 5.17 / 5.19, load 4.1 to 7.9 | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| kyto PCA alone, end to end | 2.409 / **1.879** | 1.056 / **0.769** | 0.837 / **0.585** |
| kyto PCA stage (`_execute`, the whole output) | 2.281 / 2.306 | 0.883 / 0.902 | 0.652 / 0.664 |
| end to end minus stage | 0.120 / **-0.427** | 0.176 / **-0.121** | 0.188 / **-0.079** |
| scanpy `pp.pca` (re-run, load 4.8; matrix in memory) | 2.056 | 2.048 | 2.049 |

| Peak memory, MB (5 runs, median) | base 5.17 | 5.19 |
|---|---|---|
| kyto PCA alone, end to end, 10 threads | 451 to 461 (**455**) | 322 to 327 (**326**) |
| kyto PCA alone, end to end, 1 thread | 377 to 387 (**379**) | 276 to 282 (**279**) |

`thread_scaling.py` (median of 5, one run per build, base then 5.19; load 3.8 to 6.6):

| Seconds, base 5.17 / 5.19 | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| preprocessing (`_execute`) | 2.894 / 2.890 | 1.058 / 1.074 | 0.820 / 0.819 |
| QC through PCA (`embedding().to_numpy()`) | 5.983 / **5.588** | 2.384 / **2.142** | 1.837 / **1.621** |
| PCA alone (`embedding().to_numpy()`) | 2.403 / **1.875** | 1.065 / **0.776** | 0.840 / **0.586** |

- **Bit identity (R11):** the PCA-alone embedding, and QC through PCA's embedding and HVG
  gene table, are bit-identical between base and 5.19 at 1, 4 and 10 threads, and across
  thread counts (`thread_scaling.py` on both builds). Against scanpy (`pca_scanpy.py`): worst
  relative difference of the first 10 columns 2.9e-5, largest angle 1.7e-5 rad, as before.
- **Where kyto wins and loses:** PCA alone, end to end, is now faster than scanpy's `pp.pca`
  on 1 thread too (0.91x scanpy's time), and takes 0.29x at 10 threads. Here scanpy has the
  matrix in memory, while kyto decodes the stored layer. The PCA stage itself still takes
  1.12x scanpy's time on 1 thread. Preprocessing on 1 thread is still 2.9x scanpy's (5.16).
  PCA alone scales 3.2x from 1 to 10 threads end to end; the held read and the passes
  (the stage) are what is left. The sink's writes still run on rayon's global pool, not the
  run's (task 5.20); with no layer to write, PCA alone's sink now writes only the tables and
  the embedding.

## The sink on the run's thread pool (task 5.20)

Measured Nov 20, 2026, on the task 5.20 tree (base `5008943`, 5.19; release extensions built
by `uv sync`), same machine, data and scripts as 5.12 to 5.19 above. **Not a quiet machine:**
a game client ran throughout; the 1-minute load average was 3.4 to 6.5 during the timed runs
(stated per table). Base and 5.20 ran alternating, each a fresh process, as in 5.17 and 5.19:
a warm-up per thread count, then the timed call once; the median of 5 processes per build.
**Peak memory:** `ru_maxrss` of a fresh process running the case end to end once, 5
alternating runs per build and thread count.

- **Change:** the sink's writes (`StoreWriter::create`, every `push`, the tables, embeddings
  and graphs and `finish` after the stream, and the `.h5ad` export's reads of its temporary
  store) run on the run's rayon pool of `threads` threads. Before, they ran on the sink's
  consumer thread, outside any pool, so their rayon work (layers in parallel, 5.17's inner
  chunks, zarrs' decodes) went to rayon's global pool, one thread per core, whatever
  `threads` was (R7). The producer now enters the pool per window and hands blocks over
  from outside it, so the writes interleave with the windows on the same threads.
- **Thread count, checked:** `crates/kyto-exec/tests/sink_threads.rs` holds every thread of
  rayon's global pool with a waiting job, then sinks a two-layer plan to a store and to an
  `.h5ad` at 1 and 4 threads. Work that reached the global pool would never run; every sink
  finishes. On the base build the same test hangs at 1 thread (its 120 s deadline fails it).
- **Where 1 thread gets slower:** only where the sink writes layers. A handle terminal after
  5.19 writes only its member (no layer for an embedding), so PCA alone and QC through PCA
  do not get slower (both within noise, the medians slightly lower). `sink_zarr` of PCA
  alone writes the `counts`, `normalized` and `log1p` layers (54 zstd shards): its zstd work
  used to overlap the 1-thread stream on other cores and now shares the one thread.

| Seconds (median of 5), base 5.19 / 5.20, load 5.0 to 6.5 | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| kyto PCA alone, end to end (`embedding("X_pca").to_numpy()`) | 1.884 / 1.860 | 0.768 / 0.755 | 0.587 / 0.575 |
| kyto PCA stage (`_execute`, the whole output) | 2.306 / 2.282 | 0.889 / 0.880 | 0.660 / 0.654 |
| scanpy `pp.pca` (re-run, load 4.5; matrix in memory) | 1.955 | 1.948 | 1.940 |

| Seconds (median of 5), base 5.19 / 5.20, load 3.4 to 5.4 | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| kyto PCA alone, `sink_zarr` (3 layers, embedding, tables) | 2.376 / **2.849** (+20%) | 1.047 / **1.094** (+4.5%) | 0.827 / 0.818 |

`thread_scaling.py` (median of 5, one run per build, base then 5.20; load 3.8 to 6.4):

| Seconds, base 5.19 / 5.20 | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| preprocessing (`_execute`) | 2.884 / 2.876 | 1.037 / 1.058 | 0.812 / 0.814 |
| QC through PCA (`embedding().to_numpy()`) | 5.573 / 5.527 | 2.126 / 2.101 | 1.619 / 1.599 |
| PCA alone (`embedding().to_numpy()`) | 1.861 / 1.849 | 0.775 / 0.759 | 0.588 / 0.579 |

| Peak memory, MB (5 runs, median), base 5.19 / 5.20 | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| PCA alone, end to end (load 4.3 to 5.4) | 279 / 278 | 325 / 322 | 328 / 322 |
| QC through PCA, end to end (load 3.8 to 4.3) | 1002 / 1002 | 1047 / 1057 | 1110 / 1118 |

- **Bit identity (R11):** `sink_zarr` of PCA alone at 1, 4 and 10 threads, base and 5.20: all
  135 files identical except `provenance.json` (target path, timestamps). `sink_h5ad` of the
  same plan: every dataset and attribute identical (h5py walk); the raw bytes differ between
  any two runs, base against base too, only in HDF5's object timestamps. The PCA-alone
  embedding, and QC through PCA's embedding and HVG gene table (`thread_scaling.py`), are
  bit-identical across builds and thread counts. Against scanpy (`pca_scanpy.py`): worst
  relative difference of the first 10 columns 2.9e-5, largest angle 1.7e-5 rad, as before.
- **Where kyto wins and loses:** PCA alone end to end takes 0.95x scanpy's `pp.pca` time on 1
  thread and 0.30x at 10. A `sink_zarr` that writes layers is now 20% slower on 1 thread than
  before: the price of R7. The 4- and 10-thread peaks of QC through PCA are 1.0% and 0.7%
  higher, within the spread of the 5 runs (1037 to 1084 MB at 4 threads, 1100 to 1145 MB
  at 10).
- **Not covered:** Polars' own thread pool (`POOL`, Parquet table reads and writes) is
  sized by `POLARS_MAX_THREADS`, not by `threads`. It is not rayon's global pool, and the
  Parquet table I/O of every run (not only the sink's) can use it.

## Every block buffer through the stream's pool (task 5.18)

Measured Nov 20, 2026, on the task 5.18 tree (base `bc3852f`, 5.20; release extensions built
by `uv sync`, the base in a worktree of its own), same machine, data and scripts as 5.12 to
5.20 above. **Not a quiet machine:** a game client ran throughout; the 1-minute load average
was 3.6 to 6.5 during the runs (stated per table). Base and 5.18 ran alternating, each a
fresh process: for time, a warm-up and then one timed run; for peak, one run and its
`ru_maxrss`; 5 rounds, the median of 5 per build, case and thread count.

- **Profile.** RSS traced on every progress call of QC through PCA at 10 threads (resident
  and `reusable` from `task_info(TASK_VM_INFO)`; `reusable` counts pages the allocator freed
  but kept). RSS grew about 95 MB per window inside each streaming pass and fell back at the
  pass's end, and `reusable` stayed at 0 to 2 MB: the growth was live heap, not allocator
  retention or fragmentation. The cause: the store's reader joined every block into new
  buffers, the consumer gave every block back to the stream's `CellBlockPool`, and only the
  operators took buffers from it. The pool keeps up to `(window + 1) x (layers + operators)`
  blocks' buffers, so each window parked about one window of read buffers in it, while the
  next window's reads allocated new ones, until the cap or the pass's end. The same happened
  in the output's last stream.
- **Change:** `kyto_store::BlockBuffers` (implemented by `CellBlockPool`) and
  `CellBlocks::with_buffers`: every stream's reader joins its blocks into buffers taken from
  the stream's pool, so a block's buffers go round (read, operators, consumer, pool, the next
  read) and no stream allocates block buffers after its first window. The values written
  are the same, so the bits are too. `estimate()`'s pool term changed to match: later
  blocks take their reads' buffers from the pool too.
- **Not used:** the plan's candidate (shrink layers to the HVG genes after their last
  full-width reader). Since 5.19 the handle terminal's last stream reads no layer, and the
  first pass now stays flat, so there is no full-width stream left for it to shrink.

RSS by pass, QC through PCA at 10 threads (MB at the pass's first progress call -> its
highest; 14 passes: the HVG statistics, the HVG filter's, the held PCA read and passes, the
output; load 4.8):

| pass | 1 (HVG statistics) | 2 | 3 to 13 | 14 (output) | peak |
|---|---|---|---|---|---|
| base 5.20 | 55 -> **972** (459, 579, 682, 779, 882, 972 by window) | 580 -> 1,022 | 580 to 637, flat | 718 -> **1,146** | 1,147 |
| 5.18 | 54 -> **500** (454, 479, 485, 499, 499, 500 by window) | 506 -> 542 | 542 to 598, flat | 617 -> 627 | **628** |

| Peak memory, MB (5 runs, median [range]), load 3.6 to 6.0 | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| QC through PCA, base 5.20 | 1,014 [1,013-1,016] | 1,080 [1,064-1,085] | 1,135 [1,113-1,156] |
| QC through PCA, 5.18 | **512** [511-514] | **670** [664-672] | **646** [631-660] |
| preprocessing (`_execute`), base / 5.18 | 903 / **461** | 1,185 / **777** | 1,129 / **710** |
| PCA alone, base / 5.18 | 278 / 278 | 323 / 305 | 321 / 306 |
| scanpy read through PCA (re-run, 3 runs, load 4.8) | | | 2,455 / 2,851 / 2,811 |

At 10 threads QC through PCA peaks 43% below base and 50% below 5.16's 1,290 MB (the
Done-when's reference); it is 23% of scanpy's peak for the same workflow.

| Seconds (median of 5), base 5.20 / 5.18, load 3.6 to 6.5 | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| QC through PCA (`embedding().to_numpy()`) | 5.518 / 5.540 (+0.4%) | 2.107 / 2.095 | 1.588 / **1.542** (-2.9%) |
| preprocessing (`_execute`) | 2.864 / 2.852 | 1.060 / 1.033 | 0.814 / **0.776** (-4.7%) |
| PCA alone (`embedding("X_pca").to_numpy()`) | 1.854 / 1.874 (+1.1%) | 0.761 / **0.781 (+2.6%)** | 0.579 / 0.583 (+0.7%) |

- **PCA alone at 4 threads is 2.3 to 2.6% slower than the base build** (this table, and 8
  alternating rounds at load 5.8: base 0.853, 5.18 0.873 s). The time is lost in the PCA's
  passes over the held blocks, not in the held read (progress traced, the read at 0.047 s in
  both), and the passes do not touch the reader or the pool. The same 5.18 build with the
  reads' pool switched off (a temporary environment switch, removed) took 0.865 s (minimum
  0.843 against 0.844 with the pool): the gap is between the two builds (code layout), not
  what the change does at run time. Above the Done-when's 2%: the user accepted the gap
  and the commit with it (Nov 20, 2026).
- **Bit identity (R11):** the QC-through-PCA and PCA-alone embeddings and the
  preprocessing HVG gene table are byte-identical between base and 5.18 at 1, 4 and 10
  threads, and across thread counts on 5.18.
- **Estimate:** on the fixtures (`estimate_fixtures.rs`) whole blocks are within -6.4% to
  +24.1% of the measured heap (the high end: pbmc3k at 1024-cell blocks, whose third and
  last block holds 652 cells, read as if whole), split and cut blocks never below -2.5%.
  The test's consumer now keeps each block until the next one arrives (a slow terminal,
  the case the estimate bounds): a consumer that gives a block back at once races the next
  block's read for its buffers, and the measured peak of one row moved between +6% and +26%
  from run to run.
- **Memory limit (`memory_limit.rs`):** a first version of the new pool term counted
  only the buffers a window holds at once, as if later blocks took them all back at once;
  it fell 25 to 37% below the measured heap on the test's 84,000-cell store, the fit chose
  2,048-cell blocks, and peak RSS rose from 42.4 MB (base, 512 x 2 blocks) to 56.3 MB
  (6 runs each, release), over 64 MB in debug builds. The term now counts the buffers the
  consumer holds (a later block allocates as many again; counted once the stream runs past
  its second window), splits the later window into reading (only its reads' buffers have
  left the pool) and computing, and counts a read's pooled buffer at the largest pooled
  layer (LIFO: it may be a dense `scale` block's). Estimate against the measured heap on
  that store: -1% to -11% at 512 x 2, 1,024 x 1 and x 2, 2,048 x 1 (base +13% to +20%);
  the fit now chooses 1,024 x 1 and peak RSS is 38.1 to 38.5 MB.
- **Where kyto loses:** preprocessing on 1 thread is still 2.9x scanpy's time (5.16), and
  PCA alone's 4-thread gap above.

## Reproducing

Scripts and commands: [benchmarks/scanpy_workflow](benchmarks/scanpy_workflow/README.md).
