# Held inputs: same leaves, same bits (task 5.12)

**Works:** a node with several statistics passes (`pca`, `seurat_v3` HVG) streams its
input once through the stages below it and keeps each stream block's input layer (a
compact `clone`: `AlignedBuffer::clone` copies exactly the entries), then folds every
pass over the held blocks with the streamed path's leaves: one leaf per run of
consecutive blocks of one store cell block (all blocks for a `sequential` pass), the
leaves pushed into `BlockOrderTree` in block order. Those are the blocks
`reduce_blocks` would fold, in the same order, so the bits match the streamed path on
any thread count and block split (`kyto-exec/src/held_input.rs`). Fold the cell blocks
in chunks of the pool's thread count, not all at once, or every leaf (PCA: `n_genes x
(l x 8 + 16)`) is alive together.

**Choice:** `PhysicalOptions::hold_inputs` (default on). `within_memory_limit` searches
with held inputs first and only then streams every pass. Progress: a held node adds one
pass (the read) to the total; that is also how to tell from Python which path ran.

**Watch:** the estimate holds the input layer's schema `nnz`, an upper bound: after a
data-dependent gene filter (the HVG filter) it is the whole matrix's (570 MB for 71 MB
held at 50k x 2,000 HVGs), so a memory limit may stream when holding would fit. Forcing
the streamed path from Python needs a limit that also shrinks blocks. Tests that count
store passes must subtract `Execution::passes_saved` (`multi_pass.rs`).

**Task 5.15, parallel inside a leaf:** a pass is L2-bandwidth bound, not short of tasks:
each stored value reads a 480-byte row of the dense basis (60 `f64`), so splitting rows
or genes naively stops at ~3x (4 P-cores share one L2). What scales (stage 2.32 s to
0.67 s, and 1 thread 10% faster): row tiles that walk the genes 128 at a time with a
per-row cursor (that range of the basis stays in L1), per-gene sums over fixed 256-gene
ranges, and a leaf's held blocks (5.14's 820-cell parts) folded as one group. Every sum
keeps its order (cells, then each row's genes), so the bits never depend on threads; size
the parts by genes, never by the thread count. Dense products: every thread whose gemm
runs keeps its own packing buffers, so 32-wide parallel groups cost ~20 MB of peak at 10
threads; `pca_dense::multiply` cuts Rayleigh-Ritz's products in 2 fixed halves and the
QR's stay sequential. Do not turn on faer's `rayon` feature: it switches faer's global
default from sequential to the pool's threads for every caller. Per-leaf scratch (`w`
for 4,096 rows) was not the peak's cause: chunking it changed nothing.
Watch: `embedding().to_numpy()` writes the whole plan into a temporary store first
(~0.35 s at 50k x 2,000, barely scaling: 5.17); time a stage with `_execute`.
