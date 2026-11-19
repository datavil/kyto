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
