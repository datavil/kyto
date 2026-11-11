# Deterministic reductions: leaves per store cell block, a fixed tree

Task 5.6. Any reduction across blocks (statistics passes today; batches, `scale`,
`pca`, per-gene columns later) goes through `kyto_exec`'s `reduce_pass`
(`reduction.rs`), never a fold on the consumer or in thread-finish order.

- **Leaf = one store cell block**, folded in cell order from an empty leaf. Parts of a
  cell block split by a memory limit continue the same leaf (sequentially, even across
  windows), so sums do not depend on `max_block_cells`. A leaf per *stream block*
  would make HVG sums change with the memory limit, and the fit reads the process
  RSS, so the same call could give other bits run to run.
- **Tree = binary counter** (`BlockOrderTree`): shape fixed by the number of leaves,
  at most `log2(n) + 1` subtrees held. Count those leaves in the estimate.
- Results still change with the store's cell block size (another tiling): that is
  the input, so R11 holds; say so wherever a test compares tilings.
- Testing it: fingerprint every bit independent of block splitting (hash the layer
  name per row, not per block, or block sizes change the hash), use values spanning
  many orders of magnitude so order changes show, delay storage reads per run to
  shuffle finish order, and mutation-check (push leaves in finish order: tests fail).
