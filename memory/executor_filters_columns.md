# Filters and computed columns in the executor (task 5.8)

**Broke (R11):** evaluating a Polars expression per block gave other bits for a block of
one row: Polars treats a length-1 column as a scalar and runs other float kernels
(`x / c` instead of `x * (1 / c)`), so `score / -2 / 1.43` differed by 1 ulp between
blocks of 8 and of 3 cells. Found only by a 4000-case proptest run, not at 256.
**Works:** `expression.rs` evaluates a one-row frame as two equal rows and keeps the
first value (`one_row_blocks_give_the_same_bits`). Any new per-block Polars evaluation
must go through `Prepared::evaluate`.

**Broke (test oracle):** the 4.4 reference evaluator computed in `f64` and took the
median over all cells; the executor writes `float32` map outputs and takes scanpy's
median over non-zero totals. Ties between layers (equal counts after `log1p`) then
compared differently. **Works:** the shared `support/mod.rs` rounds each map output to
`float32` and skips zero totals; the 4.4 tests are unchanged by it (both runs use it).

**Also:**
- kyto-algo depends on kyto-exec, so the executor cannot call kyto-algo: the per-cell
  QC code (`qc.rs`) moved into kyto-exec, kyto-algo re-exports it. Algorithms that run
  inside the stream belong in kyto-exec (as 5.4's HVG statistics already did).
- Reading an `Enum` column with many categories through `read_table` (Polars
  `ParquetReader`) peaks far above what it holds: pbmc3k `symbol` (32 738 categories)
  +54 MB peak for 5 MB held; `gene_id` (String) +2.5 MB. Gene steps read the gene table,
  so this lands in their peak memory; `estimate()` does not model table reads (task
  5.10).
- Linking Polars' lazy engine into kyto-exec raised the test binaries' baseline RSS by
  about 5 MB on macOS (more code kept by the linker), which took 5.5's RSS check past
  its margin; the user raised `ALLOCATOR_OVERHEAD_PERCENT` to 100 (Nov 12, 2026).
- `uint32` leaves (matrix reads, detected genes) enter Polars as `Float64` in both
  `kyto_plan::column_dtype`'s stand-ins and the executor (`cell_steps::widen`,
  `gene_steps::dense_row`), so arithmetic does not wrap; a plain read is cast back to
  `UInt32`. Keep the two in step: the declared dtype comes from the stand-ins.
- `take_cells` counts the cells it meets per stream (`Operator::fresh` gives every
  stream, statistics passes included, new counters) and errors at the stream's end.
