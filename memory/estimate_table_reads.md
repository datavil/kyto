# Table reads, Enum columns and the block pool in the estimate (task 5.10)

**Broke:** Polars' `ParquetReader` picks `ParallelStrategy::Auto`, which decodes row
groups in parallel when a file has more row groups than columns (every kyto table:
one row group per tile block). Each row group of an `Enum` column then builds a lookup
of all its categories at once: pbmc3k's `symbol` (32 738 categories, 32 row groups)
peaked at ~50 MB to hold 5 MB. **Works:** kyto-store's `table_reader` reads with
`ParallelStrategy::Columns`: the gene table peaks at 13.4 MB for 6.2 MB held (2.16x,
asserted in `estimate_fixtures.rs`); `symbol` alone 12.2 MB for 5.0 MB (2.4x: Polars
builds the categories from the file's Arrow schema, a transient kyto cannot avoid).
`ParallelStrategy::None` and `set_low_memory` measured the same.

**In memory (Polars):** an `Enum` category costs ~150 bytes (text, view, lookup), a
`String` value ~36 bytes for 15-character IDs; categories are shared between frames of
the same column. The estimate assumes 40 and 160 (`ASSUMED_STRING_BYTES`,
`ASSUMED_CATEGORY_BYTES`).

**Broke (estimate):** filter plans measured up to 2x their estimate at small blocks
although tables were only 6 MB: kyto-exec's block pool keeps `(window + 1) x (layers +
operators)` buffers and every cell step (even `with_cells`, or a filter keeping every
cell) is an operator; the reader never takes from the pool, so over a long stream the
pool fills. **Works:** count pool entries per operator, and bound the free buffers by
what earlier blocks allocated (the first window all its buffers, later blocks only the
reader's) minus what the consumer holds and the later window takes back.

**How it was found:** one `peak_alloc` probe test running variants of the plan (scan,
QC column only, cell filter that keeps all, one that drops cells, gene filter) at 64,
256, 1024 and 4096 cells a block: a fixed cost per step that grows with block size
pointed at the pool, not the copy.
