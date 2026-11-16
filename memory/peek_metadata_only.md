# `peek`: a metadata-only run is decided on the optimized plan (task 8.3)

**Works:** whether a plan's cell table can be read from the tables alone is asked of the
optimized plan, not of the logical one: build `Output::members` over the cell columns,
`optimize` it (projection pushdown leaves out `log1p`, `pca`, `scale`), and refuse if
any node left reads a `Layer`, `Embedding` or `Graph` (`Plan::peek_output`, `KY-E219`;
walk the nodes from the top so the step named is the reader, not the scan). The zero
tile-read check is `Pipeline::run_on` over a storage wrapper that records every key
(`crates/kyto-exec/tests/it/inspection.rs`), plus, from Python, deleting the store's
`layers`, `embeddings` and `graphs` folders before the calls.

**Watch:** a stream's `PipelineBlock::columns` holds computed columns only. A stored
column is read like `select` reads it: a hidden `with_cells` column of
`CellExpr::column(name)` (`cell_frame`). Leave out columns whose `Plan::resolve` gives
`KY-E301`: sinks and `cells.to_polars()` drop stale columns, `names()` and `schema()`
still list them.

**Breaks:** stopping the stream after `n` cells. `take_cells` of a cell that an earlier
filter removed is only known at the end of the run (`KY-E210`), so an early stop
returned rows for a plan that cannot run (bug-hunter). Run the tables-only stream to
its end and cut the frame.

**Watch:** the optimizer keeps a gene filter on data (`filter(cells_detected() >= 3)`)
in a cell-columns output, so `peek` refuses after it although no cell column depends on
it. Dropping gene-axis steps from such an output is an optimizer rule, not done.
