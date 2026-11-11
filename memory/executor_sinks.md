# Sinks in the executor (task 5.9)

**Works:** a sink drains the output's stream into `StoreWriter` (one `push` per block,
layers in name order) and builds the tables at the end: the stored cell table `take`n
at the kept positions plus the blocks' computed columns stacked, the gene table from
`BlockStream::genes()` (or the stored one). `.h5ad` = the same store written to a
hidden `.kyto-sink-*` temporary next to the target, then 2.7's `sink_h5ad`; both
writers rename on success, so a cancel (`KY-E503`) or error leaves no target.

**Works (stale columns):** do not copy the lineage rules. Ask the plan:
`plan.resolve(&CellExpr::column(name))` failing with `KY-E301` at the sink node is
exactly "stale here"; such columns are dropped, the others keep their lineage
(`StoreWriter::set_lineage`). Columns the plan computes have no lineage in the output
schema, so a `with_cells` column replacing a stored one loses it.

**Broke (borrow):** `for block in stream.by_ref()` borrows the stream mutably, so
`stream.recycle(block)` inside the loop does not compile; use
`while let Some(block) = stream.next()`.

**Also:** the plan's sink node has no `overwrite`; it is an argument of
`Pipeline::sink` (R7), and kyto-py's `_execute` passes `false` until 7.3. An empty
output has no computed-column frame: those columns are written empty with the dtype
the plan declares. Check the target (exists, is or lies inside the scanned store)
before running: `StoreWriter::create` can only check it after the statistics passes.
