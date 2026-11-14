# Per-gene QC metrics and `log1p` on expressions (task 6.13)

**Works:** a per-gene metric over the plan's cells is a gene-axis `Source` leaf
(`Source::GeneQc { layer, metric }`), like 6.1's per-cell `Source::Qc`, not a plan node.
A gene step whose expression is not metadata-only was already `CellScope::CrossCell`
with one statistics pass (built for `cell_profile`), so the pass count, the block on
pushing a cell filter below it and the layer read came for free. The gene step's one
pass (`gene_steps::collect_inputs`) gathers its cell profiles and, per layer, the
detected cells and `f64` sums of every gene present (`gene_qc::GeneCounts`).

**Works:** `reduction::reduce_blocks` is 5.6's `reduce_pass` made generic over the leaf
(new leaf, fold a block, combine): any new reduction over the stream should call it
instead of looping over windows itself (same bits on any thread count and block size).

**Breaks (test oracle):** the 4.4 reference evaluator runs every step; the optimizer
leaves out a step whose outputs nothing reads (deviations.md "Unused steps"). A random
gene filter on data in `filters_columns`' random chains removes genes often, and a
`with_cells` that reads a removed gene but is overwritten later then fails in the
reference only. So filters on QC metrics are fixed cases there (all 16 rule
combinations); only `with_genes` columns of the metrics are in the random chains.

**Also:**
- Polars orders `NaN` above every number: `percent_dropout() >= 0` keeps the genes of a
  plan with no cell left (`NaN`), `< 200` drops them (deviations.md "`NaN` in float
  metadata").
- `Expr::log1p` lowers to Polars `log1p` (feature `log` on kyto-core's polars):
  `Float32` stays `Float32`, every other number becomes `Float64`, null stays null.
  `column_dtype` asks Polars, so the declared dtype follows without a table.
- `uv run cargo ...` syncs first and rebuilds the extension; while the tree does not
  compile use `uv sync --no-install-project` once (cmake for the HDF5 build), then
  `uv run --no-sync cargo check/test`.
