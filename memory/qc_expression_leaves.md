# QC metrics are expression leaves (task 6.1)

**Works:** a per-cell metric computed from the matrix is a `Source` leaf
(`Source::Qc { layer, metric }`), not a plan node: it resolves at plan build (layer
named, gene set checked, R6), goes into `with_cells`/`filter_cells` like any numeric
expression, round-trips through plan JSON (`ExprEntry::Qc`, gene set as a nested
`ExprJson`), and is computed per block by `kyto_algo::CellQc::evaluate`, so it follows
whatever rows and genes the stream carries (never stale). The gene set is a
metadata-only `GeneExpr` evaluated with Polars on `genes.parquet` when the run starts.

**Breaks:** a `Source` can now hold a nested `GeneExpr`, whose literals are floats, so
`Source` is `PartialEq` only (Eq/Hash dropped). Code matching on `Source` must handle
`Qc` (reads its layer and the gene set's gene columns, `members.rs`).

**Same bits:** a gene-set sum is the row-sum kernel over the row with other genes'
values zeroed, so each value keeps its lane and a set of every gene equals
`total_counts` bit for bit. Percent is `in / total * 100` as scanpy (0/0 = NaN).

**Not wired:** nothing in the executor evaluates expressions yet (`with_cells` is
`KY-E601`, 5.2 Open (1)); the 6.1 parity test (`crates/kyto-algo/tests/qc_parity.rs`)
streams the layer and calls `CellQc` itself.
