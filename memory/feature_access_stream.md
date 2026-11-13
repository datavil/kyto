# Feature access reads a stream, not a temporary store (task 7.4)

**Works:** a terminal that needs a few cell values (`pull`, `select`, `pull_genes`) adds
them as hidden `with_cells` columns (`__kyto_feature_<i>`, resolved with
`Plan::resolve` first so errors quote the user's feature, not the hidden name), builds
`Output::members(plan, [CellColumn(...)])`, optimizes and runs it through
`run::run_with`, and keeps each `PipelineBlock`'s `cells` (store positions) and
`columns`. Projection pushdown then reads only the layers the features read (a test
deletes the other layer first), and nothing is written. Cell IDs come from the store's
cell table gathered by position; gene results from `BlockStream::genes()` (a
`with_genes` step always yields it). 7.3's terminals instead sink the whole output to a
temporary store; this path is the cheaper one when a terminal reads a few members.

**Works:** counts and IDs need no run when the plan knows them:
`output_schema().n_cells()/n_genes()` are exact without a filter on data, and
`cell_axis_changed_by().is_none()` means the store's cell IDs are the output's.

**Gotcha:** a stream that yields no block (no cell left) gives no column to take a dtype
from: build the empty column from the plan's declared `ColumnDtype`; an `Enum` then has
no categories (deviations.md, "Empty categorical feature").
