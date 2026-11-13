//! Reading the cell and gene tables in the cost estimate (task 5.10).
//!
//! Does: tell which of the store's tables a plan's run reads, as the executor (tasks
//! 5.8, 5.9) reads them, and the bytes they hold and take while read, from
//! `schema.json`'s columns alone ([`TableReads`]). The executor reads a whole table
//! (every column) once per run, before the stream starts, and keeps it until the run
//! ends:
//!
//! - the cell table, for a cell filter, `take_cells`, `with_cells`, a gene step over a
//!   cell's profile (it finds the cell's position) and a sink;
//! - the gene table (5.8's `Genes`), for a gene filter, `select_genes`, `with_genes`,
//!   a cell step reading one gene's values or a QC metric over a gene set, and a sink.
//!
//! A column in memory (Polars) holds per value its width (`column_value_bytes`), a
//! `String` [`ASSUMED_STRING_BYTES`] (a 16-byte view and its text, whose length
//! `schema.json` does not record), an `Enum` a 4-byte code, and an `Enum` its
//! categories once, [`ASSUMED_CATEGORY_BYTES`] each. Reading a table (kyto-store's
//! `read_table`, columns in parallel) takes for a moment half again its largest
//! column, or of all its `Enum` columns if more (they decode at once, each building
//! its categories again; bug-hunter: 16 columns of 32 738 categories peaked at 2.2x
//! what they hold), on top of what it then holds. Measured on pbmc3k's gene table (32 738 genes, `gene_id`,
//! `symbol` with 32 738 categories): 6.2 MB held, 13.4 MB at most while read.
//! Does not: count a sink's second read of the tables (the sink holds the run's
//! frames and the kept positions; an upper bound is to count the tables once more, not
//! done), the table a gene filter builds next to the one it replaces, or the columns a
//! block takes from the cell table (a block's rows).

use kyto_core::{ColumnDtype, ExprNode, Source, TableSchema};

use crate::estimate::column_value_bytes;
use crate::filter::Predicate;
use crate::members::for_each_leaf;
use crate::optimized_plan::{Operation, OptimizedNode};
use crate::plan::Plan;
use crate::with_columns::Columns;

/// Bytes a `String` table value holds in memory the estimate assumes: Polars' 16-byte
/// view and its text (pbmc3k's IDs: 35 to 37 bytes a value). `schema.json` does not
/// record string lengths.
pub const ASSUMED_STRING_BYTES: u64 = 40;

/// Bytes one category of an `Enum` table column holds in memory the estimate assumes:
/// its text, view and Polars' lookup from text to code (pbmc3k's `symbol`: 32 738
/// categories hold 5.0 MB, 150 bytes each).
pub const ASSUMED_CATEGORY_BYTES: u64 = 160;

/// The tables a plan's run reads: what they hold until the run ends, and the most
/// reading one of them takes for a moment on top.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TableReads {
    /// Bytes of every table read, held until the run ends.
    pub(crate) held: u128,
    /// Bytes reading the largest one takes on top of what it holds, for a moment
    /// before the stream starts.
    pub(crate) reading: u128,
}

impl TableReads {
    /// The tables the optimized plan's `nodes` read (the scan's store's).
    pub(crate) fn new(nodes: &[OptimizedNode]) -> Self {
        let mut cells = false;
        let mut genes = false;
        let mut store = None;
        for node in nodes {
            match node.operation() {
                Operation::Scan { plan, .. } => store = Some(plan.root().schema()),
                Operation::Fused { .. } => {}
                Operation::Node(plan) => {
                    let (reads_cells, reads_genes) = node_tables(plan);
                    cells |= reads_cells;
                    genes |= reads_genes;
                }
            }
        }
        let Some(store) = store else {
            return Self::default();
        };
        let mut reads = Self::default();
        for (read, table, rows) in [
            (cells, &store.cells, store.n_cells),
            (genes, &store.genes, store.n_genes),
        ] {
            if !read {
                continue;
            }
            let (held, transient) = table_bytes(table, rows);
            reads.held += held;
            reads.reading = reads.reading.max(transient);
        }
        reads
    }
}

/// Whether the logical node `plan` makes the executor read the cell table and the
/// gene table.
fn node_tables(plan: &Plan) -> (bool, bool) {
    match plan {
        Plan::Filter(node) => match node.params().predicate() {
            Predicate::Cells(predicate) => (true, reads_gene_values(predicate.node())),
            Predicate::Genes(predicate) => (reads_profiles(predicate.node()), true),
        },
        Plan::WithColumns(node) => match node.params().columns() {
            Columns::Cells(columns) => (
                true,
                columns
                    .iter()
                    .any(|(_, expr)| reads_gene_values(expr.node())),
            ),
            Columns::Genes(columns) => (
                columns.iter().any(|(_, expr)| reads_profiles(expr.node())),
                true,
            ),
        },
        Plan::TakeCells(_) => (true, false),
        Plan::SelectGenes(_) => (false, true),
        Plan::Sink(_) => (true, true),
        Plan::Scan(_)
        | Plan::WithDefaultLayer(_)
        | Plan::MapElementwise(_)
        | Plan::CrossCell(_)
        | Plan::Neighbors(_)
        | Plan::Cluster(_)
        | Plan::Umap(_) => (false, false),
    }
}

/// Whether a cell expression finds a gene among the genes present: one gene's
/// values, or a QC metric over a gene set.
fn reads_gene_values(root: &ExprNode) -> bool {
    any_leaf(root, |source| match source {
        Source::Expression { .. } => true,
        Source::Qc { metric, .. } => metric.genes().is_some(),
        _ => false,
    })
}

/// Whether a gene expression finds a cell in the cell table: one cell's profile.
fn reads_profiles(root: &ExprNode) -> bool {
    any_leaf(root, |source| matches!(source, Source::Profile { .. }))
}

fn any_leaf(root: &ExprNode, test: impl Fn(&Source) -> bool) -> bool {
    let mut found = false;
    for_each_leaf(root, &mut |source| found |= test(source));
    found
}

/// Bytes a table of `rows` rows holds in memory, and what reading it takes on top for
/// a moment: half again its largest column, or of all its `Enum` columns if more
/// (columns decode in parallel, and each `Enum` column builds its categories again
/// while read; an upper bound, as the thread count is not a plan argument).
fn table_bytes(table: &TableSchema, rows: u64) -> (u128, u128) {
    let rows = u128::from(rows);
    let (held, largest, enums) =
        table
            .columns
            .iter()
            .fold((0, 0, 0), |(held, largest, enums), column| {
                let (bytes, is_enum) = match column.dtype {
                    ColumnDtype::String => (rows * u128::from(ASSUMED_STRING_BYTES), false),
                    ColumnDtype::Enum { n_categories, .. } => (
                        rows * 4 + u128::from(n_categories) * u128::from(ASSUMED_CATEGORY_BYTES),
                        true,
                    ),
                    ref dtype => (rows * column_value_bytes(dtype), false),
                };
                (
                    held + bytes,
                    largest.max(bytes),
                    enums + if is_enum { bytes } else { 0 },
                )
            });
    (held, largest.max(enums) * 3 / 2)
}
