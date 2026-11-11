//! One step of the executor's stream: the rows of one cell block (or of a part of it,
//! under a memory limit, task 5.5), in every layer the plan carries at that point.
//!
//! Does: hold the cell block's number, the table position of each row, one
//! [`CellBlock`] per layer (task 5.2) and the cell columns `with_cells` steps computed
//! for its rows (task 5.8), all with the same rows.
//! Does not: hold the stored cell table columns (they are at the rows' table
//! positions in the store), embeddings or graphs (terminals read those, 7.3/7.4), or
//! check that its parts agree (the executor builds them so).

use std::collections::BTreeMap;

use kyto_store::CellBlock;
use polars::prelude::DataFrame;

/// The rows of one cell block in each layer carried by the stream.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineBlock {
    /// The store's cell block `b` the rows come from (format.md §4.1); blocks split
    /// from one cell block under a memory limit share it (task 5.5).
    pub cell_block: u64,
    /// Table position in the store of the cell of each row, increasing.
    pub cells: Vec<u64>,
    /// The rows in each layer, by layer name; every block has `cells.len()` rows over
    /// the same genes.
    pub layers: BTreeMap<String, CellBlock>,
    /// The cell columns computed by the plan's `with_cells` steps (task 5.8), one row
    /// per row of the block; no columns (and `cells.len()` rows) without such steps. A
    /// computed column replaces a stored one of the same name.
    pub columns: DataFrame,
}
