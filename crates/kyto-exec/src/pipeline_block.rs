//! One step of the executor's stream: the rows of one cell block (or of a part of it,
//! under a memory limit, task 5.5), in every layer the plan carries at that point.
//!
//! Does: hold the cell block's number, the table position of each row and one
//! [`CellBlock`] per layer, all with the same rows (task 5.2).
//! Does not: hold table columns, embeddings or graphs (terminals read those, 7.3/7.4),
//! or check that its layers agree (the executor builds them so).

use std::collections::BTreeMap;

use kyto_store::CellBlock;

/// The rows of one cell block in each layer carried by the stream.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineBlock {
    /// The store's cell block `b` the rows come from (format.md §4.1); blocks split
    /// from one cell block under a memory limit share it (task 5.5).
    pub cell_block: u64,
    /// Table position of the cell of each row, increasing.
    pub cells: Vec<u64>,
    /// The rows in each layer, by layer name; every block has `cells.len()` rows over
    /// the same genes.
    pub layers: BTreeMap<String, CellBlock>,
}
