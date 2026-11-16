//! `peek` (task 8.3): a plan's cell table as an output that metadata alone answers.
//!
//! Does: build the output that reads every cell column of a plan, optimize it, and
//! check that no node left in it reads anything but table columns (no layer, embedding
//! or graph), so running it reads the store's tables and no matrix tile.
//! Does not: run anything or cut the rows (the caller keeps the first `n` cells), or
//! answer for a plan whose cell table needs data (`KY-E219`). The check is on what the
//! optimized plan reads: a gene filter on data is refused although it changes no cell
//! column, since the optimizer keeps it.

use std::sync::Arc;

use kyto_error::KytoError;

use crate::members::Member;
use crate::node::LogicalNode;
use crate::optimize::{OptimizeOptions, Output, optimize};
use crate::optimized_plan::{Operation, OptimizedNode};
use crate::plan::Plan;

const CELL_ID: &str = "cell_id";

/// Whether `member` is a column of the cell or gene table.
fn is_table_column(member: &Member) -> bool {
    matches!(member, Member::CellColumn(_) | Member::GeneColumn(_))
}

/// The plan method that built `node`, as error messages name it.
fn method(node: &OptimizedNode) -> &'static str {
    match node.operation() {
        Operation::Scan { plan, .. } | Operation::Node(plan) => plan.node().method(),
        Operation::Fused { maps, .. } => maps.first().map_or("log1p", LogicalNode::method),
    }
}

impl Plan {
    /// The output `peek` runs: every column of the plan's cell table (the cell IDs come
    /// along), read from the store's tables alone. Reads nothing itself.
    ///
    /// # Errors
    /// `KY-E219` if a step the cell table depends on reads a layer, an embedding or a
    /// graph (a QC column, a filter on expression values, a clustering), naming the
    /// step and what it reads.
    pub fn peek_output(self: &Arc<Self>) -> Result<Output, KytoError> {
        let columns = self
            .schema()
            .cells
            .columns
            .iter()
            .filter(|column| column.name != CELL_ID)
            .map(|column| Member::CellColumn(column.name.clone()));
        let output = Output::members(self, columns)?;
        let optimized = optimize(std::slice::from_ref(&output), OptimizeOptions::ALL);
        // From the top down, so the step named is the one that reads the data, not the
        // scan that reads it for that step.
        for node in optimized.nodes().iter().rev() {
            let reads = match node.operation() {
                Operation::Scan { reads, .. } => reads.clone(),
                Operation::Node(_) | Operation::Fused { .. } => node.reads(),
            };
            if let Some(member) = reads.iter().find(|member| !is_table_column(member)) {
                return Err(KytoError::PeekNeedsData {
                    step: method(node).to_owned(),
                    member: member.to_string(),
                });
            }
        }
        Ok(output)
    }
}
