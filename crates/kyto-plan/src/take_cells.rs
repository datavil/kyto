//! The node that keeps the cells named by ID, in the order given.
//!
//! Does: hold the cell IDs, each checked when built to be non-empty (`KY-E210`) and
//! named once, and no more of them than the plan's bound on cells (`KY-E215`).
//! Changes the cell axis: its output schema has exactly that many cells, every layer and cell graph `nnz` becomes a bound, and stored derived
//! gene columns are stale downstream (`KY-E301`). Per-cell: each kept cell is copied
//! as it is.
//! Does not: check that the cells exist. The plan holds no cell IDs (millions of rows,
//! task 3.5), so an unknown or filtered-out ID is the executor's error (5.x).

use std::fmt;
use std::sync::Arc;

use kyto_core::Axis;
use kyto_error::KytoError;
use serde::{Deserialize, Serialize};

use crate::node::{CellScope, LogicalNode, Transform};
use crate::plan::Plan;
use crate::plan_schema::PlanSchema;
use crate::select_genes::{check_unique, write_ids};

/// Keeps the cells named by ID, in the order given.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TakeCells {
    cell_ids: Arc<[String]>,
}

impl TakeCells {
    /// The cell IDs, in output order.
    pub fn cell_ids(&self) -> &[String] {
        &self.cell_ids
    }
}

impl LogicalNode for TakeCells {
    fn name(&self) -> &'static str {
        "TAKE_CELLS"
    }

    fn method(&self) -> &'static str {
        "take_cells"
    }

    fn changes_axis(&self) -> Option<Axis> {
        Some(Axis::Cells)
    }

    fn cell_scope(&self) -> CellScope {
        CellScope::PerCell
    }

    fn describe(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_ids(formatter, &self.cell_ids)
    }
}

impl Transform for TakeCells {
    fn output_schema(&self, input: &PlanSchema) -> PlanSchema {
        let n_cells = self.cell_ids.len() as u64;
        input
            .clone()
            .with_axis_changed(Axis::Cells, Some(n_cells), n_cells)
    }
}

impl Plan {
    /// This plan keeping only the cells `cell_ids`, in that order (a new plan; `self`
    /// is unchanged). Stored derived gene columns are stale downstream (`KY-E301`).
    ///
    /// # Errors
    /// `KY-E210` for an empty ID; `KY-E215` for an ID listed twice or more IDs than
    /// the plan's bound on cells; `KY-E216` if this plan ends in a sink.
    pub fn take_cells(
        self: &Arc<Self>,
        cell_ids: impl IntoIterator<Item = String>,
    ) -> Result<Arc<Self>, KytoError> {
        self.check_open("take_cells")?;
        let cell_ids: Arc<[String]> = cell_ids.into_iter().collect();
        if cell_ids.iter().any(String::is_empty) {
            return Err(KytoError::UnknownName {
                kind: "cell".to_owned(),
                name: String::new(),
                expression: "take_cells([..., '', ...])".to_owned(),
                problem: "cell IDs are never empty".to_owned(),
            });
        }
        check_unique("take_cells", "cell_ids", &cell_ids)?;
        let bound = self.output_schema().axis_len_bound(Axis::Cells);
        if cell_ids.len() as u64 > bound {
            return Err(Self::invalid_argument(
                "take_cells",
                "cell_ids",
                &format!("[{} IDs]", cell_ids.len()),
                &format!("the plan has at most {bound} cells, so some IDs name no cell"),
                &format!("at most {bound} distinct cell IDs"),
            ));
        }
        Ok(self.extend(TakeCells { cell_ids }, Self::TakeCells))
    }
}
