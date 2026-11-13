//! The node that clusters cells on a graph: Leiden or Louvain.
//!
//! Does: hold the method, the graph (a cell graph of the plan: `KY-E109` otherwise),
//! the resolution (positive and finite: `KY-E215`), the seed and the output column.
//! Its output schema adds the cell column `into` (replacing one of that name) as
//! `String`, with no lineage: scanpy stores cluster labels as categorical strings
//! (`'0'`, `'1'`, ...), and how many there are is known only when the plan runs, so
//! the `Enum` category count waits for the sink. Changes no axis; cross-cell.
//! Does not: cluster (kyto-exec's `leiden.rs`, task 6.8, which hands the column to the
//! stream as an `Enum` of the clusters' numbers).

use std::fmt;
use std::sync::Arc;

use kyto_core::{Axis, ColumnDtype, ColumnSchema};
use kyto_error::KytoError;
use serde::{Deserialize, Serialize};

use crate::node::{CellScope, LogicalNode, Transform};
use crate::plan::Plan;
use crate::plan_schema::PlanSchema;

/// The community detection method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterMethod {
    /// Leiden.
    Leiden,
    /// Louvain.
    Louvain,
}

impl ClusterMethod {
    /// The method's name, also the plan method and scanpy's default column.
    pub fn name(self) -> &'static str {
        match self {
            Self::Leiden => "leiden",
            Self::Louvain => "louvain",
        }
    }
}

/// The parameters of a clustering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cluster {
    /// Leiden or Louvain.
    pub method: ClusterMethod,
    /// The cell graph clustered (scanpy's `obsp['connectivities']` by default).
    pub graph: String,
    /// Higher values give more clusters.
    pub resolution: f64,
    /// The random seed (R7).
    pub seed: u64,
    /// The cell column written (scanpy's `key_added`).
    pub into: String,
}

impl LogicalNode for Cluster {
    fn name(&self) -> &'static str {
        match self.method {
            ClusterMethod::Leiden => "LEIDEN",
            ClusterMethod::Louvain => "LOUVAIN",
        }
    }

    fn method(&self) -> &'static str {
        self.method.name()
    }

    fn changes_axis(&self) -> Option<Axis> {
        None
    }

    fn cell_scope(&self) -> CellScope {
        CellScope::CrossCell
    }

    fn describe(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?} -> {:?} resolution={} seed={}",
            self.graph, self.into, self.resolution, self.seed
        )
    }
}

impl Transform for Cluster {
    fn output_schema(&self, input: &PlanSchema) -> PlanSchema {
        input.clone().with_column(
            Axis::Cells,
            ColumnSchema::new(self.into.clone(), ColumnDtype::String),
        )
    }
}

impl Plan {
    /// This plan with the cells clustered on `params.graph` into cell column
    /// `params.into`.
    ///
    /// # Errors
    /// `KY-E109` if the plan has no cell graph of that name; `KY-E215` for a
    /// resolution that is not positive and finite, or an `into` that breaks
    /// format.md §5.6 (empty, or with `:`) or is reserved (`cell_id`, `gene_id`);
    /// `KY-E216` if this plan ends in a sink.
    pub fn cluster(self: &Arc<Self>, params: Cluster) -> Result<Arc<Self>, KytoError> {
        let method = params.method.name();
        self.check_open(method)?;
        self.output_schema()
            .graph(&params.graph)
            .filter(|graph| graph.axis == Axis::Cells)
            .ok_or_else(|| self.missing_member("cell graph", &params.graph))?;
        Self::check_positive(method, "resolution", params.resolution)?;
        Self::check_column_name(method, "into", Axis::Cells, &params.into)?;
        Ok(self.extend(params, Self::Cluster))
    }
}
