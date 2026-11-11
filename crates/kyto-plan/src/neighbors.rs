//! The node that builds the k-nearest-neighbour graphs of the cells in an embedding.
//!
//! Does: hold the embedding (a cell embedding of the plan: `KY-E109` otherwise), how
//! many of its components to use, `n_neighbors` (counting the cell itself, as scanpy
//! does), the metric, exact or approximate search, the seed and scanpy's `key_added`.
//! Its output schema adds two `float32` cell graphs, replacing graphs of the same name:
//! `distances` and `connectivities`, or `<key>_distances` and `<key>_connectivities`.
//! Their `nnz` are bounds: `n_cells x (n_neighbors - 1)` distances, and twice that
//! for the symmetrised connectivities. Changes no axis; cross-cell (every cell's
//! neighbours are searched among all cells).
//! Does not: search (6.6) or weight (6.7), or build from a layer: name an embedding
//! (scanpy falls back to `X` or computes a PCA; kyto asks, R7).

use std::fmt;
use std::sync::Arc;

use kyto_core::{Axis, FloatDtype, GraphSchema};
use kyto_error::KytoError;
use serde::{Deserialize, Serialize};

use crate::node::{CellScope, LogicalNode, Transform};
use crate::plan::Plan;
use crate::plan_schema::PlanSchema;

/// The distance between two cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Euclidean distance (scanpy's default).
    Euclidean,
    /// One minus the cosine similarity.
    Cosine,
}

/// How neighbours are searched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeighborSearch {
    /// Exact blocked search.
    Exact,
    /// Approximate search (HNSW).
    Approximate,
}

/// The parameters of a neighbour search.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Neighbors {
    /// The cell embedding searched.
    pub embedding: String,
    /// How many leading components to use; `None` uses all.
    pub n_components: Option<u64>,
    /// Neighbours per cell, the cell itself included.
    pub n_neighbors: u64,
    /// The distance.
    pub metric: Metric,
    /// Exact or approximate search.
    pub search: NeighborSearch,
    /// The random seed (R7).
    pub seed: u64,
    /// The prefix of the graph names (scanpy's `key_added`); `None` for none.
    pub key: Option<String>,
}

impl Neighbors {
    /// The names of the distance and the connectivity graph it writes.
    pub fn graph_names(&self) -> (String, String) {
        match &self.key {
            Some(key) => (format!("{key}_distances"), format!("{key}_connectivities")),
            None => ("distances".to_owned(), "connectivities".to_owned()),
        }
    }
}

impl LogicalNode for Neighbors {
    fn name(&self) -> &'static str {
        "NEIGHBORS"
    }

    fn method(&self) -> &'static str {
        "neighbors"
    }

    fn changes_axis(&self) -> Option<Axis> {
        None
    }

    fn cell_scope(&self) -> CellScope {
        CellScope::CrossCell
    }

    fn describe(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (distances, connectivities) = self.graph_names();
        write!(formatter, "{:?}", self.embedding)?;
        if let Some(n_components) = self.n_components {
            write!(formatter, "[:{n_components}]")?;
        }
        write!(
            formatter,
            " -> {distances:?}, {connectivities:?} n_neighbors={} metric={:?} search={:?} seed={}",
            self.n_neighbors, self.metric, self.search, self.seed
        )
    }
}

impl Transform for Neighbors {
    fn output_schema(&self, input: &PlanSchema) -> PlanSchema {
        let (distances, connectivities) = self.graph_names();
        let per_cell = self.n_neighbors.saturating_sub(1);
        let nnz = input.axis_len_bound(Axis::Cells).saturating_mul(per_cell);
        let graph = |name: String, nnz: u64| GraphSchema {
            name,
            axis: Axis::Cells,
            dtype: FloatDtype::Float32,
            nnz,
        };
        input
            .clone()
            .with_graph(graph(distances, nnz), false)
            .with_graph(graph(connectivities, nnz.saturating_mul(2)), false)
    }
}

impl Plan {
    /// This plan with the neighbour graphs of the cells in `params.embedding`.
    ///
    /// # Errors
    /// `KY-E109` if the plan has no cell embedding of that name; `KY-E215` for
    /// `n_neighbors` below 2, `n_components` of 0 or above the embedding's width, or
    /// an empty `key`; `KY-E216` if this plan ends in a sink.
    pub fn neighbors(self: &Arc<Self>, params: Neighbors) -> Result<Arc<Self>, KytoError> {
        self.check_open("neighbors")?;
        let embedding = self
            .output_schema()
            .embedding(&params.embedding)
            .filter(|embedding| embedding.axis == Axis::Cells)
            .ok_or_else(|| self.missing_member("cell embedding", &params.embedding))?;
        if params.n_neighbors < 2 {
            return Err(Self::invalid_argument(
                "neighbors",
                "n_neighbors",
                &params.n_neighbors.to_string(),
                "it counts the cell itself, so below 2 a cell has no neighbour",
                "an int of at least 2",
            ));
        }
        if let Some(n_components) = params.n_components
            && (n_components == 0 || n_components > embedding.width)
        {
            return Err(Self::invalid_argument(
                "neighbors",
                "n_components",
                &n_components.to_string(),
                &format!(
                    "embedding '{}' has {} components",
                    embedding.name, embedding.width
                ),
                &format!("an int from 1 to {}, or None for all", embedding.width),
            ));
        }
        if let Some(key) = &params.key {
            Self::check_name("neighbors", "key", key)?;
        }
        Ok(self.extend(params, Self::Neighbors))
    }
}
