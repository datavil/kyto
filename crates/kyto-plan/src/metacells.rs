//! The node that groups cells into metacells by e-means (task 6.16, opt-in).
//!
//! Does: hold the cell embedding grouped (a cell embedding of the plan: `KY-E109`
//! otherwise), how many of its leading components to use, the number of metacells
//! (at least 1, at most the cells reaching the node: `KY-E215`, checked here when the
//! count is known and when the plan runs otherwise), the epochs (1 to
//! [`METACELLS_MAX_EPOCHS`]), the seed and the cell column written. Its output schema
//! adds the cell column `into` as `String` (each cell's metacell, `'0'`, `'1'`, ...,
//! as cluster labels are) and the `float32` cell embedding `<into>_centroids`
//! ([`centroids_name`]): each cell's row is its metacell's centroid, so every sink
//! stores it as any cell embedding. Changes no axis; cross-cell.
//! Does not: group (kyto-exec's `metacells.rs` runs e-means), or get inserted by any
//! other step, default or device rule: it runs only when `metacells` is called
//! (decided by the user, Nov 19, 2026).

use std::fmt;
use std::sync::Arc;

use kyto_core::{Axis, ColumnDtype, ColumnSchema, EmbeddingSchema, FloatDtype};
use kyto_error::KytoError;
use serde::{Deserialize, Serialize};

use crate::names::MemberGroup;
use crate::node::{CellScope, LogicalNode, Transform};
use crate::plan::Plan;
use crate::plan_schema::PlanSchema;

/// The most epochs `metacells` runs: the paper found no gain past 25, and Lloyd's
/// k-means at 25 iterations is the bar it is measured against.
pub const METACELLS_MAX_EPOCHS: u64 = 25;

/// The name of the centroid embedding a `metacells` node writing column `into` adds.
pub fn centroids_name(into: &str) -> String {
    format!("{into}_centroids")
}

/// The parameters of a metacell grouping.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Metacells {
    /// The cell embedding grouped (a `pca` embedding, typically).
    pub embedding: String,
    /// How many leading components to use; `None` uses all.
    pub n_components: Option<u64>,
    /// The number of metacells (e-means' `k`).
    pub n_metacells: u64,
    /// Passes over the cells.
    pub epochs: u64,
    /// The random seed (R7): the initial cells, the batch order and the revivals.
    pub seed: u64,
    /// The cell column written; the centroids go to `<into>_centroids`.
    pub into: String,
}

impl Metacells {
    /// The width of the centroid embedding, given the grouped embedding's width.
    pub fn width(&self, embedding_width: u64) -> u64 {
        self.n_components.unwrap_or(embedding_width)
    }
}

impl LogicalNode for Metacells {
    fn name(&self) -> &'static str {
        "METACELLS"
    }

    fn method(&self) -> &'static str {
        "metacells"
    }

    fn changes_axis(&self) -> Option<Axis> {
        None
    }

    fn cell_scope(&self) -> CellScope {
        CellScope::CrossCell
    }

    fn describe(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}", self.embedding)?;
        if let Some(n_components) = self.n_components {
            write!(formatter, "[:{n_components}]")?;
        }
        write!(
            formatter,
            " -> {:?}, {:?} n_metacells={} epochs={} seed={}",
            self.into,
            centroids_name(&self.into),
            self.n_metacells,
            self.epochs,
            self.seed
        )
    }
}

impl Transform for Metacells {
    fn output_schema(&self, input: &PlanSchema) -> PlanSchema {
        let width = input
            .embedding(&self.embedding)
            .map_or(self.n_components.unwrap_or(1), |embedding| {
                self.width(embedding.width)
            });
        input
            .clone()
            .with_column(
                Axis::Cells,
                ColumnSchema::new(self.into.clone(), ColumnDtype::String),
            )
            .with_embedding(EmbeddingSchema {
                name: centroids_name(&self.into),
                axis: Axis::Cells,
                width,
                dtype: FloatDtype::Float32,
            })
    }
}

impl Plan {
    /// This plan with the cells grouped into `params.n_metacells` metacells by
    /// e-means on `params.embedding`: the cell column `params.into` and the cell
    /// embedding `<into>_centroids`. `seed` is required (R7).
    ///
    /// # Errors
    /// `KY-E109` if the plan has no cell embedding of that name; `KY-E215` for
    /// `n_metacells` of 0 or above the plan's cell count (when it is known),
    /// `n_components` of 0 or above the embedding's width, `epochs` of 0 or above
    /// [`METACELLS_MAX_EPOCHS`], an `into` that breaks format.md §5.6 or is reserved,
    /// or a centroid embedding name that breaks format.md §2.1 or differs from an
    /// embedding of the plan only in case; `KY-E216` if this plan ends in a sink.
    pub fn metacells(self: &Arc<Self>, params: Metacells) -> Result<Arc<Self>, KytoError> {
        const METHOD: &str = "metacells";
        self.check_open(METHOD)?;
        let schema = self.output_schema();
        let embedding = schema
            .embedding(&params.embedding)
            .filter(|embedding| embedding.axis == Axis::Cells)
            .ok_or_else(|| self.missing_member("cell embedding", &params.embedding))?;
        if let Some(n_components) = params.n_components
            && (n_components == 0 || n_components > embedding.width)
        {
            return Err(Self::invalid_argument(
                METHOD,
                "n_components",
                &n_components.to_string(),
                &format!(
                    "embedding '{}' has {} components",
                    embedding.name, embedding.width
                ),
                &format!("an int from 1 to {}, or None for all", embedding.width),
            ));
        }
        let n_cells = schema.n_cells();
        if params.n_metacells == 0 || n_cells.is_some_and(|n| params.n_metacells > n) {
            let (problem, accepted) = match n_cells {
                Some(n) if params.n_metacells > 0 => (
                    format!("the plan has {n} cells, fewer than the metacells asked for"),
                    format!("an int from 1 to {n} (n/50 to n/500 is typical)"),
                ),
                _ => (
                    "a grouping needs at least one metacell".to_owned(),
                    "an int of at least 1, at most the number of cells".to_owned(),
                ),
            };
            return Err(Self::invalid_argument(
                METHOD,
                "n_metacells",
                &params.n_metacells.to_string(),
                &problem,
                &accepted,
            ));
        }
        if params.epochs == 0 || params.epochs > METACELLS_MAX_EPOCHS {
            return Err(Self::invalid_argument(
                METHOD,
                "epochs",
                &params.epochs.to_string(),
                &format!(
                    "e-means runs from 1 to {METACELLS_MAX_EPOCHS} passes over the cells \
                     (3 reach Lloyd's k-means at 25 iterations)"
                ),
                &format!("an int from 1 to {METACELLS_MAX_EPOCHS}"),
            ));
        }
        Self::check_column_name(METHOD, "into", Axis::Cells, &params.into)?;
        let centroids = centroids_name(&params.into);
        self.check_member_name(
            METHOD,
            "into",
            &params.into,
            MemberGroup::Embeddings,
            &centroids,
        )?;
        if centroids == params.embedding {
            return Err(Self::invalid_argument(
                METHOD,
                "into",
                &format!("'{}'", params.into),
                &format!(
                    "its centroid embedding '{centroids}' would replace the embedding grouped"
                ),
                "a name whose '<into>_centroids' is not the input embedding",
            ));
        }
        Ok(self.extend(params, Self::Metacells))
    }
}
