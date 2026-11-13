//! The node that lays out a cell graph in a few dimensions: UMAP.
//!
//! Does: hold scanpy's `tl.umap` arguments: the graph (a cell graph of the plan:
//! `KY-E109` otherwise), `n_components`, `min_dist` and `spread` (the curve the layout
//! fits), the epochs (`None`: scanpy's 500 up to 10,000 cells, 200 above), the learning
//! rate `alpha`, the repulsion `gamma`, the negative samples per edge, the
//! initialisation (spectral or random), the seed and the cell embedding written.
//! Its output schema adds the `float32` cell embedding `into`, `n_components` wide,
//! replacing one of that name. Changes no axis; cross-cell.
//! Does not: lay out (6.9 runs it in kyto-exec), or start from another embedding
//! (scanpy's `init_pos` naming an `obsm` key or `paga`).

use std::fmt;
use std::sync::Arc;

use kyto_core::{Axis, EmbeddingSchema, FloatDtype};
use kyto_error::KytoError;
use serde::{Deserialize, Serialize};

use crate::names::MemberGroup;
use crate::node::{CellScope, LogicalNode, Transform};
use crate::plan::Plan;
use crate::plan_schema::PlanSchema;

/// The most dimensions a UMAP layout has: every cell holds that many positions, and
/// umap is a low-dimensional layout (a larger value aborted on allocation).
pub const UMAP_MAX_COMPONENTS: u64 = 100;

/// Where the layout starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UmapInit {
    /// The graph's spectral layout (scanpy's default).
    Spectral,
    /// Uniform random positions.
    Random,
}

/// The parameters of a UMAP layout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Umap {
    /// The cell graph laid out (scanpy's `obsp['connectivities']` by default).
    pub graph: String,
    /// The dimensions of the layout.
    pub n_components: u64,
    /// The smallest distance between points in the layout.
    pub min_dist: f64,
    /// The scale of the layout's distances (with `min_dist`).
    pub spread: f64,
    /// The epochs of the optimisation; `None` for scanpy's default (`maxiter`).
    pub n_epochs: Option<u64>,
    /// The initial learning rate.
    pub alpha: f64,
    /// The weight of the negative samples (repulsion).
    pub gamma: f64,
    /// Negative samples per positive sample.
    pub negative_sample_rate: u64,
    /// Where the layout starts.
    pub init: UmapInit,
    /// The random seed (R7).
    pub seed: u64,
    /// The cell embedding written (scanpy's `X_umap`).
    pub into: String,
}

impl Default for Umap {
    /// scanpy's defaults: the `connectivities` graph into `X_umap`, seed 0.
    fn default() -> Self {
        Self {
            graph: "connectivities".to_owned(),
            n_components: 2,
            min_dist: 0.5,
            spread: 1.0,
            n_epochs: None,
            alpha: 1.0,
            gamma: 1.0,
            negative_sample_rate: 5,
            init: UmapInit::Spectral,
            seed: 0,
            into: "X_umap".to_owned(),
        }
    }
}

impl LogicalNode for Umap {
    fn name(&self) -> &'static str {
        "UMAP"
    }

    fn method(&self) -> &'static str {
        "umap"
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
            "{:?} -> {:?} n_components={} min_dist={} spread={} n_epochs=",
            self.graph, self.into, self.n_components, self.min_dist, self.spread
        )?;
        match self.n_epochs {
            Some(n_epochs) => write!(formatter, "{n_epochs}")?,
            None => write!(formatter, "auto")?,
        }
        write!(
            formatter,
            " alpha={} gamma={} negative_sample_rate={} init={:?} seed={}",
            self.alpha, self.gamma, self.negative_sample_rate, self.init, self.seed
        )
    }
}

impl Transform for Umap {
    fn output_schema(&self, input: &PlanSchema) -> PlanSchema {
        input.clone().with_embedding(EmbeddingSchema {
            name: self.into.clone(),
            axis: Axis::Cells,
            width: self.n_components,
            dtype: FloatDtype::Float32,
        })
    }
}

impl Plan {
    /// This plan with the UMAP layout of the cell graph `params.graph` as the cell
    /// embedding `params.into`. `seed` is required (R7).
    ///
    /// # Errors
    /// `KY-E109` if the plan has no cell graph of that name; `KY-E215` for
    /// `n_components` of 0 or above [`UMAP_MAX_COMPONENTS`], a `spread` or `alpha`
    /// that is not positive and finite, a `min_dist` or `gamma` that is negative or not
    /// finite, a `min_dist` above
    /// `spread` (umap-learn refuses it too), or an `into` that breaks format.md §2.1
    /// or differs from an embedding of the plan only in case; `KY-E216` if this plan
    /// ends in a sink.
    pub fn umap(self: &Arc<Self>, params: Umap) -> Result<Arc<Self>, KytoError> {
        self.check_open("umap")?;
        self.output_schema()
            .graph(&params.graph)
            .filter(|graph| graph.axis == Axis::Cells)
            .ok_or_else(|| self.missing_member("cell graph", &params.graph))?;
        if params.n_components == 0 || params.n_components > UMAP_MAX_COMPONENTS {
            return Err(Self::invalid_argument(
                "umap",
                "n_components",
                &params.n_components.to_string(),
                &format!(
                    "a layout has from 1 to {UMAP_MAX_COMPONENTS} dimensions (it holds \
                     n_cells x n_components positions)"
                ),
                &format!("an int from 1 to {UMAP_MAX_COMPONENTS}"),
            ));
        }
        Self::check_positive("umap", "spread", params.spread)?;
        Self::check_positive("umap", "alpha", params.alpha)?;
        for (parameter, value) in [("min_dist", params.min_dist), ("gamma", params.gamma)] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(Self::invalid_argument(
                    "umap",
                    parameter,
                    &value.to_string(),
                    "it is not a non-negative finite number",
                    "a finite float of at least 0",
                ));
            }
        }
        if params.min_dist > params.spread {
            return Err(Self::invalid_argument(
                "umap",
                "min_dist",
                &params.min_dist.to_string(),
                &format!("it is above spread={}", params.spread),
                "a float from 0 to spread",
            ));
        }
        self.check_member_name(
            "umap",
            "into",
            &params.into,
            MemberGroup::Embeddings,
            &params.into,
        )?;
        Ok(self.extend(params, Self::Umap))
    }
}
