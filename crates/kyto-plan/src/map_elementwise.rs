//! The node that maps one layer to another value by value: `normalize_total`, `log1p`.
//!
//! Does: hold the operation, the input layer (the default layer written in when none
//! is named, R6) and the output layer `into`, checked when built. Its output schema
//! adds `into` as a `float32` layer, or replaces the layer of that name, with the
//! input layer's `nnz`: both operations keep zeros at zero. Writing a layer that
//! exists makes stored derived columns computed from it stale (`KY-E301`). Changes no
//! axis. `log1p` and `normalize_total` with a `target_sum` are per-cell;
//! `normalize_total` without one scales to the median total over cells (scanpy's
//! default), which is cross-cell.
//! Does not: compute anything (6.2, fused into one pass by 4.4), or take scanpy's
//! other parameters (`exclude_highly_expressed`, `base`, ...), which 6.2 adds.

use std::fmt;
use std::sync::Arc;

use kyto_core::{Axis, LayerDtype, LayerSchema};
use kyto_error::KytoError;
use serde::{Deserialize, Serialize};

use crate::node::{CellScope, LogicalNode, Transform};
use crate::plan::Plan;
use crate::plan_schema::{PlanSchema, SparseMember};

/// A value-by-value operation on a layer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "op")]
pub enum ElementwiseOp {
    /// Scale each cell to a total of `target_sum`, or of the median total over cells.
    NormalizeTotal {
        /// The total each cell is scaled to; `None` is the median over cells.
        target_sum: Option<f64>,
    },
    /// `ln(1 + x)`.
    Log1p,
}

impl ElementwiseOp {
    fn method(self) -> &'static str {
        match self {
            Self::NormalizeTotal { .. } => "normalize_total",
            Self::Log1p => "log1p",
        }
    }
}

/// Maps a layer value by value into a layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapElementwise {
    op: ElementwiseOp,
    layer: String,
    into: String,
}

impl MapElementwise {
    /// The operation.
    pub fn op(&self) -> ElementwiseOp {
        self.op
    }

    /// The input layer.
    pub fn layer(&self) -> &str {
        &self.layer
    }

    /// The output layer (`into`).
    pub fn output_layer(&self) -> &str {
        &self.into
    }
}

impl LogicalNode for MapElementwise {
    fn name(&self) -> &'static str {
        match self.op {
            ElementwiseOp::NormalizeTotal { .. } => "NORMALIZE_TOTAL",
            ElementwiseOp::Log1p => "LOG1P",
        }
    }

    fn method(&self) -> &'static str {
        self.op.method()
    }

    fn changes_axis(&self) -> Option<Axis> {
        None
    }

    fn cell_scope(&self) -> CellScope {
        match self.op {
            ElementwiseOp::NormalizeTotal { target_sum: None } => CellScope::CrossCell,
            ElementwiseOp::NormalizeTotal {
                target_sum: Some(_),
            }
            | ElementwiseOp::Log1p => CellScope::PerCell,
        }
    }

    fn describe(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?} -> {:?}", self.layer, self.into)?;
        if let ElementwiseOp::NormalizeTotal { target_sum } = self.op {
            match target_sum {
                Some(target_sum) => write!(formatter, " target_sum={target_sum}")?,
                None => formatter.write_str(" target_sum=median")?,
            }
        }
        Ok(())
    }
}

impl Transform for MapElementwise {
    fn output_schema(&self, input: &PlanSchema) -> PlanSchema {
        let source = SparseMember::Layer(self.layer.clone());
        let nnz = input.layer(&self.layer).map_or(0, |layer| layer.nnz);
        let layer = LayerSchema {
            name: self.into.clone(),
            dtype: LayerDtype::Float32,
            nnz,
        };
        input.clone().with_layer(layer, input.nnz_is_exact(&source))
    }

    fn writes_layer(&self) -> Option<&str> {
        Some(&self.into)
    }
}

impl Plan {
    /// This plan with `layer` (or the default layer) scaled so each cell sums to
    /// `target_sum` (the median total over cells when `None`), written to `into`.
    ///
    /// # Errors
    /// `KY-E215` for a `target_sum` that is not a positive finite number or an empty
    /// `into`; and those of [`Plan::log1p`].
    pub fn normalize_total(
        self: &Arc<Self>,
        target_sum: Option<f64>,
        layer: Option<&str>,
        into: &str,
    ) -> Result<Arc<Self>, KytoError> {
        self.check_open("normalize_total")?;
        if let Some(target_sum) = target_sum {
            Self::check_positive("normalize_total", "target_sum", target_sum)?;
        }
        self.map_elementwise(ElementwiseOp::NormalizeTotal { target_sum }, layer, into)
    }

    /// This plan with `ln(1 + x)` of `layer` (or the default layer) written to `into`.
    ///
    /// # Errors
    /// `KY-E109` for a layer not in the plan; `KY-E211` for no layer and no default
    /// layer; `KY-E215` for an empty `into`; `KY-E216` if this plan ends in a sink.
    pub fn log1p(
        self: &Arc<Self>,
        layer: Option<&str>,
        into: &str,
    ) -> Result<Arc<Self>, KytoError> {
        self.map_elementwise(ElementwiseOp::Log1p, layer, into)
    }

    fn map_elementwise(
        self: &Arc<Self>,
        op: ElementwiseOp,
        layer: Option<&str>,
        into: &str,
    ) -> Result<Arc<Self>, KytoError> {
        let method = op.method();
        self.check_open(method)?;
        let layer = self.input_layer(method, layer)?;
        Self::check_name(method, "into", into)?;
        let node = MapElementwise {
            op,
            layer,
            into: into.to_owned(),
        };
        Ok(self.extend(node, Self::MapElementwise))
    }
}
