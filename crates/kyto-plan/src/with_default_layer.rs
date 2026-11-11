//! The node that sets the layer short gene names refer to (used by the resolver, 3.5).
//!
//! Does: hold the layer name, checked against the input's schema when built. Its
//! output schema is its input's. Changes no axis; per-cell.
//! Does not: change the data or the schema; it only records a choice (R7: the layer is
//! an argument of the plan, not a global setting).

use std::fmt;
use std::sync::Arc;

use kyto_core::Axis;
use kyto_error::KytoError;
use serde::{Deserialize, Serialize};

use crate::node::{CellScope, LogicalNode, Transform};
use crate::plan::Plan;
use crate::plan_schema::PlanSchema;

/// Sets the default layer of everything downstream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WithDefaultLayer {
    layer: String,
}

impl WithDefaultLayer {
    /// The layer name, as stored in the schema.
    pub fn layer(&self) -> &str {
        &self.layer
    }
}

impl LogicalNode for WithDefaultLayer {
    fn name(&self) -> &'static str {
        "WITH_DEFAULT_LAYER"
    }

    fn method(&self) -> &'static str {
        "with_default_layer"
    }

    fn changes_axis(&self) -> Option<Axis> {
        None
    }

    fn cell_scope(&self) -> CellScope {
        CellScope::PerCell
    }

    fn describe(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}", self.layer)
    }
}

impl Transform for WithDefaultLayer {
    fn output_schema(&self, input: &PlanSchema) -> PlanSchema {
        input.clone()
    }

    fn sets_default_layer(&self) -> Option<&str> {
        Some(&self.layer)
    }
}

impl Plan {
    /// This plan with its default layer set to `layer` (a new plan; `self` is unchanged).
    ///
    /// # Errors
    /// `KY-E109` if the schema has no layer named `layer` (names are case-sensitive);
    /// `KY-E216` if this plan ends in a sink.
    pub fn with_default_layer(self: &Arc<Self>, layer: &str) -> Result<Arc<Self>, KytoError> {
        self.check_open("with_default_layer")?;
        self.layer_named(layer)?;
        let node = WithDefaultLayer {
            layer: layer.to_owned(),
        };
        Ok(self.extend(node, Self::WithDefaultLayer))
    }
}
