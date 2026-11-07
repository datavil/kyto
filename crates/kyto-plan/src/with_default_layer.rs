//! The node that sets the layer short gene names refer to (used by the resolver, 3.5).
//!
//! Does: hold its input and the layer name, checked against the schema when built.
//! Does not: change the data or the schema; it only records a choice (R7: the layer is
//! an argument of the plan, not a global setting).

use std::sync::Arc;

use kyto_core::AtlasSchema;

use crate::plan::{Input, Plan};

/// Sets the default layer of everything downstream.
#[derive(Debug, Clone)]
pub struct WithDefaultLayer {
    pub(crate) input: Input,
    layer: String,
    /// The input's output schema, shared (this node does not change it).
    schema: Arc<AtlasSchema>,
}

impl WithDefaultLayer {
    /// The node over `input`; the caller has checked that `layer` exists.
    pub(crate) fn new(input: Arc<Plan>, layer: String) -> Self {
        let schema = Arc::clone(input.schema());
        Self {
            input: Input::new(input),
            layer,
            schema,
        }
    }

    /// The layer name, as stored in the schema.
    pub fn layer(&self) -> &str {
        &self.layer
    }

    /// The node's output schema: its input's.
    pub fn schema(&self) -> &Arc<AtlasSchema> {
        &self.schema
    }
}
