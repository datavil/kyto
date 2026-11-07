//! The root node of every plan: a store and its schema.
//!
//! Does: hold the store location and the `AtlasSchema` read from it when the plan was
//! built (R6: the plan sees one fixed schema, whatever happens to the store later).
//! Does not: open or read the store; the caller passes the schema in.

use std::sync::Arc;

use kyto_core::AtlasSchema;

/// Reads a store: the root of every plan.
#[derive(Debug, Clone, PartialEq)]
pub struct Scan {
    source: String,
    schema: Arc<AtlasSchema>,
}

impl Scan {
    /// A scan of the store at `source` whose `schema.json` is `schema`.
    pub fn new(source: impl Into<String>, schema: AtlasSchema) -> Self {
        Self {
            source: source.into(),
            schema: Arc::new(schema),
        }
    }

    /// The store location as given to the scan.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The store's schema, as read when the plan was built.
    pub fn schema(&self) -> &Arc<AtlasSchema> {
        &self.schema
    }
}
