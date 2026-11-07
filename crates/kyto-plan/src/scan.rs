//! The root node of every plan: a store, its schema and its gene IDs.
//!
//! Does: hold the store location, and the `AtlasSchema` and gene IDs read from it when
//! the plan was built (R6: the plan resolves names against this one fixed snapshot,
//! whatever happens to the store later).
//! Does not: open or read the store; the caller passes the schema and gene IDs in.

use std::sync::Arc;

use kyto_core::{AtlasSchema, GeneIds};

/// Reads a store: the root of every plan.
#[derive(Debug, Clone, PartialEq)]
pub struct Scan {
    source: String,
    schema: Arc<AtlasSchema>,
    gene_ids: Arc<GeneIds>,
}

impl Scan {
    /// A scan of the store at `source` whose `schema.json` is `schema` and whose
    /// `genes.parquet` holds `gene_ids`.
    pub fn new(source: impl Into<String>, schema: AtlasSchema, gene_ids: GeneIds) -> Self {
        Self {
            source: source.into(),
            schema: Arc::new(schema),
            gene_ids: Arc::new(gene_ids),
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

    /// The store's gene IDs, as read when the plan was built.
    pub fn gene_ids(&self) -> &Arc<GeneIds> {
        &self.gene_ids
    }
}
