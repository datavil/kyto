//! The root node of every plan: a store, its schema and its gene IDs.
//!
//! Does: hold the store location, and the `AtlasSchema` and gene IDs read from it when
//! the plan was built (R6: the plan resolves names against this one fixed snapshot,
//! whatever happens to the store later). Its output schema is the store's, every size
//! exact. Changes no axis; per-cell (it streams the store's blocks).
//! Does not: open or read the store; the caller passes the schema and gene IDs in.

use std::fmt;
use std::sync::Arc;

use kyto_core::{AtlasSchema, Axis, GeneIds};

use crate::node::{CellScope, LogicalNode};
use crate::plan_schema::PlanSchema;

/// Reads a store: the root of every plan.
#[derive(Debug, Clone, PartialEq)]
pub struct Scan {
    source: Arc<str>,
    schema: Arc<AtlasSchema>,
    gene_ids: Arc<GeneIds>,
}

impl Scan {
    /// A scan of the store at `source` whose `schema.json` is `schema` and whose
    /// `genes.parquet` holds `gene_ids`.
    pub fn new(source: impl Into<String>, schema: AtlasSchema, gene_ids: GeneIds) -> Self {
        Self {
            source: Arc::from(source.into()),
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

    /// The scan's output schema: the store's, with every size exact.
    pub fn output_schema(&self) -> PlanSchema {
        PlanSchema::of_store(AtlasSchema::clone(&self.schema))
    }

    pub(crate) fn shared_source(&self) -> Arc<str> {
        Arc::clone(&self.source)
    }
}

impl LogicalNode for Scan {
    fn name(&self) -> &'static str {
        "SCAN"
    }

    fn method(&self) -> &'static str {
        "scan"
    }

    fn changes_axis(&self) -> Option<Axis> {
        None
    }

    fn cell_scope(&self) -> CellScope {
        CellScope::PerCell
    }

    fn describe(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?} ({} cells x {} genes)",
            self.source(),
            self.schema.n_cells,
            self.schema.n_genes
        )
    }
}
