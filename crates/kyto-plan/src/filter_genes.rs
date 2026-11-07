//! The node that keeps the genes where a gene predicate is true (taken early from 4.1's
//! `Filter` for task 3.8, gene axis only).
//!
//! Does: hold its input and a boolean [`GeneExpr`] predicate, resolved against the
//! input when the node is built (R6), and mark the gene axis as changed, so a stored
//! derived cell column read downstream is stale (`KY-E301`).
//! Does not: evaluate the predicate (the executor, 5.x), know which genes it keeps
//! (so it adds nothing to `Plan::removed_genes`, task 3.7), filter cells (4.1
//! generalises it), or change the schema: which genes remain is known only when the
//! plan runs.

use std::sync::Arc;

use kyto_core::{AtlasSchema, GeneExpr};

use crate::plan::{Input, Plan};

/// Keeps the genes where a boolean gene predicate is true.
#[derive(Debug, Clone)]
pub struct FilterGenes {
    pub(crate) input: Input,
    predicate: GeneExpr,
    /// The input's output schema, shared (see the module doc).
    schema: Arc<AtlasSchema>,
}

impl FilterGenes {
    /// The node over `input`; the caller has resolved `predicate` against `input` and
    /// checked that it is boolean.
    pub(crate) fn new(input: Arc<Plan>, predicate: GeneExpr) -> Self {
        let schema = Arc::clone(input.schema());
        Self {
            input: Input::new(input),
            predicate,
            schema,
        }
    }

    /// The resolved predicate.
    pub fn predicate(&self) -> &GeneExpr {
        &self.predicate
    }

    /// The node's output schema: its input's.
    pub fn schema(&self) -> &Arc<AtlasSchema> {
        &self.schema
    }
}
