//! The genes a plan has removed, and the node that removed each (task 3.7, R9).
//!
//! Does: record, for each gene ID removed upstream, the plan node that removed it, so
//! the resolver can refuse a later read of that gene with `KY-E203` when the plan is
//! built.
//! Does not: decide which genes a node removes (the plan nodes do, task 4.1), or know
//! genes removed by a filter on data (`filter_genes(gene_column(...) > ...)`): which
//! genes such a filter keeps is known only when the plan runs, so the executor reports
//! those reads (task 5.x) with the same error.

use std::collections::HashMap;
use std::fmt;

/// Gene IDs removed earlier in a plan, each with the node that removed it.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct RemovedGenes {
    removed_by: HashMap<String, String>,
}

impl RemovedGenes {
    /// No gene removed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `removed_by` (a plan node, as messages name it: `select_genes`)
    /// removed each of `gene_ids`. A gene recorded already keeps its first node, so
    /// record nodes from the scan downward: the upstream node is the one that removed it.
    pub fn insert(&mut self, gene_ids: impl IntoIterator<Item = String>, removed_by: &str) {
        for gene_id in gene_ids {
            self.removed_by
                .entry(gene_id)
                .or_insert_with(|| removed_by.to_owned());
        }
    }

    /// The node that removed `gene_id`, or `None` if it was not removed.
    pub fn removed_by(&self, gene_id: &str) -> Option<&str> {
        self.removed_by.get(gene_id).map(String::as_str)
    }

    /// The number of removed genes.
    pub fn len(&self) -> usize {
        self.removed_by.len()
    }

    /// Whether no gene was removed.
    pub fn is_empty(&self) -> bool {
        self.removed_by.is_empty()
    }
}

/// The count only: a filter can remove tens of thousands of genes.
impl fmt::Debug for RemovedGenes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "RemovedGenes({} genes)", self.removed_by.len())
    }
}
