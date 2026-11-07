//! The gene IDs of a store, as the feature resolver looks them up (task 3.5).
//!
//! Does: hold the `gene_id` column of `genes.parquet`, read once when a plan is
//! scanned, as a set (so gene names resolve and are checked at plan build, R6) and in
//! table order (the column names of a layer, `Matrix.names()`, task 3.6).
//! Does not: read the table (kyto-store does) or look up symbols (`by="symbol"` is task
//! 3.9).

use std::collections::HashSet;
use std::fmt;

/// The gene IDs of a store, in table order and as a set.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct GeneIds {
    ordered: Vec<String>,
    ids: HashSet<String>,
}

impl GeneIds {
    /// The IDs `ids`, in the given (table) order. Duplicates collapse in the set and
    /// stay in the order (a valid store has none, format.md §5.2).
    pub fn new(ids: impl IntoIterator<Item = String>) -> Self {
        let ordered: Vec<String> = ids.into_iter().collect();
        let ids = ordered.iter().cloned().collect();
        Self { ordered, ids }
    }

    /// Whether `gene_id` is one of the IDs (exact byte comparison, format.md §5.2).
    pub fn contains(&self, gene_id: &str) -> bool {
        self.ids.contains(gene_id)
    }

    /// The IDs in table order: gene `j` is the `j`-th (format.md §5.1).
    pub fn ordered(&self) -> &[String] {
        &self.ordered
    }

    /// The number of distinct IDs.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether there are no IDs.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// The count only: a store can have tens of thousands of genes.
impl fmt::Debug for GeneIds {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "GeneIds({} genes)", self.ids.len())
    }
}
