//! The gene IDs of a store, as the feature resolver looks them up (task 3.5).
//!
//! Does: hold the `gene_id` column of `genes.parquet` as a set, read once when a plan
//! is scanned, so gene names resolve and are checked at plan build (R6).
//! Does not: read the table (kyto-store does), keep the gene order, or look up symbols
//! (`by="symbol"` is task 3.9).

use std::collections::HashSet;
use std::fmt;

/// The set of gene IDs of a store.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct GeneIds {
    ids: HashSet<String>,
}

impl GeneIds {
    /// The set of `ids`. Duplicates collapse (a valid store has none, format.md §5.2).
    pub fn new(ids: impl IntoIterator<Item = String>) -> Self {
        Self {
            ids: ids.into_iter().collect(),
        }
    }

    /// Whether `gene_id` is one of the IDs (exact byte comparison, format.md §5.2).
    pub fn contains(&self, gene_id: &str) -> bool {
        self.ids.contains(gene_id)
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
