//! The gene identity of a store, as the feature resolver looks it up (tasks 3.5, 3.9).
//!
//! Does: hold the `gene_id` column of `genes.parquet`, read once when a plan is
//! scanned, as a set (so gene names resolve and are checked at plan build, R6) and in
//! table order (the column names of a layer, `Matrix.names()`, task 3.6); and, when the
//! store has one, its `symbol` column as an index from each symbol to the IDs of the
//! genes that carry it, in table order (`by="symbol"`, task 3.9). Symbols need not be
//! unique (format.md §5.2); a null symbol is no symbol.
//! Does not: read the table (kyto-store does), decide what an ambiguous or unknown
//! symbol means (the resolver raises `KY-E204` / `KY-E210`), or suggest close matches
//! (task 8.2).

use std::collections::{HashMap, HashSet};
use std::fmt;

/// The gene IDs of a store, in table order and as a set, and its gene symbols if read.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct GeneIds {
    ordered: Vec<String>,
    ids: HashSet<String>,
    /// Symbol to the table positions of the genes carrying it, ascending. `None` when
    /// no symbols were read.
    symbols: Option<HashMap<String, Vec<usize>>>,
}

impl GeneIds {
    /// The IDs `ids`, in the given (table) order, with no symbols. Duplicates collapse
    /// in the set and stay in the order (a valid store has none, format.md §5.2;
    /// `kyto_store::read_ids` refuses them).
    pub fn new(ids: impl IntoIterator<Item = String>) -> Self {
        let ordered: Vec<String> = ids.into_iter().collect();
        let ids = ordered.iter().cloned().collect();
        Self {
            ordered,
            ids,
            symbols: None,
        }
    }

    /// These IDs with the gene symbols `symbols`, one per gene in table order (`None`
    /// for a gene without a symbol). Symbols past the last gene are ignored and genes
    /// past the last symbol have none; `kyto_store::read_symbols` gives exactly one per
    /// gene.
    #[must_use]
    pub fn with_symbols(mut self, symbols: impl IntoIterator<Item = Option<String>>) -> Self {
        let mut index: HashMap<String, Vec<usize>> = HashMap::new();
        for (position, symbol) in symbols.into_iter().take(self.ordered.len()).enumerate() {
            if let Some(symbol) = symbol {
                index.entry(symbol).or_default().push(position);
            }
        }
        self.symbols = Some(index);
        self
    }

    /// Whether `gene_id` is one of the IDs (exact byte comparison, format.md §5.2).
    pub fn contains(&self, gene_id: &str) -> bool {
        self.ids.contains(gene_id)
    }

    /// Whether symbols were read (the store has a `symbol` column).
    pub fn has_symbols(&self) -> bool {
        self.symbols.is_some()
    }

    /// The IDs of the genes whose symbol is `symbol` (exact byte comparison), in table
    /// order: empty when none has it or no symbols were read.
    pub fn with_symbol(&self, symbol: &str) -> Vec<&str> {
        self.symbols
            .as_ref()
            .and_then(|index| index.get(symbol))
            .map(|positions| {
                positions
                    .iter()
                    .map(|&position| self.ordered[position].as_str())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Each gene's symbol, in table order (`None` for a gene without one), or `None`
    /// when no symbols were read: the input of [`GeneIds::with_symbols`] back (the
    /// plan JSON writes it, task 4.3).
    pub fn symbols(&self) -> Option<Vec<Option<&str>>> {
        let index = self.symbols.as_ref()?;
        let mut symbols = vec![None; self.ordered.len()];
        for (symbol, positions) in index {
            for &position in positions {
                symbols[position] = Some(symbol.as_str());
            }
        }
        Some(symbols)
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

/// The counts only: a store can have tens of thousands of genes.
impl fmt::Debug for GeneIds {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.symbols {
            Some(index) => write!(
                formatter,
                "GeneIds({} genes, {} symbols)",
                self.ids.len(),
                index.len()
            ),
            None => write!(formatter, "GeneIds({} genes)", self.ids.len()),
        }
    }
}
