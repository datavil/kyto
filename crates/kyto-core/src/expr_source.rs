//! Where the values of an expression come from: the leaves of an expression tree.
//!
//! Does: name the four source tags (`Column`, `Expression`, `Profile`, `Embedding`),
//! plus `Feature`, a name not yet resolved to one of them (task 3.4), `GeneSymbol`, a
//! gene named by symbol and not yet resolved to its ID (task 3.9), and `Recompute`, a
//! derived column computed again from its lineage (task 3.8); and say whether a source
//! is metadata (a table column) or reads the matrix or an embedding.
//!
//! Does not: check that the named column, layer, gene, cell or embedding exists (task
//! 3.5 resolves names against the schema), or decide which axis a source belongs to
//! ([`crate::CellExpr`] and [`crate::GeneExpr`] constructors do).

use crate::FeatureName;

/// The source tag of an expression leaf.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Source {
    /// A column of the cell table (on a cell expression) or of the gene table (on a gene
    /// expression).
    Column {
        /// The column name, as written (resolved to a full name by task 3.5).
        name: String,
    },
    /// The values of one gene across cells: a cell expression.
    Expression {
        /// The layer to read; `None` means the plan's default layer.
        layer: Option<String>,
        /// The gene ID, as written.
        gene_id: String,
    },
    /// The values of the one gene whose symbol is `symbol` across cells, not resolved
    /// yet: what `gene_expression('CD3E', by='symbol')` builds. The feature resolver
    /// replaces it with [`Source::Expression`] of that gene's ID when the plan is
    /// built (R6, task 3.9); a symbol that several genes carry is `KY-E204`.
    GeneSymbol {
        /// The layer to read; `None` means the plan's default layer.
        layer: Option<String>,
        /// The gene symbol, as written.
        symbol: String,
    },
    /// The values of one cell across genes: a gene expression.
    Profile {
        /// The layer to read; `None` means the plan's default layer.
        layer: Option<String>,
        /// The cell ID, as written.
        cell_id: String,
    },
    /// One component (column) of an embedding of the expression's axis.
    Embedding {
        /// The embedding name.
        name: String,
        /// The component, 0-based.
        component: u64,
    },
    /// A feature name given to `feature(...)` and not resolved yet: the feature
    /// resolver (task 3.5) replaces it with one of the tags above when the plan is
    /// built (R6). Not metadata until then.
    Feature {
        /// The parsed name.
        name: FeatureName,
    },
    /// A derived column of this axis's table computed again from the lineage the store
    /// records for it (task 3.8, format.md §8), not read as stored: what
    /// `cell_column('n_genes').recompute()` builds. Valid after a step that changed
    /// the column's inputs (`KY-E301` otherwise). The executor computes it (5.x); it
    /// reads the matrix, so it is not metadata.
    Recompute {
        /// The column name, as written.
        name: String,
    },
}

impl Source {
    /// Whether the source is a metadata column (lowers to Polars), not matrix or
    /// embedding data.
    pub fn is_metadata(&self) -> bool {
        matches!(self, Self::Column { .. })
    }
}
