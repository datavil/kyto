//! Where the values of an expression come from: the leaves of an expression tree.
//!
//! Does: name the four source tags (`Column`, `Expression`, `Profile`, `Embedding`) and
//! say whether a source is metadata (a table column) or reads the matrix or an
//! embedding.
//! Does not: check that the named column, layer, gene, cell or embedding exists (task
//! 3.5 resolves names against the schema), or decide which axis a source belongs to
//! ([`crate::CellExpr`] and [`crate::GeneExpr`] constructors do).

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
}

impl Source {
    /// Whether the source is a metadata column (lowers to Polars), not matrix or
    /// embedding data.
    pub fn is_metadata(&self) -> bool {
        matches!(self, Self::Column { .. })
    }
}
