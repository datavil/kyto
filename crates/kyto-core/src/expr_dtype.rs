//! The dtype an expression carries: a kind, known from its source and operations.
//!
//! Does: name the four kinds an expression can have (`Boolean`, `Numeric`, `String`,
//! `Unknown`), map a stored column dtype to its kind, and say which kinds two operands
//! of one operation may have.
//! Does not: resolve a column's dtype from a schema (task 3.5 replaces `Unknown` with the
//! stored kind) or track bit widths (the planner reads them from the schema).

use crate::schema_column::ColumnDtype;

/// The dtype kind of a [`crate::CellExpr`] or [`crate::GeneExpr`].
///
/// A kind, not a storage dtype: `Numeric` covers every integer and float width. A
/// column named by `cell_column` / `gene_column` is `Unknown` until the plan resolves it
/// against the schema (task 3.5); every other source is `Numeric`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExprDtype {
    /// Not known until the plan is built (an unresolved column).
    Unknown,
    /// `True` / `False`: comparisons, `&`, `|`, `~` and the `str` predicates.
    Boolean,
    /// Integers and floats: matrix values, embeddings and arithmetic.
    Numeric,
    /// Text: the ID column and categorical (`Enum`) columns.
    String,
}

impl ExprDtype {
    /// The kind as written in messages and stubs: `boolean`, `numeric`, `string` or
    /// `unknown`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Boolean => "boolean",
            Self::Numeric => "numeric",
            Self::String => "string",
        }
    }

    /// The kind of a stored column of dtype `dtype`. `Enum` columns hold text
    /// (format.md §5.4), so they are `String`.
    pub fn of_column(dtype: ColumnDtype) -> Self {
        match dtype {
            ColumnDtype::Boolean => Self::Boolean,
            ColumnDtype::Int8
            | ColumnDtype::Int16
            | ColumnDtype::Int32
            | ColumnDtype::Int64
            | ColumnDtype::UInt8
            | ColumnDtype::UInt16
            | ColumnDtype::UInt32
            | ColumnDtype::UInt64
            | ColumnDtype::Float32
            | ColumnDtype::Float64 => Self::Numeric,
            ColumnDtype::String | ColumnDtype::Enum { .. } => Self::String,
        }
    }

    /// Whether a value of this kind can stand where `accepted` kinds are expected.
    /// `Unknown` is accepted anywhere; the plan checks it once it is resolved.
    pub(crate) fn is_one_of(self, accepted: &[Self]) -> bool {
        self == Self::Unknown || accepted.contains(&self)
    }

    /// Whether two operands of one comparison may be compared: same kind, or one of
    /// them `Unknown`.
    pub(crate) fn matches(self, other: Self) -> bool {
        self == Self::Unknown || other == Self::Unknown || self == other
    }
}
