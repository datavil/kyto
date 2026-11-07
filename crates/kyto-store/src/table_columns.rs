//! Column dtypes of the metadata tables and their `schema.json` entries (format.md §5.3,
//! §5.4, §8).
//!
//! Does: map a Polars dtype to one of the v0 column dtypes, and describe a table's
//! columns as the `columns` list of `schema.json`.
//! Does not: write `schema.json` (task 2.5) or check IDs (see `table_check`).

use kyto_error::KytoError;
use polars::prelude::{DataFrame, DataType};

/// A column dtype allowed in v0 tables (format.md §5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnDtype {
    /// `Boolean`.
    Boolean,
    /// `Int8`.
    Int8,
    /// `Int16`.
    Int16,
    /// `Int32`.
    Int32,
    /// `Int64`.
    Int64,
    /// `UInt8`.
    UInt8,
    /// `UInt16`.
    UInt16,
    /// `UInt32`.
    UInt32,
    /// `UInt64`.
    UInt64,
    /// `Float32`.
    Float32,
    /// `Float64`.
    Float64,
    /// `String`: the ID column only.
    String,
    /// `Enum` with a fixed category list (format.md §5.4).
    Enum {
        /// Number of categories, used or not.
        n_categories: u64,
        /// Whether the categories are ordered. Polars `Enum` does not carry this; it is
        /// kept in `schema.json` only, and the caller sets it (from anndata, 2.6).
        ordered: bool,
    },
}

impl ColumnDtype {
    /// The v0 dtype of a Polars dtype, or `None` if v0 does not allow it. `ordered` is
    /// `false` for every `Enum`.
    pub fn from_polars(dtype: &DataType) -> Option<Self> {
        Some(match dtype {
            DataType::Boolean => Self::Boolean,
            DataType::Int8 => Self::Int8,
            DataType::Int16 => Self::Int16,
            DataType::Int32 => Self::Int32,
            DataType::Int64 => Self::Int64,
            DataType::UInt8 => Self::UInt8,
            DataType::UInt16 => Self::UInt16,
            DataType::UInt32 => Self::UInt32,
            DataType::UInt64 => Self::UInt64,
            DataType::Float32 => Self::Float32,
            DataType::Float64 => Self::Float64,
            DataType::String => Self::String,
            DataType::Enum(categories, _) => Self::Enum {
                n_categories: categories.categories().len() as u64,
                ordered: false,
            },
            _ => return None,
        })
    }

    /// The Polars dtype name, as `schema.json` writes it (format.md §8).
    pub fn name(self) -> &'static str {
        match self {
            Self::Boolean => "Boolean",
            Self::Int8 => "Int8",
            Self::Int16 => "Int16",
            Self::Int32 => "Int32",
            Self::Int64 => "Int64",
            Self::UInt8 => "UInt8",
            Self::UInt16 => "UInt16",
            Self::UInt32 => "UInt32",
            Self::UInt64 => "UInt64",
            Self::Float32 => "Float32",
            Self::Float64 => "Float64",
            Self::String => "String",
            Self::Enum { .. } => "Enum",
        }
    }
}

/// One entry of a table's `columns` list in `schema.json` (format.md §8).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableColumn {
    /// Column name.
    pub name: String,
    /// Column dtype.
    pub dtype: ColumnDtype,
}

impl TableColumn {
    /// The `schema.json` object: `name`, `dtype`, then `n_categories` and `ordered`
    /// exactly when the dtype is `Enum`. `lineage` (3.8) is not written here.
    pub fn to_json(&self) -> serde_json::Value {
        let mut object = serde_json::Map::new();
        object.insert("name".to_owned(), self.name.clone().into());
        object.insert("dtype".to_owned(), self.dtype.name().into());
        if let ColumnDtype::Enum {
            n_categories,
            ordered,
        } = self.dtype
        {
            object.insert("n_categories".to_owned(), n_categories.into());
            object.insert("ordered".to_owned(), ordered.into());
        }
        serde_json::Value::Object(object)
    }
}

/// Describe every column of a stored table, in order (format.md §8).
///
/// `frame` is a table as [`crate::read_table`] returns it.
///
/// # Errors
/// `KY-E112` if a column has a dtype v0 does not allow (only for frames that did not
/// come from [`crate::read_table`]).
pub fn table_columns(
    axis: crate::TableAxis,
    frame: &DataFrame,
) -> Result<Vec<TableColumn>, KytoError> {
    frame
        .columns()
        .iter()
        .map(|column| {
            let dtype = ColumnDtype::from_polars(column.dtype()).ok_or_else(|| {
                KytoError::InvalidTableColumn {
                    axis: axis.as_str().to_owned(),
                    column: column.name().to_string(),
                    problem: format!("has dtype {}, which v0 stores do not allow", column.dtype()),
                    accepted: ACCEPTED_DTYPES.to_owned(),
                }
            })?;
            Ok(TableColumn {
                name: column.name().to_string(),
                dtype,
            })
        })
        .collect()
}

/// What a non-ID column may hold, for `KY-E112` hints.
pub(crate) const ACCEPTED_DTYPES: &str = "use Boolean, Int8 to Int64, UInt8 to UInt64, Float32, Float64, String or Enum; cast Categorical to String or Enum";
