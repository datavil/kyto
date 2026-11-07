//! Column dtypes of the metadata tables and their `schema.json` entries (format.md §5.3,
//! §5.4, §8).
//!
//! Does: map a Polars dtype to one of the v0 column dtypes (the types themselves are
//! `kyto_core`'s), and describe a table's columns as the `columns` list of `schema.json`.
//! Does not: write `schema.json` (see `store_json`, task 2.5) or check IDs (see `table_check`).

use kyto_error::KytoError;
use polars::prelude::{DataFrame, DataType};

// The column types are part of the schema model (task 3.1); kyto-store re-exports them.
pub use kyto_core::{ColumnDtype, ColumnSchema};

/// The v0 dtype of a Polars dtype, or `None` if v0 does not allow it. `ordered` is
/// `false` for every `Enum`.
pub(crate) fn column_dtype(dtype: &DataType) -> Option<ColumnDtype> {
    Some(match dtype {
        DataType::Boolean => ColumnDtype::Boolean,
        DataType::Int8 => ColumnDtype::Int8,
        DataType::Int16 => ColumnDtype::Int16,
        DataType::Int32 => ColumnDtype::Int32,
        DataType::Int64 => ColumnDtype::Int64,
        DataType::UInt8 => ColumnDtype::UInt8,
        DataType::UInt16 => ColumnDtype::UInt16,
        DataType::UInt32 => ColumnDtype::UInt32,
        DataType::UInt64 => ColumnDtype::UInt64,
        DataType::Float32 => ColumnDtype::Float32,
        DataType::Float64 => ColumnDtype::Float64,
        DataType::String => ColumnDtype::String,
        DataType::Enum(categories, _) => ColumnDtype::Enum {
            n_categories: categories.categories().len() as u64,
            ordered: false,
        },
        _ => return None,
    })
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
) -> Result<Vec<ColumnSchema>, KytoError> {
    frame
        .columns()
        .iter()
        .map(|column| {
            let dtype =
                column_dtype(column.dtype()).ok_or_else(|| KytoError::InvalidTableColumn {
                    axis: axis.as_str().to_owned(),
                    column: column.name().to_string(),
                    problem: format!("has dtype {}, which v0 stores do not allow", column.dtype()),
                    accepted: ACCEPTED_DTYPES.to_owned(),
                })?;
            Ok(ColumnSchema::new(column.name().to_string(), dtype))
        })
        .collect()
}

/// What a non-ID column may hold, for `KY-E112` hints.
pub(crate) const ACCEPTED_DTYPES: &str = "use Boolean, Int8 to Int64, UInt8 to UInt64, Float32, Float64, String or Enum; cast Categorical to String or Enum";
