//! Write a cell or gene table as an anndata dataframe group (`obs`, `var`; task 2.7).
//!
//! Does: write the ID column as the index (`_index` names it: `cell_id`, `gene_id`) and
//! every other column in table order as anndata 0.13's h5py writer does: numbers and
//! booleans as `array` 0.2.0, integers and booleans with nulls as `nullable-integer` /
//! `nullable-boolean` 0.1.0 (values 0 / false under the mask), `Enum` columns as
//! `categorical` 0.2.0 (signed codes of pandas' width, `-1` for null, categories as a
//! `string-array`, the `ordered` flag from `schema.json`).
//! Does not: write float nulls as anything but `NaN` (anndata has no nullable float
//! encoding; deviations.md), or read the matrix.

use hdf5_metno::H5Type;
use kyto_error::KytoError;
use polars::prelude::{Column, DataFrame, DataType, Series};

use crate::export_h5::H5Out;
use crate::table_axis::TableAxis;

/// The column name anndata reserves for the index (`write_dataframe`).
const RESERVED_COLUMN: &str = "_index";

/// Write `frame` (the `axis` table, ID column first, checked by `read_table`) as the
/// dataframe group `member`. `ordered` lists this table's ordered `Enum` columns.
///
/// # Errors
/// - `KY-E124` for a column name HDF5 or anndata cannot hold (`/`, `.`, `_index`) or a
///   string with a NUL character.
/// - `KY-E110` if a write fails.
pub(crate) fn write_table(
    out: &H5Out,
    member: &str,
    axis: TableAxis,
    frame: &DataFrame,
    ordered: &[&str],
) -> Result<(), KytoError> {
    let id_name = axis.id_column();
    let columns: Vec<&Column> = frame
        .columns()
        .iter()
        .filter(|column| column.name().as_str() != id_name)
        .collect();
    let names: Vec<&str> = columns
        .iter()
        .map(|column| column.name().as_str())
        .collect();
    for name in &names {
        check_column_name(out, axis, name)?;
    }
    let group = out.group(member, ("dataframe", "0.2.0"))?;
    out.attr_str(member, &group, "_index", id_name)?;
    out.attr_strings(member, &group, "column-order", &names)?;

    let ids = series(out, member, frame, id_name)?;
    let ids = ids.str().map_err(|error| out.write_error(member, error))?;
    let id_values: Vec<&str> = ids.iter().map(|id| id.unwrap_or_default()).collect();
    out.strings(
        &format!("{member}/{id_name}"),
        id_values.into_iter(),
        &|row| format!("{} {id_name} of row {row}", axis.as_str()),
    )?;
    for column in columns {
        let name = column.name().as_str();
        let column_member = format!("{member}/{name}");
        let values = column.as_materialized_series();
        write_column(
            out,
            &column_member,
            &format!("{} column '{name}'", axis.as_str()),
            values,
            ordered.contains(&name),
        )?;
    }
    Ok(())
}

/// `KY-E124` for a name that is not a valid HDF5 member or that anndata reserves.
fn check_column_name(out: &H5Out, axis: TableAxis, name: &str) -> Result<(), KytoError> {
    let location = format!("{} column '{name}'", axis.as_str());
    let advice =
        "rename the column (h5ad column names cannot be empty, contain '/', be '.' or be '_index')";
    if name.is_empty() {
        return Err(out.unsupported(&location, "h5ad names cannot be empty", advice));
    }
    if name.contains('/') {
        return Err(out.unsupported(&location, "h5ad names cannot contain '/'", advice));
    }
    if name == "." {
        return Err(out.unsupported(&location, "'.' is not a valid HDF5 member name", advice));
    }
    if name == RESERVED_COLUMN {
        return Err(out.unsupported(
            &location,
            "anndata reserves the column name '_index'",
            advice,
        ));
    }
    if name.contains('\0') {
        return Err(out.unsupported(
            &location,
            "HDF5 names cannot hold the NUL character",
            advice,
        ));
    }
    Ok(())
}

fn series(out: &H5Out, member: &str, frame: &DataFrame, name: &str) -> Result<Series, KytoError> {
    frame
        .column(name)
        .map(|column| column.as_materialized_series().clone())
        .map_err(|error| out.write_error(member, error))
}

/// Write one non-ID column.
fn write_column(
    out: &H5Out,
    member: &str,
    location: &str,
    values: &Series,
    is_ordered: bool,
) -> Result<(), KytoError> {
    let failed = |error: polars::error::PolarsError| out.write_error(member, error);
    macro_rules! numeric {
        ($accessor:ident, $zero:expr) => {{
            let chunked = values.$accessor().map_err(failed)?;
            let items: Vec<_> = chunked.iter().collect();
            plain_or_nullable(out, member, &items, $zero)
        }};
    }
    match values.dtype() {
        DataType::Boolean => numeric!(bool, false),
        DataType::Int8 => numeric!(i8, 0),
        DataType::Int16 => numeric!(i16, 0),
        DataType::Int32 => numeric!(i32, 0),
        DataType::Int64 => numeric!(i64, 0),
        DataType::UInt8 => numeric!(u8, 0),
        DataType::UInt16 => numeric!(u16, 0),
        DataType::UInt32 => numeric!(u32, 0),
        DataType::UInt64 => numeric!(u64, 0),
        // Float nulls become NaN: anndata has no nullable float encoding.
        DataType::Float32 => {
            let chunked = values.f32().map_err(failed)?;
            let items: Vec<f32> = chunked
                .iter()
                .map(|value| value.unwrap_or(f32::NAN))
                .collect();
            out.dataset(member, &items, Some(("array", "0.2.0")))
                .map(|_| ())
        }
        DataType::Float64 => {
            let chunked = values.f64().map_err(failed)?;
            let items: Vec<f64> = chunked
                .iter()
                .map(|value| value.unwrap_or(f64::NAN))
                .collect();
            out.dataset(member, &items, Some(("array", "0.2.0")))
                .map(|_| ())
        }
        DataType::Enum(frozen, _) => {
            let categories: Vec<&str> = frozen.categories().values_iter().collect();
            categorical(out, member, location, values, &categories, is_ordered)
        }
        other => Err(out.unsupported(
            location,
            &format!("dtype {other} is not a v0 column dtype"),
            "write the store with kyto (format.md section 5.3)",
        )),
    }
}

/// A column without nulls as an `array`, with nulls as a nullable group.
fn plain_or_nullable<T: H5Type + Copy>(
    out: &H5Out,
    member: &str,
    items: &[Option<T>],
    zero: T,
) -> Result<(), KytoError> {
    let values: Vec<T> = items.iter().map(|item| item.unwrap_or(zero)).collect();
    if items.iter().all(Option::is_some) {
        return out
            .dataset(member, &values, Some(("array", "0.2.0")))
            .map(|_| ());
    }
    let kind = if T::type_descriptor() == bool::type_descriptor() {
        "nullable-boolean"
    } else {
        "nullable-integer"
    };
    out.group(member, (kind, "0.1.0"))?;
    let mask: Vec<bool> = items.iter().map(Option::is_none).collect();
    out.dataset(
        &format!("{member}/values"),
        &values,
        Some(("array", "0.2.0")),
    )?;
    out.dataset(&format!("{member}/mask"), &mask, Some(("array", "0.2.0")))?;
    Ok(())
}

/// An `Enum` column as a `categorical` group.
fn categorical(
    out: &H5Out,
    member: &str,
    location: &str,
    values: &Series,
    categories: &[&str],
    is_ordered: bool,
) -> Result<(), KytoError> {
    let group = out.group(member, ("categorical", "0.2.0"))?;
    out.attr_scalar(member, &group, "ordered", &is_ordered)?;
    let failed = |error: polars::error::PolarsError| out.write_error(member, error);
    // The physical codes are unsigned positions in the category list; null is -1.
    let physical = values
        .to_physical_repr()
        .cast(&DataType::Int64)
        .map_err(failed)?;
    let codes: Vec<i64> = physical
        .i64()
        .map_err(failed)?
        .iter()
        .map(|code| code.unwrap_or(-1))
        .collect();
    let codes_member = format!("{member}/codes");
    let encoding = Some(("array", "0.2.0"));
    // pandas' code width (`coerce_indexer_dtype`): the smallest signed integer above
    // the category count.
    let n_categories = categories.len() as u64;
    if n_categories < i8::MAX as u64 {
        let narrow: Vec<i8> = codes.iter().map(|&code| code as i8).collect();
        out.dataset(&codes_member, &narrow, encoding)?;
    } else if n_categories < i16::MAX as u64 {
        let narrow: Vec<i16> = codes.iter().map(|&code| code as i16).collect();
        out.dataset(&codes_member, &narrow, encoding)?;
    } else if n_categories < i32::MAX as u64 {
        let narrow: Vec<i32> = codes.iter().map(|&code| code as i32).collect();
        out.dataset(&codes_member, &narrow, encoding)?;
    } else {
        out.dataset(&codes_member, &codes, encoding)?;
    }
    out.strings(
        &format!("{member}/categories"),
        categories.iter().copied(),
        &|position| format!("category {position} of {location}"),
    )?;
    Ok(())
}
