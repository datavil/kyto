//! `.h5ad` input (anndata >= 0.8 on-disk format, task 2.6, format.md §12).
//!
//! Does: read `obs` and `var` (dataframe 0.2.0: `_index`, `column-order`; columns
//! `array`, `string-array`, `categorical`, `nullable-integer`, `nullable-boolean`,
//! `nullable-string-array`) into the cell and gene tables, and offer `X` and every
//! `layers/*` member (dense `array`, `csr_matrix`, `csc_matrix`) as row sources.
//! Does not: import `obsm`, `varm`, `obsp`, `varp`, `uns` or `raw` (format v0 gaps,
//! deviations.md), or files written by anndata < 0.8 (no `encoding-type`).
//!
//! Values match `anndata.read_h5ad`: categorical code `-1` and masked nullable entries
//! become null; `NaN` in a float column becomes null too, since anndata has no
//! nullable float encoding and pandas reads `NaN` as missing (format.md §5.5, decided
//! by the user, Nov 7, 2026, task 3.7); string categories keep their order and
//! `ordered` flag; numeric categoricals become plain numeric columns (Enum holds
//! strings only). Matrix values are not touched: a stored `NaN` there stays a value.

use hdf5_metno::types::TypeDescriptor;
use hdf5_metno::{Dataset, Location};
use kyto_error::KytoError;
use polars::prelude::{
    ChunkApply, Column, DataFrame, DataType, FrozenCategories, IdxCa, IntoColumn, IntoSeries,
    NamedFrom, Series,
};

use crate::import_convert::{Import, ImportLayer};
use crate::import_h5::{
    H5File, attr_bool, attr_string, attr_strings, attr_u64s, descriptor, is_float64, length_1d,
    matrix_dtype, numeric_series, read_all, read_strings,
};
use crate::import_h5_matrix::{CscEntries, CsrRows, DenseRows, SparseParts};
use crate::import_rows::RowSource;
use crate::import_transpose::TransposedRows;
use crate::table_axis::TableAxis;

/// What to do with a file anndata < 0.8 wrote.
const OLD_ANNDATA: &str =
    "read the file with anndata >= 0.8 and save it again with write_h5ad, then convert it";

/// The `(encoding-type, encoding-version)` of a member, if it has them.
fn encoding(location: &Location) -> Option<(String, String)> {
    let kind = attr_string(location, "encoding-type")?;
    let version = attr_string(location, "encoding-version").unwrap_or_default();
    Some((kind, version))
}

/// The versions of each encoding this importer reads (anndata 0.8 to 0.13).
fn known_version(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "anndata" | "csr_matrix" | "csc_matrix" | "dict" => "0.1.0",
        "nullable-integer" | "nullable-boolean" | "nullable-string-array" => "0.1.0",
        "array" | "string-array" | "categorical" | "dataframe" => "0.2.0",
        _ => return None,
    })
}

/// The encoding type of `member`, checked against [`known_version`].
///
/// # Errors
/// `KY-E120` if it has no encoding, or an unknown type or version.
fn checked_encoding(
    file: &H5File,
    member: &str,
    location: &Location,
    accepted: &str,
) -> Result<String, KytoError> {
    let Some((kind, version)) = encoding(location) else {
        return Err(file.unsupported(member, "it has no encoding-type attribute", OLD_ANNDATA));
    };
    match known_version(&kind) {
        Some(known) if known == version => Ok(kind),
        Some(known) => Err(file.unsupported(
            member,
            &format!("encoding-type '{kind}' has version '{version}', not '{known}'"),
            "convert files written by anndata 0.8 to 0.13, or save the file again with one of them",
        )),
        None => Err(file.unsupported(
            member,
            &format!("encoding-type '{kind}' is not supported"),
            accepted,
        )),
    }
}

/// Read an `.h5ad` file: tables now, matrices as row sources.
pub(crate) fn read_h5ad<'a>(
    file: &'a H5File,
    x_layer: &str,
    budget_bytes: u64,
    block_rows: u64,
) -> Result<Import<'a>, KytoError> {
    checked_encoding(file, "/", &file.file, OLD_ANNDATA)?;
    let mut ordered = Vec::new();
    let cells = read_dataframe(file, "obs", TableAxis::Cells, &mut ordered)?;
    let genes = read_dataframe(file, "var", TableAxis::Genes, &mut ordered)?;
    let shape = (cells.height() as u64, genes.height() as u64);
    let mut layers = Vec::new();
    if file.exists("X") {
        layers.push(ImportLayer {
            name: x_layer.to_owned(),
            source: matrix(file, "X", shape, budget_bytes, block_rows)?,
        });
    }
    if file.exists("layers") {
        let group = file.group("layers", "store layers as a dict group")?;
        checked_encoding(file, "layers", &group, "store layers as a dict group")?;
        let names = group
            .member_names()
            .map_err(|error| file.read_error("layers", error))?;
        for name in names {
            let member = format!("layers/{name}");
            layers.push(ImportLayer {
                name,
                source: matrix(file, &member, shape, budget_bytes, block_rows)?,
            });
        }
    }
    Ok(Import {
        kind: "h5ad",
        cells,
        genes,
        ordered,
        layers,
    })
}

/// A matrix member (`X` or a layer) as a row source.
fn matrix<'a>(
    file: &'a H5File,
    member: &str,
    shape: (u64, u64),
    budget_bytes: u64,
    block_rows: u64,
) -> Result<Box<dyn RowSource + 'a>, KytoError> {
    let accepted = "store the matrix as a dense array, csr_matrix or csc_matrix";
    let (n_cells, n_genes) = shape;
    let shape_error = |found: &[u64]| KytoError::InvalidInput {
        path: file.label(member),
        location: "the whole matrix".to_owned(),
        problem: format!("its shape is {found:?}, but obs and var give [{n_cells}, {n_genes}]"),
        advice: "save the file again with anndata, which keeps X, obs and var aligned".to_owned(),
    };
    if let Ok(dataset) = file.file.dataset(member) {
        checked_encoding(file, member, &dataset, accepted)
            .and_then(|kind| expect_kind(file, member, &kind, &["array"], accepted))?;
        let found: Vec<u64> = dataset
            .shape()
            .iter()
            .map(|&length| length as u64)
            .collect();
        if found != [n_cells, n_genes] {
            return Err(shape_error(&found));
        }
        let dtype = matrix_dtype(file, member, &dataset)?;
        return Ok(Box::new(DenseRows::new(
            file,
            member,
            dataset,
            dtype,
            n_genes,
            budget_bytes,
        )));
    }
    let group = file.group(member, accepted)?;
    let kind = checked_encoding(file, member, &group, accepted)?;
    expect_kind(file, member, &kind, &["csr_matrix", "csc_matrix"], accepted)?;
    let found = attr_u64s(&group, "shape")
        .ok_or_else(|| file.unsupported(member, "it has no shape attribute", accepted))?;
    if found != [n_cells, n_genes] {
        return Err(shape_error(&found));
    }
    let data = file.dataset(&format!("{member}/data"), accepted)?;
    let dtype = matrix_dtype(file, &format!("{member}/data"), &data)?;
    // scipy sums repeated float32 entries in float32, float64 ones in float64.
    let f32_sums = !is_float64(file, &format!("{member}/data"), &data)?;
    if kind == "csr_matrix" {
        let parts = SparseParts::open(file, member, n_cells, n_genes)?;
        Ok(Box::new(
            CsrRows::new(file, parts, dtype).with_f32_sums(f32_sums),
        ))
    } else {
        let parts = SparseParts::open(file, member, n_genes, n_cells)?;
        let entries = CscEntries::new(file, parts)?;
        Ok(Box::new(
            TransposedRows::new(
                entries,
                dtype,
                n_cells,
                n_genes,
                file.label(member),
                budget_bytes,
                block_rows,
            )?
            .with_f32_sums(f32_sums),
        ))
    }
}

/// `KY-E120` unless `kind` is one of `allowed`.
fn expect_kind(
    file: &H5File,
    member: &str,
    kind: &str,
    allowed: &[&str],
    accepted: &str,
) -> Result<(), KytoError> {
    if allowed.contains(&kind) {
        Ok(())
    } else {
        Err(file.unsupported(
            member,
            &format!("encoding-type '{kind}' is not supported here"),
            accepted,
        ))
    }
}

/// A dataframe group as a table: the index as the ID column, then the columns in
/// `column-order`. Ordered categoricals are added to `ordered`.
fn read_dataframe(
    file: &H5File,
    member: &str,
    axis: TableAxis,
    ordered: &mut Vec<(TableAxis, String)>,
) -> Result<DataFrame, KytoError> {
    let accepted = "store obs and var as anndata dataframes";
    let group = file.group(member, accepted)?;
    let kind = checked_encoding(file, member, &group, accepted)?;
    expect_kind(file, member, &kind, &["dataframe"], accepted)?;
    let index = attr_string(&group, "_index")
        .ok_or_else(|| file.unsupported(member, "it has no _index attribute", OLD_ANNDATA))?;
    let column_order = attr_strings(&group, "column-order")
        .ok_or_else(|| file.unsupported(member, "it has no column-order attribute", OLD_ANNDATA))?;
    let index_member = format!("{member}/{index}");
    let ids = string_element(file, &index_member)?;
    let n_rows = ids.len() as u64;
    let mut columns: Vec<Column> = vec![Series::new(axis.id_column().into(), ids).into_column()];
    for name in column_order {
        let column_member = format!("{member}/{name}");
        let (series, is_ordered) = column(file, &column_member, &name)?;
        if series.len() as u64 != n_rows {
            return Err(KytoError::InvalidInput {
                path: file.label(&column_member),
                location: "the whole column".to_owned(),
                problem: format!("it has {} rows, but the index has {n_rows}", series.len()),
                advice: "save the file again with anndata, which keeps columns aligned".to_owned(),
            });
        }
        if is_ordered {
            ordered.push((axis, name));
        }
        columns.push(series.into_column());
    }
    DataFrame::new(n_rows as usize, columns).map_err(|error| KytoError::InvalidInput {
        path: file.label(member),
        location: "the column names".to_owned(),
        problem: error.to_string(),
        advice: "give every column a unique name".to_owned(),
    })
}

/// A string element (`string-array` or `nullable-string-array`) as optional strings.
fn string_element(file: &H5File, member: &str) -> Result<Vec<Option<String>>, KytoError> {
    let accepted = "store the index as strings (string-array or nullable-string-array)";
    if let Ok(dataset) = file.file.dataset(member) {
        let kind = checked_encoding(file, member, &dataset, accepted)?;
        expect_kind(file, member, &kind, &["string-array"], accepted)?;
        return Ok(read_strings(file, member, &dataset)?
            .into_iter()
            .map(Some)
            .collect());
    }
    let group = file.group(member, accepted)?;
    let kind = checked_encoding(file, member, &group, accepted)?;
    expect_kind(file, member, &kind, &["nullable-string-array"], accepted)?;
    let (values, mask) = masked(file, member)?;
    let strings = read_strings(file, &format!("{member}/values"), &values)?;
    check_mask_length(file, member, strings.len(), mask.len())?;
    Ok(strings
        .into_iter()
        .zip(mask)
        .map(|(text, missing)| (!missing).then_some(text))
        .collect())
}

/// The `values` dataset and `mask` of a nullable group.
fn masked(file: &H5File, member: &str) -> Result<(Dataset, Vec<bool>), KytoError> {
    let accepted = "a nullable array has datasets values and mask";
    let values = file.dataset(&format!("{member}/values"), accepted)?;
    let mask_member = format!("{member}/mask");
    let mask_dataset = file.dataset(&mask_member, accepted)?;
    let mask = read_all::<bool>(file, &mask_member, &mask_dataset)?;
    Ok((values, mask))
}

fn check_mask_length(
    file: &H5File,
    member: &str,
    n_values: usize,
    n_mask: usize,
) -> Result<(), KytoError> {
    if n_values == n_mask {
        return Ok(());
    }
    Err(KytoError::InvalidInput {
        path: file.label(member),
        location: "mask".to_owned(),
        problem: format!("values has {n_values} entries but mask has {n_mask}"),
        advice: "save the file again with anndata".to_owned(),
    })
}

/// `series` with every `NaN` of a float column made null (anndata's missing float);
/// other dtypes unchanged.
fn nan_to_null(series: Series) -> Series {
    match (series.f32(), series.f64()) {
        (Ok(values), _) => values
            .apply(|value| value.filter(|value| !value.is_nan()))
            .into_series(),
        (_, Ok(values)) => values
            .apply(|value| value.filter(|value| !value.is_nan()))
            .into_series(),
        _ => series,
    }
}

/// One dataframe column as a series named `name`, and whether it is an ordered
/// categorical.
fn column(file: &H5File, member: &str, name: &str) -> Result<(Series, bool), KytoError> {
    let accepted = "store columns as numeric or boolean arrays, strings, categoricals or nullable integers, booleans or strings";
    if let Ok(dataset) = file.file.dataset(member) {
        let kind = checked_encoding(file, member, &dataset, accepted)?;
        expect_kind(file, member, &kind, &["array", "string-array"], accepted)?;
        length_1d(file, member, &dataset)?;
        let series = if kind == "string-array" {
            Series::new(name.into(), read_strings(file, member, &dataset)?)
        } else {
            nan_to_null(numeric_series(file, member, name, &dataset, None)?)
        };
        return Ok((series, false));
    }
    let group = file.group(member, accepted)?;
    let kind = checked_encoding(file, member, &group, accepted)?;
    match kind.as_str() {
        "categorical" => categorical(file, member, name, &group),
        "nullable-integer" | "nullable-boolean" => {
            let (values, mask) = masked(file, member)?;
            let values_member = format!("{member}/values");
            check_mask_length(
                file,
                member,
                length_1d(file, &values_member, &values)? as usize,
                mask.len(),
            )?;
            let series = numeric_series(file, &values_member, name, &values, Some(&mask))?;
            Ok((series, false))
        }
        "nullable-string-array" => Ok((
            Series::new(name.into(), string_element(file, member)?),
            false,
        )),
        other => Err(file.unsupported(
            member,
            &format!("encoding-type '{other}' is not supported for a column"),
            accepted,
        )),
    }
}

/// A categorical column: string categories give an `Enum` (order and unused
/// categories kept), numeric categories a numeric column. Code `-1` is null.
fn categorical(
    file: &H5File,
    member: &str,
    name: &str,
    group: &Location,
) -> Result<(Series, bool), KytoError> {
    let accepted = "a categorical has datasets codes and categories";
    let codes_member = format!("{member}/codes");
    let codes_dataset = file.dataset(&codes_member, accepted)?;
    let codes = read_all::<i64>(file, &codes_member, &codes_dataset)?;
    let categories_member = format!("{member}/categories");
    let categories = file.dataset(&categories_member, accepted)?;
    let n_categories = length_1d(file, &categories_member, &categories)?;
    let mut positions = Vec::with_capacity(codes.len());
    for (row, &code) in codes.iter().enumerate() {
        if code < -1 || code >= n_categories as i64 {
            return Err(KytoError::InvalidInput {
                path: file.label(&codes_member),
                location: format!("row {row}"),
                problem: format!("code {code} is outside -1..{n_categories}"),
                advice: "save the file again with anndata".to_owned(),
            });
        }
        positions.push((code >= 0).then_some(code as u32));
    }
    let is_string = matches!(
        descriptor(file, &categories_member, &categories)?,
        TypeDescriptor::VarLenUnicode
            | TypeDescriptor::VarLenAscii
            | TypeDescriptor::FixedAscii(_)
            | TypeDescriptor::FixedUnicode(_)
    );
    if !is_string {
        let values = nan_to_null(numeric_series(
            file,
            &categories_member,
            name,
            &categories,
            None,
        )?);
        let series = values
            .take(&IdxCa::new("".into(), positions))
            .map_err(|error| file.read_error(member, error))?;
        return Ok((series, false));
    }
    let labels = read_strings(file, &categories_member, &categories)?;
    let invalid_categories = |problem: String| KytoError::InvalidInput {
        path: file.label(&categories_member),
        location: "the categories".to_owned(),
        problem,
        advice: "categories must be unique".to_owned(),
    };
    let frozen = FrozenCategories::new(labels.iter().map(String::as_str))
        .map_err(|error| invalid_categories(error.to_string()))?;
    let values: Vec<Option<&str>> = positions
        .iter()
        .map(|position| position.map(|position| labels[position as usize].as_str()))
        .collect();
    let series = Series::new(name.into(), values)
        .strict_cast(&DataType::from_frozen_categories(frozen))
        .map_err(|error| invalid_categories(error.to_string()))?;
    let is_ordered = attr_bool(group, "ordered").unwrap_or(false);
    Ok((series, is_ordered))
}
