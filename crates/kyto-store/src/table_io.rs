//! Write and read a cell or gene table as Parquet with Polars (format.md §5).
//!
//! Does: [`write_table`] checks a table, turns every non-ID `String` column into an
//! `Enum` and writes `cells.parquet` or `genes.parquet` with the settings of §5.7;
//! [`read_table`] reads one back and checks it; [`read_ids`] and [`read_symbols`] read
//! only the ID column or the gene `symbol` column (metadata for plan build, tasks 3.5
//! and 3.9).
//! Does not: check the row count against the matrix, write `schema.json` or write
//! atomically (see `store_writer`, task 2.5), or import anndata tables (task 2.6). A
//! failed write can leave a partial file behind.

use std::collections::{BTreeSet, HashSet};
use std::fs::File;
use std::path::Path;

use kyto_error::KytoError;
use polars::prelude::{
    Column, DataFrame, DataType, FrozenCategories, ParquetCompression, ParquetReader,
    ParquetWriter, SerReader, StatisticsOptions,
};

use crate::table_axis::TableAxis;
use crate::table_check::{PlainStrings, check_table};
use crate::table_columns::{ColumnSchema, table_columns};
use crate::tiling::Tiling;
use crate::zarr_arrays::store_io;

/// Write `frame` as the `axis` table of the store at `store` (format.md §5).
///
/// The first column must be the String ID column (`cell_id` or `gene_id`). Every other
/// `String` column is stored as an `Enum` whose categories are its distinct non-null
/// values in byte order (format.md §5.4), as anndata sorts the categories of string
/// columns. `Enum` columns keep their categories, used or not, in order. Row groups
/// hold one cell block (cells) or one gene block (genes) of `tiling` (format.md §5.7).
///
/// # Errors
/// - `KY-E112` for a missing or misplaced ID column, or a bad column name or dtype.
/// - `KY-E111` for null or empty IDs.
/// - `KY-E101` for duplicate IDs.
/// - `KY-E104` if the row group size from `tiling` is 0.
/// - `KY-E103` if the table file already exists.
/// - `KY-E110` if the file system or the Parquet writer fails.
pub fn write_table(
    store: &Path,
    axis: TableAxis,
    frame: &DataFrame,
    tiling: Tiling,
) -> Result<(), KytoError> {
    write_table_columns(store, axis, frame, tiling).map(|_| ())
}

/// [`write_table`], returning the stored columns as `schema.json` lists them (format.md
/// §8): string columns as they were stored, as `Enum`.
pub(crate) fn write_table_columns(
    store: &Path,
    axis: TableAxis,
    frame: &DataFrame,
    tiling: Tiling,
) -> Result<Vec<ColumnSchema>, KytoError> {
    check_table(axis, frame, PlainStrings::Allowed)?;
    let row_group_size = match axis.row_group_size(tiling) {
        0 => {
            return Err(KytoError::InvalidParameter {
                parameter: match axis {
                    TableAxis::Cells => "cell_block_size",
                    TableAxis::Genes => "gene_block_size",
                }
                .to_owned(),
                value: "0".to_owned(),
                accepted: "a positive number of rows".to_owned(),
            });
        }
        // Larger than any table: one row group.
        size => usize::try_from(size).unwrap_or(usize::MAX),
    };
    let mut stored = encode_strings(axis, frame)?;

    let path = store.join(axis.file_name());
    let path_text = path.display().to_string();
    let file = match File::create_new(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(KytoError::MemberExists {
                kind: "table".to_owned(),
                name: axis.as_str().to_owned(),
                existing: axis.file_name().to_owned(),
            });
        }
        Err(error) => return Err(store_io("create table", &path_text, error)),
    };
    let write_error =
        |error: polars::prelude::PolarsError| store_io("write table", &path_text, error);
    // One `write_batch` per block of rows, each a single chunk: one row group each
    // (`ParquetWriter::finish` would merge a short last group into the one before).
    stored.rechunk_mut();
    let mut writer = ParquetWriter::new(file)
        // No level: Polars' default zstd level, 3, the layers' default too. (Polars does
        // not export its level type; a pinned Polars version keeps the bytes fixed.)
        .with_compression(ParquetCompression::Zstd(None))
        .with_statistics(StatisticsOptions::default())
        .batched(stored.schema())
        .map_err(write_error)?;
    let mut offset = 0;
    while offset < stored.height() {
        let block = stored.slice(offset as i64, row_group_size);
        writer.write_batch(&block).map_err(write_error)?;
        offset += block.height();
    }
    writer.finish().map_err(write_error)?;
    table_columns(axis, &stored)
}

/// Read the `axis` table of the store at `store` and check it (format.md §5).
///
/// Returns a new frame the caller owns: the String ID column first, string columns as
/// `Enum`.
///
/// # Errors
/// - `KY-E110` if the file cannot be opened or is not Parquet.
/// - `KY-E113` if the table breaks format.md §5.2, §5.3, §5.4 or §5.6.
pub fn read_table(store: &Path, axis: TableAxis) -> Result<DataFrame, KytoError> {
    let path = store.join(axis.file_name());
    let path_text = path.display().to_string();
    let file = File::open(&path).map_err(|error| store_io("open table", &path_text, error))?;
    let frame = ParquetReader::new(file)
        .finish()
        .map_err(|error| store_io("read table", &path_text, error))?;
    check_table(axis, &frame, PlainStrings::Forbidden).map_err(|error| {
        let message = error.to_string();
        let prefix = format!("{}: ", error.code());
        KytoError::CorruptTable {
            path: path_text.clone(),
            problem: message.strip_prefix(&prefix).unwrap_or(&message).to_owned(),
        }
    })?;
    Ok(frame)
}

/// Read the ID column of the `axis` table of the store at `store`, and only that column
/// (task 3.5: plans resolve gene names against it). `n_rows` is the axis length from
/// `schema.json`.
///
/// Returns the IDs in table order, a new vector the caller owns.
///
/// # Errors
/// - `KY-E110` if the file cannot be opened, is not Parquet or has no ID column.
/// - `KY-E113` if the ID column is not `String`, holds a null or a duplicate, or does
///   not have `n_rows` rows (format.md §5.1, §5.2).
pub fn read_ids(store: &Path, axis: TableAxis, n_rows: u64) -> Result<Vec<String>, KytoError> {
    let path = store.join(axis.file_name());
    let path_text = path.display().to_string();
    let corrupt = |problem: String| KytoError::CorruptTable {
        path: path_text.clone(),
        problem,
    };
    let file = File::open(&path).map_err(|error| store_io("open table", &path_text, error))?;
    let frame = ParquetReader::new(file)
        .with_columns(Some(vec![axis.id_column().to_owned()]))
        .finish()
        .map_err(|error| store_io("read table", &path_text, error))?;
    let column = frame
        .column(axis.id_column())
        .map_err(|error| store_io("read table", &path_text, error))?;
    let ids = column.str().map_err(|_| {
        corrupt(format!(
            "the ID column '{}' is {}, not String",
            axis.id_column(),
            column.dtype()
        ))
    })?;
    if u64::try_from(ids.len()).ok() != Some(n_rows) {
        return Err(corrupt(format!(
            "it has {} rows, but schema.json says {n_rows}",
            ids.len()
        )));
    }
    let mut seen = HashSet::with_capacity(ids.len());
    ids.iter()
        .map(|id| {
            let id = id.ok_or_else(|| {
                corrupt(format!("the ID column '{}' holds a null", axis.id_column()))
            })?;
            if !seen.insert(id) {
                return Err(corrupt(format!(
                    "the ID column '{}' holds '{id}' more than once",
                    axis.id_column()
                )));
            }
            Ok(id.to_owned())
        })
        .collect()
}

/// Read the `symbol` column of the gene table of the store at `store`, and only that
/// column (task 3.9: `by='symbol'` resolves a symbol to its gene ID at plan build).
/// `n_rows` is `n_genes` from `schema.json`. Call it only when `schema.json` lists a
/// `symbol` column of dtype `Enum` (format.md §5.2: a string column, stored as `Enum`).
///
/// Returns one symbol per gene in table order, `None` where it is null, a new vector
/// the caller owns. Symbols need not be unique.
///
/// # Errors
/// - `KY-E110` if the file cannot be opened, is not Parquet or has no `symbol` column.
/// - `KY-E113` if the column is not `Enum` or does not have `n_rows` rows.
pub fn read_symbols(store: &Path, n_rows: u64) -> Result<Vec<Option<String>>, KytoError> {
    const SYMBOL: &str = "symbol";
    let path = store.join(TableAxis::Genes.file_name());
    let path_text = path.display().to_string();
    let corrupt = |problem: String| KytoError::CorruptTable {
        path: path_text.clone(),
        problem,
    };
    let file = File::open(&path).map_err(|error| store_io("open table", &path_text, error))?;
    let frame = ParquetReader::new(file)
        .with_columns(Some(vec![SYMBOL.to_owned()]))
        .finish()
        .map_err(|error| store_io("read table", &path_text, error))?;
    let column = frame
        .column(SYMBOL)
        .map_err(|error| store_io("read table", &path_text, error))?;
    if !matches!(column.dtype(), DataType::Enum(..)) {
        return Err(corrupt(format!(
            "the column '{SYMBOL}' is {}, not Enum",
            column.dtype()
        )));
    }
    if u64::try_from(column.len()).ok() != Some(n_rows) {
        return Err(corrupt(format!(
            "it has {} rows, but schema.json says {n_rows}",
            column.len()
        )));
    }
    let text = column
        .cast(&DataType::String)
        .map_err(|error| store_io("read table", &path_text, error))?;
    let symbols = text
        .str()
        .map_err(|error| store_io("read table", &path_text, error))?;
    Ok(symbols
        .iter()
        .map(|symbol| symbol.map(str::to_owned))
        .collect())
}

/// `frame` with every non-ID `String` column cast to an `Enum` of its sorted distinct
/// values.
fn encode_strings(axis: TableAxis, frame: &DataFrame) -> Result<DataFrame, KytoError> {
    let columns = frame
        .columns()
        .iter()
        .enumerate()
        .map(|(position, column)| {
            if position == 0 || column.dtype() != &DataType::String {
                return Ok(column.clone());
            }
            string_to_enum(column).map_err(|error| KytoError::InvalidTableColumn {
                axis: axis.as_str().to_owned(),
                column: column.name().to_string(),
                problem: format!("cannot be stored as an Enum: {error}"),
                accepted: "use at most 2^32 - 1 distinct values".to_owned(),
            })
        })
        .collect::<Result<Vec<Column>, KytoError>>()?;
    DataFrame::new(frame.height(), columns)
        .map_err(|error| store_io("build table", axis.file_name(), error))
}

fn string_to_enum(column: &Column) -> polars::prelude::PolarsResult<Column> {
    let strings = column.str()?;
    let categories: BTreeSet<&str> = strings.iter().flatten().collect();
    let frozen = FrozenCategories::new(categories.iter().copied())?;
    column.strict_cast(&DataType::from_frozen_categories(frozen))
}
