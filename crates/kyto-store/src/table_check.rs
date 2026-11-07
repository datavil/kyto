//! Check a cell or gene table against format.md §5.2, §5.3 and §5.6.
//!
//! Does: check column names and positions, dtypes, and that IDs are non-null,
//! non-empty and unique. Returns the input errors `KY-E101`, `KY-E111` and `KY-E112`;
//! the reader turns them into `KY-E113`.
//! Does not: convert strings to `Enum` or touch the file system (see `table_io`).

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use kyto_error::KytoError;
use polars::prelude::{Column, DataFrame, DataType};

use crate::table_axis::{RESERVED_ID_COLUMNS, TableAxis};
use crate::table_columns::{ACCEPTED_DTYPES, ColumnDtype, column_dtype};

/// How many offending IDs or rows an error lists.
const SHOWN: usize = 5;

/// Whether a non-ID `String` column is accepted: before writing it becomes an `Enum`;
/// in a stored table it is a format error (format.md §5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlainStrings {
    Allowed,
    Forbidden,
}

/// Check every rule of format.md §5.2, §5.3 and §5.6 on `frame`.
///
/// # Errors
/// - `KY-E112` for a missing or misplaced ID column, a bad column name or dtype.
/// - `KY-E111` for null or empty IDs.
/// - `KY-E101` for duplicate IDs.
pub(crate) fn check_table(
    axis: TableAxis,
    frame: &DataFrame,
    plain_strings: PlainStrings,
) -> Result<(), KytoError> {
    let column_error =
        |column: &str, problem: String, accepted: String| KytoError::InvalidTableColumn {
            axis: axis.as_str().to_owned(),
            column: column.to_owned(),
            problem,
            accepted,
        };
    let id_name = axis.id_column();
    let Some(id_column) = frame
        .columns()
        .first()
        .filter(|column| column.name() == id_name)
    else {
        let problem = match frame
            .columns()
            .iter()
            .position(|column| column.name() == id_name)
        {
            Some(position) => format!("is column {position}, not the first column"),
            None => "is missing".to_owned(),
        };
        return Err(column_error(
            id_name,
            problem,
            format!("put the IDs first, in a String column named '{id_name}'"),
        ));
    };
    if id_column.dtype() != &DataType::String {
        return Err(column_error(
            id_name,
            format!("has dtype {}, not String", id_column.dtype()),
            format!("cast '{id_name}' to String"),
        ));
    }
    for column in &frame.columns()[1..] {
        check_column(column, plain_strings).map_err(|(problem, accepted)| {
            column_error(column.name().as_str(), problem, accepted)
        })?;
    }
    check_ids(axis, id_column)
}

/// Check the name and dtype of a non-ID column. Returns `(problem, accepted)`.
fn check_column(column: &Column, plain_strings: PlainStrings) -> Result<(), (String, String)> {
    let name = column.name().as_str();
    if name.is_empty() {
        return Err((
            "has an empty name".to_owned(),
            "give every column a name".to_owned(),
        ));
    }
    if name.contains(':') {
        return Err((
            "has ':' in its name".to_owned(),
            "rename it without ':', which feature names use as a separator".to_owned(),
        ));
    }
    if RESERVED_ID_COLUMNS.contains(&name) {
        return Err((
            "uses a reserved ID column name".to_owned(),
            "rename it; 'cell_id' and 'gene_id' are only the first column of their own table"
                .to_owned(),
        ));
    }
    match column_dtype(column.dtype()) {
        Some(ColumnDtype::String) if plain_strings == PlainStrings::Forbidden => Err((
            "has dtype String; only the ID column may".to_owned(),
            "store string columns as Enum".to_owned(),
        )),
        Some(_) => Ok(()),
        None => Err((
            format!("has dtype {}, which v0 stores do not allow", column.dtype()),
            ACCEPTED_DTYPES.to_owned(),
        )),
    }
}

/// Check that IDs are non-null, non-empty and unique (exact bytes, format.md §5.2).
fn check_ids(axis: TableAxis, id_column: &Column) -> Result<(), KytoError> {
    let ids = id_column
        .str()
        .map_err(|error| KytoError::InvalidTableColumn {
            axis: axis.as_str().to_owned(),
            column: axis.id_column().to_owned(),
            problem: error.to_string(),
            accepted: format!("cast '{}' to String", axis.id_column()),
        })?;
    let mut null_rows = Vec::new();
    let mut empty_rows = Vec::new();
    // Each ID's count; `duplicated` keeps duplicated IDs in order of first repeat.
    let mut counts: HashMap<&str, u64> = HashMap::with_capacity(ids.len());
    let mut duplicated: Vec<&str> = Vec::new();
    for (row, id) in ids.iter().enumerate() {
        match id {
            None => null_rows.push(row),
            Some("") => empty_rows.push(row),
            Some(id) => match counts.entry(id) {
                Entry::Vacant(entry) => {
                    entry.insert(1);
                }
                Entry::Occupied(mut entry) => {
                    if *entry.get() == 1 {
                        duplicated.push(id);
                    }
                    *entry.get_mut() += 1;
                }
            },
        }
    }
    for (rows, problem) in [(&null_rows, "null"), (&empty_rows, "empty")] {
        if !rows.is_empty() {
            return Err(KytoError::InvalidIds {
                axis: axis.as_str().to_owned(),
                n_invalid: rows.len() as u64,
                problem: problem.to_owned(),
                rows: rows.iter().take(SHOWN).map(usize::to_string).collect(),
            });
        }
    }
    if !duplicated.is_empty() {
        return Err(KytoError::DuplicateIds {
            axis: axis.as_str().to_owned(),
            n_duplicated: duplicated.len() as u64,
            ids: duplicated
                .iter()
                .take(SHOWN)
                .map(|&id| id.to_owned())
                .collect(),
        });
    }
    Ok(())
}
