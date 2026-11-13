//! The tables of a combined store (task 2.9): `anndata.concat`'s rules for the cell and
//! gene tables.
//!
//! Does: [`join_ids`] joins the sources' gene IDs as pandas `Index.intersection`
//! (`"inner"`, order of the first source) or `Index.union` (`"outer"`, sorted unless all
//! sources are equal); [`merge_gene_table`] keeps the gene columns every source has with
//! equal values over the merged genes (anndata `merge="same"`); [`concat_cell_tables`]
//! stacks the cell tables as `pd.concat(join=...)` after anndata's `unify_dtypes`
//! (categoricals get the natural-sorted union of their categories) and adds the label
//! column.
//! Does not: read or write anything, or check the IDs (the store writer does).

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use kyto_error::KytoError;
use polars::prelude::{
    Column, DataFrame, DataType, FrozenCategories, IdxCa, IntoColumn, NamedFrom, Series,
};

use crate::table_axis::TableAxis;

/// How the gene axes of the sources are joined (`anndata.concat(join=...)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Join {
    /// Genes every source has, in the first source's order.
    #[default]
    Inner,
    /// Genes any source has; absent values are 0 (R9).
    Outer,
}

impl Join {
    /// Parse the Python argument: `"inner"` or `"outer"`.
    ///
    /// # Errors
    /// `KY-E104` for any other value.
    pub fn parse(value: &str) -> Result<Self, KytoError> {
        match value {
            "inner" => Ok(Self::Inner),
            "outer" => Ok(Self::Outer),
            other => Err(KytoError::InvalidParameter {
                parameter: "join".to_owned(),
                value: other.to_owned(),
                accepted: "'inner' or 'outer'".to_owned(),
            }),
        }
    }
}

/// One source's table and the `Enum` columns it marks ordered.
pub(crate) struct SourceTable<'a> {
    pub frame: &'a DataFrame,
    pub ordered: Vec<String>,
}

/// The gene IDs of the combined store, joined as pandas joins the `var` indexes.
pub(crate) fn join_ids(sources: &[Vec<String>], join: Join) -> Vec<String> {
    let mut sources = sources.iter();
    let Some(first) = sources.next() else {
        return Vec::new();
    };
    sources.fold(first.clone(), |merged, ids| match join {
        // pandas keeps the left order (both monotonic gives the same order).
        Join::Inner => {
            let right: HashSet<&str> = ids.iter().map(String::as_str).collect();
            merged
                .into_iter()
                .filter(|id| right.contains(id.as_str()))
                .collect()
        }
        // pandas returns the left index when both are equal or the right one is
        // empty, the right one when the left is empty, and the sorted union otherwise.
        Join::Outer => {
            if &merged == ids || ids.is_empty() {
                merged
            } else if merged.is_empty() {
                ids.clone()
            } else {
                let mut union: Vec<String> = merged;
                let seen: HashSet<String> = union.iter().cloned().collect();
                union.extend(ids.iter().filter(|id| !seen.contains(*id)).cloned());
                union.sort_unstable();
                union
            }
        }
    })
}

/// For each merged gene, its row in a source's gene table (`None` if absent).
pub(crate) fn source_rows(merged: &[String], ids: &[String]) -> Vec<Option<u64>> {
    let rows: HashMap<&str, u64> = ids
        .iter()
        .enumerate()
        .map(|(row, id)| (id.as_str(), row as u64))
        .collect();
    merged
        .iter()
        .map(|id| rows.get(id.as_str()).copied())
        .collect()
}

/// The combined gene table: `gene_id` as merged, then every column all sources have
/// (first source's order) whose values over the merged genes are equal in all of them,
/// nulls in the same places (anndata `merge="same"`). A kept column has the first
/// source's dtype; `Enum` values compare as text. Returns the table and the kept
/// columns the first source marks ordered.
///
/// # Errors
/// `KY-E110` if Polars fails.
pub(crate) fn merge_gene_table(
    sources: &[SourceTable<'_>],
    merged: &[String],
    rows: &[Vec<Option<u64>>],
) -> Result<(DataFrame, Vec<String>), KytoError> {
    let id = TableAxis::Genes.id_column();
    let mut columns = vec![Series::new(id.into(), merged).into_column()];
    let mut ordered = Vec::new();
    let Some(first) = sources.first() else {
        return frame(merged.len(), columns).map(|frame| (frame, ordered));
    };
    for column in first.frame.columns().iter().skip(1) {
        let name = column.name().as_str();
        let Some(reindexed) = sources
            .iter()
            .zip(rows)
            .map(|(source, rows)| {
                source
                    .frame
                    .column(name)
                    .ok()
                    .map(|column| take_rows(column, rows))
            })
            .collect::<Option<Result<Vec<Column>, KytoError>>>()
            .transpose()?
        else {
            continue;
        };
        if all_equal(&reindexed)? {
            if first.ordered.iter().any(|ordered| ordered == name) {
                ordered.push(name.to_owned());
            }
            columns.push(
                reindexed
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| column.clone()),
            );
        }
    }
    frame(merged.len(), columns).map(|frame| (frame, ordered))
}

/// Rows `rows` of `column` (null where `None`).
fn take_rows(column: &Column, rows: &[Option<u64>]) -> Result<Column, KytoError> {
    let positions: Vec<Option<u32>> = rows
        .iter()
        .map(|row| row.and_then(|row| u32::try_from(row).ok()))
        .collect();
    column
        .take(&IdxCa::new("".into(), positions))
        .map_err(polars_error)
}

/// Whether the columns hold the same values and nulls (and dtypes, `Enum` as text).
fn all_equal(columns: &[Column]) -> Result<bool, KytoError> {
    let comparable = |column: &Column| -> Result<Series, KytoError> {
        let series = column.as_materialized_series().clone();
        match series.dtype() {
            DataType::Enum(..) => series.cast(&DataType::String).map_err(polars_error),
            _ => Ok(series),
        }
    };
    let Some(first) = columns.first() else {
        return Ok(true);
    };
    let first = comparable(first)?;
    for column in &columns[1..] {
        let other = comparable(column)?;
        if other.dtype() != first.dtype() || !other.equals_missing(&first) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The combined cell table: the sources' tables stacked, then `label` naming each
/// cell's source by `keys` (an `Enum` with categories `keys`, replacing a column of
/// that name in place, else last). Columns are those of every source (`"inner"`, first
/// source's order) or of any source (`"outer"`, order of appearance); a source without
/// a column gives nulls. Dtypes as anndata unifies them: `Enum` in every source gives
/// an `Enum` of the natural-sorted union of categories (unordered, unless every source
/// with categories has the same ordered ones); otherwise the Polars supertype (an
/// `Enum` mixed with other dtypes becomes text). Returns the table and its ordered
/// columns.
///
/// # Errors
/// `KY-E112` if a column's dtypes have no common type; `KY-E110` if Polars fails.
pub(crate) fn concat_cell_tables(
    sources: &[SourceTable<'_>],
    join: Join,
    label: &str,
    keys: &[String],
) -> Result<(DataFrame, Vec<String>), KytoError> {
    let id = TableAxis::Cells.id_column();
    let heights: Vec<usize> = sources.iter().map(|source| source.frame.height()).collect();
    let n_cells: usize = heights.iter().sum();
    let mut names: Vec<String> = Vec::new();
    for source in sources {
        for column in source.frame.columns().iter().skip(1) {
            let name = column.name().to_string();
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    if join == Join::Inner {
        names.retain(|name| {
            sources
                .iter()
                .all(|source| source.frame.column(name).is_ok())
        });
        // pandas keeps the first frame's order for the intersection.
        if let Some(first) = sources.first() {
            let order: Vec<String> = first
                .frame
                .columns()
                .iter()
                .map(|column| column.name().to_string())
                .collect();
            names.sort_by_key(|name| order.iter().position(|other| other == name));
        }
    }
    let mut columns = vec![stack(id, sources, &DataType::String)?.into_column()];
    let mut ordered = Vec::new();
    for name in &names {
        if name == label {
            columns.push(label_column(label, keys, &heights)?);
            continue;
        }
        let (column, is_ordered) = unified_column(name, sources, n_cells)?;
        if is_ordered {
            ordered.push(name.clone());
        }
        columns.push(column);
    }
    if !names.iter().any(|name| name == label) {
        columns.push(label_column(label, keys, &heights)?);
    }
    frame(n_cells, columns).map(|frame| (frame, ordered))
}

/// One column of the combined cell table, and whether it is an ordered `Enum`.
fn unified_column(
    name: &str,
    sources: &[SourceTable<'_>],
    n_cells: usize,
) -> Result<(Column, bool), KytoError> {
    let with_column: Vec<(&SourceTable<'_>, &Column)> = sources
        .iter()
        .filter_map(|source| {
            source
                .frame
                .column(name)
                .ok()
                .map(|column| (source, column))
        })
        .collect();
    // A source without cells adds no values, so its dtype does not decide (unless no
    // source has cells).
    let with_cells: Vec<(&SourceTable<'_>, &Column)> = with_column
        .iter()
        .filter(|(source, _)| source.frame.height() > 0)
        .copied()
        .collect();
    let present = if with_cells.is_empty() {
        with_column
    } else {
        with_cells
    };
    let all_enum = present
        .iter()
        .all(|(_, column)| matches!(column.dtype(), DataType::Enum(..)));
    if all_enum {
        let categories = enum_categories(name, &present);
        let stacked = stack(name, sources, &DataType::String)?;
        let (frozen, is_ordered) = match categories {
            EnumCategories::Ordered(categories) => (categories, true),
            EnumCategories::Unordered(categories) => (categories, false),
        };
        let frozen =
            FrozenCategories::new(frozen.iter().map(String::as_str)).map_err(polars_error)?;
        let column = stacked
            .strict_cast(&DataType::from_frozen_categories(frozen))
            .map_err(polars_error)?;
        debug_assert_eq!(column.len(), n_cells);
        return Ok((column.into_column(), is_ordered));
    }
    let mut dtype: Option<DataType> = None;
    for (_, column) in &present {
        let column_dtype = match column.dtype() {
            DataType::Enum(..) => DataType::String,
            other => other.clone(),
        };
        dtype = Some(match dtype {
            None => column_dtype,
            Some(dtype) => common_dtype(&dtype, &column_dtype).ok_or_else(
                || KytoError::InvalidTableColumn {
                    axis: TableAxis::Cells.as_str().to_owned(),
                    column: name.to_owned(),
                    problem: format!(
                        "has dtypes {dtype} and {column_dtype} in different sources, which have no common type"
                    ),
                    accepted: "give the column one dtype in every source, or drop it from all but one".to_owned(),
                },
            )?,
        });
    }
    let dtype = dtype.unwrap_or(DataType::Null);
    let column = stack(name, sources, &dtype)?;
    Ok((column.into_column(), false))
}

/// The dtype pandas gives two numeric columns stacked: a float if either is one
/// (`float32` from `float32` and integers of up to 2 bytes), else the narrowest integer holding both (`float64` for
/// `uint64` with a signed one). `None` for other mixes (pandas would give `object`).
fn common_dtype(left: &DataType, right: &DataType) -> Option<DataType> {
    use DataType::{Float32, Float64, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64};
    if left == right {
        return Some(left.clone());
    }
    // (is float, is signed, bytes)
    let kind = |dtype: &DataType| -> Option<(bool, bool, u8)> {
        Some(match dtype {
            Int8 => (false, true, 1),
            Int16 => (false, true, 2),
            Int32 => (false, true, 4),
            Int64 => (false, true, 8),
            UInt8 => (false, false, 1),
            UInt16 => (false, false, 2),
            UInt32 => (false, false, 4),
            UInt64 => (false, false, 8),
            Float32 => (true, true, 4),
            Float64 => (true, true, 8),
            _ => return None,
        })
    };
    let ((left_float, left_signed, left_bytes), (right_float, right_signed, right_bytes)) =
        (kind(left)?, kind(right)?);
    if left_float || right_float {
        // numpy: float32 holds integers of up to 2 bytes exactly.
        let fits_float32 = |float: bool, bytes: u8| if float { bytes == 4 } else { bytes <= 2 };
        return Some(
            if fits_float32(left_float, left_bytes) && fits_float32(right_float, right_bytes) {
                Float32
            } else {
                Float64
            },
        );
    }
    let signed_of = |bytes: u8| match bytes {
        1 => Int8,
        2 => Int16,
        4 => Int32,
        _ => Int64,
    };
    Some(match (left_signed, right_signed) {
        (true, true) => signed_of(left_bytes.max(right_bytes)),
        (false, false) => match left_bytes.max(right_bytes) {
            1 => UInt8,
            2 => UInt16,
            4 => UInt32,
            _ => UInt64,
        },
        _ => {
            let (signed, unsigned) = if left_signed {
                (left_bytes, right_bytes)
            } else {
                (right_bytes, left_bytes)
            };
            if unsigned == 8 {
                Float64
            } else {
                signed_of(signed.max(unsigned * 2))
            }
        }
    })
}

/// The categories of a column that is an `Enum` in every source that has it.
enum EnumCategories {
    Ordered(Vec<String>),
    Unordered(Vec<String>),
}

/// anndata's `try_unifying_dtype` and `pd.concat` for categoricals: the first source's
/// categories if they hold all others, else the union natural-sorted, unordered; or the categories of every non-empty source when all
/// are ordered and equal. (anndata gives an object column for other ordered mixes;
/// kyto keeps an unordered `Enum`, see deviations.md.)
fn enum_categories(name: &str, present: &[(&SourceTable<'_>, &Column)]) -> EnumCategories {
    let lists: Vec<(Vec<String>, bool)> = present
        .iter()
        .map(|(source, column)| {
            let categories = match column.dtype() {
                DataType::Enum(categories, _) => categories
                    .categories()
                    .iter()
                    .flatten()
                    .map(str::to_owned)
                    .collect(),
                _ => Vec::new(),
            };
            let is_ordered = source.ordered.iter().any(|ordered| ordered == name);
            (categories, is_ordered)
        })
        .collect();
    let non_empty: Vec<&(Vec<String>, bool)> = lists
        .iter()
        .filter(|(categories, _)| !categories.is_empty())
        .collect();
    if lists.iter().any(|(_, is_ordered)| *is_ordered)
        && let Some((first, _)) = non_empty.first()
        && non_empty
            .iter()
            .all(|(categories, is_ordered)| *is_ordered && categories == first)
    {
        return EnumCategories::Ordered(first.clone());
    }
    // pandas' `Index.union` of the categories, then natsort's stable sort.
    let all: Vec<Vec<String>> = lists
        .into_iter()
        .map(|(categories, _)| categories)
        .collect();
    let mut union = join_ids(&all, Join::Outer);
    // pandas' `astype` keeps a categorical whose categories are the same set (unordered
    // dtypes compare equal), and `pd.concat` takes the first frame's categories: the
    // first source's order wins when it holds every category.
    if let Some(first) = all.first()
        && first.len() == union.len()
    {
        return EnumCategories::Unordered(first.clone());
    }
    union.sort_by(|left, right| natural_order(left, right));
    EnumCategories::Unordered(union)
}

/// The `label` column: `keys[s]` for every cell of source `s`, an `Enum` of `keys`.
fn label_column(label: &str, keys: &[String], heights: &[usize]) -> Result<Column, KytoError> {
    let values: Vec<&str> = keys
        .iter()
        .zip(heights)
        .flat_map(|(key, height)| std::iter::repeat_n(key.as_str(), *height))
        .collect();
    let frozen = FrozenCategories::new(keys.iter().map(String::as_str)).map_err(polars_error)?;
    Series::new(label.into(), values)
        .strict_cast(&DataType::from_frozen_categories(frozen))
        .map(IntoColumn::into_column)
        .map_err(polars_error)
}

/// Column `name` of every source cast to `dtype` and stacked (nulls for a source
/// without it).
fn stack(name: &str, sources: &[SourceTable<'_>], dtype: &DataType) -> Result<Series, KytoError> {
    let mut stacked = Series::new_empty(name.into(), dtype);
    for source in sources {
        let part = match source.frame.column(name) {
            Ok(_) if source.frame.height() == 0 => Series::new_empty(name.into(), dtype),
            Ok(column) => column
                .as_materialized_series()
                .cast(dtype)
                .map_err(polars_error)?,
            Err(_) => Series::full_null(name.into(), source.frame.height(), dtype),
        };
        stacked.append(&part).map_err(polars_error)?;
    }
    Ok(stacked.rechunk())
}

/// The order of natsort's `natsorted` (default algorithm): text split into runs of
/// digits, compared as numbers, and other text, compared by code point; a key starting
/// with a digit gets an empty text run first. Equal keys compare equal (the sort is
/// stable, as Python's).
pub(crate) fn natural_order(left: &str, right: &str) -> Ordering {
    let (left, right) = (natural_key(left), natural_key(right));
    for (left, right) in left.iter().zip(&right) {
        let order = match (left, right) {
            (Run::Text(left), Run::Text(right)) => left.cmp(right),
            (Run::Number(left), Run::Number(right)) => {
                let (left, right) = (left.trim_start_matches('0'), right.trim_start_matches('0'));
                left.len().cmp(&right.len()).then_with(|| left.cmp(right))
            }
            // Runs alternate from an initial text run, so kinds always match by position.
            (Run::Text(_), Run::Number(_)) => Ordering::Less,
            (Run::Number(_), Run::Text(_)) => Ordering::Greater,
        };
        if order != Ordering::Equal {
            return order;
        }
    }
    left.len().cmp(&right.len())
}

/// One run of a natural sort key.
enum Run<'a> {
    Text(&'a str),
    Number(&'a str),
}

/// Split into alternating text and digit runs, starting with a (maybe empty) text run.
fn natural_key(text: &str) -> Vec<Run<'_>> {
    let mut runs = Vec::new();
    let mut start = 0;
    let mut digits = false;
    for (position, character) in text.char_indices() {
        let is_digit = character.is_ascii_digit();
        if is_digit != digits {
            let run = &text[start..position];
            runs.push(if digits {
                Run::Number(run)
            } else {
                Run::Text(run)
            });
            start = position;
            digits = is_digit;
        }
    }
    let run = &text[start..];
    if digits {
        runs.push(Run::Number(run));
    } else if !run.is_empty() {
        runs.push(Run::Text(run));
    }
    // natsort drops a trailing empty text run, keeps a leading one.
    if runs.is_empty() {
        runs.push(Run::Text(""));
    }
    runs
}

fn frame(height: usize, columns: Vec<Column>) -> Result<DataFrame, KytoError> {
    DataFrame::new(height, columns).map_err(polars_error)
}

fn polars_error(error: impl std::fmt::Display) -> KytoError {
    KytoError::StoreIo {
        operation: "combine tables".to_owned(),
        path: "concat".to_owned(),
        detail: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use polars::prelude::DataType;

    use super::{Join, common_dtype, join_ids, natural_order};

    #[test]
    fn numeric_dtypes_combine_as_pandas() {
        use DataType::{
            Float32, Float64, Int8, Int16, Int32, Int64, String, UInt8, UInt16, UInt32, UInt64,
        };
        // pd.concat([pd.Series([1], dtype=a), pd.Series([1], dtype=b)]).dtype
        let cases = [
            (UInt32, Int16, Some(Int64)),
            (UInt64, Int8, Some(Float64)),
            (Float32, Int8, Some(Float32)),
            (Float32, Int32, Some(Float64)),
            (UInt8, UInt16, Some(UInt16)),
            (Int8, UInt8, Some(Int16)),
            (Float32, UInt16, Some(Float32)),
            (String, Int8, None),
        ];
        for (left, right, expected) in cases {
            assert_eq!(common_dtype(&left, &right), expected, "{left} + {right}");
            assert_eq!(common_dtype(&right, &left), expected, "{right} + {left}");
        }
    }

    fn ids(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn gene_joins_match_pandas() {
        // pd.Index(["c","a","b"]).intersection(pd.Index(["b","c","d"])) -> ['c', 'b']
        let sources = [ids(&["c", "a", "b"]), ids(&["b", "c", "d"])];
        assert_eq!(join_ids(&sources, Join::Inner), ids(&["c", "b"]));
        // .union(...) -> ['a', 'b', 'c', 'd'] (sorted)
        assert_eq!(join_ids(&sources, Join::Outer), ids(&["a", "b", "c", "d"]));
        // Equal indexes keep their order: pd.Index(["c","a"]).union(pd.Index(["c","a"])).
        let equal = [ids(&["c", "a"]), ids(&["c", "a"])];
        assert_eq!(join_ids(&equal, Join::Outer), ids(&["c", "a"]));
        // An empty side returns the other unsorted.
        let empty = [ids(&["c", "a"]), ids(&[])];
        assert_eq!(join_ids(&empty, Join::Outer), ids(&["c", "a"]));
    }

    #[test]
    fn natural_order_matches_natsort() {
        // natsorted(["a10", "a2", "b", "10", "2", "a02", ""]) ->
        // ['', '2', '10', 'a2', 'a02', 'a10', 'b']
        let mut values = vec!["a10", "a2", "b", "10", "2", "a02", ""];
        values.sort_by(|left, right| natural_order(left, right));
        assert_eq!(values, ["", "2", "10", "a2", "a02", "a10", "b"]);
    }
}
