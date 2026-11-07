//! Duplicate cell and gene IDs on import (task 2.6, format.md §5.2).
//!
//! Does: [`OnDuplicateIds`] picks between refusing duplicates (`KY-E101`, the default)
//! and making them unique exactly as anndata's `obs_names_make_unique` /
//! `var_names_make_unique` do ([`make_unique`]).
//! Does not: touch null or empty IDs (they stay invalid, `KY-E111`).

use std::collections::{HashMap, HashSet};

use kyto_error::KytoError;
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};

use crate::table_axis::TableAxis;

/// What [`crate::convert`] does with IDs that occur more than once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnDuplicateIds {
    /// Refuse the input with `KY-E101` before writing anything.
    #[default]
    Error,
    /// Append `-1`, `-2`, ... to the second and later occurrences, skipping suffixes
    /// that are already IDs (anndata's `make_index_unique` with `join="-"`).
    Suffix,
}

impl OnDuplicateIds {
    /// Parse the Python argument: `"error"` or `"suffix"`.
    ///
    /// # Errors
    /// `KY-E104` for any other value.
    pub fn parse(value: &str) -> Result<Self, KytoError> {
        match value {
            "error" => Ok(Self::Error),
            "suffix" => Ok(Self::Suffix),
            other => Err(KytoError::InvalidParameter {
                parameter: "on_duplicate_ids".to_owned(),
                value: other.to_owned(),
                accepted: "'error' or 'suffix'".to_owned(),
            }),
        }
    }
}

/// Make the non-null `ids` unique in place, as anndata's `make_index_unique`.
pub(crate) fn make_unique(ids: &mut [Option<String>]) {
    let mut seen = HashSet::new();
    let duplicates: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter_map(|(position, id)| {
            let id = id.as_deref()?;
            (!seen.insert(id)).then_some(position)
        })
        .collect();
    if duplicates.is_empty() {
        return;
    }
    let mut taken: HashSet<String> = ids.iter().flatten().cloned().collect();
    let mut counters: HashMap<String, u64> = HashMap::new();
    for position in duplicates {
        let Some(id) = ids[position].clone() else {
            continue;
        };
        let counter = counters.entry(id.clone()).or_insert(0);
        loop {
            *counter += 1;
            let candidate = format!("{id}-{counter}");
            if taken.insert(candidate.clone()) {
                ids[position] = Some(candidate);
                break;
            }
        }
    }
}

/// Replace the ID column (the first) of `frame` with unique IDs.
pub(crate) fn suffix_ids(frame: &mut DataFrame, axis: TableAxis) -> Result<(), KytoError> {
    let name = axis.id_column();
    let Some(column) = frame.columns().first() else {
        return Ok(());
    };
    let Ok(strings) = column.str() else {
        return Ok(());
    };
    let mut ids: Vec<Option<String>> = strings.iter().map(|id| id.map(str::to_owned)).collect();
    make_unique(&mut ids);
    let replacement = Series::new(name.into(), ids).into_column();
    frame
        .replace_column(0, replacement)
        .map(|_| ())
        .map_err(|error| KytoError::StoreIo {
            operation: "make IDs unique".to_owned(),
            path: axis.as_str().to_owned(),
            detail: error.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::{OnDuplicateIds, make_unique};

    fn unique(ids: &[Option<&str>]) -> Vec<Option<String>> {
        let mut ids: Vec<Option<String>> = ids.iter().map(|id| id.map(str::to_owned)).collect();
        make_unique(&mut ids);
        ids
    }

    #[test]
    fn suffixes_match_anndata() {
        // anndata: make_index_unique(pd.Index(["a", "a", "b", "a-1", "a", None, None]))
        // gives ['a', 'a-2', 'b', 'a-1', 'a-3', nan, nan].
        let ids = unique(&[
            Some("a"),
            Some("a"),
            Some("b"),
            Some("a-1"),
            Some("a"),
            None,
            None,
        ]);
        let expected = [
            Some("a"),
            Some("a-2"),
            Some("b"),
            Some("a-1"),
            Some("a-3"),
            None,
            None,
        ];
        assert_eq!(ids, expected.map(|id| id.map(str::to_owned)));
        let ids = unique(&[Some("x"), Some("y")]);
        assert_eq!(ids, [Some("x".to_owned()), Some("y".to_owned())]);
    }

    #[test]
    fn parse_accepts_two_values() {
        assert_eq!(
            OnDuplicateIds::parse("error").unwrap(),
            OnDuplicateIds::Error
        );
        assert_eq!(
            OnDuplicateIds::parse("suffix").unwrap(),
            OnDuplicateIds::Suffix
        );
        let error = OnDuplicateIds::parse("rename").unwrap_err();
        assert_eq!(error.code(), "KY-E104");
    }
}
