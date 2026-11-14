//! `Atlas.to_anndata` (task 7.8): a plan's output as an `anndata.AnnData`, the input
//! scanpy and cellestial (kyto's plotting library) read.
//!
//! Does: check `var_names` and the cell embedding names before reading anything, run
//! the plan into a temporary `.h5ad` (5.9's sink, 2.7's export) and hand it to
//! `kyto._anndata.read`, which reads it with anndata and applies scanpy's names: cell
//! embeddings as `X_<name>` in `obsm`, and optionally a gene column as `var_names`.
//! Does not: give `Atlas` any `obs`/`var` access (R4), keep the file after returning,
//! or write anything kyto's stores do not hold (no `uns`: format v0 has none).

use std::collections::BTreeMap;
use std::sync::Arc;

use kyto_core::{Axis, ColumnDtype};
use kyto_error::KytoError;
use kyto_plan::{Plan, Sink};
use pyo3::prelude::*;

use crate::errors::to_py_err;

/// Each cell embedding's `obsm` key, by embedding name.
type ObsmKeys = BTreeMap<String, String>;

/// The prefix scanpy gives the cell embeddings in `obsm` (`X_pca`, `X_umap`).
const OBSM_PREFIX: &str = "X_";

/// Run `plan` and return its output as an `anndata.AnnData` (see `Atlas.to_anndata`).
///
/// # Errors
/// `KY-E215` for a `var_names` that is not a text gene column, `KY-E103` when two cell
/// embeddings would get the same `obsm` key, those of the sink (`KY-E215`, `KY-E401`,
/// `KY-E502`, `KY-E601`, a `KeyboardInterrupt`), `KY-E110` if the temporary folder
/// cannot be created, and `ModuleNotFoundError` without anndata (checked first).
pub fn to_anndata<'py>(
    py: Python<'py>,
    plan: &Arc<Plan>,
    var_names: Option<&str>,
    memory_limit: Option<&Bound<'_, PyAny>>,
    threads: Option<&Bound<'_, PyAny>>,
    progress: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(column) = var_names {
        check_var_names(plan, column).map_err(|error| to_py_err(py, &error))?;
    }
    let obsm_keys = obsm_keys(plan).map_err(|error| to_py_err(py, &error))?;
    // Fail before running when anndata is missing (it is not a kyto dependency).
    py.import("anndata")?;
    let folder = tempfile::Builder::new()
        .prefix("kyto-anndata-")
        .tempdir()
        .map_err(|error| {
            let error = KytoError::StoreIo {
                operation: "create a temporary folder for to_anndata()".to_owned(),
                path: std::env::temp_dir().display().to_string(),
                detail: error.to_string(),
            };
            to_py_err(py, &error)
        })?;
    let target = folder.path().join("atlas.h5ad");
    crate::materialize::sink(
        py,
        "to_anndata",
        plan,
        Sink::H5ad {
            path: target.display().to_string(),
        },
        false,
        memory_limit,
        threads,
        progress,
    )?;
    // anndata reads the whole file into memory; the folder is deleted on return.
    py.import("kyto._anndata")?
        .call_method1("read", (target.display().to_string(), obsm_keys, var_names))
}

/// `KY-E215` unless `column` is a `String` or `Enum` column of the plan's gene table.
fn check_var_names(plan: &Plan, column: &str) -> Result<(), KytoError> {
    let text_columns: Vec<&str> = plan
        .schema()
        .genes
        .columns
        .iter()
        .filter(|schema| matches!(schema.dtype, ColumnDtype::String | ColumnDtype::Enum { .. }))
        .map(|schema| schema.name.as_str())
        .collect();
    if text_columns.contains(&column) {
        return Ok(());
    }
    let problem = match plan
        .schema()
        .genes
        .columns
        .iter()
        .find(|schema| schema.name == column)
    {
        Some(schema) => format!(
            "gene column '{column}' holds {}, not text",
            schema.dtype.name()
        ),
        None => format!("the plan's gene table has no column '{column}'"),
    };
    let accepted = text_columns
        .iter()
        .map(|name| format!("'{name}'"))
        .collect::<Vec<_>>()
        .join(", ");
    Err(KytoError::InvalidArgument {
        function: "to_anndata".to_owned(),
        parameter: "var_names".to_owned(),
        value: format!("'{column}'"),
        problem,
        accepted: format!("None or a text gene column: {accepted}"),
    })
}

/// The `obsm` key of every cell embedding the plan outputs: `X_<name>`, or the name
/// itself when it already starts with `X_`. `KY-E103` when two embeddings get one key
/// (`pca` next to `X_pca`).
fn obsm_keys(plan: &Plan) -> Result<ObsmKeys, KytoError> {
    let names: Vec<&str> = plan
        .schema()
        .embeddings
        .iter()
        .filter(|embedding| embedding.axis == Axis::Cells)
        .map(|embedding| embedding.name.as_str())
        .collect();
    let mut keys = ObsmKeys::new();
    for name in &names {
        let key = obsm_key(name);
        if key != *name && names.contains(&key.as_str()) {
            return Err(KytoError::MemberExists {
                kind: "embedding".to_owned(),
                name: (*name).to_owned(),
                existing: key,
            });
        }
        keys.insert((*name).to_owned(), key);
    }
    Ok(keys)
}

/// scanpy's `obsm` key for the cell embedding `name`.
fn obsm_key(name: &str) -> String {
    if name.starts_with(OBSM_PREFIX) {
        name.to_owned()
    } else {
        format!("{OBSM_PREFIX}{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::obsm_key;

    #[test]
    fn obsm_keys_take_scanpy_prefix_once() {
        assert_eq!(obsm_key("pca"), "X_pca");
        assert_eq!(obsm_key("umap"), "X_umap");
        assert_eq!(obsm_key("X_pca"), "X_pca");
        // Case matters, as in anndata's keys.
        assert_eq!(obsm_key("x_pca"), "X_x_pca");
    }
}
