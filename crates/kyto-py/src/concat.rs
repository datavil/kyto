//! Binding of `kyto.concat` (task 2.9).
//!
//! Does: turn the Python arguments into [`kyto_store::ConcatOptions`], run
//! [`kyto_store::concat`] with the GIL released, and raise its coded error.
//! Does not: check paths or read anything itself; the docstring and the public
//! signature live in `python/kyto/_concat.py`.

use std::path::PathBuf;

use kyto_store::{ConcatOptions, ConvertOptions, Join, OnDuplicateIds};
use pyo3::prelude::*;

use crate::errors::to_py_err;

/// Combine `sources` into a new kyto store at `target`.
#[pyfunction]
#[pyo3(signature = (sources, target, /, *, label, keys, join, layer, on_duplicate_ids, overwrite))]
#[allow(clippy::too_many_arguments)]
pub fn concat(
    py: Python<'_>,
    sources: Vec<PathBuf>,
    target: PathBuf,
    label: String,
    keys: Option<Vec<String>>,
    join: &str,
    layer: String,
    on_duplicate_ids: &str,
    overwrite: bool,
) -> PyResult<()> {
    let join = Join::parse(join).map_err(|error| to_py_err(py, &error))?;
    let on_duplicate_ids =
        OnDuplicateIds::parse(on_duplicate_ids).map_err(|error| to_py_err(py, &error))?;
    let options = ConcatOptions {
        label,
        keys,
        join,
        on_duplicate_ids,
        overwrite,
        convert: ConvertOptions {
            layer,
            on_duplicate_ids,
            ..ConvertOptions::default()
        },
    };
    py.detach(|| kyto_store::concat(&sources, &target, &options))
        .map_err(|error| to_py_err(py, &error))
}
