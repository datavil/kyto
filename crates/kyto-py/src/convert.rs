//! Binding of `kyto.convert` (task 2.6).
//!
//! Does: turn the Python arguments into [`kyto_store::ConvertOptions`], run
//! [`kyto_store::convert`] with the GIL released, and raise its coded error.
//! Does not: check paths or read anything itself; the docstring and the public
//! signature live in `python/kyto/_convert.py`.

use std::path::PathBuf;

use kyto_store::{ConvertOptions, OnDuplicateIds};
use pyo3::prelude::*;

use crate::errors::to_py_err;

/// Import `source` into a new kyto store at `target`.
#[pyfunction]
#[pyo3(signature = (source, target, /, *, layer, on_duplicate_ids, overwrite))]
pub fn convert(
    py: Python<'_>,
    source: PathBuf,
    target: PathBuf,
    layer: String,
    on_duplicate_ids: &str,
    overwrite: bool,
) -> PyResult<()> {
    let on_duplicate_ids =
        OnDuplicateIds::parse(on_duplicate_ids).map_err(|error| to_py_err(py, &error))?;
    let options = ConvertOptions {
        layer,
        on_duplicate_ids,
        overwrite,
        ..ConvertOptions::default()
    };
    py.detach(|| kyto_store::convert(&source, &target, &options))
        .map_err(|error| to_py_err(py, &error))
}
