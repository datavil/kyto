//! Private binding `kyto._kyto.sink_h5ad` (task 2.7).
//!
//! Does: turn the Python arguments into [`kyto_store::SinkH5adOptions`], run
//! [`kyto_store::sink_h5ad`] with the GIL released, and raise its coded error.
//! Does not: define the public surface: `sink_h5ad()` becomes a terminal of the lazy
//! plan in 7.2/7.3 (user decision, Nov 7, 2026); until then it is private, for tests.

use std::path::PathBuf;

use kyto_store::SinkH5adOptions;
use pyo3::prelude::*;

use crate::errors::to_py_err;

/// Export the store at `store` as an `.h5ad` file at `target`.
#[pyfunction]
#[pyo3(signature = (store, target, /, *, x_layer, overwrite))]
pub fn sink_h5ad(
    py: Python<'_>,
    store: PathBuf,
    target: PathBuf,
    x_layer: Option<String>,
    overwrite: bool,
) -> PyResult<()> {
    let options = SinkH5adOptions {
        x_layer,
        overwrite,
        ..SinkH5adOptions::default()
    };
    py.detach(|| kyto_store::sink_h5ad(&store, &target, &options))
        .map_err(|error| to_py_err(py, &error))
}
