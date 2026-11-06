//! Converts a [`KytoError`] into the matching Python class from `kyto.errors`.
//!
//! Does: build the registered Python exception (code, fields) for a Rust error, and
//! expose test hooks that raise every registered example (round trip, task 8.1).
//! Does not: render messages (Python renders them from the same `errors.toml`
//! templates; the round-trip test checks both renderings agree) or define exceptions.

use kyto_error::{FieldValue, KytoError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// Convert a Rust error into its `kyto.errors` exception, ready to return to Python.
///
/// If the Python side cannot be built (an internal bug: `kyto.errors` is out of date
/// with the extension), the error that stopped it is returned instead.
pub fn to_py_err(py: Python<'_>, error: &KytoError) -> PyErr {
    match build(py, error) {
        Ok(converted) => converted,
        Err(failure) => failure,
    }
}

fn build(py: Python<'_>, error: &KytoError) -> PyResult<PyErr> {
    let class = py.import("kyto.errors")?.getattr(error.python_class())?;
    let fields = PyDict::new(py);
    for (name, value) in error.fields() {
        match value {
            FieldValue::Str(text) => fields.set_item(name, text)?,
            FieldValue::Int(number) => fields.set_item(name, number)?,
            FieldValue::StrList(items) => fields.set_item(name, PyTuple::new(py, items)?)?,
        }
    }
    let instance = class.call((error.code(),), Some(&fields))?;
    Ok(PyErr::from_value(instance))
}

fn example(code: &str) -> Option<KytoError> {
    KytoError::examples()
        .into_iter()
        .find(|error| error.code() == code)
}

/// Test hook: raise the registered example error for `code` as its Python class.
///
/// Returns `False` (raises nothing) if `code` is not registered.
#[pyfunction]
pub fn raise_error_example(py: Python<'_>, code: &str) -> PyResult<bool> {
    match example(code) {
        Some(error) => Err(to_py_err(py, &error)),
        None => Ok(false),
    }
}

/// Test hook: the Rust `Display` text and hint of the example error for `code`.
///
/// Returns `None` if `code` is not registered.
#[pyfunction]
pub fn error_example_text(code: &str) -> Option<(String, String)> {
    example(code).map(|error| (error.to_string(), error.hint()))
}
