//! The Python `Atlas`: an immutable lazy plan (task 3.2), and the private `scan` that
//! builds one.
//!
//! Does: wrap an `Arc<kyto_plan::Plan>` in a frozen pyclass whose every method returns a
//! new `Atlas` (R2); build the root plan from a store's `schema.json`.
//! Does not: read tables or the matrix; hold any state besides the plan (R7); expose
//! `__getitem__`, `.X`, `obs` or `var` (R4). The public `kyto.scan` and the chain
//! methods arrive with 7.1/7.2; `scan` is private until then.

use std::path::PathBuf;
use std::sync::Arc;

use kyto_plan::Plan;
use pyo3::prelude::*;

use crate::errors::to_py_err;

/// A lazy, immutable plan over a kyto store.
///
/// An ``Atlas`` records what to compute; it reads no matrix data. Every method returns
/// a new ``Atlas`` and leaves the one it was called on unchanged (R2), so an ``Atlas``
/// can be shared and reused freely. Plans built from one another share their common
/// part; nothing is copied.
///
/// Examples
/// --------
/// >>> atlas = scan(store)
/// >>> counts = atlas.with_default_layer("counts")
/// >>> print(atlas)
/// SCAN "..." (2 cells x 3 genes)
#[pyclass(frozen, module = "kyto", name = "Atlas")]
pub struct Atlas {
    plan: Arc<Plan>,
}

#[pymethods]
impl Atlas {
    /// Set the layer that short gene names refer to.
    ///
    /// Parameters
    /// ----------
    /// layer : str
    ///     Name of a layer of the store, as stored (case-sensitive).
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// MissingMemberError
    ///     ``KY-E109``: the store has no layer named ``layer``.
    ///
    /// Examples
    /// --------
    /// >>> atlas = scan(store)
    /// >>> print(atlas.with_default_layer("counts"))
    /// WITH_DEFAULT_LAYER "counts"
    /// SCAN "..." (2 cells x 3 genes)
    fn with_default_layer(&self, py: Python<'_>, layer: &str) -> PyResult<Self> {
        self.plan
            .with_default_layer(layer)
            .map(|plan| Self { plan })
            .map_err(|error| to_py_err(py, &error))
    }

    /// The plan, one node per line, from the last step down to the scan.
    fn __str__(&self) -> String {
        self.plan.to_string()
    }

    /// The plan, one node per line, inside ``<Atlas ...>``.
    fn __repr__(&self) -> String {
        format!("<Atlas\n{}\n>", self.plan)
    }
}

/// Build the root `Atlas` of the local store at `store` (private until 7.1).
#[pyfunction]
#[pyo3(signature = (store, /))]
pub fn scan(py: Python<'_>, store: PathBuf) -> PyResult<Atlas> {
    let schema = py
        .detach(|| kyto_store::open_atlas_schema(&store))
        .map_err(|error| to_py_err(py, &error))?;
    Ok(Atlas {
        plan: Plan::scan(store.display().to_string(), schema),
    })
}
