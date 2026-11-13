//! `Atlas.rank_genes_groups` (task 7.2 over 6.10): differential expression as a lazy
//! Polars frame.
//!
//! Does: check the arguments and the plan's members when the method is called (R5, R6:
//! the layer and the `groupby` column must be in the plan's output), then return a
//! `pl.LazyFrame` built with `pl.defer`: the plan runs (`kyto_algo::rank_genes_groups`,
//! GIL released) only when the frame is collected (R1), and the result is handed to a
//! new `pl.DataFrame` of the six documented columns without a copy (7.5).
//! Does not: report progress or react to Ctrl-C while it runs (`run_plan` drains a
//! pipeline; DE consumes its own).

use std::num::NonZeroUsize;
use std::sync::Arc;

use kyto_algo::{RANK_GENES_COLUMNS, RankGenesGroups, rank_genes_groups};
use kyto_error::KytoError;
use kyto_exec::ExecuteOptions;
use kyto_plan::{Member, Output, PhysicalOptions, Plan};
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict};

use crate::errors::to_py_err;

/// Check `parameters` against `plan` without reading data: the layer (or the default
/// layer, `KY-E211`) and the `groupby` column must be in the plan's output (`KY-E109`).
pub fn check(plan: &Arc<Plan>, parameters: &RankGenesGroups) -> Result<(), KytoError> {
    let layer =
        match &parameters.layer {
            Some(layer) => layer.clone(),
            None => plan.default_layer().map(str::to_owned).ok_or_else(|| {
                KytoError::NoDefaultLayer {
                    expression: "rank_genes_groups(layer=None)".to_owned(),
                    layers: plan
                        .schema()
                        .layers
                        .iter()
                        .map(|layer| layer.name.clone())
                        .collect(),
                }
            })?,
        };
    Output::members(
        plan,
        [
            Member::Layer(layer),
            Member::CellColumn(parameters.groupby.clone()),
        ],
    )
    .map(|_| ())
}

/// The lazy frame: `pl.defer` over a function that runs the test when collected.
pub fn lazy_frame<'py>(
    py: Python<'py>,
    plan: Arc<Plan>,
    collected: Option<Arc<tempfile::TempDir>>,
    parameters: RankGenesGroups,
    threads: NonZeroUsize,
) -> PyResult<Bound<'py, PyAny>> {
    let polars = py.import("polars")?;
    let schema = PyDict::new(py);
    for (index, name) in RANK_GENES_COLUMNS.iter().enumerate() {
        let dtype = if index < 2 { "String" } else { "Float64" };
        schema.set_item(*name, polars.getattr(dtype)?)?;
    }
    let run = PyCFunction::new_closure(
        py,
        None,
        None,
        move |arguments: &Bound<'_, pyo3::types::PyTuple>,
              _: Option<&Bound<'_, PyDict>>|
              -> PyResult<Py<PyAny>> {
            let py = arguments.py();
            // The frame keeps the plan's store alive (a collected Atlas's temporary
            // store) as long as it can be collected.
            let _store = &collected;
            let frame = py
                .detach(|| {
                    rank_genes_groups(
                        &plan,
                        &parameters,
                        PhysicalOptions::default(),
                        &ExecuteOptions::new(threads),
                    )
                })
                .map_err(|error| to_py_err(py, &error))?;
            to_polars(py, &frame).map(Bound::unbind)
        },
    )?;
    let keywords = PyDict::new(py);
    keywords.set_item("schema", schema)?;
    polars.call_method("defer", (run,), Some(&keywords))
}

/// The frame's six columns, in order, as a new `pl.DataFrame` (handed over without a
/// copy, `handoff::to_polars`).
fn to_polars<'py>(py: Python<'py>, frame: &DataFrame) -> PyResult<Bound<'py, PyAny>> {
    // The frame's columns and dtypes are kyto-algo's contract (`RANK_GENES_COLUMNS`,
    // tested there): a mismatch is a bug, not a user error.
    let Ok(frame) = frame.select(RANK_GENES_COLUMNS.iter().copied()) else {
        unreachable!("rank_genes_groups returns every column of RANK_GENES_COLUMNS")
    };
    crate::handoff::to_polars(py, frame)
}
