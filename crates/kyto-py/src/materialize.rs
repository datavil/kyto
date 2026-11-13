//! The store a handle terminal reads (task 7.3): the plan's output, written to disk
//! when the plan has steps.
//!
//! Does: give the path of a kyto store holding exactly the plan's output. A bare scan
//! (a `SCAN`, possibly under `WITH_DEFAULT_LAYER` steps, which change no data) is its
//! own store and is read in place; any other plan is run into a temporary store
//! (`kyto-terminal-*` in the system temporary directory, as 7.2's `collect()`; GIL,
//! progress and Ctrl-C through 5.7's `run_plan`), deleted when the [`Materialized`]
//! is dropped. A bare scan whose store changed since the scan is `KY-E127`. Also the sink that `Atlas.collect` and `sink_zarr` share ([`sink`]).
//! Does not: stream a member straight to Python without the temporary store (an Open
//! of 7.3), or convert anything to
//! Python objects (`handoff.rs`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use kyto_error::KytoError;
use kyto_plan::{Plan, Sink};
use pyo3::prelude::*;

use crate::errors::to_py_err;
use crate::run::run_plan;
use crate::terminal;

/// The store holding a plan's output, kept alive as long as this value.
pub struct Materialized {
    /// The store's path.
    pub store: PathBuf,
    /// The temporary store this terminal wrote, deleted on drop (`None` for a bare
    /// scan).
    _scratch: Option<tempfile::TempDir>,
    /// The `collect()` store a bare scan reads, kept alive while it is read.
    _collected: Option<Arc<tempfile::TempDir>>,
}

/// The store `plan` scans, if the plan is a bare scan: a `SCAN` under nothing but
/// `WITH_DEFAULT_LAYER` steps (they change no data).
fn bare_scan(plan: &Plan) -> Option<&str> {
    match plan {
        Plan::Scan(node) => Some(node.params().source()),
        Plan::WithDefaultLayer(node) => node.input().and_then(|input| bare_scan(input)),
        _ => None,
    }
}

/// Run `plan` into `target` (a kyto store or an `.h5ad` file; the sink of `collect`,
/// `sink_zarr`, `sink_h5ad` and the handle terminals), replacing an existing target
/// only with `overwrite`: `memory_limit`, `threads` and `progress` as those terminals
/// take them.
#[allow(clippy::too_many_arguments)]
pub fn sink(
    py: Python<'_>,
    function: &str,
    plan: &Arc<Plan>,
    target: Sink,
    overwrite: bool,
    memory_limit: Option<&Bound<'_, PyAny>>,
    threads: Option<&Bound<'_, PyAny>>,
    progress: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let plan = plan.sink(target).map_err(|error| to_py_err(py, &error))?;
    let run = terminal::run_arguments(py, function, &plan, memory_limit, threads, progress)?;
    run_plan(py, &run.physical, run.threads, progress, overwrite).map(|_| ())
}

/// A kyto store target at `path`.
pub fn zarr(path: &Path) -> Sink {
    Sink::Zarr {
        path: path.display().to_string(),
    }
}

/// The store holding `plan`'s output, for the terminal `function` (`Class.method`).
///
/// A bare scan is read in place (the run arguments are still checked: `KY-E215`);
/// any other plan is run into a temporary store first.
///
/// # Errors
/// Those of [`sink`] (`KY-E215`, `KY-E401`, `KY-E502`, `KY-E601` for a step the
/// executor does not run yet, a `KeyboardInterrupt`), and `KY-E110` if the temporary folder cannot be created.
pub fn materialize(
    py: Python<'_>,
    function: &str,
    plan: &Arc<Plan>,
    collected: Option<&Arc<tempfile::TempDir>>,
    memory_limit: Option<&Bound<'_, PyAny>>,
    threads: Option<&Bound<'_, PyAny>>,
    progress: Option<&Bound<'_, PyAny>>,
) -> PyResult<Materialized> {
    if let Some(source) = bare_scan(plan) {
        // Nothing runs, but the arguments mean the same on every plan.
        terminal::memory_limit(py, function, memory_limit)?;
        terminal::threads(py, function, threads)?;
        terminal::check_progress(py, function, progress)?;
        // The plan answers from the scan's snapshot (handles, estimates): a store
        // replaced since then is refused, never read as it is now (bug-hunter).
        let stored = py
            .detach(|| kyto_store::open_atlas_schema(Path::new(source)))
            .map_err(|error| to_py_err(py, &error))?;
        if &stored != plan.schema() {
            let error = KytoError::StoreChanged {
                store: source.to_owned(),
                problem: "its schema.json is not the one the plan read".to_owned(),
            };
            return Err(to_py_err(py, &error));
        }
        return Ok(Materialized {
            store: PathBuf::from(source),
            _scratch: None,
            _collected: collected.cloned(),
        });
    }
    let folder = tempfile::Builder::new()
        .prefix("kyto-terminal-")
        .tempdir()
        .map_err(|error| {
            let error = KytoError::StoreIo {
                operation: format!("create a temporary folder for {function}()"),
                path: std::env::temp_dir().display().to_string(),
                detail: error.to_string(),
            };
            to_py_err(py, &error)
        })?;
    let store = folder.path().join("atlas.kyto");
    sink(
        py,
        function,
        plan,
        zarr(&store),
        false,
        memory_limit,
        threads,
        progress,
    )?;
    Ok(Materialized {
        store,
        _scratch: Some(folder),
        _collected: None,
    })
}
