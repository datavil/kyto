//! The arguments every terminal of `Atlas` shares (task 7.2): `memory_limit`, `threads`
//! and `progress`, checked and turned into a physical plan and a thread count.
//!
//! Does: check `threads` (at least 1, `None` for one per core) and `progress` (callable
//! or `None`) with `KY-E215`; parse `memory_limit` (`"4GB"` or a byte count, task 5.5)
//! and fit the physical plan to it (`KY-E401` before anything runs).
//! Does not: run the plan (`run::run_plan` does) or hold any state (R7).

use std::num::{NonZeroU64, NonZeroUsize};

use kyto_error::KytoError;
use kyto_plan::{MemoryLimit, PhysicalOptions, PhysicalPlan, Plan};
use pyo3::prelude::*;
use pyo3::types::PyString;
use std::sync::Arc;

use crate::errors::to_py_err;

/// How a terminal runs: the physical plan (fitted to the memory limit) and its threads.
pub struct RunArguments {
    /// The physical plan, fitted to the memory limit.
    pub physical: PhysicalPlan,
    /// The threads of the run.
    pub threads: NonZeroUsize,
}

/// `KY-E215` for argument `parameter` of `function`.
pub fn invalid(
    function: &str,
    parameter: &str,
    value: String,
    problem: &str,
    accepted: &str,
) -> KytoError {
    KytoError::InvalidArgument {
        function: function.to_owned(),
        parameter: parameter.to_owned(),
        value,
        problem: problem.to_owned(),
        accepted: accepted.to_owned(),
    }
}

/// The threads of a run: `None` is one per CPU core; `KY-E215` for anything but an int
/// of 1 or more.
pub fn threads(
    py: Python<'_>,
    function: &str,
    threads: Option<&Bound<'_, PyAny>>,
) -> PyResult<NonZeroUsize> {
    match crate::chain::optional_integer(py, function, "threads", threads, 1)? {
        None => Ok(std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)),
        Some(threads) => Ok(usize::try_from(threads)
            .ok()
            .and_then(NonZeroUsize::new)
            .unwrap_or(NonZeroUsize::MAX)),
    }
}

/// `KY-E215` unless `progress` is `None` or callable.
pub fn check_progress(
    py: Python<'_>,
    function: &str,
    progress: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    match progress {
        Some(progress) if !progress.is_callable() => {
            let error = invalid(
                function,
                "progress",
                progress.repr()?.to_string(),
                "it is not callable",
                "a callable taking (done, total), or None",
            );
            Err(to_py_err(py, &error))
        }
        _ => Ok(()),
    }
}

/// The memory limit: `None`, a size such as `"4GB"` (task 5.5's units) or a byte count
/// of at least 1. `KY-E215` for anything else.
pub fn memory_limit(
    py: Python<'_>,
    function: &str,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<MemoryLimit>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if let Ok(text) = value.cast::<PyString>() {
        return MemoryLimit::parse(&text.to_cow()?, function)
            .map(Some)
            .map_err(|error| to_py_err(py, &error));
    }
    let bytes = if value.is_instance_of::<pyo3::types::PyBool>() {
        None
    } else {
        value.extract::<u64>().ok().and_then(NonZeroU64::new)
    };
    match bytes {
        Some(bytes) => Ok(Some(MemoryLimit::new(bytes))),
        None => {
            let error = invalid(
                function,
                "memory_limit",
                value.repr()?.to_string(),
                "it is not a size",
                "a size such as '4GB' or '512MiB', a number of bytes of 1 or more, or None",
            );
            Err(to_py_err(py, &error))
        }
    }
}

/// Check a terminal's shared arguments and build the physical plan of `plan` (every
/// optimizer rule, the CPU), fitted to the memory limit. Nothing is read.
pub fn run_arguments(
    py: Python<'_>,
    function: &str,
    plan: &Arc<Plan>,
    limit: Option<&Bound<'_, PyAny>>,
    threads_argument: Option<&Bound<'_, PyAny>>,
    progress: Option<&Bound<'_, PyAny>>,
) -> PyResult<RunArguments> {
    let limit = memory_limit(py, function, limit)?;
    let threads = threads(py, function, threads_argument)?;
    check_progress(py, function, progress)?;
    let options = PhysicalOptions::default();
    let physical = match limit {
        None => plan.physical(options),
        Some(limit) => plan
            .physical_within_memory_limit(options, limit)
            .map_err(|error| to_py_err(py, &error))?,
    };
    Ok(RunArguments { physical, threads })
}
