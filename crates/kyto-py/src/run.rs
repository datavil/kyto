//! Running a plan from Python (task 5.7): the GIL released while it computes, an
//! optional progress callback, and Ctrl-C cancelling the run cleanly.
//!
//! Does: run one output of a physical plan through the executor (kyto-exec) on a worker
//! thread and drain its stream, while the calling Python thread waits with the GIL
//! released, waking every [`TICK`] to take the GIL back briefly: it runs Python's signal
//! handlers (`check_signals`; Ctrl-C raises `KeyboardInterrupt` there) and calls the
//! progress callback, on the calling thread, when the progress changed. A raised
//! exception (from a signal handler or the callback) cancels the run through its
//! `RunControl`, waits for the worker (the run has stopped and its buffers are freed
//! before this returns; the rayon pool's threads exit just after) and is re-raised as it is. Other Python threads run while
//! the plan computes.
//! A plan ending in a sink writes it (task 5.9; never over an existing target: the
//! public `sink_zarr`/`sink_h5ad` and their `overwrite` are 7.3).
//! Does not: return the data (the terminals are 7.2/7.3: each drains its stream
//! through [`run_plan`]), or apply a memory limit (5.5's `fit_memory_limit`
//! is for 7.2's `collect(memory_limit=...)`).

use std::num::NonZeroUsize;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use kyto_error::KytoError;
use kyto_exec::{ExecuteOptions, Pipeline, Progress, RunControl};
use kyto_plan::PhysicalPlan;
use pyo3::prelude::*;

use crate::errors::to_py_err;

/// How long the calling thread waits, without the GIL, between two looks at signals and
/// progress: Ctrl-C cancels within about this long plus one block.
pub const TICK: Duration = Duration::from_millis(20);

/// Run output 0 of `physical` on `threads` threads and drain its stream; the number of
/// cell rows the output streamed.
///
/// `progress`, when given, is called on this thread as `progress(done, total)` (cell
/// rows computed and in the whole run, summed over its passes) whenever they changed
/// since the last call, so last with `done == total` after a run that finished (never
/// for a store without cells).
///
/// # Errors
/// - The executor's errors (`KY-E601` for a node it does not run yet, reading errors).
/// - `KY-E502` if the worker thread cannot be started.
/// - Whatever a signal handler (Ctrl-C: `KeyboardInterrupt`) or `progress` raises: the
///   run is cancelled and has ended before it is re-raised.
pub fn run_plan(
    py: Python<'_>,
    physical: &PhysicalPlan,
    threads: NonZeroUsize,
    progress: Option<&Bound<'_, PyAny>>,
) -> PyResult<u64> {
    let pipeline = Pipeline::new(physical, 0).map_err(|error| to_py_err(py, &error))?;
    let control = RunControl::new();
    let options = ExecuteOptions::new(threads).with_control(control.clone());
    let (sender, receiver) = mpsc::channel();
    let worker = std::thread::Builder::new()
        .name("kyto-run".to_owned())
        .spawn(move || {
            // The receiver may be gone only if this thread outlived the call: never.
            let _ = sender.send(drain(&pipeline, options));
        })
        .map_err(|error| {
            let error = KytoError::ThreadStart {
                threads: 1,
                problem: error.to_string(),
            };
            to_py_err(py, &error)
        })?;
    let mut reported = Progress::default();
    let mut receiver = receiver;
    loop {
        // A `Receiver` is `Send` but not `Sync`: move it into the detached wait and back.
        let (back, received) = py.detach(move || {
            let received = receiver.recv_timeout(TICK);
            (receiver, received)
        });
        receiver = back;
        // Signals first: Ctrl-C wins over a run that has just finished.
        let looked = py
            .check_signals()
            .and_then(|()| report(progress, &control, &mut reported));
        if let Err(raised) = looked {
            control.cancel();
            // The run stops at its next block; wait for it without the GIL. A panic
            // in it was a bug, already reported by the panic hook.
            let _ = py.detach(|| worker.join());
            return Err(raised);
        }
        match received {
            Err(RecvTimeoutError::Timeout) => {}
            Ok(result) => {
                join(py, worker);
                let cells = result.map_err(|error| to_py_err(py, &error))?;
                report(progress, &control, &mut reported)?;
                return Ok(cells);
            }
            Err(RecvTimeoutError::Disconnected) => {
                // The worker ended without a result: it panicked; re-raise the panic.
                join(py, worker);
                unreachable!("a worker that sends nothing has panicked");
            }
        }
    }
}

/// Run `pipeline` and drain its stream, recycling every block; the cell rows streamed.
/// A sink writes its target instead (task 5.9); the cells written.
fn drain(pipeline: &Pipeline, options: ExecuteOptions) -> Result<u64, KytoError> {
    if pipeline.is_sink() {
        return pipeline.sink(options, false);
    }
    let mut stream = pipeline.run(options)?;
    let mut cells = 0_u64;
    while let Some(block) = stream.next() {
        let block = block?;
        cells += block.cells.len() as u64;
        stream.recycle(block);
    }
    Ok(cells)
}

/// Wait for the worker without the GIL; re-raise its panic (a kyto bug) on this thread,
/// where PyO3 turns it into a `PanicException`.
fn join(py: Python<'_>, worker: std::thread::JoinHandle<()>) {
    if let Err(payload) = py.detach(|| worker.join()) {
        std::panic::resume_unwind(payload);
    }
}

/// Call `progress(done, total)` if the run's progress moved since `reported`.
fn report(
    progress: Option<&Bound<'_, PyAny>>,
    control: &RunControl,
    reported: &mut Progress,
) -> PyResult<()> {
    let Some(progress) = progress else {
        return Ok(());
    };
    let now = control.progress();
    if now.total == 0 || now == *reported {
        return Ok(());
    }
    *reported = now;
    progress.call1((now.done, now.total))?;
    Ok(())
}
