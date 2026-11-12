//! Cancelling a run and following its progress (task 5.7).
//!
//! Does: a [`RunControl`] is shared by a running pipeline and whoever started it (for
//! example the Python thread waiting for the run). The caller may [`RunControl::cancel`]
//! at any time; the run checks the flag before it reads each block and before it
//! computes each block (a sink once more right before it renames its target into
//! place, `sink.rs`), then stops with `KY-E503`: the stream ends (its producer thread
//! joined when it is dropped), its buffers are freed, and its rayon pool's threads exit
//! once the pool is dropped (rayon does not join them). The run counts the cell rows it
//! has computed, summed over its passes, so the caller can read
//! [`RunControl::progress`] from any thread.
//! Does not: interrupt a tile read or a kernel already started (a block is the unit), call
//! back into the caller (the caller polls; kyto-py turns that into the Python progress
//! callback), or watch for signals (the caller does, on its own thread).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use kyto_error::KytoError;

/// How far a run has come: cell rows computed and cell rows in the whole run, each
/// summed over the run's passes (every statistics pass streams every cell once, then
/// the output's stream does).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Progress {
    /// Cell rows computed so far (a block counts once all its operators ran).
    pub done: u64,
    /// Cell rows the run computes in all: `n_cells` times its passes; 0 until the run
    /// has started.
    pub total: u64,
}

/// The cancel flag and progress counters of one run, shared (cheap to clone: every
/// clone is the same run).
#[derive(Debug, Clone, Default)]
pub struct RunControl {
    shared: Arc<Shared>,
}

#[derive(Debug, Default)]
struct Shared {
    cancelled: AtomicBool,
    done: AtomicU64,
    total: AtomicU64,
}

impl RunControl {
    /// A control for a new run: not cancelled, no progress.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask the run to stop. It stops at its next block with `KY-E503` (a sink past its
    /// last block, right before it renames its target into place); calling this again,
    /// or after the run ended, does nothing.
    pub fn cancel(&self) {
        self.shared.cancelled.store(true, Ordering::Relaxed);
    }

    /// Whether [`RunControl::cancel`] was called.
    pub fn is_cancelled(&self) -> bool {
        self.shared.cancelled.load(Ordering::Relaxed)
    }

    /// The run's progress now.
    pub fn progress(&self) -> Progress {
        Progress {
            done: self.shared.done.load(Ordering::Relaxed),
            total: self.shared.total.load(Ordering::Relaxed),
        }
    }

    /// `KY-E503` if the run was cancelled.
    pub(crate) fn check(&self) -> Result<(), KytoError> {
        if !self.is_cancelled() {
            return Ok(());
        }
        let Progress { done, total } = self.progress();
        Err(KytoError::Cancelled { done, total })
    }

    /// Set the cell rows of the whole run (when it starts).
    pub(crate) fn set_total(&self, total: u64) {
        self.shared.total.store(total, Ordering::Relaxed);
    }

    /// Count `cells` more cell rows computed.
    pub(crate) fn add_done(&self, cells: u64) {
        self.shared.done.fetch_add(cells, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::{Progress, RunControl};
    use kyto_error::KytoError;

    #[test]
    fn clones_share_one_run() {
        let control = RunControl::new();
        let other = control.clone();
        assert!(control.check().is_ok());
        other.set_total(10);
        other.add_done(4);
        assert_eq!(control.progress(), Progress { done: 4, total: 10 });
        other.cancel();
        assert!(control.is_cancelled());
        let error = control.check().unwrap_err();
        assert!(matches!(error, KytoError::Cancelled { done: 4, total: 10 }));
        assert_eq!(error.code(), "KY-E503");
    }
}
