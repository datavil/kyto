//! The last check before a writer makes its target appear.
//!
//! Does: name the callback a caller hands to `StoreWriter::finish_checked` and
//! `sink_h5ad_checked`. The writer calls it once, when everything is written and
//! synced, right before the rename that puts the target in place. If it fails, its
//! error is returned, the temporary is deleted and the target is as it was. The
//! executor passes its cancel check (task 5.7), so a sink cancelled at any moment
//! before the rename leaves no target.
//! Does not: stop a writer midway (it runs once, at the end), or undo a rename: after
//! the check passes, the target appears.

use kyto_error::KytoError;

/// A check run right before a writer renames its finished output to the target; an
/// error stops the writer, which leaves the target as it was.
pub type CommitCheck<'a> = &'a (dyn Fn() -> Result<(), KytoError> + 'a);

/// The check of a writer nobody stops: always `Ok`.
pub(crate) fn proceed() -> Result<(), KytoError> {
    Ok(())
}
