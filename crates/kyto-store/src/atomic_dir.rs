//! Write a store in a temporary sibling directory and rename it into place on success
//! (format.md §10.1).
//!
//! Does: check the target path, delete stale temporary directories of the same target
//! that killed writers left behind, create `<parent>/.<name>.kyto-tmp-<random>` guarded
//! by a held lock file, fsync a finished tree, rename it to the target (replacing an
//! existing kyto store when asked), and delete the temporary directory on failure or
//! drop.
//! Does not: know what a store holds (see `store_writer`). Object stores (2.8) have no
//! rename; this is for local file systems.
//!
//! The lock file `<parent>/.<name>.kyto-tmp-<random>.lock` sits next to the temporary
//! directory, not inside it, and is held (an OS file lock) until the rename is done. The
//! OS drops the lock when its process dies, even by SIGKILL, so a later writer can tell
//! a stale temporary directory (lock free) from one that is still being written (lock
//! held) and deletes only the stale ones.

use std::fs::{File, OpenOptions};
use std::hash::{BuildHasher, Hasher};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use kyto_error::KytoError;
use zarrs::filesystem::FilesystemStore;

use crate::store_version::check_format_version;
use crate::zarr_arrays::store_io;

/// Marker between the target name and the random part of a temporary directory.
const TMP_MARKER: &str = ".kyto-tmp-";
/// Marker of an old store renamed aside during an overwrite.
const OLD_MARKER: &str = ".kyto-old-";
/// Extension of a temporary directory's lock file.
const LOCK_EXTENSION: &str = ".lock";

/// A temporary store directory that becomes the target only on [`AtomicDir::commit`].
///
/// Dropped without a commit, it deletes the temporary directory (best effort).
#[derive(Debug)]
pub(crate) struct AtomicDir {
    target: PathBuf,
    parent: PathBuf,
    /// Target file name, for temporary and old names.
    name: String,
    temporary: PathBuf,
    lock_path: PathBuf,
    /// The held lock; `None` once released.
    lock: Option<File>,
    overwrite: bool,
    committed: bool,
}

impl AtomicDir {
    /// Check `target` and create its temporary directory (format.md §10.1 step 1).
    ///
    /// Stale temporary directories of the same target are deleted first.
    ///
    /// # Errors
    /// - `KY-E118` if `target` exists and `overwrite` is false, or it exists and is not a
    ///   kyto store (overwrite replaces only stores).
    /// - `KY-E110` if `target` has no file name or parent, or the file system fails.
    pub(crate) fn create(target: &Path, overwrite: bool) -> Result<Self, KytoError> {
        let target_text = target.display().to_string();
        let name = target
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                store_io(
                    "create store",
                    &target_text,
                    "the path has no UTF-8 file name",
                )
            })?
            .to_owned();
        let parent = match target.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
            _ => PathBuf::from("."),
        };
        check_target(target, overwrite)?;
        remove_stale(&parent, &name);
        let (temporary, lock_path, lock) = loop {
            let random = random_suffix();
            let temporary = parent.join(format!(".{name}{TMP_MARKER}{random}"));
            let lock_path = parent.join(format!(".{name}{TMP_MARKER}{random}{LOCK_EXTENSION}"));
            // `create_new` makes the random part ours alone; retry on a collision.
            let lock = match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(lock) => lock,
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(store_io(
                        "create lock file",
                        &lock_path.display().to_string(),
                        error,
                    ));
                }
            };
            if let Err(error) = lock.lock() {
                drop(lock);
                let _ = std::fs::remove_file(&lock_path);
                return Err(store_io("lock", &lock_path.display().to_string(), error));
            }
            // Another writer's cleanup may have taken the fresh, not yet locked file for
            // an orphan and deleted it; then this lock guards nothing, so start over.
            if !lock_path.exists() {
                continue;
            }
            match std::fs::create_dir(&temporary) {
                Ok(()) => break (temporary, lock_path, lock),
                Err(error) => {
                    drop(lock);
                    let _ = std::fs::remove_file(&lock_path);
                    if error.kind() == ErrorKind::AlreadyExists {
                        continue;
                    }
                    return Err(store_io(
                        "create temporary directory",
                        &temporary.display().to_string(),
                        error,
                    ));
                }
            }
        };
        Ok(Self {
            target: target.to_path_buf(),
            parent,
            name,
            temporary,
            lock_path,
            lock: Some(lock),
            overwrite,
            committed: false,
        })
    }

    /// The temporary directory the store is written into.
    pub(crate) fn path(&self) -> &Path {
        &self.temporary
    }

    /// Fsync the temporary tree, rename it to the target and fsync the parent
    /// (format.md §10.1 steps 2 and 3). With `overwrite`, an existing store is renamed
    /// aside first and deleted after the new one is in place.
    ///
    /// # Errors
    /// - `KY-E118` if the target appeared since [`AtomicDir::create`] and may not be
    ///   replaced.
    /// - `KY-E110` if the file system fails; the target is then as it was.
    pub(crate) fn commit(mut self) -> Result<(), KytoError> {
        sync_tree(&self.temporary)?;
        sync_dir(&self.temporary)?;
        let target_text = self.target.display().to_string();
        let rename_error =
            |error: std::io::Error| store_io("rename the new store to", &target_text, error);
        let mut old = None;
        if self.target.symlink_metadata().is_ok() {
            check_target(&self.target, self.overwrite)?;
            let aside = self
                .parent
                .join(format!(".{}{OLD_MARKER}{}", self.name, random_suffix()));
            std::fs::rename(&self.target, &aside).map_err(|error| {
                store_io("rename the old store aside from", &target_text, error)
            })?;
            old = Some(aside);
        }
        if let Err(error) = std::fs::rename(&self.temporary, &self.target) {
            if let Some(aside) = &old {
                // Put the old store back, so a failed overwrite changes nothing.
                let _ = std::fs::rename(aside, &self.target);
            }
            return Err(rename_error(error));
        }
        self.committed = true;
        let synced = sync_dir(&self.parent);
        if let Some(aside) = old {
            // The new store is in place; a failure here only leaves the old one behind.
            let _ = std::fs::remove_dir_all(aside);
        }
        self.release_lock();
        synced
    }

    /// Unlock and delete the lock file.
    fn release_lock(&mut self) {
        if let Some(lock) = self.lock.take() {
            // Close before deleting: Windows does not delete an open file at once.
            drop(lock);
            let _ = std::fs::remove_file(&self.lock_path);
        }
    }
}

impl Drop for AtomicDir {
    fn drop(&mut self) {
        if !self.committed {
            // Best effort (format.md §10.1 step 4); a leftover is deleted by the next
            // write to the same target.
            let _ = std::fs::remove_dir_all(&self.temporary);
        }
        self.release_lock();
    }
}

/// `KY-E118` unless `target` is absent, or it is a kyto store and `overwrite` is set.
fn check_target(target: &Path, overwrite: bool) -> Result<(), KytoError> {
    if target.symlink_metadata().is_err() {
        return Ok(());
    }
    let exists = |problem: &str| KytoError::StoreExists {
        path: target.display().to_string(),
        problem: problem.to_owned(),
    };
    let is_store = is_kyto_store(target);
    match (overwrite, is_store) {
        (true, true) => Ok(()),
        (true, false) => Err(exists(
            "it exists and is not a kyto store, so overwrite does not replace it",
        )),
        (false, true) => Err(exists("a kyto store already exists there")),
        (false, false) => Err(exists("the path already exists")),
    }
}

/// True if `path` is a directory with a kyto root `zarr.json` of any `format_version`.
fn is_kyto_store(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    let Ok(storage) = FilesystemStore::new(path).map(Arc::new) else {
        return false;
    };
    match check_format_version(&*storage, &path.display().to_string()) {
        Ok(_) => true,
        Err(error) => error.code() == "KY-E115",
    }
}

/// Delete the temporary directories of target `name` in `parent` whose lock is free:
/// their writer died (format.md §10.1 step 4). Lock files left without a directory are
/// deleted too. Best effort: errors are ignored.
fn remove_stale(parent: &Path, name: &str) {
    let prefix = format!(".{name}{TMP_MARKER}");
    let Ok(entries) = std::fs::read_dir(parent) else {
        return;
    };
    let mut temporaries = Vec::new();
    let mut locks = Vec::new();
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        let Some(random) = file_name.strip_prefix(&prefix) else {
            continue;
        };
        match random.strip_suffix(LOCK_EXTENSION) {
            Some(random) if is_random(random) => locks.push(file_name.to_owned()),
            None if is_random(random) => temporaries.push(file_name.to_owned()),
            _ => {}
        }
    }
    for temporary in &temporaries {
        let lock_path = parent.join(format!("{temporary}{LOCK_EXTENSION}"));
        let Ok(lock) = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
        else {
            continue;
        };
        if lock.try_lock().is_err() {
            continue; // Its writer is alive.
        }
        let _ = std::fs::remove_dir_all(parent.join(temporary));
        drop(lock);
        let _ = std::fs::remove_file(&lock_path);
    }
    for lock_name in &locks {
        let directory = lock_name.trim_end_matches(LOCK_EXTENSION);
        if temporaries.iter().any(|temporary| temporary == directory) {
            continue; // Handled above.
        }
        let lock_path = parent.join(lock_name);
        let Ok(lock) = OpenOptions::new().write(true).open(&lock_path) else {
            continue;
        };
        if lock.try_lock().is_ok() {
            drop(lock);
            let _ = std::fs::remove_file(&lock_path);
        }
    }
}

/// True for the random part of a temporary name: 16 lowercase hex digits.
fn is_random(text: &str) -> bool {
    text.len() == 16
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// 16 hex digits from the standard library's randomly keyed hasher. Not part of the
/// store, so it does not affect determinism (format.md §10.3).
fn random_suffix() -> String {
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u32(std::process::id());
    if let Ok(elapsed) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        hasher.write_u128(elapsed.as_nanos());
    }
    format!("{:016x}", hasher.finish())
}

/// Fsync every file and directory below `directory` (format.md §10.1 step 2).
fn sync_tree(directory: &Path) -> Result<(), KytoError> {
    let io_error =
        |path: &Path, error: std::io::Error| store_io("sync", &path.display().to_string(), error);
    let entries = std::fs::read_dir(directory).map_err(|error| io_error(directory, error))?;
    for entry in entries {
        let entry = entry.map_err(|error| io_error(directory, error))?;
        let path = entry.path();
        let file_type = entry.file_type().map_err(|error| io_error(&path, error))?;
        if file_type.is_dir() {
            sync_tree(&path)?;
            sync_dir(&path)?;
        } else {
            // Write access: Windows flushes only handles opened for writing.
            OpenOptions::new()
                .write(true)
                .open(&path)
                .and_then(|file| file.sync_all())
                .map_err(|error| io_error(&path, error))?;
        }
    }
    Ok(())
}

/// Fsync a directory, so the names in it are durable. Windows cannot open a directory
/// as a file and makes renames durable itself, so this is a no-op there.
fn sync_dir(directory: &Path) -> Result<(), KytoError> {
    if cfg!(windows) {
        return Ok(());
    }
    File::open(directory)
        .and_then(|file| file.sync_all())
        .map_err(|error| store_io("sync", &directory.display().to_string(), error))
}

#[cfg(test)]
mod tests {
    use super::{AtomicDir, is_random, random_suffix};

    #[test]
    fn random_suffixes_are_recognised_and_differ() {
        let first = random_suffix();
        assert!(is_random(&first));
        assert_ne!(first, random_suffix());
        assert!(!is_random("0123456789abcdeg"));
        assert!(!is_random("0123"));
    }

    #[test]
    fn drop_without_commit_leaves_nothing() {
        let parent = tempfile::tempdir().unwrap();
        let target = parent.path().join("a.kyto");
        let dir = AtomicDir::create(&target, false).unwrap();
        std::fs::write(dir.path().join("file"), b"x").unwrap();
        drop(dir);
        assert_eq!(std::fs::read_dir(parent.path()).unwrap().count(), 0);
    }

    #[test]
    fn a_live_temporary_is_not_removed_by_a_second_writer() {
        let parent = tempfile::tempdir().unwrap();
        let target = parent.path().join("a.kyto");
        let first = AtomicDir::create(&target, false).unwrap();
        let second = AtomicDir::create(&target, false).unwrap();
        assert!(first.path().is_dir());
        assert!(second.path().is_dir());
        assert_ne!(first.path(), second.path());
    }
}
