//! Write one output file in a temporary sibling and rename it into place on success
//! (the file counterpart of `atomic_dir`, for exports such as `sink_h5ad`, task 2.7).
//!
//! Does: check the target path (absent, or an existing file with `overwrite`), give a
//! temporary path `<parent>/.kyto-part-<random>` (no target name in it, so a
//! long target name still fits the file system's name limit), fsync and rename it to the
//! target on commit, and delete it on failure or drop.
//! Does not: clean up temporaries a killed process left behind (no lock file, unlike
//! stores); they are hidden files the user can delete. Never replaces a directory.

use std::path::{Path, PathBuf};

use kyto_error::KytoError;

use crate::atomic_dir::{random_suffix, sync_dir};
use crate::commit_check::CommitCheck;
use crate::zarr_arrays::store_io;

/// Prefix of a temporary file, before its random part.
const PART_MARKER: &str = ".kyto-part-";

/// A temporary file that becomes the target only on [`AtomicFile::commit`].
#[derive(Debug)]
pub(crate) struct AtomicFile {
    target: PathBuf,
    parent: PathBuf,
    temporary: PathBuf,
    overwrite: bool,
    committed: bool,
}

impl AtomicFile {
    /// Check `target` and choose a free temporary path next to it. Creates nothing.
    ///
    /// # Errors
    /// - `KY-E123` if `target` exists and `overwrite` is false, or it is not a file.
    /// - `KY-E110` if `target` has no file name, ends in a path separator or its parent
    ///   directory is missing.
    pub(crate) fn create(target: &Path, overwrite: bool) -> Result<Self, KytoError> {
        let target_text = target.display().to_string();
        // `Path::file_name` ignores a trailing separator; the rename would not.
        if target_text.ends_with(std::path::is_separator) {
            return Err(store_io(
                "create",
                &target_text,
                "the path ends in a separator, so it names a directory, not a file",
            ));
        }
        if target.file_name().is_none() {
            return Err(store_io(
                "create",
                &target_text,
                "the path has no file name",
            ));
        }
        let parent = match target.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
            _ => PathBuf::from("."),
        };
        if !parent.is_dir() {
            return Err(store_io(
                "create",
                &target_text,
                "its parent directory does not exist",
            ));
        }
        check_target(target, overwrite)?;
        let temporary = loop {
            let candidate = parent.join(format!("{PART_MARKER}{}", random_suffix()));
            if candidate.symlink_metadata().is_err() {
                break candidate;
            }
        };
        Ok(Self {
            target: target.to_path_buf(),
            parent,
            temporary,
            overwrite,
            committed: false,
        })
    }

    /// The temporary path to write the output to.
    pub(crate) fn path(&self) -> &Path {
        &self.temporary
    }

    /// Fsync the temporary file, rename it to the target and fsync the parent.
    /// `before_rename` runs once the file is synced, right before the rename.
    ///
    /// # Errors
    /// - The error of `before_rename`; the target is then as it was.
    /// - `KY-E123` if the target appeared since [`AtomicFile::create`] and may not be
    ///   replaced.
    /// - `KY-E110` if the file system fails; the target is then as it was.
    pub(crate) fn commit(mut self, before_rename: CommitCheck<'_>) -> Result<(), KytoError> {
        let temporary_text = self.temporary.display().to_string();
        std::fs::File::open(&self.temporary)
            .and_then(|file| file.sync_all())
            .map_err(|error| store_io("sync", &temporary_text, error))?;
        before_rename()?;
        check_target(&self.target, self.overwrite)?;
        std::fs::rename(&self.temporary, &self.target).map_err(|error| {
            store_io(
                "rename the new file to",
                &self.target.display().to_string(),
                error,
            )
        })?;
        self.committed = true;
        sync_dir(&self.parent)
    }
}

impl Drop for AtomicFile {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.temporary);
        }
    }
}

/// `KY-E123` unless `target` is absent, or it is a file (or a link to one) and
/// `overwrite` is set. A link is replaced by the new file; the file it points to is
/// left as it is.
fn check_target(target: &Path, overwrite: bool) -> Result<(), KytoError> {
    let Ok(metadata) = target.symlink_metadata() else {
        return Ok(());
    };
    let exists = |problem: &str| KytoError::OutputExists {
        path: target.display().to_string(),
        problem: problem.to_owned(),
    };
    if target.is_dir() {
        return Err(exists(
            "a directory exists there, and overwrite never replaces one",
        ));
    }
    if !overwrite {
        return Err(exists(if metadata.is_file() {
            "a file already exists there"
        } else {
            "the path already exists"
        }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::AtomicFile;
    use crate::commit_check::proceed;

    #[test]
    fn commit_renames_and_drop_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.h5ad");
        let file = AtomicFile::create(&target, false).unwrap();
        std::fs::write(file.path(), b"new").unwrap();
        file.commit(&proceed).unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"new");

        let error = AtomicFile::create(&target, false).unwrap_err();
        assert_eq!(error.code(), "KY-E123");
        assert!(
            error.to_string().contains("a file already exists"),
            "{error}"
        );

        let file = AtomicFile::create(&target, true).unwrap();
        std::fs::write(file.path(), b"newer").unwrap();
        let temporary = file.path().to_path_buf();
        drop(file);
        assert!(!temporary.exists());
        assert_eq!(std::fs::read(&target).unwrap(), b"new");

        let file = AtomicFile::create(&target, true).unwrap();
        std::fs::write(file.path(), b"newer").unwrap();
        file.commit(&proceed).unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"newer");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn never_replaces_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.h5ad");
        std::fs::create_dir(&target).unwrap();
        let error = AtomicFile::create(&target, true).unwrap_err();
        assert_eq!(error.code(), "KY-E123");
        assert!(error.to_string().contains("directory"), "{error}");
        let missing = dir.path().join("missing").join("out.h5ad");
        assert_eq!(
            AtomicFile::create(&missing, false).unwrap_err().code(),
            "KY-E110"
        );
    }
}
