# HDF5 locks leak into child processes (errno 35)

**Symptom (Nov 2026, after the per-crate test binaries of `4f13b25`):** random gate
failures `H5Fopen(): unable to lock file, errno = 35, 'Resource temporarily
unavailable'` when a test reopens an `.h5ad` that `sink_h5ad` just wrote
(kyto-store `export_h5ad`, kyto-exec `filter_qc_parity`, `embeddings_graphs`). Each
test passed alone.

**Cause:** HDF5's sec2 driver opens files without `O_CLOEXEC` and `flock`s them
(`LOCK_EX` for a writer, `LOCK_SH` for a reader). A `flock` belongs to the open file
description, so any child process forked while the file is open (by another thread,
here the parity tests' `python` and `store_writer_kill`'s re-run of the binary)
inherits the descriptor and keeps the writer's exclusive lock after the writer
closes it. The rename keeps the inode, so the next open of the target fails until
the child exits. It is in-process, not cross-process: separate gates never share
files. Merging the test binaries put HDF5 writers and process-spawning tests in one
process; parallel gates only made the windows longer. `hdf5-metno` serialises its
calls with a global lock, so threading itself was never the problem.

**Fix:** `H5Out::create` (`export_h5.rs`) writes the temporary file with
`file_locking(false)` on its file access list: it has a random name only the writer
opens, so the lock guarded nothing, and without it there is nothing to inherit.
Library users hit the same error (a host that spawns a subprocess while
`sink_h5ad` runs), so the fix is in kyto, not the harness, and not a global
`HDF5_USE_FILE_LOCKING=FALSE`. Regression test:
`export_h5ad::a_child_started_during_the_write_does_not_lock_the_output` (spawns
`sleep` children while writing; fails at once without the fix).

**Still open:** readers (`import_h5`, `concat`) keep HDF5's shared lock, which also
leaks into children; it blocks only a later *writer* of the same input file, and it
protects against reading a file h5py is writing, so it stays.
