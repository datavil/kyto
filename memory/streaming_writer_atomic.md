# Streaming writer: provisional array shapes, OS file locks for stale temporaries (task 2.5)

**Works:** write chunks before the final shape is known through arrays built with the
largest shape the format allows (`u32::MAX` cells, `i64::MAX` entries): a regular-grid
chunk's bytes do not depend on the array shape. Build the real-shape arrays at `finish`
only to write `zarr.json`. `tests/store_writer.rs::bytes_do_not_depend_on_batching_or_threads`
checks the layer files equal `write_layer`'s byte for byte.

**Broke (by design review):** deleting another writer's `.kyto-tmp-` directory is not
safe by name alone: zarrs' `FilesystemStore` recreates missing parent directories, so a
live writer whose temporary was deleted mid-run would carry on and could rename a store
with missing shards into place. Stale temporaries are told apart by an OS file lock
(`File::lock` / `try_lock`, released when the process dies, even by SIGKILL) on a lock
file next to, not inside, the temporary (format.md §10.1 step 5). After locking, check
the lock file still exists: another writer's cleanup may have deleted it first.

**Also:**
- `File::lock` is Rust 1.89; the workspace `rust-version` is 1.95 (bumped Nov 7, 2026,
  see polars_parquet_tables.md), so clippy's `incompatible_msrv` no longer fires.
- zarrs omits `attributes` from a group `zarr.json` when it is empty.
- The kill test re-runs its own test binary (`current_exe()`, an `#[ignore]` child test
  gated by an env var) and kills it with `Child::kill` (SIGKILL / TerminateProcess).
