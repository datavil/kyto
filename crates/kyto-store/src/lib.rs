//! kyto-store: the on-disk format of a kyto atlas.
//!
//! Does: the Zarr v3 layout, tiled sparse matrix I/O, metadata tables and
//! importers (`.h5ad`, 10x `.h5`, `.mtx`).
//! Does not: plan, optimize or compute anything; it only reads and writes tiles.
#![forbid(unsafe_code)]
