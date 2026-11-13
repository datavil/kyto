//! kyto-store integration tests: one test binary, one module per former
//! `tests/<name>.rs` file, so HDF5, Polars and kyto are linked once (AGENTS.md,
//! Agent-friendliness). Add a new test file here as a module, never as `tests/*.rs`.

mod backends;

mod atlas_schema;
mod concat;
mod export_h5ad;
mod gene_symbols;
mod import_parity;
mod layer_format;
mod members;
mod missing_values;
mod remote_storage;
mod round_trip;
mod store_writer;
mod store_writer_kill;
mod table_format;
mod table_round_trip;
mod tile_reader;
