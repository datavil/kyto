//! kyto-algo integration tests: one test binary, one module per former
//! `tests/<name>.rs` file, so HDF5, Polars and kyto are linked once (AGENTS.md,
//! Agent-friendliness). Add a new test file here as a module, never as `tests/*.rs`.

mod qc_parity;
mod rank_genes_parity;
