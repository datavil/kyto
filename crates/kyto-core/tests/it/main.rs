//! kyto-core integration tests: one test binary, one module per former
//! `tests/<name>.rs` file, so HDF5, Polars and kyto are linked once (AGENTS.md,
//! Agent-friendliness). Add a new test file here as a module, never as `tests/*.rs`.

/// Fixed seed and case count of the property tests (task 10.1), shared by every crate.
#[path = "../../../../tests/property_cases.rs"]
mod property_cases;

mod expressions;
mod feature_names;
mod gene_qc_metrics;
mod gene_symbols;
mod missing_values;
mod qc_metrics;
mod resolve;
mod schema_round_trip;
mod suggestions;
