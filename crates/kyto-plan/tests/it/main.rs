//! kyto-plan integration tests: one test binary, one module per former
//! `tests/<name>.rs` file, so HDF5, Polars and kyto are linked once (AGENTS.md,
//! Agent-friendliness). Add a new test file here as a module, never as `tests/*.rs`.

mod support;

mod build_validation;
mod column_dtype_polars;
mod estimate;
mod explain;
mod handles;
mod lineage;
mod logical_nodes;
mod normalize_log1p_explain;
mod optimizer;
mod physical_plan;
mod plan_chain;
mod plan_json;
mod plan_validate;
mod resolve_in_plan;
