//! kyto-exec integration tests: one test binary, one module per former
//! `tests/<name>.rs` file, so HDF5, Polars and kyto are linked once (AGENTS.md,
//! Agent-friendliness). Add a new test file here as a module, never as `tests/*.rs`.

// The 4.4 reference evaluator, shared with kyto-plan's tests.
#[path = "../../../kyto-plan/tests/it/support.rs"]
mod plan_support;

mod cluster_parity;
mod connectivities_parity;
mod deterministic_reductions;
mod embeddings_graphs;
mod filter_qc_parity;
mod filters_columns;
mod hvg_parity;
mod kernels;
mod loess_parity;
mod multi_pass;
mod neighbors;
mod neighbors_parity;
mod normalize_log1p_parity;
mod pca;
mod pca_parity;
mod pipeline;
mod run_control;
mod scale_parity;
mod scrublet_parity;
mod sinks;
mod thread_scaling;
mod umap_parity;
