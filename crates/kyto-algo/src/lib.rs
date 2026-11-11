//! kyto-algo: single-cell algorithms on top of the executor.
//!
//! Does: QC metrics, normalization, HVG, scaling, PCA, nearest neighbours,
//! connectivities, Leiden/Louvain, UMAP layout and differential expression.
//! Does not: storage, planning or Python bindings.
//!
//! So far: per-cell QC metrics computed on each block (task 6.1). They live in
//! kyto-exec since task 5.8 (the executor computes them inside its stream, and this
//! crate depends on the executor), re-exported here unchanged.
#![forbid(unsafe_code)]

pub use kyto_exec::{CellQc, GeneSet, QcValues};
