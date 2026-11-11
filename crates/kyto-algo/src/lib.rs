//! kyto-algo: single-cell algorithms on top of the executor.
//!
//! Does: QC metrics, normalization, HVG, scaling, PCA, nearest neighbours,
//! connectivities, Leiden/Louvain, UMAP layout and differential expression.
//! Does not: storage, planning or Python bindings.
//!
//! So far: per-cell QC metrics computed on each block ([`qc`], task 6.1).
#![forbid(unsafe_code)]

pub mod qc;

pub use qc::{CellQc, GeneSet, QcValues};
