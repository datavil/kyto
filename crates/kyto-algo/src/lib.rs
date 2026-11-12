//! kyto-algo: single-cell algorithms on top of the executor.
//!
//! Does: QC metrics, normalization, HVG, scaling, PCA, nearest neighbours,
//! connectivities, Leiden/Louvain, UMAP layout and differential expression.
//! Does not: storage, planning or Python bindings.
//!
//! So far: per-cell QC metrics computed on each block (task 6.1). They live in
//! kyto-exec since task 5.8 (the executor computes them inside its stream, and this
//! crate depends on the executor), re-exported here unchanged. Differential
//! expression (task 6.10, [`rank_genes_groups`]): Welch's t-test and Wilcoxon's
//! rank-sum test per group, folded from a plan's output stream into one Polars frame.
#![forbid(unsafe_code)]

mod distributions;
mod group_labels;
mod rank_genes;
mod rank_genes_moments;
mod rank_genes_wilcoxon;

pub use kyto_exec::{CellQc, GeneSet, QcValues};
pub use rank_genes::{
    DEFAULT_MAX_RANK_VALUES, DifferentialTest, PValueCorrection, RANK_GENES_COLUMNS,
    RankGenesGroups, rank_genes_groups,
};
