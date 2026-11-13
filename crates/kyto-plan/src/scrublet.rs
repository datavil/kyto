//! The parameters of doublet detection: Scrublet, as scanpy's `pp.scrublet`.
//!
//! Does: hold scanpy 1.12.5 `pp.scrublet`'s arguments with its defaults (the batch
//! column, the doublets simulated per cell, the expected doublet rate and its
//! uncertainty, the principal components, the neighbours, a fixed threshold or the
//! automatic one, the seed) and the two cell columns written: the `Float64` score
//! (scanpy's `doublet_score`) and the `Boolean` call (`predicted_doublet`). The node
//! itself is a cross-cell step (`CrossCellOp::Scrublet`, built by `Plan::scrublet`).
//! Does not: detect doublets (kyto-exec's `scrublet.rs`, task 6.12), or take scanpy's
//! advanced arguments (`adata_sim`, `synthetic_doublet_umi_subsampling`,
//! `knn_dist_metric`, `normalize_variance`, `log_transform`, `mean_center`,
//! `use_approx_neighbors`, `get_doublet_neighbor_parents`): deviations.md.

use std::fmt;

use kyto_core::{ColumnDtype, ColumnSchema};
use serde::{Deserialize, Serialize};

/// The parameters of a Scrublet run (scanpy's names and defaults).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scrublet {
    /// A cell column: each batch is scored on its own cells (scanpy's `batch_key`).
    pub batch_key: Option<String>,
    /// Doublets simulated per observed cell.
    pub sim_doublet_ratio: f64,
    /// The expected fraction of doublets.
    pub expected_doublet_rate: f64,
    /// The uncertainty of `expected_doublet_rate`.
    pub stdev_doublet_rate: f64,
    /// Principal components the cells and doublets are embedded in.
    pub n_prin_comps: u64,
    /// Neighbours of the classifier; `None`: `round(0.5 * sqrt(n_cells))` per batch.
    pub n_neighbors: Option<u64>,
    /// The score above which a cell is called a doublet; `None`: the minimum between
    /// the two modes of the simulated doublets' scores (scikit-image's
    /// `threshold_minimum`).
    pub threshold: Option<f64>,
    /// The random seed of the simulated doublets (R7).
    pub seed: u64,
    /// The `Float64` cell column of the scores (scanpy's `doublet_score`).
    pub score_into: String,
    /// The `Boolean` cell column of the calls (scanpy's `predicted_doublet`).
    pub predicted_into: String,
}

impl Default for Scrublet {
    /// scanpy's defaults, seed 0, scanpy's column names.
    fn default() -> Self {
        Self {
            batch_key: None,
            sim_doublet_ratio: 2.0,
            expected_doublet_rate: 0.05,
            stdev_doublet_rate: 0.02,
            n_prin_comps: 30,
            n_neighbors: None,
            threshold: None,
            seed: 0,
            score_into: "doublet_score".to_owned(),
            predicted_into: "predicted_doublet".to_owned(),
        }
    }
}

impl Scrublet {
    /// The two cell columns written: the score (`Float64`), then the call
    /// (`Boolean`). A cell Scrublet does not score (fewer than 3 genes detected among
    /// the genes seen in at least 3 cells of its batch) is null in both.
    pub fn columns(&self) -> [ColumnSchema; 2] {
        [
            ColumnSchema::new(self.score_into.clone(), ColumnDtype::Float64),
            ColumnSchema::new(self.predicted_into.clone(), ColumnDtype::Boolean),
        ]
    }
}

impl fmt::Display for Scrublet {
    /// The arguments as `explain` prints them.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "-> [{:?}, {:?}]",
            self.score_into, self.predicted_into
        )?;
        if let Some(batch_key) = &self.batch_key {
            write!(formatter, " batch_key={batch_key:?}")?;
        }
        write!(
            formatter,
            " sim_doublet_ratio={} expected_doublet_rate={} stdev_doublet_rate={} \
             n_prin_comps={} n_neighbors=",
            self.sim_doublet_ratio,
            self.expected_doublet_rate,
            self.stdev_doublet_rate,
            self.n_prin_comps
        )?;
        match self.n_neighbors {
            Some(n_neighbors) => write!(formatter, "{n_neighbors}")?,
            None => write!(formatter, "auto")?,
        }
        write!(formatter, " threshold=")?;
        match self.threshold {
            Some(threshold) => write!(formatter, "{threshold}")?,
            None => write!(formatter, "auto")?,
        }
        write!(formatter, " seed={}", self.seed)
    }
}
