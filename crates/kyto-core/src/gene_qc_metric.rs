//! Per-gene QC metrics as expression leaves (task 6.13): what `cells_detected()`,
//! `gene_total_counts()`, `gene_mean_counts()` and `percent_dropout()` read.
//!
//! Does: name the metrics a [`crate::Source::GeneQc`] leaf computes from one layer over
//! the cells present where the gene step runs (the per-gene columns of
//! `scanpy.pp.calculate_qc_metrics`), so `filter(cells_detected() >= 3)` is scanpy's
//! `filter_genes(min_cells=3)` inside a plan, and the values never go stale (3.8).
//! Does not: compute anything (`kyto-exec`'s `gene_qc` module does, in the gene
//! step's one pass over the stream), or check the layer (the resolver does, at plan
//! build, R6).

use serde::{Deserialize, Serialize};

use crate::expr::{ExprNode, GeneExpr};
use crate::expr_dtype::ExprDtype;
use crate::expr_source::Source;

/// A per-gene QC metric of one layer, as `scanpy.pp.calculate_qc_metrics` computes it
/// per gene. In plan JSON a metric is written as its constructor's name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeneQcMetric {
    /// The number of cells with a non-zero value of the gene: scanpy's
    /// `n_cells_by_counts`.
    #[serde(rename = "cells_detected")]
    CellsDetected,
    /// The sum of the gene's values over the cells: scanpy's per-gene `total_counts`.
    #[serde(rename = "gene_total_counts")]
    TotalCounts,
    /// The gene's total divided by the number of cells: scanpy's `mean_counts` (`NaN`
    /// without cells, as `0 / 0`).
    #[serde(rename = "gene_mean_counts")]
    MeanCounts,
    /// `(1 - cells_detected / cells) * 100`: scanpy's `pct_dropout_by_counts` (`NaN`
    /// without cells).
    #[serde(rename = "percent_dropout")]
    PercentDropout,
}

impl GeneQcMetric {
    /// Every metric, in the order of scanpy's columns.
    pub const ALL: [Self; 4] = [
        Self::CellsDetected,
        Self::TotalCounts,
        Self::MeanCounts,
        Self::PercentDropout,
    ];

    /// The Python constructor that builds this metric: `cells_detected`, ...
    pub fn function(self) -> &'static str {
        match self {
            Self::CellsDetected => "cells_detected",
            Self::TotalCounts => "gene_total_counts",
            Self::MeanCounts => "gene_mean_counts",
            Self::PercentDropout => "percent_dropout",
        }
    }

    /// The metric the constructor `function` builds, if it names one.
    pub fn from_function(function: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|metric| metric.function() == function)
    }
}

impl GeneExpr {
    /// The QC metric `metric` of each gene, computed from `layer` (`None`: the plan's
    /// default layer) over the cells present when the plan is read. Numeric.
    pub fn qc(layer: Option<String>, metric: GeneQcMetric) -> Self {
        Self::from_node(
            ExprNode::Source(Source::GeneQc { layer, metric }),
            ExprDtype::Numeric,
        )
    }
}
