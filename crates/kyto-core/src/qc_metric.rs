//! Per-cell QC metrics as expression leaves (task 6.1): what `total_counts()`,
//! `genes_detected()`, `total_counts_in(genes)` and `percent_counts_in(genes)` read.
//!
//! Does: name the metrics a [`crate::Source::Qc`] leaf computes from one layer of the
//! cells' rows, and hold the gene set of the two gene-set metrics as a boolean
//! metadata predicate over the gene table (`gene_column('symbol').str.starts_with('MT-')`),
//! so the metric is computed when the plan is read, over the genes present then, and
//! never goes stale (3.8).
//! Does not: compute anything (`kyto-algo`'s `qc` module does, on each block), or check
//! the layer or the predicate's columns (the resolver does, at plan build, R6).

use kyto_error::KytoError;

use crate::expr::{CellExpr, ExprNode, GeneExpr};
use crate::expr_dtype::ExprDtype;
use crate::expr_source::Source;

/// A per-cell QC metric of one layer, as `scanpy.pp.calculate_qc_metrics` computes it
/// per cell.
#[derive(Debug, Clone, PartialEq)]
pub enum QcMetric {
    /// The sum of the cell's values over every gene: scanpy's `total_counts`.
    TotalCounts,
    /// The number of genes with a non-zero value in the cell: scanpy's
    /// `n_genes_by_counts`.
    GenesDetected,
    /// The sum of the cell's values over the genes where `genes` is true: scanpy's
    /// `total_counts_<qc_var>`.
    TotalCountsIn {
        /// The gene set: a boolean predicate on gene-table columns.
        genes: GeneExpr,
    },
    /// `100 * total_counts_in(genes) / total_counts()`: scanpy's `pct_counts_<qc_var>`
    /// (`NaN` for a cell without counts, as scanpy's `0 / 0`).
    PercentCountsIn {
        /// The gene set: a boolean predicate on gene-table columns.
        genes: GeneExpr,
    },
}

impl QcMetric {
    /// The Python constructor that builds this metric: `total_counts`, ...
    pub fn function(&self) -> &'static str {
        match self {
            Self::TotalCounts => "total_counts",
            Self::GenesDetected => "genes_detected",
            Self::TotalCountsIn { .. } => "total_counts_in",
            Self::PercentCountsIn { .. } => "percent_counts_in",
        }
    }

    /// The gene set of a gene-set metric.
    pub fn genes(&self) -> Option<&GeneExpr> {
        match self {
            Self::TotalCounts | Self::GenesDetected => None,
            Self::TotalCountsIn { genes } | Self::PercentCountsIn { genes } => Some(genes),
        }
    }

    /// The same metric over the gene set `genes` (a gene-set metric) or as it is.
    pub(crate) fn with_genes(&self, genes: GeneExpr) -> Self {
        match self {
            Self::TotalCounts | Self::GenesDetected => self.clone(),
            Self::TotalCountsIn { .. } => Self::TotalCountsIn { genes },
            Self::PercentCountsIn { .. } => Self::PercentCountsIn { genes },
        }
    }
}

impl CellExpr {
    /// The QC metric `metric` of each cell, computed from `layer` (`None`: the plan's
    /// default layer) when the plan is read. Numeric.
    ///
    /// # Errors
    /// `KY-E205` if a gene-set metric's gene set is not a boolean expression (a
    /// column of unknown dtype is accepted and checked when the plan is built).
    pub fn qc(layer: Option<String>, metric: QcMetric) -> Result<Self, KytoError> {
        let mut depth = 1;
        if let Some(genes) = metric.genes() {
            if !matches!(genes.dtype(), ExprDtype::Boolean | ExprDtype::Unknown) {
                return Err(gene_set_error(&metric, genes));
            }
            depth = Self::deeper(metric.function(), genes.depth())?;
        }
        Ok(Self::with_depth(
            ExprNode::Source(Source::Qc { layer, metric }),
            ExprDtype::Numeric,
            depth,
        ))
    }
}

/// `KY-E205` for a gene set that is not a boolean predicate on gene-table columns.
pub(crate) fn gene_set_error(metric: &QcMetric, genes: &GeneExpr) -> KytoError {
    let got = if genes.dtype() == ExprDtype::Boolean {
        "a boolean expression that reads matrix or embedding values".to_owned()
    } else {
        format!("a {} expression", genes.dtype().name())
    };
    KytoError::ExpressionOperand {
        operation: metric.function().to_owned(),
        accepted: "a boolean expression on gene-table columns, such as \
                   gene_column('symbol').str.starts_with('MT-')"
            .to_owned(),
        got,
        operand: genes.to_string(),
    }
}
