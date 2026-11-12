//! What the statistics passes of cross-cell nodes compute (task 5.4): the results a
//! run keeps, by node.
//!
//! Does: hold, for each cross-cell node of the optimized plan that ran its statistics
//! passes, what they gave: the median `normalize_total`'s target (the median of the
//! positive cell totals) and `highly_variable_genes`' per-gene moments (sums and sums
//! of squares, `f64`, over every cell and, with a `batch_key`, over each batch), with
//! `seurat_v3`'s clip fit and the clipped moments of its second pass, and the gene
//! columns task 6.3 computes from them ([`HvgValues`]). Means and variances follow
//! scanpy's `fast_array_utils` `mean_var` (variance with `ddof=1`).
//! Does not: compute anything itself (see `cross_cell.rs` and `hvg.rs`).

use std::collections::BTreeMap;

use kyto_plan::HvgFlavor;

use crate::neighbors::NeighborStatistics;
use crate::pca::PcaStatistics;

pub use crate::scale::ScaleStatistics;

/// Per-gene sums and sums of squares of a layer over a group of cells.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneMoments {
    pub(crate) n_cells: u64,
    pub(crate) sums: Vec<f64>,
    pub(crate) sums_of_squares: Vec<f64>,
    /// Cells with a positive value, per gene (scanpy's `filter_genes(min_cells=1)`).
    pub(crate) positives: Vec<u64>,
    /// Welford's moments of the stored values, for `seurat_v3` with batches (scanpy's
    /// `aggregate(..., func=["mean", "var"])`); `None` otherwise.
    pub(crate) welford: Option<Welford>,
}

/// Welford's running moments of each gene's stored (non-zero) values.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Welford {
    pub(crate) counts: Vec<u64>,
    pub(crate) means: Vec<f64>,
    pub(crate) squared_deviations: Vec<f64>,
}

impl Welford {
    fn new(n_genes: usize) -> Self {
        Self {
            counts: vec![0; n_genes],
            means: vec![0.0; n_genes],
            squared_deviations: vec![0.0; n_genes],
        }
    }

    /// Add `value` to gene `gene` (scanpy's `mean_var_csr` loop).
    pub(crate) fn add(&mut self, gene: usize, value: f64) {
        let count = self.counts[gene] + 1;
        self.counts[gene] = count;
        let mean = self.means[gene];
        let delta = value - mean;
        let mean = mean + delta / count as f64;
        self.means[gene] = mean;
        self.squared_deviations[gene] += delta * (value - mean);
    }

    /// `self` (the cells before) and `right` combined (Chan et al.'s parallel update).
    fn combine(&mut self, right: &Self) {
        for gene in 0..self.counts.len() {
            let (left_count, right_count) = (self.counts[gene], right.counts[gene]);
            if right_count == 0 {
                continue;
            }
            if left_count == 0 {
                self.counts[gene] = right_count;
                self.means[gene] = right.means[gene];
                self.squared_deviations[gene] = right.squared_deviations[gene];
                continue;
            }
            let count = left_count + right_count;
            let delta = right.means[gene] - self.means[gene];
            let (left_n, right_n, n) = (left_count as f64, right_count as f64, count as f64);
            self.means[gene] += delta * (right_n / n);
            self.squared_deviations[gene] +=
                right.squared_deviations[gene] + delta * delta * (left_n * right_n / n);
            self.counts[gene] = count;
        }
    }
}

impl GeneMoments {
    /// Zero moments of `n_genes` genes over no cells, with Welford's moments if
    /// `welford`.
    pub(crate) fn new(n_genes: usize, welford: bool) -> Self {
        Self {
            n_cells: 0,
            sums: vec![0.0; n_genes],
            sums_of_squares: vec![0.0; n_genes],
            positives: vec![0; n_genes],
            welford: welford.then(|| Welford::new(n_genes)),
        }
    }

    /// `self` (the cells before) and `right` as the moments of both.
    pub(crate) fn combine(&mut self, right: &Self) {
        self.n_cells += right.n_cells;
        for (sum, right) in self.sums.iter_mut().zip(&right.sums) {
            *sum += right;
        }
        for (squares, right) in self.sums_of_squares.iter_mut().zip(&right.sums_of_squares) {
            *squares += right;
        }
        for (positives, right) in self.positives.iter_mut().zip(&right.positives) {
            *positives += right;
        }
        if let (Some(welford), Some(right)) = (&mut self.welford, &right.welford) {
            welford.combine(right);
        }
    }

    /// The cells summed over (every cell of the group, zeros included).
    pub fn n_cells(&self) -> u64 {
        self.n_cells
    }

    /// Each gene's sum over the cells.
    pub fn sums(&self) -> &[f64] {
        &self.sums
    }

    /// Each gene's sum of squares over the cells (each value squared in `float32`, as
    /// scanpy's numba kernel squares `float32` data, for the first pass).
    pub fn sums_of_squares(&self) -> &[f64] {
        &self.sums_of_squares
    }

    /// Each gene's number of cells with a positive value.
    pub fn positives(&self) -> &[u64] {
        &self.positives
    }

    /// Each gene's mean over the cells (NaN without cells).
    pub fn means(&self) -> Vec<f64> {
        let n = self.n_cells as f64;
        self.sums.iter().map(|sum| sum / n).collect()
    }

    /// Each gene's variance over the cells with `ddof=1`, as `fast_array_utils`'
    /// `mean_var(correction=1)`: `(sum_of_squares / n - mean^2) * (n / (n - 1))`, not
    /// scaled for one cell, NaN without cells.
    pub fn variances(&self) -> Vec<f64> {
        let n = self.n_cells as f64;
        self.sums
            .iter()
            .zip(&self.sums_of_squares)
            .map(|(sum, squares)| {
                let mean = sum / n;
                let variance = squares / n - mean * mean;
                if self.n_cells == 1 {
                    variance
                } else {
                    variance * (n / (n - 1.0))
                }
            })
            .collect()
    }
}

/// What `highly_variable_genes`' statistics passes computed, and the gene columns
/// computed from them.
#[derive(Debug, Clone, PartialEq)]
pub struct HighlyVariableGeneStatistics {
    pub(crate) flavor: HvgFlavor,
    pub(crate) moments: GeneMoments,
    pub(crate) clipped: Option<ClippedMoments>,
    pub(crate) batches: Vec<BatchStatistics>,
    pub(crate) columns: Vec<(String, HvgValues)>,
}

/// One batch's statistics (`batch_key` given).
#[derive(Debug, Clone, PartialEq)]
pub struct BatchStatistics {
    pub(crate) label: String,
    pub(crate) moments: GeneMoments,
    pub(crate) clipped: Option<ClippedMoments>,
}

impl BatchStatistics {
    /// The batch: the `batch_key` column's value as text.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The first pass's moments over the batch's cells.
    pub fn moments(&self) -> &GeneMoments {
        &self.moments
    }

    /// `seurat_v3`'s second pass over the batch's cells; `None` for other flavours.
    pub fn clipped(&self) -> Option<&ClippedMoments> {
        self.clipped.as_ref()
    }
}

/// `seurat_v3`'s second pass: each gene's moments of `min(value, clip)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ClippedMoments {
    pub(crate) clip: Vec<f64>,
    pub(crate) regularized_std: Vec<f64>,
    /// The means the fit used (`mean_var`'s, or Welford's with batches).
    pub(crate) means: Vec<f64>,
    pub(crate) moments: GeneMoments,
}

impl ClippedMoments {
    /// Each gene's clip value, from the first pass: `mean + sqrt(n) * reg_std`.
    pub fn clip(&self) -> &[f64] {
        &self.clip
    }

    /// Each gene's regularized standard deviation, `sqrt(10^loess(log10 mean))`
    /// (scanpy's `reg_std`; 1 for a gene of variance 0).
    pub fn regularized_std(&self) -> &[f64] {
        &self.regularized_std
    }

    /// The moments of the clipped values.
    pub fn moments(&self) -> &GeneMoments {
        &self.moments
    }
}

/// The values of one gene column `highly_variable_genes` writes, one per gene.
#[derive(Debug, Clone, PartialEq)]
pub enum HvgValues {
    /// A `Boolean` column.
    Boolean(Vec<bool>),
    /// A `Float64` column.
    Float64(Vec<f64>),
    /// An `Int64` column.
    Int64(Vec<i64>),
}

impl HighlyVariableGeneStatistics {
    /// The flavour the statistics are for.
    pub fn flavor(&self) -> HvgFlavor {
        self.flavor
    }

    /// The first pass over every cell: moments of `expm1(value)` for `seurat` (scanpy
    /// undoes the log first), of the values as they are for `cell_ranger` and
    /// `seurat_v3`. With batches, the batches' moments added in label order.
    pub fn moments(&self) -> &GeneMoments {
        &self.moments
    }

    /// `seurat_v3`'s second pass without batches; `None` for the other flavours and
    /// with batches (see [`HighlyVariableGeneStatistics::batches`]).
    pub fn clipped(&self) -> Option<&ClippedMoments> {
        self.clipped.as_ref()
    }

    /// Each batch's statistics, in label order; empty without `batch_key`.
    pub fn batches(&self) -> &[BatchStatistics] {
        &self.batches
    }

    /// The gene columns computed, in the order the plan's schema lists them
    /// (`kyto_plan::hvg_columns`), one value per gene present.
    pub fn columns(&self) -> &[(String, HvgValues)] {
        &self.columns
    }

    /// The gene column `name`, if computed.
    pub fn column(&self, name: &str) -> Option<&HvgValues> {
        self.columns
            .iter()
            .find(|(column, _)| column == name)
            .map(|(_, values)| values)
    }
}

/// What one cross-cell node's statistics passes computed.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NodeStatistics {
    /// A median `normalize_total`: the target every cell was scaled to.
    MedianTotal(f64),
    /// `highly_variable_genes`.
    HighlyVariableGenes(Box<HighlyVariableGeneStatistics>),
    /// `scale`: each gene's mean and standard deviation (task 6.4).
    Scale(Box<ScaleStatistics>),
    /// `pca`: the embedding, loadings and variances (task 6.5).
    Pca(Box<PcaStatistics>),
    /// `neighbors`: the distance graph (task 6.6; no statistics passes, it searches
    /// the embedding after the passes of the node that wrote it).
    Neighbors(Box<NeighborStatistics>),
}

/// The statistics of every cross-cell node a run computed, by the node's index in the
/// optimized plan (`OptimizedPlan::nodes`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Statistics {
    pub(crate) nodes: BTreeMap<usize, NodeStatistics>,
}

impl Statistics {
    /// The statistics of optimized node `node`, if it ran statistics passes.
    pub fn get(&self, node: usize) -> Option<&NodeStatistics> {
        self.nodes.get(&node)
    }

    /// Every node's statistics, by node index.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &NodeStatistics)> {
        self.nodes
            .iter()
            .map(|(node, statistics)| (*node, statistics))
    }
}
