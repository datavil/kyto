//! What the statistics passes of cross-cell nodes compute (task 5.4): the results a
//! run keeps, by node.
//!
//! Does: hold, for each cross-cell node of the optimized plan that ran its statistics
//! passes, what they gave: the median `normalize_total`'s target (the median of the
//! positive cell totals) and `highly_variable_genes`' per-gene moments (sums and sums
//! of squares over every cell, `f64`), with the clipped moments of `seurat_v3`'s second
//! pass. Means and variances follow scanpy's `_get_mean_var` (variance with `ddof=1`).
//! Does not: rank or flag genes, fit `seurat_v3`'s loess or compute dispersions (task
//! 6.3), or compute anything itself (see `cross_cell.rs`).

use std::collections::BTreeMap;

use kyto_plan::HvgFlavor;

/// Per-gene sums and sums of squares of a layer over every cell.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneMoments {
    pub(crate) n_cells: u64,
    pub(crate) sums: Vec<f64>,
    pub(crate) sums_of_squares: Vec<f64>,
}

impl GeneMoments {
    /// Zero moments of `n_genes` genes over no cells.
    pub(crate) fn new(n_genes: usize) -> Self {
        Self {
            n_cells: 0,
            sums: vec![0.0; n_genes],
            sums_of_squares: vec![0.0; n_genes],
        }
    }

    /// The cells summed over (every cell of the stream, zeros included).
    pub fn n_cells(&self) -> u64 {
        self.n_cells
    }

    /// Each gene's sum over the cells.
    pub fn sums(&self) -> &[f64] {
        &self.sums
    }

    /// Each gene's sum of squares over the cells.
    pub fn sums_of_squares(&self) -> &[f64] {
        &self.sums_of_squares
    }

    /// Each gene's mean over the cells (NaN without cells).
    pub fn means(&self) -> Vec<f64> {
        let n = self.n_cells as f64;
        self.sums.iter().map(|sum| sum / n).collect()
    }

    /// Each gene's variance over the cells with `ddof=1`, as scanpy's `_get_mean_var`:
    /// `(sum_of_squares / n - mean^2) * n / (n - 1)` (NaN with fewer than two cells).
    pub fn variances(&self) -> Vec<f64> {
        let n = self.n_cells as f64;
        self.sums
            .iter()
            .zip(&self.sums_of_squares)
            .map(|(sum, squares)| {
                let mean = sum / n;
                (squares / n - mean * mean) * n / (n - 1.0)
            })
            .collect()
    }
}

/// What `highly_variable_genes`' statistics passes computed.
#[derive(Debug, Clone, PartialEq)]
pub struct HighlyVariableGeneStatistics {
    pub(crate) flavor: HvgFlavor,
    pub(crate) moments: GeneMoments,
    pub(crate) clipped: Option<ClippedMoments>,
}

/// `seurat_v3`'s second pass: each gene's moments of `min(value, clip)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ClippedMoments {
    pub(crate) clip: Vec<f64>,
    pub(crate) moments: GeneMoments,
}

impl ClippedMoments {
    /// Each gene's clip value, from the first pass.
    pub fn clip(&self) -> &[f64] {
        &self.clip
    }

    /// The moments of the clipped values.
    pub fn moments(&self) -> &GeneMoments {
        &self.moments
    }
}

impl HighlyVariableGeneStatistics {
    /// The flavour the statistics are for.
    pub fn flavor(&self) -> HvgFlavor {
        self.flavor
    }

    /// The first pass: moments of `expm1(value)` for `seurat` (scanpy undoes the log
    /// first), of the values as they are for `cell_ranger` and `seurat_v3`.
    pub fn moments(&self) -> &GeneMoments {
        &self.moments
    }

    /// `seurat_v3`'s second pass; `None` for the other flavours.
    pub fn clipped(&self) -> Option<&ClippedMoments> {
        self.clipped.as_ref()
    }
}

/// What one cross-cell node's statistics passes computed.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NodeStatistics {
    /// A median `normalize_total`: the target every cell was scaled to.
    MedianTotal(f64),
    /// `highly_variable_genes`.
    HighlyVariableGenes(HighlyVariableGeneStatistics),
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
