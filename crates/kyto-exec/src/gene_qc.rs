//! Per-gene QC metrics (task 6.13): computes the resolved [`GeneQcMetric`] leaves of a
//! gene step from one pass over the cells present at that step.
//!
//! Does: count, for each gene present, the cells with a non-zero value and sum its
//! values in `f64` over the blocks of one layer ([`GeneCounts`]), as a leaf of a
//! deterministic reduction (5.6: one leaf per store cell block, rows folded in cell
//! order, leaves combined in a fixed tree, so the same bits on any thread count and
//! block size, R11); and turn the counts into each metric's column as
//! `scanpy.pp.calculate_qc_metrics` defines it per gene: `n_cells_by_counts`,
//! `total_counts`, `mean_counts = total / cells` and
//! `pct_dropout_by_counts = (1 - detected / cells) * 100` (both `NaN` without cells).
//! Does not: run the pass (`gene_steps.rs` does, with the step's cell profiles), pick
//! the layer (the resolver did, at plan build), or compute the `log1p_` variants
//! (`.log1p()` on the expression does).

use kyto_core::GeneQcMetric;
use kyto_error::KytoError;
use kyto_store::{CellBlock, CellValues};
use polars::prelude::{NamedFrom, Series};

/// What one layer's rows give each gene present: the cells seen, and per gene the
/// cells with a non-zero value and the sum of its values.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GeneCounts {
    cells: u64,
    detected: Vec<u64>,
    totals: Vec<f64>,
}

impl GeneCounts {
    /// No cells yet, over `n_genes` genes.
    pub(crate) fn empty(n_genes: usize) -> Self {
        Self {
            cells: 0,
            detected: vec![0; n_genes],
            totals: vec![0.0; n_genes],
        }
    }

    /// Add the rows of `block`, in cell order. A stored `NaN` counts as non-zero, as
    /// the per-cell `genes_detected()` and numpy's `count_nonzero`.
    ///
    /// # Errors
    /// `KY-E117` if `block` is not over this many genes (a kyto bug).
    pub(crate) fn fold(&mut self, block: &CellBlock) -> Result<(), KytoError> {
        if block.n_genes() != self.totals.len() as u64 {
            return Err(KytoError::InvalidCellBlock {
                context: "per-gene QC metrics".to_owned(),
                problem: format!(
                    "a block of {} genes reaches counts over {}",
                    block.n_genes(),
                    self.totals.len()
                ),
            });
        }
        self.cells += block.n_rows() as u64;
        let indices = block.indices();
        match block.data() {
            CellValues::F32(values) => {
                for (gene, value) in indices.iter().zip(values) {
                    self.add(*gene, f64::from(*value));
                }
            }
            CellValues::U32(values) => {
                for (gene, value) in indices.iter().zip(values) {
                    self.add(*gene, f64::from(*value));
                }
            }
        }
        Ok(())
    }

    fn add(&mut self, gene: u32, value: f64) {
        let gene = gene as usize;
        self.detected[gene] += u64::from(value != 0.0);
        self.totals[gene] += value;
    }

    /// The counts of `self`'s cells followed by `right`'s.
    pub(crate) fn combine(mut self, right: Self) -> Self {
        self.cells += right.cells;
        for (left, right) in self.detected.iter_mut().zip(right.detected) {
            *left += right;
        }
        for (left, right) in self.totals.iter_mut().zip(right.totals) {
            *left += right;
        }
        self
    }

    /// `metric` of each gene, as `Float64` (the dtype every numeric leaf takes part in
    /// arithmetic with; a plain read of detected cells is cast to its declared
    /// `UInt32`).
    pub(crate) fn values(&self, metric: GeneQcMetric) -> Series {
        let cells = self.cells as f64;
        let values: Vec<f64> = match metric {
            GeneQcMetric::CellsDetected => self.detected.iter().map(|n| *n as f64).collect(),
            GeneQcMetric::TotalCounts => self.totals.clone(),
            GeneQcMetric::MeanCounts => self.totals.iter().map(|total| total / cells).collect(),
            GeneQcMetric::PercentDropout => self
                .detected
                .iter()
                .map(|n| (1.0 - *n as f64 / cells) * 100.0)
                .collect(),
        };
        Series::new("".into(), values)
    }
}

#[cfg(test)]
mod tests {
    use kyto_core::GeneQcMetric;
    use kyto_store::CellBlock;

    use super::GeneCounts;

    fn block(indptr: Vec<u64>, indices: Vec<u32>, data: Vec<f32>) -> CellBlock {
        CellBlock::from_f32(indptr, indices, data, 3).unwrap()
    }

    fn floats(counts: &GeneCounts, metric: GeneQcMetric) -> Vec<f64> {
        let values = counts.values(metric);
        values.f64().unwrap().iter().map(Option::unwrap).collect()
    }

    #[test]
    fn metrics_follow_scanpy_per_gene() {
        // 4 cells x 3 genes: [[1, 0, 2], [0, 0, 3], [4, 0, 0], [0, 0, 0]].
        let mut left = GeneCounts::empty(3);
        left.fold(&block(vec![0, 2, 3], vec![0, 2, 2], vec![1.0, 2.0, 3.0]))
            .unwrap();
        let mut right = GeneCounts::empty(3);
        right
            .fold(&block(vec![0, 1, 1], vec![0], vec![4.0]))
            .unwrap();
        let counts = left.combine(right);
        assert_eq!(
            floats(&counts, GeneQcMetric::CellsDetected),
            [2.0, 0.0, 2.0]
        );
        assert_eq!(floats(&counts, GeneQcMetric::TotalCounts), [5.0, 0.0, 5.0]);
        assert_eq!(floats(&counts, GeneQcMetric::MeanCounts), [1.25, 0.0, 1.25]);
        assert_eq!(
            floats(&counts, GeneQcMetric::PercentDropout),
            [50.0, 100.0, 50.0]
        );
    }

    #[test]
    fn no_cells_give_nan_means_and_percentages() {
        let counts = GeneCounts::empty(2);
        assert_eq!(floats(&counts, GeneQcMetric::CellsDetected), [0.0, 0.0]);
        assert!(
            floats(&counts, GeneQcMetric::MeanCounts)
                .iter()
                .all(|value| value.is_nan())
        );
        assert!(
            floats(&counts, GeneQcMetric::PercentDropout)
                .iter()
                .all(|value| value.is_nan())
        );
    }

    #[test]
    fn a_block_over_other_genes_is_refused() {
        let mut counts = GeneCounts::empty(2);
        let error = counts
            .fold(&block(vec![0, 1], vec![0], vec![1.0]))
            .unwrap_err();
        assert_eq!(error.code(), "KY-E117");
    }
}
