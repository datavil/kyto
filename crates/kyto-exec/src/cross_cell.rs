//! Cross-cell nodes in the executor (task 5.4): statistics passes over the whole
//! input stream, then, for a node with an apply pass, a per-cell map with what they
//! found.
//!
//! Does: lower a median `normalize_total` (alone, or first in a fused pass) and
//! `highly_variable_genes` (no batches) to a [`CrossCellStage`] that runs exactly the
//! statistics passes the physical plan gives the node (task 4.5); fold the blocks of
//! a pass into leaves ([`Partial`], one per store cell block, in cell order) that
//! combine into the pass's result in a fixed tree in block order (task 5.6,
//! `reduction.rs`, R11), kept by an [`Accumulator`]; and turn the
//! result into [`NodeStatistics`] and, for `normalize_total`, the maps of its apply pass
//! with the median as their target. Median `normalize_total`: one pass of cell totals
//! (`f64`, [`Kernels::row_sums`], rounded to `f32` as scanpy keeps them, task 6.2),
//! target = median of the positive totals as `np.median(counts[counts > 0])` on
//! `float32` (NaN when there is none, as numpy). HVG `seurat` and
//! `cell_ranger`: one pass of per-gene moments (of `expm1(value)` for `seurat`, as
//! scanpy undoes the log first); `seurat_v3`: moments, then moments of
//! `min(value, clip)` per gene with `clip = mean + sqrt(n) * reg_std`.
//! Does not: fit `seurat_v3`'s loess (`reg_std` here is the gene's own standard
//! deviation, 1 for a constant gene, a stand-in until 6.3), rank or flag genes (6.3),
//! batches (6.3), `scale` (6.4) or `pca` (6.5), or run passes (see `reduction.rs`).

use std::collections::BTreeSet;

use kyto_error::KytoError;
use kyto_plan::{ElementwiseOp, HvgFlavor, MapElementwise};
use kyto_store::{CellBlock, CellValues};

use crate::elementwise::{ElementwiseMaps, cell_totals};
use crate::kernels::{KernelValue, Kernels};
use crate::statistics::{
    ClippedMoments, GeneMoments, HighlyVariableGeneStatistics, NodeStatistics,
};

/// A cross-cell node of the stream, with the statistics passes the physical plan
/// gives it.
#[derive(Debug, Clone)]
pub(crate) struct CrossCellStage {
    /// The node's index in the optimized plan.
    pub(crate) node: usize,
    /// The physical plan's statistics passes for the node.
    pub(crate) passes: u32,
    pub(crate) kind: CrossCellKind,
}

/// What a cross-cell node computes.
#[derive(Debug, Clone)]
pub(crate) enum CrossCellKind {
    /// A median `normalize_total`, then (fused) maps after it; `keep` as
    /// [`ElementwiseMaps::new`].
    MedianNormalize {
        maps: Vec<MapElementwise>,
        keep: Option<BTreeSet<String>>,
    },
    /// `highly_variable_genes` of `layer`, without batches.
    HighlyVariableGenes { layer: String, flavor: HvgFlavor },
}

impl CrossCellStage {
    /// A stage for optimized node `node` with the plan's `passes`.
    ///
    /// # Panics
    /// If `passes` is not the number the node's statistics need: the physical plan
    /// (kyto-plan) and the executor disagree, a kyto bug.
    pub(crate) fn new(node: usize, passes: u32, kind: CrossCellKind) -> Self {
        let needed = match &kind {
            CrossCellKind::MedianNormalize { .. } => 1,
            CrossCellKind::HighlyVariableGenes { flavor, .. } => match flavor {
                HvgFlavor::Seurat | HvgFlavor::CellRanger => 1,
                HvgFlavor::SeuratV3 => 2,
            },
        };
        assert_eq!(
            passes, needed,
            "the physical plan gives node {node} {passes} statistics passes, its statistics need {needed}"
        );
        Self { node, passes, kind }
    }

    /// The layer each statistics pass reads from the stream.
    pub(crate) fn input_layer(&self) -> &str {
        match &self.kind {
            CrossCellKind::MedianNormalize { maps, .. } => {
                maps.first().map_or("", MapElementwise::layer)
            }
            CrossCellKind::HighlyVariableGenes { layer, .. } => layer,
        }
    }

    /// An empty accumulator for a stream of `n_genes` genes.
    pub(crate) fn accumulator(&self, n_genes: usize) -> Accumulator {
        match &self.kind {
            CrossCellKind::MedianNormalize { .. } => Accumulator::Totals(Vec::new()),
            CrossCellKind::HighlyVariableGenes { flavor, .. } => Accumulator::Genes {
                flavor: *flavor,
                moments: GeneMoments::new(n_genes),
                clipped: None,
            },
        }
    }

    /// The node's statistics, and the maps of its apply pass if it has one.
    pub(crate) fn finish(
        &self,
        accumulator: Accumulator,
    ) -> (NodeStatistics, Option<ElementwiseMaps>) {
        match (accumulator, &self.kind) {
            (Accumulator::Totals(mut totals), CrossCellKind::MedianNormalize { maps, keep }) => {
                let median = median(&mut totals);
                let apply = ElementwiseMaps::new(maps.clone(), keep.as_ref()).with_median(median);
                (NodeStatistics::MedianTotal(median), Some(apply))
            }
            (
                Accumulator::Genes {
                    flavor,
                    moments,
                    clipped,
                },
                _,
            ) => (
                NodeStatistics::HighlyVariableGenes(HighlyVariableGeneStatistics {
                    flavor,
                    moments,
                    clipped,
                }),
                None,
            ),
            (Accumulator::Totals(_), CrossCellKind::HighlyVariableGenes { .. }) => {
                unreachable!("a stage's accumulator comes from CrossCellStage::accumulator")
            }
        }
    }
}

/// What a stage's passes have summed so far: the result of every pass that ended.
#[derive(Debug, Clone)]
pub(crate) enum Accumulator {
    /// The positive cell totals, in cell order.
    Totals(Vec<f64>),
    /// Per-gene moments; `clipped` once the first pass of `seurat_v3` ended (its clip
    /// set, its moments those of the second pass).
    Genes {
        flavor: HvgFlavor,
        moments: GeneMoments,
        clipped: Option<ClippedMoments>,
    },
}

/// What one leaf of a pass's reduction holds (task 5.6): the sums over one store cell
/// block, in cell order, or two leaves' sums combined (`reduction.rs`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Partial {
    /// The positive cell totals, in cell order.
    Totals(Vec<f64>),
    /// Per-gene moments.
    Moments(GeneMoments),
}

impl Partial {
    /// `self` (the cells before) and `right` (the cells after) as one partial: totals
    /// one after the other, moments added gene by gene.
    pub(crate) fn combine(self, right: Self) -> Self {
        match (self, right) {
            (Self::Totals(mut left), Self::Totals(right)) => {
                left.extend(right);
                Self::Totals(left)
            }
            (Self::Moments(mut left), Self::Moments(right)) => {
                left.n_cells += right.n_cells;
                for (sum, right) in left.sums.iter_mut().zip(&right.sums) {
                    *sum += right;
                }
                for (squares, right) in left.sums_of_squares.iter_mut().zip(&right.sums_of_squares)
                {
                    *squares += right;
                }
                Self::Moments(left)
            }
            _ => unreachable!("one pass's partials come from Accumulator::leaf"),
        }
    }
}

impl Accumulator {
    /// An empty leaf of the next pass's reduction.
    pub(crate) fn leaf(&self) -> Partial {
        match self {
            Self::Totals(_) => Partial::Totals(Vec::new()),
            Self::Genes { moments, .. } => Partial::Moments(GeneMoments::new(moments.sums.len())),
        }
    }

    /// Add `block` (the stream's input layer of rows of one cell block) to `leaf`, a
    /// leaf of pass `pass` (0-based); a leaf's blocks come in cell order.
    ///
    /// # Errors
    /// `KY-E117` if the block's genes are not the accumulator's (a stream that
    /// changed shape: a kyto bug, reported, never a panic).
    pub(crate) fn fold(
        &self,
        pass: u32,
        leaf: &mut Partial,
        block: &CellBlock,
    ) -> Result<(), KytoError> {
        match (self, leaf) {
            (Self::Totals(_), Partial::Totals(totals)) => {
                let mut row_totals = Vec::new();
                let kernels = Kernels::detected();
                match block.data() {
                    CellValues::F32(values) => {
                        cell_totals(kernels, block.indptr(), values, &mut row_totals);
                    }
                    CellValues::U32(values) => {
                        cell_totals(kernels, block.indptr(), values, &mut row_totals);
                    }
                }
                // scanpy keeps the totals in the data's float32 (task 6.2); NaN is not
                // positive, as numpy's `counts > 0`.
                totals.extend(
                    row_totals
                        .into_iter()
                        .map(|total| f64::from(total as f32))
                        .filter(|total| *total > 0.0),
                );
                Ok(())
            }
            (
                Self::Genes {
                    flavor, clipped, ..
                },
                Partial::Moments(moments),
            ) => {
                if block.n_genes() as usize != moments.sums.len() {
                    return Err(KytoError::InvalidCellBlock {
                        context: "highly_variable_genes statistics pass".to_owned(),
                        problem: format!(
                            "a statistics pass over {} genes got a block of {} genes",
                            moments.sums.len(),
                            block.n_genes()
                        ),
                    });
                }
                let mut gene_values = GeneValues::new(block);
                if pass == 0 {
                    let expm1 = *flavor == HvgFlavor::Seurat;
                    add_moments_by_gene(moments, &mut gene_values, |_, value| {
                        if expm1 { value.exp_m1() } else { value }
                    });
                } else {
                    let Some(clipped) = clipped else {
                        unreachable!("end_pass(0) of seurat_v3 sets the clip")
                    };
                    let clip = &clipped.clip;
                    add_moments_by_gene(moments, &mut gene_values, |gene, value| {
                        value.min(clip[gene])
                    });
                }
                Ok(())
            }
            _ => unreachable!("a pass's leaves come from Accumulator::leaf"),
        }
    }

    /// End pass `pass` with its reduction over every leaf (`None` for a stream without
    /// cells): keep it, and for `seurat_v3`'s first pass set the second pass's clip.
    pub(crate) fn end_pass(&mut self, pass: u32, reduced: Option<Partial>) {
        let reduced = reduced.unwrap_or_else(|| self.leaf());
        match (self, reduced) {
            (Self::Totals(totals), Partial::Totals(reduced)) => *totals = reduced,
            (
                Self::Genes {
                    flavor,
                    moments,
                    clipped,
                },
                Partial::Moments(reduced),
            ) => {
                if pass == 0 {
                    *moments = reduced;
                    if *flavor == HvgFlavor::SeuratV3 {
                        *clipped = Some(ClippedMoments {
                            clip: seurat_v3_clip(moments),
                            moments: GeneMoments::new(moments.sums.len()),
                        });
                    }
                } else if let Some(clipped) = clipped {
                    clipped.moments = reduced;
                }
            }
            _ => unreachable!("a pass's reduction comes from Accumulator::leaf"),
        }
    }
}

/// `seurat_v3`'s clip per gene: `mean + sqrt(n) * reg_std`, `reg_std` the gene's own
/// standard deviation (1 for a gene of variance 0 or NaN, as scanpy's unfitted genes):
/// a stand-in for scanpy's loess-regressed one until 6.3.
fn seurat_v3_clip(moments: &GeneMoments) -> Vec<f64> {
    let sqrt_n = (moments.n_cells as f64).sqrt();
    moments
        .means()
        .into_iter()
        .zip(moments.variances())
        .map(|(mean, variance)| {
            let reg_std = if variance > 0.0 { variance.sqrt() } else { 1.0 };
            reg_std * sqrt_n + mean
        })
        .collect()
}

fn add_moments_by_gene(
    moments: &mut GeneMoments,
    gene_values: &mut GeneValues<'_>,
    transform: impl Fn(usize, f64) -> f64,
) {
    moments.n_cells += gene_values.rows;
    gene_values.for_each(|gene, value| {
        let value = transform(gene, value);
        moments.sums[gene] += value;
        moments.sums_of_squares[gene] += value * value;
    });
}

/// The (gene, value as `f64`) of every stored value of a block, in storage order.
struct GeneValues<'a> {
    block: &'a CellBlock,
    rows: u64,
}

impl<'a> GeneValues<'a> {
    fn new(block: &'a CellBlock) -> Self {
        let rows = block.indptr().len().saturating_sub(1) as u64;
        Self { block, rows }
    }

    fn for_each(&mut self, mut add: impl FnMut(usize, f64)) {
        fn each<T: KernelValue>(indices: &[u32], values: &[T], add: &mut impl FnMut(usize, f64)) {
            for (&gene, &value) in indices.iter().zip(values) {
                add(gene as usize, value.to_f64());
            }
        }
        match self.block.data() {
            CellValues::F32(values) => each(self.block.indices(), values, &mut add),
            CellValues::U32(values) => each(self.block.indices(), values, &mut add),
        }
    }
}

/// `np.median` of `values` (`float32` values held as `f64`, sorted here): the middle
/// value, or the mean of the two middle values in `float32` as numpy's `float32` mean
/// (`f32(f32(a + b) / 2)`, task 6.2); NaN for none.
fn median(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let n = values.len();
    match n {
        0 => f64::NAN,
        _ if n % 2 == 1 => values[n / 2],
        _ => f64::from((values[n / 2 - 1] as f32 + values[n / 2] as f32) / 2.0),
    }
}

/// The maps of a fused pass or a single map, with whether the first one is a median
/// `normalize_total` (the node is then cross-cell).
pub(crate) fn starts_with_median(maps: &[MapElementwise]) -> bool {
    maps.first()
        .is_some_and(|map| matches!(map.op(), ElementwiseOp::NormalizeTotal { target_sum: None }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run every pass of `stage` over `block`, one leaf a pass.
    fn run_passes(stage: &CrossCellStage, n_genes: usize, block: &CellBlock) -> Accumulator {
        let mut accumulator = stage.accumulator(n_genes);
        for pass in 0..stage.passes {
            let mut leaf = accumulator.leaf();
            accumulator.fold(pass, &mut leaf, block).unwrap();
            accumulator.end_pass(pass, Some(leaf));
        }
        accumulator
    }

    #[test]
    fn median_is_numpys() {
        assert!(median(&mut []).is_nan());
        assert_eq!(median(&mut [3.0]), 3.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0]), 3.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        // float32 mean of the two middle values: 2^24 + 1 is not a float32.
        assert_eq!(
            median(&mut [16_777_216.0, 2.0, 1.0, 16_777_218.0]),
            8_388_609.0
        );
        assert_eq!(
            median(&mut [16_777_218.0, 16_777_216.0]),
            16_777_216.0,
            "(2^24 + 2^24 + 2) / 2 in float32"
        );
    }

    #[test]
    fn gene_moments_follow_get_mean_var() {
        // Cells: [1, 0] [3, 2]
        let block = CellBlock::from_u32(vec![0, 1, 3], vec![0, 0, 1], vec![1, 3, 2], 2).unwrap();
        let stage = CrossCellStage::new(
            0,
            1,
            CrossCellKind::HighlyVariableGenes {
                layer: "counts".to_owned(),
                flavor: HvgFlavor::CellRanger,
            },
        );
        let accumulator = run_passes(&stage, 2, &block);
        let (NodeStatistics::HighlyVariableGenes(statistics), None) = stage.finish(accumulator)
        else {
            panic!("expected HVG statistics and no apply pass")
        };
        let moments = statistics.moments();
        assert_eq!(moments.n_cells(), 2);
        assert_eq!(moments.sums(), [4.0, 2.0]);
        assert_eq!(moments.sums_of_squares(), [10.0, 4.0]);
        assert_eq!(moments.means(), [2.0, 1.0]);
        assert_eq!(moments.variances(), [2.0, 2.0]);
        assert!(statistics.clipped().is_none());
    }

    #[test]
    fn seurat_v3_clips_in_its_second_pass() {
        // Gene 0 over 4 cells: [0, 0, 0, 8]: mean 2, variance 16, clip 2 + 2 * 4 = 10.
        // Gene 1: [1, 1, 1, 1]: variance 0, reg_std 1, clip 1 + 2 = 3.
        let block = CellBlock::from_u32(
            vec![0, 1, 2, 3, 5],
            vec![1, 1, 1, 0, 1],
            vec![1, 1, 1, 8, 1],
            2,
        )
        .unwrap();
        let stage = CrossCellStage::new(
            0,
            2,
            CrossCellKind::HighlyVariableGenes {
                layer: "counts".to_owned(),
                flavor: HvgFlavor::SeuratV3,
            },
        );
        let accumulator = run_passes(&stage, 2, &block);
        let (NodeStatistics::HighlyVariableGenes(statistics), None) = stage.finish(accumulator)
        else {
            panic!("expected HVG statistics")
        };
        let clipped = statistics.clipped().unwrap();
        assert_eq!(clipped.clip(), [10.0, 3.0]);
        assert_eq!(clipped.moments().sums(), [8.0, 4.0]);
        assert_eq!(clipped.moments().n_cells(), 4);
    }

    #[test]
    fn seurat_moments_are_of_expm1() {
        let block = CellBlock::from_f32(vec![0, 1], vec![0], vec![1.0], 1).unwrap();
        let stage = CrossCellStage::new(
            0,
            1,
            CrossCellKind::HighlyVariableGenes {
                layer: "log1p".to_owned(),
                flavor: HvgFlavor::Seurat,
            },
        );
        let accumulator = run_passes(&stage, 1, &block);
        let (NodeStatistics::HighlyVariableGenes(statistics), _) = stage.finish(accumulator) else {
            panic!("expected HVG statistics")
        };
        assert_eq!(statistics.moments().sums(), [1.0_f64.exp_m1()]);
    }

    #[test]
    #[should_panic(expected = "statistics passes")]
    fn a_pass_count_the_statistics_do_not_need_is_a_bug() {
        let _ = CrossCellStage::new(
            3,
            1,
            CrossCellKind::HighlyVariableGenes {
                layer: "counts".to_owned(),
                flavor: HvgFlavor::SeuratV3,
            },
        );
    }
}
