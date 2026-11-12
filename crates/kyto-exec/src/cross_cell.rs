//! Cross-cell nodes in the executor (task 5.4): statistics passes over the whole
//! input stream, then, for a node with an apply pass, a per-cell map with what they
//! found.
//!
//! Does: lower a median `normalize_total` (alone, or first in a fused pass) and
//! `highly_variable_genes` (with or without batches, task 6.3) to a [`CrossCellStage`]
//! that runs exactly the statistics passes the physical plan gives the node (task
//! 4.5); fold the blocks of a pass into leaves ([`Partial`], one per store cell block,
//! in cell order) that combine into the pass's result in a fixed tree in block order
//! (task 5.6, `reduction.rs`, R11), kept by an [`Accumulator`]; and turn the result
//! into [`NodeStatistics`] and, for `normalize_total`, the maps of its apply pass with
//! the median as their target. Median `normalize_total`: one pass of cell totals
//! (`f64`, [`Kernels::row_sums`], rounded to `f32` as scanpy keeps them, task 6.2),
//! target = median of the positive totals as `np.median(counts[counts > 0])` on
//! `float32` (NaN when there is none, as numpy). HVG: see `hvg_pass.rs` (what its
//! passes sum: `seurat` and `cell_ranger` one pass, `seurat_v3` two) and `hvg.rs`
//! (the gene columns, computed by the pipeline once the passes ended).
//! `scale` (task 6.4): one pass of per-gene moments, then its apply pass (see
//! `scale.rs`).
//! Does not: `pca` (6.5), or run passes (see `reduction.rs`).

use std::collections::BTreeSet;
use std::sync::Arc;

use kyto_error::KytoError;
use kyto_plan::{ElementwiseOp, HvgFlavor, MapElementwise};
use kyto_store::{CellBlock, CellValues};
use polars::prelude::DataFrame;

use crate::elementwise::{ElementwiseMaps, cell_totals};
use crate::hvg_pass::{BatchColumn, GroupPartial, HvgAccumulator};
use crate::kernels::Kernels;
use crate::operator::Operator;
use crate::pipeline_block::PipelineBlock;
use crate::scale::{ScaleMap, ScaleStatistics, fold_moments};
use crate::statistics::{GeneMoments, NodeStatistics};

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
    /// `highly_variable_genes` of `layer`, per batch of the cell column `batch_key`.
    HighlyVariableGenes {
        layer: String,
        flavor: HvgFlavor,
        n_top_genes: Option<u64>,
        batch_key: Option<String>,
    },
    /// `scale` of `layer` into layer `into`.
    Scale {
        layer: String,
        into: String,
        zero_center: bool,
        max_value: Option<f64>,
    },
}

impl CrossCellStage {
    /// A stage for optimized node `node` with the plan's `passes`.
    ///
    /// # Panics
    /// If `passes` is not the number the node's statistics need: the physical plan
    /// (kyto-plan) and the executor disagree, a kyto bug.
    pub(crate) fn new(node: usize, passes: u32, kind: CrossCellKind) -> Self {
        let needed = match &kind {
            CrossCellKind::MedianNormalize { .. } | CrossCellKind::Scale { .. } => 1,
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
            CrossCellKind::HighlyVariableGenes { layer, .. }
            | CrossCellKind::Scale { layer, .. } => layer,
        }
    }

    /// The `batch_key` cell column the passes read, if any.
    pub(crate) fn batch_key(&self) -> Option<&str> {
        match &self.kind {
            CrossCellKind::HighlyVariableGenes { batch_key, .. } => batch_key.as_deref(),
            CrossCellKind::MedianNormalize { .. } | CrossCellKind::Scale { .. } => None,
        }
    }

    /// An empty accumulator for a stream of `n_genes` genes; `cells` is the store's
    /// cell table (`store` its path), which a `batch_key` is read from.
    pub(crate) fn accumulator(
        &self,
        n_genes: usize,
        cells: Option<Arc<DataFrame>>,
        store: &str,
    ) -> Accumulator {
        match &self.kind {
            CrossCellKind::MedianNormalize { .. } => Accumulator::Totals(Vec::new()),
            CrossCellKind::Scale { .. } => {
                Accumulator::Moments(Box::new(GeneMoments::new(n_genes, false)))
            }
            CrossCellKind::HighlyVariableGenes {
                layer,
                flavor,
                batch_key,
                ..
            } => {
                let batch = batch_key
                    .as_ref()
                    .zip(cells)
                    .map(|(name, table)| BatchColumn {
                        name: name.clone(),
                        table,
                        store: store.to_owned(),
                    });
                Accumulator::Genes(Box::new(HvgAccumulator::new(
                    layer, *flavor, n_genes, batch,
                )))
            }
        }
    }

    /// The node's statistics (for HVG without its gene columns, which need the gene
    /// table: see `hvg::columns`), and the operator of its apply pass if it has one.
    pub(crate) fn finish(&self, accumulator: Accumulator) -> (NodeStatistics, Option<Operator>) {
        match (accumulator, &self.kind) {
            (Accumulator::Totals(mut totals), CrossCellKind::MedianNormalize { maps, keep }) => {
                let median = median(&mut totals);
                let apply = ElementwiseMaps::new(maps.clone(), keep.as_ref()).with_median(median);
                (
                    NodeStatistics::MedianTotal(median),
                    Some(Operator::Maps(apply)),
                )
            }
            (Accumulator::Genes(genes), _) => (
                NodeStatistics::HighlyVariableGenes(Box::new(genes.finish())),
                None,
            ),
            (
                Accumulator::Moments(moments),
                CrossCellKind::Scale {
                    layer,
                    into,
                    zero_center,
                    max_value,
                },
            ) => {
                let statistics = ScaleStatistics::from_moments(&moments);
                let apply = ScaleMap::new(
                    layer.clone(),
                    into.clone(),
                    *zero_center,
                    *max_value,
                    statistics.clone(),
                );
                (
                    NodeStatistics::Scale(Box::new(statistics)),
                    Some(Operator::Scale(apply)),
                )
            }
            (Accumulator::Totals(_) | Accumulator::Moments(_), _) => {
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
    /// `highly_variable_genes`' moments and fits.
    Genes(Box<HvgAccumulator>),
    /// `scale`'s per-gene moments.
    Moments(Box<GeneMoments>),
}

/// What one leaf of a pass's reduction holds (task 5.6): the sums over one store cell
/// block, in cell order, or two leaves' sums combined (`reduction.rs`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Partial {
    /// The positive cell totals, in cell order.
    Totals(Vec<f64>),
    /// Per-gene moments per group of cells.
    Groups(GroupPartial),
    /// Per-gene moments of every cell (`scale`).
    Moments(Box<GeneMoments>),
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
            (Self::Groups(left), Self::Groups(right)) => Self::Groups(left.combine(right)),
            (Self::Moments(mut left), Self::Moments(right)) => {
                left.combine(&right);
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
            Self::Genes(genes) => Partial::Groups(genes.leaf()),
            Self::Moments(moments) => {
                Partial::Moments(Box::new(GeneMoments::new(moments.sums.len(), false)))
            }
        }
    }

    /// Whether pass `pass` must fold every block into one leaf, in cell order (its sums
    /// depend on the order of the cells: `seurat_v3`'s Welford moments with batches).
    pub(crate) fn sequential(&self, pass: u32) -> bool {
        match self {
            Self::Totals(_) | Self::Moments(_) => false,
            Self::Genes(genes) => genes.sequential(pass),
        }
    }

    /// Add `layer` (the stream's input layer of `block`, rows of one cell block) to
    /// `leaf`, a leaf of pass `pass` (0-based); a leaf's blocks come in cell order.
    ///
    /// # Errors
    /// `KY-E117` if the block's genes are not the accumulator's (a stream that
    /// changed shape: a kyto bug, reported, never a panic); HVG's batch errors.
    pub(crate) fn fold(
        &self,
        pass: u32,
        leaf: &mut Partial,
        layer: &CellBlock,
        block: &PipelineBlock,
    ) -> Result<(), KytoError> {
        match (self, leaf) {
            (Self::Totals(_), Partial::Totals(totals)) => {
                let mut row_totals = Vec::new();
                let kernels = Kernels::detected();
                match layer.data() {
                    CellValues::F32(values) => {
                        cell_totals(kernels, layer.indptr(), values, &mut row_totals);
                    }
                    CellValues::U32(values) => {
                        cell_totals(kernels, layer.indptr(), values, &mut row_totals);
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
            (Self::Genes(genes), Partial::Groups(groups)) => genes.fold(pass, groups, layer, block),
            (Self::Moments(_), Partial::Moments(moments)) => fold_moments(moments, layer),
            _ => unreachable!("a pass's leaves come from Accumulator::leaf"),
        }
    }

    /// End pass `pass` with its reduction over every leaf (`None` for a stream without
    /// cells): keep it, and for `seurat_v3`'s first pass fit the second pass's clip.
    ///
    /// # Errors
    /// `KY-E215` if `seurat_v3` read a value that is not a count.
    pub(crate) fn end_pass(
        &mut self,
        pass: u32,
        reduced: Option<Partial>,
    ) -> Result<(), KytoError> {
        match (self, reduced) {
            (Self::Totals(totals), Some(Partial::Totals(reduced))) => *totals = reduced,
            (Self::Totals(totals), None) => totals.clear(),
            (Self::Genes(genes), Some(Partial::Groups(reduced))) => {
                genes.end_pass(pass, Some(reduced))?;
            }
            (Self::Genes(genes), None) => genes.end_pass(pass, None)?,
            (Self::Moments(moments), Some(Partial::Moments(reduced))) => *moments = reduced,
            (Self::Moments(moments), None) => {
                **moments = GeneMoments::new(moments.sums.len(), false);
            }
            _ => unreachable!("a pass's reduction comes from Accumulator::leaf"),
        }
        Ok(())
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
    use polars::prelude::DataFrame;

    use super::*;

    /// A stream block holding `layer` alone, over cells `0..rows`.
    fn pipeline_block(layer: &CellBlock) -> PipelineBlock {
        let rows = layer.indptr().len() - 1;
        PipelineBlock {
            cell_block: 0,
            cells: (0..rows as u64).collect(),
            layers: std::collections::BTreeMap::new(),
            columns: DataFrame::empty_with_height(rows),
        }
    }

    /// Run every pass of `stage` over `block`, one leaf a pass.
    fn run_passes(stage: &CrossCellStage, n_genes: usize, block: &CellBlock) -> Accumulator {
        let mut accumulator = stage.accumulator(n_genes, None, "store");
        let rows = pipeline_block(block);
        for pass in 0..stage.passes {
            let mut leaf = accumulator.leaf();
            accumulator.fold(pass, &mut leaf, block, &rows).unwrap();
            accumulator.end_pass(pass, Some(leaf)).unwrap();
        }
        accumulator
    }

    fn hvg(flavor: HvgFlavor) -> CrossCellKind {
        CrossCellKind::HighlyVariableGenes {
            layer: "counts".to_owned(),
            flavor,
            n_top_genes: None,
            batch_key: None,
        }
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
    fn gene_moments_follow_mean_var() {
        // Cells: [1, 0] [3, 2]
        let block = CellBlock::from_u32(vec![0, 1, 3], vec![0, 0, 1], vec![1, 3, 2], 2).unwrap();
        let stage = CrossCellStage::new(0, 1, hvg(HvgFlavor::CellRanger));
        let accumulator = run_passes(&stage, 2, &block);
        let (NodeStatistics::HighlyVariableGenes(statistics), None) = stage.finish(accumulator)
        else {
            panic!("expected HVG statistics and no apply pass")
        };
        let moments = statistics.moments();
        assert_eq!(moments.n_cells(), 2);
        assert_eq!(moments.sums(), [4.0, 2.0]);
        assert_eq!(moments.sums_of_squares(), [10.0, 4.0]);
        assert_eq!(moments.positives(), [2, 1]);
        assert_eq!(moments.means(), [2.0, 1.0]);
        assert_eq!(moments.variances(), [2.0, 2.0]);
        assert!(statistics.clipped().is_none());
    }

    #[test]
    fn seurat_v3_clips_in_its_second_pass() {
        // 8 cells, 25 genes: gene 0 is 1 in every cell (variance 0, so reg_std 1 and
        // clip 1 + sqrt(8)); genes 1..25 vary, enough points for
        // the loess (skmisc refuses fewer).
        let (n_cells, n_genes) = (8_u32, 25_u32);
        let (mut indptr, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
        for cell in 0..n_cells {
            for gene in 0..n_genes {
                let value = if gene == 0 {
                    1
                } else {
                    (cell * gene + gene) % (gene + 3)
                };
                if value != 0 {
                    indices.push(gene);
                    values.push(value);
                }
            }
            indptr.push(indices.len() as u64);
        }
        let block = CellBlock::from_u32(indptr, indices, values, u64::from(n_genes)).unwrap();
        let stage = CrossCellStage::new(0, 2, hvg(HvgFlavor::SeuratV3));
        let accumulator = run_passes(&stage, n_genes as usize, &block);
        let (NodeStatistics::HighlyVariableGenes(statistics), None) = stage.finish(accumulator)
        else {
            panic!("expected HVG statistics")
        };
        let clipped = statistics.clipped().unwrap();
        assert_eq!(clipped.clip()[0], 1.0 + 8_f64.sqrt());
        assert_eq!(clipped.regularized_std()[0], 1.0);
        assert_eq!(clipped.moments().sums()[0], 8.0);
        assert_eq!(clipped.moments().n_cells(), 8);
        assert!(clipped.clip()[1..].iter().all(|clip| clip.is_finite()));
    }

    #[test]
    fn seurat_v3_refuses_values_that_are_not_counts() {
        let block = CellBlock::from_f32(vec![0, 1], vec![0], vec![1.5], 1).unwrap();
        let stage = CrossCellStage::new(0, 2, hvg(HvgFlavor::SeuratV3));
        let mut accumulator = stage.accumulator(1, None, "store");
        let mut leaf = accumulator.leaf();
        accumulator
            .fold(0, &mut leaf, &block, &pipeline_block(&block))
            .unwrap();
        let error = accumulator.end_pass(0, Some(leaf)).unwrap_err();
        assert_eq!(error.code(), "KY-E215");
        assert!(error.to_string().contains("1.5"), "{error}");
    }

    #[test]
    fn seurat_moments_are_of_expm1_in_float32() {
        let block = CellBlock::from_f32(vec![0, 1], vec![0], vec![1.0], 1).unwrap();
        let stage = CrossCellStage::new(0, 1, hvg(HvgFlavor::Seurat));
        let accumulator = run_passes(&stage, 1, &block);
        let (NodeStatistics::HighlyVariableGenes(statistics), _) = stage.finish(accumulator) else {
            panic!("expected HVG statistics")
        };
        let expm1 = 1.0_f64.exp_m1() as f32;
        assert_eq!(statistics.moments().sums(), [f64::from(expm1)]);
        assert_eq!(
            statistics.moments().sums_of_squares(),
            [f64::from(expm1 * expm1)]
        );
    }

    #[test]
    #[should_panic(expected = "statistics passes")]
    fn a_pass_count_the_statistics_do_not_need_is_a_bug() {
        let _ = CrossCellStage::new(3, 1, hvg(HvgFlavor::SeuratV3));
    }
}
