//! `scale` in the executor (task 6.4): one statistics pass of per-gene moments, then
//! an apply pass that z-scores each gene into a new `float32` layer.
//!
//! Does: in scanpy 1.12.5's steps (`pp.scale` on a CSR matrix), sum each gene's values
//! and squares in `f64` (fast_array_utils' `mean_var`: a `float32` value is squared in
//! `float32`, a `uint32` count is cast to `float64` first, as scanpy casts integer
//! data); mean `sum / n`, variance `(squares / n - mean^2) * n / (n - 1)`
//! ([`GeneMoments`]), standard deviation its square root with 0 replaced by 1. The
//! apply pass computes in `f64` and rounds once to `float32`:
//! - `zero_center`: `(x - mean) / std` for every gene of every cell (a dense row; the
//!   exact zeros are not stored, R9), clipped to `[-max_value, max_value]`. scanpy
//!   returns this as a dense `float64` array; kyto rounds it to `float32` (layers are
//!   `float32`, deviations.md).
//! - otherwise: `x / std` for the stored values only (the block keeps its structure),
//!   clipped above at `max_value` only, as scanpy's `scale_and_clip_csr`.
//!
//! A NaN stays NaN (no comparison clips it, as scanpy's numba loops).
//! Does not: scanpy's `mask_obs`, or write scanpy's `var['mean']`/`var['std']` gene
//! columns (the statistics are kept in [`ScaleStatistics`]; deviations.md).

use std::sync::Arc;

use kyto_error::KytoError;
use kyto_store::{CellBlock, CellData, CellValues, ValueDtype};

use crate::cell_block_pool::CellBlockPool;
use crate::kernels::KernelValue;
use crate::pipeline_block::PipelineBlock;
use crate::statistics::GeneMoments;

/// Add the rows of `layer` to `moments` (scanpy's `mean_var` sums).
///
/// # Errors
/// `KY-E117` if the block's genes are not the moments' (a stream that changed shape:
/// a kyto bug, reported, never a panic).
pub(crate) fn fold_moments(moments: &mut GeneMoments, layer: &CellBlock) -> Result<(), KytoError> {
    if layer.n_genes() as usize != moments.sums.len() {
        return Err(KytoError::InvalidCellBlock {
            context: "scale statistics pass".to_owned(),
            problem: format!(
                "a statistics pass over {} genes got a block of {} genes",
                moments.sums.len(),
                layer.n_genes()
            ),
        });
    }
    moments.n_cells += layer.indptr().len().saturating_sub(1) as u64;
    match layer.data() {
        CellValues::F32(values) => {
            for (&gene, &value) in layer.indices().iter().zip(values) {
                let gene = gene as usize;
                moments.sums[gene] += f64::from(value);
                // fast_array_utils squares the `float32` value in `float32`.
                moments.sums_of_squares[gene] += f64::from(value * value);
            }
        }
        CellValues::U32(values) => {
            for (&gene, &value) in layer.indices().iter().zip(values) {
                let gene = gene as usize;
                // scanpy casts integer data to `float64` first.
                let value = f64::from(value);
                moments.sums[gene] += value;
                moments.sums_of_squares[gene] += value * value;
            }
        }
    }
    Ok(())
}

/// What `scale`'s statistics pass found: each gene's mean and standard deviation,
/// as scanpy's `var['mean']` and `var['std']`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleStatistics {
    pub(crate) means: Arc<Vec<f64>>,
    pub(crate) standard_deviations: Arc<Vec<f64>>,
}

impl ScaleStatistics {
    /// The statistics of `moments`: means, and standard deviations (`ddof=1`) with 0
    /// replaced by 1, as scanpy's `std[std == 0] = 1`.
    pub(crate) fn from_moments(moments: &GeneMoments) -> Self {
        let standard_deviations = moments
            .variances()
            .into_iter()
            .map(|variance| {
                let std = variance.sqrt();
                if std == 0.0 { 1.0 } else { std }
            })
            .collect();
        Self {
            means: Arc::new(moments.means()),
            standard_deviations: Arc::new(standard_deviations),
        }
    }

    /// Each gene's mean over the cells (NaN without cells).
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// Each gene's standard deviation over the cells (`ddof=1`), 1 for a gene without
    /// variation (scanpy's `var['std']`).
    pub fn standard_deviations(&self) -> &[f64] {
        &self.standard_deviations
    }
}

/// The apply pass of `scale`: writes layer `into` from layer `layer`.
#[derive(Debug, Clone)]
pub(crate) struct ScaleMap {
    layer: String,
    into: String,
    zero_center: bool,
    max_value: Option<f64>,
    statistics: ScaleStatistics,
}

impl ScaleMap {
    pub(crate) fn new(
        layer: String,
        into: String,
        zero_center: bool,
        max_value: Option<f64>,
        statistics: ScaleStatistics,
    ) -> Self {
        Self {
            layer,
            into,
            zero_center,
            max_value,
            statistics,
        }
    }

    /// Write the scaled layer into `block` (replacing a layer of the same name).
    ///
    /// # Errors
    /// `KY-E109` if the input layer is not in the block (never for a plan built by
    /// kyto); `KY-E117` if the block's genes are not the statistics' (a kyto bug).
    pub(crate) fn apply(
        &self,
        mut block: PipelineBlock,
        pool: &CellBlockPool,
        store: &str,
    ) -> Result<PipelineBlock, KytoError> {
        let input = block
            .layers
            .get(&self.layer)
            .ok_or_else(|| KytoError::MissingMember {
                kind: "layer".to_owned(),
                name: self.layer.clone(),
                store: store.to_owned(),
                suggestions: Box::default(),
            })?;
        let n_genes = self.statistics.means.len();
        if input.n_genes() as usize != n_genes {
            return Err(KytoError::InvalidCellBlock {
                context: "scale apply pass".to_owned(),
                problem: format!(
                    "statistics of {n_genes} genes, a block of {} genes",
                    input.n_genes()
                ),
            });
        }
        let output = match input.data() {
            CellValues::F32(values) => self.scale_block(input, values, pool)?,
            CellValues::U32(values) => self.scale_block(input, values, pool)?,
        };
        if let Some(replaced) = block.layers.insert(self.into.clone(), output) {
            pool.recycle(replaced);
        }
        Ok(block)
    }

    fn scale_block<T: KernelValue>(
        &self,
        input: &CellBlock,
        values: &[T],
        pool: &CellBlockPool,
    ) -> Result<CellBlock, KytoError> {
        let means = self.statistics.means.as_slice();
        let stds = self.statistics.standard_deviations.as_slice();
        let indptr = input.indptr();
        let indices = input.indices();
        let mut parts = pool.take(ValueDtype::F32);
        let CellData::F32(data) = &mut parts.data else {
            unreachable!("the pool gives the dtype asked for")
        };
        parts.indptr.push(0);
        if self.zero_center {
            let n_genes = means.len();
            let rows = indptr.len().saturating_sub(1);
            parts.indices.reserve(rows * n_genes);
            data.reserve(rows * n_genes);
            for row in 0..rows {
                let entries = indptr[row] as usize..indptr[row + 1] as usize;
                let mut stored = indices[entries.clone()]
                    .iter()
                    .zip(&values[entries])
                    .peekable();
                for gene in 0..n_genes {
                    let x = match stored.peek() {
                        Some((index, value)) if **index as usize == gene => {
                            let x = value.to_f64();
                            stored.next();
                            x
                        }
                        _ => 0.0,
                    };
                    let scaled = clip((x - means[gene]) / stds[gene], self.max_value, true);
                    let scaled = scaled as f32;
                    if scaled != 0.0 {
                        parts.indices.push(gene as u32);
                        data.push(scaled);
                    }
                }
                parts.indptr.push(parts.indices.len() as u64);
            }
        } else {
            parts.indptr.extend_from_slice(&indptr[1..]);
            parts.indices.extend_from_slice(indices);
            data.reserve(values.len());
            for (&gene, value) in indices.iter().zip(values) {
                let scaled = clip(value.to_f64() / stds[gene as usize], self.max_value, false);
                data.push(scaled as f32);
            }
        }
        CellBlock::from_parts(parts, input.n_genes())
    }
}

/// scanpy's clip: above `max_value` becomes `max_value`, below `-max_value` becomes
/// `-max_value` when `zero_center` (scanpy's `clip_array`; its sparse path without
/// centring clips above only). NaN is kept.
fn clip(value: f64, max_value: Option<f64>, zero_center: bool) -> f64 {
    match max_value {
        Some(max) if value > max => max,
        Some(max) if zero_center && value < -max => -max,
        _ => value,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use polars::prelude::DataFrame;

    use super::*;

    /// Cells: [1, 0, 0] [3, 2, 0] [0, 4, 0]; gene 2 never varies.
    fn counts() -> CellBlock {
        CellBlock::from_u32(vec![0, 1, 3, 4], vec![0, 0, 1, 1], vec![1, 3, 2, 4], 3).unwrap()
    }

    fn statistics(block: &CellBlock) -> ScaleStatistics {
        let mut moments = GeneMoments::new(block.n_genes() as usize, false);
        fold_moments(&mut moments, block).unwrap();
        ScaleStatistics::from_moments(&moments)
    }

    fn run(block: CellBlock, zero_center: bool, max_value: Option<f64>) -> CellBlock {
        let map = ScaleMap::new(
            "counts".to_owned(),
            "scaled".to_owned(),
            zero_center,
            max_value,
            statistics(&block),
        );
        let rows = block.indptr().len() - 1;
        let pipeline_block = PipelineBlock {
            cell_block: 0,
            cells: (0..rows as u64).collect(),
            layers: BTreeMap::from([("counts".to_owned(), block)]),
            columns: DataFrame::empty_with_height(rows),
        };
        let pool = CellBlockPool::new(4);
        let mut out = map.apply(pipeline_block, &pool, "store").unwrap();
        out.layers.remove("scaled").unwrap()
    }

    fn dense(block: &CellBlock) -> Vec<Vec<f32>> {
        let CellValues::F32(values) = block.data() else {
            panic!("scale writes float32")
        };
        let indptr = block.indptr();
        (0..indptr.len() - 1)
            .map(|row| {
                let mut dense = vec![0.0; block.n_genes() as usize];
                for entry in indptr[row] as usize..indptr[row + 1] as usize {
                    dense[block.indices()[entry] as usize] = values[entry];
                }
                dense
            })
            .collect()
    }

    #[test]
    fn statistics_are_mean_var_with_ddof_1() {
        let statistics = statistics(&counts());
        assert_eq!(statistics.means(), [4.0 / 3.0, 2.0, 0.0]);
        // Gene 0: values 1, 3, 0; variance (10/3 - 16/9) * 3/2 = 7/3.
        let expected = ((10.0 / 3.0 - (4.0_f64 / 3.0).powi(2)) * 1.5).sqrt();
        assert_eq!(statistics.standard_deviations()[0], expected);
        assert_eq!(statistics.standard_deviations()[1], 2.0);
        assert_eq!(statistics.standard_deviations()[2], 1.0, "std 0 becomes 1");
    }

    #[test]
    fn zero_centred_is_dense_and_clipped_both_ways() {
        let block = counts();
        let statistics = statistics(&block);
        let stds = statistics.standard_deviations().to_vec();
        let out = run(block, true, Some(1.0));
        let expected = |x: f64, gene: usize| -> f32 {
            (((x - statistics.means()[gene]) / stds[gene]).clamp(-1.0, 1.0)) as f32
        };
        assert_eq!(
            dense(&out),
            vec![
                vec![expected(1.0, 0), expected(0.0, 1), 0.0],
                vec![expected(3.0, 0), expected(2.0, 1), 0.0],
                vec![expected(0.0, 0), expected(4.0, 1), 0.0],
            ]
        );
        // Gene 2 (all 0) and gene 1 in cell 1 (its mean) are exact zeros, not stored.
        assert_eq!(out.indices().len(), 5);
        assert!(dense(&out).iter().flatten().all(|v| v.abs() <= 1.0));
    }

    #[test]
    fn not_centred_keeps_the_structure_and_clips_above_only() {
        let block = counts();
        let stds = statistics(&block).standard_deviations().to_vec();
        let out = run(block.clone(), false, Some(1.0));
        assert_eq!(out.indptr(), block.indptr());
        assert_eq!(out.indices(), block.indices());
        let CellValues::F32(values) = out.data() else {
            panic!()
        };
        assert_eq!(
            values,
            [(1.0 / stds[0]) as f32, 1.0, (2.0 / stds[1]) as f32, 1.0]
        );
        // Without a max_value nothing is clipped.
        let out = run(block, false, None);
        let CellValues::F32(values) = out.data() else {
            panic!()
        };
        assert_eq!(values[1], (3.0 / stds[0]) as f32);
    }

    #[test]
    fn float32_squares_are_float32() {
        // 4097 squared is not a float32: scanpy squares in float32.
        let block = CellBlock::from_f32(vec![0, 1, 1], vec![0], vec![4097.0], 1).unwrap();
        let mut moments = GeneMoments::new(1, false);
        fold_moments(&mut moments, &block).unwrap();
        assert_eq!(moments.sums_of_squares(), [f64::from(4097.0_f32 * 4097.0)]);
        assert_ne!(moments.sums_of_squares(), [4097.0 * 4097.0]);
        assert_eq!(moments.n_cells(), 2);
    }

    #[test]
    fn nan_is_kept() {
        assert!(clip(f64::NAN, Some(1.0), true).is_nan());
        assert_eq!(clip(-3.0, Some(1.0), false), -3.0);
        assert_eq!(clip(-3.0, Some(1.0), true), -1.0);
    }

    #[test]
    fn a_block_of_other_genes_is_a_bug_reported() {
        let mut moments = GeneMoments::new(2, false);
        let error = fold_moments(&mut moments, &counts()).unwrap_err();
        assert_eq!(error.code(), "KY-E117");
    }
}
