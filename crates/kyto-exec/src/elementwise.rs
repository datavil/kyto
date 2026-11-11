//! The per-cell value-by-value maps of a pipeline: `normalize_total(target_sum=...)` and
//! `log1p`, one map or a fused pass of them (task 4.4's elementwise fusion).
//!
//! Does: apply the maps in order to each block, each writing a `float32` layer (its
//! logical dtype, so a fused pass gives the bits the same maps give one by one, task
//! 4.4 Open (2)), and drop the layers a fused pass does not keep. Output buffers come
//! from the [`CellBlockPool`]; replaced and dropped blocks go back to it. On the 5.3
//! [`Kernels`], in scanpy's `float32` steps (task 6.2, so the bits are scanpy's):
//! `normalize_total` sums each row in `f64` ([`Kernels::row_sums`], lane order) and
//! rounds the total to `f32` (scanpy keeps `counts_per_cell` in the data's `float32`),
//! divides it by the target in `f32` (the divisor, [`normalize_divisor`]; a divisor of
//! 0, a cell without counts, becomes 1 and leaves the cell as it is, as scanpy's
//! `counts += counts == 0`) and divides each value by it in `f32`
//! ([`Kernels::divide_rows`]); `log1p` is kyto's [`crate::log1p`] ([`Kernels::log1p`],
//! correctly rounded where numpy's `float32` `log1p` may be 1 ulp off, deviations.md);
//! both keep zeros at zero, so a block keeps its structure.
//! A median `normalize_total` runs here once its statistics pass found the median
//! ([`ElementwiseMaps::with_median`], task 5.4).
//! Does not: find the median (`cross_cell.rs`), or scanpy's other parameters
//! (`exclude_highly_expressed`, `log1p`'s `base`: not asked for by 6.2).

use std::collections::BTreeSet;

use kyto_error::KytoError;
use kyto_plan::{ElementwiseOp, MapElementwise};
use kyto_store::{CellBlock, CellData, CellValues, ValueDtype};

use crate::cell_block_pool::CellBlockPool;
use crate::kernels::{KernelValue, Kernels};
use crate::pipeline_block::PipelineBlock;

/// Maps run in one pass over each block, then the layers to drop.
#[derive(Debug, Clone)]
pub(crate) struct ElementwiseMaps {
    maps: Vec<MapElementwise>,
    drop: Vec<String>,
    /// The target of a median `normalize_total`, once its statistics pass found it
    /// (task 5.4).
    median: Option<f64>,
}

impl ElementwiseMaps {
    /// `maps` (each per-cell) run in order; then the maps' outputs not in `keep` are
    /// dropped (`None` keeps them all, as a single unfused map does).
    pub(crate) fn new(maps: Vec<MapElementwise>, keep: Option<&BTreeSet<String>>) -> Self {
        let drop = match keep {
            None => Vec::new(),
            Some(keep) => maps
                .iter()
                .map(|map| map.output_layer().to_owned())
                .filter(|layer| !keep.contains(layer))
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
        };
        Self {
            maps,
            drop,
            median: None,
        }
    }

    /// These maps with `median` as the target of a median `normalize_total`.
    pub(crate) fn with_median(mut self, median: f64) -> Self {
        self.median = Some(median);
        self
    }

    /// Run the maps over `block`, then drop what the pass does not keep.
    ///
    /// # Errors
    /// `KY-E109` if a map's input layer is not in the block (a plan the optimizer
    /// trimmed wrongly; never for a plan built by kyto).
    pub(crate) fn apply(
        &self,
        mut block: PipelineBlock,
        pool: &CellBlockPool,
        store: &str,
    ) -> Result<PipelineBlock, KytoError> {
        for map in &self.maps {
            let input = block
                .layers
                .get(map.layer())
                .ok_or_else(|| KytoError::MissingMember {
                    kind: "layer".to_owned(),
                    name: map.layer().to_owned(),
                    store: store.to_owned(),
                })?;
            let op = match map.op() {
                ElementwiseOp::NormalizeTotal { target_sum: None } => {
                    ElementwiseOp::NormalizeTotal {
                        target_sum: self.median,
                    }
                }
                op => op,
            };
            let output = map_block(op, input, pool)?;
            if let Some(replaced) = block.layers.insert(map.output_layer().to_owned(), output) {
                pool.recycle(replaced);
            }
        }
        for layer in &self.drop {
            if let Some(dropped) = block.layers.remove(layer) {
                pool.recycle(dropped);
            }
        }
        Ok(block)
    }
}

/// `op` of every value of `input`, as a new `float32` block with the same structure.
fn map_block(
    op: ElementwiseOp,
    input: &CellBlock,
    pool: &CellBlockPool,
) -> Result<CellBlock, KytoError> {
    let mut parts = pool.take(ValueDtype::F32);
    parts.indptr.extend_from_slice(input.indptr());
    parts.indices.extend_from_slice(input.indices());
    if let CellData::F32(data) = &mut parts.data {
        // AlignedBuffer has no `resize`: zeros first, then the kernel writes over them.
        let len = input.indices().len();
        data.reserve(len);
        while data.len() < len {
            data.extend_from_slice(&ZEROS[..(len - data.len()).min(ZEROS.len())]);
        }
        match input.data() {
            CellValues::F32(values) => map_values(op, input.indptr(), values, data.as_mut_slice()),
            CellValues::U32(values) => map_values(op, input.indptr(), values, data.as_mut_slice()),
        }
    }
    CellBlock::from_parts(parts, input.n_genes())
}

/// Zeros to grow an output buffer by.
static ZEROS: [f32; 1024] = [0.0; 1024];

/// `op` of each value of the rows `indptr` cuts from `values`, written to `out`.
fn map_values<T: KernelValue>(op: ElementwiseOp, indptr: &[u64], values: &[T], out: &mut [f32]) {
    let kernels = Kernels::detected();
    match op {
        ElementwiseOp::NormalizeTotal { target_sum } => {
            let mut totals = Vec::new();
            cell_totals(kernels, indptr, values, &mut totals);
            let divisors: Vec<f32> = totals
                .into_iter()
                .map(|total| normalize_divisor(total, target_sum))
                .collect();
            kernels.divide_rows(indptr, values, &divisors, out);
        }
        // A u32 count is rounded to f32 first: the input layer as stored.
        ElementwiseOp::Log1p => kernels.log1p(values, out),
    }
}

/// Each row's total as scanpy's `normalize_total` sums it: of the values as `float32`
/// (scanpy casts integer data to `float32` first), in `f64` ([`Kernels::row_sums`]),
/// into `totals` (cleared first). A `uint32` count above 2^24 that is not a `float32`
/// is rounded before summing (bug-hunter, 6.2); other blocks are summed as they are.
pub(crate) fn cell_totals<T: KernelValue>(
    kernels: Kernels,
    indptr: &[u64],
    values: &[T],
    totals: &mut Vec<f64>,
) {
    let exact = values
        .iter()
        .all(|value| value.to_f64().to_bits() == f64::from(value.to_f32()).to_bits());
    if exact {
        kernels.row_sums(indptr, values, totals);
    } else {
        let rounded: Vec<f32> = values.iter().map(|value| value.to_f32()).collect();
        kernels.row_sums(indptr, &rounded, totals);
    }
}

/// What `normalize_total` divides a cell's values by, in scanpy's `float32` steps:
/// `f32(total) / f32(target_sum)`, or 1 when that is 0 (a cell without counts, or a
/// quotient that underflows: scanpy's `counts += counts == 0`). A NaN total gives NaN.
/// `None` (a median not found yet) is 1; the pipeline always sets the median first.
pub(crate) fn normalize_divisor(total: f64, target_sum: Option<f64>) -> f32 {
    let Some(target_sum) = target_sum else {
        return 1.0;
    };
    // scanpy: counts_per_cell (float32) / target_sum, a float32 division.
    let divisor = total as f32 / target_sum as f32;
    if divisor == 0.0 { 1.0 } else { divisor }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts() -> CellBlock {
        // Rows: [1, 3] [] [2]
        CellBlock::from_u32(vec![0, 2, 2, 3], vec![0, 2, 1], vec![1, 3, 2], 3).unwrap()
    }

    fn values(block: &CellBlock) -> Vec<f32> {
        match block.data() {
            CellValues::F32(values) => values.to_vec(),
            CellValues::U32(_) => panic!("expected float32"),
        }
    }

    #[test]
    fn normalize_total_scales_each_row_to_the_target() {
        let pool = CellBlockPool::new(2);
        let block = map_block(
            ElementwiseOp::NormalizeTotal {
                target_sum: Some(10.0),
            },
            &counts(),
            &pool,
        )
        .unwrap();
        assert_eq!(values(&block), [2.5, 7.5, 10.0]);
        assert_eq!(block.indptr(), counts().indptr());
        assert_eq!(block.indices(), counts().indices());
    }

    #[test]
    fn normalize_total_leaves_a_zero_total_cell_as_it_is() {
        let pool = CellBlockPool::new(2);
        let block = CellBlock::from_f32(vec![0, 2], vec![0, 1], vec![-1.0, 1.0], 2).unwrap();
        let output = map_block(
            ElementwiseOp::NormalizeTotal {
                target_sum: Some(10.0),
            },
            &block,
            &pool,
        )
        .unwrap();
        assert_eq!(values(&output), [-1.0, 1.0]);
    }

    #[test]
    fn normalize_total_divides_in_float32_as_scanpy() {
        // Totals 5 and 7: scanpy's divisors f32(5) / f32(1e4) and f32(7) / f32(1e4),
        // then value / divisor in f32 (not value * (1e4 / total) in f64: 1 of 5 counts
        // is 1999.9999 in scanpy, not 2000).
        let pool = CellBlockPool::new(2);
        let block = CellBlock::from_u32(vec![0, 2, 3], vec![0, 1, 2], vec![1, 4, 7], 3).unwrap();
        let output = map_block(
            ElementwiseOp::NormalizeTotal {
                target_sum: Some(1e4),
            },
            &block,
            &pool,
        )
        .unwrap();
        let first = 5.0_f32 / 1e4_f32;
        let second = 7.0_f32 / 1e4_f32;
        assert_eq!(values(&output), [1.0 / first, 4.0 / first, 7.0 / second]);
        assert_eq!(values(&output)[0], 1999.9999);
    }

    #[test]
    fn counts_above_2_pow_24_are_summed_as_float32_as_scanpy() {
        // The bug-hunter's row: scanpy casts to float32, then sums.
        let counts = [
            16_777_266_u32,
            16_777_226,
            16_777_221,
            16_777_267,
            16_777_217,
        ];
        let pool = CellBlockPool::new(2);
        let block = CellBlock::from_u32(vec![0, 5], (0..5).collect(), counts.to_vec(), 5).unwrap();
        let output = map_block(
            ElementwiseOp::NormalizeTotal {
                target_sum: Some(1e4),
            },
            &block,
            &pool,
        )
        .unwrap();
        // scanpy 1.12.5 on this uint32 row.
        assert_eq!(
            values(&output),
            [2000.0033, 1999.9985, 1999.9978, 2000.0035, 1999.9973]
        );
    }

    #[test]
    fn normalize_divisor_edges() {
        assert_eq!(normalize_divisor(0.0, Some(1e4)), 1.0);
        assert_eq!(normalize_divisor(5.0, None), 1.0);
        // A quotient that underflows to 0 leaves the cell as it is, as scanpy.
        assert_eq!(normalize_divisor(1e-30, Some(1e30)), 1.0);
        assert!(normalize_divisor(f64::NAN, Some(1e4)).is_nan());
        assert_eq!(normalize_divisor(-2.0, Some(4.0)), -0.5);
        // The total is rounded to f32 first (2^24 + 1 is not a float32).
        assert_eq!(normalize_divisor(16_777_217.0, Some(1.0)), 16_777_216.0);
    }

    #[test]
    fn log1p_is_kyto_log1p() {
        let pool = CellBlockPool::new(2);
        let block = map_block(ElementwiseOp::Log1p, &counts(), &pool).unwrap();
        assert_eq!(
            values(&block),
            [crate::log1p(1.0), crate::log1p(3.0), crate::log1p(2.0)]
        );
    }

    #[test]
    fn a_block_longer_than_the_zeros_is_mapped_whole() {
        let pool = CellBlockPool::new(2);
        let n = ZEROS.len() * 2 + 3;
        let block = CellBlock::from_u32(
            vec![0, n as u64],
            (0..n as u32).collect(),
            (0..n as u32).collect(),
            n as u64,
        )
        .unwrap();
        let output = map_block(ElementwiseOp::Log1p, &block, &pool).unwrap();
        let expected: Vec<f32> = (0..n).map(|value| crate::log1p(value as f32)).collect();
        assert_eq!(values(&output), expected);
    }
}
