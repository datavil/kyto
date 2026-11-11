//! The per-cell value-by-value maps of a pipeline: `normalize_total(target_sum=...)` and
//! `log1p`, one map or a fused pass of them (task 4.4's elementwise fusion).
//!
//! Does: apply the maps in order to each block, each writing a `float32` layer (its
//! logical dtype, so a fused pass gives the bits the same maps give one by one, task
//! 4.4 Open (2)), and drop the layers a fused pass does not keep. Output buffers come
//! from the [`CellBlockPool`]; replaced and dropped blocks go back to it. Scalar code:
//! `normalize_total` sums each row in `f64` and multiplies by `target_sum / total`
//! (a cell whose total is 0 stays as it is, as scanpy's `counts += counts == 0`),
//! `log1p` is `f32::ln_1p`; both keep zeros at zero, so a block keeps its structure.
//! Does not: SIMD (task 5.3), the median `normalize_total` (cross-cell, a statistics
//! pass first: 5.4), or scanpy's other parameters and its exact float steps (6.2 owns
//! parity).

use std::collections::BTreeSet;

use kyto_error::KytoError;
use kyto_plan::{ElementwiseOp, MapElementwise};
use kyto_store::{CellBlock, CellData, CellValues, ValueDtype};

use crate::cell_block_pool::CellBlockPool;
use crate::pipeline_block::PipelineBlock;

/// Maps run in one pass over each block, then the layers to drop.
#[derive(Debug, Clone)]
pub(crate) struct ElementwiseMaps {
    maps: Vec<MapElementwise>,
    drop: Vec<String>,
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
        Self { maps, drop }
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
            let output = map_block(map.op(), input, pool)?;
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
        data.reserve(input.indices().len());
        match input.data() {
            CellValues::F32(values) => map_rows(op, input.indptr(), values, |data_value| {
                data.push(data_value);
            }),
            CellValues::U32(values) => map_rows(op, input.indptr(), values, |data_value| {
                data.push(data_value);
            }),
        }
    }
    CellBlock::from_parts(parts, input.n_genes())
}

/// Call `write` with `op` of each value, row by row.
fn map_rows<T: Copy + Into<f64>>(
    op: ElementwiseOp,
    indptr: &[u64],
    values: &[T],
    mut write: impl FnMut(f32),
) {
    for row in indptr.windows(2) {
        // indptr comes from a checked CellBlock: increasing, ending at values.len().
        let row = &values[row[0] as usize..row[1] as usize];
        match op {
            ElementwiseOp::NormalizeTotal { target_sum } => {
                let total: f64 = row.iter().map(|&value| value.into()).sum();
                // A zero total leaves the cell as it is (scanpy: counts += counts == 0).
                // A median target never reaches here (cross-cell, rejected when lowered).
                let factor = match target_sum {
                    Some(target_sum) if total != 0.0 => target_sum / total,
                    _ => 1.0,
                };
                for &value in row {
                    write((value.into() * factor) as f32);
                }
            }
            ElementwiseOp::Log1p => {
                for &value in row {
                    // Rounded to f32 first: the input layer as stored (a u32 count is
                    // exact in f64; an f32 value is already f32).
                    write((value.into() as f32).ln_1p());
                }
            }
        }
    }
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
    fn log1p_is_f32_ln_1p() {
        let pool = CellBlockPool::new(2);
        let block = map_block(ElementwiseOp::Log1p, &counts(), &pool).unwrap();
        assert_eq!(
            values(&block),
            [1.0_f32.ln_1p(), 3.0_f32.ln_1p(), 2.0_f32.ln_1p()]
        );
    }
}
