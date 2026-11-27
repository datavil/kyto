//! `pca`'s passes on a GPU (task 9.8): the sparse x dense products through kyto-gpu's
//! CubeCL matrix kernels, the rest on the host as on the CPU.
//!
//! Does: per fold (a group of blocks, or one part of a split cell block), the group's
//! rows of the genes computed over as one CSR copy (`f32` values, positions among those
//! genes), then by pass:
//! - the sketch: `Omega`'s rows drawn on the host (as the CPU's, rounded to `f32`; its
//!   column sums from the rounded values, so the centring matches), and `X^T [Omega 1]`
//!   on the GPU (the sketch and, from the column of ones, the gene sums);
//! - a product: each row's centred projection `w = x Q - mu^T Q` on the GPU (`X Q`,
//!   `Q` as `f32` pairs and exact products, so the centring cancels without losing
//!   the rows' precision when means are large), rounded to `f32` and summed on the host
//!   in row order, then `X^T W` on the GPU;
//! - the last pass: each row's `(x - mu) V` on the GPU, and each gene's centred squares
//!   on the host in `f64` (the CPU's loop: the same bits).
//!
//! A nonzero value outside 1e-10 to 1e10 in magnitude is refused (`KY-E215`): its
//! products would leave `f32`'s normal range (the CPU runs it).
//! The GPU's sums of exact products are compensated `f32` pairs ([`PairSums`]) that a leaf
//! carries from fold to fold, so the parts 5.14 cuts a cell block into (which depend on
//! the threads) give the bits of the whole block: the same result on any thread count,
//! and the same bits on rerun (the kernels' fixed order, R11). Leaves become `f64` when
//! they combine, in the CPU's tree. The products are `f32` where the CPU's are `f64`,
//! so the result is within the PCA parity tolerances of the CPU's, not equal
//! (deviations.md).
//! Does not: the Rayleigh-Ritz step, the basis or the variances (`pca.rs`, `f64` on the
//! host: genes x `l`, small), or run without a GPU (`pca.rs` dispatches here only when
//! the step was placed on one).

use kyto_error::KytoError;
use kyto_gpu::{DeviceKernels, PairSums, SparseRows};
use kyto_store::{CellBlock, CellValues};

use super::{PcaAccumulator, PcaPartial, omega_row};
use crate::pca_dense::Dense;
use crate::pipeline_block::PipelineBlock;

/// The operation the GPU's errors name.
const OPERATION: &str = "pca";

/// A fold's rows of the genes computed over, in cell order, as the kernels read them.
struct GroupRows {
    indptr: Vec<u32>,
    /// Each value's position among the genes computed over.
    columns: Vec<u32>,
    values: Vec<f32>,
    /// The values as the CPU reads them (`f64`, a count exactly).
    exact: Vec<f64>,
    cells: Vec<u64>,
}

impl GroupRows {
    fn sparse(&self, n_columns: usize) -> SparseRows<'_> {
        SparseRows::new(&self.indptr, &self.columns, &self.values, n_columns)
    }

    fn n_rows(&self) -> usize {
        self.cells.len()
    }
}

/// `value` as `u32` offsets, or `KY-E505` (a block too large for the GPU's indices).
fn offset(gpu: &DeviceKernels, value: usize) -> Result<u32, KytoError> {
    u32::try_from(value).map_err(|_| KytoError::DeviceFailure {
        device: gpu.device().name().to_owned(),
        operation: OPERATION.to_owned(),
        problem: "a block holds more than 2^32 values; pass a smaller memory_limit".to_owned(),
    })
}

/// The rows of `group` (blocks of `pca`'s stream genes, checked by the caller) of the
/// genes `pca` computes over.
fn group_rows(
    pca: &PcaAccumulator,
    gpu: &DeviceKernels,
    group: &[(&CellBlock, &PipelineBlock)],
) -> Result<GroupRows, KytoError> {
    let mut rows = GroupRows {
        indptr: vec![0],
        columns: Vec::new(),
        values: Vec::new(),
        exact: Vec::new(),
        cells: Vec::new(),
    };
    let subset = pca.subset.as_deref();
    for &(layer, block) in group {
        let indptr = layer.indptr();
        let indices = layer.indices();
        for (row, bounds) in indptr.windows(2).enumerate() {
            for entry in bounds[0] as usize..bounds[1] as usize {
                let gene = indices[entry];
                let position = match subset {
                    Some(subset) => match subset[gene as usize] {
                        u32::MAX => continue,
                        position => position,
                    },
                    None => gene,
                };
                let exact = match layer.data() {
                    CellValues::F32(values) => f64::from(values[entry]),
                    CellValues::U32(values) => f64::from(values[entry]),
                };
                if exact.is_finite() && exact != 0.0 && !GPU_RANGE.contains(&exact.abs()) {
                    return Err(out_of_range(pca, exact));
                }
                rows.columns.push(position);
                rows.values.push(exact as f32);
                rows.exact.push(exact);
            }
            rows.indptr.push(offset(gpu, rows.values.len())?);
            rows.cells.push(block.cells[row]);
        }
    }
    Ok(rows)
}

/// The magnitudes of nonzero values the GPU's `f32` products keep exact: their
/// products with each other and with the basis stay inside `f32`'s normal range
/// (bug-hunter: values of 1e-20 underflowed to a silently wrong PCA, 1e30 overflowed).
const GPU_RANGE: std::ops::RangeInclusive<f64> = 1e-10..=1e10;

/// `KY-E215` for a value outside [`GPU_RANGE`].
fn out_of_range(pca: &PcaAccumulator, value: f64) -> KytoError {
    KytoError::InvalidArgument {
        function: "pca".to_owned(),
        parameter: "layer".to_owned(),
        value: format!("'{}'", pca.layer),
        problem: format!(
            "it holds {value:e}, outside the magnitudes pca's float32 products on a GPU \
             keep exact (1e-10 to 1e10)"
        ),
        accepted: "nonzero values from 1e-10 to 1e10 in magnitude on a GPU, or \
                   on(device='cpu') for this plan"
            .to_owned(),
    }
}

/// `values` rounded to `f32`.
fn rounded(values: &[f64]) -> Vec<f32> {
    values.iter().map(|&value| value as f32).collect()
}

/// [`PcaAccumulator::fold`] on `gpu`: `leaf` (a GPU leaf of the pass, from
/// [`PcaAccumulator::leaf`]) continued with the rows of `group`.
///
/// # Errors
/// `KY-E505` if a copy or a kernel fails, or a block passes the GPU's 2^32 indices;
/// `KY-E215` for a nonzero value outside 1e-10 to 1e10 in magnitude.
pub(super) fn fold(
    pca: &PcaAccumulator,
    gpu: &DeviceKernels,
    leaf: &mut PcaPartial,
    group: &[(&CellBlock, &PipelineBlock)],
) -> Result<(), KytoError> {
    let rows = group_rows(pca, gpu, group)?;
    let n_rows = rows.n_rows();
    let sparse = rows.sparse(pca.n_genes);
    match leaf {
        PcaPartial::GpuSketch {
            n_cells,
            omega_sums,
            sketch,
        } => {
            let width = pca.width;
            // [Omega 1]: each row's Omega (rounded to f32) and a 1 for the gene sums.
            let mut dense = vec![1.0_f32; n_rows * (width + 1)];
            let mut omega = vec![0.0; width];
            for (cell, row) in rows.cells.iter().zip(dense.chunks_mut(width + 1)) {
                omega_row(pca.seed, *cell, &mut omega);
                for ((target, sum), &value) in row.iter_mut().zip(omega_sums.iter_mut()).zip(&omega)
                {
                    *target = value as f32;
                    *sum += f64::from(*target);
                }
            }
            *n_cells += n_rows as u64;
            gpu.sparse_transposed_dense(&sparse, &dense, width + 1, sketch, OPERATION)
        }
        PcaPartial::GpuProduct { product, sums } => {
            let width = pca.block.columns;
            let starts: Vec<f64> = pca.block_means.iter().map(|mean| -mean).collect();
            let projected =
                gpu.sparse_dense(&sparse, &pca.block.values, width, &starts, OPERATION)?;
            let projected = rounded(&projected);
            for row in projected.chunks(width.max(1)).take(n_rows) {
                for (sum, &value) in sums.iter_mut().zip(row) {
                    *sum += f64::from(value);
                }
            }
            gpu.sparse_transposed_dense(&sparse, &projected, width, product, OPERATION)
        }
        PcaPartial::Rows {
            cells,
            rows: embedding,
            squares,
            stored,
        } => {
            let Some(loadings) = &pca.loadings else {
                unreachable!("the Rayleigh-Ritz step ends the product passes")
            };
            let width = pca.mean_projection.len();
            let starts: Vec<f64> = pca.mean_projection.iter().map(|mean| -mean).collect();
            let projected =
                gpu.sparse_dense(&sparse, &loadings.values, width, &starts, OPERATION)?;
            cells.extend_from_slice(&rows.cells);
            embedding.extend(projected);
            // Each gene's centred squares in row order, as the CPU adds them.
            for (&gene, &value) in rows.columns.iter().zip(&rows.exact) {
                let centred = value - pca.means[gene as usize];
                squares[gene as usize] += centred * centred;
                stored[gene as usize] += 1;
            }
            Ok(())
        }
        PcaPartial::Sketch { .. } | PcaPartial::Product { .. } => {
            unreachable!("a GPU pca's leaves come from PcaAccumulator::leaf")
        }
    }
}

/// An empty GPU leaf of the sketch pass: `n_genes x (width + 1)` pairs (the sketch and
/// the gene sums).
pub(super) fn sketch_leaf(n_genes: usize, width: usize) -> PcaPartial {
    PcaPartial::GpuSketch {
        n_cells: 0,
        omega_sums: vec![0.0; width],
        sketch: PairSums::zeros(n_genes * (width + 1)),
    }
}

/// An empty GPU leaf of a product pass over `block`.
pub(super) fn product_leaf(block: &Dense) -> PcaPartial {
    PcaPartial::GpuProduct {
        product: PairSums::zeros(block.rows * block.columns),
        sums: vec![0.0; block.columns],
    }
}

/// The CPU leaf a GPU leaf stands for: its pairs as `f64` (the sketch's last column
/// split off as the gene sums). Other leaves as they are.
pub(super) fn settle(partial: PcaPartial) -> PcaPartial {
    match partial {
        PcaPartial::GpuSketch {
            n_cells,
            omega_sums,
            sketch,
        } => {
            let width = omega_sums.len();
            let values = sketch.to_f64();
            let n_genes = values.len() / (width + 1);
            let mut sums = Vec::with_capacity(n_genes);
            let mut plain = Vec::with_capacity(n_genes * width);
            for row in values.chunks_exact(width + 1) {
                plain.extend_from_slice(&row[..width]);
                sums.push(row[width]);
            }
            PcaPartial::Sketch {
                n_cells,
                sums,
                omega_sums,
                sketch: plain,
            }
        }
        PcaPartial::GpuProduct { product, sums } => PcaPartial::Product {
            product: product.to_f64(),
            sums,
        },
        other => other,
    }
}
