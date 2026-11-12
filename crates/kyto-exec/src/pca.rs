//! `pca` in the executor (task 6.5): a streamed randomized block Krylov method (Halko
//! et al. 2011, §4.3 and §4.6; Musco and Musco 2015) on the genes' side, with the
//! input centred implicitly, so a sparse layer is never densified.
//!
//! Does: with `X` the stream's cells x genes at the node (values as `f64`), `mu` the
//! gene means, `n` the cells and `C = (X - 1 mu^T)^T (X - 1 mu^T)` (never formed):
//! - pass 0: the gene sums and the sketch `X^T Omega`, where `Omega` has
//!   `l = min(n_components + `[`PCA_OVERSAMPLES`]`, n_genes)` columns of uniform values
//!   in `[-1, 1)` drawn per cell from the seed and the cell's table position (so
//!   independent of threads and blocks, R11, and of libm, no Gaussian draw); then the
//!   centred sketch `X^T Omega - mu (1^T Omega)` is orthonormalised into the first
//!   block `Q_0`.
//! - passes `1..=q` (`q` = [`PCA_POWER_ITERATIONS`]): `C Q_j = X^T W - mu (1^T W)` with
//!   each cell's centred projection `w = x Q_j - mu^T Q_j`, one sparse row at a time
//!   (centring each row first keeps large means accurate); each product (kept) grows the basis by its
//!   new directions (`pca_dense.rs`). After the last, or as soon as a block adds no
//!   direction (the basis spans the data; the passes left are skipped), the
//!   Rayleigh-Ritz step on the whole basis gives the eigenvalues `lambda` of `C` and
//!   the loadings `V`, with scikit-learn's signs.
//! - the last pass: each cell's row `(x - mu) V` (scanpy's `X_pca`, in cell order) and
//!   each gene's centred sum of squares (for the total variance).
//!
//! Variance `lambda / (n - 1)` and its ratio to the total variance (the genes'
//! variances with `ddof=1`, as scikit-learn's `explained_variance_ratio_`). Every
//! reduction folds through `reduction.rs`'s fixed tree (R11).
//! Does not: write the embeddings into the stream's blocks or a store (they are
//! statistics, [`PcaStatistics`]; terminals, 7.3), scanpy's other solvers, or
//! `zero_center=False`.

use std::sync::Arc;

use kyto_error::KytoError;
use kyto_plan::{PCA_OVERSAMPLES, PCA_POWER_ITERATIONS};
use kyto_store::{CellBlock, CellValues};

use crate::kernels::KernelValue;
use crate::pca_dense::{Dense, extend_basis, rayleigh_ritz};
use crate::pipeline_block::PipelineBlock;

/// What `pca`'s passes found (scanpy's `obsm['X_pca']`, `varm['PCs']` and
/// `uns['pca']`).
#[derive(Debug, Clone, PartialEq)]
pub struct PcaStatistics {
    n_components: usize,
    pub(crate) cells: Arc<Vec<u64>>,
    embedding: Arc<Vec<f32>>,
    loadings: Arc<Vec<f32>>,
    variance: Vec<f64>,
    variance_ratio: Vec<f64>,
    means: Arc<Vec<f64>>,
}

impl PcaStatistics {
    /// The number of components.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// The store table position of each embedding row: the cells reaching the node,
    /// in stream order.
    pub fn cells(&self) -> &[u64] {
        &self.cells
    }

    /// The cell embedding (scanpy's `X_pca`), row-major: `cells().len()` rows of
    /// `n_components()` values.
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    /// The loadings (scanpy's `varm['PCs']`), row-major: one row of `n_components()`
    /// values per gene of the stream at the node, in its order. Each component's
    /// largest absolute loading is positive (scikit-learn's `svd_flip`).
    pub fn loadings(&self) -> &[f32] {
        &self.loadings
    }

    /// Each component's variance, largest first (scanpy's `uns['pca']['variance']`).
    pub fn variance(&self) -> &[f64] {
        &self.variance
    }

    /// Each component's share of the total variance (scanpy's
    /// `uns['pca']['variance_ratio']`).
    pub fn variance_ratio(&self) -> &[f64] {
        &self.variance_ratio
    }

    /// Each gene's mean over the cells reaching the node (the implicit centring).
    pub fn means(&self) -> &[f64] {
        &self.means
    }
}

/// The passes `pca` runs: one sketch, `q` products, one projection.
pub(crate) const PCA_PASSES: u32 = 2 + PCA_POWER_ITERATIONS;

/// What one leaf of a `pca` pass sums (task 5.6).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PcaPartial {
    /// Pass 0: cells, gene sums, `Omega`'s column sums and the sketch (genes x `l`).
    Sketch {
        n_cells: u64,
        sums: Vec<f64>,
        omega_sums: Vec<f64>,
        sketch: Vec<f64>,
    },
    /// Passes `1..=q`: `X^T w` with each cell's centred row `w = (x - mu) Q`
    /// (genes x the block's width), and the sum of the `w`.
    Product { product: Vec<f64>, sums: Vec<f64> },
    /// The last pass: the cells and their rows of `(x - mu) V`, in cell order, each
    /// gene's sum of `(x - mu)^2` over its stored values and their count.
    Rows {
        cells: Vec<u64>,
        rows: Vec<f64>,
        squares: Vec<f64>,
        stored: Vec<u64>,
    },
}

impl PcaPartial {
    /// `self` (the cells before) and `right` (the cells after) as one partial.
    pub(crate) fn combine(self, right: Self) -> Self {
        match (self, right) {
            (
                Self::Sketch {
                    n_cells,
                    mut sums,
                    mut omega_sums,
                    mut sketch,
                },
                Self::Sketch {
                    n_cells: right_cells,
                    sums: right_sums,
                    omega_sums: right_omega,
                    sketch: right_sketch,
                },
            ) => {
                add(&mut sums, &right_sums);
                add(&mut omega_sums, &right_omega);
                add(&mut sketch, &right_sketch);
                Self::Sketch {
                    n_cells: n_cells + right_cells,
                    sums,
                    omega_sums,
                    sketch,
                }
            }
            (
                Self::Product {
                    mut product,
                    mut sums,
                },
                Self::Product {
                    product: right_product,
                    sums: right_sums,
                },
            ) => {
                add(&mut product, &right_product);
                add(&mut sums, &right_sums);
                Self::Product { product, sums }
            }
            (
                Self::Rows {
                    mut cells,
                    mut rows,
                    mut squares,
                    mut stored,
                },
                Self::Rows {
                    cells: right_cells,
                    rows: right_rows,
                    squares: right_squares,
                    stored: right_stored,
                },
            ) => {
                cells.extend(right_cells);
                rows.extend(right_rows);
                add(&mut squares, &right_squares);
                for (left, right) in stored.iter_mut().zip(right_stored) {
                    *left += right;
                }
                Self::Rows {
                    cells,
                    rows,
                    squares,
                    stored,
                }
            }
            _ => unreachable!("one pass's partials come from PcaAccumulator::leaf"),
        }
    }
}

fn add(left: &mut [f64], right: &[f64]) {
    for (left, right) in left.iter_mut().zip(right) {
        *left += right;
    }
}

/// The result of the passes so far.
#[derive(Debug, Clone)]
pub(crate) struct PcaAccumulator {
    layer: String,
    n_components: usize,
    seed: u64,
    n_genes: usize,
    /// Columns of `Omega` (`l`).
    width: usize,
    n_cells: u64,
    means: Vec<f64>,
    /// The basis's blocks `Q_0, Q_1, ...` (genes x their widths).
    basis: Dense,
    /// `C` times each basis column, in the basis's order.
    products: Dense,
    /// The block the next product pass multiplies.
    block: Dense,
    /// `mu^T` times the block (subtracted from each cell's `x Q`).
    block_means: Vec<f64>,
    /// Set by the Rayleigh-Ritz step: eigenvalues of `C` and the loadings.
    eigenvalues: Vec<f64>,
    loadings: Option<Dense>,
    /// `mu^T V`, subtracted from each cell's `x V`.
    mean_projection: Vec<f64>,
    cells: Vec<u64>,
    rows: Vec<f64>,
    total_variance: f64,
}

impl PcaAccumulator {
    /// An accumulator for `n_components` components of `layer`, over `n_genes` genes.
    pub(crate) fn new(layer: &str, n_components: u64, seed: u64, n_genes: usize) -> Self {
        let n_components = usize::try_from(n_components).unwrap_or(usize::MAX);
        let width = n_components
            .saturating_add(PCA_OVERSAMPLES as usize)
            .min(n_genes);
        Self {
            layer: layer.to_owned(),
            n_components,
            seed,
            n_genes,
            width,
            n_cells: 0,
            means: Vec::new(),
            basis: Dense::zeros(n_genes, 0),
            products: Dense::zeros(n_genes, 0),
            block: Dense::zeros(n_genes, 0),
            block_means: Vec::new(),
            eigenvalues: Vec::new(),
            loadings: None,
            mean_projection: Vec::new(),
            cells: Vec::new(),
            rows: Vec::new(),
            total_variance: f64::NAN,
        }
    }

    /// An empty leaf of pass `pass`.
    pub(crate) fn leaf(&self, pass: u32) -> PcaPartial {
        if pass == 0 {
            PcaPartial::Sketch {
                n_cells: 0,
                sums: vec![0.0; self.n_genes],
                omega_sums: vec![0.0; self.width],
                sketch: vec![0.0; self.n_genes * self.width],
            }
        } else if pass < PCA_PASSES - 1 {
            PcaPartial::Product {
                product: vec![0.0; self.n_genes * self.block.columns],
                sums: vec![0.0; self.block.columns],
            }
        } else {
            PcaPartial::Rows {
                cells: Vec::new(),
                rows: Vec::new(),
                squares: vec![0.0; self.n_genes],
                stored: vec![0; self.n_genes],
            }
        }
    }

    /// Whether pass `pass` has nothing to do: a product pass after the basis stopped
    /// growing.
    pub(crate) fn skips(&self, pass: u32) -> bool {
        pass > 0 && pass < PCA_PASSES - 1 && self.loadings.is_some()
    }

    /// Add `layer`'s rows (those of `block`, in cell order) to `leaf`, a leaf of one
    /// of the passes.
    ///
    /// # Errors
    /// `KY-E117` if the block's genes are not the accumulator's (a kyto bug, reported).
    pub(crate) fn fold(
        &self,
        leaf: &mut PcaPartial,
        layer: &CellBlock,
        block: &PipelineBlock,
    ) -> Result<(), KytoError> {
        if layer.n_genes() as usize != self.n_genes {
            return Err(KytoError::InvalidCellBlock {
                context: "pca statistics pass".to_owned(),
                problem: format!(
                    "a statistics pass over {} genes got a block of {} genes",
                    self.n_genes,
                    layer.n_genes()
                ),
            });
        }
        match layer.data() {
            CellValues::F32(values) => {
                self.fold_rows(leaf, layer.indptr(), layer.indices(), values, block)
            }
            CellValues::U32(values) => {
                self.fold_rows(leaf, layer.indptr(), layer.indices(), values, block)
            }
        }
        Ok(())
    }

    fn fold_rows<T: KernelValue>(
        &self,
        leaf: &mut PcaPartial,
        indptr: &[u64],
        indices: &[u32],
        values: &[T],
        block: &PipelineBlock,
    ) {
        let n_rows = indptr.len().saturating_sub(1);
        let row = |index: usize| {
            let start = indptr[index] as usize;
            let end = indptr[index + 1] as usize;
            indices[start..end]
                .iter()
                .zip(&values[start..end])
                .map(|(&gene, &value)| (gene as usize, value.to_f64()))
        };
        match leaf {
            PcaPartial::Sketch {
                n_cells,
                sums,
                omega_sums,
                sketch,
            } => {
                let mut omega = vec![0.0; self.width];
                for index in 0..n_rows {
                    omega_row(self.seed, block.cells[index], &mut omega);
                    *n_cells += 1;
                    add(omega_sums, &omega);
                    for (gene, value) in row(index) {
                        sums[gene] += value;
                        let target = &mut sketch[gene * self.width..(gene + 1) * self.width];
                        for (target, omega) in target.iter_mut().zip(&omega) {
                            *target += value * omega;
                        }
                    }
                }
            }
            PcaPartial::Product { product, sums } => {
                let width = self.block.columns;
                let mut projected = vec![0.0; width];
                for index in 0..n_rows {
                    // The centred row's projection: x Q - mu^T Q.
                    for (target, mean) in projected.iter_mut().zip(&self.block_means) {
                        *target = -mean;
                    }
                    for (gene, value) in row(index) {
                        for (target, basis) in projected.iter_mut().zip(self.block.row(gene)) {
                            *target += value * basis;
                        }
                    }
                    for (gene, value) in row(index) {
                        let target = &mut product[gene * width..(gene + 1) * width];
                        for (target, projected) in target.iter_mut().zip(&projected) {
                            *target += value * projected;
                        }
                    }
                    add(sums, &projected);
                }
            }
            PcaPartial::Rows {
                cells,
                rows,
                squares,
                stored,
            } => {
                let Some(loadings) = &self.loadings else {
                    unreachable!("the Rayleigh-Ritz step ends the product passes")
                };
                for index in 0..n_rows {
                    cells.push(block.cells[index]);
                    let start = rows.len();
                    rows.extend(self.mean_projection.iter().map(|value| -value));
                    let out = &mut rows[start..];
                    for (gene, value) in row(index) {
                        for (target, loading) in out.iter_mut().zip(loadings.row(gene)) {
                            *target += value * loading;
                        }
                        let centred = value - self.means[gene];
                        squares[gene] += centred * centred;
                        stored[gene] += 1;
                    }
                }
            }
        }
    }

    /// End pass `pass` with its reduction (`None` for a stream without cells).
    ///
    /// # Errors
    /// `KY-E215` if fewer than 2 cells or more components than cells or genes reach
    /// the node, or the layer holds a value that is not finite.
    pub(crate) fn end_pass(
        &mut self,
        pass: u32,
        reduced: Option<PcaPartial>,
    ) -> Result<(), KytoError> {
        match reduced {
            Some(PcaPartial::Sketch {
                n_cells,
                sums,
                omega_sums,
                mut sketch,
            }) => {
                self.check_cells(n_cells)?;
                self.n_cells = n_cells;
                let n = n_cells as f64;
                self.means = sums.iter().map(|sum| sum / n).collect();
                if let Some(gene) = self.means.iter().position(|mean| !mean.is_finite()) {
                    return Err(self.invalid(
                        format!("gene {gene} holds a value that is not finite (NaN or infinity)"),
                        "a layer of finite values",
                    ));
                }
                for (gene, mean) in self.means.iter().enumerate() {
                    let target = &mut sketch[gene * self.width..(gene + 1) * self.width];
                    for (target, omega_sum) in target.iter_mut().zip(&omega_sums) {
                        *target -= mean * omega_sum;
                    }
                }
                let sketch = Dense {
                    rows: self.n_genes,
                    columns: self.width,
                    values: sketch,
                };
                self.set_block(extend_basis(&self.basis, &sketch)?);
                if self.block.columns == 0 {
                    self.finish_basis()?;
                }
            }
            Some(PcaPartial::Product { product, sums }) => {
                let width = self.block.columns;
                let mut product = Dense {
                    rows: self.n_genes,
                    columns: width,
                    values: product,
                };
                // C Q = sum (x - mu)^T w = X^T W - mu (sum w), with w = (x - mu) Q per
                // cell: each row centred before the product, so large means cancel only
                // in the last bits (not `X^T X Q - n mu mu^T Q`, bug-hunter).
                for (gene, mean) in self.means.iter().enumerate() {
                    for (target, sum) in product.row_mut(gene).iter_mut().zip(&sums) {
                        *target -= mean * sum;
                    }
                }
                self.basis = self.basis.beside(&self.block);
                self.products = self.products.beside(&product);
                if pass == PCA_PASSES - 2 {
                    self.finish_basis()?;
                } else {
                    self.set_block(extend_basis(&self.basis, &product)?);
                    if self.block.columns == 0 {
                        self.finish_basis()?;
                    }
                }
            }
            Some(PcaPartial::Rows {
                cells,
                rows,
                squares,
                stored,
            }) => {
                let n = self.n_cells as f64;
                let sum: f64 = squares
                    .iter()
                    .zip(&stored)
                    .zip(&self.means)
                    .map(|((squares, stored), mean)| {
                        squares + (self.n_cells - stored) as f64 * mean * mean
                    })
                    .sum();
                self.total_variance = sum / (n - 1.0);
                self.cells = cells;
                self.rows = rows;
            }
            None => self.check_cells(0)?,
        }
        Ok(())
    }

    /// The node's statistics.
    pub(crate) fn finish(self) -> PcaStatistics {
        let Some(loadings) = self.loadings else {
            unreachable!("a run's last pass follows the Rayleigh-Ritz step")
        };
        let denominator = self.n_cells as f64 - 1.0;
        // Every cell equal (no variance at all): the eigenvalues are rounding, so 0, and
        // each ratio 0 / 0 is NaN as scikit-learn's (bug-hunter: was inf).
        let constant = self.total_variance == 0.0;
        let variance: Vec<f64> = self
            .eigenvalues
            .iter()
            .map(|value| if constant { 0.0 } else { value / denominator })
            .collect();
        let variance_ratio = variance
            .iter()
            .map(|value| value / self.total_variance)
            .collect();
        PcaStatistics {
            n_components: self.n_components,
            cells: Arc::new(self.cells),
            embedding: Arc::new(self.rows.iter().map(|&value| value as f32).collect()),
            loadings: Arc::new(loadings.values.iter().map(|&value| value as f32).collect()),
            variance,
            variance_ratio,
            means: Arc::new(self.means),
        }
    }

    /// The Rayleigh-Ritz step on the basis so far: eigenvalues, loadings, `mu^T V`.
    fn finish_basis(&mut self) -> Result<(), KytoError> {
        let (eigenvalues, loadings) =
            rayleigh_ritz(&self.basis, &self.products, self.n_components)?;
        let mut mean_projection = vec![0.0; self.n_components];
        for (gene, mean) in self.means.iter().enumerate() {
            for (target, loading) in mean_projection.iter_mut().zip(loadings.row(gene)) {
                *target += mean * loading;
            }
        }
        self.eigenvalues = eigenvalues;
        self.loadings = Some(loadings);
        self.mean_projection = mean_projection;
        self.set_block(Dense::zeros(self.n_genes, 0));
        Ok(())
    }

    /// Make `block` the one the next product pass multiplies.
    fn set_block(&mut self, block: Dense) {
        let mut block_means = vec![0.0; block.columns];
        for (gene, mean) in self.means.iter().enumerate() {
            for (target, value) in block_means.iter_mut().zip(block.row(gene)) {
                *target += mean * value;
            }
        }
        self.block = block;
        self.block_means = block_means;
    }

    fn check_cells(&self, n_cells: u64) -> Result<(), KytoError> {
        if n_cells < 2 {
            let reach = if n_cells == 1 {
                "1 cell reaches".to_owned()
            } else {
                format!("{n_cells} cells reach")
            };
            return Err(self.invalid(
                format!("{reach} pca (the filters before it keep fewer than 2)"),
                "a plan keeping at least 2 cells before pca",
            ));
        }
        let limit = (n_cells as usize).min(self.n_genes);
        if self.n_components > limit {
            return Err(KytoError::InvalidArgument {
                function: "pca".to_owned(),
                parameter: "n_components".to_owned(),
                value: self.n_components.to_string(),
                problem: format!(
                    "{n_cells} cells and {} genes reach pca, so it has at most {limit} components",
                    self.n_genes
                ),
                accepted: format!("an int from 1 to {limit}"),
            });
        }
        Ok(())
    }

    fn invalid(&self, problem: String, accepted: &str) -> KytoError {
        KytoError::InvalidArgument {
            function: "pca".to_owned(),
            parameter: "layer".to_owned(),
            value: format!("'{}'", self.layer),
            problem,
            accepted: accepted.to_owned(),
        }
    }
}

/// Fill `out` with the row of `Omega` for the cell at table position `cell`: uniform
/// values in `[-1, 1)` from SplitMix64 seeded by `seed` and `cell` (integer arithmetic
/// only, so the same values on every platform).
fn omega_row(seed: u64, cell: u64, out: &mut [f64]) {
    let mut state = mix(seed ^ mix(cell.wrapping_add(0x9E37_79B9_7F4A_7C15)));
    for value in out {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        // 53 random bits as a multiple of 2^-52 in [0, 2), shifted to [-1, 1).
        *value = (mix(state) >> 11) as f64 * f64::powi(2.0, -52) - 1.0;
    }
}

/// SplitMix64's output function.
pub(crate) fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omega_is_uniform_in_minus_one_to_one_and_depends_on_seed_and_cell() {
        let mut row = vec![0.0; 1000];
        omega_row(0, 7, &mut row);
        assert!(row.iter().all(|value| (-1.0..1.0).contains(value)));
        let mean = row.iter().sum::<f64>() / 1000.0;
        assert!(mean.abs() < 0.1);
        let mut other = vec![0.0; 1000];
        omega_row(0, 8, &mut other);
        assert_ne!(row, other);
        omega_row(1, 7, &mut other);
        assert_ne!(row, other);
        omega_row(0, 7, &mut other);
        assert_eq!(row, other);
    }
}
