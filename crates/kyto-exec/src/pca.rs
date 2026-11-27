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
//! Each fold splits a block's work over the pool's threads (task 5.15): the row-wise
//! steps (`Omega`'s rows, `w`, the embedding rows) over tiles of rows, each tile walking
//! the genes a range at a time so its rows of the dense matrix stay in cache; the
//! per-gene sums over disjoint ranges of genes. Every sum still adds its terms in cell
//! order and each row's in gene order, so the bits are those of folding row by row, on
//! any thread count.
//!
//! Variance `lambda / (n - 1)` and its ratio to the total variance (the genes'
//! variances with `ddof=1`, as scikit-learn's `explained_variance_ratio_`). Every
//! reduction folds through `reduction.rs`'s fixed tree (R11).
//! Over a gene set (`pca(genes=...)`, scanpy's `mask_var`, task 6.14) `X` is the
//! stream's cells x the genes of the set: every other gene's values are skipped as
//! each row is read, so the sums run in the order a `filter_genes` before the node
//! gives them (the same bits), and those genes' loadings are 0 (scanpy's
//! `varm['PCs']`).
//! Does not: write the embeddings into the stream's blocks or a store (they are
//! statistics, [`PcaStatistics`]; terminals, 7.3), scanpy's other solvers, or
//! `zero_center=False`.

mod gpu;

use std::sync::Arc;

use kyto_error::KytoError;
use kyto_gpu::{DeviceKernels, PairSums};
use kyto_plan::{PCA_OVERSAMPLES, PCA_POWER_ITERATIONS};
use kyto_store::{CellBlock, CellValues};
use rayon::prelude::*;

use crate::kernels::KernelValue;
use crate::pca_dense::{Dense, extend_basis, rayleigh_ritz};
use crate::pipeline_block::PipelineBlock;

/// What `pca`'s passes found (scanpy's `obsm['X_pca']`, `varm['PCs']` and
/// `uns['pca']`).
#[derive(Debug, Clone, PartialEq)]
pub struct PcaStatistics {
    n_components: usize,
    pub(crate) cells: Arc<Vec<u64>>,
    /// The store table position of each loadings row (set by the pipeline).
    pub(crate) genes: Arc<Vec<u64>>,
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

    /// The store table position of each loadings row: the genes of the stream at the
    /// node, in its order (task 5.11).
    pub fn genes(&self) -> &[u64] {
        &self.genes
    }

    /// The cell embedding (scanpy's `X_pca`), row-major: `cells().len()` rows of
    /// `n_components()` values.
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    /// The loadings (scanpy's `varm['PCs']`), row-major: one row of `n_components()`
    /// values per gene of the stream at the node, in its order (zeros for a gene
    /// outside the node's gene set). Each component's largest absolute loading is
    /// positive (scikit-learn's `svd_flip`).
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

    /// Each gene's mean over the cells reaching the node (the implicit centring): one
    /// per gene the components are computed over (the node's gene set, every gene of
    /// the stream without one), in the stream's order.
    pub fn means(&self) -> &[f64] {
        &self.means
    }
}

/// The passes `pca` runs: one sketch, `q` products, one projection.
pub(crate) const PCA_PASSES: u32 = 2 + PCA_POWER_ITERATIONS;

/// Genes per range whose sums one thread adds in a leaf's fold (task 5.15): the
/// range's sums stay in the core's cache while every row's values for it go by. Any
/// split gives the same bits.
const GENES_PER_PART: usize = 256;

/// Genes a tile of rows projects at a time (their rows of the matrix stay in cache).
const GENES_PER_TILE: usize = 128;

/// Rows per tile of a row-wise step (one parallel task).
const ROWS_PER_TILE: usize = 64;

/// Genes `start..end` (positions among the genes computed over) and the stream's genes
/// they are found among (`stream_start..stream_end`, genes outside a gene set
/// included).
#[derive(Debug, Clone, Copy)]
struct GenePart {
    start: usize,
    end: usize,
    stream_start: u32,
    stream_end: u32,
}

/// One block's rows of a pass's input layer, as `fold_rows` reads them.
struct BlockRows<'a, T> {
    indptr: &'a [u64],
    indices: &'a [u32],
    values: &'a [T],
    /// The rows' cells (table positions).
    cells: &'a [u64],
}

impl<'a, T> BlockRows<'a, T> {
    fn new(layer: &'a CellBlock, values: &'a [T], block: &'a PipelineBlock) -> Self {
        Self {
            indptr: layer.indptr(),
            indices: layer.indices(),
            values,
            cells: &block.cells,
        }
    }

    fn n_rows(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }
}

/// `values` (`per_gene` values per gene, genes in order) cut at `parts`' bounds.
fn split_genes<'a, V>(
    mut values: &'a mut [V],
    parts: &[GenePart],
    per_gene: usize,
) -> Vec<&'a mut [V]> {
    let mut split = Vec::with_capacity(parts.len());
    for part in parts {
        let (head, tail) = values.split_at_mut((part.end - part.start) * per_gene);
        split.push(head);
        values = tail;
    }
    split
}

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
    /// Pass 0 on a GPU (task 9.8, `pca/gpu.rs`): the sketch and the gene sums as
    /// compensated `f32` pairs a leaf's folds continue (genes x `l + 1`, the sums in
    /// the last column), with the cells and `Omega`'s column sums.
    GpuSketch {
        n_cells: u64,
        omega_sums: Vec<f64>,
        sketch: PairSums,
    },
    /// Passes `1..=q` on a GPU: `X^T w` as pairs, and the sum of the `w`.
    GpuProduct { product: PairSums, sums: Vec<f64> },
}

impl PcaPartial {
    /// `self` (the cells before) and `right` (the cells after) as one partial (a GPU
    /// leaf as the `f64` leaf it stands for).
    pub(crate) fn combine(self, right: Self) -> Self {
        match (gpu::settle(self), gpu::settle(right)) {
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
    /// The GPU the products run on (task 9.8, `pca/gpu.rs`), `None` on the CPU.
    gpu: Option<DeviceKernels>,
    n_components: usize,
    seed: u64,
    /// The genes the components are computed over (those of the gene set).
    n_genes: usize,
    /// The genes of the stream's blocks.
    stream_genes: usize,
    /// With a gene set: the position among its genes of each gene of the stream,
    /// `u32::MAX` for a gene outside it.
    subset: Option<Vec<u32>>,
    /// With a gene set: the stream gene of each of its genes, in order.
    set_genes: Option<Vec<u32>>,
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
    /// An accumulator for `n_components` components of `layer`, whose blocks hold
    /// `stream_genes` genes: over all of them, or over those flagged in `gene_set`
    /// (one flag per gene of the stream).
    pub(crate) fn new(
        layer: &str,
        n_components: u64,
        seed: u64,
        stream_genes: usize,
        gene_set: Option<&[bool]>,
        gpu: Option<DeviceKernels>,
    ) -> Self {
        let n_components = usize::try_from(n_components).unwrap_or(usize::MAX);
        let subset = gene_set.map(|flags| {
            let mut next = 0_u32;
            flags
                .iter()
                .map(|&flag| {
                    if flag {
                        next += 1;
                        next - 1
                    } else {
                        u32::MAX
                    }
                })
                .collect::<Vec<u32>>()
        });
        let set_genes = gene_set.map(|flags| {
            (0_u32..)
                .zip(flags)
                .filter(|&(_, &flag)| flag)
                .map(|(gene, _)| gene)
                .collect::<Vec<u32>>()
        });
        let n_genes = match gene_set {
            Some(flags) => flags.iter().filter(|&&flag| flag).count(),
            None => stream_genes,
        };
        let width = n_components
            .saturating_add(PCA_OVERSAMPLES as usize)
            .min(n_genes);
        Self {
            layer: layer.to_owned(),
            gpu,
            n_components,
            seed,
            n_genes,
            stream_genes,
            subset,
            set_genes,
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
        if self.gpu.is_some() && pass < PCA_PASSES - 1 {
            return if pass == 0 {
                gpu::sketch_leaf(self.n_genes, self.width)
            } else {
                gpu::product_leaf(&self.block)
            };
        }
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

    /// Add the rows of `group` (each block's input layer and the block, in cell order)
    /// to `leaf`, a leaf of one of the passes: as folding the blocks one by one, row by
    /// row, but with a whole run of a leaf's blocks split over the pool's threads
    /// (task 5.15).
    ///
    /// # Errors
    /// `KY-E117` if a block's genes are not the accumulator's (a kyto bug, reported).
    pub(crate) fn fold(
        &self,
        leaf: &mut PcaPartial,
        group: &[(&CellBlock, &PipelineBlock)],
    ) -> Result<(), KytoError> {
        let flags = self.subset.as_ref().map_or(self.stream_genes, Vec::len);
        for (layer, _) in group {
            if layer.n_genes() as usize != self.stream_genes || flags != self.stream_genes {
                return Err(KytoError::InvalidCellBlock {
                    context: "pca statistics pass".to_owned(),
                    problem: format!(
                        "a statistics pass over {} genes (a gene set over {flags}) got a block of {} genes",
                        self.stream_genes,
                        layer.n_genes()
                    ),
                });
            }
        }
        if let Some(kernels) = &self.gpu {
            return gpu::fold(self, kernels, leaf, group);
        }
        let f32_rows: Option<Vec<BlockRows<'_, f32>>> = group
            .iter()
            .map(|(layer, block)| match layer.data() {
                CellValues::F32(values) => Some(BlockRows::new(layer, values, block)),
                CellValues::U32(_) => None,
            })
            .collect();
        if let Some(rows) = f32_rows {
            self.fold_rows(leaf, &rows);
            return Ok(());
        }
        let u32_rows: Option<Vec<BlockRows<'_, u32>>> = group
            .iter()
            .map(|(layer, block)| match layer.data() {
                CellValues::U32(values) => Some(BlockRows::new(layer, values, block)),
                CellValues::F32(_) => None,
            })
            .collect();
        if let Some(rows) = u32_rows {
            self.fold_rows(leaf, &rows);
            return Ok(());
        }
        // Blocks of both value types (not from one stream): one at a time, in order.
        for &(layer, block) in group {
            self.fold(leaf, &[(layer, block)])?;
        }
        Ok(())
    }

    fn fold_rows<T: KernelValue>(&self, leaf: &mut PcaPartial, blocks: &[BlockRows<'_, T>]) {
        // Each row of the group: its block and its row there, in cell order.
        let rows_at: Vec<(usize, usize)> = blocks
            .iter()
            .enumerate()
            .flat_map(|(block, rows)| (0..rows.n_rows()).map(move |row| (block, row)))
            .collect();
        let n_rows = rows_at.len();
        let cell = |index: usize| {
            let (block, row) = rows_at[index];
            blocks[block].cells[row]
        };
        // Row `index`'s stored genes and values.
        let entries = |index: usize| {
            let (block, row) = rows_at[index];
            let rows = &blocks[block];
            let start = rows.indptr[row] as usize;
            let end = rows.indptr[row + 1] as usize;
            (&rows.indices[start..end], &rows.values[start..end])
        };
        let subset = self.subset.as_deref();
        let position = move |gene: u32| match subset {
            Some(subset) => match subset[gene as usize] {
                u32::MAX => None,
                position => Some(position as usize),
            },
            None => Some(gene as usize),
        };
        // Row `index`'s stored values of the genes of `part`, by position from the
        // part's start (genes outside a gene set skipped) (a
        // row's genes are strictly increasing, so they are one run of the row).
        let part_row =
            |part: GenePart, index: usize| {
                let (genes, values) = entries(index);
                let low = genes.partition_point(|&gene| gene < part.stream_start);
                let high = low + genes[low..].partition_point(|&gene| gene < part.stream_end);
                genes[low..high].iter().zip(&values[low..high]).filter_map(
                    move |(&gene, &value)| Some((position(gene)? - part.start, value.to_f64())),
                )
            };
        // Each gene sums its rows in cell order whatever the split (task 5.15): the
        // row-wise steps run in parallel over tiles of rows, each gene's sums over
        // disjoint ranges of genes, so the bits are those of folding the rows one by one.
        let parts = self.gene_parts(GENES_PER_PART);
        let tile_parts = self.gene_parts(GENES_PER_TILE);
        // Each row's `offsets`-subtracted projection on `matrix` (genes x `offsets.len()`)
        // into `out`: `-offsets + sum value * matrix[gene]`, its values added in the
        // row's gene order. A tile of rows walks the genes a few at a time, each row
        // resuming where it stopped, so a tile's rows and genes stay in the core's cache.
        let project = |matrix: &Dense, offsets: &[f64], out: &mut [f64]| {
            let width = offsets.len();
            if width == 0 {
                return;
            }
            out.par_chunks_mut(ROWS_PER_TILE * width)
                .enumerate()
                .for_each(|(tile, out)| {
                    let first = tile * ROWS_PER_TILE;
                    for row in out.chunks_mut(width) {
                        for (target, offset) in row.iter_mut().zip(offsets) {
                            *target = -offset;
                        }
                    }
                    // Each row's values not yet projected.
                    let mut rest: Vec<_> =
                        (first..first + out.len() / width).map(&entries).collect();
                    for part in &tile_parts {
                        for (row, (genes, values)) in out.chunks_mut(width).zip(&mut rest) {
                            let mut cursor = 0;
                            while let Some(&gene) = genes.get(cursor) {
                                if gene >= part.stream_end {
                                    break;
                                }
                                if let Some(position) = position(gene) {
                                    let value = values[cursor].to_f64();
                                    for (target, factor) in row.iter_mut().zip(matrix.row(position))
                                    {
                                        *target += value * factor;
                                    }
                                }
                                cursor += 1;
                            }
                            *genes = &genes[cursor..];
                            *values = &values[cursor..];
                        }
                    }
                });
        };
        match leaf {
            PcaPartial::Sketch {
                n_cells,
                sums,
                omega_sums,
                sketch,
            } => {
                let width = self.width;
                let mut omega = vec![0.0; n_rows * width];
                omega
                    .par_chunks_mut(width.max(1))
                    .with_min_len(ROWS_PER_TILE)
                    .enumerate()
                    .for_each(|(index, omega)| omega_row(self.seed, cell(index), omega));
                // Counted by rows, not by `Omega`'s rows (none at width 0, bug-hunter).
                *n_cells += n_rows as u64;
                for omega in omega.chunks(width.max(1)).take(n_rows) {
                    add(omega_sums, omega);
                }
                parts
                    .par_iter()
                    .zip(split_genes(sums, &parts, 1))
                    .zip(split_genes(sketch, &parts, width))
                    .for_each(|((part, sums), sketch)| {
                        for index in 0..n_rows {
                            let omega = &omega[index * width..(index + 1) * width];
                            for (gene, value) in part_row(*part, index) {
                                sums[gene] += value;
                                let target = &mut sketch[gene * width..(gene + 1) * width];
                                for (target, omega) in target.iter_mut().zip(omega) {
                                    *target += value * omega;
                                }
                            }
                        }
                    });
            }
            PcaPartial::Product { product, sums } => {
                let width = self.block.columns;
                // Each centred row's projection: x Q - mu^T Q.
                let mut projected = vec![0.0; n_rows * width];
                project(&self.block, &self.block_means, &mut projected);
                for projected in projected.chunks(width.max(1)).take(n_rows) {
                    add(sums, projected);
                }
                parts
                    .par_iter()
                    .zip(split_genes(product, &parts, width))
                    .for_each(|(part, product)| {
                        for index in 0..n_rows {
                            let projected = &projected[index * width..(index + 1) * width];
                            for (gene, value) in part_row(*part, index) {
                                let target = &mut product[gene * width..(gene + 1) * width];
                                for (target, projected) in target.iter_mut().zip(projected) {
                                    *target += value * projected;
                                }
                            }
                        }
                    });
            }
            PcaPartial::GpuSketch { .. } | PcaPartial::GpuProduct { .. } => {
                unreachable!("a GPU leaf is folded by pca/gpu.rs")
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
                let width = self.mean_projection.len();
                cells.extend((0..n_rows).map(cell));
                let start = rows.len();
                rows.resize(start + n_rows * width, 0.0);
                project(loadings, &self.mean_projection, &mut rows[start..]);
                parts
                    .par_iter()
                    .zip(split_genes(squares, &parts, 1))
                    .zip(split_genes(stored, &parts, 1))
                    .for_each(|((part, squares), stored)| {
                        let means = &self.means[part.start..part.end];
                        for index in 0..n_rows {
                            for (gene, value) in part_row(*part, index) {
                                let centred = value - means[gene];
                                squares[gene] += centred * centred;
                                stored[gene] += 1;
                            }
                        }
                    });
            }
        }
    }

    /// The genes split into ranges of `size` genes (the last one shorter).
    fn gene_parts(&self, size: usize) -> Vec<GenePart> {
        // The stream gene a position starts at (the stream's end past the last).
        let stream_start = |position: usize| -> u32 {
            let stream = match &self.set_genes {
                Some(set_genes) => set_genes
                    .get(position)
                    .map_or(self.stream_genes, |&gene| gene as usize),
                None => position.min(self.stream_genes),
            };
            u32::try_from(stream).unwrap_or(u32::MAX)
        };
        (0..self.n_genes.div_ceil(size))
            .map(|part| {
                let start = part * size;
                let end = (start + size).min(self.n_genes);
                GenePart {
                    start,
                    end,
                    stream_start: stream_start(start),
                    stream_end: stream_start(end),
                }
            })
            .collect()
    }

    /// End pass `pass` with its reduction (`None` for a stream without cells).
    ///
    /// # Errors
    /// `KY-E215` if fewer than 2 cells or more components than cells or genes (those
    /// of the gene set) reach the node, or the layer holds a value that is not finite.
    pub(crate) fn end_pass(
        &mut self,
        pass: u32,
        reduced: Option<PcaPartial>,
    ) -> Result<(), KytoError> {
        match reduced.map(gpu::settle) {
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
            Some(PcaPartial::GpuSketch { .. } | PcaPartial::GpuProduct { .. }) => {
                unreachable!("settled above")
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
        // One row per gene of the stream: zeros outside the gene set.
        let loadings: Vec<f32> = match &self.subset {
            Some(subset) => subset
                .iter()
                .flat_map(|&position| {
                    (0..self.n_components).map(move |component| (position, component))
                })
                .map(|(position, component)| match position {
                    u32::MAX => 0.0,
                    position => loadings.row(position as usize)[component] as f32,
                })
                .collect(),
            None => loadings.values.iter().map(|&value| value as f32).collect(),
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
            genes: Arc::new(Vec::new()),
            embedding: Arc::new(self.rows.iter().map(|&value| value as f32).collect()),
            loadings: Arc::new(loadings),
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
        if self.subset.is_some() && self.n_genes == 0 {
            // An empty gene set (bug-hunter: the range "1 to 0" said nothing).
            return Err(KytoError::InvalidArgument {
                function: "pca".to_owned(),
                parameter: "genes".to_owned(),
                value: "a gene set without genes".to_owned(),
                problem: "the predicate is true for none of the genes reaching pca".to_owned(),
                accepted: format!(
                    "a predicate true for at least {} genes (n_components)",
                    self.n_components
                ),
            });
        }
        let limit = (n_cells as usize).min(self.n_genes);
        if self.n_components > limit {
            return Err(KytoError::InvalidArgument {
                function: "pca".to_owned(),
                parameter: "n_components".to_owned(),
                value: self.n_components.to_string(),
                problem: format!(
                    "{n_cells} cells and {} genes{} reach pca, so it has at most {limit} components",
                    self.n_genes,
                    if self.subset.is_some() {
                        " (those of its gene set, `genes`)"
                    } else {
                        ""
                    }
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

    /// Stream genes of the split-fold tests: several of `GENES_PER_PART` and
    /// `GENES_PER_TILE`, so a fold crosses parts and tiles.
    const GENES: usize = 700;

    /// Seeded sparse rows (strictly increasing genes) with values over many orders of
    /// magnitude, so any change in the order of additions shows in the bits.
    fn sparse_rows(n_rows: usize, seed: u64) -> Vec<(Vec<u32>, Vec<f32>)> {
        let mut state = seed;
        let mut next = || {
            state = mix(state.wrapping_add(0x9E37_79B9_7F4A_7C15));
            state
        };
        (0..n_rows)
            .map(|_| {
                let (mut genes, mut values) = (Vec::new(), Vec::new());
                for gene in 0..GENES as u32 {
                    if next() % 7 == 0 {
                        genes.push(gene);
                        let exponent = (next() % 9) as i32 - 4;
                        values.push((next() % 1000) as f32 / 997.0 * 10_f32.powi(exponent));
                    }
                }
                (genes, values)
            })
            .collect()
    }

    /// A block of `rows` over cells starting at `first`.
    fn block(rows: &[(Vec<u32>, Vec<f32>)], first: u64) -> (CellBlock, PipelineBlock) {
        let mut parts = kyto_store::CellBlockParts::new(kyto_store::ValueDtype::F32);
        parts.indptr.push(0);
        for (genes, values) in rows {
            parts.indices.extend_from_slice(genes);
            if let kyto_store::CellData::F32(data) = &mut parts.data {
                data.extend_from_slice(values);
            }
            parts.indptr.push(parts.indices.len() as u64);
        }
        let layer = CellBlock::from_parts(parts, GENES as u64).unwrap();
        let pipeline = PipelineBlock {
            cell_block: 0,
            cells: (first..first + rows.len() as u64).collect(),
            layers: std::collections::BTreeMap::new(),
            columns: polars::prelude::DataFrame::empty_with_height(rows.len()),
        };
        (layer, pipeline)
    }

    /// The fold before task 5.15: one row at a time, each row's values in gene order.
    fn reference_fold(pca: &PcaAccumulator, leaf: &mut PcaPartial, rows: &[(u64, &[u32], &[f32])]) {
        let row = |genes: &[u32], values: &[f32]| -> Vec<(usize, f64)> {
            genes
                .iter()
                .zip(values)
                .filter_map(|(&gene, &value)| {
                    let position = match &pca.subset {
                        Some(subset) => match subset[gene as usize] {
                            u32::MAX => return None,
                            position => position as usize,
                        },
                        None => gene as usize,
                    };
                    Some((position, f64::from(value)))
                })
                .collect()
        };
        match leaf {
            PcaPartial::Sketch {
                n_cells,
                sums,
                omega_sums,
                sketch,
            } => {
                let mut omega = vec![0.0; pca.width];
                for &(cell, genes, values) in rows {
                    omega_row(pca.seed, cell, &mut omega);
                    *n_cells += 1;
                    add(omega_sums, &omega);
                    for (gene, value) in row(genes, values) {
                        sums[gene] += value;
                        let target = &mut sketch[gene * pca.width..(gene + 1) * pca.width];
                        for (target, omega) in target.iter_mut().zip(&omega) {
                            *target += value * omega;
                        }
                    }
                }
            }
            PcaPartial::Product { product, sums } => {
                let width = pca.block.columns;
                for &(_, genes, values) in rows {
                    let mut projected: Vec<f64> =
                        pca.block_means.iter().map(|mean| -mean).collect();
                    let entries = row(genes, values);
                    for &(gene, value) in &entries {
                        for (target, basis) in projected.iter_mut().zip(pca.block.row(gene)) {
                            *target += value * basis;
                        }
                    }
                    for &(gene, value) in &entries {
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
                rows: out,
                squares,
                stored,
            } => {
                let loadings = pca.loadings.as_ref().unwrap();
                for &(cell, genes, values) in rows {
                    cells.push(cell);
                    let start = out.len();
                    out.extend(pca.mean_projection.iter().map(|value| -value));
                    for (gene, value) in row(genes, values) {
                        for (target, loading) in out[start..].iter_mut().zip(loadings.row(gene)) {
                            *target += value * loading;
                        }
                        let centred = value - pca.means[gene];
                        squares[gene] += centred * centred;
                        stored[gene] += 1;
                    }
                }
            }
            PcaPartial::GpuSketch { .. } | PcaPartial::GpuProduct { .. } => {
                unreachable!("the reference folds CPU leaves")
            }
        }
    }

    /// Every pass kind folds a group of blocks with the bits of the row-by-row fold, on
    /// 1 and 4 threads, with and without a gene set, and one block at a time too.
    #[test]
    fn the_split_fold_gives_the_bits_of_folding_row_by_row() {
        let sizes = [150, 7, 203];
        let all: Vec<_> = sizes
            .iter()
            .enumerate()
            .map(|(index, &n)| sparse_rows(n, index as u64 + 1))
            .collect();
        let mut first = 0;
        let blocks: Vec<_> = all
            .iter()
            .map(|rows| {
                let built = block(rows, first);
                first += rows.len() as u64;
                built
            })
            .collect();
        let group: Vec<_> = blocks.iter().map(|(layer, block)| (layer, block)).collect();
        let reference_rows: Vec<(u64, &[u32], &[f32])> = all
            .iter()
            .flatten()
            .zip(0_u64..)
            .map(|((genes, values), cell)| (cell, genes.as_slice(), values.as_slice()))
            .collect();
        let flags: Vec<bool> = (0..GENES).map(|gene| gene % 3 != 1).collect();
        for gene_set in [None, Some(flags.as_slice())] {
            for threads in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                pool.install(|| {
                    let mut pca = PcaAccumulator::new("x", 5, 3, GENES, gene_set, None);
                    for pass in [0, 1, PCA_PASSES - 1] {
                        if pass == PCA_PASSES - 1 {
                            pca.finish_basis().unwrap();
                        }
                        let mut expected = pca.leaf(pass);
                        reference_fold(&pca, &mut expected, &reference_rows);
                        let mut grouped = pca.leaf(pass);
                        pca.fold(&mut grouped, &group).unwrap();
                        assert_eq!(grouped, expected, "pass {pass}, {threads} threads");
                        let mut one_by_one = pca.leaf(pass);
                        for pair in &group {
                            pca.fold(&mut one_by_one, std::slice::from_ref(pair))
                                .unwrap();
                        }
                        assert_eq!(
                            one_by_one, expected,
                            "pass {pass} by block, {threads} threads"
                        );
                        if pass < PCA_PASSES - 1 {
                            pca.end_pass(pass, Some(grouped)).unwrap();
                        }
                    }
                });
            }
        }
    }
}
