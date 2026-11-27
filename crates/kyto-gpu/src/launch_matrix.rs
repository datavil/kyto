//! Running the 9.8 matrix kernels on a GPU: the host side of `kernels_matrix.rs`.
//!
//! Does: PCA's products of a block's CSR rows with a dense matrix,
//! [`DeviceKernels::sparse_dense`] (`X D`) and [`DeviceKernels::sparse_transposed_dense`]
//! (`X^T D`, added to compensated sums carried between calls, [`PairSums`]), and exact
//! neighbours' candidates, [`DeviceKernels::nearest_keys`] (tiles of queries against
//! every point: squared Euclidean distances or the GEMM's cosine distances, then each
//! query's `m` smallest keys on the device). One round trip per call. Each method's
//! numerics are stated on it: `f32` products summed as compensated `f32` pairs (about
//! `f32` products' precision, not the CPU's `f64` bits), `f32` keys.
//! Does not: decide what the results mean (kyto-exec's `pca_gpu.rs` and
//! `neighbors_gpu.rs`), or choose devices.
//!
//! **Hooks for 9.3 and 9.4.** These methods are the one entry point of PCA's products
//! and of neighbours' candidates on any GPU, CUDA included (CubeCL compiles the same
//! kernels for it). 9.3 makes cuSPARSE / cuBLAS the CUDA path by branching on
//! `self.device == Device::Cuda` inside [`DeviceKernels::sparse_dense`] and
//! [`DeviceKernels::sparse_transposed_dense`] (same arguments and results), and 9.4's
//! brute-force neighbours reuse [`DeviceKernels::nearest_keys`] (its keys go through
//! kyto-exec's host check, which makes the graph equal the CPU's exact search).

use kyto_error::KytoError;

use crate::launch::DeviceKernels;
#[cfg(any(feature = "wgpu", feature = "cuda"))]
use crate::launch::failure;

/// A block's rows in CSR over `n_columns` columns: `indptr` (from 0) cuts `columns`
/// (increasing within a row, below `n_columns`) and their `f32` values.
#[derive(Debug, Clone, Copy)]
pub struct SparseRows<'a> {
    indptr: &'a [u32],
    columns: &'a [u32],
    values: &'a [f32],
    n_columns: usize,
}

impl<'a> SparseRows<'a> {
    /// The rows `indptr` cuts from `columns` and `values`.
    ///
    /// # Panics
    /// If `indptr` is empty, does not start at 0, decreases or does not end at the
    /// number of values, `columns` and `values` differ in length, or a column is not
    /// below `n_columns` (a caller's bug, not user input).
    pub fn new(indptr: &'a [u32], columns: &'a [u32], values: &'a [f32], n_columns: usize) -> Self {
        assert!(
            indptr.first() == Some(&0)
                && indptr.windows(2).all(|pair| pair[0] <= pair[1])
                && indptr.last().map(|&end| end as usize) == Some(values.len()),
            "indptr cuts the values from 0"
        );
        assert_eq!(columns.len(), values.len(), "a column per value");
        assert!(
            columns.iter().all(|&column| (column as usize) < n_columns),
            "columns below n_columns"
        );
        Self {
            indptr,
            columns,
            values,
            n_columns,
        }
    }

    /// The number of rows.
    pub fn n_rows(&self) -> usize {
        self.indptr.len() - 1
    }

    /// The number of columns.
    pub fn n_columns(&self) -> usize {
        self.n_columns
    }

    /// Each value's column, row by row.
    pub fn columns(&self) -> &'a [u32] {
        self.columns
    }

    /// The number of stored values.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// Sums carried between calls as compensated `f32` pairs: each sum is `high + low`,
/// `low` the rounding errors TwoSum collected. Kept as pairs (not `f64`) so a sum
/// continued over several calls has the bits of one call (R11: a leaf split into
/// parts gives the bits of the whole block).
#[derive(Debug, Clone, PartialEq)]
pub struct PairSums {
    high: Vec<f32>,
    low: Vec<f32>,
}

impl PairSums {
    /// `len` sums of 0.
    pub fn zeros(len: usize) -> Self {
        Self {
            high: vec![0.0; len],
            low: vec![0.0; len],
        }
    }

    /// The number of sums.
    pub fn len(&self) -> usize {
        self.high.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.high.is_empty()
    }

    /// Each sum in `f64`: `high + low`, or `high` alone when either is not finite (an
    /// overflowed or NaN sum).
    pub fn to_f64(&self) -> Vec<f64> {
        self.high
            .iter()
            .zip(&self.low)
            .map(|(&high, &low)| pair_value(high, low))
            .collect()
    }
}

/// `high + low` in `f64`, or `high` when either is not finite.
fn pair_value(high: f32, low: f32) -> f64 {
    let (high, low) = (f64::from(high), f64::from(low));
    if high.is_finite() && low.is_finite() {
        high + low
    } else {
        high
    }
}

/// `value` as the pair `high + low` of `f32`s nearest it (`low` the rest of `high`).
#[cfg(any(feature = "wgpu", feature = "cuda"))]
fn split(value: f64) -> (f32, f32) {
    let high = value as f32;
    let low = if high.is_finite() {
        (value - f64::from(high)) as f32
    } else {
        0.0
    };
    (high, low)
}

/// What [`DeviceKernels::nearest_keys`] ranks by.
#[derive(Debug, Clone, Copy)]
pub enum PairwiseKey<'a> {
    /// The squared Euclidean distance, in `f32`.
    SquaredEuclidean,
    /// The cosine distance `1 - dot / (|q| |p|)` clipped to `[0, 2]` (1 for a zero
    /// row), from the GEMM's `f32` dot products and these norms.
    Cosine {
        /// Each query's norm.
        query_norms: &'a [f32],
        /// Each point's norm.
        point_norms: &'a [f32],
    },
}

/// The bytes of keys one tile of [`DeviceKernels::nearest_keys`] holds on the device.
pub const NEAREST_TILE_BYTES: usize = 64 << 20;

// Without a GPU runtime the methods' bodies are `match self.never {}`.
#[cfg_attr(not(any(feature = "wgpu", feature = "cuda")), allow(unused_variables))]
impl DeviceKernels {
    /// `X D`: for each row of `rows` and column `j` of the row-major `dense` (one row
    /// per column of `rows`, `width` values a row), `start[j] + sum value * dense[column,
    /// j]` over the row's values in order, row-major (`rows.n_rows() x width`). `dense`
    /// goes to the GPU as `f32` pairs, each product with it is exact to about twice
    /// `f32`'s precision and the sum with `start[j]` is a compensated `f32` pair, so a
    /// centring `start` that cancels most of the sum keeps that precision; non-finite
    /// sums come back as their `high` part.
    ///
    /// # Errors
    /// `KY-E505` if a copy or the kernel fails, or the output holds more than 2^32
    /// values; `operation` names the plan step.
    ///
    /// # Panics
    /// If `dense` is not `rows.n_columns() x width` or `start` not `width` long.
    pub fn sparse_dense(
        &self,
        rows: &SparseRows<'_>,
        dense: &[f64],
        width: usize,
        start: &[f64],
        operation: &str,
    ) -> Result<Vec<f64>, KytoError> {
        assert_eq!(
            dense.len(),
            rows.n_columns * width,
            "dense is n_columns x width"
        );
        assert_eq!(start.len(), width, "a start per column");
        #[cfg(any(feature = "wgpu", feature = "cuda"))]
        return run::sparse_dense(self, rows, dense, width, start, operation);
        #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
        match self.never {}
    }

    /// `X^T D` added to `sums` (`rows.n_columns() x width`, row-major): each column `c`
    /// of `rows` and column `j` of the row-major `dense` (one row per row of `rows`)
    /// continues the pair `sums[c * width + j]` with `value * dense[row, j]` for each
    /// value of column `c`, in row order: each product exact (TwoProduct), the sums
    /// compensated. So folding a block's rows in parts gives the bits of folding them at
    /// once.
    ///
    /// # Errors
    /// `KY-E505` if a copy or the kernel fails, or a size passes 2^32.
    ///
    /// # Panics
    /// If `dense` is not `rows.n_rows() x width` or `sums` not `rows.n_columns() x
    /// width`.
    pub fn sparse_transposed_dense(
        &self,
        rows: &SparseRows<'_>,
        dense: &[f32],
        width: usize,
        sums: &mut PairSums,
        operation: &str,
    ) -> Result<(), KytoError> {
        assert_eq!(
            dense.len(),
            rows.n_rows() * width,
            "dense is n_rows x width"
        );
        assert_eq!(
            sums.len(),
            rows.n_columns * width,
            "sums are n_columns x width"
        );
        #[cfg(any(feature = "wgpu", feature = "cuda"))]
        return run::sparse_transposed_dense(self, rows, dense, width, sums, operation);
        #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
        match self.never {}
    }

    /// Each query's `m` smallest keys against every point (`queries` and `points`
    /// row-major, `width` values a row), the point `skip[query]` left out (`u32::MAX`
    /// for none), ranked by (key, point): `(keys, points)`, `m` a query, smallest
    /// first. Keys in `f32` (`kernels_matrix.rs`), in tiles of queries whose keys fit
    /// [`NEAREST_TILE_BYTES`] on the device; only the `m` per query come back.
    ///
    /// # Errors
    /// `KY-E505` if a copy or a kernel fails, or there are 2^32 points or more.
    ///
    /// # Panics
    /// If `m` is 0 or above the points (less `skip`), `width` is 0 or does not divide
    /// both inputs, `skip` has not one entry per query, or cosine norms do not match.
    #[allow(clippy::too_many_arguments)]
    pub fn nearest_keys(
        &self,
        queries: &[f32],
        points: &[f32],
        width: usize,
        key: PairwiseKey<'_>,
        skip: &[u32],
        m: usize,
        operation: &str,
    ) -> Result<(Vec<f32>, Vec<u32>), KytoError> {
        assert!(
            width > 0 && queries.len().is_multiple_of(width) && points.len().is_multiple_of(width),
            "rows of width values"
        );
        let (n_queries, n_points) = (queries.len() / width, points.len() / width);
        assert_eq!(skip.len(), n_queries, "a skip per query");
        let skips_none = skip.iter().all(|&point| point == u32::MAX);
        assert!(
            m >= 1 && (m < n_points || (m == n_points && skips_none)),
            "1 <= m <= the points left after skip"
        );
        if let PairwiseKey::Cosine {
            query_norms,
            point_norms,
        } = key
        {
            assert_eq!(query_norms.len(), n_queries, "a norm per query");
            assert_eq!(point_norms.len(), n_points, "a norm per point");
        }
        #[cfg(any(feature = "wgpu", feature = "cuda"))]
        return run::nearest_keys(self, queries, points, width, key, skip, m, operation);
        #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
        match self.never {}
    }
}

#[cfg(any(feature = "wgpu", feature = "cuda"))]
mod run {
    //! The launches, only in a build with a GPU runtime.

    use cubecl::prelude::*;
    use cubecl::server::Handle;
    use kyto_error::KytoError;

    use super::{
        DeviceKernels, NEAREST_TILE_BYTES, PairSums, PairwiseKey, SparseRows, failure, pair_value,
        split,
    };
    use crate::kernels_matrix as cube;
    use crate::launch::run::{CUBE_THREADS, cubes};

    /// A launch argument over `handle`, `len` 4-byte elements.
    fn buffer(handle: &Handle, len: usize) -> BufferArg {
        // SAFETY: every handle passed here was allocated (or uploaded) by this module
        // for exactly `len` elements of the kernel argument's 4-byte type.
        unsafe { BufferArg::from_raw_parts(handle.clone(), len) }
    }

    /// `value` as `u32`, or `KY-E505` naming `what`.
    fn fits(
        gpu: &DeviceKernels,
        value: usize,
        what: &str,
        operation: &str,
    ) -> Result<u32, KytoError> {
        u32::try_from(value).map_err(|_| {
            failure(
                gpu.device,
                operation,
                format!("{what} pass 2^32 on the GPU; pass a smaller memory_limit"),
            )
        })
    }

    pub(super) fn sparse_dense(
        gpu: &DeviceKernels,
        rows: &SparseRows<'_>,
        dense: &[f64],
        width: usize,
        start: &[f64],
        operation: &str,
    ) -> Result<Vec<f64>, KytoError> {
        let n_rows = rows.n_rows();
        let starts: Vec<(f32, f32)> = start.iter().map(|&value| split(value)).collect();
        if n_rows == 0 || width == 0 {
            return Ok(Vec::new());
        }
        if rows.nnz() == 0 {
            // No products: each value is its start, as the kernel leaves it.
            let row: Vec<f64> = starts
                .iter()
                .map(|&(high, low)| pair_value(high, low))
                .collect();
            return Ok(row.repeat(n_rows));
        }
        let len = n_rows * width;
        let width_u32 = fits(gpu, width, "the dense columns", operation)?;
        fits(gpu, len, "the output values", operation)?;
        let (start_high, start_low): (Vec<f32>, Vec<f32>) = starts.into_iter().unzip();
        let (dense_high, dense_low): (Vec<f32>, Vec<f32>) =
            dense.iter().map(|&value| split(value)).unzip();
        let handles = [
            gpu.upload(rows.indptr),
            gpu.upload(rows.columns),
            gpu.upload(rows.values),
            gpu.upload(&dense_high),
            gpu.upload(&dense_low),
            gpu.upload(&start_high),
            gpu.upload(&start_low),
        ];
        let (high, low) = (gpu.output::<f32>(len), gpu.output::<f32>(len));
        cube::sparse_dense::launch(
            &gpu.client,
            cubes(len),
            CubeDim::new_1d(CUBE_THREADS),
            buffer(&handles[0], n_rows + 1),
            buffer(&handles[1], rows.nnz()),
            buffer(&handles[2], rows.nnz()),
            buffer(&handles[3], dense.len()),
            buffer(&handles[4], dense.len()),
            buffer(&handles[5], width),
            buffer(&handles[6], width),
            buffer(&high, len),
            buffer(&low, len),
            width_u32,
            1.0,
        );
        let high = gpu.download::<f32>(high, operation)?;
        let low = gpu.download::<f32>(low, operation)?;
        Ok(high
            .iter()
            .zip(&low)
            .map(|(&high, &low)| pair_value(high, low))
            .collect())
    }

    pub(super) fn sparse_transposed_dense(
        gpu: &DeviceKernels,
        rows: &SparseRows<'_>,
        dense: &[f32],
        width: usize,
        sums: &mut PairSums,
        operation: &str,
    ) -> Result<(), KytoError> {
        if rows.nnz() == 0 || width == 0 {
            return Ok(());
        }
        let n_columns = rows.n_columns;
        let len = n_columns * width;
        let width_u32 = fits(gpu, width, "the dense columns", operation)?;
        fits(gpu, len, "the output values", operation)?;
        fits(gpu, rows.n_rows(), "the rows", operation)?;
        // The rows in CSC: each column's entries in row order (a counting sort).
        let mut column_ptr = vec![0_u32; n_columns + 1];
        for &column in rows.columns {
            column_ptr[column as usize + 1] += 1;
        }
        for column in 0..n_columns {
            column_ptr[column + 1] += column_ptr[column];
        }
        let mut next: Vec<u32> = column_ptr[..n_columns].to_vec();
        let mut row_of = vec![0_u32; rows.nnz()];
        let mut values = vec![0.0_f32; rows.nnz()];
        for (row, bounds) in rows.indptr.windows(2).enumerate() {
            for entry in bounds[0] as usize..bounds[1] as usize {
                let column = rows.columns[entry] as usize;
                let at = next[column] as usize;
                next[column] += 1;
                row_of[at] = row as u32;
                values[at] = rows.values[entry];
            }
        }
        let handles = [
            gpu.upload(&column_ptr),
            gpu.upload(&row_of),
            gpu.upload(&values),
            gpu.upload(dense),
            gpu.upload(&sums.high),
            gpu.upload(&sums.low),
        ];
        cube::sparse_transposed_dense::launch(
            &gpu.client,
            cubes(len),
            CubeDim::new_1d(CUBE_THREADS),
            buffer(&handles[0], n_columns + 1),
            buffer(&handles[1], row_of.len()),
            buffer(&handles[2], values.len()),
            buffer(&handles[3], dense.len()),
            buffer(&handles[4], len),
            buffer(&handles[5], len),
            width_u32,
            1.0,
        );
        let [_, _, _, _, high, low] = handles;
        sums.high = gpu.download::<f32>(high, operation)?;
        sums.low = gpu.download::<f32>(low, operation)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn nearest_keys(
        gpu: &DeviceKernels,
        queries: &[f32],
        points: &[f32],
        width: usize,
        key: PairwiseKey<'_>,
        skip: &[u32],
        m: usize,
        operation: &str,
    ) -> Result<(Vec<f32>, Vec<u32>), KytoError> {
        let (n_queries, n_points) = (queries.len() / width, points.len() / width);
        let mut found_keys = Vec::with_capacity(n_queries * m);
        let mut found_points = Vec::with_capacity(n_queries * m);
        if n_queries == 0 {
            return Ok((found_keys, found_points));
        }
        let n_points_u32 = fits(gpu, n_points, "the points", operation)?;
        let width_u32 = fits(gpu, width, "the components", operation)?;
        let m_u32 = fits(gpu, m, "the candidates", operation)?;
        let tile = (NEAREST_TILE_BYTES / 4 / n_points).clamp(1, n_queries);
        let points_handle = gpu.upload(points);
        let point_norms = match key {
            PairwiseKey::Cosine { point_norms, .. } => Some(gpu.upload(point_norms)),
            PairwiseKey::SquaredEuclidean => None,
        };
        for first in (0..n_queries).step_by(tile) {
            let last = (first + tile).min(n_queries);
            let count = last - first;
            let len = count * n_points;
            let tile_queries = gpu.upload(&queries[first * width..last * width]);
            let keys = gpu.output::<f32>(len);
            match key {
                PairwiseKey::SquaredEuclidean => cube::squared_distances::launch(
                    &gpu.client,
                    cubes(len),
                    CubeDim::new_1d(CUBE_THREADS),
                    buffer(&tile_queries, count * width),
                    buffer(&points_handle, points.len()),
                    buffer(&keys, len),
                    width_u32,
                    n_points_u32,
                ),
                PairwiseKey::Cosine { query_norms, .. } => {
                    cube::dot_products::launch(
                        &gpu.client,
                        cubes(len),
                        CubeDim::new_1d(CUBE_THREADS),
                        buffer(&tile_queries, count * width),
                        buffer(&points_handle, points.len()),
                        buffer(&keys, len),
                        width_u32,
                        n_points_u32,
                    );
                    let norms = gpu.upload(&query_norms[first..last]);
                    let Some(point_norms) = &point_norms else {
                        unreachable!("uploaded for the cosine key")
                    };
                    cube::cosine_keys::launch(
                        &gpu.client,
                        cubes(len),
                        CubeDim::new_1d(CUBE_THREADS),
                        buffer(&keys, len),
                        buffer(&norms, count),
                        buffer(point_norms, n_points),
                        n_points_u32,
                    );
                }
            }
            let tile_skip = gpu.upload(&skip[first..last]);
            let (out_keys, out_points) =
                (gpu.output::<f32>(count * m), gpu.output::<u32>(count * m));
            cube::smallest::launch(
                &gpu.client,
                cubes(count),
                CubeDim::new_1d(CUBE_THREADS),
                buffer(&keys, len),
                buffer(&tile_skip, count),
                buffer(&out_keys, count * m),
                buffer(&out_points, count * m),
                n_points_u32,
                m_u32,
            );
            found_keys.extend(gpu.download::<f32>(out_keys, operation)?);
            found_points.extend(gpu.download::<u32>(out_points, operation)?);
        }
        Ok((found_keys, found_points))
    }
}
