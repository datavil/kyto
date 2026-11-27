//! The CubeCL matrix kernels of task 9.8, written once for every GPU runtime (wgpu on
//! Metal, Vulkan and DX12, and CUDA): PCA's sparse x dense products, and the tiled
//! distances and top-k selection of exact neighbours.
//!
//! Does: a block's CSR rows times a dense matrix ([`sparse_dense`], `X D`, one thread per
//! output value walking its row's entries in order) and the transposed product
//! ([`sparse_transposed_dense`], `X^T D`, one thread per output value walking its
//! column's entries in row order, on the host's CSC copy of the block), both summed as
//! compensated `f32` pairs of exact products (TwoProduct) that a later call continues ([`sparse_transposed_dense`]
//! reads its pairs back in, so a leaf split over many calls gives the bits of one
//! call); a tile of query rows against every point row ([`squared_distances`], and the
//! dense GEMM `A B^T` of [`dot_products`] with [`cosine_keys`] for the cosine
//! distance); and each query's `m` smallest keys ([`smallest`]). One thread per output
//! in a fixed order, no atomics and no cross-thread reduction: a device gives the same
//! bits run to run. Only `f32` and `u32` arithmetic (Metal has no `f64`).
//! Does not: move data or launch (`launch_matrix.rs`), or decide what the results mean
//! (kyto-exec's `pca_gpu.rs` and `neighbors_gpu.rs`, which keep the `f64` reductions
//! and check every neighbour on the host).
//!
//! As in `kernels.rs`, each TwoSum step is multiplied by a runtime `one` (1, passed at
//! launch, exact) so a shader compiler with fast math on (Metal's default) cannot
//! reassociate `(sum + term) - sum` into `term` and drop the error.

// The `#[cube]` macro expands each kernel into launch code clippy reads as ours, and
// takes comparisons, not `clamp`.
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_range_contains,
    clippy::manual_clamp
)]

use cubecl::prelude::*;

/// The rounding error of the `f32` product `product = a * b`: `a * b - product`
/// exactly (Dekker's TwoProduct without FMA, each half 12 bits so the partial products
/// are exact). Each step multiplied by the runtime `one` so fast math keeps it. Exact for
/// `|a|, |b|` below about `8e34` with no partial product below `f32`'s normal range.
#[cube]
fn product_error(a: f32, b: f32, product: f32, one: f32) -> f32 {
    let split = 4097.0f32; // 2^12 + 1
    let a_scaled = (a * split) * one;
    let a_high = (a_scaled - (a_scaled - a) * one) * one;
    let a_low = (a - a_high) * one;
    let b_scaled = (b * split) * one;
    let b_high = (b_scaled - (b_scaled - b) * one) * one;
    let b_low = (b - b_high) * one;
    (((a_high * b_high - product) * one + a_high * b_low) * one + a_low * b_high) * one
        + a_low * b_low
}

/// `X D` for a block's CSR rows (`indptr`, `columns`, `values`) and a dense row-major
/// `D` of `width` columns given as pairs (`dense_high + dense_low`): output value
/// `row * width + j` is the pair `start[j] + sum value * D[column, j]` over the row's
/// entries in order, as `high + low`. Each product `value * dense_high` is exact
/// ([`product_error`]) and `value * dense_low` is added to the error term; the sum is
/// Knuth's TwoSum, `low` collecting each addition's rounding error, started at the pair
/// `start_high[j] + start_low[j]`. So large values whose centring (`start`) cancels most
/// of the sum keep about twice `f32`'s precision.
#[cube(launch)]
pub fn sparse_dense(
    indptr: &[u32],
    columns: &[u32],
    values: &[f32],
    dense_high: &[f32],
    dense_low: &[f32],
    start_high: &[f32],
    start_low: &[f32],
    high: &mut [f32],
    low: &mut [f32],
    width: u32,
    one: f32,
) {
    let position = ABSOLUTE_POS;
    if position < high.len() {
        let width = width as usize;
        let row = position / width;
        let column = position % width;
        let mut sum = start_high[column];
        let mut error = start_low[column];
        for entry in indptr[row] as usize..indptr[row + 1] as usize {
            let value = values[entry];
            let at = columns[entry] as usize * width + column;
            let factor = dense_high[at];
            let term = value * factor;
            error += (product_error(value, factor, term, one) + value * dense_low[at]) * one;
            let total = (sum + term) * one;
            let virtual_term = (total - sum) * one;
            let virtual_sum = (total - virtual_term) * one;
            error += ((sum - virtual_sum) * one + (term - virtual_term) * one) * one;
            sum = total;
        }
        high[position] = sum;
        low[position] = error;
    }
}

/// `X^T D` added to the pairs `high + low`, for a block's rows in CSC (`column_ptr`
/// over the columns, each entry's `rows` and `values`, rows increasing within a
/// column) and a dense row-major `D` of `width` columns, one row per block row: output
/// value `column * width + j` continues its pair with `value * D[row, j]` for each
/// entry of the column, in row order (exact products and TwoSum, as [`sparse_dense`]).
#[cube(launch)]
pub fn sparse_transposed_dense(
    column_ptr: &[u32],
    rows: &[u32],
    values: &[f32],
    dense: &[f32],
    high: &mut [f32],
    low: &mut [f32],
    width: u32,
    one: f32,
) {
    let position = ABSOLUTE_POS;
    if position < high.len() {
        let width = width as usize;
        let column = position / width;
        let j = position % width;
        let mut sum = high[position];
        let mut error = low[position];
        for entry in column_ptr[column] as usize..column_ptr[column + 1] as usize {
            let value = values[entry];
            let factor = dense[rows[entry] as usize * width + j];
            let term = value * factor;
            error += product_error(value, factor, term, one);
            let total = (sum + term) * one;
            let virtual_term = (total - sum) * one;
            let virtual_sum = (total - virtual_term) * one;
            error += ((sum - virtual_sum) * one + (term - virtual_term) * one) * one;
            sum = total;
        }
        high[position] = sum;
        low[position] = error;
    }
}

/// The squared Euclidean distance of each query row to each point row (both row-major,
/// `width` values a row): output `query * n_points + point`, the squares of the
/// differences summed in component order in `f32`.
#[cube(launch)]
pub fn squared_distances(
    queries: &[f32],
    points: &[f32],
    out: &mut [f32],
    width: u32,
    n_points: u32,
) {
    let position = ABSOLUTE_POS;
    if position < out.len() {
        let width = width as usize;
        let n_points = n_points as usize;
        let query = (position / n_points) * width;
        let point = (position % n_points) * width;
        let mut sum = 0.0f32;
        for component in 0..width {
            let difference = queries[query + component] - points[point + component];
            sum += difference * difference;
        }
        out[position] = sum;
    }
}

/// The dense GEMM `A B^T`: the dot product of each query row with each point row
/// (row-major, `width` values a row), output `query * n_points + point`, summed in
/// component order in `f32`.
#[cube(launch)]
pub fn dot_products(queries: &[f32], points: &[f32], out: &mut [f32], width: u32, n_points: u32) {
    let position = ABSOLUTE_POS;
    if position < out.len() {
        let width = width as usize;
        let n_points = n_points as usize;
        let query = (position / n_points) * width;
        let point = (position % n_points) * width;
        let mut sum = 0.0f32;
        for component in 0..width {
            sum += queries[query + component] * points[point + component];
        }
        out[position] = sum;
    }
}

/// Dot products made cosine distances in place: `1 - dot / (|q| |p|)` clipped to
/// `[0, 2]`, and 1 when either norm is 0 (scikit-learn's `cosine_distances`, as the
/// CPU's).
#[cube(launch)]
pub fn cosine_keys(keys: &mut [f32], query_norms: &[f32], point_norms: &[f32], n_points: u32) {
    let position = ABSOLUTE_POS;
    if position < keys.len() {
        let n_points = n_points as usize;
        let query_norm = query_norms[position / n_points];
        let point_norm = point_norms[position % n_points];
        let mut key = 1.0f32;
        if query_norm != 0.0f32 && point_norm != 0.0f32 {
            key = 1.0f32 - keys[position] / (query_norm * point_norm);
            if key < 0.0f32 {
                key = 0.0f32;
            }
            if key > 2.0f32 {
                key = 2.0f32;
            }
        }
        keys[position] = key;
    }
}

/// Each query's `m` smallest keys among its row of `n_points` keys, the point
/// `skip[query]` left out (the query itself; `u32::MAX` for none), ranked by (key,
/// point): written to `out_keys` and `out_points` at `query * m`, smallest first. A
/// thread per query walks the points in order and keeps them by insertion, so a tie
/// goes to the earlier point.
#[cube(launch)]
pub fn smallest(
    keys: &[f32],
    skip: &[u32],
    out_keys: &mut [f32],
    out_points: &mut [u32],
    n_points: u32,
    m: u32,
) {
    let query = ABSOLUTE_POS;
    if query < skip.len() {
        let row = query * n_points as usize;
        let base = query * m as usize;
        let mut count = 0u32;
        for point in 0..n_points {
            let key = keys[row + point as usize];
            let mut keep = point != skip[query];
            if keep && count == m {
                // Full: only a smaller key enters (an equal one is a later point).
                keep = key < out_keys[base + (m - 1u32) as usize];
            }
            if keep {
                let mut at = count;
                if count < m {
                    count += 1u32;
                } else {
                    at = m - 1u32;
                }
                // Shift the larger keys up one place (the last one drops out).
                let mut moving = at > 0u32;
                while moving {
                    let before = out_keys[base + (at - 1u32) as usize];
                    if before > key {
                        out_keys[base + at as usize] = before;
                        out_points[base + at as usize] = out_points[base + (at - 1u32) as usize];
                        at -= 1u32;
                        moving = at > 0u32;
                    } else {
                        moving = false;
                    }
                }
                out_keys[base + at as usize] = key;
                out_points[base + at as usize] = point;
            }
        }
    }
}
