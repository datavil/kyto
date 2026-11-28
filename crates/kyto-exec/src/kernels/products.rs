//! The `f32` dot products of a block of rows with a tile of rows, as one matrix product
//! (task 12.4): `out[i * m + j] = queries[i] · points[j]`.
//!
//! Does: on macOS, call Accelerate's `cblas_sgemm` (it runs on the Apple matrix unit);
//! elsewhere, `faer`'s `f32` matmul on the calling thread (`Par::Seq`). The summation
//! order, fused multiply-adds and blocking are the library's, so the bits are **not**
//! fixed across platforms, library versions or block shapes: callers use these values
//! only where a stated error bound makes the result independent of them (exact
//! neighbours filter candidates with them and rank with [`super::Kernels`]' fixed-order
//! `f64` keys). The bound any order meets: `|computed - exact| <= w u / (1 - w u)
//! sum |q_i p_i|` for `w` components, `u = 2^-24`, plus underflow.
//! Does not: thread the product (each call is one thread's work; callers parallelise
//! over blocks), or promise the same bits anywhere (above).

/// `out[i * m + j]` = the dot product of row `i` of `queries` with row `j` of `points`,
/// both row-major with `width` values a row, `m = points.len() / width`.
///
/// # Panics
/// If `width` is 0, either slice is not whole rows, `out` is not `rows x m` values, or a
/// dimension does not fit in an `i32` (Accelerate's integer).
pub fn f32_products(queries: &[f32], points: &[f32], width: usize, out: &mut [f32]) {
    assert!(width > 0, "rows have at least one value");
    assert_eq!(queries.len() % width, 0, "queries are whole rows");
    assert_eq!(points.len() % width, 0, "points are whole rows");
    let rows = queries.len() / width;
    let columns = points.len() / width;
    assert_eq!(out.len(), rows * columns, "out is rows x columns");
    if rows == 0 || columns == 0 {
        return;
    }
    products(queries, points, rows, columns, width, out);
}

#[cfg(target_os = "macos")]
fn products(
    queries: &[f32],
    points: &[f32],
    rows: usize,
    columns: usize,
    width: usize,
    out: &mut [f32],
) {
    const ROW_MAJOR: i32 = 101;
    const NO_TRANSPOSE: i32 = 111;
    const TRANSPOSE: i32 = 112;
    let largest = i32::MAX as usize;
    assert!(
        rows <= largest && columns <= largest && width <= largest,
        "dimensions fit in i32"
    );
    let (rows, columns, width) = (rows as i32, columns as i32, width as i32);
    // SAFETY: `cblas_sgemm` reads `rows x width` values of `queries` (lda = width) and
    // `columns x width` of `points` (ldb = width, transposed), and writes `rows x columns`
    // values of `out` (ldc = columns); `f32_products` checked those lengths, the slices
    // do not overlap (`out` is a unique borrow), and beta = 0 means `out` is not read.
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANSPOSE,
            TRANSPOSE,
            rows,
            columns,
            width,
            1.0,
            queries.as_ptr(),
            width,
            points.as_ptr(),
            width,
            0.0,
            out.as_mut_ptr(),
            columns,
        );
    }
}

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        transpose_a: i32,
        transpose_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[cfg(not(target_os = "macos"))]
fn products(
    queries: &[f32],
    points: &[f32],
    rows: usize,
    columns: usize,
    width: usize,
    out: &mut [f32],
) {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, MatMut, MatRef, Par};
    let left = MatRef::from_row_major_slice(queries, rows, width);
    let right = MatRef::from_row_major_slice(points, columns, width).transpose();
    let product = MatMut::from_row_major_slice_mut(out, rows, columns);
    matmul(product, Accum::Replace, left, right, 1.0, Par::Seq);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn products_of_small_rows() {
        let queries = [1.0, 2.0, 0.0, -1.0];
        let points = [3.0, 4.0, 1.0, 1.0, 0.5, -2.0];
        let mut out = [0.0; 6];
        f32_products(&queries, &points, 2, &mut out);
        assert_eq!(out, [11.0, 3.0, -3.5, -4.0, -1.0, 2.0]);
    }

    #[test]
    fn products_within_the_bound() {
        let width = 37;
        let value = |index: usize| ((index * 7919 % 1000) as f32 - 500.0) / 37.0;
        let queries: Vec<f32> = (0..5 * width).map(value).collect();
        let points: Vec<f32> = (0..9 * width).map(|i| value(i + 3)).collect();
        let mut out = vec![0.0; 45];
        f32_products(&queries, &points, width, &mut out);
        let unit = f64::from(f32::EPSILON) / 2.0;
        for (i, query) in queries.chunks_exact(width).enumerate() {
            for (j, point) in points.chunks_exact(width).enumerate() {
                let exact: f64 = query
                    .iter()
                    .zip(point)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let magnitude: f64 = query
                    .iter()
                    .zip(point)
                    .map(|(&a, &b)| (f64::from(a) * f64::from(b)).abs())
                    .sum();
                let error = (f64::from(out[i * 9 + j]) - exact).abs();
                assert!(error <= width as f64 * unit * magnitude * 1.01, "{i} {j}");
            }
        }
    }
}
