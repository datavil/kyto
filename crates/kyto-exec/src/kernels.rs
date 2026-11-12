//! SIMD kernels over the values of CSR rows: row sums, row sums of squares, row nnz,
//! row scaling, row division (task 6.2) and `log1p` (task 5.3); and over dense rows:
//! squared distances and dot products from one row to many (nearest neighbours, task
//! 6.6).
//!
//! Does: run each kernel through `pulp`'s runtime dispatch ([`Kernels::detected`]: AVX-512
//! with `x86-v4`, AVX2 + FMA with `x86-v3`, NEON on aarch64) or on pulp's scalar
//! fallback ([`Kernels::scalar`]). The kernel bodies are plain Rust compiled once per
//! instruction set inside the dispatched function, so the compiler vectorizes them with
//! that set's registers. Every kernel gives the same bits on every instruction set and
//! thread count (R11): Rust never reorders or contracts float operations (no fused
//! multiply-add), so the order written here is the order computed, and that order does
//! not depend on the vector width:
//! - sums run in [`SUM_LANES`] `f64` accumulators per row, value `j` of a row into
//!   accumulator `j % SUM_LANES`, then a fixed tree ([`combine_lanes`]); this is not the
//!   left-to-right sum, so an `f32` row may differ from it in the last bits (a `u32` row
//!   sums exactly in `f64` in any order, up to 2^53);
//! - a squared distance or dot product of two dense rows runs the same way: component
//!   `j`'s difference (or product) in `f64` into lane `j % SUM_LANES`, then the tree;
//! - `log1p` is kyto's own ([`log1p`]), computed in `f64` from the `f32` value and
//!   rounded once to `f32`, as `f32::ln_1p` (the platform's libm) cannot be vectorized
//!   and is not the same function on every platform.
//!
//! A NaN stays a NaN, but its payload bits are not promised.
//! Does not: per-gene (column) reductions (a scatter by gene index, task 5.4/6.x),
//! choose float steps for parity with scanpy (the callers do, task 6.2), or check CSR structure: the
//! callers pass a checked `CellBlock`'s `indptr` and values.

mod log1p;

pub use log1p::log1p;

use pulp::{Arch, Simd, WithSimd};

/// Accumulators per row of a sum; value `j` of a row goes into accumulator
/// `j % SUM_LANES`. Fixed, so a sum does not depend on the vector width (8 `f64` is one
/// AVX-512, two AVX2 or four NEON registers).
pub const SUM_LANES: usize = 8;

/// A value type the kernels read: a layer's `f32` values or `u32` counts.
pub trait KernelValue: Copy + Send + Sync + sealed::Sealed {
    /// The value as `f64` (exact for both types).
    fn to_f64(self) -> f64;
    /// The value as `f32` (a `u32` above 2^24 rounds to nearest, as `as f32`).
    fn to_f32(self) -> f32;
    /// Whether the value is zero (`-0.0` is zero, a NaN is not).
    fn is_zero(self) -> bool;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
}

impl KernelValue for f32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline(always)]
    fn is_zero(self) -> bool {
        self == 0.0
    }
}

impl KernelValue for u32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn is_zero(self) -> bool {
        self == 0
    }
}

/// The kernels on one instruction set. Cheap to copy; the output never depends on it.
#[derive(Debug, Clone, Copy)]
pub struct Kernels {
    arch: Arch,
}

impl Kernels {
    /// The best instruction set this CPU has (detected at run time).
    pub fn detected() -> Self {
        Self { arch: Arch::new() }
    }

    /// pulp's scalar fallback, the reference the dispatched kernels must equal. It is
    /// compiled for the target's baseline, so the compiler may still vectorize it with
    /// the baseline set (SSE2 on x86-64, NEON on aarch64).
    pub fn scalar() -> Self {
        Self { arch: Arch::Scalar }
    }

    /// The instruction set the kernels run on: `"avx512"`, `"avx2"`, `"neon"` or
    /// `"scalar"`.
    pub fn instruction_set(&self) -> &'static str {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        match self.arch {
            Arch::V4(_) => "avx512",
            Arch::V3(_) => "avx2",
            _ => "scalar",
        }
        #[cfg(target_arch = "aarch64")]
        match self.arch {
            Arch::Neon(_) => "neon",
            _ => "scalar",
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        "scalar"
    }

    /// The sum of each row's values in `f64`, written to `sums` (cleared first; one per
    /// row, 0 for an empty row). Summed in [`SUM_LANES`] lanes (module doc).
    ///
    /// # Panics
    /// If `indptr` is not a row pointer into `values` (a checked `CellBlock`'s are).
    pub fn row_sums<T: KernelValue>(&self, indptr: &[u64], values: &[T], sums: &mut Vec<f64>) {
        self.arch.dispatch(RowSums {
            indptr,
            values,
            out: sums,
            square: false,
        });
    }

    /// The sum of each row's squared values in `f64` (each value squared in `f64`),
    /// written to `sums` (cleared first). Summed in [`SUM_LANES`] lanes (module doc).
    ///
    /// # Panics
    /// If `indptr` is not a row pointer into `values`.
    pub fn row_sums_of_squares<T: KernelValue>(
        &self,
        indptr: &[u64],
        values: &[T],
        sums: &mut Vec<f64>,
    ) {
        self.arch.dispatch(RowSums {
            indptr,
            values,
            out: sums,
            square: true,
        });
    }

    /// The number of nonzero values of each row (a stored zero is not counted, a NaN is),
    /// written to `counts` (cleared first).
    ///
    /// # Panics
    /// If `indptr` is not a row pointer into `values`.
    pub fn row_nnz<T: KernelValue>(&self, indptr: &[u64], values: &[T], counts: &mut Vec<u64>) {
        self.arch.dispatch(RowNnz {
            indptr,
            values,
            out: counts,
        });
    }

    /// Each value times its row's factor, `(value as f64 * factor) as f32`, written to
    /// `out` (same length as `values`).
    ///
    /// # Panics
    /// If `indptr` is not a row pointer into `values`, `factors` has not one factor per
    /// row, or `out` is not as long as `values`.
    pub fn scale_rows<T: KernelValue>(
        &self,
        indptr: &[u64],
        values: &[T],
        factors: &[f64],
        out: &mut [f32],
    ) {
        assert_eq!(
            factors.len(),
            indptr.len().saturating_sub(1),
            "one factor per row"
        );
        assert_eq!(out.len(), values.len(), "out as long as values");
        self.arch.dispatch(ScaleRows {
            indptr,
            values,
            factors,
            out,
        });
    }

    /// Each value divided by its row's divisor in `f32`, `value as f32 / divisor`,
    /// written to `out` (same length as `values`): numpy's `float32` division, as
    /// scanpy's `normalize_total` divides (task 6.2). Not [`Kernels::scale_rows`] by
    /// the reciprocal, which rounds differently.
    ///
    /// # Panics
    /// If `indptr` is not a row pointer into `values`, `divisors` has not one divisor
    /// per row, or `out` is not as long as `values`.
    pub fn divide_rows<T: KernelValue>(
        &self,
        indptr: &[u64],
        values: &[T],
        divisors: &[f32],
        out: &mut [f32],
    ) {
        assert_eq!(
            divisors.len(),
            indptr.len().saturating_sub(1),
            "one divisor per row"
        );
        assert_eq!(out.len(), values.len(), "out as long as values");
        self.arch.dispatch(DivideRows {
            indptr,
            values,
            divisors,
            out,
        });
    }

    /// The squared Euclidean distance from `query` to each row of `points` (row-major,
    /// `query.len()` values a row), written to `out` (one per row): each difference and
    /// square in `f64`, summed in [`SUM_LANES`] lanes (module doc). An empty `query`
    /// gives 0 (task 6.6).
    ///
    /// # Panics
    /// If `points` is not `out.len()` rows of `query.len()` values.
    pub fn squared_distances(&self, query: &[f32], points: &[f32], out: &mut [f64]) {
        assert_eq!(
            points.len(),
            out.len().saturating_mul(query.len()),
            "points are out.len() rows of query.len() values"
        );
        self.arch.dispatch(RowPairs {
            query,
            points,
            out,
            dot: false,
        });
    }

    /// The dot product of `query` with each row of `points` (row-major, `query.len()`
    /// values a row), written to `out` (one per row): each product in `f64`, summed in
    /// [`SUM_LANES`] lanes (module doc). An empty `query` gives 0 (task 6.6).
    ///
    /// # Panics
    /// If `points` is not `out.len()` rows of `query.len()` values.
    pub fn dot_products(&self, query: &[f32], points: &[f32], out: &mut [f64]) {
        assert_eq!(
            points.len(),
            out.len().saturating_mul(query.len()),
            "points are out.len() rows of query.len() values"
        );
        self.arch.dispatch(RowPairs {
            query,
            points,
            out,
            dot: true,
        });
    }

    /// [`log1p`] of each value (a `u32` rounded to `f32` first), written to `out`.
    ///
    /// # Panics
    /// If `out` is not as long as `values`.
    pub fn log1p<T: KernelValue>(&self, values: &[T], out: &mut [f32]) {
        assert_eq!(out.len(), values.len(), "out as long as values");
        self.arch.dispatch(Log1p { values, out });
    }
}

/// The rows of `values` as `indptr` cuts them.
#[inline(always)]
fn rows<'a, T>(indptr: &'a [u64], values: &'a [T]) -> impl Iterator<Item = &'a [T]> + 'a {
    indptr
        .windows(2)
        .map(move |bounds| &values[bounds[0] as usize..bounds[1] as usize])
}

/// The fixed tree adding the lanes of a sum: `((a0 + a4) + (a2 + a6)) + ((a1 + a5) +
/// (a3 + a7))`, halving the lanes as a vector reduction does.
#[inline(always)]
pub fn combine_lanes(lanes: [f64; SUM_LANES]) -> f64 {
    let half = [
        lanes[0] + lanes[4],
        lanes[1] + lanes[5],
        lanes[2] + lanes[6],
        lanes[3] + lanes[7],
    ];
    let quarter = [half[0] + half[2], half[1] + half[3]];
    quarter[0] + quarter[1]
}

/// The lane sum of one row (of squares if `square`).
#[inline(always)]
fn lane_sum<T: KernelValue>(row: &[T], square: bool) -> f64 {
    let mut lanes = [0.0_f64; SUM_LANES];
    let (chunks, tail) = row.as_chunks::<SUM_LANES>();
    if square {
        for chunk in chunks {
            for lane in 0..SUM_LANES {
                let value = chunk[lane].to_f64();
                lanes[lane] += value * value;
            }
        }
        for (lane, value) in tail.iter().enumerate() {
            let value = value.to_f64();
            lanes[lane] += value * value;
        }
    } else {
        for chunk in chunks {
            for lane in 0..SUM_LANES {
                lanes[lane] += chunk[lane].to_f64();
            }
        }
        for (lane, value) in tail.iter().enumerate() {
            lanes[lane] += value.to_f64();
        }
    }
    combine_lanes(lanes)
}

struct RowSums<'a, T> {
    indptr: &'a [u64],
    values: &'a [T],
    out: &'a mut Vec<f64>,
    square: bool,
}

impl<T: KernelValue> WithSimd for RowSums<'_, T> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, _simd: S) {
        let Self {
            indptr,
            values,
            out,
            square,
        } = self;
        out.clear();
        out.reserve(indptr.len().saturating_sub(1));
        if square {
            out.extend(rows(indptr, values).map(|row| lane_sum(row, true)));
        } else {
            out.extend(rows(indptr, values).map(|row| lane_sum(row, false)));
        }
    }
}

struct RowNnz<'a, T> {
    indptr: &'a [u64],
    values: &'a [T],
    out: &'a mut Vec<u64>,
}

impl<T: KernelValue> WithSimd for RowNnz<'_, T> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, _simd: S) {
        self.out.clear();
        self.out.reserve(self.indptr.len().saturating_sub(1));
        self.out.extend(rows(self.indptr, self.values).map(|row| {
            row.iter()
                .map(|&value| u64::from(!value.is_zero()))
                .sum::<u64>()
        }));
    }
}

struct ScaleRows<'a, T> {
    indptr: &'a [u64],
    values: &'a [T],
    factors: &'a [f64],
    out: &'a mut [f32],
}

impl<T: KernelValue> WithSimd for ScaleRows<'_, T> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, _simd: S) {
        for (bounds, &factor) in self.indptr.windows(2).zip(self.factors) {
            let range = bounds[0] as usize..bounds[1] as usize;
            let row = &self.values[range.clone()];
            for (out, &value) in self.out[range].iter_mut().zip(row) {
                *out = (value.to_f64() * factor) as f32;
            }
        }
    }
}

struct DivideRows<'a, T> {
    indptr: &'a [u64],
    values: &'a [T],
    divisors: &'a [f32],
    out: &'a mut [f32],
}

impl<T: KernelValue> WithSimd for DivideRows<'_, T> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, _simd: S) {
        for (bounds, &divisor) in self.indptr.windows(2).zip(self.divisors) {
            let range = bounds[0] as usize..bounds[1] as usize;
            let row = &self.values[range.clone()];
            for (out, &value) in self.out[range].iter_mut().zip(row) {
                *out = value.to_f32() / divisor;
            }
        }
    }
}

/// The lane sum of `query` against one row of the same length: of products if `dot`,
/// else of squared differences.
#[inline(always)]
fn pair_sum(query: &[f32], row: &[f32], dot: bool) -> f64 {
    let mut lanes = [0.0_f64; SUM_LANES];
    let (query_chunks, query_tail) = query.as_chunks::<SUM_LANES>();
    let (row_chunks, row_tail) = row.as_chunks::<SUM_LANES>();
    if dot {
        for (left, right) in query_chunks.iter().zip(row_chunks) {
            for lane in 0..SUM_LANES {
                lanes[lane] += f64::from(left[lane]) * f64::from(right[lane]);
            }
        }
        for (lane, (&left, &right)) in query_tail.iter().zip(row_tail).enumerate() {
            lanes[lane] += f64::from(left) * f64::from(right);
        }
    } else {
        for (left, right) in query_chunks.iter().zip(row_chunks) {
            for lane in 0..SUM_LANES {
                let difference = f64::from(left[lane]) - f64::from(right[lane]);
                lanes[lane] += difference * difference;
            }
        }
        for (lane, (&left, &right)) in query_tail.iter().zip(row_tail).enumerate() {
            let difference = f64::from(left) - f64::from(right);
            lanes[lane] += difference * difference;
        }
    }
    combine_lanes(lanes)
}

struct RowPairs<'a> {
    query: &'a [f32],
    points: &'a [f32],
    out: &'a mut [f64],
    dot: bool,
}

impl WithSimd for RowPairs<'_> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, _simd: S) {
        let Self {
            query,
            points,
            out,
            dot,
        } = self;
        if query.is_empty() {
            out.fill(0.0);
            return;
        }
        let rows = points.chunks_exact(query.len());
        if dot {
            for (out, row) in out.iter_mut().zip(rows) {
                *out = pair_sum(query, row, true);
            }
        } else {
            for (out, row) in out.iter_mut().zip(rows) {
                *out = pair_sum(query, row, false);
            }
        }
    }
}

struct Log1p<'a, T> {
    values: &'a [T],
    out: &'a mut [f32],
}

impl<T: KernelValue> WithSimd for Log1p<'_, T> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, _simd: S) {
        for (out, &value) in self.out.iter_mut().zip(self.values) {
            *out = log1p(value.to_f32());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_count_and_scale_rows() {
        // Rows: [1, 3] [] [2, 0, 4]
        let indptr = [0, 2, 2, 5];
        let values = [1_u32, 3, 2, 0, 4];
        for kernels in [Kernels::detected(), Kernels::scalar()] {
            let mut sums = Vec::new();
            kernels.row_sums(&indptr, &values, &mut sums);
            assert_eq!(sums, [4.0, 0.0, 6.0]);
            kernels.row_sums_of_squares(&indptr, &values, &mut sums);
            assert_eq!(sums, [10.0, 0.0, 20.0]);
            let mut counts = vec![7];
            kernels.row_nnz(&indptr, &values, &mut counts);
            assert_eq!(counts, [3 - 1, 0, 2]);
            let mut out = [0.0; 5];
            kernels.scale_rows(&indptr, &values, &[0.5, 9.0, 2.0], &mut out);
            assert_eq!(out, [0.5, 1.5, 4.0, 0.0, 8.0]);
        }
    }

    #[test]
    fn lane_sum_uses_the_fixed_tree() {
        // 1e16 + 1 loses the 1 in f64; the lanes decide which ones survive.
        let row = [1e16_f32, 1.0, 1.0, 1.0, -1e16, 1.0, 1.0, 1.0, 1.0];
        let mut lanes = [0.0; SUM_LANES];
        for (index, &value) in row.iter().enumerate() {
            lanes[index % SUM_LANES] += f64::from(value);
        }
        let mut sums = Vec::new();
        Kernels::detected().row_sums(&[0, row.len() as u64], &row, &mut sums);
        assert_eq!(sums, [combine_lanes(lanes)]);
    }

    #[test]
    fn instruction_set_is_named() {
        assert_eq!(Kernels::scalar().instruction_set(), "scalar");
        #[cfg(target_arch = "aarch64")]
        assert_eq!(Kernels::detected().instruction_set(), "neon");
    }
}
