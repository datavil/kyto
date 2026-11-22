//! The CubeCL kernels of task 9.2, written once for every GPU runtime (CUDA, wgpu on
//! Metal, Vulkan and DX12): the 5.3 CPU kernels over the values of CSR rows.
//!
//! Does: per row, the sum of its values (compensated, [`row_sums_f32`], or exact for
//! counts, [`row_sums_u32`]) and its nonzero count ([`row_nnz`]); per value, the
//! division by its row's divisor ([`divide_rows`], `normalize_total`'s scaling) and
//! `log1p` ([`log1p`]). One thread per row or per value, each in a fixed order, with no
//! atomics and no cross-thread reduction, so a device gives the same bits run to run.
//! Only `f32` and `u32` arithmetic: Metal has no `f64`. So the bits are not the CPU's
//! (`f64` sums in 8 lanes, `f64` `log1p`): within 1e-6 relative of them (task 9.2's
//! check), and equal for counts below 2^24 (exact integer sums).
//! Does not: move data or launch (`launch.rs`), or say which plan steps use them
//! (`backend.rs`).

// The `#[cube]` macro expands each kernel into launch code clippy reads as ours, and
// takes comparisons, not `Range::contains`.
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_range_contains
)]

use cubecl::prelude::*;

/// The values `kernel` reads as `f32`: a layer's `f32` or a count's `u32` (rounded to
/// nearest, as the CPU's `as f32`).
#[cube]
fn value_f32<T: Numeric>(value: T) -> f32 {
    f32::cast_from(value)
}

/// Each row's sum of `f32` values as a compensated pair: `high[row] + low[row]` (taken
/// in `f64` by the host) is the sum with about twice `f32`'s precision. Knuth's
/// TwoSum, left to right: `low` collects each addition's exact rounding error. A row
/// with a non-finite value or sum leaves `low` non-finite: the host takes `high` then.
///
/// `one` is 1, passed at launch: each step of TwoSum is multiplied by it (exact), so a
/// shader compiler with fast math on (Metal's default) cannot reassociate
/// `(sum + value) - sum` into `value` and drop the error term, as it does otherwise
/// (measured on an M4: plain `f32` sums, 1.5e-6 off).
#[cube(launch)]
pub fn row_sums_f32<T: Numeric>(
    indptr: &[u32],
    values: &[T],
    high: &mut [f32],
    low: &mut [f32],
    one: f32,
) {
    let row = ABSOLUTE_POS;
    if row < high.len() {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let mut sum = 0.0f32;
        let mut error = 0.0f32;
        for position in start..end {
            let value = value_f32::<T>(values[position]);
            let total = (sum + value) * one;
            let virtual_value = (total - sum) * one;
            let virtual_sum = (total - virtual_value) * one;
            error += ((sum - virtual_sum) * one + (value - virtual_value) * one) * one;
            sum = total;
        }
        high[row] = sum;
        low[row] = error;
    }
}

/// Each row's exact sum of `u32` counts, as the low and high 32 bits of a `u64`
/// (`high * 2^32 + low`): integer adds with a carry, so it is exact in any order.
#[cube(launch)]
pub fn row_sums_u32(indptr: &[u32], values: &[u32], high: &mut [u32], low: &mut [u32]) {
    let row = ABSOLUTE_POS;
    if row < high.len() {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let mut sum_low = 0u32;
        let mut sum_high = 0u32;
        for position in start..end {
            let next = sum_low + values[position];
            if next < sum_low {
                sum_high += 1u32;
            }
            sum_low = next;
        }
        high[row] = sum_high;
        low[row] = sum_low;
    }
}

/// Each row's number of nonzero values (a stored zero, `-0.0` too, is not counted; a
/// NaN and a subnormal are), as the CPU counts them.
#[cube(launch)]
pub fn row_nnz<T: Numeric>(indptr: &[u32], values: &[T], counts: &mut [u32]) {
    let row = ABSOLUTE_POS;
    if row < counts.len() {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let mut count = 0u32;
        for position in start..end {
            // By bits, not `value != 0`: a GPU may flush a subnormal to zero in a
            // comparison (Metal does), and the CPU counts it. `-0.0` is zero, a NaN is
            // not (bits << 1 drops the sign).
            let bits = u32::reinterpret(value_f32::<T>(values[position]));
            if (bits << 1u32) != 0u32 {
                count += 1u32;
            }
        }
        counts[row] = count;
    }
}

/// Each value divided by its row's divisor in `f32` (`normalize_total`'s scaling, as
/// the CPU's `divide_rows`); `rows[i]` is the row of value `i`.
#[cube(launch)]
pub fn divide_rows<T: Numeric>(rows: &[u32], values: &[T], divisors: &[f32], out: &mut [f32]) {
    let position = ABSOLUTE_POS;
    if position < out.len() {
        let divisor = divisors[rows[position] as usize];
        out[position] = value_f32::<T>(values[position]) / divisor;
    }
}

/// `log1p` of each value (a `u32` rounded to `f32` first): [`log1p_f32`].
#[cube(launch)]
pub fn log1p<T: Numeric>(values: &[T], out: &mut [f32]) {
    let position = ABSOLUTE_POS;
    if position < out.len() {
        out[position] = log1p_f32(value_f32::<T>(values[position]));
    }
}

/// `ln(1 + x)` in `f32` arithmetic only (no `f64` on Metal), within a few ulp: musl's
/// `log1pf` (from FreeBSD's `s_log1pf.c`), with its early returns as selects. A
/// platform's `log` is not used: WGSL promises only an absolute error near 1.
#[cube]
pub fn log1p_f32(x: f32) -> f32 {
    let ln2_high = 6.931_381e-1_f32; // 0x3f317180
    let ln2_low = 9.058_001e-6_f32; // 0x3717f7d1
    let lg1 = 0.666_666_6f32; // 0xaaaaaa.0p-24
    let lg2 = 0.400_009_72f32; // 0xccce13.0p-25
    let lg3 = 0.284_987_87f32; // 0x91e9ee.0p-25
    let lg4 = 0.242_790_79f32; // 0xf89e26.0p-26

    let bits = u32::reinterpret(x);
    // 1 + x reduced to [sqrt(2)/2, sqrt(2)) times 2^k, with the rounding of 1 + x
    // corrected by c (musl's k != 0 branch).
    let one_plus = 1.0f32 + x;
    let reduced_bits = u32::reinterpret(one_plus) + (0x3f80_0000u32 - 0x3f35_04f3u32);
    let k_shifted = (reduced_bits >> 23u32) as i32 - 0x7f;
    let mut correction = 0.0f32;
    if k_shifted < 25 {
        if k_shifted >= 2 {
            correction = 1.0f32 - (one_plus - x);
        } else {
            correction = x - (one_plus - 1.0f32);
        }
        correction /= one_plus;
    }
    let mantissa = f32::reinterpret((reduced_bits & 0x007f_ffffu32) + 0x3f35_04f3u32);
    let mut f = mantissa - 1.0f32;
    let mut k = k_shifted;
    // sqrt(2)/2 - 1 <= x < sqrt(2) - 1: no reduction (musl's k = 0 branch).
    let small = bits < 0x3ed4_13d0u32 || bits >= 0x8000_0000u32;
    if small && bits <= 0xbe95_f619u32 {
        k = 0;
        correction = 0.0f32;
        f = x;
    }
    let s = f / (2.0f32 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (lg2 + w * lg4);
    let t2 = z * (lg1 + w * lg3);
    let r = t2 + t1;
    let half_f_squared = 0.5f32 * f * f;
    let dk = k as f32;
    let mut result =
        s * (half_f_squared + r) + (dk * ln2_low + correction) - half_f_squared + f + dk * ln2_high;
    // Special cases, in musl's order of tests.
    let magnitude = bits & 0x7fff_ffffu32;
    if magnitude < 0x3380_0000u32 {
        // |x| < 2^-24: log1p(x) rounds to x (zeros keep their sign).
        result = x;
    }
    if bits >= 0x7f80_0000u32 && bits < 0x8000_0000u32 {
        // +inf, and NaNs with the sign bit clear.
        result = x;
    }
    if bits == 0xbf80_0000u32 {
        result = f32::reinterpret(0xff80_0000u32); // log1p(-1) = -inf
    }
    if bits > 0xbf80_0000u32 {
        // x < -1, -inf and NaNs with the sign bit set: NaN.
        result = f32::reinterpret(0x7fc0_0000u32);
    }
    result
}
