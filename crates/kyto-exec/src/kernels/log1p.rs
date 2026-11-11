//! kyto's `log1p` of one `f32`, written to vectorize: no branches (each special case is a
//! select at the end), no table, no libm, no fused multiply-add.
//!
//! Does: compute `ln(1 + x)` in `f64` and round once to `f32`. For an `f32` `x` with
//! `|x| >= 2^-28`, `1 + x` is exact in `f64` (at most 24 + 28 bits); smaller `x` give
//! `x` itself (the true value rounds to it). `ln(u)` takes `u = 2^k * m` with
//! `m` in `[sqrt(1/2), sqrt(2))`, `s = (m - 1) / (m + 1)`, `ln(m) = 2s + 2s * R(s^2)` with
//! the series `R(z) = z/3 + z^2/5 + ... + z^9/19` (truncation below 1e-16 relative, as
//! `|s| <= 0.1716`), plus `k * ln 2` in two parts. The `f64` result is within a few
//! `f64` ulps, so the `f32` rounding is the correctly rounded one except for values
//! within ~1e-16 of a halfway point between two `f32`s.
//! Does not: promise the bits of `f32::ln_1p` (libm differs by platform); a NaN keeps
//! its payload only when it was the input.

/// `|x|` below this gives `x` (also keeps `-0.0` and subnormals as they are).
const TINY: f32 = 3.725_290_3e-9; // 2^-28

/// `ln 2` split so that `k * LN2_HI` is exact for `|k| < 2^20` (fdlibm's split).
/// `6.93147180369123816490e-01` and `1.90821492927058770002e-10`, as bits.
const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000);
const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76);

/// `sqrt(2)`: mantissas at or above it are halved (and `k` raised).
const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// `1 / (2i + 1)` for `i = 1..=9`: the series of `atanh(s) / s - 1` in `z = s^2`.
const SERIES: [f64; 9] = [
    1.0 / 3.0,
    1.0 / 5.0,
    1.0 / 7.0,
    1.0 / 9.0,
    1.0 / 11.0,
    1.0 / 13.0,
    1.0 / 15.0,
    1.0 / 17.0,
    1.0 / 19.0,
];

/// `ln(1 + x)` rounded to `f32`: `-1` gives `-inf`, below `-1` NaN, `+inf` gives
/// `+inf`, `0` and `-0` themselves. The scalar version of [`super::Kernels::log1p`].
///
/// # Examples
/// ```
/// assert_eq!(kyto_exec::log1p(0.0), 0.0);
/// assert_eq!(kyto_exec::log1p(1.0), std::f32::consts::LN_2);
/// assert_eq!(kyto_exec::log1p(-1.0), f32::NEG_INFINITY);
/// ```
#[inline(always)]
pub fn log1p(x: f32) -> f32 {
    let wide = f64::from(x);
    let u = 1.0 + wide;
    // u = 2^exponent * mantissa, mantissa in [1, 2); u >= 2^-24 is normal for x > -1.
    let bits = u.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mantissa = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    let high = mantissa >= SQRT_2;
    let mantissa = if high { mantissa * 0.5 } else { mantissa };
    let k = (exponent + i64::from(high)) as f64;
    let f = mantissa - 1.0; // exact (Sterbenz)
    let s = f / (2.0 + f);
    let z = s * s;
    let mut series = SERIES[8];
    for &coefficient in SERIES[..8].iter().rev() {
        series = coefficient + z * series;
    }
    let two_s = s + s;
    let ln_mantissa = two_s + two_s * (z * series);
    let result = (k * LN2_HI + (ln_mantissa + k * LN2_LO)) as f32;
    // Special cases, as selects.
    let result = if x.abs() < TINY { x } else { result };
    let result = if x == f32::INFINITY { x } else { result };
    let result = if x == -1.0 { f32::NEG_INFINITY } else { result };
    let result = if x < -1.0 { f32::NAN } else { result };
    if x.is_nan() { x } else { result }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Units in the last place between two finite `f32`s of the same sign.
    fn ulps(a: f32, b: f32) -> u32 {
        a.to_bits().abs_diff(b.to_bits())
    }

    #[test]
    fn special_values() {
        assert_eq!(log1p(0.0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(log1p(-0.0).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(log1p(f32::INFINITY), f32::INFINITY);
        assert_eq!(log1p(-1.0), f32::NEG_INFINITY);
        assert!(log1p(-1.5).is_nan());
        assert!(log1p(f32::NEG_INFINITY).is_nan());
        assert!(log1p(f32::NAN).is_nan());
        assert_eq!(log1p(f32::MIN_POSITIVE), f32::MIN_POSITIVE);
        assert_eq!(log1p(1e-45), 1e-45);
        assert_eq!(log1p(1.0), std::f32::consts::LN_2);
        assert_eq!(log1p(f32::MAX), f32::MAX.ln_1p());
    }

    #[test]
    fn within_one_ulp_of_libm_and_rounds_the_f64_value() {
        // A walk over the f32 range from -1 up, every 4099th value.
        let mut bits = (-1.0_f32).to_bits();
        let mut checked = 0;
        while bits > 0x8000_0000 {
            let x = f32::from_bits(bits);
            assert!(ulps(log1p(x), x.ln_1p()) <= 1, "{x}");
            assert_eq!(log1p(x), f64::from(x).ln_1p() as f32, "{x}");
            bits -= 4099;
            checked += 1;
        }
        let mut bits = 0_u32;
        while bits < f32::INFINITY.to_bits() {
            let x = f32::from_bits(bits);
            assert!(ulps(log1p(x), x.ln_1p()) <= 1, "{x}");
            assert_eq!(log1p(x), f64::from(x).ln_1p() as f32, "{x}");
            bits += 4099;
            checked += 1;
        }
        assert!(checked > 500_000);
    }
}
