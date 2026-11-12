//! Tail probabilities and p-value corrections for differential expression (task 6.10).
//!
//! Does: the two-sided p-values scanpy's `rank_genes_groups` takes from scipy (the
//! normal distribution's `2 * norm.sf(|z|)` for Wilcoxon, Student's t `2 * t.sf(|t|,
//! df)` for Welch's t-test) and statsmodels' Benjamini-Hochberg correction, in `f64`
//! with full relative precision in the far tails (p-values of `1e-300` keep their
//! digits).
//! Does not: rank or test anything (`rank_genes.rs`).

/// `sqrt(1/2)`, cephes' `SQRTH`.
const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// cephes' `MAXLOG`: `erfc(x)` is 0 once `x^2` exceeds it.
const MAX_LOG: f64 = 7.097_827_128_933_84e2;

/// The two-sided p-value of a standard normal statistic: `2 * norm.sf(|z|)` as scipy
/// computes it (cephes' `ndtr(-|z|)`, then doubled; 0 beyond `|z|` of about 37.7,
/// where cephes' `erfc` underflows instead of going subnormal). `NaN` stays `NaN`.
pub(crate) fn normal_two_sided(z: f64) -> f64 {
    let x = -z.abs() * SQRT_HALF;
    if x * x > MAX_LOG {
        return 0.0;
    }
    let tail = if x.abs() < SQRT_HALF {
        0.5 + 0.5 * libm::erf(x)
    } else {
        0.5 * libm::erfc(x.abs())
    };
    2.0 * tail
}

/// The two-sided p-value of Student's t statistic `t` with `df` degrees of freedom:
/// `2 * t.sf(|t|, df) = I_{df / (df + t^2)}(df / 2, 1 / 2)`. `NaN` in either stays
/// `NaN`; an infinite `t` gives 0, `t = 0` gives 1.
pub(crate) fn student_two_sided(t: f64, df: f64) -> f64 {
    if t.is_nan() || df.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    if t.is_infinite() {
        return 0.0;
    }
    if t == 0.0 {
        return 1.0;
    }
    let squared = t * t;
    if df.is_infinite() {
        return normal_two_sided(t);
    }
    // x = df / (df + t^2) and 1 - x = t^2 / (df + t^2), each computed without
    // cancellation.
    let denominator = df + squared;
    let x = df / denominator;
    let one_minus_x = squared / denominator;
    regularized_incomplete_beta(0.5 * df, 0.5, x, one_minus_x).clamp(0.0, 1.0)
}

/// The regularized incomplete beta function `I_x(a, b)` for `a, b > 0`, given `x` and
/// `1 - x` separately (so a tail close to 1 keeps its digits). Lentz's continued
/// fraction (Numerical Recipes' `betacf`), on the side where it converges fast.
fn regularized_incomplete_beta(a: f64, b: f64, x: f64, one_minus_x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if one_minus_x <= 0.0 {
        return 1.0;
    }
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(b, a, one_minus_x, x);
    }
    // ln(x^a (1 - x)^b / B(a, b)): ln x through ln_1p when x is close to 1.
    let ln_x = if one_minus_x < 0.5 {
        (-one_minus_x).ln_1p()
    } else {
        x.ln()
    };
    let ln_one_minus_x = if x < 0.5 {
        (-x).ln_1p()
    } else {
        one_minus_x.ln()
    };
    let ln_beta = libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b);
    let front = (a * ln_x + b * ln_one_minus_x - ln_beta).exp() / a;
    front * beta_continued_fraction(a, b, x)
}

/// The continued fraction of `I_x(a, b)` (modified Lentz).
fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    const EPSILON: f64 = 1e-16;
    const MAX_TERMS: u32 = 100_000;
    let clamp_tiny = |value: f64| {
        if value.abs() < TINY { TINY } else { value }
    };
    let mut c = 1.0;
    let mut d = 1.0 / clamp_tiny(1.0 - (a + b) * x / (a + 1.0));
    let mut fraction = d;
    for m in 1..=MAX_TERMS {
        let m = f64::from(m);
        let m2 = 2.0 * m;
        // Even step.
        let numerator = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2));
        d = 1.0 / clamp_tiny(1.0 + numerator * d);
        c = clamp_tiny(1.0 + numerator / c);
        fraction *= d * c;
        // Odd step.
        let numerator = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0));
        d = 1.0 / clamp_tiny(1.0 + numerator * d);
        c = clamp_tiny(1.0 + numerator / c);
        let step = d * c;
        fraction *= step;
        if (step - 1.0).abs() < EPSILON {
            break;
        }
    }
    fraction
}

/// Benjamini-Hochberg adjusted p-values, as statsmodels' `multipletests(pvals,
/// method="fdr_bh")` (scanpy sets `NaN` to 1 first): sorted ascending, `p / (i / n)`,
/// the running minimum from the largest down, capped at 1, in the input's order.
pub(crate) fn benjamini_hochberg(p_values: &[f64]) -> Vec<f64> {
    let n = p_values.len();
    let p_values: Vec<f64> = p_values
        .iter()
        .map(|&p| if p.is_nan() { 1.0 } else { p })
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    // Ties may come in any order: equal p-values end with equal adjusted ones.
    order.sort_by(|&left, &right| p_values[left].total_cmp(&p_values[right]));
    let mut adjusted = vec![0.0; n];
    let mut running = f64::INFINITY;
    for rank in (0..n).rev() {
        let position = order[rank];
        let ecdf = (rank + 1) as f64 / n as f64;
        running = running.min(p_values[position] / ecdf);
        adjusted[position] = running.min(1.0);
    }
    adjusted
}

/// Bonferroni adjusted p-values over `n_tests` tests: `min(p * n_tests, 1)` (scanpy
/// keeps `NaN`, here as scipy's `np.minimum`: `NaN`).
pub(crate) fn bonferroni(p_values: &[f64], n_tests: usize) -> Vec<f64> {
    p_values
        .iter()
        .map(|&p| {
            let scaled = p * n_tests as f64;
            if scaled.is_nan() {
                f64::NAN
            } else {
                scaled.min(1.0)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(kyto: f64, expected: f64, relative: f64) -> bool {
        (kyto - expected).abs() <= relative * expected.abs()
    }

    #[test]
    fn normal_tails_match_scipy() {
        // scipy.stats.norm.sf(z) * 2
        for (z, expected) in [
            (0.0, 1.0),
            (0.5, 0.617_075_077_451_973_8),
            (1.96, 0.049_995_790_296_440_87),
            (-3.0, 0.002_699_796_063_260_186),
            (11.965_752, 5.370_877_888_990_152e-33),
            (37.5, 9.210_706_019_163_912e-308),
            (37.74, 0.0),
        ] {
            let kyto = normal_two_sided(z);
            assert!(
                close(kyto, expected, 1e-8) || kyto == expected,
                "{z}: {kyto} vs {expected}"
            );
        }
        assert!(normal_two_sided(f64::NAN).is_nan());
    }

    #[test]
    fn student_tails_match_scipy() {
        // 2 * scipy.stats.t.sf(abs(t), df)
        for (t, df, expected) in [
            (0.0, 10.0, 1.0),
            (1.0, 1.0, 0.500_000_000_000_000_1),
            (2.0, 5.0, 0.101_939_478_829_858_32),
            (-2.5, 3.7, 0.071_822_022_911_826_78),
            (14.917_345, 180.3, 2.672_121_395_767_047_8e-33),
            (0.01, 1e6, 0.992_021_289_365_547_6),
            (40.0, 2000.0, 1.427_861_853_356_283e-257),
        ] {
            let kyto = student_two_sided(t, df);
            assert!(
                close(kyto, expected, 1e-9),
                "{t} {df}: {kyto} vs {expected}"
            );
        }
        assert!(student_two_sided(f64::NAN, 3.0).is_nan());
        assert_eq!(student_two_sided(f64::INFINITY, 3.0), 0.0);
    }

    #[test]
    fn benjamini_hochberg_matches_statsmodels() {
        // statsmodels.stats.multitest.multipletests(p, method="fdr_bh")[1]
        let p = [0.01, 0.04, 0.03, 0.04, 0.5, f64::NAN];
        let expected = [0.06, 0.06, 0.06, 0.06, 0.6, 1.0];
        let adjusted = benjamini_hochberg(&p);
        for (kyto, expected) in adjusted.iter().zip(expected) {
            assert!(close(*kyto, expected, 1e-15), "{adjusted:?}");
        }
        assert_eq!(bonferroni(&[0.01, 0.5], 4), vec![0.04, 1.0]);
    }
}
