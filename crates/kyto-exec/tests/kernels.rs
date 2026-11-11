//! Task 5.3 Done-when: every kernel equals its scalar version on `proptest` inputs.
//!
//! Each kernel runs dispatched (`Kernels::detected()`, the best instruction set here)
//! and on pulp's scalar fallback (`Kernels::scalar()`), and both must give the bits of a
//! plain value-at-a-time reference written below (a NaN equals any NaN: payloads are not
//! promised). Inputs: rows of any length up to past several `SUM_LANES` chunks, any `f32`
//! bits (NaN, infinities, subnormals, `-0.0`) or any `u32`, and rows of small counts as
//! real layers hold. An ignored test walks every `f32` through `log1p` (release only).

use kyto_exec::{KernelValue, Kernels, SUM_LANES, combine_lanes, log1p};
use proptest::prelude::*;

/// Both instruction sets the kernels must agree on.
fn kernel_sets() -> [Kernels; 2] {
    [Kernels::detected(), Kernels::scalar()]
}

/// CSR `indptr` and values of `rows`.
fn flatten<T: Copy>(rows: &[Vec<T>]) -> (Vec<u64>, Vec<T>) {
    let mut indptr = vec![0_u64];
    let mut values = Vec::new();
    for row in rows {
        values.extend_from_slice(row);
        indptr.push(values.len() as u64);
    }
    (indptr, values)
}

/// Bits of `value`, all NaNs as one.
fn f64_key(value: f64) -> u64 {
    if value.is_nan() {
        u64::MAX
    } else {
        value.to_bits()
    }
}

fn f32_key(value: f32) -> u32 {
    if value.is_nan() {
        u32::MAX
    } else {
        value.to_bits()
    }
}

/// The lane sum by hand: value `j` into lane `j % SUM_LANES`, then the fixed tree
/// (written out, and checked against the public `combine_lanes`).
fn reference_sum<T: KernelValue>(row: &[T], square: bool) -> f64 {
    let mut lanes = [0.0_f64; SUM_LANES];
    for (index, &value) in row.iter().enumerate() {
        let value = value.to_f64();
        lanes[index % SUM_LANES] += if square { value * value } else { value };
    }
    let [a0, a1, a2, a3, a4, a5, a6, a7] = lanes;
    let tree = ((a0 + a4) + (a2 + a6)) + ((a1 + a5) + (a3 + a7));
    assert_eq!(f64_key(tree), f64_key(combine_lanes(lanes)));
    tree
}

fn check_all<T: KernelValue + std::fmt::Debug>(rows: &[Vec<T>], factors: &[f64]) {
    let (indptr, values) = flatten(rows);
    let factors = &factors[..rows.len()];
    for kernels in kernel_sets() {
        let set = kernels.instruction_set();

        let mut sums = vec![f64::NAN; 3];
        kernels.row_sums(&indptr, &values, &mut sums);
        let expected: Vec<u64> = rows
            .iter()
            .map(|row| f64_key(reference_sum(row, false)))
            .collect();
        assert_eq!(
            sums.iter().map(|&sum| f64_key(sum)).collect::<Vec<_>>(),
            expected,
            "row_sums {set}"
        );

        kernels.row_sums_of_squares(&indptr, &values, &mut sums);
        let expected: Vec<u64> = rows
            .iter()
            .map(|row| f64_key(reference_sum(row, true)))
            .collect();
        assert_eq!(
            sums.iter().map(|&sum| f64_key(sum)).collect::<Vec<_>>(),
            expected,
            "row_sums_of_squares {set}"
        );

        let mut counts = vec![9; 5];
        kernels.row_nnz(&indptr, &values, &mut counts);
        let expected: Vec<u64> = rows
            .iter()
            .map(|row| row.iter().filter(|value| value.to_f64() != 0.0).count() as u64)
            .collect();
        assert_eq!(counts, expected, "row_nnz {set}");

        let mut out = vec![f32::NAN; values.len()];
        kernels.scale_rows(&indptr, &values, factors, &mut out);
        let expected: Vec<u32> = rows
            .iter()
            .zip(factors)
            .flat_map(|(row, &factor)| {
                row.iter()
                    .map(move |value| f32_key((value.to_f64() * factor) as f32))
            })
            .collect();
        assert_eq!(
            out.iter().map(|&value| f32_key(value)).collect::<Vec<_>>(),
            expected,
            "scale_rows {set}"
        );

        kernels.log1p(&values, &mut out);
        let expected: Vec<u32> = values
            .iter()
            .map(|value| f32_key(log1p(value.to_f32())))
            .collect();
        assert_eq!(
            out.iter().map(|&value| f32_key(value)).collect::<Vec<_>>(),
            expected,
            "log1p {set}"
        );
    }
}

/// Rows of up to 4 chunks of lanes plus a tail.
fn rows_of<T: std::fmt::Debug>(
    value: impl Strategy<Value = T>,
) -> impl Strategy<Value = Vec<Vec<T>>> {
    prop::collection::vec(prop::collection::vec(value, 0..(4 * SUM_LANES + 7)), 0..12)
}

/// Any `f32` bits, with NaN, `-0.0` and the infinities forced in (`any::<f32>()` gives
/// no NaN).
fn any_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        8 => any::<f32>(),
        1 => Just(f32::NAN),
        1 => Just(-f32::NAN),
        1 => Just(-0.0_f32),
        1 => Just(0.0_f32),
        1 => Just(f32::INFINITY),
        1 => Just(f32::NEG_INFINITY),
    ]
}

fn factors() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(
        prop_oneof![Just(1.0), Just(0.0), any::<f64>(), 1e-3..1e4_f64],
        12,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn f32_kernels_equal_their_scalar_version(rows in rows_of(any_f32()), factors in factors()) {
        check_all(&rows, &factors);
    }

    #[test]
    fn normalized_f32_kernels_equal_their_scalar_version(
        rows in rows_of(prop_oneof![Just(0.0_f32), 0.0..1e4_f32, -1.0..1.0_f32]),
        factors in factors(),
    ) {
        check_all(&rows, &factors);
    }

    #[test]
    fn u32_kernels_equal_their_scalar_version(rows in rows_of(any::<u32>()), factors in factors()) {
        check_all(&rows, &factors);
    }

    #[test]
    fn count_kernels_equal_their_scalar_version(rows in rows_of(0..40_u32), factors in factors()) {
        check_all(&rows, &factors);
    }
}

#[test]
fn kernels_run_on_the_detected_instruction_set() {
    let set = Kernels::detected().instruction_set();
    #[cfg(target_arch = "aarch64")]
    assert_eq!(set, "neon");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    assert!(["avx512", "avx2", "scalar"].contains(&set));
}

/// Every `f32` from -1 up through `log1p`: kyto's value against libm's `f32::ln_1p` and
/// against `f64::ln_1p` rounded to `f32`. Prints the counts; asserts at most 1 ulp from
/// libm. Run on purpose: `cargo test --release -p kyto-exec --test kernels -- --ignored
/// --nocapture`.
#[test]
#[ignore = "walks 3.2e9 floats; run in release on purpose"]
fn log1p_over_every_f32() {
    let kernels = Kernels::detected();
    let mut inputs = Vec::with_capacity(1 << 20);
    let mut outputs = vec![0.0_f32; 1 << 20];
    let (mut total, mut libm_differs, mut f64_differs, mut max_ulps) = (0_u64, 0_u64, 0_u64, 0_u32);
    let negative = (0x8000_0000_u32..=(-1.0_f32).to_bits()).map(f32::from_bits);
    let positive = (0_u32..=f32::INFINITY.to_bits()).map(f32::from_bits);
    let mut values = negative.chain(positive).peekable();
    while values.peek().is_some() {
        inputs.clear();
        inputs.extend(values.by_ref().take(1 << 20));
        let outputs = &mut outputs[..inputs.len()];
        kernels.log1p(&inputs, outputs);
        for (&x, &value) in inputs.iter().zip(outputs.iter()) {
            total += 1;
            let libm = x.ln_1p();
            if value.to_bits() != libm.to_bits() {
                libm_differs += 1;
                let ulps = value.to_bits().abs_diff(libm.to_bits());
                max_ulps = max_ulps.max(ulps);
            }
            if value.to_bits() != (f64::from(x).ln_1p() as f32).to_bits() {
                f64_differs += 1;
            }
        }
    }
    println!(
        "log1p over {total} f32: differs from f32::ln_1p in {libm_differs} (max {max_ulps} ulp), from f64::ln_1p as f32 in {f64_differs}"
    );
    assert!(max_ulps <= 1);
}
