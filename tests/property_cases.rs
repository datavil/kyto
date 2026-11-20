//! The configuration of every Rust property test (task 10.1): a fixed seed and a case
//! count, so `cargo test` draws the same cases on every run and every machine.
//!
//! Shared by the crates' test binaries as a module (`#[path]`), not a crate. Two
//! environment variables, also read by the hypothesis suite (`tests/test_properties.py`),
//! make long runs:
//!
//! - `KYTO_PROPERTY_SCALE`: multiplies every test's case count (default 1).
//! - `KYTO_PROPERTY_SEED`: the seed (default [`SEED`]); change it to draw other cases.
//!
//! For example `KYTO_PROPERTY_SCALE=20 KYTO_PROPERTY_SEED=7 uv run cargo test --workspace`.
//! Failures are not persisted to files: a failing case prints its seed and minimal input.
//! It does not choose the strategies, which stay with each test.

use proptest::test_runner::{Config, RngSeed};

/// The seed every property test uses unless `KYTO_PROPERTY_SEED` says otherwise.
pub const SEED: u64 = 0x6b79_746f; // "kyto"

/// A positive integer from the environment variable `name`, or `default` if unset.
///
/// # Panics
///
/// If the variable is set but is not a positive integer: a typo must not silently run
/// the default.
fn positive(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Err(_) => default,
        Ok(text) => match text.trim().parse::<u64>() {
            Ok(value) if value > 0 => value,
            _ => panic!("{name} must be a positive integer, got {text:?}"),
        },
    }
}

/// `base` cases times `KYTO_PROPERTY_SCALE`.
pub fn cases(base: u32) -> u32 {
    let scale = u32::try_from(positive("KYTO_PROPERTY_SCALE", 1)).unwrap_or(u32::MAX);
    base.saturating_mul(scale)
}

/// The proptest configuration of a test with `base` cases: fixed seed, scaled cases, no
/// failure file (integration tests have no `lib.rs` for proptest to place it by).
pub fn config(base: u32) -> Config {
    Config {
        cases: cases(base),
        rng_seed: RngSeed::Fixed(positive("KYTO_PROPERTY_SEED", SEED)),
        failure_persistence: None,
        ..Config::default()
    }
}
