//! The memory limit (task 5.5): fit a physical plan's blocks to a number of bytes.
//!
//! Does: parse a limit given as bytes or text (`"4GB"`, [`MemoryLimit::parse`]) and
//! build the physical plan whose [`crate::CostEstimate::peak_memory`] fits it
//! ([`PhysicalPlan::within_memory_limit`]): the options as given if they fit, else
//! smaller blocks (each of the store's cell blocks split into blocks of fewer cells,
//! [`PhysicalOptions::max_block_cells`]) and fewer blocks in flight, keeping the most
//! cells in flight; when even one block of one cell does not fit, `KY-E401` with the
//! smallest peak the plan can run in, before anything is read (R7: the limit is an
//! argument, never a global setting).
//! Does not: measure memory or watch the process (the estimate is the model, 4.6),
//! count what the process held before the plan (the interpreter, other data), or
//! shrink what does not stream (statistics, columns, embeddings, graphs, the opening
//! of a layer): those are the floor `KY-E401` reports.

use std::num::NonZeroU64;
use std::sync::Arc;

use kyto_error::KytoError;

use crate::estimate::CostEstimate;
use crate::optimized_plan::OptimizedPlan;
use crate::physical_plan::{Blocks, PhysicalOptions, PhysicalPlan};
use crate::plan::Plan;

/// A memory limit in bytes (at least 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryLimit(NonZeroU64);

/// Units [`MemoryLimit::parse`] accepts, with their bytes: decimal (`KB` = 1000) and
/// binary (`KiB` = 1024), as dask's `parse_bytes`. Matched without case.
const UNITS: [(&str, u64); 9] = [
    ("b", 1),
    ("kb", 1_000),
    ("mb", 1_000_000),
    ("gb", 1_000_000_000),
    ("tb", 1_000_000_000_000),
    ("kib", 1 << 10),
    ("mib", 1 << 20),
    ("gib", 1 << 30),
    ("tib", 1 << 40),
];

impl MemoryLimit {
    /// A limit of `bytes` bytes.
    pub fn new(bytes: NonZeroU64) -> Self {
        Self(bytes)
    }

    /// The limit in bytes.
    pub fn bytes(self) -> u64 {
        self.0.get()
    }

    /// Parse `text`: a number of bytes (`"1000000"`) or a number and a unit, with or
    /// without a space (`"4GB"`, `"1.5 GiB"`, `"512mb"`); units in [`UNITS`]. A
    /// fraction of a byte rounds down. `function` names the method the limit was given
    /// to, for the error.
    ///
    /// # Errors
    /// `KY-E215` if `text` is not such a size, is 0 bytes or is more than `2^64 - 1`.
    pub fn parse(text: &str, function: &str) -> Result<Self, KytoError> {
        let invalid = |problem: &str| {
            KytoError::InvalidArgument {
            function: function.to_owned(),
            parameter: "memory_limit".to_owned(),
            value: format!("'{text}'"),
            problem: problem.to_owned(),
            accepted: "a positive number of bytes, or a size such as '4GB', '512MiB' or '1.5 GiB' (units B, KB, MB, GB, TB, KiB, MiB, GiB, TiB)".to_owned(),
        }
        };
        let trimmed = text.trim();
        let split = trimmed
            .find(|character: char| !(character.is_ascii_digit() || character == '.'))
            .unwrap_or(trimmed.len());
        let (number, unit) = trimmed.split_at(split);
        let (whole, fraction) = number.split_once('.').unwrap_or((number, ""));
        let is_digits = |part: &str| part.bytes().all(|byte| byte.is_ascii_digit());
        if whole.is_empty() && fraction.is_empty() || !is_digits(whole) || !is_digits(fraction) {
            return Err(invalid("it does not start with a number"));
        }
        let unit = unit.trim_start().to_ascii_lowercase();
        let multiplier = if unit.is_empty() {
            1
        } else {
            UNITS
                .iter()
                .find(|(name, _)| *name == unit)
                .map(|(_, bytes)| *bytes)
                .ok_or_else(|| invalid("its unit is not one kyto knows"))?
        };
        // Exact arithmetic: whole * multiplier + fraction * multiplier / 10^digits.
        let too_large = || invalid("it is more than 2^64 - 1 bytes");
        let mut bytes = 0_u128;
        for digit in whole.bytes() {
            bytes = bytes * 10 + u128::from(digit - b'0');
            if bytes > u128::from(u64::MAX) {
                return Err(too_large());
            }
        }
        bytes *= u128::from(multiplier);
        // The fraction's first 19 digits (below 2^64; a multiplier is below 2^41, so
        // the product fits u128); later digits are below a byte.
        let (mut numerator, mut denominator) = (0_u128, 1_u128);
        for digit in fraction.bytes().take(19) {
            numerator = numerator * 10 + u128::from(digit - b'0');
            denominator *= 10;
        }
        bytes += numerator * u128::from(multiplier) / denominator;
        let bytes = u64::try_from(bytes).map_err(|_| too_large())?;
        NonZeroU64::new(bytes)
            .map(Self)
            .ok_or_else(|| invalid("a plan needs more than 0 bytes"))
    }
}

impl PhysicalPlan {
    /// The physical plan of `optimized` whose estimated peak memory fits `limit`.
    ///
    /// `options` as given if the plan fits. Otherwise the blocks shrink: every
    /// combination of blocks in flight (the given number, halved down to 1) and cells
    /// per block (the given cells, halved down to 1) is estimated, and the fitting one
    /// with the most cells in flight wins (then the larger blocks, then more blocks in
    /// flight). The estimate is the model (task 4.6): what it does not count (the
    /// process's own memory, threads' scratch) is not in the limit.
    ///
    /// # Errors
    /// `KY-E401` if no combination fits, with the smallest peak memory of all (one
    /// block of one cell when the blocks dominate; what does not stream otherwise).
    /// Nothing has been read.
    pub fn within_memory_limit(
        optimized: OptimizedPlan,
        options: PhysicalOptions,
        limit: MemoryLimit,
    ) -> Result<Self, KytoError> {
        let plan = Self::new(optimized, options);
        let fits = |estimate: CostEstimate| estimate.peak_memory() <= limit.bytes();
        let estimate = plan.estimate();
        if fits(estimate) {
            return Ok(plan);
        }
        let Some(block_cells) = plan.largest_block() else {
            // Nothing streams: no block to shrink.
            return Err(too_small(estimate.peak_memory(), limit));
        };
        let optimized = plan.optimized().clone();
        let mut best: Option<(u128, u64, u64)> = None;
        let mut smallest = estimate.peak_memory();
        for in_flight in halvings(options.blocks_in_flight.get()) {
            for cells in halvings(block_cells) {
                let candidate = PhysicalOptions {
                    blocks_in_flight: nonzero(in_flight),
                    max_block_cells: Some(nonzero(cells)),
                    ..options
                };
                let peak = Self::new(optimized.clone(), candidate)
                    .estimate()
                    .peak_memory();
                smallest = smallest.min(peak);
                if peak > limit.bytes() {
                    continue;
                }
                let key = (u128::from(in_flight) * u128::from(cells), cells, in_flight);
                if best.is_none_or(|best| key > best) {
                    best = Some(key);
                }
            }
        }
        let Some((_, cells, in_flight)) = best else {
            return Err(too_small(smallest, limit));
        };
        Ok(Self::new(
            optimized,
            PhysicalOptions {
                blocks_in_flight: nonzero(in_flight),
                max_block_cells: Some(nonzero(cells)),
                ..options
            },
        ))
    }

    /// The most cells of any streamed node's blocks, `None` if nothing streams.
    fn largest_block(&self) -> Option<u64> {
        self.executions()
            .iter()
            .filter_map(|execution| match execution.blocks() {
                Blocks::Cells(cells) => Some(cells),
                Blocks::InMemory | Blocks::None => None,
            })
            .max()
    }
}

impl Plan {
    /// This plan's physical plan with every optimizer rule, fitted to `limit`
    /// ([`PhysicalPlan::within_memory_limit`]), for a terminal that reads all of its
    /// output. The plan itself is unchanged.
    ///
    /// # Errors
    /// `KY-E401` if the plan cannot run within `limit`.
    pub fn physical_within_memory_limit(
        self: &Arc<Self>,
        options: PhysicalOptions,
        limit: MemoryLimit,
    ) -> Result<PhysicalPlan, KytoError> {
        PhysicalPlan::within_memory_limit(self.optimize(), options, limit)
    }
}

/// `value`, `value / 2`, ... down to 1 (`value` at least 1).
fn halvings(value: u64) -> impl Iterator<Item = u64> {
    std::iter::successors(Some(value.max(1)), |&value| {
        (value > 1).then_some(value / 2)
    })
}

fn nonzero(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap_or(NonZeroU64::MIN)
}

fn too_small(required_bytes: u64, limit: MemoryLimit) -> KytoError {
    KytoError::MemoryLimit {
        required_bytes,
        limit_bytes: limit.bytes(),
    }
}

#[cfg(test)]
mod tests {
    use super::{MemoryLimit, halvings};

    fn parse(text: &str) -> u64 {
        MemoryLimit::parse(text, "collect").unwrap().bytes()
    }

    #[test]
    fn sizes_parse_in_decimal_and_binary_units() {
        assert_eq!(parse("4GB"), 4_000_000_000);
        assert_eq!(parse("4 gb"), 4_000_000_000);
        assert_eq!(parse(" 512MiB "), 512 << 20);
        assert_eq!(parse("1.5 GiB"), 3 << 29);
        assert_eq!(parse("1000000"), 1_000_000);
        assert_eq!(parse("7b"), 7);
        assert_eq!(parse("0.5KB"), 500);
        assert_eq!(parse(".25kib"), 256);
        assert_eq!(parse("1.0009b"), 1);
        assert_eq!(parse("2TiB"), 2 << 40);
        assert_eq!(parse("18446744073709551615"), u64::MAX);
    }

    #[test]
    fn bad_sizes_are_invalid_arguments() {
        for (text, problem) in [
            ("", "does not start with a number"),
            ("GB", "does not start with a number"),
            (".", "does not start with a number"),
            ("-1GB", "does not start with a number"),
            ("1.2.3GB", "does not start with a number"),
            ("4 GBs", "its unit is not one kyto knows"),
            ("4 G", "its unit is not one kyto knows"),
            ("0", "more than 0 bytes"),
            ("0.0001b", "more than 0 bytes"),
            ("18446744073709551616", "more than 2^64 - 1 bytes"),
            ("20000000TB", "more than 2^64 - 1 bytes"),
        ] {
            let error = MemoryLimit::parse(text, "collect").unwrap_err();
            assert_eq!(error.code(), "KY-E215", "{text}");
            let message = error.to_string();
            assert!(message.contains("collect(..., memory_limit="), "{message}");
            assert!(message.contains(problem), "{text}: {message}");
        }
    }

    #[test]
    fn halvings_end_at_one() {
        assert_eq!(halvings(5).collect::<Vec<_>>(), vec![5, 2, 1]);
        assert_eq!(halvings(1).collect::<Vec<_>>(), vec![1]);
        assert_eq!(halvings(0).collect::<Vec<_>>(), vec![1]);
    }
}
