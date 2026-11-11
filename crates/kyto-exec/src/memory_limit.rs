//! The memory limit of a run (task 5.5): a bound on the process's resident memory.
//!
//! Does: fit a physical plan to a [`MemoryLimit`] on the process's peak resident set
//! (RSS, decided by the user, Nov 11, 2026): read what the process already holds
//! ([`process_resident_bytes`]), leave the allocator its overhead
//! ([`ALLOCATOR_OVERHEAD_PERCENT`]) and give the rest to the plan's estimate
//! ([`kyto_plan::PhysicalPlan::within_memory_limit`] shrinks its blocks to fit), or
//! raise `KY-E401` with the limit the process would need, before anything is read.
//! Does not: watch memory while running (the estimate is the model, task 4.6), or
//! account for other threads of the process that allocate while the plan runs.

use kyto_error::KytoError;
use kyto_plan::{MemoryLimit, OptimizedPlan, PhysicalOptions, PhysicalPlan};

/// Heap bytes the allocator keeps resident beyond what the plan holds, in percent of
/// it: freed buffers it caches and fragmentation (measured 28% to 95% over the 5.5
/// dataset on macOS depending on the block size, `tests/memory_limit.rs`; raised from
/// 50 to 100 by the user, Nov 12, 2026 (task 5.8); provisional, re-measured with the
/// benchmarks of 10.3).
pub const ALLOCATOR_OVERHEAD_PERCENT: u64 = 100;

/// The resident set of this process now, in bytes, or `None` where the operating
/// system does not tell (then a run counts nothing as held before it).
pub fn process_resident_bytes() -> Option<u64> {
    let pid = sysinfo::get_current_pid().ok()?;
    let mut system = sysinfo::System::new();
    system.refresh_processes_specifics(
        sysinfo::ProcessesToUpdate::Some(&[pid]),
        false,
        sysinfo::ProcessRefreshKind::nothing().with_memory(),
    );
    system.process(pid).map(sysinfo::Process::memory)
}

/// The physical plan of `optimized` whose run keeps the process's peak resident set
/// under `limit`: [`fit_memory_limit_with`] with what the process holds now.
///
/// # Errors
/// `KY-E401` if the plan cannot run within `limit`, with the limit it would need.
pub fn fit_memory_limit(
    optimized: OptimizedPlan,
    options: PhysicalOptions,
    limit: MemoryLimit,
) -> Result<PhysicalPlan, KytoError> {
    let resident = process_resident_bytes().unwrap_or(0);
    fit_memory_limit_with(optimized, options, limit, resident)
}

/// The physical plan of `optimized` whose estimated peak memory, with the allocator's
/// overhead, fits in `limit` minus `resident` bytes the process already holds.
///
/// # Errors
/// `KY-E401` if even the smallest blocks do not fit, with `required_bytes` the
/// smallest limit that would: `resident` plus the plan's smallest peak with the
/// allocator's overhead (rounded up).
pub fn fit_memory_limit_with(
    optimized: OptimizedPlan,
    options: PhysicalOptions,
    limit: MemoryLimit,
    resident: u64,
) -> Result<PhysicalPlan, KytoError> {
    let budget = plan_budget(limit.bytes().saturating_sub(resident));
    let plan_limit = std::num::NonZeroU64::new(budget).unwrap_or(std::num::NonZeroU64::MIN);
    let result =
        PhysicalPlan::within_memory_limit(optimized, options, MemoryLimit::new(plan_limit));
    match result {
        Ok(plan) if plan.estimate().peak_memory() <= budget => Ok(plan),
        Ok(plan) => Err(too_small(plan.estimate().peak_memory(), resident, limit)),
        Err(KytoError::MemoryLimit { required_bytes, .. }) => {
            Err(too_small(required_bytes, resident, limit))
        }
        Err(error) => Err(error),
    }
}

/// The bytes a plan's estimate may take out of `available` resident bytes.
fn plan_budget(available: u64) -> u64 {
    let budget = u128::from(available) * 100 / u128::from(100 + ALLOCATOR_OVERHEAD_PERCENT);
    u64::try_from(budget).unwrap_or(u64::MAX)
}

/// `KY-E401` for a plan whose smallest estimated peak is `plan_bytes`.
fn too_small(plan_bytes: u64, resident: u64, limit: MemoryLimit) -> KytoError {
    let with_overhead =
        (u128::from(plan_bytes) * u128::from(100 + ALLOCATOR_OVERHEAD_PERCENT)).div_ceil(100);
    let required = with_overhead + u128::from(resident);
    KytoError::MemoryLimit {
        required_bytes: u64::try_from(required).unwrap_or(u64::MAX),
        limit_bytes: limit.bytes(),
    }
}

#[cfg(test)]
mod tests {
    use super::{plan_budget, process_resident_bytes};

    #[test]
    fn the_budget_leaves_the_allocator_its_overhead() {
        assert_eq!(plan_budget(200), 100);
        assert_eq!(plan_budget(0), 0);
        assert_eq!(plan_budget(u64::MAX), u64::MAX / 2);
    }

    #[test]
    fn this_process_has_a_resident_set() {
        if cfg!(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "windows"
        )) {
            assert!(process_resident_bytes().unwrap() > 0);
        }
    }
}
