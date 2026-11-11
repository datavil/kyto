# Counting allocations: `stats_alloc`, one test per binary, bound growth not zero (task 5.1)

**Works:** `stats_alloc` (dev-dependency, no `unsafe` in kyto) as `#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;`, then
`Region::new(GLOBAL).change()` gives `allocations` and `reallocations` since the
region began (`crates/kyto-exec/tests/cell_block_stream.rs`). Record the cumulative
count after each block and assert it stops changing; print the vector on failure.

**Watch out:**
- The allocator is global: a second test in the same binary runs on another thread
  and adds its allocations. Keep one `#[test]` per counting file.
- The first lock of a `std::sync::Mutex` allocated once on macOS; a new pool (new
  lock) pays it again. Warm-up bounds allow one more allocation for it.
- With varying block sizes, recycled buffers still grow when a block is larger than
  any they held; `reserve` at least doubles capacity, so with sizes within 2x each
  buffer grows at most twice. Assert a bound and a flat tail, not zero from block 1.
- Check the test against a broken pool (`CellBlockPool::new(0)` keeps nothing): the
  count must then climb by 3 per block.
