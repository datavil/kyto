//! The 5.1 Done-when check: a 100-block stream through a `CellBlockPool` makes no new
//! allocations once its buffers have grown, counted by a global allocator.
//!
//! One test only: the counting allocator sees every thread of the binary, so a second
//! test running at the same time would add its allocations to the count.

#![allow(clippy::unwrap_used)]

use std::alloc::System;
use std::collections::VecDeque;

use kyto_exec::{CellBlock, CellBlockPool, CellData, CellValues, ValueDtype};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

const BLOCKS: usize = 100;
const ROWS: u32 = 512;
const GENES: u32 = 256;
/// Blocks held at once downstream, as `PhysicalOptions::blocks_in_flight` (default 2).
const IN_FLIGHT: usize = 2;
/// Buffers per block: `indptr`, `indices`, `data`.
const BUFFERS: usize = 3;

/// Allocations and reallocations since `region` started.
fn allocations(region: &Region<'_, System>) -> usize {
    let change = region.change();
    change.allocations + change.reallocations
}

/// Entries of row `row` in block `block`: from `GENES / 2` to `GENES` when `varying`,
/// else `GENES * 3 / 4`; a fixed hash, no generator.
fn row_nnz(block: usize, row: u32, varying: bool) -> u32 {
    if !varying {
        return GENES * 3 / 4;
    }
    let hash = (block as u32).wrapping_mul(2_654_435_761) ^ row.wrapping_mul(40_503);
    GENES / 2 + hash.rotate_left(13) % (GENES / 2 + 1)
}

/// Block `block` of the stream in buffers from `pool`: the producer reserves the
/// block's nnz first, as a reader that knows its tiles' nnz would. Even blocks hold
/// `float32` values, odd ones `uint32`, so buffers change dtype on reuse.
fn produce(pool: &CellBlockPool, block: usize, varying: bool) -> CellBlock {
    let dtype = if block.is_multiple_of(2) {
        ValueDtype::F32
    } else {
        ValueDtype::U32
    };
    let nnz: usize = (0..ROWS)
        .map(|row| row_nnz(block, row, varying) as usize)
        .sum();
    let mut parts = pool.take(dtype);
    parts.indptr.reserve(ROWS as usize + 1);
    parts.indices.reserve(nnz);
    match &mut parts.data {
        CellData::F32(values) => values.reserve(nnz),
        CellData::U32(values) => values.reserve(nnz),
    }
    parts.indptr.push(0);
    for row in 0..ROWS {
        for gene in 0..row_nnz(block, row, varying) {
            parts.indices.push(gene);
            match &mut parts.data {
                CellData::F32(values) => values.push((row + gene) as f32),
                CellData::U32(values) => values.push(row + gene),
            }
        }
        parts.indptr.push(parts.indices.len() as u64);
    }
    CellBlock::from_parts(parts, u64::from(GENES)).unwrap()
}

/// The same block in fresh vectors, without a pool (the control).
fn produce_unpooled(block: usize) -> CellBlock {
    let (mut indptr, mut indices, mut data) = (vec![0_u64], Vec::new(), Vec::new());
    for row in 0..ROWS {
        for gene in 0..row_nnz(block, row, true) {
            indices.push(gene);
            data.push(row + gene);
        }
        indptr.push(indices.len() as u64);
    }
    CellBlock::from_u32(indptr, indices, data, u64::from(GENES)).unwrap()
}

/// What a consumer computes from a block: the sum of its values.
fn consume(block: &CellBlock) -> f64 {
    match block.data() {
        CellValues::F32(values) => values.iter().map(|&value| f64::from(value)).sum(),
        CellValues::U32(values) => values.iter().map(|&value| f64::from(value)).sum(),
    }
}

/// Stream `BLOCKS` blocks with `IN_FLIGHT` held downstream: the cumulative
/// allocation count after each block, and the sum of every value.
fn stream(region: &Region<'_, System>, varying: bool) -> (Vec<usize>, f64) {
    let pool = CellBlockPool::new(IN_FLIGHT + 1);
    let mut in_flight = VecDeque::with_capacity(IN_FLIGHT + 1);
    let mut counts = Vec::with_capacity(BLOCKS);
    let mut total = 0.0;
    let start = allocations(region);
    for block in 0..BLOCKS {
        in_flight.push_back(produce(&pool, block, varying));
        if in_flight.len() > IN_FLIGHT {
            let done = in_flight.pop_front().unwrap();
            total += consume(&done);
            pool.recycle(done);
        }
        counts.push(allocations(region) - start);
    }
    for done in in_flight {
        total += consume(&done);
        pool.recycle(done);
    }
    (counts, total)
}

/// Sum of every value of the varying stream, computed without blocks.
fn expected_total() -> f64 {
    let mut total = 0.0;
    for block in 0..BLOCKS {
        for row in 0..ROWS {
            for gene in 0..row_nnz(block, row, true) {
                total += f64::from(row + gene);
            }
        }
    }
    total
}

#[test]
fn allocation_count_stays_flat_across_a_100_block_stream() {
    let region = Region::new(GLOBAL);

    // Blocks of one size: the first IN_FLIGHT + 1 blocks allocate their buffers (plus,
    // once, the pool's three free lists and the lock, which allocates on first use on
    // some platforms); the other 97 allocate nothing.
    let (counts, _) = stream(&region, false);
    let warm = counts[IN_FLIGHT];
    assert!(warm <= BUFFERS * (IN_FLIGHT + 1) + 4, "warm-up: {counts:?}");
    assert_eq!(
        counts[BLOCKS - 1],
        warm,
        "allocations after warm-up: {counts:?}"
    );

    // Varying sizes: a buffer grows when a block is larger than any it held, and
    // capacity at least doubles, so each buffer grows at most twice (sizes vary by 2x);
    // after that the count is flat.
    let (counts, total) = stream(&region, true);
    let last = counts[BLOCKS - 1];
    assert!(
        last <= 3 * BUFFERS * (IN_FLIGHT + 1) + 4,
        "too many: {counts:?}"
    );
    assert_eq!(
        counts[BLOCKS / 4],
        last,
        "allocations still growing: {counts:?}"
    );
    assert_eq!(total, expected_total());
    eprintln!(
        "allocations: {last} over {BLOCKS} varying blocks, flat from block {}",
        counts.iter().position(|&count| count == last).unwrap_or(0)
    );

    // Control: fresh vectors allocate per block (three vectors, each growing by
    // doubling, then three aligned copies), so the counter does count.
    let start = allocations(&region);
    for block in 0..BLOCKS {
        let block = produce_unpooled(block);
        std::hint::black_box(consume(&block));
    }
    assert!(allocations(&region) - start >= 6 * BLOCKS);
}
