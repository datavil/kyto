//! A pool of reusable [`CellBlock`] buffers (plan 5.1).
//!
//! Does: hand out empty, 64-byte aligned [`CellBlockParts`] that keep the capacity of
//! blocks recycled before ([`CellBlockPool::take`]), and take finished blocks back
//! ([`CellBlockPool::recycle`]), so a stream of blocks stops allocating once its buffers
//! have grown to the largest block (the Done-when check of 5.1). One buffer serves
//! `float32` and `uint32` values alike. Thread-safe (`&self`), for 5.2's operators.
//! Does not: bound the bytes it keeps by itself (it keeps at most `max_blocks` blocks'
//! buffers, each as large as the largest block it held), size buffers ahead (the producer
//! reserves what it knows), or decide block sizes (the physical plan, task 4.5).

use std::sync::{Mutex, MutexGuard, PoisonError};

use kyto_store::{AlignedBuffer, CellBlock, CellBlockParts, CellData, ValueDtype};

/// Reusable buffers for [`CellBlock`]s.
///
/// Take parts, fill them, build the block with [`CellBlock::from_parts`], and give it
/// back with [`CellBlockPool::recycle`] when it is consumed. Parts come back empty with
/// their capacity; the most recently recycled come first.
#[derive(Debug)]
pub struct CellBlockPool {
    max_blocks: usize,
    free: Mutex<FreeBuffers>,
}

/// Buffers of recycled blocks, one list per buffer kind.
#[derive(Debug, Default)]
struct FreeBuffers {
    indptr: Vec<AlignedBuffer<u64>>,
    indices: Vec<AlignedBuffer<u32>>,
    data: Vec<CellData>,
}

impl CellBlockPool {
    /// An empty pool that keeps the buffers of at most `max_blocks` recycled blocks
    /// (buffers recycled beyond that are freed). Allocates nothing.
    ///
    /// Use the number of blocks in flight (`PhysicalOptions::blocks_in_flight`): a
    /// stream that holds `n` blocks at once needs `n` blocks' buffers to stop
    /// allocating.
    pub fn new(max_blocks: usize) -> Self {
        Self {
            max_blocks,
            free: Mutex::new(FreeBuffers::default()),
        }
    }

    /// The most blocks' buffers the pool keeps.
    pub fn max_blocks(&self) -> usize {
        self.max_blocks
    }

    /// Number of recycled blocks' buffers the pool holds now.
    pub fn free_blocks(&self) -> usize {
        self.lock().indptr.len()
    }

    /// Empty buffers for a block of `dtype` values: recycled ones (capacity kept) when
    /// the pool has some, else new ones that allocate only when filled.
    pub fn take(&self, dtype: ValueDtype) -> CellBlockParts {
        let mut free = self.lock();
        let indptr = free.indptr.pop().unwrap_or_default();
        let indices = free.indices.pop().unwrap_or_default();
        let data = free
            .data
            .pop()
            .map_or_else(|| CellData::new(dtype), |data| data.into_cleared(dtype));
        CellBlockParts {
            indptr,
            indices,
            data,
        }
    }

    /// Give a consumed block's buffers back for reuse (freed when the pool already
    /// holds `max_blocks`).
    pub fn recycle(&self, block: CellBlock) {
        self.recycle_parts(block.into_parts());
    }

    /// Give buffers back for reuse, as [`CellBlockPool::recycle`] does; for parts that
    /// never became a block (an error while filling them).
    pub fn recycle_parts(&self, parts: CellBlockParts) {
        let CellBlockParts {
            mut indptr,
            mut indices,
            data,
        } = parts;
        indptr.clear();
        indices.clear();
        let data = data.into_cleared(ValueDtype::U32);
        let mut free = self.lock();
        if free.indptr.len() < self.max_blocks {
            free.indptr.push(indptr);
            free.indices.push(indices);
            free.data.push(data);
        }
    }

    /// The free lists. A panic while they were held cannot leave them invalid (each
    /// push or pop is whole), so a poisoned lock is used as it is.
    fn lock(&self) -> MutexGuard<'_, FreeBuffers> {
        self.free.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use kyto_store::{BUFFER_ALIGNMENT, CellBlock, CellData, CellValues, ValueDtype};

    use super::CellBlockPool;

    /// A block of `rows` rows with `per_row` entries each, from `pool`.
    fn fill(pool: &CellBlockPool, dtype: ValueDtype, rows: u64, per_row: u32) -> CellBlock {
        let mut parts = pool.take(dtype);
        parts.indptr.push(0);
        for row in 0..rows {
            for gene in 0..per_row {
                parts.indices.push(gene);
                match &mut parts.data {
                    CellData::F32(values) => values.push(row as f32 + 0.5),
                    CellData::U32(values) => values.push(row as u32 + 1),
                }
            }
            parts.indptr.push((row + 1) * u64::from(per_row));
        }
        CellBlock::from_parts(parts, u64::from(per_row)).unwrap()
    }

    #[test]
    fn recycled_buffers_come_back_empty_with_their_capacity() {
        let pool = CellBlockPool::new(1);
        let block = fill(&pool, ValueDtype::F32, 10, 7);
        assert_eq!((block.n_rows(), block.nnz()), (10, 70));
        pool.recycle(block);
        assert_eq!(pool.free_blocks(), 1);
        let parts = pool.take(ValueDtype::U32);
        assert_eq!(pool.free_blocks(), 0);
        assert!(parts.indptr.is_empty() && parts.indices.is_empty() && parts.data.is_empty());
        assert!(parts.indptr.capacity() >= 11);
        assert!(parts.indices.capacity() >= 70);
        assert!(matches!(&parts.data, CellData::U32(values) if values.capacity() >= 70));
        pool.recycle_parts(parts);
        let block = fill(&pool, ValueDtype::U32, 2, 3);
        assert_eq!(block.data(), CellValues::U32(&[1, 1, 1, 2, 2, 2]));
        assert_eq!(block.indptr(), [0, 3, 6]);
    }

    #[test]
    fn keeps_at_most_max_blocks() {
        let pool = CellBlockPool::new(2);
        let blocks: Vec<_> = (0..3).map(|_| fill(&pool, ValueDtype::U32, 1, 1)).collect();
        blocks.into_iter().for_each(|block| pool.recycle(block));
        assert_eq!((pool.free_blocks(), pool.max_blocks()), (2, 2));
        let none = CellBlockPool::new(0);
        none.recycle(fill(&none, ValueDtype::F32, 1, 1));
        assert_eq!(none.free_blocks(), 0);
    }

    #[test]
    fn pooled_blocks_are_64_byte_aligned() {
        let pool = CellBlockPool::new(1);
        for per_row in [0, 1, 5, 33] {
            let block = fill(&pool, ValueDtype::F32, 3, per_row);
            let CellValues::F32(values) = block.data() else {
                panic!("float32 block")
            };
            for address in [
                block.indptr().as_ptr() as usize,
                block.indices().as_ptr() as usize,
                values.as_ptr() as usize,
            ] {
                assert_eq!(address % BUFFER_ALIGNMENT, 0);
            }
            pool.recycle(block);
        }
    }

    #[test]
    fn bad_parts_are_reported() {
        let pool = CellBlockPool::new(1);
        let mut parts = pool.take(ValueDtype::F32);
        parts.indptr.extend_from_slice(&[0, 1]);
        let error = CellBlock::from_parts(parts, 3).unwrap_err();
        assert_eq!(error.code(), "KY-E117");
        assert!(
            error.to_string().contains("CellBlock::from_parts"),
            "{error}"
        );
    }
}
