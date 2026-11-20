# Every buffer of a stream goes through its pool, reads too (task 5.18)

**Symptom:** on the 50k data at 10 threads, RSS climbed about 95 MB per window inside each
streaming pass (QC through PCA's first pass 516 to 1,004 MB, the output's stream 645 to
1,121 MB) and fell back when the pass ended. `task_info(TASK_VM_INFO)` showed `reusable`
(pages the allocator freed but kept) at 0 MB: live heap, not allocator retention.

**Cause:** the store's reader joined each block into new `AlignedBuffer`s, but the
consumer gave every block back to the stream's `CellBlockPool`, and only the operators
took from it. The pool keeps up to `(window + 1) x (layers + operators)` blocks' buffers,
so each window parked about `window` read blocks' buffers in it until the cap or the
pass's end, while the next window's reads allocated new ones.

**Fix:** `kyto_store::BlockBuffers` (implemented by `CellBlockPool`) and
`CellBlocks::with_buffers`: `Windows::open` gives the pool to every reader, so a block's
buffers circulate (read, operators, consumer, pool, next read). Peak QC through PCA
1,174 to about 650 MB; no pass grows after its first window; slightly faster (fewer
page faults). Same bits: only where the values are written changed.

**Lessons:**
- A pool is only bounded if **every** producer of the buffers it receives also takes from
  it. A buffer that enters a pool from a path that never takes is a leak up to the cap.
- To tell retention from live memory on macOS, read `resident` and `reusable` from
  `task_info(TASK_VM_INFO)` (ctypes) per progress call; `ru_maxrss` alone cannot.
- With reads from the pool, a test consumer that gives a block back at once races the
  next block's read for its buffers; `estimate_fixtures.rs` now holds each block until
  the next arrives (a slow terminal, the case the estimate bounds), else its peak is
  timing-dependent.
- When a change lowers what the estimate counts, run `memory_limit.rs` against base: the
  fit picks larger blocks from a lower estimate, so an estimate that is now too low shows
  up there as RSS, not in the fixtures (they passed while the term was 25 to 37% low on
  that store). With reads from the pool, count the buffers the consumer holds (later
  blocks allocate as many again), the reading phase (only the reads have left the pool),
  and a read taking a larger layer's pooled buffer (the pool is LIFO).

**Pricing a short last window (task 5.21):** `estimate()` prices the reading phase window by
window: second window (only what the first allocated), a full window without the last
block (only if a fourth window exists), and the window holding the last block (read at its
own rows). What the bug hunt found in the first version, and how to check:
- Every per-read term must scale with the window's reads. The `inflated` term (a read
  holding a larger layer's pooled buffer) left at a full window's count made a shorter last
  window price *higher* (`scale`, +13 to +23%) and the estimate non-monotonic in stream
  length: fewer reads also leave more buffers free in the pool.
- A read joins into a pooled buffer (no growth) only if the pool has one for it: with the
  consumer holding the only one (2 blocks, 1 in flight) it allocates and grows.
- Reads cut from whole blocks (5.14) are not per block: keep their pool term.
- The fixtures have at most 3 blocks; to check a window-model change, convert pbmc3k at 512
  and 256 cells a block (temporarily) and probe `estimate()` old vs new over many shapes.
