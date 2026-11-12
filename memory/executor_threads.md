# Executor threads: one rayon pool per run, `install`, timing tests in release (task 5.2)

**Works:** `kyto_exec::BlockStream` builds a `rayon::ThreadPool` of exactly
`ExecuteOptions::threads` (R7: no global pool size) and runs the producer as
`pool.install(|| ...)` on a thread of its own. Everything rayon below it (the tile
reader's `par_iter` over tiles and blocks, zarrs' codec concurrency, the operators'
`into_par_iter` over blocks) then runs on that pool, so the thread count holds end to end.
An indexed parallel collect (`Vec::into_par_iter().map().collect::<Vec<_>>()`) keeps
cell order on any thread count (R11). A rendezvous `sync_channel(0)` gives
backpressure with at most `blocks_in_flight + 1` blocks held; Drop drops the receiver
before joining, so a producer blocked in `send` gets an error and returns.

**Watch out:**
- Each block carries every layer the scan read; drop what the output does not read at
  the end of the stream, or projected-away layers ride along (a test caught it).
- Thread scaling is a timing check: keep it `#[ignore]` and run it in release
  (`uv run cargo test --release -p kyto-exec --test it -- thread_scaling::
  --ignored --nocapture`), with a warm-up run and the median of several. On this 10-core Mac
  (4 performance + 6 efficiency cores) pbmc3k at 128 x 4096 tiles gave 1.84x, 2.74x,
  4.09x at 2, 4, 8 threads with as many blocks in flight; past 4 threads the
  efficiency cores add less.
- With the default 2 blocks in flight, parallelism over blocks stops at 2; the rest
  comes from tiles within a block (gene blocks). More threads need more blocks in
  flight, which costs memory (4.6's estimate counts them).
