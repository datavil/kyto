# Measuring peak memory: `peak_alloc`, realloc moves, zarrs decode buffers (task 4.6)

**Works:** a test binary with `#[global_allocator] static A: peak_alloc::PeakAlloc`
(dev-dependency, no `unsafe` in kyto) measures heap bytes: take `current_usage()`, call
`reset_peak_usage()`, run, then `peak_usage() - before`. Its realloc allocates, copies
and frees, so a growing `Vec` counts old plus new while it moves; that is real memory
and the estimate models it. One `#[global_allocator]` per test binary: keep such tests
in their own file (`crates/kyto-plan/tests/estimate_fixtures.rs`).

**What the reader (`LayerReader`, task 2.4) really holds** (pbmc3k, one cell block:
60.8 MB measured for 18.3 MB of entries):
- the block's tiles (entries + `(C + 1) x 4` `indptr` per tile) stay alive while the
  joined CellBlock is built;
- the joined `indices`/`data` grow by `push` from empty: capacity doubles from 4, so up
  to 2x the entries, and the last growth holds 1.5x more for a moment;
- zarrs decodes whole inner chunks (`K` = 65 536 entries, 256 KiB) per tile read, so
  small layers and small tiles are dominated by decode buffers;
- opening a layer decodes a whole `tile_offsets` chunk (65 536 `u64`) and copies it:
  1 MiB for a moment, the peak of any tiny store.

**Watch out:** decode buffers scale with the tiles read at once, i.e. with the thread
count; `estimate()` assumes every tile of a block at once (an upper bound, far too high
for tiny gene blocks on few threads). Pre-sizing the joined arrays from the tiles' nnz
(5.1's pooled buffers) would remove the doubling; then `estimate.rs` must change too,
and `estimate_fixtures.rs` will say so.
