# Peak RSS is not the heap: baseline and allocator overhead (task 5.5)

**Measured** (macOS, debug test binary, 205 MB dataset, `normalize_total` + `log1p`):
RSS = the process before the run (~19.5 MB) + heap peak (`peak_alloc`) + 30% to 95%
of the heap that the allocator keeps (freed buffers it caches, C zstd outside the Rust
allocator, thread stacks); the share is largest for small blocks. `/usr/bin/time -l`
"maximum resident set size" is bytes on macOS; Linux `VmHWM` in `/proc/self/status`.

**Decision (user, Nov 11, 2026):** `memory_limit` bounds the process RSS: the plan's
estimate must fit `(limit - RSS now) / 2` (`kyto_exec::fit_memory_limit`,
`ALLOCATOR_OVERHEAD_PERCENT = 100`, raised from 50 by the user, Nov 12, 2026, task
5.8). At 50% the 5.5 test failed once kyto-exec linked Polars' lazy engine (5.8): the
test binary's baseline RSS rose ~5 MB (19.5 to 24.5 MB) and peak RSS reached 64.7 to
68.5 MB over the 64 MiB limit; the overhead measured at 1024-cell blocks is ~70-80%.

**How to measure:** run the limited plan in a child process (the test binary running
one `#[ignore]`d test), so writing the store and the reference run do not count; read
RSS with `sysinfo` (no `unsafe`), not `getrusage` (libc, unsafe).
