# Peak RSS is not the heap: baseline and allocator overhead (task 5.5)

**Measured** (macOS, debug test binary, 205 MB dataset, `normalize_total` + `log1p`):
RSS = the process before the run (~19.5 MB) + heap peak (`peak_alloc`) + 30% to 95%
of the heap that the allocator keeps (freed buffers it caches, C zstd outside the Rust
allocator, thread stacks); the share is largest for small blocks. `/usr/bin/time -l`
"maximum resident set size" is bytes on macOS; Linux `VmHWM` in `/proc/self/status`.

**Decision (user, Nov 11, 2026):** `memory_limit` bounds the process RSS: the plan's
estimate must fit `(limit - RSS now) / 1.5` (`kyto_exec::fit_memory_limit`,
`ALLOCATOR_OVERHEAD_PERCENT = 50`). The 50% is below the overhead measured at 1024-cell
blocks (~70%); the 5.5 test passes with ~6% to spare, helped by block halving.

**How to measure:** run the limited plan in a child process (the test binary running
one `#[ignore]`d test), so writing the store and the reference run do not count; read
RSS with `sysinfo` (no `unsafe`), not `getrusage` (libc, unsafe).
