# Polars debug info vs. link memory on the arm runner (task 2.3, unconfirmed)

**Broke (suspected):** after polars joined kyto-store (d30fc21), `cargo test` on
`ubuntu-24.04-arm` was cancelled twice while linking the test binaries (no error, ~3 min
silent link, far under the 30-minute timeout; run 42325035823). Likely out of memory or
disk while linking full dependency debug info; every other runner passed.

**Tried:** workspace `[profile.dev.package."*"] debug = "line-tables-only"`, which keeps
file:line in backtraces (checked inside polars frames). Not yet verified in CI: check the
arm `test` job of the next on-demand or release run. If it still dies, try fewer link
jobs (`CARGO_BUILD_JOBS`) or one test binary for the table tests.
