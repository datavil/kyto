# A faster local gate (task 1.7)

**Works:**

- **The forced extension rebuild is already cheap.** `uv sync --reinstall-package kyto`
  runs maturin, whose cargo build in `target/` is incremental: about 2 s when nothing
  changed. Its cargo fingerprints (dep-info, so `include_str!` files such as
  `plan_json.schema.json` count) are a more reliable staleness check than uv's
  `cache-keys` (only `.rs` and Cargo files) or a hash of the sources, so the gate keeps
  it and only drops the separate plain `uv sync` line it implies.
- **cargo-nextest** (`scripts/rust_tests.py`): one process per test and binaries side
  by side, then `cargo test --doc` (nextest runs no doctests); `cargo test` when nextest
  is missing. Same 620 test names (`cargo test -- --list` against
  `cargo nextest list --run-ignored all`).
- **One GPU-feature build for lint and tests:** `RUSTC_WORKSPACE_WRAPPER=clippy-driver`
  with `CLIPPY_ARGS=-D__CLIPPY_HACKERY__warnings__CLIPPY_HACKERY__` makes `cargo test`
  compile the workspace crates through clippy (as `cargo clippy -- -D warnings` does in
  check mode), so the wgpu/cubecl/cudarc dependencies build once. Checked with canaries:
  a `len_zero` under `cfg(feature = "wgpu")` and an unused variable in a `cuda`-only
  test both fail it. `CLIPPY_ARGS`' separator is clippy's internal protocol; if a
  clippy release changes it, the workspace lints (`clippy::all = deny`) still fail on
  clippy lints, but rustc warnings would pass: recheck with a canary after a toolchain
  bump.

**Breaks:**

- **Memory-peak tests beside other tests.** `estimate_fixtures` failed under nextest
  (2 blocks in flight measured 35% below the estimate: with 10 tests side by side the
  two blocks rarely overlap) and passes alone. `.config/nextest.toml` gives every
  own-process binary (`crates/*/tests/*.rs` outside `it/`) the machine alone
  (`threads-required = "num-test-threads"`); `tests/test_agents_commands.py` fails if
  a new one is missing from the filter.
- Draft scripts left under `target/` break the gate: pyright checks every `.py` file
  outside dot-directories, ignored or not.
