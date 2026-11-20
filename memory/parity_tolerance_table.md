# One parity tolerance table (task 10.2)

**Works:** every tolerance a parity test applies lives in `tests/parity/tolerances.toml`
(`[<algorithm>.<what>]` with `value`, `kind`, `task`, `tests`, one-line `reason`). Rust
reads it with `crate::tolerances::tolerance("pca.variance")` (`tests/parity/tolerances.rs`,
a `#[path]` module like `property_cases.rs`, `include_str!` plus a line scan, so no
`toml` crate); Python with `from parity import tolerance`. `tests/test_parity_suite.py`
keeps it honest: the scan and `tomllib` agree, each call's file is listed under its
entry (and the other way round), no `f32`/`f64` `const` in a Rust `*_parity.rs`, and
every 6.x task has a `  - Parity:` line with resolving links.

**Adding a parity check:** add the entry (with what you measured), call
`tolerance("...")`, list the file in `tests`, and link the test from the task's
`Parity:` line. A new 6.x task needs a `Parity:` line as soon as it is written.

**Breaks:** `tolerance("a.b")` inside a `format!`/`eprintln!` string is a Rust 2021
reserved prefix error (`{tolerance("x")}` parses `x"` as a prefixed literal): bind it to
a local first. Keep each `value = ` on one line as a plain float (the Rust scan reads
only that form; the suite test catches anything else).
