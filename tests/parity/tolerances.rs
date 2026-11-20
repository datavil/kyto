//! The parity tolerance table (task 10.2) for the Rust parity tests:
//! [`tolerance`] reads a value from `tests/parity/tolerances.toml`, the one place
//! where the tolerances against scanpy and anndata live (each with its reason).
//!
//! Shared by the crates' test binaries as a module (`#[path]`), not a crate, as
//! `tests/property_cases.rs`. The table is compiled in (`include_str!`) and read by a
//! line scan of its `[section]` headers and `value = <float>` lines, so no TOML
//! dependency is needed; `tests/test_parity_suite.py` checks that this scan and a
//! real TOML parser agree. It does not compare anything: each test applies its value.

use std::sync::OnceLock;

/// The table's text, compiled into the test binary.
const TABLE: &str = include_str!("tolerances.toml");

/// The value of the tolerance `name` (`<algorithm>.<what>`, a section of the table).
///
/// # Panics
///
/// If the table has no such section: a test must not run without its tolerance.
pub fn tolerance(name: &str) -> f64 {
    static ENTRIES: OnceLock<Vec<(String, f64)>> = OnceLock::new();
    ENTRIES
        .get_or_init(|| parse(TABLE))
        .iter()
        .find(|(entry, _)| entry == name)
        .map(|(_, value)| *value)
        .unwrap_or_else(|| panic!("no tolerance {name:?} in tests/parity/tolerances.toml"))
}

/// Each section's name and `value`, in table order.
///
/// # Panics
///
/// On a `value` line outside a section or one that is not a float.
fn parse(text: &str) -> Vec<(String, f64)> {
    let mut entries = Vec::new();
    let mut section: Option<&str> = None;
    for line in text.lines().map(str::trim) {
        if let Some(header) = line
            .strip_prefix('[')
            .and_then(|rest| rest.strip_suffix(']'))
        {
            section = Some(header);
        } else if let Some(value) = line.strip_prefix("value = ") {
            let name = section.unwrap_or_else(|| panic!("value outside a section: {line}"));
            let value = value
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("{name}: value {value:?} is not a float"));
            entries.push((name.to_owned(), value));
        }
    }
    entries
}
