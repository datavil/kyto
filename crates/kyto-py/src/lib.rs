//! kyto-py: the Python bindings.
//!
//! Does: convert Python arguments, call the Rust crates, return Python objects.
//! Does not: hold any logic of its own. Python imports this as `kyto._kyto`.
#![forbid(unsafe_code)]

use pyo3::prelude::*;

/// The package version, taken from the workspace `Cargo.toml` (the single source).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The compiled extension module, imported by Python as `kyto._kyto`.
#[pymodule]
fn _kyto(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", VERSION)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn version_matches_workspace() {
        assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
        assert!(!VERSION.is_empty());
    }
}
