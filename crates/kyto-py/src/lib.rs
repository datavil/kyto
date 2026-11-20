//! kyto-py: the Python bindings.
//!
//! Does: convert Python arguments, call the Rust crates, return Python objects.
//! Does not: hold any logic of its own. Python imports this as `kyto._kyto`.
#![forbid(unsafe_code)]

use pyo3::prelude::*;

pub mod atlas;
pub mod chain;
pub mod concat;
pub mod convert;
pub mod errors;
pub mod expr;
pub mod feature_access;
pub mod handles;
pub mod handoff;
pub mod materialize;
pub mod plan_validate;
pub mod qc;
pub mod rank_genes;
pub mod run;
pub mod sink_h5ad;
pub mod terminal;
pub mod to_anndata;

/// The package version, taken from the workspace `Cargo.toml` (the single source).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Stub entries no `#[gen_stub_*]` annotation makes (task 7.6): the version, and the
// expression classes written by hand in `crates/kyto-py/stubs/expressions.pyi`, so the
// stub's `__all__` lists every public name of the module.
pyo3_stub_gen::module_variable!("kyto._kyto", "__version__", &str);
pyo3_stub_gen::export_verbatim!("kyto._kyto", "CellExpr");
pyo3_stub_gen::export_verbatim!("kyto._kyto", "GeneExpr");
pyo3_stub_gen::export_verbatim!("kyto._kyto", "CellStringNamespace");
pyo3_stub_gen::export_verbatim!("kyto._kyto", "GeneStringNamespace");

/// The compiled extension module, imported by Python as `kyto._kyto`.
#[pymodule]
fn _kyto(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", VERSION)?;
    module.add_class::<atlas::Atlas>()?;
    module.add_function(wrap_pyfunction!(atlas::scan, module)?)?;
    module.add_function(wrap_pyfunction!(atlas::from_json, module)?)?;
    module.add_function(wrap_pyfunction!(plan_validate::validate_plan, module)?)?;
    module.add_function(wrap_pyfunction!(plan_validate::plan_json_schema, module)?)?;
    module.add_class::<handles::PyTable>()?;
    module.add_class::<handles::PyMatrix>()?;
    module.add_class::<handles::PyEmbedding>()?;
    module.add_class::<handles::PyGraph>()?;
    module.add_class::<handles::PyModality>()?;
    module.add_class::<expr::PyCellExpr>()?;
    module.add_class::<expr::PyGeneExpr>()?;
    module.add_class::<expr::PyCellStringNamespace>()?;
    module.add_class::<expr::PyGeneStringNamespace>()?;
    module.add_function(wrap_pyfunction!(expr::cell_column, module)?)?;
    module.add_function(wrap_pyfunction!(expr::gene_column, module)?)?;
    module.add_function(wrap_pyfunction!(expr::gene_expression, module)?)?;
    module.add_function(wrap_pyfunction!(expr::cell_profile, module)?)?;
    module.add_function(wrap_pyfunction!(expr::cell_embedding, module)?)?;
    module.add_function(wrap_pyfunction!(expr::gene_embedding, module)?)?;
    module.add_function(wrap_pyfunction!(expr::feature, module)?)?;
    module.add_function(wrap_pyfunction!(qc::total_counts, module)?)?;
    module.add_function(wrap_pyfunction!(qc::genes_detected, module)?)?;
    module.add_function(wrap_pyfunction!(qc::total_counts_in, module)?)?;
    module.add_function(wrap_pyfunction!(qc::percent_counts_in, module)?)?;
    module.add_function(wrap_pyfunction!(qc::cells_detected, module)?)?;
    module.add_function(wrap_pyfunction!(qc::gene_total_counts, module)?)?;
    module.add_function(wrap_pyfunction!(qc::gene_mean_counts, module)?)?;
    module.add_function(wrap_pyfunction!(qc::percent_dropout, module)?)?;
    module.add_function(wrap_pyfunction!(convert::convert, module)?)?;
    module.add_function(wrap_pyfunction!(concat::concat, module)?)?;
    module.add_function(wrap_pyfunction!(sink_h5ad::sink_h5ad, module)?)?;
    module.add_function(wrap_pyfunction!(errors::raise_error_example, module)?)?;
    module.add_function(wrap_pyfunction!(errors::error_example_text, module)?)?;
    module.add_function(wrap_pyfunction!(handoff::handoff_example, module)?)?;
    Ok(())
}

/// What `pyo3-stub-gen` gathered from the `#[gen_stub_*]` annotations, read by the
/// `stub_gen` binary that writes `python/kyto/_kyto.pyi` (task 7.6). It must live in this
/// crate: the annotations register through `inventory` in the library.
///
/// # Errors
///
/// When the repository's `pyproject.toml` cannot be read or parsed.
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    pyo3_stub_gen::StubInfo::from_pyproject_toml(root.join("pyproject.toml"))
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
