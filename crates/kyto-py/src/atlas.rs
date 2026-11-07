//! The Python `Atlas`: an immutable lazy plan (task 3.2), and the private `scan` that
//! builds one.
//!
//! Does: wrap an `Arc<kyto_plan::Plan>` in a frozen pyclass whose every method returns a
//! new `Atlas` (R2); resolve an expression's names against the plan (task 3.5); build
//! the root plan from a store's `schema.json` and the `gene_id` column of its gene table;
//! hand out the read-only handles (task 3.6; the accessors are 7.3's, taken early);
//! filter genes (`filter_genes`, 4.1/7.2's gene filter taken early for task 3.8).
//! Does not: read other columns or the matrix; hold any state besides the plan (R7); expose
//! `__getitem__`, `.X`, `obs` or `var` (R4). The public `kyto.scan` and the chain
//! methods arrive with 7.1/7.2; `scan` is private until then.

use std::path::PathBuf;
use std::sync::Arc;

use kyto_core::GeneIds;
use kyto_error::KytoError;
use kyto_plan::Plan;
use pyo3::prelude::*;
use pyo3::{IntoPyObjectExt, PyAny};

use crate::errors::to_py_err;
use crate::expr::{PyCellExpr, PyGeneExpr};
use crate::handles::{PyEmbedding, PyGraph, PyMatrix, PyModality, PyTable};

/// A lazy, immutable plan over a kyto store.
///
/// An ``Atlas`` records what to compute; it reads no matrix data. Every method returns
/// a new ``Atlas`` and leaves the one it was called on unchanged (R2), so an ``Atlas``
/// can be shared and reused freely. Plans built from one another share their common
/// part; nothing is copied.
///
/// Examples
/// --------
/// >>> atlas = scan(store)
/// >>> counts = atlas.with_default_layer("counts")
/// >>> print(atlas)
/// SCAN "..." (2 cells x 3 genes)
#[pyclass(frozen, module = "kyto", name = "Atlas")]
pub struct Atlas {
    plan: Arc<Plan>,
}

#[pymethods]
impl Atlas {
    /// Set the layer that short gene names refer to.
    ///
    /// Parameters
    /// ----------
    /// layer : str
    ///     Name of a layer of the store, as stored (case-sensitive).
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// MissingMemberError
    ///     ``KY-E109``: the store has no layer named ``layer``.
    ///
    /// Examples
    /// --------
    /// >>> atlas = scan(store)
    /// >>> print(atlas.with_default_layer("counts"))
    /// WITH_DEFAULT_LAYER "counts"
    /// SCAN "..." (2 cells x 3 genes)
    fn with_default_layer(&self, py: Python<'_>, layer: &str) -> PyResult<Self> {
        self.plan
            .with_default_layer(layer)
            .map(|plan| Self { plan })
            .map_err(|error| to_py_err(py, &error))
    }

    /// Resolve the names in an expression against this plan (R6).
    ///
    /// Every ``feature(...)`` name becomes the one column, gene or embedding component
    /// it names; every gene or cell read without a ``layer`` gets the plan's default
    /// layer; every column, gene, layer and embedding name is checked against the
    /// store's schema and gene IDs as read when the plan was built; every column gets
    /// its dtype from the schema, and the operations on it are checked again. The
    /// result names everything in full, so it keeps its meaning whatever is added to
    /// the store or the plan later. Full names are ``meta:<column>``,
    /// ``<layer>:<gene_id>`` and ``<embedding>:<component>``; a name that matches more
    /// than one feature is an error, never a guess.
    ///
    /// Parameters
    /// ----------
    /// expression : CellExpr or GeneExpr
    ///     The expression to resolve.
    ///
    /// Returns
    /// -------
    /// CellExpr or GeneExpr
    ///     A new expression of the same axis; ``expression`` is unchanged. Nothing is
    ///     read or copied.
    ///
    /// Raises
    /// ------
    /// ResolutionError
    ///     ``KY-E201``: a ``feature(...)`` name matches more than one feature.
    ///     ``KY-E210``: a name matches nothing, or a cell ID is empty.
    ///     ``KY-E211``: a gene or cell is read from the default layer, and the plan has
    ///     none.
    /// MissingMemberError
    ///     ``KY-E109``: a layer or embedding is not in the store.
    /// ExpressionError
    ///     ``KY-E205``: a column's stored dtype does not fit an operation on it.
    /// RemovedFeatureError
    ///     ``KY-E203``: a gene read names a gene removed earlier in the plan.
    /// StaleColumnError
    ///     ``KY-E301``: a stored derived cell column (its ``schema.json`` entry records
    ///     ``lineage``) is read after ``filter_genes``; read
    ///     ``cell_column(name).recompute()`` instead.
    /// LineageError
    ///     ``KY-E302``: a ``.recompute()`` names a column that records no lineage.
    ///
    /// Examples
    /// --------
    /// >>> atlas = scan(store).with_default_layer("counts")
    /// >>> atlas.resolve(feature("ENSG01") > 1)
    /// gene_expression('ENSG01', layer='counts') > 1
    /// >>> atlas.resolve(feature("meta:cell_id").str.starts_with("AAA"))
    /// cell_column('cell_id').str.starts_with('AAA')
    #[pyo3(signature = (expression, /))]
    fn resolve(&self, py: Python<'_>, expression: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(expression) = expression.cast::<PyCellExpr>() {
            let resolved = self
                .plan
                .resolve(expression.get().core())
                .map_err(|error| to_py_err(py, &error))?;
            return PyCellExpr::from_core(resolved).into_py_any(py);
        }
        if let Ok(expression) = expression.cast::<PyGeneExpr>() {
            let resolved = self
                .plan
                .resolve(expression.get().core())
                .map_err(|error| to_py_err(py, &error))?;
            return PyGeneExpr::from_core(resolved).into_py_any(py);
        }
        Err(to_py_err(
            py,
            &wrong_type("resolve", "a CellExpr or a GeneExpr", expression),
        ))
    }

    /// Keep the genes where a boolean gene expression is true.
    ///
    /// The predicate is resolved against this plan now (R6): its names are checked and
    /// stored in full. Which genes it keeps is known only when the plan runs. From
    /// here on, a derived cell column that the store computed over all genes (one whose
    /// ``schema.json`` entry records ``lineage``, such as a stored ``n_genes``) is stale:
    /// reading it raises ``KY-E301``; read ``cell_column(name).recompute()`` instead.
    ///
    /// Parameters
    /// ----------
    /// predicate : GeneExpr[Boolean] or GeneExpr[Unknown]
    ///     One value per gene; genes where it is ``True`` are kept, genes where it is
    ///     ``False`` or null are dropped. A column read (``Unknown`` until resolved)
    ///     must be stored as ``Boolean``.
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// ExpressionError
    ///     ``KY-E205``: ``predicate`` is not a ``GeneExpr``, or not boolean once
    ///     resolved.
    /// ResolutionError
    ///     ``KY-E210``: a name in ``predicate`` matches nothing, and the other
    ///     errors of ``resolve``.
    ///
    /// Examples
    /// --------
    /// >>> atlas = scan(store)
    /// >>> print(atlas.filter_genes(gene_column("feature_type") == "Gene Expression"))
    /// FILTER_GENES gene_column('feature_type') == 'Gene Expression'
    /// SCAN "..." (2 cells x 3 genes)
    #[pyo3(signature = (predicate, /))]
    fn filter_genes(&self, py: Python<'_>, predicate: &Bound<'_, PyAny>) -> PyResult<Self> {
        let Ok(predicate) = predicate.cast::<PyGeneExpr>() else {
            return Err(to_py_err(
                py,
                &wrong_type("filter_genes", "a boolean GeneExpr", predicate),
            ));
        };
        self.plan
            .filter_genes(predicate.get().core())
            .map(|plan| Self { plan })
            .map_err(|error| to_py_err(py, &error))
    }

    /// The cell table: a read-only, lazy handle (nothing is read).
    ///
    /// Returns
    /// -------
    /// Table
    ///     A new handle on this plan's cell table.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).cells.names()
    /// ['cell_id']
    #[getter]
    fn cells(&self) -> PyTable {
        PyTable::new(kyto_plan::Table::new(&self.plan, kyto_core::Axis::Cells))
    }

    /// The gene table: a read-only, lazy handle (nothing is read).
    ///
    /// Returns
    /// -------
    /// Table
    ///     A new handle on this plan's gene table.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).genes.names()
    /// ['gene_id', 'symbol', 'feature_type']
    #[getter]
    fn genes(&self) -> PyTable {
        PyTable::new(kyto_plan::Table::new(&self.plan, kyto_core::Axis::Genes))
    }

    /// A layer, as a read-only, lazy matrix handle (nothing is read).
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Name of a layer of the store, as stored (case-sensitive).
    ///
    /// Returns
    /// -------
    /// Matrix
    ///     A new handle on the layer.
    ///
    /// Raises
    /// ------
    /// MissingMemberError
    ///     ``KY-E109``: the store has no layer named ``name``.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).layer("counts").schema()
    /// {'name': 'counts', 'dtype': 'uint32', 'nnz': 3}
    #[pyo3(signature = (name, /))]
    fn layer(&self, py: Python<'_>, name: &str) -> PyResult<PyMatrix> {
        kyto_plan::Matrix::new(&self.plan, name)
            .map(PyMatrix::new)
            .map_err(|error| to_py_err(py, &error))
    }

    /// An embedding, as a read-only, lazy handle (nothing is read).
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Name of an embedding of the store, as stored (case-sensitive).
    ///
    /// Returns
    /// -------
    /// Embedding
    ///     A new handle on the embedding.
    ///
    /// Raises
    /// ------
    /// MissingMemberError
    ///     ``KY-E109``: the store has no embedding named ``name``.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).embedding("X_pca")
    /// Traceback (most recent call last):
    /// ...
    /// kyto.errors.MissingMemberError: KY-E109: no embedding named 'X_pca' in the store at '...'...
    #[pyo3(signature = (name, /))]
    fn embedding(&self, py: Python<'_>, name: &str) -> PyResult<PyEmbedding> {
        kyto_plan::Embedding::new(&self.plan, name)
            .map(PyEmbedding::new)
            .map_err(|error| to_py_err(py, &error))
    }

    /// A graph, as a read-only, lazy handle (nothing is read).
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Name of a graph of the store, as stored (case-sensitive).
    ///
    /// Returns
    /// -------
    /// Graph
    ///     A new handle on the graph.
    ///
    /// Raises
    /// ------
    /// MissingMemberError
    ///     ``KY-E109``: the store has no graph named ``name``.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).graph("connectivities")
    /// Traceback (most recent call last):
    /// ...
    /// kyto.errors.MissingMemberError: KY-E109: no graph named 'connectivities' in the store at '...'...
    #[pyo3(signature = (name, /))]
    fn graph(&self, py: Python<'_>, name: &str) -> PyResult<PyGraph> {
        kyto_plan::Graph::new(&self.plan, name)
            .map(PyGraph::new)
            .map_err(|error| to_py_err(py, &error))
    }

    /// A modality, as a read-only, lazy handle (nothing is read).
    ///
    /// Store format v0 holds no modalities, so for now this always raises.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Name of a modality of the store, as stored (case-sensitive).
    ///
    /// Returns
    /// -------
    /// Modality
    ///     A new handle on the modality.
    ///
    /// Raises
    /// ------
    /// MissingMemberError
    ///     ``KY-E109``: the store has no modality named ``name``.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).modality("adt")
    /// Traceback (most recent call last):
    /// ...
    /// kyto.errors.MissingMemberError: KY-E109: no modality named 'adt' in the store at '...'...
    #[pyo3(signature = (name, /))]
    fn modality(&self, py: Python<'_>, name: &str) -> PyResult<PyModality> {
        kyto_plan::Modality::new(&self.plan, name)
            .map(PyModality::new)
            .map_err(|error| to_py_err(py, &error))
    }

    /// The plan, one node per line, from the last step down to the scan.
    fn __str__(&self) -> String {
        self.plan.to_string()
    }

    /// The plan, one node per line, inside ``<Atlas ...>``.
    fn __repr__(&self) -> String {
        format!("<Atlas\n{}\n>", self.plan)
    }
}

/// `KY-E205`: `operation` needs `accepted` and got `value`, a Python object of another
/// type.
fn wrong_type(operation: &str, accepted: &str, value: &Bound<'_, PyAny>) -> KytoError {
    let type_name = value
        .get_type()
        .name()
        .map_or_else(|_| "object".to_owned(), |name| name.to_string());
    let article = if type_name.starts_with(['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']) {
        "an"
    } else {
        "a"
    };
    KytoError::ExpressionOperand {
        operation: operation.to_owned(),
        accepted: accepted.to_owned(),
        got: format!("{article} {type_name}"),
        operand: value
            .repr()
            .map_or_else(|_| format!("<{type_name}>"), |text| text.to_string()),
    }
}

/// Build the root `Atlas` of the local store at `store` (private until 7.1).
#[pyfunction]
#[pyo3(signature = (store, /))]
pub fn scan(py: Python<'_>, store: PathBuf) -> PyResult<Atlas> {
    let (schema, gene_ids) = py
        .detach(|| {
            let schema = kyto_store::open_atlas_schema(&store)?;
            let gene_ids =
                kyto_store::read_ids(&store, kyto_store::TableAxis::Genes, schema.n_genes)?;
            Ok::<_, kyto_error::KytoError>((schema, GeneIds::new(gene_ids)))
        })
        .map_err(|error| to_py_err(py, &error))?;
    Ok(Atlas {
        plan: Plan::scan(store.display().to_string(), schema, gene_ids),
    })
}
