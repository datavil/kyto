//! The Python handles `Table`, `Matrix`, `Embedding`, `Graph` and `Modality` (task 3.6).
//!
//! Does: wrap the kyto-plan handles in frozen pyclasses with no constructor, no setter
//! and no method that mutates (R2); convert `names()`, `schema()` (the handle's part of
//! `schema.json`, format.md §8, as new dicts and lists) and `estimate()` to Python; give
//! each handle its fixed `to_*` terminals, which raise `KY-E601` until task 7.3 runs
//! them.
//! Does not: read data, or expose `__getitem__`, `.X`, `obs`, `var` (R4) or
//! `__array__` (no implicit dense array, R8).

use kyto_core::{
    Axis, ColumnDtype, ColumnSchema, EmbeddingSchema, FloatDtype, GraphSchema, LayerDtype,
    LayerSchema, ModalitySchema, TableSchema,
};
use kyto_error::KytoError;
use kyto_plan::Estimate;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::errors::to_py_err;

/// The task that implements the handle terminals.
const TERMINALS_TASK: &str = "7.3";

/// `KY-E601` for the terminal `method` (written `Class.method`).
fn not_implemented(py: Python<'_>, method: &str) -> PyErr {
    to_py_err(
        py,
        &KytoError::NotImplementedYet {
            method: method.to_owned(),
            task: TERMINALS_TASK.to_owned(),
        },
    )
}

fn axis_name(axis: Axis) -> &'static str {
    match axis {
        Axis::Cells => "cells",
        Axis::Genes => "genes",
    }
}

fn layer_dtype_name(dtype: LayerDtype) -> &'static str {
    match dtype {
        LayerDtype::Float32 => "float32",
        LayerDtype::Uint32 => "uint32",
    }
}

fn float_dtype_name(dtype: FloatDtype) -> &'static str {
    match dtype {
        FloatDtype::Float32 => "float32",
        FloatDtype::Float64 => "float64",
    }
}

/// A `columns` entry of `schema.json` (format.md §8), as a new dict.
fn column_dict<'py>(py: Python<'py>, column: &ColumnSchema) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", &column.name)?;
    dict.set_item("dtype", column.dtype.name())?;
    if let ColumnDtype::Enum {
        n_categories,
        ordered,
    } = column.dtype
    {
        dict.set_item("n_categories", n_categories)?;
        dict.set_item("ordered", ordered)?;
    }
    if let Some(lineage) = &column.lineage {
        let entry = PyDict::new(py);
        entry.set_item("layer", &lineage.layer)?;
        entry.set_item("node", &lineage.node)?;
        dict.set_item("lineage", entry)?;
    }
    Ok(dict)
}

/// A `cells` or `genes` entry of `schema.json`, as a new dict.
fn table_dict<'py>(py: Python<'py>, table: &TableSchema) -> PyResult<Bound<'py, PyDict>> {
    let columns = PyList::empty(py);
    for column in &table.columns {
        columns.append(column_dict(py, column)?)?;
    }
    let dict = PyDict::new(py);
    dict.set_item("columns", columns)?;
    Ok(dict)
}

/// A `layers` entry of `schema.json`, as a new dict.
fn layer_dict<'py>(py: Python<'py>, layer: &LayerSchema) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", &layer.name)?;
    dict.set_item("dtype", layer_dtype_name(layer.dtype))?;
    dict.set_item("nnz", layer.nnz)?;
    Ok(dict)
}

/// An `embeddings` entry of `schema.json`, as a new dict.
fn embedding_dict<'py>(
    py: Python<'py>,
    embedding: &EmbeddingSchema,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", &embedding.name)?;
    dict.set_item("axis", axis_name(embedding.axis))?;
    dict.set_item("width", embedding.width)?;
    dict.set_item("dtype", float_dtype_name(embedding.dtype))?;
    Ok(dict)
}

/// A `graphs` entry of `schema.json`, as a new dict.
fn graph_dict<'py>(py: Python<'py>, graph: &GraphSchema) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", &graph.name)?;
    dict.set_item("axis", axis_name(graph.axis))?;
    dict.set_item("dtype", float_dtype_name(graph.dtype))?;
    dict.set_item("nnz", graph.nnz)?;
    Ok(dict)
}

/// A modality's schema (provisional, task 3.1), as a new dict.
fn modality_dict<'py>(py: Python<'py>, modality: &ModalitySchema) -> PyResult<Bound<'py, PyDict>> {
    let layers = PyList::empty(py);
    for layer in &modality.layers {
        layers.append(layer_dict(py, layer)?)?;
    }
    let dict = PyDict::new(py);
    dict.set_item("name", &modality.name)?;
    dict.set_item("n_features", modality.n_features)?;
    dict.set_item("features", table_dict(py, &modality.features)?)?;
    dict.set_item("layers", layers)?;
    Ok(dict)
}

/// An [`Estimate`] as a new dict (`None` for what is not known).
fn estimate_dict<'py>(py: Python<'py>, estimate: Estimate) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("n_rows", estimate.n_rows)?;
    dict.set_item("n_columns", estimate.n_columns)?;
    dict.set_item("nnz", estimate.nnz)?;
    dict.set_item("dense_bytes", estimate.dense_bytes)?;
    Ok(dict)
}

/// The cell or gene table of a plan: read-only and lazy.
///
/// Taken from ``Atlas.cells`` or ``Atlas.genes``. It reads nothing: ``names()``,
/// ``schema()`` and ``estimate()`` answer from the schema read when the plan was built,
/// and the terminals ``to_polars()`` and ``lazy()`` run the plan (task 7.3). It has no
/// method that changes it or the plan (R2).
///
/// Examples
/// --------
/// >>> atlas = scan(store)
/// >>> atlas.cells
/// <Table cells: 2 rows x 1 columns>
#[pyclass(frozen, module = "kyto", name = "Table")]
pub struct PyTable {
    handle: kyto_plan::Table,
}

impl PyTable {
    /// Wrap a kyto-plan table handle.
    pub fn new(handle: kyto_plan::Table) -> Self {
        Self { handle }
    }
}

#[pymethods]
impl PyTable {
    /// The column names, ID column first, in stored order.
    ///
    /// Returns
    /// -------
    /// list of str
    ///     A new list; reads no data.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).genes.names()
    /// ['gene_id', 'symbol', 'feature_type']
    fn names(&self) -> Vec<String> {
        self.handle.names()
    }

    /// The table's entry in ``schema.json`` (format.md section 8).
    ///
    /// Returns
    /// -------
    /// dict
    ///     A new ``{"columns": [...]}`` dict, one dict per column with ``name`` and
    ///     ``dtype``, ``n_categories`` and ``ordered`` for an ``Enum`` column, and
    ///     ``lineage`` when the store records one. Changing it changes nothing else.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).cells.schema()
    /// {'columns': [{'name': 'cell_id', 'dtype': 'String'}]}
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        table_dict(py, self.handle.schema())
    }

    /// What the table looks like, from metadata alone (no data is read).
    ///
    /// Returns
    /// -------
    /// dict
    ///     A new dict: ``n_rows`` (cells or genes), ``n_columns``, ``nnz`` (``None``)
    ///     and ``dense_bytes`` (``None``: string sizes are not in the schema). A value
    ///     not known without running the plan is ``None``. The cost estimate (bytes
    ///     read, peak memory) is task 4.6.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).genes.estimate()
    /// {'n_rows': 3, 'n_columns': 3, 'nnz': None, 'dense_bytes': None}
    fn estimate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        estimate_dict(py, self.handle.estimate())
    }

    /// Run the plan and return the table as a Polars ``DataFrame``.
    ///
    /// Returns
    /// -------
    /// polars.DataFrame
    ///     A copy the caller owns (R2): changing it never changes the store or a
    ///     later read.
    ///
    /// Raises
    /// ------
    /// NotImplementedYetError
    ///     ``KY-E601``: terminals run with task 7.3.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).cells.to_polars()
    /// Traceback (most recent call last):
    /// ...
    /// kyto.errors.NotImplementedYetError: KY-E601: 'Table.to_polars' is not implemented yet...
    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Err(not_implemented(py, "Table.to_polars"))
    }

    /// The table as a Polars ``LazyFrame`` over the plan.
    ///
    /// Returns
    /// -------
    /// polars.LazyFrame
    ///     A new lazy frame; collecting it returns a copy the caller owns (R2).
    ///
    /// Raises
    /// ------
    /// NotImplementedYetError
    ///     ``KY-E601``: terminals run with task 7.3.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).cells.lazy()
    /// Traceback (most recent call last):
    /// ...
    /// kyto.errors.NotImplementedYetError: KY-E601: 'Table.lazy' is not implemented yet...
    fn lazy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Err(not_implemented(py, "Table.lazy"))
    }

    /// One line, from metadata only.
    fn __repr__(&self) -> String {
        let estimate = self.handle.estimate();
        format!(
            "<Table {}: {} rows x {} columns>",
            axis_name(self.handle.axis()),
            count(estimate.n_rows),
            count(estimate.n_columns),
        )
    }
}

/// A count for `repr`, `?` when not known without running the plan.
fn count(value: Option<u64>) -> String {
    value.map_or_else(|| "?".to_owned(), |value| value.to_string())
}

/// A layer of a plan, a sparse cells x genes matrix: read-only and lazy.
///
/// Taken from ``Atlas.layer(name)``. It reads nothing: ``names()``, ``schema()`` and
/// ``estimate()`` answer from metadata, and the terminals ``to_scipy()`` and
/// ``to_dense(max_bytes=...)`` run the plan (task 7.3). There is no implicit dense
/// array (R8): no ``__array__``, and dense output only through ``to_dense``. It has
/// no method that changes it or the plan (R2).
///
/// Examples
/// --------
/// >>> scan(store).layer("counts")
/// <Matrix 'counts': 2 cells x 3 genes, nnz 3, uint32>
#[pyclass(frozen, module = "kyto", name = "Matrix")]
pub struct PyMatrix {
    handle: kyto_plan::Matrix,
}

impl PyMatrix {
    /// Wrap a kyto-plan matrix handle.
    pub fn new(handle: kyto_plan::Matrix) -> Self {
        Self { handle }
    }
}

#[pymethods]
impl PyMatrix {
    /// The column names: the gene IDs, in stored order.
    ///
    /// Returns
    /// -------
    /// list of str
    ///     A new list, read from the gene table when the plan was built.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).layer("counts").names()
    /// ['ENSG01', 'ENSG02', 'ENSG03']
    fn names(&self) -> Vec<String> {
        self.handle.names().to_vec()
    }

    /// The layer's entry in ``schema.json`` (format.md section 8).
    ///
    /// Returns
    /// -------
    /// dict
    ///     A new dict with ``name``, ``dtype`` (``"float32"`` or ``"uint32"``) and
    ///     ``nnz``. Changing it changes nothing else.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).layer("counts").schema()
    /// {'name': 'counts', 'dtype': 'uint32', 'nnz': 3}
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        layer_dict(py, self.handle.schema())
    }

    /// What the matrix looks like, from metadata alone (no data is read).
    ///
    /// Returns
    /// -------
    /// dict
    ///     A new dict: ``n_rows`` (cells), ``n_columns`` (genes), ``nnz`` (stored
    ///     entries) and ``dense_bytes`` (what ``to_dense`` would allocate). A value not
    ///     known without running the plan is ``None``. The cost estimate (bytes read,
    ///     peak memory) is task 4.6.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).layer("counts").estimate()
    /// {'n_rows': 2, 'n_columns': 3, 'nnz': 3, 'dense_bytes': 24}
    fn estimate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        estimate_dict(py, self.handle.estimate())
    }

    /// Run the plan and return the matrix as a SciPy CSR matrix.
    ///
    /// Returns
    /// -------
    /// scipy.sparse.csr_matrix
    ///     Cells x genes in the layer's stored dtype; a copy the caller owns (R2).
    ///
    /// Raises
    /// ------
    /// NotImplementedYetError
    ///     ``KY-E601``: terminals run with task 7.3.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).layer("counts").to_scipy()
    /// Traceback (most recent call last):
    /// ...
    /// kyto.errors.NotImplementedYetError: KY-E601: 'Matrix.to_scipy' is not implemented yet...
    fn to_scipy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Err(not_implemented(py, "Matrix.to_scipy"))
    }

    /// Run the plan and return the matrix as a dense NumPy array, if it fits.
    ///
    /// The only way to get a dense matrix (R8): absent entries become 0 (R9).
    ///
    /// Parameters
    /// ----------
    /// max_bytes : int
    ///     The largest array, in bytes, the caller accepts. Required: there is no
    ///     default, so densifying is always a stated choice. ``estimate()["dense_bytes"]``
    ///     gives the size without reading the matrix.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Cells x genes in the layer's stored dtype; a copy the caller owns (R2).
    ///
    /// Raises
    /// ------
    /// NotImplementedYetError
    ///     ``KY-E601``: terminals run with task 7.3.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).layer("counts").to_dense(max_bytes=1024)
    /// Traceback (most recent call last):
    /// ...
    /// kyto.errors.NotImplementedYetError: KY-E601: 'Matrix.to_dense' is not implemented yet...
    #[pyo3(signature = (*, max_bytes))]
    fn to_dense(&self, py: Python<'_>, max_bytes: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // Checked by task 7.3, which defines what values it takes.
        let _ = max_bytes;
        Err(not_implemented(py, "Matrix.to_dense"))
    }

    /// One line, from metadata only.
    fn __repr__(&self) -> String {
        let estimate = self.handle.estimate();
        let schema = self.handle.schema();
        format!(
            "<Matrix '{}': {} cells x {} genes, nnz {}, {}>",
            schema.name,
            count(estimate.n_rows),
            count(estimate.n_columns),
            count(estimate.nnz),
            layer_dtype_name(schema.dtype),
        )
    }
}

/// An embedding of a plan, a dense array with one row per cell or gene: read-only and
/// lazy.
///
/// Taken from ``Atlas.embedding(name)``. It reads nothing: ``names()``, ``schema()``
/// and ``estimate()`` answer from metadata, and the terminal ``to_numpy()`` runs the
/// plan (task 7.3). An embedding is stored dense, so ``to_numpy()`` densifies nothing
/// (R8). It has no method that changes it or the plan (R2).
///
/// Examples
/// --------
/// >>> scan(store_with_members).embedding("X_pca")
/// <Embedding 'X_pca': 2 cells x 2 components, float32>
#[pyclass(frozen, module = "kyto", name = "Embedding")]
pub struct PyEmbedding {
    handle: kyto_plan::Embedding,
}

impl PyEmbedding {
    /// Wrap a kyto-plan embedding handle.
    pub fn new(handle: kyto_plan::Embedding) -> Self {
        Self { handle }
    }
}

#[pymethods]
impl PyEmbedding {
    /// The column names: ``<embedding>:<component>`` for each component, from 0.
    ///
    /// Returns
    /// -------
    /// list of str
    ///     A new list; reads no data. For a cell embedding these are the full names
    ///     ``feature(...)`` accepts (docs/feature_names.md).
    ///
    /// Examples
    /// --------
    /// >>> scan(store_with_members).embedding("X_pca").names()
    /// ['X_pca:0', 'X_pca:1']
    fn names(&self) -> Vec<String> {
        self.handle.names()
    }

    /// The embedding's entry in ``schema.json`` (format.md section 8).
    ///
    /// Returns
    /// -------
    /// dict
    ///     A new dict with ``name``, ``axis`` (``"cells"`` or ``"genes"``), ``width``
    ///     and ``dtype`` (``"float32"`` or ``"float64"``). Changing it changes nothing
    ///     else.
    ///
    /// Examples
    /// --------
    /// >>> scan(store_with_members).embedding("X_pca").schema()
    /// {'name': 'X_pca', 'axis': 'cells', 'width': 2, 'dtype': 'float32'}
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        embedding_dict(py, self.handle.schema())
    }

    /// What the embedding looks like, from metadata alone (no data is read).
    ///
    /// Returns
    /// -------
    /// dict
    ///     A new dict: ``n_rows`` (cells or genes), ``n_columns`` (components), ``nnz``
    ///     (``None``: it is dense) and ``dense_bytes`` (what ``to_numpy`` would
    ///     allocate). A value not known without running the plan is ``None``.
    ///
    /// Examples
    /// --------
    /// >>> scan(store_with_members).embedding("X_pca").estimate()
    /// {'n_rows': 2, 'n_columns': 2, 'nnz': None, 'dense_bytes': 16}
    fn estimate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        estimate_dict(py, self.handle.estimate())
    }

    /// Run the plan and return the embedding as a NumPy array.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Rows x components in the stored dtype; a copy the caller owns (R2).
    ///
    /// Raises
    /// ------
    /// NotImplementedYetError
    ///     ``KY-E601``: terminals run with task 7.3.
    ///
    /// Examples
    /// --------
    /// >>> scan(store_with_members).embedding("X_pca").to_numpy()
    /// Traceback (most recent call last):
    /// ...
    /// kyto.errors.NotImplementedYetError: KY-E601: 'Embedding.to_numpy' is not implemented yet...
    fn to_numpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Err(not_implemented(py, "Embedding.to_numpy"))
    }

    /// One line, from metadata only.
    fn __repr__(&self) -> String {
        let estimate = self.handle.estimate();
        let schema = self.handle.schema();
        format!(
            "<Embedding '{}': {} {} x {} components, {}>",
            schema.name,
            count(estimate.n_rows),
            axis_name(schema.axis),
            count(estimate.n_columns),
            float_dtype_name(schema.dtype),
        )
    }
}

/// A graph of a plan, a sparse cells x cells matrix of edge weights: read-only and
/// lazy.
///
/// Taken from ``Atlas.graph(name)``. It reads nothing: ``names()``, ``schema()`` and
/// ``estimate()`` answer from metadata, and the terminals ``to_scipy()`` and
/// ``to_edges()`` run the plan (task 7.3). It has no method that changes it or the
/// plan (R2).
///
/// Examples
/// --------
/// >>> scan(store_with_members).graph("connectivities")
/// <Graph 'connectivities': 2 x 2 cells, nnz 3, float32>
#[pyclass(frozen, module = "kyto", name = "Graph")]
pub struct PyGraph {
    handle: kyto_plan::Graph,
}

impl PyGraph {
    /// Wrap a kyto-plan graph handle.
    pub fn new(handle: kyto_plan::Graph) -> Self {
        Self { handle }
    }
}

#[pymethods]
impl PyGraph {
    /// The column names of its edge list (``to_edges()``).
    ///
    /// Returns
    /// -------
    /// list of str
    ///     A new list: ``['source', 'target', 'weight']``.
    ///
    /// Examples
    /// --------
    /// >>> scan(store_with_members).graph("connectivities").names()
    /// ['source', 'target', 'weight']
    fn names(&self) -> Vec<String> {
        self.handle.names()
    }

    /// The graph's entry in ``schema.json`` (format.md section 8).
    ///
    /// Returns
    /// -------
    /// dict
    ///     A new dict with ``name``, ``axis`` (``"cells"``), ``dtype`` (``"float32"``
    ///     or ``"float64"``) and ``nnz``. Changing it changes nothing else.
    ///
    /// Examples
    /// --------
    /// >>> scan(store_with_members).graph("connectivities").schema()
    /// {'name': 'connectivities', 'axis': 'cells', 'dtype': 'float32', 'nnz': 3}
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        graph_dict(py, self.handle.schema())
    }

    /// What the graph looks like, from metadata alone (no data is read).
    ///
    /// Returns
    /// -------
    /// dict
    ///     A new dict: ``n_rows`` and ``n_columns`` (cells), ``nnz`` (stored edges)
    ///     and ``dense_bytes`` (a dense cells x cells array). A value not known
    ///     without running the plan is ``None``.
    ///
    /// Examples
    /// --------
    /// >>> scan(store_with_members).graph("connectivities").estimate()
    /// {'n_rows': 2, 'n_columns': 2, 'nnz': 3, 'dense_bytes': 16}
    fn estimate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        estimate_dict(py, self.handle.estimate())
    }

    /// Run the plan and return the graph as a SciPy CSR matrix.
    ///
    /// Returns
    /// -------
    /// scipy.sparse.csr_matrix
    ///     Cells x cells edge weights in the stored dtype; a copy the caller owns
    ///     (R2). A stored edge may weigh 0 (format.md section 7).
    ///
    /// Raises
    /// ------
    /// NotImplementedYetError
    ///     ``KY-E601``: terminals run with task 7.3.
    ///
    /// Examples
    /// --------
    /// >>> scan(store_with_members).graph("connectivities").to_scipy()
    /// Traceback (most recent call last):
    /// ...
    /// kyto.errors.NotImplementedYetError: KY-E601: 'Graph.to_scipy' is not implemented yet...
    fn to_scipy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Err(not_implemented(py, "Graph.to_scipy"))
    }

    /// Run the plan and return the edges as a Polars ``DataFrame``.
    ///
    /// Returns
    /// -------
    /// polars.DataFrame
    ///     One row per stored edge, with the columns ``names()``; a copy the caller
    ///     owns (R2).
    ///
    /// Raises
    /// ------
    /// NotImplementedYetError
    ///     ``KY-E601``: terminals run with task 7.3.
    ///
    /// Examples
    /// --------
    /// >>> scan(store_with_members).graph("connectivities").to_edges()
    /// Traceback (most recent call last):
    /// ...
    /// kyto.errors.NotImplementedYetError: KY-E601: 'Graph.to_edges' is not implemented yet...
    fn to_edges(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Err(not_implemented(py, "Graph.to_edges"))
    }

    /// One line, from metadata only.
    fn __repr__(&self) -> String {
        let estimate = self.handle.estimate();
        let schema = self.handle.schema();
        format!(
            "<Graph '{}': {} x {} {}, nnz {}, {}>",
            schema.name,
            count(estimate.n_rows),
            count(estimate.n_columns),
            axis_name(schema.axis),
            count(estimate.nnz),
            float_dtype_name(schema.dtype),
        )
    }
}

/// A modality of a plan, a second feature space on the same cells: read-only and lazy.
///
/// Taken from ``Atlas.modality(name)``. Provisional and minimal: store format v0 holds
/// no modalities, so no store has one yet, and the format version that adds them fixes
/// the schema and the terminals. It has no method that changes it or the plan (R2).
///
/// Examples
/// --------
/// >>> scan(store).modality("adt")
/// Traceback (most recent call last):
/// ...
/// kyto.errors.MissingMemberError: KY-E109: no modality named 'adt' in the store at '...'...
#[pyclass(frozen, module = "kyto", name = "Modality")]
pub struct PyModality {
    handle: kyto_plan::Modality,
}

impl PyModality {
    /// Wrap a kyto-plan modality handle.
    pub fn new(handle: kyto_plan::Modality) -> Self {
        Self { handle }
    }
}

#[pymethods]
impl PyModality {
    /// The names of its layers.
    ///
    /// Returns
    /// -------
    /// list of str
    ///     A new list; reads no data.
    fn names(&self) -> Vec<String> {
        self.handle.names()
    }

    /// The modality's schema (provisional, as format v0 stores none).
    ///
    /// Returns
    /// -------
    /// dict
    ///     A new dict with ``name``, ``n_features``, ``features`` (a table entry) and
    ///     ``layers`` (layer entries). Changing it changes nothing else.
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        modality_dict(py, self.handle.schema())
    }

    /// What the modality looks like, from metadata alone (no data is read).
    ///
    /// Returns
    /// -------
    /// dict
    ///     A new dict: ``n_rows`` (cells), ``n_columns`` (features), ``nnz`` and
    ///     ``dense_bytes`` (``None``).
    fn estimate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        estimate_dict(py, self.handle.estimate())
    }

    /// One line, from metadata only.
    fn __repr__(&self) -> String {
        let estimate = self.handle.estimate();
        format!(
            "<Modality '{}': {} cells x {} features>",
            self.handle.schema().name,
            count(estimate.n_rows),
            count(estimate.n_columns),
        )
    }
}

#[cfg(test)]
mod tests {
    use pyo3::PyClass;
    use pyo3::pyclass::boolean_struct::True;

    use super::{PyEmbedding, PyGraph, PyMatrix, PyModality, PyTable};

    fn frozen<T: PyClass<Frozen = True>>() {}

    /// Every handle is a frozen pyclass: PyO3 then refuses setters and `&mut self`
    /// methods at compile time (R2). This fails to compile if one is not frozen.
    #[test]
    fn handles_are_frozen() {
        frozen::<PyTable>();
        frozen::<PyMatrix>();
        frozen::<PyEmbedding>();
        frozen::<PyGraph>();
        frozen::<PyModality>();
    }
}
