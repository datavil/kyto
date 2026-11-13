//! The Python handles `Table`, `Matrix`, `Embedding`, `Graph` and `Modality` (task 3.6).
//!
//! Does: wrap the kyto-plan handles in frozen pyclasses with no constructor, no setter
//! and no method that mutates (R2); convert `names()`, `schema()` (the handle's part of
//! `schema.json`, format.md §8, as new dicts and lists) and `estimate()` to Python; give
//! each handle its fixed `to_*` terminals (task 7.3): they run the plan into a store
//! (`materialize.rs`; a bare scan is read in place), read the member and hand it over
//! as a new Polars, NumPy or SciPy object (`handoff.rs`). Every handle keeps the
//! temporary store of a `collect()`ed `Atlas` alive (7.2 Open (4)).
//! Does not: read data outside the terminals, or expose `__getitem__`,
//! `.X`, `obs`, `var` (R4) or `__array__` (no implicit dense array, R8).

use std::sync::Arc;

use kyto_core::{
    Axis, ColumnDtype, ColumnSchema, EmbeddingSchema, FloatDtype, GraphSchema, LayerDtype,
    LayerSchema, ModalitySchema, TableSchema,
};
use kyto_error::KytoError;
use kyto_plan::{Estimate, Plan};
use kyto_store::{CsrMatrix, FloatValues, TableAxis};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::errors::to_py_err;
use crate::handoff;
use crate::materialize::{Materialized, materialize};

/// The temporary store of a `collect()`ed `Atlas`, shared by the handles taken from it.
type Collected = Option<Arc<tempfile::TempDir>>;

/// The store holding `plan`'s output, for the terminal `method`, with the shared run
/// arguments.
fn run(
    py: Python<'_>,
    method: &str,
    plan: &Arc<Plan>,
    collected: &Collected,
    memory_limit: Option<&Bound<'_, PyAny>>,
    threads: Option<&Bound<'_, PyAny>>,
    progress: Option<&Bound<'_, PyAny>>,
) -> PyResult<Materialized> {
    materialize(
        py,
        method,
        plan,
        collected.as_ref(),
        memory_limit,
        threads,
        progress,
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
    collected: Collected,
}

impl PyTable {
    /// Wrap a kyto-plan table handle, keeping `collected` (the plan's `collect()`
    /// store, if any) alive.
    pub fn new(handle: kyto_plan::Table, collected: Collected) -> Self {
        Self { handle, collected }
    }

    /// Run the plan and read the table as a new Python Polars frame (task 7.3).
    fn frame<'py>(
        &self,
        py: Python<'py>,
        method: &str,
        memory_limit: Option<&Bound<'_, PyAny>>,
        threads: Option<&Bound<'_, PyAny>>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = run(
            py,
            method,
            self.handle.plan(),
            &self.collected,
            memory_limit,
            threads,
            progress,
        )?;
        let axis = match self.handle.axis() {
            Axis::Cells => TableAxis::Cells,
            Axis::Genes => TableAxis::Genes,
        };
        let frame = py
            .detach(|| kyto_store::read_table(&store.store, axis))
            .map_err(|error| to_py_err(py, &error))?;
        handoff::to_polars(py, frame)
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
    /// The table after every step of the plan: its rows are the cells (or genes) the
    /// plan keeps, in store order, and its columns those of ``names()``, computed
    /// columns included. A plan with steps is run into a temporary store first (as
    /// ``Atlas.collect``); a plain ``scan`` is read in place. The GIL is released
    /// while the plan runs; Ctrl-C cancels it cleanly.
    ///
    /// Parameters
    /// ----------
    /// memory_limit : str or int or None, default=None
    ///     The most memory running the plan may use, as ``Atlas.collect`` takes it;
    ///     the result is held in memory besides.
    /// threads : int or None, default=None
    ///     Threads that run the plan; ``None`` for one per CPU core. The result does
    ///     not depend on it (R11).
    /// progress : callable or None, default=None
    ///     Called as ``progress(done, total)`` on the calling thread while the plan
    ///     runs; never for a plain ``scan``, which runs nothing.
    ///
    /// Returns
    /// -------
    /// polars.DataFrame
    ///     A copy the caller owns (R2): read from the store into new memory that is
    ///     handed to Polars without a further copy (the Arrow C stream interface), so
    ///     changing it never changes the store or a later read. String columns are
    ///     ``Enum``, as stored.
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
    ///     accepted.
    /// MemoryLimitError
    ///     ``KY-E401``: the plan cannot run within ``memory_limit``.
    /// NotImplementedYetError
    ///     ``KY-E601``: the plan has a step the executor does not run yet.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).genes.to_polars()["gene_id"].to_list()
    /// ['ENSG01', 'ENSG02', 'ENSG03']
    #[pyo3(signature = (*, memory_limit = None, threads = None, progress = None))]
    fn to_polars<'py>(
        &self,
        py: Python<'py>,
        memory_limit: Option<&Bound<'_, PyAny>>,
        threads: Option<&Bound<'_, PyAny>>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.frame(py, "Table.to_polars", memory_limit, threads, progress)
    }

    /// Run the plan and return the table as a Polars ``LazyFrame``.
    ///
    /// Runs the plan now, as ``to_polars`` (the table's schema, such as the
    /// categories of an ``Enum`` column, is known only then), and returns a lazy
    /// query over the result, to go on with in Polars.
    ///
    /// Parameters
    /// ----------
    /// memory_limit : str or int or None, default=None
    ///     The most memory running the plan may use, as ``Atlas.collect`` takes it;
    ///     the result is held in memory besides.
    /// threads : int or None, default=None
    ///     Threads that run the plan; ``None`` for one per CPU core. The result does
    ///     not depend on it (R11).
    /// progress : callable or None, default=None
    ///     Called as ``progress(done, total)`` on the calling thread while the plan
    ///     runs; never for a plain ``scan``, which runs nothing.
    ///
    /// Returns
    /// -------
    /// polars.LazyFrame
    ///     A new lazy frame over a copy the caller owns (R2), handed over as
    ///     ``to_polars``'s.
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
    ///     accepted.
    /// MemoryLimitError
    ///     ``KY-E401``: the plan cannot run within ``memory_limit``.
    /// NotImplementedYetError
    ///     ``KY-E601``: the plan has a step the executor does not run yet.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).cells.lazy().select("cell_id").collect()["cell_id"].to_list()
    /// ['AAAC-1', 'AAAG-1']
    #[pyo3(signature = (*, memory_limit = None, threads = None, progress = None))]
    fn lazy<'py>(
        &self,
        py: Python<'py>,
        memory_limit: Option<&Bound<'_, PyAny>>,
        threads: Option<&Bound<'_, PyAny>>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.frame(py, "Table.lazy", memory_limit, threads, progress)?
            .call_method0("lazy")
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
    collected: Collected,
}

/// A layer read whole, in its stored value dtype.
enum Layer {
    Float32(CsrMatrix<f32>),
    Uint32(CsrMatrix<u32>),
}

impl PyMatrix {
    /// Wrap a kyto-plan matrix handle, keeping `collected` (the plan's `collect()`
    /// store, if any) alive.
    pub fn new(handle: kyto_plan::Matrix, collected: Collected) -> Self {
        Self { handle, collected }
    }

    /// Run the plan and read the layer whole (task 7.3). With `max_bytes`, a dense
    /// copy larger than it is refused (`KY-E402`) before the layer is read.
    #[allow(clippy::too_many_arguments)]
    fn read(
        &self,
        py: Python<'_>,
        method: &str,
        max_bytes: Option<u64>,
        memory_limit: Option<&Bound<'_, PyAny>>,
        threads: Option<&Bound<'_, PyAny>>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Layer> {
        let store = run(
            py,
            method,
            self.handle.plan(),
            &self.collected,
            memory_limit,
            threads,
            progress,
        )?;
        let name = self.handle.schema().name.as_str();
        let dtype = self.handle.schema().dtype;
        py.detach(|| {
            let schema = kyto_store::open_atlas_schema(&store.store)?;
            let (n_cells, n_genes, tiling) = (schema.n_cells, schema.n_genes, schema.tiling);
            // Both layer dtypes are 4 bytes a value.
            let bytes = handoff::dense_bytes::<u32>(n_cells, n_genes);
            if let Some(max_bytes) = max_bytes
                && bytes > u128::from(max_bytes)
            {
                return Err(handoff::dense_too_large(
                    method,
                    n_cells,
                    n_genes,
                    bytes,
                    format!("max_bytes is {max_bytes}"),
                ));
            }
            Ok(match dtype {
                LayerDtype::Float32 => Layer::Float32(kyto_store::read_layer(
                    &store.store,
                    name,
                    n_cells,
                    n_genes,
                    tiling,
                )?),
                LayerDtype::Uint32 => Layer::Uint32(kyto_store::read_layer(
                    &store.store,
                    name,
                    n_cells,
                    n_genes,
                    tiling,
                )?),
            })
        })
        .map_err(|error: KytoError| to_py_err(py, &error))
    }
}

#[pymethods]
impl PyMatrix {
    /// The column names: the gene IDs, in stored order.
    ///
    /// Returns
    /// -------
    /// list of str
    ///     A new list, read from the gene table when the plan was built. After
    ///     a gene ``filter``, the genes that may remain: which ones is known only
    ///     when the plan runs, and ``estimate()`` gives ``n_columns`` as ``None``.
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
    /// The layer after every step of the plan: the cells and genes the plan keeps, in
    /// store order. A plan with steps is run into a temporary store first (as
    /// ``Atlas.collect``); a plain ``scan`` is read in place. The GIL is released
    /// while the plan runs; Ctrl-C cancels it cleanly.
    ///
    /// Parameters
    /// ----------
    /// memory_limit : str or int or None, default=None
    ///     The most memory running the plan may use, as ``Atlas.collect`` takes it;
    ///     the result is held in memory besides.
    /// threads : int or None, default=None
    ///     Threads that run the plan; ``None`` for one per CPU core. The result does
    ///     not depend on it (R11).
    /// progress : callable or None, default=None
    ///     Called as ``progress(done, total)`` on the calling thread while the plan
    ///     runs; never for a plain ``scan``, which runs nothing.
    ///
    /// Returns
    /// -------
    /// scipy.sparse.csr_matrix
    ///     Cells x genes in the layer's stored dtype (``float32`` or ``uint32``), with
    ///     sorted indices and no stored zeros, ``int32`` index arrays (``int64`` when
    ///     ``nnz`` or the genes need them); a copy the caller owns (R2): the arrays
    ///     hold memory read from the store, handed to NumPy without a further copy
    ///     (one index array is converted to SciPy's dtype).
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
    ///     accepted.
    /// MemoryLimitError
    ///     ``KY-E401``: the plan cannot run within ``memory_limit``.
    /// NotImplementedYetError
    ///     ``KY-E601``: the plan has a step the executor does not run yet.
    ///
    /// Examples
    /// --------
    /// >>> matrix = scan(store).layer("counts").to_scipy()
    /// >>> matrix.shape, matrix.dtype, matrix.nnz
    /// ((2, 3), dtype('uint32'), 3)
    #[pyo3(signature = (*, memory_limit = None, threads = None, progress = None))]
    fn to_scipy<'py>(
        &self,
        py: Python<'py>,
        memory_limit: Option<&Bound<'_, PyAny>>,
        threads: Option<&Bound<'_, PyAny>>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match self.read(py, "Matrix.to_scipy", None, memory_limit, threads, progress)? {
            Layer::Float32(matrix) => handoff::to_scipy(py, matrix),
            Layer::Uint32(matrix) => handoff::to_scipy(py, matrix),
        }
    }

    /// Run the plan and return the matrix as a dense NumPy array, if it fits.
    ///
    /// The only way to get a dense matrix (R8): absent entries become 0 (R9). When the
    /// size is known from metadata (``estimate()["dense_bytes"]``) it is checked
    /// before the plan runs; otherwise (after a filter) once the plan has run, before
    /// the layer is read.
    ///
    /// Parameters
    /// ----------
    /// max_bytes : int
    ///     The largest array, in bytes, the caller accepts. Required: there is no
    ///     default, so densifying is always a stated choice. ``estimate()["dense_bytes"]``
    ///     gives the size without reading the matrix.
    /// memory_limit : str or int or None, default=None
    ///     The most memory running the plan may use, as ``Atlas.collect`` takes it;
    ///     the result is held in memory besides.
    /// threads : int or None, default=None
    ///     Threads that run the plan; ``None`` for one per CPU core. The result does
    ///     not depend on it (R11).
    /// progress : callable or None, default=None
    ///     Called as ``progress(done, total)`` on the calling thread while the plan
    ///     runs; never for a plain ``scan``, which runs nothing.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Cells x genes, C order, in the layer's stored dtype (``float32`` or
    ///     ``uint32``); a copy the caller owns (R2): densified once into new memory,
    ///     handed to NumPy without a further copy.
    ///
    /// Raises
    /// ------
    /// MemoryLimitError
    ///     ``KY-E402``: the array is larger than ``max_bytes``, or cannot be
    ///     allocated. ``KY-E401``: the plan cannot run within ``memory_limit``.
    /// PlanError
    ///     ``KY-E215``: a ``max_bytes`` that is not an int of 0 or more, or a
    ///     ``memory_limit``, ``threads`` or ``progress`` that is not accepted.
    /// NotImplementedYetError
    ///     ``KY-E601``: the plan has a step the executor does not run yet.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).layer("counts").to_dense(max_bytes=1024)
    /// array([[5, 0, 1],
    ///        [0, 7, 0]], dtype=uint32)
    /// >>> scan(store).layer("counts").to_dense(max_bytes=8)
    /// Traceback (most recent call last):
    /// ...
    /// kyto.errors.MemoryLimitError: KY-E402: 'Matrix.to_dense' cannot allocate a dense 2 x 3 array of 24 bytes: max_bytes is 8...
    #[pyo3(signature = (*, max_bytes, memory_limit = None, threads = None, progress = None))]
    fn to_dense<'py>(
        &self,
        py: Python<'py>,
        max_bytes: &Bound<'_, PyAny>,
        memory_limit: Option<&Bound<'_, PyAny>>,
        threads: Option<&Bound<'_, PyAny>>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        const METHOD: &str = "Matrix.to_dense";
        let max_bytes = crate::chain::integer(py, METHOD, "max_bytes", max_bytes, 0)?;
        let estimate = self.handle.estimate();
        if let (Some(rows), Some(columns), Some(bytes)) =
            (estimate.n_rows, estimate.n_columns, estimate.dense_bytes)
            && bytes > u128::from(max_bytes)
        {
            let error = handoff::dense_too_large(
                METHOD,
                rows,
                columns,
                bytes,
                format!("max_bytes is {max_bytes}"),
            );
            return Err(to_py_err(py, &error));
        }
        match self.read(py, METHOD, Some(max_bytes), memory_limit, threads, progress)? {
            Layer::Float32(matrix) => handoff::to_dense(py, METHOD, &matrix, max_bytes),
            Layer::Uint32(matrix) => handoff::to_dense(py, METHOD, &matrix, max_bytes),
        }
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
/// and ``estimate()`` answer from metadata, and the terminals ``to_numpy()`` and
/// ``to_torch(device=...)`` run the plan (task 7.3). An embedding is stored dense, so ``to_numpy()`` densifies nothing (R8). It has no method that changes it or the plan (R2).
///
/// Examples
/// --------
/// >>> scan(store_with_members).embedding("X_pca")
/// <Embedding 'X_pca': 2 cells x 2 components, float32>
#[pyclass(frozen, module = "kyto", name = "Embedding")]
pub struct PyEmbedding {
    handle: kyto_plan::Embedding,
    collected: Collected,
}

impl PyEmbedding {
    /// Wrap a kyto-plan embedding handle, keeping `collected` (the plan's `collect()`
    /// store, if any) alive.
    pub fn new(handle: kyto_plan::Embedding, collected: Collected) -> Self {
        Self { handle, collected }
    }

    /// Run the plan and read the embedding as a new rows x components NumPy array.
    fn array<'py>(
        &self,
        py: Python<'py>,
        method: &str,
        memory_limit: Option<&Bound<'_, PyAny>>,
        threads: Option<&Bound<'_, PyAny>>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = run(
            py,
            method,
            self.handle.plan(),
            &self.collected,
            memory_limit,
            threads,
            progress,
        )?;
        let name = self.handle.schema().name.as_str();
        let embedding = py
            .detach(|| kyto_store::read_embedding(&store.store, name))
            .map_err(|error| to_py_err(py, &error))?;
        let (rows, width) = (embedding.n_rows(), embedding.width);
        match embedding.values {
            FloatValues::F32(values) => handoff::to_array2(py, rows, width, values),
            FloatValues::F64(values) => handoff::to_array2(py, rows, width, values),
        }
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
    /// The embedding after every step of the plan: one row per cell (or gene) the
    /// plan keeps, in store order. It is stored dense, so this densifies nothing (R8).
    ///
    /// Parameters
    /// ----------
    /// memory_limit : str or int or None, default=None
    ///     The most memory running the plan may use, as ``Atlas.collect`` takes it;
    ///     the result is held in memory besides.
    /// threads : int or None, default=None
    ///     Threads that run the plan; ``None`` for one per CPU core. The result does
    ///     not depend on it (R11).
    /// progress : callable or None, default=None
    ///     Called as ``progress(done, total)`` on the calling thread while the plan
    ///     runs; never for a plain ``scan``, which runs nothing.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Rows x components, C order, in the stored dtype; a copy the caller owns
    ///     (R2): read from the store into new memory, handed to NumPy without a
    ///     further copy.
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
    ///     accepted.
    /// MemoryLimitError
    ///     ``KY-E401``: the plan cannot run within ``memory_limit``.
    /// NotImplementedYetError
    ///     ``KY-E601``: the plan has a step the executor does not run yet.
    ///
    /// Examples
    /// --------
    /// >>> atlas = scan(store).log1p(layer="counts", into="log1p").pca(layer="log1p", into="X_pca", n_components=1, seed=0)
    /// >>> array = atlas.embedding("X_pca").to_numpy()
    /// >>> array.shape, array.dtype
    /// ((2, 1), dtype('float32'))
    #[pyo3(signature = (*, memory_limit = None, threads = None, progress = None))]
    fn to_numpy<'py>(
        &self,
        py: Python<'py>,
        memory_limit: Option<&Bound<'_, PyAny>>,
        threads: Option<&Bound<'_, PyAny>>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.array(py, "Embedding.to_numpy", memory_limit, threads, progress)
    }

    /// Run the plan and return the embedding as a PyTorch tensor on ``device``.
    ///
    /// As ``to_numpy``, then ``torch.from_numpy(array).to(device)``: on the CPU the
    /// tensor shares the new array's memory (no further copy); any other device gets
    /// one more copy, from the host. A tensor made on the device without the host
    /// copy (DLPack) is task 9.5. Needs ``torch`` installed.
    ///
    /// Parameters
    /// ----------
    /// device : str
    ///     A device PyTorch accepts, such as ``"cpu"``, ``"cuda"``, ``"cuda:1"`` or
    ///     ``"mps"``. Required (R7: the device is always an argument).
    /// memory_limit : str or int or None, default=None
    ///     The most memory running the plan may use, as ``Atlas.collect`` takes it;
    ///     the result is held in memory besides.
    /// threads : int or None, default=None
    ///     Threads that run the plan; ``None`` for one per CPU core. The result does
    ///     not depend on it (R11).
    /// progress : callable or None, default=None
    ///     Called as ``progress(done, total)`` on the calling thread while the plan
    ///     runs; never for a plain ``scan``, which runs nothing.
    ///
    /// Returns
    /// -------
    /// torch.Tensor
    ///     Rows x components in the stored dtype, on ``device``; a copy the caller
    ///     owns (R2).
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
    ///     accepted.
    /// MemoryLimitError
    ///     ``KY-E401``: the plan cannot run within ``memory_limit``.
    /// NotImplementedYetError
    ///     ``KY-E601``: the plan has a step the executor does not run yet.
    /// ModuleNotFoundError
    ///     ``torch`` is not installed.
    ///
    /// Examples
    /// --------
    /// >>> atlas = scan(store).log1p(layer="counts", into="log1p").pca(layer="log1p", into="X_pca", n_components=1, seed=0)
    /// >>> atlas.embedding("X_pca").to_torch(device="cpu").shape  # doctest: +SKIP
    /// torch.Size([2, 1])
    #[pyo3(signature = (*, device, memory_limit = None, threads = None, progress = None))]
    fn to_torch<'py>(
        &self,
        py: Python<'py>,
        device: &str,
        memory_limit: Option<&Bound<'_, PyAny>>,
        threads: Option<&Bound<'_, PyAny>>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let array = self.array(py, "Embedding.to_torch", memory_limit, threads, progress)?;
        let keywords = PyDict::new(py);
        keywords.set_item("device", device)?;
        py.import("torch")?
            .call_method1("from_numpy", (array,))?
            .call_method("to", (), Some(&keywords))
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
/// ``to_edges()`` run the plan (task 7.3). It has no
/// method that changes it or the plan (R2).
///
/// Examples
/// --------
/// >>> scan(store_with_members).graph("connectivities")
/// <Graph 'connectivities': 2 x 2 cells, nnz 3, float32>
#[pyclass(frozen, module = "kyto", name = "Graph")]
pub struct PyGraph {
    handle: kyto_plan::Graph,
    collected: Collected,
}

impl PyGraph {
    /// Wrap a kyto-plan graph handle, keeping `collected` (the plan's `collect()`
    /// store, if any) alive.
    pub fn new(handle: kyto_plan::Graph, collected: Collected) -> Self {
        Self { handle, collected }
    }

    /// Run the plan and read the graph whole.
    fn read(
        &self,
        py: Python<'_>,
        method: &str,
        memory_limit: Option<&Bound<'_, PyAny>>,
        threads: Option<&Bound<'_, PyAny>>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<kyto_store::Graph> {
        let store = run(
            py,
            method,
            self.handle.plan(),
            &self.collected,
            memory_limit,
            threads,
            progress,
        )?;
        let name = self.handle.schema().name.as_str();
        py.detach(|| kyto_store::read_graph(&store.store, name))
            .map_err(|error| to_py_err(py, &error))
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
    /// Parameters
    /// ----------
    /// memory_limit : str or int or None, default=None
    ///     The most memory running the plan may use, as ``Atlas.collect`` takes it;
    ///     the result is held in memory besides.
    /// threads : int or None, default=None
    ///     Threads that run the plan; ``None`` for one per CPU core. The result does
    ///     not depend on it (R11).
    /// progress : callable or None, default=None
    ///     Called as ``progress(done, total)`` on the calling thread while the plan
    ///     runs; never for a plain ``scan``, which runs nothing.
    ///
    /// Returns
    /// -------
    /// scipy.sparse.csr_matrix
    ///     Cells x cells edge weights in the stored dtype; a copy the caller owns
    ///     (R2), handed over as ``Matrix.to_scipy``'s. A stored edge may weigh 0
    ///     (format.md section 7).
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
    ///     accepted.
    /// MemoryLimitError
    ///     ``KY-E401``: the plan cannot run within ``memory_limit``.
    /// NotImplementedYetError
    ///     ``KY-E601``: the plan has a step the executor does not run yet.
    ///
    /// Examples
    /// --------
    /// >>> atlas = scan(store).log1p(layer="counts", into="log1p").pca(layer="log1p", into="X_pca", n_components=1, seed=0).neighbors("X_pca", n_neighbors=2)
    /// >>> graph = atlas.graph("distances").to_scipy()
    /// >>> graph.shape, graph.dtype
    /// ((2, 2), dtype('float32'))
    #[pyo3(signature = (*, memory_limit = None, threads = None, progress = None))]
    fn to_scipy<'py>(
        &self,
        py: Python<'py>,
        memory_limit: Option<&Bound<'_, PyAny>>,
        threads: Option<&Bound<'_, PyAny>>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let graph = self.read(py, "Graph.to_scipy", memory_limit, threads, progress)?;
        let n_cells = graph.n_cells();
        let kyto_store::Graph {
            indptr,
            indices,
            weights,
            ..
        } = graph;
        match weights {
            FloatValues::F32(data) => handoff::to_scipy(
                py,
                CsrMatrix {
                    n_cells,
                    n_genes: n_cells,
                    indptr,
                    indices,
                    data,
                },
            ),
            FloatValues::F64(data) => handoff::to_scipy(
                py,
                CsrMatrix {
                    n_cells,
                    n_genes: n_cells,
                    indptr,
                    indices,
                    data,
                },
            ),
        }
    }

    /// Run the plan and return the edges as a Polars ``DataFrame``.
    ///
    /// Parameters
    /// ----------
    /// memory_limit : str or int or None, default=None
    ///     The most memory running the plan may use, as ``Atlas.collect`` takes it;
    ///     the result is held in memory besides.
    /// threads : int or None, default=None
    ///     Threads that run the plan; ``None`` for one per CPU core. The result does
    ///     not depend on it (R11).
    /// progress : callable or None, default=None
    ///     Called as ``progress(done, total)`` on the calling thread while the plan
    ///     runs; never for a plain ``scan``, which runs nothing.
    ///
    /// Returns
    /// -------
    /// polars.DataFrame
    ///     One row per stored edge, row by row: ``source`` and ``target`` are cell
    ///     positions (``UInt32``, rows of ``cells.to_polars()`` and of ``to_scipy()``)
    ///     and ``weight`` the stored weight (``Float32`` or ``Float64``); a copy the
    ///     caller owns (R2), handed over as ``Table.to_polars``'s.
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
    ///     accepted.
    /// MemoryLimitError
    ///     ``KY-E401``: the plan cannot run within ``memory_limit``.
    /// NotImplementedYetError
    ///     ``KY-E601``: the plan has a step the executor does not run yet.
    ///
    /// Examples
    /// --------
    /// >>> atlas = scan(store).log1p(layer="counts", into="log1p").pca(layer="log1p", into="X_pca", n_components=1, seed=0).neighbors("X_pca", n_neighbors=2)
    /// >>> atlas.graph("distances").to_edges().columns
    /// ['source', 'target', 'weight']
    #[pyo3(signature = (*, memory_limit = None, threads = None, progress = None))]
    fn to_edges<'py>(
        &self,
        py: Python<'py>,
        memory_limit: Option<&Bound<'_, PyAny>>,
        threads: Option<&Bound<'_, PyAny>>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let graph = self.read(py, "Graph.to_edges", memory_limit, threads, progress)?;
        let frame = handoff::edges(graph).map_err(|error| to_py_err(py, &error))?;
        handoff::to_polars(py, frame)
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
