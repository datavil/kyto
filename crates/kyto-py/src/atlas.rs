//! The Python `Atlas`: an immutable lazy plan (task 3.2), and the private `scan` that
//! builds one.
//!
//! Does: wrap an `Arc<kyto_plan::Plan>` in a frozen pyclass whose every method returns a
//! new `Atlas` (R2); resolve an expression's names against the plan (task 3.5); build
//! the root plan from a store's `schema.json` and the `gene_id` column of its gene table;
//! hand out the read-only handles (task 3.6; the accessors are 7.3's, taken early);
//! filter genes (`filter_genes`, 4.1/7.2's gene filter taken early for task 3.8); write the
//! plan JSON (`to_json`) and read it back (`from_json`, task 4.3); run the plan to
//! nowhere (`_execute`, task 5.7's private hook: GIL, progress, cancel).
//! Does not: read other columns or the matrix; hold any state besides the plan (R7); expose
//! `__getitem__`, `.X`, `obs` or `var` (R4). The public `kyto.scan` and the chain
//! methods arrive with 7.1/7.2; `scan` is private until then.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

use kyto_core::GeneIds;
use kyto_error::KytoError;
use kyto_plan::{PhysicalOptions, Plan};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use pyo3::{IntoPyObjectExt, PyAny};

use crate::errors::to_py_err;
use crate::expr::{PyCellExpr, PyGeneExpr};
use crate::handles::{PyEmbedding, PyGraph, PyMatrix, PyModality, PyTable};
use crate::run::run_plan;

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

    /// Write the plan as JSON text, to store it or send it elsewhere.
    ///
    /// The text is versioned (``"version": 1``) and lists the plan's nodes from the
    /// scan up, each with its stable ID (its position from the scan, 0), the ID of its
    /// input, and its step: the plan method that built it and its arguments with every
    /// name resolved in full (R6). The scan holds the store's location, its schema and
    /// its gene IDs and symbols as read when the plan was built, so ``from_json`` needs
    /// no store. Equal plans write equal text, and ``from_json(atlas.to_json())``
    /// gives back an equal plan.
    ///
    /// Returns
    /// -------
    /// str
    ///     The plan JSON, on one line (a new string; the plan is unchanged). Nothing
    ///     is read.
    ///
    /// Examples
    /// --------
    /// >>> genes = gene_column("feature_type") == "Gene Expression"
    /// >>> atlas = scan(store).filter_genes(genes)
    /// >>> text = atlas.to_json()
    /// >>> text[:39]
    /// '{"format":"kyto-plan","version":1,"node'
    /// >>> print(from_json(text))
    /// FILTER_GENES gene_column('feature_type') == 'Gene Expression'
    /// SCAN "..." (2 cells x 3 genes)
    fn to_json(&self) -> String {
        self.plan.to_json()
    }

    /// How the plan would run: its logical and optimized trees (task 4.7).
    ///
    /// The logical tree lists the plan's steps from the last down to the scan, each with
    /// its node ID and its parameters, every name resolved in full as the plan stored it
    /// (R6: a ``feature`` or a gene read by symbol shows the gene ID and layer it
    /// names). The optimized tree is the plan optimized with every rule (for a terminal
    /// reading all of its output): one line for what the output reads, then one line per
    /// node from the top down to the scan, each with the logical nodes it came from
    /// (``(node 1, 2)`` for a fused pass) and, in brackets, how it runs: ``device``
    /// (``cpu``), ``passes`` over its input's blocks (with the ``statistics`` passes over
    /// all cells and the ``apply`` pass when there are statistics passes), and
    /// ``blocks=N cells`` (streamed in blocks of the store's cell block size), ``in
    /// memory`` or ``no data``. ``estimate()`` gives the cost of the whole plan. Nothing
    /// is read.
    ///
    /// Parameters
    /// ----------
    /// format : {"text", "json"}, default="text"
    ///     ``"text"``: the two trees as text, under ``logical plan`` and ``optimized
    ///     plan``. ``"json"``: the same as a dict with ``logical`` (one dict per step:
    ///     ``id``, ``name``, ``parameters``, ``input``) and ``optimized`` (``outputs``:
    ///     ``node``, ``reads``; ``nodes``: ``index``, ``name``, ``parameters``,
    ///     ``reads`` (a scan's, else ``None``), ``input``, ``origin``, ``device``,
    ///     ``passes``, ``statistics_passes``, ``apply_pass``, ``blocks`` (``"cells"``,
    ///     ``"in_memory"`` or ``"none"``) and ``block_cells``), in the text's order.
    ///
    /// Returns
    /// -------
    /// str or dict
    ///     A ``str`` for ``format="text"``, a ``dict`` for ``format="json"``; new
    ///     objects each call (the plan is unchanged).
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: ``format`` is not ``"text"`` or ``"json"``.
    ///
    /// Examples
    /// --------
    /// >>> genes = gene_column("feature_type") == "Gene Expression"
    /// >>> print(scan(store).filter_genes(genes).explain())
    /// logical plan
    ///   node 1 FILTER_GENES gene_column('feature_type') == 'Gene Expression' <- node 0
    ///   node 0 SCAN "..." (2 cells x 3 genes)
    /// optimized plan
    ///   output #1 reads [...]
    ///   #1 FILTER_GENES ... <- #0 (node 1) [device=cpu passes=1 blocks=4096 cells]
    ///   #0 SCAN ... (node 0) [device=cpu passes=1 blocks=4096 cells]
    /// >>> explained = scan(store).filter_genes(genes).explain(format="json")
    /// >>> [step["name"] for step in explained["logical"]]
    /// ['FILTER_GENES', 'SCAN']
    /// >>> explained["optimized"]["nodes"][0]["passes"]
    /// 1
    #[pyo3(signature = (*, format = "text"))]
    fn explain<'py>(&self, py: Python<'py>, format: &str) -> PyResult<Bound<'py, PyAny>> {
        match format {
            "text" => self.plan.explain().into_bound_py_any(py),
            "json" => py
                .import("json")?
                .call_method1("loads", (self.plan.explain_json(),)),
            _ => {
                let error = KytoError::InvalidArgument {
                    function: "explain".to_owned(),
                    parameter: "format".to_owned(),
                    value: PyString::new(py, format).repr()?.to_string(),
                    problem: "it is not a format explain writes".to_owned(),
                    accepted: "'text' or 'json'".to_owned(),
                };
                Err(to_py_err(py, &error))
            }
        }
    }

    /// What running the plan would cost, from its schema alone (task 4.6).
    ///
    /// Estimated for the plan as ``explain()`` shows it (every optimizer rule, a
    /// terminal reading all of its output, the CPU, 2 cell blocks in flight).
    /// ``passes`` counts the passes over the store's matrix: each statistics pass
    /// streams the data again from the store, and one more pass delivers what is read
    /// from the stream. ``bytes_read`` is the decoded size of the matrix tiles those
    /// passes read (fewer bytes are on disk, compressed; the cell and gene tables are
    /// not counted). ``peak_memory`` is the most memory the plan holds at once: the
    /// blocks in flight in every streamed step (sized for blocks of mean density) plus
    /// statistics and in-memory results; the store reader's part is within 25% of
    /// what it allocates on the test fixtures. Nothing is read.
    ///
    /// Returns
    /// -------
    /// CostEstimate
    ///     A new dict with ``passes``, ``bytes_read`` and ``peak_memory`` (bytes, as
    ///     ``int``) and ``device`` (``"cpu"``).
    ///
    /// Examples
    /// --------
    /// >>> scan(store).estimate()
    /// {'passes': 1, 'bytes_read': 16428, 'peak_memory': 1048592, 'device': 'cpu'}
    fn estimate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let estimate = self.plan.estimate();
        let dict = PyDict::new(py);
        dict.set_item("passes", estimate.passes())?;
        dict.set_item("bytes_read", estimate.bytes_read())?;
        dict.set_item("peak_memory", estimate.peak_memory())?;
        dict.set_item("device", estimate.device().name())?;
        Ok(dict)
    }

    /// Run the plan and stream its output to nowhere (private: task 5.7's hook until
    /// the terminals of 7.2 and 7.3 return the data).
    ///
    /// Runs the plan as ``estimate()`` costs it (every optimizer rule, the output read
    /// whole, 2 cell blocks in flight) on the CPU executor, drops every block it
    /// streams and returns how many cell rows it streamed. The GIL is released while
    /// the plan computes, so other Python threads run. Ctrl-C (or any exception a
    /// signal handler raises) cancels the run: it stops at its next block, its threads
    /// end, and the exception is raised from here; the process stays usable.
    ///
    /// Parameters
    /// ----------
    /// threads : int or None, default=None
    ///     Threads that read and compute blocks; ``None`` for one per CPU core. The
    ///     result does not depend on it (R11).
    /// progress : callable or None, default=None
    ///     Called as ``progress(done, total)`` on the calling thread whenever the run
    ///     moved: cell rows computed and in the whole run, both summed over its
    ///     passes (each statistics pass, then the output's stream, computes every
    ///     cell). The last call has ``done == total``; a store without cells gets
    ///     none. An exception it raises cancels the run and is raised from here.
    ///
    /// Returns
    /// -------
    /// int
    ///     The cell rows the output streamed (the plan's cells).
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: ``threads`` is below 1, or ``progress`` is not callable.
    /// NotImplementedYetError
    ///     ``KY-E601``: the plan has a step the executor does not run yet.
    /// KeyboardInterrupt
    ///     Ctrl-C while the plan computes; the run has stopped.
    ///
    /// Examples
    /// --------
    /// >>> calls = []
    /// >>> scan(store)._execute(threads=1, progress=lambda done, total: calls.append((done, total)))
    /// 2
    /// >>> calls[-1]
    /// (2, 2)
    #[pyo3(signature = (*, threads = None, progress = None))]
    fn _execute(
        &self,
        py: Python<'_>,
        threads: Option<i64>,
        progress: Option<Bound<'_, PyAny>>,
    ) -> PyResult<u64> {
        let invalid = |parameter: &str, value: String, problem: &str, accepted: &str| {
            let error = KytoError::InvalidArgument {
                function: "Atlas._execute".to_owned(),
                parameter: parameter.to_owned(),
                value,
                problem: problem.to_owned(),
                accepted: accepted.to_owned(),
            };
            to_py_err(py, &error)
        };
        let threads = match threads {
            None => std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN),
            Some(threads) => usize::try_from(threads)
                .ok()
                .and_then(NonZeroUsize::new)
                .ok_or_else(|| {
                    invalid(
                        "threads",
                        threads.to_string(),
                        "a run needs at least one thread",
                        "an int of 1 or more, or None for one per CPU core",
                    )
                })?,
        };
        if let Some(progress) = &progress
            && !progress.is_callable()
        {
            let value = progress.repr()?.to_string();
            return Err(invalid(
                "progress",
                value,
                "it is not callable",
                "a callable taking (done, total), or None",
            ));
        }
        let physical = self.plan.physical(PhysicalOptions::default());
        run_plan(py, &physical, threads, progress.as_ref())
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
pub(crate) fn wrong_type(operation: &str, accepted: &str, value: &Bound<'_, PyAny>) -> KytoError {
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

/// Read a plan from the JSON text that ``Atlas.to_json`` wrote.
///
/// The plan is rebuilt step by step with the same checks as when it was first built,
/// against the store snapshot the text holds (schema, gene IDs and symbols). No store
/// is opened or read; the store is checked when the plan runs.
///
/// Parameters
/// ----------
/// text : str
///     A plan JSON, as ``Atlas.to_json`` writes it.
///
/// Returns
/// -------
/// Atlas
///     A new plan, equal to the one that wrote ``text``.
///
/// Raises
/// ------
/// PlanError
///     ``KY-E217``: ``text`` is not JSON, not a plan JSON, or breaks its format (the
///     message names the line and column, or the node).
///     ``KY-E218``: the text has a ``version`` this kyto does not read.
/// KytoError
///     Any error a plan method raises for its arguments, when the text was edited
///     into a plan that does not build (such as ``KY-E109`` for a missing layer).
///
/// Examples
/// --------
/// >>> text = (
/// ...     '{"format":"kyto-plan","version":1,"nodes":[{"id":0,"input":null,"step":'
/// ...     '{"method":"scan","source":"pbmc.kyto","schema":{"n_cells":2,"n_genes":1,'
/// ...     '"tiling":{"cell_block_size":4096,"gene_block_size":1024},'
/// ...     '"cells":{"columns":[{"name":"cell_id","dtype":"String"}]},'
/// ...     '"genes":{"columns":[{"name":"gene_id","dtype":"String"}]},'
/// ...     '"layers":[{"name":"counts","dtype":"uint32","nnz":2}],'
/// ...     '"embeddings":[],"graphs":[]},"gene_ids":["CD3E"],"symbols":null}}]}'
/// ... )
/// >>> print(from_json(text))
/// SCAN "pbmc.kyto" (2 cells x 1 genes)
/// >>> from_json('{"format":"kyto-plan","version":2,"nodes":[]}')
/// Traceback (most recent call last):
///     ...
/// kyto.errors.PlanError: KY-E218: the plan JSON has version 2, but this kyto reads only version 1...
#[pyfunction]
#[pyo3(signature = (text, /))]
pub fn from_json(py: Python<'_>, text: &str) -> PyResult<Atlas> {
    py.detach(|| Plan::from_json(text))
        .map(|plan| Atlas { plan })
        .map_err(|error| to_py_err(py, &error))
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
            let mut gene_ids = GeneIds::new(gene_ids);
            // Symbols for `by='symbol'` (task 3.9): read only when the gene table has a
            // string `symbol` column; otherwise the resolver says why there are none.
            let has_symbols = schema.genes.columns.iter().any(|column| {
                column.name == kyto_core::SYMBOL_COLUMN
                    && matches!(column.dtype, kyto_core::ColumnDtype::Enum { .. })
            });
            if has_symbols {
                gene_ids = gene_ids.with_symbols(kyto_store::read_symbols(&store, schema.n_genes)?);
            }
            Ok::<_, kyto_error::KytoError>((schema, gene_ids))
        })
        .map_err(|error| to_py_err(py, &error))?;
    Ok(Atlas {
        plan: Plan::scan(store.display().to_string(), schema, gene_ids),
    })
}
