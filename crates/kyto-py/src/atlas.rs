//! The Python `Atlas`: an immutable lazy plan (task 3.2), and the private `scan` that
//! builds one.
//!
//! Does: wrap an `Arc<kyto_plan::Plan>` in a frozen pyclass whose every method returns a
//! new `Atlas` (R2); resolve an expression's names against the plan (task 3.5); build
//! the root plan from a store's `schema.json` and the `gene_id` column of its gene table;
//! hand out the read-only handles (task 3.6; the accessors are 7.3's, taken early);
//! the chain methods (task 7.2: filters, selections, columns, the algorithms of 6.x, and
//! the terminals `collect` and `sink_zarr`, and `sink_h5ad`, task 7.3); write the
//! plan JSON (`to_json`) and read it back (`from_json`, task 4.3); run the plan to
//! nowhere (`_execute`, task 5.7's private hook: GIL, progress, cancel).
//! Does not: read the matrix outside the terminals (R1); hold any state besides the plan
//! and the temporary store a `collect()` wrote (R7); expose `__getitem__`, `.X`, `obs` or
//! `var` (R4). `scan` is exported as `kyto.scan` (7.1).

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

use kyto_algo::RankGenesGroups;
use kyto_core::GeneIds;
use kyto_error::KytoError;
use kyto_plan::{Cluster, ClusterMethod, Neighbors, Plan, Umap};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use pyo3::{IntoPyObjectExt, PyAny};

use crate::errors::to_py_err;
use crate::expr::{PyCellExpr, PyGeneExpr};
use crate::handles::{PyEmbedding, PyGraph, PyMatrix, PyModality, PyTable};
use crate::run::run_plan;
use crate::{chain, rank_genes, terminal};

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
    /// The temporary store a `collect()` wrote and this plan scans: deleted when the
    /// last `Atlas` holding it is dropped. `None` for any other plan.
    collected: Option<Arc<tempfile::TempDir>>,
}

impl Atlas {
    /// A new `Atlas` with `plan`, built on this one (it keeps the collected store
    /// alive), or the plan's error as a Python exception.
    fn derive(&self, py: Python<'_>, plan: Result<Arc<Plan>, KytoError>) -> PyResult<Self> {
        plan.map(|plan| Self {
            plan,
            collected: self.collected.clone(),
        })
        .map_err(|error| to_py_err(py, &error))
    }

    /// `leiden` and `louvain`: one clustering node.
    fn cluster(
        &self,
        py: Python<'_>,
        method: ClusterMethod,
        graph: &str,
        resolution: Option<&Bound<'_, PyAny>>,
        seed: &Bound<'_, PyAny>,
        into: &str,
    ) -> PyResult<Self> {
        let name = method.name();
        let parameters = Cluster {
            method,
            graph: graph.to_owned(),
            resolution: chain::optional_float(py, name, "resolution", resolution)?.unwrap_or(1.0),
            seed: chain::integer(py, name, "seed", seed, 0)?,
            into: into.to_owned(),
        };
        self.derive(py, self.plan.cluster(parameters))
    }
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
        self.derive(py, self.plan.with_default_layer(layer))
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
    ///     ``lineage``) is read after a gene ``filter``; read
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

    /// Keep the cells or the genes where a boolean expression is true.
    ///
    /// A ``CellExpr`` filters cells, a ``GeneExpr`` filters genes (R5). The predicate is
    /// resolved against this plan now (R6): its names are checked and stored in full.
    /// Which cells or genes it keeps is known only when the plan runs. After a gene
    /// filter, a derived cell column that the store computed over all genes (one whose
    /// ``schema.json`` entry records ``lineage``, such as a stored ``n_genes``) is stale:
    /// reading it raises ``KY-E301``; read ``cell_column(name).recompute()`` instead.
    ///
    /// Parameters
    /// ----------
    /// predicate : CellExpr[Boolean] or GeneExpr[Boolean]
    ///     One value per cell or gene; where it is ``True`` the cell or gene is kept,
    ///     where it is ``False`` or null it is dropped. A column read (``Unknown`` until
    ///     resolved) must be stored as ``Boolean``.
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// ExpressionError
    ///     ``KY-E205``: ``predicate`` is not a ``CellExpr`` or a ``GeneExpr``, or not
    ///     boolean once resolved.
    /// ResolutionError
    ///     ``KY-E210``: a name in ``predicate`` matches nothing, and the other
    ///     errors of ``resolve``.
    ///
    /// Examples
    /// --------
    /// >>> atlas = scan(store)
    /// >>> print(atlas.filter(gene_column("feature_type") == "Gene Expression"))
    /// FILTER_GENES gene_column('feature_type') == 'Gene Expression'
    /// SCAN "..." (2 cells x 3 genes)
    /// >>> print(atlas.filter(total_counts(layer="counts") >= 6))
    /// FILTER_CELLS total_counts(layer='counts') >= 6
    /// SCAN "..." (2 cells x 3 genes)
    #[pyo3(signature = (predicate, /))]
    fn filter(&self, py: Python<'_>, predicate: &Bound<'_, PyAny>) -> PyResult<Self> {
        let plan = if let Ok(predicate) = predicate.cast::<PyCellExpr>() {
            self.plan.filter_cells(predicate.get().core())
        } else if let Ok(predicate) = predicate.cast::<PyGeneExpr>() {
            self.plan.filter_genes(predicate.get().core())
        } else {
            Err(wrong_type(
                "filter",
                "a boolean CellExpr or GeneExpr",
                predicate,
            ))
        };
        self.derive(py, plan)
    }

    /// Keep the given genes, in the given order.
    ///
    /// Parameters
    /// ----------
    /// gene_ids : list of str
    ///     Gene IDs (the store's ``gene_id`` column, not symbols), each once.
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// ResolutionError
    ///     ``KY-E210``: a gene ID is not in the store, with the closest IDs.
    /// RemovedFeatureError
    ///     ``KY-E203``: a gene was removed earlier in the plan.
    /// PlanError
    ///     ``KY-E215``: a gene ID is listed twice.
    ///
    /// Examples
    /// --------
    /// >>> print(scan(store).select_genes(["ENSG03", "ENSG01"]))
    /// SELECT_GENES ['ENSG03', 'ENSG01']
    /// SCAN "..." (2 cells x 3 genes)
    #[pyo3(signature = (gene_ids, /))]
    fn select_genes(&self, py: Python<'_>, gene_ids: &Bound<'_, PyAny>) -> PyResult<Self> {
        let gene_ids = chain::strings(py, "select_genes", "gene_ids", gene_ids)?;
        self.derive(py, self.plan.select_genes(gene_ids))
    }

    /// Keep the given cells.
    ///
    /// The cells stream in the store's order, not in the order given (deviations.md).
    /// Whether every ID names a cell is checked when the plan runs (the cell IDs are
    /// not read when the plan is built).
    ///
    /// Parameters
    /// ----------
    /// cell_ids : list of str
    ///     Cell IDs (the store's ``cell_id`` column), each once.
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// ResolutionError
    ///     ``KY-E210``: a cell ID is empty (here), or names no cell of the plan (when
    ///     it runs).
    /// PlanError
    ///     ``KY-E215``: a cell ID is listed twice.
    ///
    /// Examples
    /// --------
    /// >>> print(scan(store).take_cells(["AAAG-1"]))
    /// TAKE_CELLS ['AAAG-1']
    /// SCAN "..." (2 cells x 3 genes)
    #[pyo3(signature = (cell_ids, /))]
    fn take_cells(&self, py: Python<'_>, cell_ids: &Bound<'_, PyAny>) -> PyResult<Self> {
        let cell_ids = chain::strings(py, "take_cells", "cell_ids", cell_ids)?;
        self.derive(py, self.plan.take_cells(cell_ids))
    }

    /// Add or replace cell columns, each computed from a cell expression.
    ///
    /// The expressions are resolved against this plan now (R6) and computed when the
    /// plan runs. A name equal to a stored column replaces it.
    ///
    /// Parameters
    /// ----------
    /// **columns : CellExpr
    ///     The new columns, by name.
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// ExpressionError
    ///     ``KY-E205``: a value is not a ``CellExpr``.
    /// ResolutionError
    ///     ``KY-E210``: a name in an expression matches nothing, and the other errors
    ///     of ``resolve``.
    /// PlanError
    ///     ``KY-E215``: no column is given.
    ///
    /// Examples
    /// --------
    /// >>> print(scan(store).with_cells(n_counts=total_counts(layer="counts")))
    /// WITH_CELLS "n_counts" = total_counts(layer='counts')
    /// SCAN "..." (2 cells x 3 genes)
    #[pyo3(signature = (**columns))]
    fn with_cells(&self, py: Python<'_>, columns: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut named = Vec::new();
        for (name, value) in columns.into_iter().flatten() {
            let Ok(expression) = value.cast::<PyCellExpr>() else {
                let error = wrong_type("with_cells", "a CellExpr", &value);
                return Err(to_py_err(py, &error));
            };
            named.push((name.extract::<String>()?, expression.get().core().clone()));
        }
        if named.is_empty() {
            return Err(to_py_err(py, &no_columns("with_cells")));
        }
        self.derive(py, self.plan.with_cells(named))
    }

    /// Add or replace gene columns, each computed from a gene expression.
    ///
    /// The expressions are resolved against this plan now (R6) and computed when the
    /// plan runs. A name equal to a stored column replaces it.
    ///
    /// Parameters
    /// ----------
    /// **columns : GeneExpr
    ///     The new columns, by name.
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// ExpressionError
    ///     ``KY-E205``: a value is not a ``GeneExpr``.
    /// ResolutionError
    ///     ``KY-E210``: a name in an expression matches nothing, and the other errors
    ///     of ``resolve``.
    /// PlanError
    ///     ``KY-E215``: no column is given.
    ///
    /// Examples
    /// --------
    /// >>> print(scan(store).with_genes(is_cd3e=gene_column("symbol") == "CD3E"))
    /// WITH_GENES "is_cd3e" = gene_column('symbol') == 'CD3E'
    /// SCAN "..." (2 cells x 3 genes)
    #[pyo3(signature = (**columns))]
    fn with_genes(&self, py: Python<'_>, columns: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut named = Vec::new();
        for (name, value) in columns.into_iter().flatten() {
            let Ok(expression) = value.cast::<PyGeneExpr>() else {
                let error = wrong_type("with_genes", "a GeneExpr", &value);
                return Err(to_py_err(py, &error));
            };
            named.push((name.extract::<String>()?, expression.get().core().clone()));
        }
        if named.is_empty() {
            return Err(to_py_err(py, &no_columns("with_genes")));
        }
        self.derive(py, self.plan.with_genes(named))
    }

    /// Add a cell embedding computed outside kyto (not implemented yet).
    ///
    /// Arrives with task 6.11 (interop hooks: scvi-tools and Harmony outputs).
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Name of the new embedding.
    /// values : numpy.ndarray
    ///     One row per cell of the plan, in ``cell_ids()`` order.
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged.
    ///
    /// Raises
    /// ------
    /// NotImplementedYetError
    ///     ``KY-E601``: always, until task 6.11.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).with_embedding("X_scvi", None)
    /// Traceback (most recent call last):
    /// ...
    /// kyto.errors.NotImplementedYetError: KY-E601: 'Atlas.with_embedding' is not implemented yet...
    #[pyo3(signature = (name, values, /))]
    fn with_embedding(
        &self,
        py: Python<'_>,
        name: &str,
        values: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let _ = (name, values);
        let error = KytoError::NotImplementedYet {
            method: "Atlas.with_embedding".to_owned(),
            task: "6.11".to_owned(),
        };
        Err(to_py_err(py, &error))
    }

    /// Scale every cell to the same total count (scanpy's ``normalize_total``).
    ///
    /// Each cell's values are multiplied by ``target_sum / total``, its total over the
    /// genes of the plan; a cell with total 0 stays 0. Writes ``float32``.
    ///
    /// Parameters
    /// ----------
    /// target_sum : float or None, default=None
    ///     The total of every cell afterwards; ``None`` for the median of the cells'
    ///     totals (one more pass over the cells).
    /// layer : str or None, default=None
    ///     The layer read; ``None`` for the default layer.
    /// into : str
    ///     The layer written (a new name, or an existing layer to replace).
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: ``target_sum`` is not a positive finite ``float32``.
    /// ResolutionError
    ///     ``KY-E211``: ``layer`` is ``None`` and the plan has no default layer.
    /// MissingMemberError
    ///     ``KY-E109``: the layer is not in the plan.
    /// StoreInputError
    ///     ``KY-E102``: ``into`` is not a valid layer name.
    ///
    /// Examples
    /// --------
    /// >>> atlas = scan(store).normalize_total(1e4, layer="counts", into="normalized")
    /// >>> print(atlas)
    /// NORMALIZE_TOTAL "counts" -> "normalized" target_sum=10000
    /// SCAN "..." (2 cells x 3 genes)
    #[pyo3(signature = (target_sum = None, *, layer = None, into))]
    fn normalize_total(
        &self,
        py: Python<'_>,
        target_sum: Option<&Bound<'_, PyAny>>,
        layer: Option<&str>,
        into: &str,
    ) -> PyResult<Self> {
        let target_sum = chain::optional_float(py, "normalize_total", "target_sum", target_sum)?;
        self.derive(py, self.plan.normalize_total(target_sum, layer, into))
    }

    /// Take ``log(1 + x)`` of every value (natural log, scanpy's ``log1p``).
    ///
    /// Writes ``float32``. Absent values stay absent (``log1p(0) == 0``).
    ///
    /// Parameters
    /// ----------
    /// layer : str or None, default=None
    ///     The layer read; ``None`` for the default layer.
    /// into : str
    ///     The layer written (a new name, or an existing layer to replace).
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// ResolutionError
    ///     ``KY-E211``: ``layer`` is ``None`` and the plan has no default layer.
    /// MissingMemberError
    ///     ``KY-E109``: the layer is not in the plan.
    ///
    /// Examples
    /// --------
    /// >>> print(scan(store).with_default_layer("counts").log1p(into="log1p"))
    /// LOG1P "counts" -> "log1p"
    /// WITH_DEFAULT_LAYER "counts"
    /// SCAN "..." (2 cells x 3 genes)
    #[pyo3(signature = (*, layer = None, into))]
    fn log1p(&self, py: Python<'_>, layer: Option<&str>, into: &str) -> PyResult<Self> {
        self.derive(py, self.plan.log1p(layer, into))
    }

    /// Scale every gene to mean 0 and variance 1 (scanpy's ``scale``).
    ///
    /// Centring stores every gene of every cell (``explain()`` notes it; PCA centres
    /// sparse data implicitly). Writes ``float32``.
    ///
    /// Parameters
    /// ----------
    /// layer : str or None, default=None
    ///     The layer read; ``None`` for the default layer.
    /// into : str
    ///     The layer written (a new name, or an existing layer to replace).
    /// zero_center : bool, default=True
    ///     Subtract each gene's mean; ``False`` only divides by the standard deviation
    ///     (and keeps the data sparse).
    /// max_value : float or None, default=None
    ///     Clip the scaled values to ``[-max_value, max_value]`` (``[.., max_value]``
    ///     without centring); ``None`` does not clip.
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: ``max_value`` is not positive.
    /// ResolutionError
    ///     ``KY-E211``: ``layer`` is ``None`` and the plan has no default layer.
    /// MissingMemberError
    ///     ``KY-E109``: the layer is not in the plan.
    ///
    /// Examples
    /// --------
    /// >>> print(scan(store).scale(layer="counts", into="scaled", max_value=10))
    /// SCALE "counts" -> "scaled" zero_center=true max_value=10
    /// SCAN "..." (2 cells x 3 genes)
    #[pyo3(signature = (*, layer = None, into, zero_center = true, max_value = None))]
    fn scale(
        &self,
        py: Python<'_>,
        layer: Option<&str>,
        into: &str,
        zero_center: bool,
        max_value: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let max_value = chain::optional_float(py, "scale", "max_value", max_value)?;
        self.derive(py, self.plan.scale(layer, into, zero_center, max_value))
    }

    /// Mark the highly variable genes (scanpy's ``highly_variable_genes``).
    ///
    /// Adds scanpy's gene columns: ``highly_variable`` (``Boolean``), ``means``,
    /// ``dispersions`` and ``dispersions_norm`` (``seurat``, ``cell_ranger``) or
    /// ``means``, ``variances``, ``variances_norm`` and ``highly_variable_rank``
    /// (``seurat_v3``), all ``Float64``, and with ``batch_key``
    /// ``highly_variable_nbatches`` (``Int64``) and ``highly_variable_intersection``.
    /// Keep the genes with ``filter(gene_column("highly_variable"))``.
    ///
    /// Parameters
    /// ----------
    /// layer : str or None, default=None
    ///     The layer read (log data for ``seurat`` and ``cell_ranger``, counts for
    ///     ``seurat_v3``); ``None`` for the default layer.
    /// flavor : {"seurat", "cell_ranger", "seurat_v3"}, default="seurat"
    ///     The method, as scanpy's.
    /// n_top_genes : int or None, default=None
    ///     Mark this many genes; ``None`` uses scanpy's dispersion cutoffs (not with
    ///     ``seurat_v3``, which needs it).
    /// batch_key : str or None, default=None
    ///     A cell column: genes are ranked in each batch and combined as scanpy does.
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: an unknown ``flavor``, ``n_top_genes`` below 1, or
    ///     ``seurat_v3`` without ``n_top_genes``.
    /// ResolutionError
    ///     ``KY-E211``: ``layer`` is ``None`` and the plan has no default layer.
    /// MissingMemberError
    ///     ``KY-E109``: the layer or ``batch_key`` is not in the plan.
    ///
    /// Examples
    /// --------
    /// >>> atlas = scan(store).highly_variable_genes(layer="counts", n_top_genes=2)
    /// >>> print(atlas)
    /// HIGHLY_VARIABLE_GENES "counts" flavor=seurat n_top_genes=2
    /// SCAN "..." (2 cells x 3 genes)
    #[pyo3(signature = (*, layer = None, flavor = "seurat", n_top_genes = None, batch_key = None))]
    fn highly_variable_genes(
        &self,
        py: Python<'_>,
        layer: Option<&str>,
        flavor: &str,
        n_top_genes: Option<&Bound<'_, PyAny>>,
        batch_key: Option<&str>,
    ) -> PyResult<Self> {
        let n_top_genes =
            chain::optional_integer(py, "highly_variable_genes", "n_top_genes", n_top_genes, 1)?;
        let plan = chain::hvg_flavor(flavor).and_then(|flavor| {
            self.plan
                .highly_variable_genes(layer, flavor, n_top_genes, batch_key)
        });
        self.derive(py, plan)
    }

    /// Principal components of a layer (scanpy's ``pca``, implicit centring).
    ///
    /// Streams the layer in ``2 + PCA_POWER_ITERATIONS`` passes (a randomized block
    /// Krylov method) and never densifies it. Adds the cell embedding ``into`` (one row
    /// per cell, ``n_components`` wide) and the gene embedding ``into + "_loadings"``.
    ///
    /// Parameters
    /// ----------
    /// layer : str or None, default=None
    ///     The layer read; ``None`` for the default layer.
    /// into : str
    ///     Name of the cell embedding written.
    /// n_components : int, default=50
    ///     Components computed.
    /// seed : int
    ///     Seed of the random start (R7, R11: the same seed gives the same result).
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: ``n_components`` is below 1 or above the cells or genes, or
    ///     ``seed`` is negative.
    /// ResolutionError
    ///     ``KY-E211``: ``layer`` is ``None`` and the plan has no default layer.
    /// MissingMemberError
    ///     ``KY-E109``: the layer is not in the plan.
    ///
    /// Examples
    /// --------
    /// >>> print(scan(store).pca(layer="counts", into="pca", n_components=1, seed=0))
    /// PCA "counts" -> "pca" n_components=1 seed=0
    /// SCAN "..." (2 cells x 3 genes)
    #[pyo3(signature = (*, layer = None, into, n_components = None, seed))]
    fn pca(
        &self,
        py: Python<'_>,
        layer: Option<&str>,
        into: &str,
        n_components: Option<&Bound<'_, PyAny>>,
        seed: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let n_components =
            chain::optional_integer(py, "pca", "n_components", n_components, 1)?.unwrap_or(50);
        let seed = chain::integer(py, "pca", "seed", seed, 0)?;
        self.derive(py, self.plan.pca(layer, into, n_components, seed))
    }

    /// The k nearest neighbours of every cell in a cell embedding (scanpy's
    /// ``neighbors``).
    ///
    /// Adds the cell graph of distances, ``"distances"`` (or ``into + "_distances"``),
    /// and declares the connectivities graph, ``"connectivities"`` (or
    /// ``into + "_connectivities"``).
    ///
    /// Parameters
    /// ----------
    /// embedding : str
    ///     The cell embedding searched, such as the ``into`` of ``pca``.
    /// n_neighbors : int, default=15
    ///     Neighbours per cell, the cell itself included (scanpy's meaning).
    /// n_components : int or None, default=None
    ///     Use the first this many columns of the embedding; ``None`` uses all.
    /// metric : {"euclidean", "cosine"}, default="euclidean"
    ///     The distance.
    /// search : {"exact", "approximate"}, default="exact"
    ///     Exact blocked search, or HNSW (approximate, faster on large atlases).
    /// seed : int, default=0
    ///     Seed of the approximate search (R7, R11).
    /// into : str or None, default=None
    ///     Prefix of the graph names; ``None`` for scanpy's names.
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: ``n_neighbors`` below 2 or above the cells, ``n_components``
    ///     out of range, a negative ``seed``, or an unknown ``metric`` or ``search``.
    /// MissingMemberError
    ///     ``KY-E109``: ``embedding`` is not a cell embedding of the plan.
    ///
    /// Examples
    /// --------
    /// >>> atlas = scan(store).pca(layer="counts", into="pca", n_components=1, seed=0)
    /// >>> print(atlas.neighbors("pca", n_neighbors=2))
    /// NEIGHBORS "pca" -> "distances", "connectivities" n_neighbors=2 metric=Euclidean search=Exact seed=0
    /// PCA "counts" -> "pca" n_components=1 seed=0
    /// SCAN "..." (2 cells x 3 genes)
    #[pyo3(signature = (
        embedding,
        /,
        *,
        n_neighbors = None,
        n_components = None,
        metric = "euclidean",
        search = "exact",
        seed = None,
        into = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn neighbors(
        &self,
        py: Python<'_>,
        embedding: &str,
        n_neighbors: Option<&Bound<'_, PyAny>>,
        n_components: Option<&Bound<'_, PyAny>>,
        metric: &str,
        search: &str,
        seed: Option<&Bound<'_, PyAny>>,
        into: Option<&str>,
    ) -> PyResult<Self> {
        let n_neighbors =
            chain::optional_integer(py, "neighbors", "n_neighbors", n_neighbors, 0)?.unwrap_or(15);
        let n_components =
            chain::optional_integer(py, "neighbors", "n_components", n_components, 1)?;
        let seed = chain::optional_integer(py, "neighbors", "seed", seed, 0)?.unwrap_or(0);
        let parameters = (|| {
            Ok::<_, KytoError>(Neighbors {
                embedding: embedding.to_owned(),
                n_components,
                n_neighbors,
                metric: chain::metric(metric)?,
                search: chain::search(search)?,
                seed,
                key: into.map(str::to_owned),
            })
        })();
        self.derive(
            py,
            parameters.and_then(|parameters| self.plan.neighbors(parameters)),
        )
    }

    /// Cluster the cells of a cell graph with Leiden (scanpy's ``leiden``).
    ///
    /// Adds the cell column ``into``: one cluster label per cell.
    ///
    /// Parameters
    /// ----------
    /// graph : str, default="connectivities"
    ///     The cell graph clustered, such as the connectivities ``neighbors`` declares.
    /// resolution : float, default=1.0
    ///     Higher values give more, smaller clusters.
    /// seed : int
    ///     Seed of the random steps (R7, R11: the same seed gives the same clusters).
    /// into : str
    ///     The cell column written.
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: ``resolution`` is not positive, ``seed`` is not an int of 0 or
    ///     more, or ``into`` is not a valid column name.
    /// MissingMemberError
    ///     ``KY-E109``: ``graph`` is not a cell graph of the plan.
    ///
    /// Examples
    /// --------
    /// >>> atlas = scan(store).pca(layer="counts", into="pca", n_components=1, seed=0)
    /// >>> atlas = atlas.neighbors("pca", n_neighbors=2).leiden(into="leiden", seed=0)
    /// >>> print(str(atlas).splitlines()[0])
    /// LEIDEN "connectivities" -> "leiden" resolution=1 seed=0
    #[pyo3(signature = (*, graph = "connectivities", resolution = None, seed, into))]
    fn leiden(
        &self,
        py: Python<'_>,
        graph: &str,
        resolution: Option<&Bound<'_, PyAny>>,
        seed: &Bound<'_, PyAny>,
        into: &str,
    ) -> PyResult<Self> {
        self.cluster(py, ClusterMethod::Leiden, graph, resolution, seed, into)
    }

    /// Cluster the cells of a cell graph with Louvain (scanpy's ``louvain``).
    ///
    /// Adds the cell column ``into``: one cluster label per cell.
    ///
    /// Parameters
    /// ----------
    /// graph : str, default="connectivities"
    ///     The cell graph clustered, such as the connectivities ``neighbors`` declares.
    /// resolution : float, default=1.0
    ///     Higher values give more, smaller clusters.
    /// seed : int
    ///     Seed of the random steps (R7, R11: the same seed gives the same clusters).
    /// into : str
    ///     The cell column written.
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: ``resolution`` is not positive, ``seed`` is not an int of 0 or
    ///     more, or ``into`` is not a valid column name.
    /// MissingMemberError
    ///     ``KY-E109``: ``graph`` is not a cell graph of the plan.
    ///
    /// Examples
    /// --------
    /// >>> atlas = scan(store).pca(layer="counts", into="pca", n_components=1, seed=0)
    /// >>> atlas = atlas.neighbors("pca", n_neighbors=2).louvain(into="louvain", seed=0)
    /// >>> print(str(atlas).splitlines()[0])
    /// LOUVAIN "connectivities" -> "louvain" resolution=1 seed=0
    #[pyo3(signature = (*, graph = "connectivities", resolution = None, seed, into))]
    fn louvain(
        &self,
        py: Python<'_>,
        graph: &str,
        resolution: Option<&Bound<'_, PyAny>>,
        seed: &Bound<'_, PyAny>,
        into: &str,
    ) -> PyResult<Self> {
        self.cluster(py, ClusterMethod::Louvain, graph, resolution, seed, into)
    }

    /// Lay the cells of a cell graph out in a few dimensions (scanpy's ``umap``).
    ///
    /// Adds the cell embedding ``into`` (``float32``, ``n_components`` wide).
    ///
    /// Parameters
    /// ----------
    /// graph : str, default="connectivities"
    ///     The cell graph laid out, such as the connectivities ``neighbors`` declares.
    /// into : str
    ///     The cell embedding written.
    /// n_components : int, default=2
    ///     Dimensions of the layout (1 to 100).
    /// min_dist : float, default=0.5
    ///     The smallest distance between points of the layout (at most ``spread``).
    /// spread : float, default=1.0
    ///     The scale of the layout's distances.
    /// n_epochs : int or None, default=None
    ///     Epochs of the optimisation; ``None`` for scanpy's default.
    /// alpha : float, default=1.0
    ///     The initial learning rate.
    /// gamma : float, default=1.0
    ///     The weight of the negative samples (repulsion).
    /// negative_sample_rate : int, default=5
    ///     Negative samples per positive sample.
    /// init : {"spectral", "random"}, default="spectral"
    ///     Where the layout starts.
    /// seed : int
    ///     Seed of the random steps (R7, R11).
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: an argument out of range (``n_components`` from 1 to 100,
    ///     positive ``spread`` and ``alpha``, ``min_dist`` from 0 to ``spread``), an
    ///     unknown ``init``, or an ``into`` that is not a valid embedding name.
    /// MissingMemberError
    ///     ``KY-E109``: ``graph`` is not a cell graph of the plan.
    ///
    /// Examples
    /// --------
    /// >>> atlas = scan(store).pca(layer="counts", into="pca", n_components=1, seed=0)
    /// >>> atlas = atlas.neighbors("pca", n_neighbors=2).umap(into="X_umap", seed=0)
    /// >>> print(str(atlas).splitlines()[0])
    /// UMAP "connectivities" -> "X_umap" n_components=2 min_dist=0.5 spread=1 n_epochs=auto alpha=1 gamma=1 negative_sample_rate=5 init=Spectral seed=0
    #[pyo3(signature = (
        *,
        graph = "connectivities",
        into,
        n_components = None,
        min_dist = None,
        spread = None,
        n_epochs = None,
        alpha = None,
        gamma = None,
        negative_sample_rate = None,
        init = "spectral",
        seed
    ))]
    #[allow(clippy::too_many_arguments)]
    fn umap(
        &self,
        py: Python<'_>,
        graph: &str,
        into: &str,
        n_components: Option<&Bound<'_, PyAny>>,
        min_dist: Option<&Bound<'_, PyAny>>,
        spread: Option<&Bound<'_, PyAny>>,
        n_epochs: Option<&Bound<'_, PyAny>>,
        alpha: Option<&Bound<'_, PyAny>>,
        gamma: Option<&Bound<'_, PyAny>>,
        negative_sample_rate: Option<&Bound<'_, PyAny>>,
        init: &str,
        seed: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let defaults = Umap::default();
        let float = |parameter: &str, value, default: f64| {
            chain::optional_float(py, "umap", parameter, value)
                .map(|value| value.unwrap_or(default))
        };
        let parameters = Umap {
            graph: graph.to_owned(),
            n_components: chain::optional_integer(py, "umap", "n_components", n_components, 1)?
                .unwrap_or(defaults.n_components),
            min_dist: float("min_dist", min_dist, defaults.min_dist)?,
            spread: float("spread", spread, defaults.spread)?,
            n_epochs: chain::optional_integer(py, "umap", "n_epochs", n_epochs, 0)?,
            alpha: float("alpha", alpha, defaults.alpha)?,
            gamma: float("gamma", gamma, defaults.gamma)?,
            negative_sample_rate: chain::optional_integer(
                py,
                "umap",
                "negative_sample_rate",
                negative_sample_rate,
                0,
            )?
            .unwrap_or(defaults.negative_sample_rate),
            init: chain::umap_init(init).map_err(|error| to_py_err(py, &error))?,
            seed: chain::integer(py, "umap", "seed", seed, 0)?,
            into: into.to_owned(),
        };
        self.derive(py, self.plan.umap(parameters))
    }

    /// Rank the genes of each group by a differential expression test (scanpy's
    /// ``rank_genes_groups`` then ``rank_genes_groups_df``).
    ///
    /// The arguments and the plan's layer and ``groupby`` column are checked now; the
    /// plan runs only when the returned frame is collected (R1), with the GIL released.
    /// One row per (group, gene): groups in order (an ``Enum`` column's categories,
    /// numbers by value, else text sorted; or as ``groups`` lists them), each group's
    /// genes by statistic, descending (ties in gene order). Cells with a null label are
    /// in no group and in every rest.
    ///
    /// Parameters
    /// ----------
    /// groupby : str
    ///     The cell column whose values are the groups (stored, or added by
    ///     ``with_cells``).
    /// method : {"t-test", "wilcoxon"}, default="t-test"
    ///     Welch's t-test, or Wilcoxon's rank-sum test (normal approximation).
    /// layer : str or None, default=None
    ///     The layer tested (log data, as scanpy expects); ``None`` for the default
    ///     layer.
    /// groups : list of str or None, default=None
    ///     The groups tested, in this order; ``None`` tests every group. The rest is
    ///     still every other cell.
    /// reference : str or None, default=None
    ///     The group every other is compared with; ``None`` compares with the rest.
    /// n_genes : int or None, default=None
    ///     Keep the first this many genes of each group; ``None`` keeps every gene.
    /// rankby_abs : bool, default=False
    ///     Order genes by the absolute statistic.
    /// tie_correct : bool, default=False
    ///     Wilcoxon only: correct the variance for ties.
    /// correction : {"benjamini-hochberg", "bonferroni"}, default="benjamini-hochberg"
    ///     How the p-values are adjusted.
    /// threads : int or None, default=None
    ///     Threads that read and compute blocks; ``None`` for one per CPU core. The
    ///     result does not depend on it (R11).
    ///
    /// Returns
    /// -------
    /// polars.LazyFrame
    ///     Columns ``group``, ``gene_id`` (``String``), ``statistic``,
    ///     ``log_fold_change``, ``p_value``, ``adjusted_p_value`` (``Float64``).
    ///     Collecting it runs the plan and returns a new frame the caller owns.
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: an unknown ``method`` or ``correction``, ``n_genes`` below 1,
    ///     ``threads`` below 1; when collected, an empty or repeated ``groups``, no
    ///     labelled cell, or a tested group of one cell.
    /// ResolutionError
    ///     ``KY-E211``: ``layer`` is ``None`` and the plan has no default layer.
    /// MissingMemberError
    ///     ``KY-E109``: the layer or ``groupby`` is not in the plan.
    /// ResolutionError
    ///     ``KY-E214`` (when collected): a ``groups`` or ``reference`` label no cell
    ///     has.
    ///
    /// Examples
    /// --------
    /// >>> frame = scan(store).rank_genes_groups("cell_id", layer="counts")
    /// >>> frame.collect_schema().names()
    /// ['group', 'gene_id', 'statistic', 'log_fold_change', 'p_value', 'adjusted_p_value']
    #[pyo3(signature = (
        groupby,
        /,
        *,
        method = "t-test",
        layer = None,
        groups = None,
        reference = None,
        n_genes = None,
        rankby_abs = false,
        tie_correct = false,
        correction = "benjamini-hochberg",
        threads = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn rank_genes_groups<'py>(
        &self,
        py: Python<'py>,
        groupby: &str,
        method: &str,
        layer: Option<String>,
        groups: Option<&Bound<'_, PyAny>>,
        reference: Option<String>,
        n_genes: Option<&Bound<'_, PyAny>>,
        rankby_abs: bool,
        tie_correct: bool,
        correction: &str,
        threads: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let groups = groups
            .filter(|groups| !groups.is_none())
            .map(|groups| chain::strings(py, "rank_genes_groups", "groups", groups))
            .transpose()?;
        let n_genes = chain::optional_integer(py, "rank_genes_groups", "n_genes", n_genes, 1)?
            .map(|value| usize::try_from(value).unwrap_or(usize::MAX))
            .and_then(NonZeroUsize::new);
        let parameters = (|| {
            let mut parameters = RankGenesGroups::new(groupby, chain::differential_test(method)?);
            parameters.layer = layer;
            parameters.groups = groups;
            parameters.reference = reference;
            parameters.n_genes = n_genes;
            parameters.rankby_abs = rankby_abs;
            parameters.tie_correct = tie_correct;
            parameters.correction = chain::correction(correction)?;
            rank_genes::check(&self.plan, &parameters)?;
            Ok::<_, KytoError>(parameters)
        })()
        .map_err(|error| to_py_err(py, &error))?;
        let threads = terminal::threads(py, "rank_genes_groups", threads)?;
        rank_genes::lazy_frame(
            py,
            Arc::clone(&self.plan),
            self.collected.clone(),
            parameters,
            threads,
        )
    }

    /// Run the plan on a device (R7: the device is an argument, not a setting).
    ///
    /// Today every plan runs on the CPU; the GPU devices arrive with task 9.1.
    ///
    /// Parameters
    /// ----------
    /// device : {"cpu"}
    ///     The device the plan's nodes run on.
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan; this one is unchanged. Nothing is read or copied.
    ///
    /// Raises
    /// ------
    /// NotImplementedYetError
    ///     ``KY-E601``: ``"auto"``, ``"cuda"``, ``"wgpu"`` or ``"metal"`` (task 9.1).
    /// PlanError
    ///     ``KY-E215``: any other device name.
    ///
    /// Examples
    /// --------
    /// >>> scan(store).on(device="cpu").estimate()["device"]
    /// 'cpu'
    #[pyo3(signature = (*, device))]
    fn on(&self, py: Python<'_>, device: &str) -> PyResult<Self> {
        self.derive(py, chain::device(device).map(|()| Arc::clone(&self.plan)))
    }

    /// Run the plan and keep its result as a new, materialized ``Atlas``.
    ///
    /// The plan's output (its layers, and its cell and gene tables with the computed
    /// columns) is written to a temporary kyto store in the system's temporary
    /// directory, and the returned ``Atlas`` scans that store (with this plan's default
    /// layer, if the output has it). The store is deleted when the last ``Atlas`` built
    /// on it is garbage-collected, so ``to_json()`` of such a plan names a store that
    /// lives only as long as the plan. To keep the result, use ``sink_zarr``. The GIL is
    /// released while the plan computes; Ctrl-C cancels the run cleanly.
    ///
    /// Parameters
    /// ----------
    /// memory_limit : str or int or None, default=None
    ///     The most memory the run may use, such as ``"4GB"``, ``"512MiB"`` or a number
    ///     of bytes: blocks shrink to fit, and a plan that cannot fit raises before
    ///     anything is read. ``None`` runs with the default blocks.
    /// threads : int or None, default=None
    ///     Threads that read and compute blocks; ``None`` for one per CPU core. The
    ///     result does not depend on it (R11).
    /// progress : callable or None, default=None
    ///     Called as ``progress(done, total)`` on the calling thread whenever the run
    ///     moved (cell rows, summed over the passes).
    ///
    /// Returns
    /// -------
    /// Atlas
    ///     A new plan with one ``SCAN`` step (and ``WITH_DEFAULT_LAYER``) over the
    ///     result; this one is unchanged. The data are copied into the temporary store.
    ///
    /// Raises
    /// ------
    /// PlanError
    ///     ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
    ///     accepted.
    /// MemoryLimitError
    ///     ``KY-E401``: the plan cannot run within ``memory_limit``.
    /// DeviceError
    ///     ``KY-E502``: the system cannot start ``threads`` threads.
    /// NotImplementedYetError
    ///     ``KY-E601``: the plan has a step the executor does not run yet.
    /// LineageError
    ///     ``KY-E303``: the plan recomputes a column (``.recompute()``) whose store
    ///     does not hold the plan that computed it.
    /// KeyboardInterrupt
    ///     Ctrl-C while the plan computes; the run has stopped.
    ///
    /// Examples
    /// --------
    /// >>> atlas = scan(store).filter(total_counts(layer="counts") >= 7)
    /// >>> print(atlas.collect(threads=1))
    /// SCAN "...atlas.kyto" (1 cells x 3 genes)
    #[pyo3(signature = (*, memory_limit = None, threads = None, progress = None))]
    fn collect(
        &self,
        py: Python<'_>,
        memory_limit: Option<&Bound<'_, PyAny>>,
        threads: Option<&Bound<'_, PyAny>>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let folder = tempfile::Builder::new()
            .prefix("kyto-collect-")
            .tempdir()
            .map_err(|error| {
                let error = KytoError::StoreIo {
                    operation: "create a temporary folder for collect()".to_owned(),
                    path: std::env::temp_dir().display().to_string(),
                    detail: error.to_string(),
                };
                to_py_err(py, &error)
            })?;
        let target = folder.path().join("atlas.kyto");
        crate::materialize::sink(
            py,
            "collect",
            &self.plan,
            crate::materialize::zarr(&target),
            false,
            memory_limit,
            threads,
            progress,
        )?;
        let mut plan = open_plan(py, target)?;
        if let Some(layer) = self.plan.default_layer()
            && plan
                .schema()
                .layers
                .iter()
                .any(|stored| stored.name == layer)
        {
            plan = plan
                .with_default_layer(layer)
                .map_err(|error| to_py_err(py, &error))?;
        }
        Ok(Self {
            plan,
            collected: Some(Arc::new(folder)),
        })
    }

    /// Run the plan and write its output as a new kyto store at ``path``.
    ///
    /// Writes the plan's layers, and its cell and gene tables with the computed columns
    /// (derived columns keep their lineage; stale ones are dropped), streaming one cell
    /// block at a time. The store is written next to ``path`` and renamed into place
    /// when complete, so a failed or cancelled run leaves nothing (and an overwritten
    /// store is as it was). The GIL is released while the plan computes; Ctrl-C
    /// cancels the run cleanly.
    ///
    /// Parameters
    /// ----------
    /// path : str or os.PathLike[str]
    ///     The new store.
    /// overwrite : bool, default=False
    ///     Replace a kyto store already at ``path``, once the new one is complete.
    ///     Anything else at ``path`` is never replaced, nor the store the plan reads.
    /// memory_limit : str or int or None, default=None
    ///     The most memory the run may use, such as ``"4GB"``, ``"512MiB"`` or a number
    ///     of bytes: blocks shrink to fit, and a plan that cannot fit raises before
    ///     anything is read. ``None`` runs with the default blocks.
    /// threads : int or None, default=None
    ///     Threads that read and compute blocks; ``None`` for one per CPU core. The
    ///     result does not depend on it (R11).
    /// progress : callable or None, default=None
    ///     Called as ``progress(done, total)`` on the calling thread whenever the run
    ///     moved (cell rows, summed over the passes).
    ///
    /// Returns
    /// -------
    /// None
    ///     The data are written to ``path``; this plan is unchanged.
    ///
    /// Raises
    /// ------
    /// StoreInputError
    ///     ``KY-E118``: ``path`` exists and ``overwrite`` is false, it is not a kyto
    ///     store, or it is (or lies inside) the store the plan reads.
    /// PlanError
    ///     ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
    ///     accepted. ``KY-E216``: the plan already ends in a sink.
    /// MemoryLimitError
    ///     ``KY-E401``: the plan cannot run within ``memory_limit``.
    /// DeviceError
    ///     ``KY-E502``: the system cannot start ``threads`` threads.
    /// NotImplementedYetError
    ///     ``KY-E601``: the plan has a step the executor does not run yet.
    /// LineageError
    ///     ``KY-E303``: the plan recomputes a column (``.recompute()``) whose store
    ///     does not hold the plan that computed it.
    /// KeyboardInterrupt
    ///     Ctrl-C while the plan computes; the run has stopped and nothing is left.
    ///
    /// Examples
    /// --------
    /// >>> target = store.parent / "normalized.kyto"
    /// >>> scan(store).normalize_total(1e4, layer="counts", into="normalized").sink_zarr(target)
    /// >>> scan(target).layer("normalized").schema()["dtype"]
    /// 'float32'
    /// >>> scan(store).log1p(layer="counts", into="log1p").sink_zarr(target, overwrite=True)
    /// >>> scan(target).layer("log1p").schema()["dtype"]
    /// 'float32'
    #[pyo3(signature = (
        path, /, *, overwrite = false, memory_limit = None, threads = None, progress = None
    ))]
    fn sink_zarr(
        &self,
        py: Python<'_>,
        path: PathBuf,
        overwrite: bool,
        memory_limit: Option<&Bound<'_, PyAny>>,
        threads: Option<&Bound<'_, PyAny>>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::materialize::sink(
            py,
            "sink_zarr",
            &self.plan,
            crate::materialize::zarr(&path),
            overwrite,
            memory_limit,
            threads,
            progress,
        )
    }

    /// Run the plan and write its output as an ``.h5ad`` file at ``path``, for
    /// scanpy and anndata.
    ///
    /// The plan's default layer becomes ``X`` (no ``X`` without one) and every other
    /// layer goes under ``layers``, all as CSR; the cell and gene tables with the
    /// computed columns become ``obs`` and ``var``. The plan is streamed into a
    /// temporary kyto store next to ``path``, exported, and the store deleted; the
    /// file is renamed into place when complete, so a failed or cancelled run leaves
    /// nothing. The GIL is released while the plan computes; Ctrl-C cancels the run
    /// cleanly.
    ///
    /// Parameters
    /// ----------
    /// path : str or os.PathLike[str]
    ///     The new file.
    /// overwrite : bool, default=False
    ///     Replace a file already at ``path``, once the new one is complete. A
    ///     directory is never replaced.
    /// memory_limit : str or int or None, default=None
    ///     The most memory the run may use, such as ``"4GB"``, ``"512MiB"`` or a number
    ///     of bytes: blocks shrink to fit, and a plan that cannot fit raises before
    ///     anything is read. ``None`` runs with the default blocks.
    /// threads : int or None, default=None
    ///     Threads that read and compute blocks; ``None`` for one per CPU core. The
    ///     result does not depend on it (R11).
    /// progress : callable or None, default=None
    ///     Called as ``progress(done, total)`` on the calling thread whenever the run
    ///     moved (cell rows, summed over the passes).
    ///
    /// Returns
    /// -------
    /// None
    ///     The data are written to ``path``; this plan is unchanged.
    ///
    /// Raises
    /// ------
    /// StoreInputError
    ///     ``KY-E123``: ``path`` exists and ``overwrite`` is false, or is a directory.
    ///     ``KY-E124``: a name ``.h5ad`` cannot hold.
    /// PlanError
    ///     ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
    ///     accepted. ``KY-E216``: the plan already ends in a sink.
    /// MemoryLimitError
    ///     ``KY-E401``: the plan cannot run within ``memory_limit``.
    /// DeviceError
    ///     ``KY-E502``: the system cannot start ``threads`` threads.
    /// NotImplementedYetError
    ///     ``KY-E601``: the plan has a step the executor does not run yet.
    /// KeyboardInterrupt
    ///     Ctrl-C while the plan computes; the run has stopped and nothing is left.
    ///
    /// Examples
    /// --------
    /// >>> target = store.parent / "pbmc.h5ad"
    /// >>> scan(store).with_default_layer("counts").sink_h5ad(target)
    /// >>> target.exists()
    /// True
    /// >>> scan(store).sink_h5ad(target)
    /// Traceback (most recent call last):
    /// ...
    /// kyto.errors.StoreInputError: KY-E123: cannot write '...pbmc.h5ad': ...
    #[pyo3(signature = (
        path, /, *, overwrite = false, memory_limit = None, threads = None, progress = None
    ))]
    fn sink_h5ad(
        &self,
        py: Python<'_>,
        path: PathBuf,
        overwrite: bool,
        memory_limit: Option<&Bound<'_, PyAny>>,
        threads: Option<&Bound<'_, PyAny>>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let target = kyto_plan::Sink::H5ad {
            path: path.display().to_string(),
        };
        crate::materialize::sink(
            py,
            "sink_h5ad",
            &self.plan,
            target,
            overwrite,
            memory_limit,
            threads,
            progress,
        )
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
        PyTable::new(
            kyto_plan::Table::new(&self.plan, kyto_core::Axis::Cells),
            self.collected.clone(),
        )
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
        PyTable::new(
            kyto_plan::Table::new(&self.plan, kyto_core::Axis::Genes),
            self.collected.clone(),
        )
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
            .map(|handle| PyMatrix::new(handle, self.collected.clone()))
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
            .map(|handle| PyEmbedding::new(handle, self.collected.clone()))
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
            .map(|handle| PyGraph::new(handle, self.collected.clone()))
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
    /// >>> atlas = scan(store).filter(genes)
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
    /// names), and, in brackets, a note when the step has a cheaper alternative (a
    /// centred ``scale`` stores every gene of every cell; PCA centres sparse data
    /// implicitly). The optimized tree is the plan optimized with every rule (for a terminal
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
    ///     ``id``, ``name``, ``parameters``, ``input``, ``note``) and ``optimized`` (``outputs``:
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
    /// >>> print(scan(store).filter(genes).explain())
    /// logical plan
    ///   node 1 FILTER_GENES gene_column('feature_type') == 'Gene Expression' <- node 0
    ///   node 0 SCAN "..." (2 cells x 3 genes)
    /// optimized plan
    ///   output #1 reads [...]
    ///   #1 FILTER_GENES ... <- #0 (node 1) [device=cpu passes=1 blocks=4096 cells]
    ///   #0 SCAN ... (node 0) [device=cpu passes=1 blocks=4096 cells]
    /// >>> explained = scan(store).filter(genes).explain(format="json")
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
        threads: Option<&Bound<'_, PyAny>>,
        progress: Option<Bound<'_, PyAny>>,
    ) -> PyResult<u64> {
        let run = terminal::run_arguments(
            py,
            "Atlas._execute",
            &self.plan,
            None,
            threads,
            progress.as_ref(),
        )?;
        run_plan(py, &run.physical, run.threads, progress.as_ref(), false)
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

/// `KY-E215`: `with_cells()`/`with_genes()` without a column.
fn no_columns(method: &str) -> KytoError {
    KytoError::InvalidArgument {
        function: method.to_owned(),
        parameter: "columns".to_owned(),
        value: "{}".to_owned(),
        problem: "no column is given".to_owned(),
        accepted: format!("one or more name=expression pairs, such as {method}(n=...)"),
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
        .map(|plan| Atlas {
            plan,
            collected: None,
        })
        .map_err(|error| to_py_err(py, &error))
}

/// Open the kyto store at ``store`` as the root of a lazy plan.
///
/// Reads the store's ``schema.json`` and its gene IDs (and gene symbols, when the gene
/// table has a ``symbol`` column), so names can be resolved and checked when the plan
/// is built (R6). Reads no matrix tile and no other column (R1). Every method of the
/// returned ``Atlas`` builds a new plan on top of this one (R2).
///
/// Parameters
/// ----------
/// store : str or os.PathLike[str]
///     Path of a local kyto store, as ``convert`` writes it.
///
/// Returns
/// -------
/// Atlas
///     A new plan with one ``SCAN`` step. Nothing of the matrix is read or copied.
///
/// Raises
/// ------
/// CorruptStoreError
///     ``KY-E116``: ``store`` does not exist or is not a kyto store.
///     ``KY-E113``, ``KY-E122``: its gene table or ``schema.json`` breaks the format.
/// FormatVersionError
///     ``KY-E115``: the store has a ``format_version`` this kyto does not read.
/// StoreIOError
///     ``KY-E110``: a file of the store cannot be read.
///
/// Examples
/// --------
/// >>> import tempfile
/// >>> from pathlib import Path
/// >>> folder = Path(tempfile.mkdtemp())
/// >>> _ = (folder / "matrix.mtx").write_text(
/// ...     "%%MatrixMarket matrix coordinate integer general\n3 2 3\n"
/// ...     "1 1 5\n3 1 1\n2 2 7\n"
/// ... )
/// >>> _ = (folder / "barcodes.tsv").write_text("AAAC-1\nAAAG-1\n")
/// >>> _ = (folder / "features.tsv").write_text(
/// ...     "ENSG01\tCD3E\tGene Expression\nENSG02\tCD4\tGene Expression\n"
/// ...     "ENSG03\tMS4A1\tGene Expression\n"
/// ... )
/// >>> convert(folder, folder / "pbmc.kyto")
/// >>> atlas = scan(folder / "pbmc.kyto")
/// >>> print(atlas)
/// SCAN "..." (2 cells x 3 genes)
/// >>> scan(folder / "missing.kyto")
/// Traceback (most recent call last):
///     ...
/// kyto.errors.CorruptStoreError: KY-E116: ...
#[pyfunction]
#[pyo3(signature = (store, /))]
pub fn scan(py: Python<'_>, store: PathBuf) -> PyResult<Atlas> {
    Ok(Atlas {
        plan: open_plan(py, store)?,
        collected: None,
    })
}

/// The root plan of the store at `store`: its schema, gene IDs and symbols (`scan`).
fn open_plan(py: Python<'_>, store: PathBuf) -> PyResult<Arc<Plan>> {
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
    Ok(Plan::scan(store.display().to_string(), schema, gene_ids))
}
