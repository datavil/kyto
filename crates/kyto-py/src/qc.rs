//! The QC metric constructors: `total_counts`, `genes_detected`, `total_counts_in`
//! and `percent_counts_in`, each a `CellExpr` leaf (task 6.1), and `cells_detected`,
//! `gene_total_counts`, `gene_mean_counts` and `percent_dropout`, each a `GeneExpr`
//! leaf (task 6.13).
//!
//! Does: build `kyto_core::CellExpr::qc` and `GeneExpr::qc` leaves, taking the gene
//! set of the two gene-set metrics as a `GeneExpr` (`KY-E205` for anything else).
//! Does not: check the layer or the gene set's columns (`Atlas.resolve` and the plan
//! do, at build), or compute the metrics (`kyto-algo`'s `qc`, when a plan reads them).

use kyto_core::{CellExpr, GeneExpr, GeneQcMetric, QcMetric};
use pyo3::prelude::*;

use crate::atlas::wrong_type;
use crate::errors::to_py_err;
use crate::expr::{PyCellExpr, PyGeneExpr};

/// The QC leaf of `metric` on `layer`.
fn qc(py: Python<'_>, layer: Option<&str>, metric: QcMetric) -> PyResult<PyCellExpr> {
    CellExpr::qc(layer.map(str::to_owned), metric)
        .map(PyCellExpr::from_core)
        .map_err(|error| to_py_err(py, &error))
}

/// The gene set `genes` given to `function`: a `GeneExpr` (`KY-E205` otherwise).
fn gene_set(py: Python<'_>, function: &str, genes: &Bound<'_, PyAny>) -> PyResult<GeneExpr> {
    match genes.cast::<PyGeneExpr>() {
        Ok(genes) => Ok(genes.get().core().clone()),
        Err(_) => Err(to_py_err(
            py,
            &wrong_type(function, "a boolean GeneExpr", genes),
        )),
    }
}

/// Each cell's total count: the sum of its values over every gene, as a ``CellExpr``.
///
/// scanpy's ``total_counts`` (``scanpy.pp.calculate_qc_metrics``), computed when a
/// plan reads it, over the genes present then, so it never goes stale. Summed in
/// ``float64`` (scanpy: ``float32``).
///
/// Parameters
/// ----------
/// layer : str or None, default=None
///     The layer to sum. ``None`` reads the plan's default layer
///     (``Atlas.with_default_layer``).
///
/// Returns
/// -------
/// CellExpr
///     A numeric expression that reads nothing until a plan using it runs.
///
/// Examples
/// --------
/// >>> total_counts(layer="counts") > 500
/// total_counts(layer='counts') > 500
#[pyfunction]
#[pyo3(signature = (*, layer = None))]
pub fn total_counts(py: Python<'_>, layer: Option<&str>) -> PyResult<PyCellExpr> {
    qc(py, layer, QcMetric::TotalCounts)
}

/// Each cell's number of genes with a non-zero value, as a ``CellExpr``.
///
/// scanpy's ``n_genes_by_counts`` (``scanpy.pp.calculate_qc_metrics``), computed when
/// a plan reads it, over the genes present then, so it never goes stale.
///
/// Parameters
/// ----------
/// layer : str or None, default=None
///     The layer to count in. ``None`` reads the plan's default layer
///     (``Atlas.with_default_layer``).
///
/// Returns
/// -------
/// CellExpr
///     A numeric (``UInt32``) expression that reads nothing until a plan using it runs.
///
/// Examples
/// --------
/// >>> genes_detected(layer="counts") >= 200
/// genes_detected(layer='counts') >= 200
#[pyfunction]
#[pyo3(signature = (*, layer = None))]
pub fn genes_detected(py: Python<'_>, layer: Option<&str>) -> PyResult<PyCellExpr> {
    qc(py, layer, QcMetric::GenesDetected)
}

/// Each cell's total count over a gene set, as a ``CellExpr``.
///
/// scanpy's ``total_counts_<qc_var>`` (``scanpy.pp.calculate_qc_metrics``), computed
/// when a plan reads it, over the genes present then, so it never goes stale. Summed
/// in ``float64`` (scanpy: ``float32``).
///
/// Parameters
/// ----------
/// genes : GeneExpr
///     The gene set: a boolean expression on gene-table columns, such as
///     ``gene_column('symbol').str.starts_with('MT-')``. A gene where it is null is
///     not in the set. Checked when the plan is built.
/// layer : str or None, default=None
///     The layer to sum. ``None`` reads the plan's default layer
///     (``Atlas.with_default_layer``).
///
/// Returns
/// -------
/// CellExpr
///     A numeric expression that reads nothing until a plan using it runs.
///
/// Raises
/// ------
/// ExpressionError
///     ``KY-E205`` if ``genes`` is not a ``GeneExpr`` or not boolean (a column of
///     unknown dtype is checked when the plan is built, as is an expression that
///     reads matrix values).
///
/// Examples
/// --------
/// >>> total_counts_in(gene_column("symbol").str.starts_with("MT-"))
/// total_counts_in(gene_column('symbol').str.starts_with('MT-'))
#[pyfunction]
#[pyo3(signature = (genes, /, *, layer = None))]
pub fn total_counts_in(
    py: Python<'_>,
    genes: &Bound<'_, PyAny>,
    layer: Option<&str>,
) -> PyResult<PyCellExpr> {
    let genes = gene_set(py, "total_counts_in", genes)?;
    qc(py, layer, QcMetric::TotalCountsIn { genes })
}

/// Each cell's percentage of counts in a gene set, as a ``CellExpr``.
///
/// scanpy's ``pct_counts_<qc_var>`` (``scanpy.pp.calculate_qc_metrics``):
/// ``total_counts_in(genes) / total_counts() * 100``, ``NaN`` for a cell without
/// counts. Computed when a plan reads it, over the genes present then, so it never
/// goes stale; in ``float64`` (scanpy: ``float32``).
///
/// Parameters
/// ----------
/// genes : GeneExpr
///     The gene set: a boolean expression on gene-table columns, such as
///     ``gene_column('symbol').str.starts_with('MT-')``. A gene where it is null is
///     not in the set. Checked when the plan is built.
/// layer : str or None, default=None
///     The layer to sum. ``None`` reads the plan's default layer
///     (``Atlas.with_default_layer``).
///
/// Returns
/// -------
/// CellExpr
///     A numeric expression that reads nothing until a plan using it runs.
///
/// Raises
/// ------
/// ExpressionError
///     ``KY-E205`` if ``genes`` is not a ``GeneExpr`` or not boolean (a column of
///     unknown dtype is checked when the plan is built, as is an expression that
///     reads matrix values).
///
/// Examples
/// --------
/// >>> mitochondrial = gene_column("symbol").str.starts_with("MT-")
/// >>> percent_counts_in(mitochondrial, layer="counts") < 5
/// percent_counts_in(gene_column('symbol').str.starts_with('MT-'), layer='counts') < 5
#[pyfunction]
#[pyo3(signature = (genes, /, *, layer = None))]
pub fn percent_counts_in(
    py: Python<'_>,
    genes: &Bound<'_, PyAny>,
    layer: Option<&str>,
) -> PyResult<PyCellExpr> {
    let genes = gene_set(py, "percent_counts_in", genes)?;
    qc(py, layer, QcMetric::PercentCountsIn { genes })
}

/// The per-gene QC leaf of `metric` on `layer` (task 6.13).
fn gene_qc(layer: Option<&str>, metric: GeneQcMetric) -> PyGeneExpr {
    PyGeneExpr::from_core(GeneExpr::qc(layer.map(str::to_owned), metric))
}

/// Each gene's number of cells with a non-zero value, as a ``GeneExpr``.
///
/// scanpy's ``n_cells_by_counts`` (``scanpy.pp.calculate_qc_metrics``), computed when
/// a plan reads it, over the cells present then (one pass over them), so it never goes
/// stale. ``atlas.filter(cells_detected() >= 3)`` is scanpy's
/// ``filter_genes(min_cells=3)``.
///
/// Parameters
/// ----------
/// layer : str or None, default=None
///     The layer to count in. ``None`` reads the plan's default layer
///     (``Atlas.with_default_layer``).
///
/// Returns
/// -------
/// GeneExpr
///     A numeric (``UInt32``) expression that reads nothing until a plan using it runs.
///
/// Examples
/// --------
/// >>> cells_detected(layer="counts") >= 3
/// cells_detected(layer='counts') >= 3
#[pyfunction]
#[pyo3(signature = (*, layer = None))]
pub fn cells_detected(layer: Option<&str>) -> PyGeneExpr {
    gene_qc(layer, GeneQcMetric::CellsDetected)
}

/// Each gene's total count: the sum of its values over every cell, as a ``GeneExpr``.
///
/// scanpy's per-gene ``total_counts`` (``scanpy.pp.calculate_qc_metrics``), computed
/// when a plan reads it, over the cells present then (one pass over them), so it never
/// goes stale. Summed in ``float64`` (scanpy: ``float32``). scanpy's
/// ``log1p_total_counts`` is ``gene_total_counts().log1p()``.
///
/// Parameters
/// ----------
/// layer : str or None, default=None
///     The layer to sum. ``None`` reads the plan's default layer
///     (``Atlas.with_default_layer``).
///
/// Returns
/// -------
/// GeneExpr
///     A numeric expression that reads nothing until a plan using it runs.
///
/// Examples
/// --------
/// >>> gene_total_counts(layer="counts").log1p()
/// gene_total_counts(layer='counts').log1p()
#[pyfunction]
#[pyo3(signature = (*, layer = None))]
pub fn gene_total_counts(layer: Option<&str>) -> PyGeneExpr {
    gene_qc(layer, GeneQcMetric::TotalCounts)
}

/// Each gene's mean value over every cell, as a ``GeneExpr``.
///
/// scanpy's ``mean_counts`` (``scanpy.pp.calculate_qc_metrics``):
/// ``gene_total_counts() / <number of cells>``, zeros included, ``NaN`` without
/// cells. Computed when a plan reads it, over the cells present then (one pass over
/// them), so it never goes stale; in ``float64`` (scanpy: ``float32``). scanpy's
/// ``log1p_mean_counts`` is ``gene_mean_counts().log1p()``.
///
/// Parameters
/// ----------
/// layer : str or None, default=None
///     The layer to average. ``None`` reads the plan's default layer
///     (``Atlas.with_default_layer``).
///
/// Returns
/// -------
/// GeneExpr
///     A numeric expression that reads nothing until a plan using it runs.
///
/// Examples
/// --------
/// >>> gene_mean_counts(layer="counts") > 0.1
/// gene_mean_counts(layer='counts') > 0.1
#[pyfunction]
#[pyo3(signature = (*, layer = None))]
pub fn gene_mean_counts(layer: Option<&str>) -> PyGeneExpr {
    gene_qc(layer, GeneQcMetric::MeanCounts)
}

/// Each gene's percentage of cells without a value, as a ``GeneExpr``.
///
/// scanpy's ``pct_dropout_by_counts`` (``scanpy.pp.calculate_qc_metrics``):
/// ``(1 - cells_detected() / <number of cells>) * 100``, ``NaN`` without cells.
/// Computed when a plan reads it, over the cells present then (one pass over them),
/// so it never goes stale.
///
/// Parameters
/// ----------
/// layer : str or None, default=None
///     The layer to count in. ``None`` reads the plan's default layer
///     (``Atlas.with_default_layer``).
///
/// Returns
/// -------
/// GeneExpr
///     A numeric expression that reads nothing until a plan using it runs.
///
/// Examples
/// --------
/// >>> percent_dropout(layer="counts") < 99
/// percent_dropout(layer='counts') < 99
#[pyfunction]
#[pyo3(signature = (*, layer = None))]
pub fn percent_dropout(layer: Option<&str>) -> PyGeneExpr {
    gene_qc(layer, GeneQcMetric::PercentDropout)
}
