//! Feature access (task 7.4): an `Atlas`'s values by feature, and its cell and gene
//! IDs and counts, as Polars objects and ints.
//!
//! Does: compute the features asked for as hidden `with_cells`/`with_genes` columns of
//! the plan (so every name is resolved and checked when the call is made, R6) and run
//! only what reading them needs (an output reading those columns: the optimizer leaves
//! out the rest, and no layer is read that the features do not read), keeping the rows
//! the stream yields; pair them with the store's cell IDs (`select`, `pull_genes`); give
//! the cell and gene IDs and counts of the plan's output from the plan when it knows
//! them (no filter on data), and by running the plan otherwise; list the feature names
//! the plan resolves (`features`, metadata only); read the scanned store's
//! `provenance.json` (`metadata`).
//! Does not: write anything (no temporary store), read a layer outside the features'
//! leaves, or decide how a name resolves (`kyto_core::Resolver` does).

use std::path::Path;
use std::sync::Arc;

use kyto_core::{Axis, CellExpr, GeneExpr};
use kyto_error::KytoError;
use kyto_exec::{ExecuteOptions, Genes, Pipeline};
use kyto_plan::{Member, Output, Plan};
use kyto_store::TableAxis;
use polars::prelude::{Column, DataFrame, DataType, IntoColumn, NamedFrom, Series};
use pyo3::prelude::*;

use crate::errors::to_py_err;
use crate::handoff;
use crate::run::run_with;
use crate::terminal::{self, invalid};

/// The prefix of the hidden columns a feature-access call computes. Never written: the
/// columns live only in the plan the call builds and runs.
const HIDDEN: &str = "__kyto_feature_";

/// A terminal's shared arguments, as the Python methods take them.
#[derive(Clone, Copy)]
pub struct RunArgs<'a, 'py> {
    /// `memory_limit`: as `Atlas.collect` takes it.
    pub memory_limit: Option<&'a Bound<'py, PyAny>>,
    /// `threads`: `None` for one per CPU core.
    pub threads: Option<&'a Bound<'py, PyAny>>,
    /// `progress`: a callable or `None`.
    pub progress: Option<&'a Bound<'py, PyAny>>,
}

impl RunArgs<'_, '_> {
    /// Check the arguments of a call that runs nothing (`KY-E215`), so they mean the
    /// same on every plan.
    fn check(&self, py: Python<'_>, function: &str) -> PyResult<()> {
        terminal::memory_limit(py, function, self.memory_limit)?;
        terminal::threads(py, function, self.threads)?;
        terminal::check_progress(py, function, self.progress)
    }
}

/// What a run of an output kept: the store position of each cell row, in stream order,
/// the hidden cell columns read, and the genes present (when the plan changes them or
/// computes gene columns).
struct Streamed {
    cells: Vec<u64>,
    columns: Vec<DataFrame>,
    genes: Option<Genes>,
}

/// Run `plan`, reading only `reads`, keeping the cell positions, the cell columns in
/// `reads` and the genes present.
fn stream(
    py: Python<'_>,
    function: &str,
    plan: &Arc<Plan>,
    reads: Vec<Member>,
    args: RunArgs<'_, '_>,
) -> PyResult<Streamed> {
    let names: Vec<String> = reads
        .iter()
        .filter_map(|member| match member {
            Member::CellColumn(name) => Some(name.clone()),
            _ => None,
        })
        .collect();
    let store = plan.root().source().to_owned();
    let output = Output::members(plan, reads).map_err(|error| to_py_err(py, &error))?;
    let run = terminal::output_run_arguments(
        py,
        function,
        output,
        args.memory_limit,
        args.threads,
        args.progress,
    )?;
    run_with(
        py,
        &run.physical,
        run.threads,
        args.progress,
        move |pipeline: &Pipeline, options: ExecuteOptions| {
            let mut stream = pipeline.run(options)?;
            let mut cells = Vec::new();
            let mut columns = Vec::new();
            while let Some(block) = stream.next() {
                let mut block = block?;
                cells.extend_from_slice(&block.cells);
                if !names.is_empty() {
                    let frame = std::mem::take(&mut block.columns);
                    columns.push(frame.select(names.iter().map(String::as_str)).map_err(
                        |error| KytoError::CorruptTable {
                            path: store.clone(),
                            problem: format!("a computed column is missing: {error}"),
                        },
                    )?);
                }
                stream.recycle(block);
            }
            Ok(Streamed {
                cells,
                columns,
                genes: stream.genes().cloned(),
            })
        },
    )
}

/// The scanned store's path, after checking that its `schema.json` is still the one
/// the plan read (`KY-E127` otherwise, as 7.3's terminals).
fn checked_store<'p>(py: Python<'_>, plan: &'p Plan) -> PyResult<&'p str> {
    let root = plan.root();
    let source = root.source();
    let stored = py
        .detach(|| kyto_store::open_atlas_schema(Path::new(source)))
        .map_err(|error| to_py_err(py, &error))?;
    if &stored != root.schema().as_ref() {
        let error = KytoError::StoreChanged {
            store: source.to_owned(),
            problem: "its schema.json is not the one the plan read".to_owned(),
        };
        return Err(to_py_err(py, &error));
    }
    Ok(source)
}

/// The cell IDs of the store at the store positions `cells`.
fn cell_ids_at(py: Python<'_>, plan: &Plan, cells: &[u64]) -> PyResult<Vec<String>> {
    let store = checked_store(py, plan)?;
    let n_cells = plan.root().schema().n_cells;
    let ids = py
        .detach(|| kyto_store::read_ids(Path::new(store), TableAxis::Cells, n_cells))
        .map_err(|error| to_py_err(py, &error))?;
    Ok(cells
        .iter()
        .map(|&position| ids[position as usize].clone())
        .collect())
}

/// Every cell ID of the store, in store order.
fn all_cell_ids(py: Python<'_>, plan: &Plan) -> PyResult<Vec<String>> {
    let store = checked_store(py, plan)?;
    let n_cells = plan.root().schema().n_cells;
    py.detach(|| kyto_store::read_ids(Path::new(store), TableAxis::Cells, n_cells))
        .map_err(|error| to_py_err(py, &error))
}

/// The hidden name of feature `index`.
fn hidden(index: usize) -> String {
    format!("{HIDDEN}{index}")
}

/// `KY-E215` for `function`'s argument `parameter`.
fn refuse(
    py: Python<'_>,
    function: &str,
    parameter: &str,
    value: &str,
    problem: &str,
    accepted: &str,
) -> PyErr {
    to_py_err(
        py,
        &invalid(function, parameter, value.to_owned(), problem, accepted),
    )
}

/// The cell columns `named` (output name, expression) of `plan`'s output as a frame:
/// `cell_id` first, then one column per feature, in the order given, each named as
/// given. Names must be distinct and not `cell_id` (`KY-E215`).
pub fn select_cells<'py>(
    py: Python<'py>,
    function: &str,
    plan: &Arc<Plan>,
    parameter: &str,
    named: Vec<(String, CellExpr)>,
    args: RunArgs<'_, 'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let frame = cell_frame(py, function, plan, parameter, named, args)?;
    handoff::to_polars(py, frame)
}

/// One cell feature of `plan`'s output as a Polars `Series` named `name`.
pub fn pull_cells<'py>(
    py: Python<'py>,
    function: &str,
    plan: &Arc<Plan>,
    name: String,
    expression: CellExpr,
    args: RunArgs<'_, 'py>,
) -> PyResult<Bound<'py, PyAny>> {
    // The frame's column name is not the output name: `pull("cell_id")` is fine.
    let frame = cell_frame(
        py,
        function,
        plan,
        "feature",
        vec![("value".to_owned(), expression)],
        args,
    )?;
    let values = frame
        .column("value")
        .map_err(|error| polars_error(py, plan, &error))?
        .clone()
        .with_name(name.as_str().into());
    let frame = DataFrame::new_infer_height(vec![values])
        .map_err(|error| polars_error(py, plan, &error))?;
    handoff::to_polars(py, frame)?.call_method1("to_series", (0,))
}

/// One gene feature of `plan`'s output as a Polars `Series` named `name`, one row per
/// gene present, in the plan's gene order.
pub fn pull_genes_column<'py>(
    py: Python<'py>,
    function: &str,
    plan: &Arc<Plan>,
    name: String,
    expression: GeneExpr,
    args: RunArgs<'_, 'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let column = hidden(0);
    let resolved = plan
        .resolve(&expression)
        .map_err(|error| to_py_err(py, &error))?;
    let computed = plan
        .with_genes([(column.clone(), resolved)])
        .map_err(|error| to_py_err(py, &error))?;
    let streamed = stream(
        py,
        function,
        &computed,
        vec![Member::GeneColumn(column.clone())],
        args,
    )?;
    let Some(genes) = streamed.genes else {
        // A `with_genes` step always streams its gene table; a missing one is a bug.
        let error = KytoError::CorruptTable {
            path: plan.root().source().to_owned(),
            problem: "the run gave no gene table for a computed gene column".to_owned(),
        };
        return Err(to_py_err(py, &error));
    };
    let values = genes
        .table()
        .column(&column)
        .map_err(|error| polars_error(py, plan, &error))?
        .clone()
        .with_name(name.as_str().into());
    let frame = DataFrame::new_infer_height(vec![values])
        .map_err(|error| polars_error(py, plan, &error))?;
    handoff::to_polars(py, frame)?.call_method1("to_series", (0,))
}

/// `cell_id` and the `named` cell features of `plan`'s output, as a Rust frame.
fn cell_frame(
    py: Python<'_>,
    function: &str,
    plan: &Arc<Plan>,
    parameter: &str,
    named: Vec<(String, CellExpr)>,
    args: RunArgs<'_, '_>,
) -> PyResult<DataFrame> {
    if named.is_empty() {
        return Err(refuse(
            py,
            function,
            parameter,
            "()",
            "no feature is given",
            "one or more",
        ));
    }
    let mut names: Vec<String> = Vec::with_capacity(named.len());
    for (name, _) in &named {
        if name == "cell_id" || names.contains(name) {
            let problem = if name == "cell_id" {
                "'cell_id' is the ID column, which comes first already"
            } else {
                "the name is given twice"
            };
            return Err(refuse(
                py,
                function,
                parameter,
                &format!("'{name}'"),
                problem,
                "distinct output names other than 'cell_id'",
            ));
        }
        names.push(name.clone());
    }
    // Resolve each feature against the plan first: errors then quote the feature, not
    // a hidden column (R6; the stored expression names everything in full).
    let mut columns = Vec::with_capacity(named.len());
    for (index, (_, expression)) in named.into_iter().enumerate() {
        let resolved = plan
            .resolve(&expression)
            .map_err(|error| to_py_err(py, &error))?;
        columns.push((hidden(index), resolved));
    }
    let hidden_names: Vec<String> = columns.iter().map(|(name, _)| name.clone()).collect();
    let computed = plan
        .with_cells(columns)
        .map_err(|error| to_py_err(py, &error))?;
    let reads = hidden_names
        .iter()
        .map(|name| Member::CellColumn(name.clone()))
        .collect();
    let streamed = stream(py, function, &computed, reads, args)?;
    let ids = cell_ids_at(py, plan, &streamed.cells)?;
    let mut frame_columns: Vec<Column> = Vec::with_capacity(names.len() + 1);
    frame_columns.push(Series::new("cell_id".into(), ids).into_column());
    for (hidden_name, name) in hidden_names.iter().zip(&names) {
        let values = column_values(py, &computed, &streamed.columns, hidden_name)?;
        frame_columns.push(values.with_name(name.as_str().into()).into_column());
    }
    DataFrame::new_infer_height(frame_columns).map_err(|error| polars_error(py, plan, &error))
}

/// Column `name` of every streamed block, in stream order; an empty column of the
/// declared dtype when nothing streamed (an `Enum` column is then `String`: its
/// categories are known only from the data).
fn column_values(
    py: Python<'_>,
    plan: &Arc<Plan>,
    blocks: &[DataFrame],
    name: &str,
) -> PyResult<Series> {
    let mut parts = blocks.iter().map(|frame| frame.column(name));
    let Some(first) = parts.next() else {
        let dtype = plan
            .schema()
            .cells
            .columns
            .iter()
            .find(|column| column.name == name)
            .map(|column| empty_dtype(&column.dtype))
            .unwrap_or(DataType::Null);
        return Ok(Series::new_empty(name.into(), &dtype));
    };
    let mut values = first
        .map_err(|error| polars_error(py, plan, &error))?
        .as_materialized_series()
        .clone();
    for part in parts {
        let part = part.map_err(|error| polars_error(py, plan, &error))?;
        values
            .append(part.as_materialized_series())
            .map_err(|error| polars_error(py, plan, &error))?;
    }
    Ok(values)
}

/// The Polars dtype of an empty column of `dtype`.
fn empty_dtype(dtype: &kyto_core::ColumnDtype) -> DataType {
    use kyto_core::ColumnDtype as D;
    match dtype {
        D::Boolean => DataType::Boolean,
        D::Int8 => DataType::Int8,
        D::Int16 => DataType::Int16,
        D::Int32 => DataType::Int32,
        D::Int64 => DataType::Int64,
        D::UInt8 => DataType::UInt8,
        D::UInt16 => DataType::UInt16,
        D::UInt32 => DataType::UInt32,
        D::UInt64 => DataType::UInt64,
        D::Float32 => DataType::Float32,
        D::Float64 => DataType::Float64,
        D::String | D::Enum { .. } => DataType::String,
    }
}

/// A Polars error while assembling a result: a kyto bug or a corrupt table
/// (`KY-E113`).
fn polars_error(py: Python<'_>, plan: &Plan, error: &polars::prelude::PolarsError) -> PyErr {
    let error = KytoError::CorruptTable {
        path: plan.root().source().to_owned(),
        problem: format!("assembling a feature-access result failed: {error}"),
    };
    to_py_err(py, &error)
}

/// The cell IDs of `plan`'s output, in order: the store's when no step changes the
/// cells (read in place, nothing runs), else from a run of the plan.
pub fn cell_ids(
    py: Python<'_>,
    function: &str,
    plan: &Arc<Plan>,
    args: RunArgs<'_, '_>,
) -> PyResult<Vec<String>> {
    if plan.cell_axis_changed_by().is_none() {
        args.check(py, function)?;
        return all_cell_ids(py, plan);
    }
    let streamed = stream(py, function, plan, Vec::new(), args)?;
    cell_ids_at(py, plan, &streamed.cells)
}

/// The gene IDs of `plan`'s output, in order: from the plan when it knows them (no
/// filter on data; nothing is read), else from a run of the plan.
pub fn gene_ids(
    py: Python<'_>,
    function: &str,
    plan: &Arc<Plan>,
    args: RunArgs<'_, '_>,
) -> PyResult<Vec<String>> {
    if plan.output_schema().n_genes().is_some() {
        args.check(py, function)?;
        return Ok(plan.genes().to_vec());
    }
    let streamed = stream(py, function, plan, Vec::new(), args)?;
    Ok(present_genes(plan, streamed.genes.as_ref()))
}

/// The IDs of the genes a run left present (`None`: every gene of the store).
fn present_genes(plan: &Plan, genes: Option<&Genes>) -> Vec<String> {
    let ordered = plan.root().gene_ids().ordered();
    match genes {
        None => ordered.to_vec(),
        Some(genes) => genes
            .positions()
            .iter()
            .map(|&position| ordered[position as usize].clone())
            .collect(),
    }
}

/// The number of cells (`Axis::Cells`) or genes of `plan`'s output: from the plan when
/// it knows it, else from a run of the plan.
pub fn count(
    py: Python<'_>,
    function: &str,
    plan: &Arc<Plan>,
    axis: Axis,
    args: RunArgs<'_, '_>,
) -> PyResult<u64> {
    if let Some(known) = plan.output_schema().axis_len(axis) {
        args.check(py, function)?;
        return Ok(known);
    }
    let streamed = stream(py, function, plan, Vec::new(), args)?;
    Ok(match axis {
        Axis::Cells => streamed.cells.len() as u64,
        Axis::Genes => streamed
            .genes
            .as_ref()
            .map_or(plan.root().schema().n_genes, |genes| genes.len() as u64),
    })
}

/// The feature names `plan` resolves, as a frame (`feature`, `kind`, `symbol`): every
/// cell column (`meta:<column>`), every gene of every layer that may remain
/// (`<layer>:<gene_id>`, with its symbol), and every component of every cell embedding
/// (`<embedding>:<component>`). Metadata only.
pub fn features(plan: &Plan) -> Result<DataFrame, polars::prelude::PolarsError> {
    let schema = plan.schema();
    let mut names: Vec<String> = Vec::new();
    let mut kinds: Vec<&str> = Vec::new();
    let mut symbols: Vec<Option<String>> = Vec::new();
    for column in &schema.cells.columns {
        names.push(format!("meta:{}", column.name));
        kinds.push("cell column");
        symbols.push(None);
    }
    let gene_ids = plan.root().gene_ids();
    let symbol_of: std::collections::HashMap<&str, &str> = gene_ids
        .symbols()
        .map(|symbols| {
            gene_ids
                .ordered()
                .iter()
                .zip(symbols)
                .filter_map(|(id, symbol)| symbol.map(|symbol| (id.as_str(), symbol)))
                .collect()
        })
        .unwrap_or_default();
    let removed = plan.removed_genes();
    for layer in &schema.layers {
        for gene in plan.genes() {
            if removed.removed_by(gene).is_some() {
                continue;
            }
            names.push(format!("{}:{gene}", layer.name));
            kinds.push("gene");
            symbols.push(
                symbol_of
                    .get(gene.as_str())
                    .map(|symbol| (*symbol).to_owned()),
            );
        }
    }
    for embedding in schema
        .embeddings
        .iter()
        .filter(|embedding| embedding.axis == Axis::Cells)
    {
        for component in 0..embedding.width {
            names.push(format!("{}:{component}", embedding.name));
            kinds.push("embedding component");
            symbols.push(None);
        }
    }
    DataFrame::new_infer_height(vec![
        Series::new("feature".into(), names).into_column(),
        Series::new("kind".into(), kinds).into_column(),
        Series::new("symbol".into(), symbols).into_column(),
    ])
}

/// The scanned store's `provenance.json` text (format.md section 9).
pub fn provenance(py: Python<'_>, plan: &Plan) -> PyResult<String> {
    let source = plan.root().source();
    let path = Path::new(source).join("provenance.json");
    py.detach(|| std::fs::read_to_string(&path))
        .map_err(|error| {
            let error = KytoError::StoreIo {
                operation: "read provenance.json".to_owned(),
                path: path.display().to_string(),
                detail: error.to_string(),
            };
            to_py_err(py, &error)
        })
}
