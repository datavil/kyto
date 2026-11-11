//! Evaluating resolved expressions in the executor (task 5.8): the predicates of
//! filters and the columns of `with_cells`/`with_genes`.
//!
//! Does: run a resolved expression tree through Polars over a frame holding the table
//! columns it reads, with every other leaf (a gene's values in a block, a QC metric, a
//! cell's profile) given as a column of values computed beforehand; turn a boolean
//! result into a row mask (null is false, R9); cast a column to its declared dtype
//! (task 4.1: the plan declares it, the executor casts).
//! Does not: compute the leaves (the cell and gene steps do), or check names and
//! dtypes (the resolver did, when the plan was built).

use std::sync::Arc;

use kyto_core::{ColumnDtype, ExprNode, Source};
use kyto_error::KytoError;
use polars::prelude::{DataFrame, DataType, IntoLazy, Series, col};

/// A resolved expression and the table columns it reads.
#[derive(Debug, Clone)]
pub(crate) struct Prepared {
    pub(crate) node: Arc<ExprNode>,
    /// The table columns it reads, in first-read order, without repeats.
    pub(crate) columns: Vec<String>,
    /// Its other leaves, in first-read order, without repeats.
    pub(crate) leaves: Vec<Source>,
}

impl Prepared {
    /// `node` with its leaves listed.
    pub(crate) fn new(node: &Arc<ExprNode>) -> Self {
        let mut columns = Vec::new();
        let mut leaves: Vec<Source> = Vec::new();
        for source in sources(node) {
            match source {
                Source::Column { name } => {
                    if !columns.contains(name) {
                        columns.push(name.clone());
                    }
                }
                other => {
                    if !leaves.contains(other) {
                        leaves.push(other.clone());
                    }
                }
            }
        }
        Self {
            node: Arc::clone(node),
            columns,
            leaves,
        }
    }

    /// The values over `frame`'s rows, `frame` holding the columns it reads and
    /// `leaf_values` one series per leaf of [`Prepared::leaves`], in that order. `table`
    /// names the table in errors.
    ///
    /// # Errors
    /// `KY-E113` if Polars cannot evaluate it over the frame (a table that changed
    /// since the plan was built), or the result does not have one value per row.
    pub(crate) fn evaluate(
        &self,
        mut frame: DataFrame,
        leaf_values: Vec<Series>,
        table: &str,
    ) -> Result<Series, KytoError> {
        let corrupt = |problem: String| KytoError::CorruptTable {
            path: table.to_owned(),
            problem,
        };
        let height = frame.height();
        for (position, values) in leaf_values.into_iter().enumerate() {
            frame
                .with_column(values.with_name(leaf_name(position).into()).into())
                .map_err(|error| corrupt(error.to_string()))?;
        }
        let leaves = &self.leaves;
        let lowered = self
            .node
            .to_polars_with(&|source| match source {
                Source::Column { name } => Some(col(name.as_str())),
                other => leaves
                    .iter()
                    .position(|leaf| leaf == other)
                    .map(|position| col(leaf_name(position))),
            })
            .ok_or_else(|| corrupt("the expression does not lower to Polars".to_owned()))?;
        // Polars runs a one-row column as a scalar, through other float kernels than
        // longer columns (`x / c` rather than `x * (1 / c)`), so a block of one row
        // would give other bits (R11). One row is evaluated as two equal rows.
        if height == 1 {
            frame = frame
                .vstack(&frame)
                .map_err(|error| corrupt(error.to_string()))?;
        }
        let result = frame
            .lazy()
            .with_column(lowered.alias(VALUE))
            .select([col(VALUE)])
            .collect()
            .map_err(|error| corrupt(error.to_string()))?;
        let mut values = result
            .column(VALUE)
            .map_err(|error| corrupt(error.to_string()))?
            .as_materialized_series()
            .clone();
        if height == 1 {
            values = values.head(Some(1));
        }
        if values.len() != height {
            return Err(corrupt(format!(
                "the expression gave {} values for {height} rows",
                values.len()
            )));
        }
        Ok(values)
    }
}

/// The column every evaluation writes its result to.
const VALUE: &str = "__kyto_value";

fn leaf_name(position: usize) -> String {
    format!("__kyto_leaf{position}")
}

/// Every source leaf of `node`, in depth-first order (with repeats). Iterative: plan
/// JSON allows 1000-deep trees (memory/plan_json_floats_and_depth.md).
fn sources(node: &ExprNode) -> Vec<&Source> {
    let mut found = Vec::new();
    let mut stack = vec![node];
    while let Some(node) = stack.pop() {
        match node {
            ExprNode::Source(source) => found.push(source),
            ExprNode::Literal(_) => {}
            ExprNode::Binary { left, right, .. } => {
                stack.push(right);
                stack.push(left);
            }
            ExprNode::Not(input) | ExprNode::Str { input, .. } | ExprNode::Null { input, .. } => {
                stack.push(input);
            }
        }
    }
    found
}

/// The rows where `values` is true; null and false rows are dropped (R9).
///
/// # Errors
/// `KY-E113` if `values` is not boolean (`table` names the table).
pub(crate) fn mask(values: &Series, table: &str) -> Result<Vec<bool>, KytoError> {
    let values = values.bool().map_err(|error| KytoError::CorruptTable {
        path: table.to_owned(),
        problem: format!("a filter predicate gave {error}"),
    })?;
    Ok(values.iter().map(|value| value == Some(true)).collect())
}

/// `values` as a column of the declared `dtype`, named `name`. A string result stays as
/// it is for a declared `String` or `Enum` (a plain read of an `Enum` column keeps its
/// categories).
///
/// # Errors
/// `KY-E113` if Polars cannot cast it.
pub(crate) fn cast(
    values: Series,
    name: &str,
    dtype: &ColumnDtype,
    table: &str,
) -> Result<Series, KytoError> {
    let target = match dtype {
        ColumnDtype::Boolean => DataType::Boolean,
        ColumnDtype::Int8 => DataType::Int8,
        ColumnDtype::Int16 => DataType::Int16,
        ColumnDtype::Int32 => DataType::Int32,
        ColumnDtype::Int64 => DataType::Int64,
        ColumnDtype::UInt8 => DataType::UInt8,
        ColumnDtype::UInt16 => DataType::UInt16,
        ColumnDtype::UInt32 => DataType::UInt32,
        ColumnDtype::UInt64 => DataType::UInt64,
        ColumnDtype::Float32 => DataType::Float32,
        ColumnDtype::Float64 => DataType::Float64,
        ColumnDtype::String | ColumnDtype::Enum { .. } => {
            if matches!(
                values.dtype(),
                DataType::String | DataType::Enum(..) | DataType::Categorical(..)
            ) {
                return Ok(values.with_name(name.into()));
            }
            DataType::String
        }
    };
    let cast = values
        .cast(&target)
        .map_err(|error| KytoError::CorruptTable {
            path: table.to_owned(),
            problem: format!("column '{name}' cannot be cast to {target}: {error}"),
        })?;
    Ok(cast.with_name(name.into()))
}
