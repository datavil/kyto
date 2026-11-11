//! The node that adds or replaces columns of the cell or the gene table, each
//! computed from an expression (Polars' `with_columns`).
//!
//! Does: hold the named expressions, each resolved against the input when built (R6;
//! an expression sees the input's columns, not its siblings', as in Polars). Names
//! must be non-empty, distinct, and not reserved (`cell_id` in the cell table,
//! `gene_id` and `symbol` in the gene table: `KY-E215`). Its output schema replaces a column of the same name in place and
//! appends the others, each with no lineage (computed by this plan, so never stale)
//! and the dtype [`column_dtype`] declares. Changes no axis. Cell columns are
//! per-cell; gene columns are cross-cell when they read more than gene metadata
//! (`cell_profile`, gene embeddings), per-cell otherwise.
//! Does not: evaluate anything (the executor, which casts each result to its declared
//! dtype so data and schema agree).

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use kyto_core::{
    Axis, BinaryOp, CellExpr, ColumnDtype, ColumnSchema, Expr, ExprAxis, ExprNode, FloatDtype,
    GeneExpr, LayerDtype, Literal, Source,
};
use kyto_error::KytoError;

use crate::node::{CellScope, LogicalNode, Transform};
use crate::plan::Plan;
use crate::plan_schema::PlanSchema;
use crate::select_genes::quoted;

/// Reserved column names no plan writes, per table: the ID columns (format.md §5.2)
/// and the gene symbols the scan read for `by='symbol'` (task 3.9).
fn reserved(axis: Axis) -> &'static [&'static str] {
    match axis {
        Axis::Cells => &["cell_id"],
        Axis::Genes => &["gene_id", "symbol"],
    }
}

/// The named expressions of a [`WithColumns`], on one axis.
#[derive(Debug, Clone, PartialEq)]
pub enum Columns {
    /// Cell table columns.
    Cells(Vec<(String, CellExpr)>),
    /// Gene table columns.
    Genes(Vec<(String, GeneExpr)>),
}

/// Adds or replaces table columns computed from expressions.
#[derive(Debug, Clone, PartialEq)]
pub struct WithColumns {
    columns: Columns,
}

impl WithColumns {
    /// The named, resolved expressions.
    pub fn columns(&self) -> &Columns {
        &self.columns
    }

    /// The table it writes to.
    pub fn axis(&self) -> Axis {
        match self.columns {
            Columns::Cells(_) => Axis::Cells,
            Columns::Genes(_) => Axis::Genes,
        }
    }
}

impl LogicalNode for WithColumns {
    fn name(&self) -> &'static str {
        match self.axis() {
            Axis::Cells => "WITH_CELLS",
            Axis::Genes => "WITH_GENES",
        }
    }

    fn method(&self) -> &'static str {
        match self.axis() {
            Axis::Cells => "with_cells",
            Axis::Genes => "with_genes",
        }
    }

    fn changes_axis(&self) -> Option<Axis> {
        None
    }

    fn cell_scope(&self) -> CellScope {
        match &self.columns {
            Columns::Genes(columns) if columns.iter().any(|(_, expr)| !expr.is_metadata_only()) => {
                CellScope::CrossCell
            }
            Columns::Cells(_) | Columns::Genes(_) => CellScope::PerCell,
        }
    }

    fn describe(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.columns {
            Columns::Cells(columns) => write_columns(formatter, columns),
            Columns::Genes(columns) => write_columns(formatter, columns),
        }
    }
}

fn write_columns<A: ExprAxis>(
    formatter: &mut fmt::Formatter<'_>,
    columns: &[(String, Expr<A>)],
) -> fmt::Result {
    for (position, (name, expr)) in columns.iter().enumerate() {
        if position > 0 {
            formatter.write_str(", ")?;
        }
        write!(formatter, "{name:?} = {expr}")?;
    }
    Ok(())
}

impl Transform for WithColumns {
    fn output_schema(&self, input: &PlanSchema) -> PlanSchema {
        fn add<A: ExprAxis>(schema: PlanSchema, columns: &[(String, Expr<A>)]) -> PlanSchema {
            let input = schema.clone();
            columns.iter().fold(schema, |schema, (name, expr)| {
                let dtype = column_dtype(&input, A::AXIS, expr.node());
                schema.with_column(A::AXIS, ColumnSchema::new(name.clone(), dtype))
            })
        }
        match &self.columns {
            Columns::Cells(columns) => add(input.clone(), columns),
            Columns::Genes(columns) => add(input.clone(), columns),
        }
    }
}

/// The dtype a column computed by `node` on the table of `axis` is declared with.
///
/// Rules, close to Polars' (the executor casts to the declared dtype):
/// - a column read keeps the column's dtype (`Enum` included); a matrix read the
///   layer's (`float32`, `uint32`); an embedding component its dtype;
/// - literals: `Boolean`, `Int64`, `Float64`, `String`;
/// - comparisons, `&`, `|`, `~`, null tests and string tests are `Boolean`;
/// - `+`, `-`, `*`: equal dtypes stay; an int literal takes the other side's dtype; a
///   float literal takes a float side's dtype and makes an int side `Float64`; two
///   different ints are `Int64`; any other mix is `Float64`;
/// - `/` with a literal is `Float32` when the other side is `Float32`; otherwise
///   `Float32` when no side is wider than `float32` and one is `Float32`, and
///   `Float64` in every other case.
pub fn column_dtype(schema: &PlanSchema, axis: Axis, node: &ExprNode) -> ColumnDtype {
    match node {
        ExprNode::Source(source) => source_dtype(schema, axis, source),
        ExprNode::Literal(literal) => literal_dtype(literal),
        ExprNode::Not(_) | ExprNode::Str { .. } | ExprNode::Null { .. } => ColumnDtype::Boolean,
        ExprNode::Binary { op, left, right } => match op {
            BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::LtEq
            | BinaryOp::Gt
            | BinaryOp::GtEq
            | BinaryOp::And
            | BinaryOp::Or => ColumnDtype::Boolean,
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                let left_dtype = column_dtype(schema, axis, left);
                let right_dtype = column_dtype(schema, axis, right);
                let literal = |node: &ExprNode| matches!(node, ExprNode::Literal(_));
                if *op == BinaryOp::Div {
                    division_dtype((left_dtype, literal(left)), (right_dtype, literal(right)))
                } else {
                    arithmetic_dtype((left_dtype, literal(left)), (right_dtype, literal(right)))
                }
            }
        },
    }
}

fn source_dtype(schema: &PlanSchema, axis: Axis, source: &Source) -> ColumnDtype {
    match source {
        Source::Column { name } | Source::Recompute { name } => schema
            .column(axis, name)
            .map_or(ColumnDtype::Float64, |column| column.dtype),
        Source::Expression { layer, .. } | Source::Profile { layer, .. } => {
            match layer.as_deref().and_then(|layer| schema.layer(layer)) {
                Some(layer) => match layer.dtype {
                    LayerDtype::Float32 => ColumnDtype::Float32,
                    LayerDtype::Uint32 => ColumnDtype::UInt32,
                },
                None => ColumnDtype::Float32,
            }
        }
        Source::Embedding { name, .. } => match schema.embedding(name).map(|e| e.dtype) {
            Some(FloatDtype::Float64) => ColumnDtype::Float64,
            Some(FloatDtype::Float32) | None => ColumnDtype::Float32,
        },
        // Resolved expressions hold neither; a gene symbol reads a layer like a gene ID.
        Source::GeneSymbol { .. } | Source::Feature { .. } => ColumnDtype::Float32,
    }
}

fn literal_dtype(literal: &Literal) -> ColumnDtype {
    match literal {
        Literal::Boolean(_) => ColumnDtype::Boolean,
        Literal::Int(_) => ColumnDtype::Int64,
        Literal::Float(_) => ColumnDtype::Float64,
        Literal::String(_) => ColumnDtype::String,
    }
}

fn is_float(dtype: ColumnDtype) -> bool {
    matches!(dtype, ColumnDtype::Float32 | ColumnDtype::Float64)
}

fn arithmetic_dtype(left: (ColumnDtype, bool), right: (ColumnDtype, bool)) -> ColumnDtype {
    let ((left, left_literal), (right, right_literal)) = (left, right);
    if left == right {
        return left;
    }
    // A literal adapts to the other side, as Polars' dynamic literals do.
    for (literal, other, is_literal) in [(left, right, left_literal), (right, left, right_literal)]
    {
        if is_literal {
            return match (literal, is_float(other)) {
                (ColumnDtype::Int64, _) | (ColumnDtype::Float64, true) => other,
                _ => ColumnDtype::Float64,
            };
        }
    }
    if is_float(left) || is_float(right) {
        ColumnDtype::Float64
    } else {
        ColumnDtype::Int64
    }
}

fn division_dtype(left: (ColumnDtype, bool), right: (ColumnDtype, bool)) -> ColumnDtype {
    let ((left, left_literal), (right, right_literal)) = (left, right);
    // A literal adapts to the other side, as in `arithmetic_dtype`.
    match (left_literal, right_literal) {
        (true, false) => {
            return if right == ColumnDtype::Float32 {
                right
            } else {
                ColumnDtype::Float64
            };
        }
        (false, true) => {
            return if left == ColumnDtype::Float32 {
                left
            } else {
                ColumnDtype::Float64
            };
        }
        _ => {}
    }
    let narrow = |dtype| {
        matches!(
            dtype,
            ColumnDtype::Float32
                | ColumnDtype::Int8
                | ColumnDtype::Int16
                | ColumnDtype::UInt8
                | ColumnDtype::UInt16
        )
    };
    if narrow(left)
        && narrow(right)
        && (left == ColumnDtype::Float32 || right == ColumnDtype::Float32)
    {
        ColumnDtype::Float32
    } else {
        ColumnDtype::Float64
    }
}

impl Plan {
    /// This plan with cell columns added or replaced, each computed by its expression,
    /// resolved against this plan now (R6).
    ///
    /// # Errors
    /// Any error of [`Plan::resolve`]; `KY-E215` for an empty, repeated or reserved
    /// name; `KY-E216` if this plan ends in a sink.
    pub fn with_cells(
        self: &Arc<Self>,
        columns: impl IntoIterator<Item = (String, CellExpr)>,
    ) -> Result<Arc<Self>, KytoError> {
        self.check_open("with_cells")?;
        let columns = self.resolve_columns("with_cells", columns)?;
        Ok(self.extend(
            WithColumns {
                columns: Columns::Cells(columns),
            },
            Self::WithColumns,
        ))
    }

    /// This plan with gene columns added or replaced; as [`Plan::with_cells`].
    ///
    /// # Errors
    /// As [`Plan::with_cells`].
    pub fn with_genes(
        self: &Arc<Self>,
        columns: impl IntoIterator<Item = (String, GeneExpr)>,
    ) -> Result<Arc<Self>, KytoError> {
        self.check_open("with_genes")?;
        let columns = self.resolve_columns("with_genes", columns)?;
        Ok(self.extend(
            WithColumns {
                columns: Columns::Genes(columns),
            },
            Self::WithColumns,
        ))
    }

    fn resolve_columns<A: ExprAxis>(
        &self,
        method: &str,
        columns: impl IntoIterator<Item = (String, Expr<A>)>,
    ) -> Result<Vec<(String, Expr<A>)>, KytoError> {
        let mut names = HashSet::new();
        columns
            .into_iter()
            .map(|(name, expr)| {
                let problem = if name.is_empty() {
                    Some("a column name is never empty".to_owned())
                } else if reserved(A::AXIS).contains(&name.as_str()) {
                    Some(format!("'{name}' is reserved for the store"))
                } else if !names.insert(name.clone()) {
                    Some(format!("'{name}' is written twice"))
                } else {
                    None
                };
                if let Some(problem) = problem {
                    return Err(Self::invalid_argument(
                        method,
                        "name",
                        &quoted(&name),
                        &problem,
                        &format!(
                            "a distinct, non-empty name other than {}",
                            reserved(A::AXIS)
                                .iter()
                                .map(|name| format!("'{name}'"))
                                .collect::<Vec<_>>()
                                .join(" and ")
                        ),
                    ));
                }
                Ok((name, self.resolve(&expr)?))
            })
            .collect()
    }
}
