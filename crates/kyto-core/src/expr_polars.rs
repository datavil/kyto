//! Lowering metadata-only expressions to Polars `Expr`.
//!
//! Does: translate an expression whose leaves are table columns and literals into the
//! equivalent Polars expression, to run on the cell or gene table.
//! Does not: lower matrix or embedding sources (the executor reads those, tasks 5.x),
//! resolve or check column names (task 3.5), or pick the table (the planner does).

use polars::prelude::{DataType, Expr as PolarsExpr, Operator, binary_expr, col, lit};

use crate::expr::{BinaryOp, Expr, ExprAxis, ExprNode, Literal, NullTest, StrFunction};
use crate::expr_source::Source;

impl<A: ExprAxis> Expr<A> {
    /// The Polars expression computing this one on its axis's table, or `None` if it
    /// reads the matrix or an embedding (see [`Expr::is_metadata_only`]).
    ///
    /// Columns lower to `col(name)` with the name as written; `/` is true division; the
    /// input of a `str` function is cast to String first, so `Enum` columns work. Missing
    /// values are Polars nulls (R9, format.md §5.5) and follow Polars: a comparison or
    /// arithmetic with a null is null, `&` and `|` are three-valued (`null | true` is
    /// true), and a filter keeps only rows where the condition is true.
    pub fn to_polars(&self) -> Option<PolarsExpr> {
        lower(self.node(), &column_leaf)
    }
}

impl ExprNode {
    /// The Polars expression of this node with each leaf [`Source`] given by `leaf`
    /// (`None` from `leaf` makes the result `None`); operators lower as in
    /// [`Expr::to_polars`]. The plan uses it to ask Polars which dtype an expression
    /// over matrix or embedding reads has, with stand-in columns (task 4.2).
    pub fn to_polars_with(
        &self,
        leaf: &dyn Fn(&Source) -> Option<PolarsExpr>,
    ) -> Option<PolarsExpr> {
        lower(self, leaf)
    }
}

fn column_leaf(source: &Source) -> Option<PolarsExpr> {
    match source {
        Source::Column { name } => Some(col(name.as_str())),
        _ => None,
    }
}

#[recursive::recursive]
fn lower(node: &ExprNode, leaf: &dyn Fn(&Source) -> Option<PolarsExpr>) -> Option<PolarsExpr> {
    Some(match node {
        ExprNode::Source(source) => leaf(source)?,
        ExprNode::Literal(Literal::Boolean(value)) => lit(*value),
        ExprNode::Literal(Literal::Int(value)) => lit(*value),
        ExprNode::Literal(Literal::Float(value)) => lit(*value),
        ExprNode::Literal(Literal::String(value)) => lit(value.as_str()),
        ExprNode::Binary { op, left, right } => {
            let operator = match op {
                BinaryOp::Eq => Operator::Eq,
                BinaryOp::NotEq => Operator::NotEq,
                BinaryOp::Lt => Operator::Lt,
                BinaryOp::LtEq => Operator::LtEq,
                BinaryOp::Gt => Operator::Gt,
                BinaryOp::GtEq => Operator::GtEq,
                BinaryOp::Add => Operator::Plus,
                BinaryOp::Sub => Operator::Minus,
                BinaryOp::Mul => Operator::Multiply,
                BinaryOp::Div => Operator::TrueDivide,
                BinaryOp::And => Operator::LogicalAnd,
                BinaryOp::Or => Operator::LogicalOr,
            };
            binary_expr(lower(left, leaf)?, operator, lower(right, leaf)?)
        }
        ExprNode::Not(input) => lower(input, leaf)?.not(),
        ExprNode::Str { function, input } => {
            // Categorical (`Enum`) columns hold text too; the `str` functions need String.
            let strings = lower(input, leaf)?.cast(DataType::String).str();
            match function {
                StrFunction::StartsWith(prefix) => strings.starts_with(lit(prefix.as_str())),
                StrFunction::EndsWith(suffix) => strings.ends_with(lit(suffix.as_str())),
            }
        }
        ExprNode::Null { test, input } => match test {
            NullTest::IsNull => lower(input, leaf)?.is_null(),
            NullTest::IsNotNull => lower(input, leaf)?.is_not_null(),
        },
    })
}
