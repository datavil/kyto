//! The node that adds or replaces columns of the cell or the gene table, each
//! computed from an expression (Polars' `with_columns`).
//!
//! Does: hold the named expressions, each resolved against the input when built (R6;
//! an expression sees the input's columns, not its siblings', as in Polars). Names
//! must be distinct, valid column names (format.md §5.6: non-empty, no `:`) and not
//! reserved (`cell_id` and `gene_id` in either table, `symbol` in the gene table:
//! `KY-E215`). Its output schema replaces a column of the same name in place and
//! appends the others, each with no lineage (computed by this plan, so never stale)
//! and the dtype [`column_dtype`] declares (Polars' own inference). Changes no axis.
//! Cell columns are
//! per-cell; gene columns are cross-cell when they read more than gene metadata
//! (`cell_profile`, gene embeddings), per-cell otherwise.
//! Does not: evaluate anything (the executor, which casts each result to its declared
//! dtype so data and schema agree).

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use kyto_core::{
    Axis, CellExpr, ColumnDtype, ColumnSchema, Expr, ExprAxis, ExprNode, FloatDtype, GeneExpr,
    LayerDtype, Source,
};
use kyto_error::KytoError;
use polars::prelude::{DataFrame, DataType, IntoLazy, Series, col};

use crate::node::{CellScope, LogicalNode, Transform};
use crate::plan::Plan;
use crate::plan_schema::PlanSchema;
use crate::select_genes::quoted;

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
/// A plain read keeps its dtype: a column's (`Enum` included), a matrix read the
/// layer's (`float32` is `Float32`, `uint32` is `UInt32`), an embedding component its
/// dtype. Any other expression gets the dtype Polars infers for it, asked of Polars
/// itself with one empty stand-in column per leaf (`Enum` and `String` leaves stand in
/// as `String`), so supertypes, literals that adapt or widen, and true division follow
/// Polars exactly (`tests/column_dtype_polars.rs`). The executor casts each result to
/// the declared dtype. `Float64` if Polars cannot say (resolution refuses such
/// expressions first).
pub fn column_dtype(schema: &PlanSchema, axis: Axis, node: &ExprNode) -> ColumnDtype {
    match node {
        ExprNode::Source(source) => source_dtype(schema, axis, source),
        _ => inferred_dtype(schema, axis, node).unwrap_or(ColumnDtype::Float64),
    }
}

/// The dtype Polars infers for `node`, each leaf read as an empty column of its dtype.
fn inferred_dtype(schema: &PlanSchema, axis: Axis, node: &ExprNode) -> Option<ColumnDtype> {
    let leaves: RefCell<Vec<(Source, DataType)>> = RefCell::new(Vec::new());
    let leaf = |source: &Source| {
        let mut leaves = leaves.borrow_mut();
        let position = match leaves.iter().position(|(known, _)| known == source) {
            Some(position) => position,
            None => {
                let dtype = polars_dtype(source_dtype(schema, axis, source));
                leaves.push((source.clone(), dtype));
                leaves.len() - 1
            }
        };
        Some(col(format!("leaf{position}")))
    };
    let lowered = node.to_polars_with(&leaf)?.alias("out");
    let columns = leaves
        .into_inner()
        .into_iter()
        .enumerate()
        .map(|(position, (_, dtype))| {
            Series::new_empty(format!("leaf{position}").into(), &dtype).into()
        })
        .collect();
    let mut frame = DataFrame::new(0, columns).ok()?.lazy().select([lowered]);
    let schema = frame.collect_schema().ok()?;
    column_dtype_of(schema.get("out")?)
}

fn polars_dtype(dtype: ColumnDtype) -> DataType {
    match dtype {
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
        ColumnDtype::String | ColumnDtype::Enum { .. } => DataType::String,
    }
}

fn column_dtype_of(dtype: &DataType) -> Option<ColumnDtype> {
    Some(match dtype {
        DataType::Boolean => ColumnDtype::Boolean,
        DataType::Int8 => ColumnDtype::Int8,
        DataType::Int16 => ColumnDtype::Int16,
        DataType::Int32 => ColumnDtype::Int32,
        DataType::Int64 => ColumnDtype::Int64,
        DataType::UInt8 => ColumnDtype::UInt8,
        DataType::UInt16 => ColumnDtype::UInt16,
        DataType::UInt32 => ColumnDtype::UInt32,
        DataType::UInt64 => ColumnDtype::UInt64,
        DataType::Float32 => ColumnDtype::Float32,
        DataType::Float64 => ColumnDtype::Float64,
        DataType::String => ColumnDtype::String,
        _ => return None,
    })
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

impl Plan {
    /// This plan with cell columns added or replaced, each computed by its expression,
    /// resolved against this plan now (R6).
    ///
    /// # Errors
    /// Any error of [`Plan::resolve`]; `KY-E215` for a name that breaks format.md
    /// §5.6 (empty, or with `:`), is reserved or is repeated; `KY-E216` if this plan
    /// ends in a sink.
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
                Self::check_column_name(method, "name", A::AXIS, &name)?;
                if !names.insert(name.clone()) {
                    return Err(Self::invalid_argument(
                        method,
                        "name",
                        &quoted(&name),
                        &format!("'{name}' is written twice"),
                        "a name used once per call",
                    ));
                }
                Ok((name, self.resolve(&expr)?))
            })
            .collect()
    }
}
