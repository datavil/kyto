//! The node that keeps the cells, or the genes, where a boolean predicate is true.
//!
//! Does: hold a [`CellExpr`] or a [`GeneExpr`] predicate, resolved against the input
//! when the node is built (R6) and checked to be boolean (`KY-E205`). Changes the
//! predicate's axis: its output schema keeps every member and column, the axis length
//! becomes an upper bound, and every `nnz` over that axis a bound. Downstream, stored
//! derived columns of the other table are stale (`KY-E301`). Per-cell, except a gene
//! predicate that reads more than gene metadata (`cell_profile` needs one cell's
//! values before it keeps any gene; a gene embedding is counted the same way, to be
//! safe), which is cross-cell.
//! Does not: evaluate the predicate (the executor, 5.x) or know which rows it keeps,
//! so it adds nothing to `Plan::removed_genes` (task 3.7); a gene removed this way
//! raises `KY-E203` from the executor.

use std::fmt;
use std::sync::Arc;

use kyto_core::{Axis, CellExpr, Expr, ExprAxis, ExprDtype, GeneExpr};
use kyto_error::KytoError;

use crate::node::{CellScope, LogicalNode, Transform};
use crate::plan::Plan;
use crate::plan_schema::PlanSchema;

/// The predicate of a [`Filter`], on one axis.
#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    /// Keep the cells where it is true.
    Cells(CellExpr),
    /// Keep the genes where it is true.
    Genes(GeneExpr),
}

/// Keeps the cells or the genes where a boolean predicate is true (rows where it is
/// false or null are dropped).
#[derive(Debug, Clone, PartialEq)]
pub struct Filter {
    predicate: Predicate,
}

impl Filter {
    /// The resolved predicate.
    pub fn predicate(&self) -> &Predicate {
        &self.predicate
    }

    /// The axis it filters.
    pub fn axis(&self) -> Axis {
        match self.predicate {
            Predicate::Cells(_) => Axis::Cells,
            Predicate::Genes(_) => Axis::Genes,
        }
    }
}

impl LogicalNode for Filter {
    fn name(&self) -> &'static str {
        match self.axis() {
            Axis::Cells => "FILTER_CELLS",
            Axis::Genes => "FILTER_GENES",
        }
    }

    fn method(&self) -> &'static str {
        match self.axis() {
            Axis::Cells => "filter_cells",
            Axis::Genes => "filter_genes",
        }
    }

    fn changes_axis(&self) -> Option<Axis> {
        Some(self.axis())
    }

    fn cell_scope(&self) -> CellScope {
        match &self.predicate {
            Predicate::Genes(predicate) if !predicate.is_metadata_only() => CellScope::CrossCell,
            Predicate::Cells(_) | Predicate::Genes(_) => CellScope::PerCell,
        }
    }

    fn describe(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.predicate {
            Predicate::Cells(predicate) => write!(formatter, "{predicate}"),
            Predicate::Genes(predicate) => write!(formatter, "{predicate}"),
        }
    }
}

impl Transform for Filter {
    fn output_schema(&self, input: &PlanSchema) -> PlanSchema {
        let axis = self.axis();
        let bound = input.axis_len_bound(axis);
        input.clone().with_axis_changed(axis, None, bound)
    }
}

impl Plan {
    /// This plan keeping only the cells where `predicate` is true (a new plan; `self`
    /// is unchanged). The predicate is resolved against this plan now (R6) and must be
    /// boolean. Which cells it keeps is known only when the plan runs; every stored
    /// derived gene column read downstream is stale from here on (`KY-E301`).
    ///
    /// # Errors
    /// Any error of [`Plan::resolve`] for `predicate`; `KY-E205` if the resolved
    /// predicate is not boolean; `KY-E216` if this plan ends in a sink.
    pub fn filter_cells(self: &Arc<Self>, predicate: &CellExpr) -> Result<Arc<Self>, KytoError> {
        self.check_open("filter_cells")?;
        let predicate = self.boolean_predicate(predicate, "filter_cells", "cell")?;
        self.filter(Predicate::Cells(predicate))
    }

    /// This plan keeping only the genes where `predicate` is true (a new plan; `self`
    /// is unchanged). The predicate is resolved against this plan now (R6) and must be
    /// boolean. Which genes it keeps is known only when the plan runs, but every
    /// stored derived cell column read downstream is stale from here on (`KY-E301`).
    ///
    /// # Errors
    /// Any error of [`Plan::resolve`] for `predicate`; `KY-E205` if the resolved
    /// predicate is not boolean; `KY-E216` if this plan ends in a sink.
    pub fn filter_genes(self: &Arc<Self>, predicate: &GeneExpr) -> Result<Arc<Self>, KytoError> {
        self.check_open("filter_genes")?;
        let predicate = self.boolean_predicate(predicate, "filter_genes", "gene")?;
        self.filter(Predicate::Genes(predicate))
    }

    fn filter(self: &Arc<Self>, predicate: Predicate) -> Result<Arc<Self>, KytoError> {
        Ok(self.extend(Filter { predicate }, Self::Filter))
    }

    /// `predicate` resolved against this plan, checked to be boolean.
    fn boolean_predicate<A: ExprAxis>(
        &self,
        predicate: &Expr<A>,
        method: &str,
        axis_word: &str,
    ) -> Result<Expr<A>, KytoError> {
        let predicate = self.resolve(predicate)?;
        if predicate.dtype() != ExprDtype::Boolean {
            return Err(KytoError::ExpressionOperand {
                operation: method.to_owned(),
                accepted: format!("a boolean {axis_word} expression"),
                got: format!("a {} expression", predicate.dtype().name()),
                operand: predicate.to_string(),
            });
        }
        Ok(predicate)
    }
}
