//! The node that keeps the genes named by ID, in the order given.
//!
//! Does: hold the gene IDs, each checked when built: a gene of the store (`KY-E210`),
//! not removed upstream (`KY-E203`), named once (`KY-E215`). Changes the gene axis:
//! its output schema has exactly that many genes, every layer `nnz` becomes a bound,
//! and every other gene is added to `Plan::removed_genes`, so reading it later raises
//! `KY-E203` at plan build (R9). Stored derived cell columns are stale downstream
//! (`KY-E301`). Per-cell: a projection, each cell keeps its values of these genes.
//! Does not: look genes up by symbol (3.9's `by=` is on `gene_expression` only) or
//! check against an earlier filter on data, whose kept genes are known only when the
//! plan runs (the executor raises `KY-E203` for a selected gene it dropped).

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use kyto_core::Axis;
use kyto_error::KytoError;
use serde::{Deserialize, Serialize};

use crate::node::{CellScope, LogicalNode, Transform};
use crate::plan::Plan;
use crate::plan_schema::PlanSchema;

/// How many IDs an `explain` line lists before it says how many more there are.
const SHOWN_IDS: usize = 5;

/// Keeps the genes named by ID, in the order given.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectGenes {
    gene_ids: Arc<[String]>,
}

impl SelectGenes {
    /// The selected gene IDs, in output order.
    pub fn gene_ids(&self) -> &[String] {
        &self.gene_ids
    }
}

impl LogicalNode for SelectGenes {
    fn name(&self) -> &'static str {
        "SELECT_GENES"
    }

    fn method(&self) -> &'static str {
        "select_genes"
    }

    fn changes_axis(&self) -> Option<Axis> {
        Some(Axis::Genes)
    }

    fn cell_scope(&self) -> CellScope {
        CellScope::PerCell
    }

    fn describe(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_ids(formatter, &self.gene_ids)
    }
}

impl Transform for SelectGenes {
    fn output_schema(&self, input: &PlanSchema) -> PlanSchema {
        let n_genes = self.gene_ids.len() as u64;
        input
            .clone()
            .with_axis_changed(Axis::Genes, Some(n_genes), n_genes)
    }

    fn selects_genes(&self) -> Option<&Arc<[String]>> {
        Some(&self.gene_ids)
    }
}

/// `['a', 'b', ...]`, listing at most [`SHOWN_IDS`] and then how many more.
pub(crate) fn write_ids(formatter: &mut fmt::Formatter<'_>, ids: &[String]) -> fmt::Result {
    formatter.write_str("[")?;
    for (position, id) in ids.iter().take(SHOWN_IDS).enumerate() {
        if position > 0 {
            formatter.write_str(", ")?;
        }
        write!(formatter, "{}", quoted(id))?;
    }
    if ids.len() > SHOWN_IDS {
        write!(formatter, ", ... {} more", ids.len() - SHOWN_IDS)?;
    }
    formatter.write_str("]")
}

/// `id` as a Python string literal (single quotes, as expressions render them).
pub(crate) fn quoted(id: &str) -> String {
    format!("'{}'", id.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// `KY-E215` if `ids` names one ID twice.
pub(crate) fn check_unique(method: &str, parameter: &str, ids: &[String]) -> Result<(), KytoError> {
    let mut seen = HashSet::with_capacity(ids.len());
    match ids.iter().find(|id| !seen.insert(id.as_str())) {
        Some(repeated) => Err(KytoError::InvalidArgument {
            function: method.to_owned(),
            parameter: parameter.to_owned(),
            value: format!("[..., {}, ...]", quoted(repeated)),
            problem: format!("{} is listed more than once", quoted(repeated)),
            accepted: "a list of distinct IDs".to_owned(),
        }),
        None => Ok(()),
    }
}

impl Plan {
    /// This plan keeping only the genes `gene_ids`, in that order (a new plan; `self`
    /// is unchanged). Every other gene is removed: reading it downstream raises
    /// `KY-E203`, and stored derived cell columns are stale (`KY-E301`).
    ///
    /// # Errors
    /// `KY-E210` for an ID that is no gene of the store; `KY-E203` for a gene removed
    /// upstream; `KY-E215` for an ID listed twice; `KY-E216` if this plan ends in a
    /// sink.
    pub fn select_genes(
        self: &Arc<Self>,
        gene_ids: impl IntoIterator<Item = String>,
    ) -> Result<Arc<Self>, KytoError> {
        self.check_open("select_genes")?;
        let gene_ids: Arc<[String]> = gene_ids.into_iter().collect();
        check_unique("select_genes", "gene_ids", &gene_ids)?;
        let state = self.state();
        for gene_id in gene_ids.iter() {
            if !state.gene_ids.contains(gene_id) {
                return Err(KytoError::UnknownName {
                    kind: "gene".to_owned(),
                    name: gene_id.clone(),
                    expression: format!("select_genes([..., {}, ...])", quoted(gene_id)),
                    problem: "no gene of the store has this gene_id".to_owned(),
                    suggestions: self.resolver().gene_suggestions(gene_id).into(),
                });
            }
            if let Some(removed_by) = state.removed_genes.removed_by(gene_id) {
                return Err(KytoError::RemovedGene {
                    gene: gene_id.clone(),
                    removed_by: removed_by.to_owned(),
                });
            }
        }
        Ok(self.extend(SelectGenes { gene_ids }, Self::SelectGenes))
    }
}
