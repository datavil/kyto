//! Recomputed columns (task 5.11, plan 3.8 Open (1)): the recipe of a stored derived
//! column, from the plan that wrote the store.
//!
//! Does: a kyto sink records the plan JSON that wrote a store in its `provenance.json`
//! (format.md §9) and, for a column a `with_cells`/`with_genes` computed from one layer,
//! `lineage` naming that layer and the node (format.md §8). [`Recipes`] reads that plan
//! back (`Plan::from_json`, no store read) and finds the expression of the node named
//! by a column's lineage; [`Recipes::substitute`] replaces each `.recompute()` leaf of
//! an expression with that expression, so the step computes the column again from the
//! layer as it is at that step, over the cells and genes present there (a QC metric
//! over the genes a filter kept). The expression was resolved when the writing plan was
//! built (R6) and computes the same dtype the plan declared for the column, so a store
//! read without filters recomputes exactly the stored values.
//! Does not: recompute a column without lineage (`KY-E302`, refused when the plan is
//! built) or one whose store holds no plan computing it (`KY-E303`: imported stores,
//! columns carried over from an earlier store, hand-written lineage).

use std::path::Path;
use std::sync::Arc;

use kyto_core::{AtlasSchema, Axis, ExprNode, Source};
use kyto_error::KytoError;
use kyto_plan::{Columns, Plan};

/// The recipes of a store's derived columns, the writing plan read on first use.
pub(crate) struct Recipes<'a> {
    store: &'a str,
    schema: &'a AtlasSchema,
    plan: Option<StoredPlan>,
}

/// The plan in a store's `provenance.json`, or why there is none.
#[derive(Clone)]
enum StoredPlan {
    Found(Arc<Plan>),
    Missing(String),
}

impl<'a> Recipes<'a> {
    /// The recipes of the store at `store`, whose schema is `schema`.
    pub(crate) fn new(store: &'a str, schema: &'a AtlasSchema) -> Self {
        Self {
            store,
            schema,
            plan: None,
        }
    }

    /// `node` with every `.recompute()` leaf of the `axis` table replaced by the
    /// expression that computed it; `node` itself (shared) when it has none.
    ///
    /// # Errors
    /// `KY-E302` for a column without lineage, `KY-E303` when the store's
    /// `provenance.json` holds no plan computing it.
    pub(crate) fn substitute(
        &mut self,
        node: &Arc<ExprNode>,
        axis: Axis,
    ) -> Result<Arc<ExprNode>, KytoError> {
        Ok(match &**node {
            ExprNode::Source(Source::Recompute { name }) => self.recipe(name, axis)?,
            ExprNode::Source(_) | ExprNode::Literal(_) => Arc::clone(node),
            ExprNode::Binary { op, left, right } => {
                let new_left = self.substitute(left, axis)?;
                let new_right = self.substitute(right, axis)?;
                if Arc::ptr_eq(&new_left, left) && Arc::ptr_eq(&new_right, right) {
                    Arc::clone(node)
                } else {
                    Arc::new(ExprNode::Binary {
                        op: *op,
                        left: new_left,
                        right: new_right,
                    })
                }
            }
            ExprNode::Not(input) => {
                let new = self.substitute(input, axis)?;
                if Arc::ptr_eq(&new, input) {
                    Arc::clone(node)
                } else {
                    Arc::new(ExprNode::Not(new))
                }
            }
            ExprNode::Str { function, input } => {
                let new = self.substitute(input, axis)?;
                if Arc::ptr_eq(&new, input) {
                    Arc::clone(node)
                } else {
                    Arc::new(ExprNode::Str {
                        function: function.clone(),
                        input: new,
                    })
                }
            }
            ExprNode::Null { test, input } => {
                let new = self.substitute(input, axis)?;
                if Arc::ptr_eq(&new, input) {
                    Arc::clone(node)
                } else {
                    Arc::new(ExprNode::Null {
                        test: *test,
                        input: new,
                    })
                }
            }
        })
    }

    /// The expression that computed column `name` of the `axis` table.
    fn recipe(&mut self, name: &str, axis: Axis) -> Result<Arc<ExprNode>, KytoError> {
        let (kind, function) = match axis {
            Axis::Cells => ("cell column", "cell_column"),
            Axis::Genes => ("gene column", "gene_column"),
        };
        let expression = format!("{function}('{name}').recompute()");
        let table = match axis {
            Axis::Cells => &self.schema.cells,
            Axis::Genes => &self.schema.genes,
        };
        let Some(lineage) = table
            .columns
            .iter()
            .find(|column| column.name == name)
            .and_then(|column| column.lineage.as_ref())
        else {
            return Err(KytoError::NoLineage {
                expression,
                kind: kind.to_owned(),
                column: name.to_owned(),
            });
        };
        let no_recipe = |problem: String| KytoError::NoRecipe {
            expression: expression.clone(),
            kind: kind.to_owned(),
            column: name.to_owned(),
            problem,
        };
        let plan = match self.plan() {
            StoredPlan::Found(plan) => plan,
            StoredPlan::Missing(problem) => {
                return Err(no_recipe(format!(
                    "{problem} (its lineage names node {})",
                    lineage.node
                )));
            }
        };
        let node = plan
            .nodes()
            .find(|node| node.id().to_string() == lineage.node);
        let found = match (node, axis) {
            (Some(Plan::WithColumns(node)), Axis::Cells) => match node.params().columns() {
                Columns::Cells(columns) => columns
                    .iter()
                    .find(|(column, _)| column == name)
                    .map(|(_, expr)| Arc::clone(expr.node())),
                Columns::Genes(_) => None,
            },
            (Some(Plan::WithColumns(node)), Axis::Genes) => match node.params().columns() {
                Columns::Genes(columns) => columns
                    .iter()
                    .find(|(column, _)| column == name)
                    .map(|(_, expr)| Arc::clone(expr.node())),
                Columns::Cells(_) => None,
            },
            _ => None,
        };
        found.ok_or_else(|| {
            no_recipe(format!(
                "node {} of the plan in provenance.json does not compute it (a column carried \
                 over from an earlier store keeps that store's lineage)",
                lineage.node
            ))
        })
    }

    /// The plan in the store's `provenance.json`, read once.
    fn plan(&mut self) -> StoredPlan {
        match &self.plan {
            Some(plan) => plan.clone(),
            None => {
                let plan = read_plan(self.store);
                self.plan = Some(plan.clone());
                plan
            }
        }
    }
}

fn read_plan(store: &str) -> StoredPlan {
    let path = Path::new(store).join("provenance.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            return StoredPlan::Missing(format!("{} cannot be read ({error})", path.display()));
        }
    };
    let provenance: serde_json::Value = match serde_json::from_str(&text) {
        Ok(provenance) => provenance,
        Err(error) => {
            return StoredPlan::Missing(format!("{} is not JSON ({error})", path.display()));
        }
    };
    match provenance.get("plan") {
        Some(serde_json::Value::Null) | None => StoredPlan::Missing(
            "provenance.json records no plan (the store was not written by a kyto sink)".to_owned(),
        ),
        Some(plan) => match Plan::from_json(&plan.to_string()) {
            Ok(plan) => StoredPlan::Found(plan),
            Err(error) => StoredPlan::Missing(format!(
                "the plan in provenance.json cannot be read ({error})"
            )),
        },
    }
}
