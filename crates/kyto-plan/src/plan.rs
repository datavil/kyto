//! The logical plan: an immutable chain of nodes, each holding its input by `Arc`.
//!
//! Does: list the logical nodes (task 4.1: one variant each, one file each), build
//! plans by wrapping an existing `Arc<Plan>` in a new node (the input is shared, never
//! copied or changed: R2), walk a plan from a node to its scan, and answer what a
//! node's output schema, default layer and removed genes are from the state cached
//! when it was built (so these answers do not walk the chain).
//! Does not: serialize (`plan_json.rs`, 4.3), optimize (`optimize.rs`, 4.4) or execute
//! plans. Build-time validation (4.2) happens in each step's builder as the plan grows:
//! names resolved, kinds checked, arguments and new names checked (`names.rs`), so a
//! built plan is valid.
//!
//! The shape follows Polars' `DslPlan`: one enum variant per node, each node owning its
//! input as `Arc<Plan>`, so building `b` from `a` costs one allocation and leaves `a`
//! untouched and still usable. Like Polars' `IR::schema()`, each node's output schema
//! is computed from its input's and its parameters ([`Transform::output_schema`]), once.
//! Equality, `Debug`, `Display` and `Drop` walk the chain with a loop, not recursion,
//! so a long chain cannot overflow the stack.
//!
//! Adding a node: a file with its parameters, their [`LogicalNode`] and [`Transform`]
//! impls and its builder (`impl Plan`), then one variant here and one arm in each of
//! `parts`, `input_mut` and `same_node`, its `Step` in `plan_json.rs` (written from
//! the parameters, read back through the builder) and its method in
//! `plan_json.schema.json` (8.4; the plan JSON corpus test fails until both match), and
//! what it reads and writes in
//! `members.rs` (the optimizer's rules move and leave out nodes by it, 4.4).

use std::fmt;
use std::sync::Arc;

use kyto_core::{Axis, Expr, ExprAxis, LayerSchema, RemovedGenes, Resolver};
use kyto_error::KytoError;

use crate::cluster::Cluster;
use crate::cross_cell::CrossCell;
use crate::filter::Filter;
use crate::map_elementwise::MapElementwise;
use crate::neighbors::Neighbors;
use crate::node::{CellScope, Input, LogicalNode, Node, NodeId, NodeState, Transform};
use crate::plan_schema::PlanSchema;
use crate::scan::Scan;
use crate::select_genes::SelectGenes;
use crate::sink::Sink;
use crate::take_cells::TakeCells;
use crate::with_columns::WithColumns;
use crate::with_default_layer::WithDefaultLayer;

/// A logical plan node and, through its input, the plan below it.
///
/// Plans are immutable: methods that extend a plan take `&Arc<Plan>` and return a new
/// `Arc<Plan>` whose input is the old one.
#[derive(Clone)]
pub enum Plan {
    /// Read a store (the root of every plan).
    Scan(Node<Scan>),
    /// Set the default layer of everything downstream.
    WithDefaultLayer(Node<WithDefaultLayer>),
    /// Keep the cells or the genes where a predicate is true.
    Filter(Node<Filter>),
    /// Keep the genes named by ID.
    SelectGenes(Node<SelectGenes>),
    /// Keep the cells named by ID.
    TakeCells(Node<TakeCells>),
    /// Map a layer value by value (`normalize_total`, `log1p`).
    MapElementwise(Node<MapElementwise>),
    /// A step over all cells (`scale`, `highly_variable_genes`, `pca`).
    CrossCell(Node<CrossCell>),
    /// Add or replace table columns from expressions.
    WithColumns(Node<WithColumns>),
    /// Build the neighbour graphs of an embedding.
    Neighbors(Node<Neighbors>),
    /// Cluster cells on a graph.
    Cluster(Node<Cluster>),
    /// Write the plan (the last node).
    Sink(Node<Sink>),
}

impl Plan {
    /// A plan that scans the store at `source`, whose schema is `schema` and whose
    /// genes are `gene_ids`.
    pub fn scan(
        source: impl Into<String>,
        schema: kyto_core::AtlasSchema,
        gene_ids: kyto_core::GeneIds,
    ) -> Arc<Self> {
        let scan = Scan::new(source, schema, gene_ids);
        let state = NodeState::root(
            scan.shared_source(),
            scan.output_schema(),
            Arc::clone(scan.gene_ids()),
        );
        Arc::new(Self::Scan(Node::new(Input::none(), scan, state)))
    }

    /// This node's declarations and cached state.
    fn parts(&self) -> (&dyn LogicalNode, &NodeState, &Input) {
        fn of<P: LogicalNode>(node: &Node<P>) -> (&dyn LogicalNode, &NodeState, &Input) {
            (node.params(), &node.state, &node.input)
        }
        match self {
            Self::Scan(node) => of(node),
            Self::WithDefaultLayer(node) => of(node),
            Self::Filter(node) => of(node),
            Self::SelectGenes(node) => of(node),
            Self::TakeCells(node) => of(node),
            Self::MapElementwise(node) => of(node),
            Self::CrossCell(node) => of(node),
            Self::WithColumns(node) => of(node),
            Self::Neighbors(node) => of(node),
            Self::Cluster(node) => of(node),
            Self::Sink(node) => of(node),
        }
    }

    fn input_mut(&mut self) -> &mut Input {
        match self {
            Self::Scan(node) => &mut node.input,
            Self::WithDefaultLayer(node) => &mut node.input,
            Self::Filter(node) => &mut node.input,
            Self::SelectGenes(node) => &mut node.input,
            Self::TakeCells(node) => &mut node.input,
            Self::MapElementwise(node) => &mut node.input,
            Self::CrossCell(node) => &mut node.input,
            Self::WithColumns(node) => &mut node.input,
            Self::Neighbors(node) => &mut node.input,
            Self::Cluster(node) => &mut node.input,
            Self::Sink(node) => &mut node.input,
        }
    }

    /// Whether the two nodes have equal parameters, ignoring their inputs.
    pub(crate) fn same_node(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Scan(left), Self::Scan(right)) => left.params() == right.params(),
            (Self::WithDefaultLayer(left), Self::WithDefaultLayer(right)) => {
                left.params() == right.params()
            }
            (Self::Filter(left), Self::Filter(right)) => left.params() == right.params(),
            (Self::SelectGenes(left), Self::SelectGenes(right)) => left.params() == right.params(),
            (Self::TakeCells(left), Self::TakeCells(right)) => left.params() == right.params(),
            (Self::MapElementwise(left), Self::MapElementwise(right)) => {
                left.params() == right.params()
            }
            (Self::CrossCell(left), Self::CrossCell(right)) => left.params() == right.params(),
            (Self::WithColumns(left), Self::WithColumns(right)) => left.params() == right.params(),
            (Self::Neighbors(left), Self::Neighbors(right)) => left.params() == right.params(),
            (Self::Cluster(left), Self::Cluster(right)) => left.params() == right.params(),
            (Self::Sink(left), Self::Sink(right)) => left.params() == right.params(),
            _ => false,
        }
    }

    pub(crate) fn state(&self) -> &NodeState {
        self.parts().1
    }

    /// The node's declarations: name, axis changed, per-cell or cross-cell.
    pub fn node(&self) -> &dyn LogicalNode {
        self.parts().0
    }

    /// The node's stable ID: its position from the scan (0).
    pub fn id(&self) -> NodeId {
        self.state().id
    }

    /// The node's name, as `explain` shows it.
    pub fn name(&self) -> &'static str {
        self.node().name()
    }

    /// The axis this node changes, if any.
    pub fn changes_axis(&self) -> Option<Axis> {
        self.node().changes_axis()
    }

    /// Whether this node works per cell or across cells.
    pub fn cell_scope(&self) -> CellScope {
        self.node().cell_scope()
    }

    /// The node's input, or `None` for a scan.
    pub fn input(&self) -> Option<&Arc<Self>> {
        self.parts().2.get()
    }

    /// This node, then its input, and so on down to the scan.
    pub fn nodes(&self) -> PlanNodes<'_> {
        PlanNodes { next: Some(self) }
    }

    /// The plan's output schema: members, columns and sizes (cached, no walk).
    pub fn output_schema(&self) -> &PlanSchema {
        &self.state().schema
    }

    /// The plan's members and columns. Its `n_cells`, `n_genes` and `nnz` are upper
    /// bounds; [`Plan::output_schema`] says which are exact.
    pub fn schema(&self) -> &kyto_core::AtlasSchema {
        self.output_schema().atlas()
    }

    /// The genes that may remain, in order: the store's, or a selection's
    /// ([`Plan::select_genes`]). Exactly these when the gene count is known; after a
    /// filter on data, an upper bound.
    pub fn genes(&self) -> &[String] {
        self.state().genes()
    }

    /// The default layer set nearest to this node, or `None` if no node sets one.
    pub fn default_layer(&self) -> Option<&str> {
        self.state().default_layer.as_deref()
    }

    /// The scan at the root of the plan (walks the chain).
    pub fn root(&self) -> &Scan {
        let mut node = self;
        loop {
            if let Self::Scan(scan) = node {
                return scan.params();
            }
            match node.input() {
                Some(input) => node = input,
                // Only a plan being dropped has no input; it is never reachable.
                None => unreachable!("plan node without input"),
            }
        }
    }

    /// The first node from the scan that changes the gene axis, as messages name it
    /// (`filter_genes`, `select_genes`), or `None` if no node does. Stored derived
    /// cell columns read after it are stale (`KY-E301`, task 3.8).
    pub fn gene_axis_changed_by(&self) -> Option<&'static str> {
        self.state().gene_axis_changed_by
    }

    /// The first node from the scan that changes the cell axis (`filter_cells`,
    /// `take_cells`), or `None`. Stored derived gene columns read after it are stale.
    pub fn cell_axis_changed_by(&self) -> Option<&'static str> {
        self.state().cell_axis_changed_by
    }

    /// The genes removed by nodes of this plan, each with the first node that removed
    /// it, as known when the plan is built (R9): those a [`Plan::select_genes`] left
    /// out. A filter on data removes genes known only when the plan runs.
    pub fn removed_genes(&self) -> &RemovedGenes {
        &self.state().removed_genes
    }

    /// `expr` with every name resolved against this plan (R6): its schema, its store's
    /// gene IDs as read when the plan was built, and its default layer. The result
    /// names everything in full (`feature('CD3E')` becomes
    /// `gene_expression('CD3E', layer='log1p')`), so it means the same thing whatever is
    /// added to the store or the plan later. See [`Resolver::resolve`].
    ///
    /// A gene removed upstream ([`Plan::removed_genes`]) is refused with `KY-E203`. A
    /// stored derived column (lineage in `schema.json`) is refused with `KY-E301`,
    /// unless it is `.recompute()`d, when read after a node that changed the other
    /// axis ([`Plan::gene_axis_changed_by`], [`Plan::cell_axis_changed_by`]) or wrote
    /// its lineage layer again.
    ///
    /// # Errors
    /// `KY-E201`, `KY-E203`, `KY-E204`, `KY-E205`, `KY-E210`, `KY-E211`, `KY-E109`,
    /// `KY-E301` or `KY-E302`, as [`Resolver::resolve`] lists them.
    pub fn resolve<A: ExprAxis>(&self, expr: &Expr<A>) -> Result<Expr<A>, KytoError> {
        let state = self.state();
        let mut resolver = Resolver::new(
            state.schema.atlas(),
            &state.gene_ids,
            state.default_layer.as_deref(),
            &state.source,
        )
        .with_removed_genes(&state.removed_genes)
        .with_rewritten_layers(&state.rewritten_layers);
        if let Some(changed_by) = state.gene_axis_changed_by {
            resolver = resolver.with_gene_axis_changed_by(changed_by);
        }
        if let Some(changed_by) = state.cell_axis_changed_by {
            resolver = resolver.with_cell_axis_changed_by(changed_by);
        }
        resolver.resolve(expr)
    }

    /// The new plan of node `params` over this one, wrapped by `wrap` (the variant).
    /// The caller has checked the parameters.
    pub(crate) fn extend<P: Transform>(
        self: &Arc<Self>,
        params: P,
        wrap: fn(Node<P>) -> Self,
    ) -> Arc<Self> {
        let state = NodeState::child(self.state(), &params);
        Arc::new(wrap(Node::new(Input::new(Arc::clone(self)), params, state)))
    }

    /// `KY-E216` if this plan ends in a sink, so `method` cannot extend it.
    pub(crate) fn check_open(&self, method: &str) -> Result<(), KytoError> {
        match self {
            Self::Sink(sink) => Err(KytoError::PlanEnded {
                method: method.to_owned(),
                sink: sink.params().method().to_owned(),
            }),
            _ => Ok(()),
        }
    }

    /// `KY-E109` for a member of kind `kind` named `name` that the plan does not have.
    pub(crate) fn missing_member(&self, kind: &str, name: &str) -> KytoError {
        KytoError::MissingMember {
            kind: kind.to_owned(),
            name: name.to_owned(),
            store: self.state().source.to_string(),
        }
    }

    /// The layer `name` of the plan, or `KY-E109`.
    pub(crate) fn layer_named(&self, name: &str) -> Result<&LayerSchema, KytoError> {
        self.output_schema()
            .layer(name)
            .ok_or_else(|| self.missing_member("layer", name))
    }

    /// The layer a step `method` reads: `layer`, checked (`KY-E109`), or the default
    /// layer written in full (R6; `KY-E211` without one).
    pub(crate) fn input_layer(
        &self,
        method: &str,
        layer: Option<&str>,
    ) -> Result<String, KytoError> {
        match layer {
            Some(layer) => Ok(self.layer_named(layer)?.name.clone()),
            None => {
                self.default_layer()
                    .map(str::to_owned)
                    .ok_or_else(|| KytoError::NoDefaultLayer {
                        expression: format!("{method}(layer=None)"),
                        layers: self
                            .schema()
                            .layers
                            .iter()
                            .map(|layer| layer.name.clone())
                            .collect(),
                    })
            }
        }
    }

    /// `KY-E215` for argument `parameter` of `method`.
    pub(crate) fn invalid_argument(
        method: &str,
        parameter: &str,
        value: &str,
        problem: &str,
        accepted: &str,
    ) -> KytoError {
        KytoError::InvalidArgument {
            function: method.to_owned(),
            parameter: parameter.to_owned(),
            value: value.to_owned(),
            problem: problem.to_owned(),
            accepted: accepted.to_owned(),
        }
    }

    /// `KY-E215` unless `value` is positive and finite.
    pub(crate) fn check_positive(
        method: &str,
        parameter: &str,
        value: f64,
    ) -> Result<(), KytoError> {
        if value.is_finite() && value > 0.0 {
            return Ok(());
        }
        Err(Self::invalid_argument(
            method,
            parameter,
            &value.to_string(),
            "it is not a positive finite number",
            "a positive finite float",
        ))
    }

    /// `KY-E215` if the name `value` of `parameter` is empty.
    pub(crate) fn check_name(method: &str, parameter: &str, value: &str) -> Result<(), KytoError> {
        if !value.is_empty() {
            return Ok(());
        }
        Err(Self::invalid_argument(
            method,
            parameter,
            "''",
            "names are never empty",
            "a non-empty name",
        ))
    }
}

/// Plans are equal when their chains have equal nodes, in the same order.
impl PartialEq for Plan {
    fn eq(&self, other: &Self) -> bool {
        let mut left = self.nodes();
        let mut right = other.nodes();
        loop {
            match (left.next(), right.next()) {
                (None, None) => return true,
                (Some(left_node), Some(right_node)) => {
                    if std::ptr::eq(left_node, right_node) {
                        return true;
                    }
                    if !left_node.same_node(right_node) {
                        return false;
                    }
                }
                _ => return false,
            }
        }
    }
}

/// One line per node, from this node down to the scan: its name and parameters.
impl fmt::Display for Plan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (position, node) in self.nodes().enumerate() {
            if position > 0 {
                formatter.write_str("\n")?;
            }
            let node = node.node();
            write!(formatter, "{} ", node.name())?;
            node.describe(formatter)?;
        }
        Ok(())
    }
}

impl fmt::Debug for Plan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Plan(\n{self}\n)")
    }
}

/// Drop the chain with a loop: a recursive drop of a long chain overflows the stack.
impl Drop for Plan {
    fn drop(&mut self) {
        let mut next = self.input_mut().take();
        while let Some(input) = next {
            // Shared inputs stay alive for their other owners; stop there. `into_inner`
            // (not `try_unwrap`) hands the node to exactly one of two threads dropping a
            // shared tail at once, so neither falls back to a recursive drop.
            next = Arc::into_inner(input).and_then(|mut plan| plan.input_mut().take());
        }
    }
}

/// Iterator over a plan's nodes, from a node down to its scan.
#[derive(Debug, Clone)]
pub struct PlanNodes<'a> {
    next: Option<&'a Plan>,
}

impl<'a> Iterator for PlanNodes<'a> {
    type Item = &'a Plan;

    fn next(&mut self) -> Option<&'a Plan> {
        let node = self.next?;
        self.next = node.input().map(|input| &**input);
        Some(node)
    }
}
