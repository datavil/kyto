//! The logical plan: an immutable chain of nodes, each holding its input by `Arc`.
//!
//! Does: build plans by wrapping an existing `Arc<Plan>` in a new node (the input is
//! shared, never copied or changed: R2), walk a plan from a node to its scan, and
//! answer what the plan's schema and default layer are.
//! Does not: execute, optimize or serialize plans (tasks 4.x).
//!
//! The shape follows Polars' `DslPlan`: one enum variant per node, each node owning its
//! inputs as `Arc<Plan>`, so building `b` from `a` costs one allocation and leaves `a`
//! untouched and still usable. Equality, `Debug`, `Display` and `Drop` walk the chain
//! with a loop, not recursion, so a long chain cannot overflow the stack.

use std::fmt;
use std::sync::Arc;

use kyto_core::{
    AtlasSchema, Expr, ExprAxis, ExprDtype, GeneExpr, GeneIds, RemovedGenes, Resolver,
};
use kyto_error::KytoError;

use crate::filter_genes::FilterGenes;
use crate::scan::Scan;
use crate::with_default_layer::WithDefaultLayer;

/// A logical plan node and, through its inputs, the plan below it.
///
/// Plans are immutable: methods that extend a plan take `&Arc<Plan>` and return a new
/// `Arc<Plan>` whose input is the old one.
#[derive(Clone)]
pub enum Plan {
    /// Read a store (the root of every plan).
    Scan(Scan),
    /// Set the default layer of everything downstream.
    WithDefaultLayer(WithDefaultLayer),
    /// Keep the genes where a gene predicate is true (4.1's `Filter`, gene axis only).
    FilterGenes(FilterGenes),
}

impl Plan {
    /// A plan that scans the store at `source`, whose schema is `schema` and whose
    /// genes are `gene_ids`.
    pub fn scan(source: impl Into<String>, schema: AtlasSchema, gene_ids: GeneIds) -> Arc<Self> {
        Arc::new(Self::Scan(Scan::new(source, schema, gene_ids)))
    }

    /// `expr` with every name resolved against this plan (R6): its schema, its store's
    /// gene IDs as read when the plan was built, and its default layer. The result
    /// names everything in full (`feature('CD3E')` becomes
    /// `gene_expression('CD3E', layer='log1p')`), so it means the same thing whatever is
    /// added to the store or the plan later. See [`Resolver::resolve`].
    ///
    /// A gene removed upstream ([`Plan::removed_genes`]) is refused with `KY-E203`; a
    /// stored derived cell column read after a node that changed the gene axis
    /// ([`Plan::gene_axis_changed_by`]) with `KY-E301`, unless it is `.recompute()`d.
    ///
    /// # Errors
    /// `KY-E201`, `KY-E203`, `KY-E205`, `KY-E210`, `KY-E211`, `KY-E109`, `KY-E301` or
    /// `KY-E302`, as [`Resolver::resolve`] lists them.
    pub fn resolve<A: ExprAxis>(&self, expr: &Expr<A>) -> Result<Expr<A>, KytoError> {
        let root = self.root();
        let removed_genes = self.removed_genes();
        let resolver = Resolver::new(
            self.schema(),
            root.gene_ids(),
            self.default_layer(),
            root.source(),
        )
        .with_removed_genes(&removed_genes);
        match self.gene_axis_changed_by() {
            Some(changed_by) => resolver.with_gene_axis_changed_by(changed_by),
            None => resolver,
        }
        .resolve(expr)
    }

    /// The first node from the scan downward that changes the gene axis, as messages
    /// name it (`filter_genes`), or `None` if no node does. Stored derived cell columns
    /// read after it are stale (`KY-E301`, task 3.8). 4.1's `SelectGenes` belongs here
    /// too.
    pub fn gene_axis_changed_by(&self) -> Option<&'static str> {
        let mut changed_by = None;
        for node in self.nodes() {
            match node {
                Self::FilterGenes(_) => changed_by = Some("filter_genes"),
                Self::Scan(_) | Self::WithDefaultLayer(_) => {}
            }
        }
        changed_by
    }

    /// The genes removed by nodes of this plan, each with the node that removed it, as
    /// known when the plan is built (R9). No node removes genes yet, so this is empty;
    /// task 4.1's `SelectGenes` adds its genes here (walking from the scan downward, see
    /// [`RemovedGenes::insert`]), and a filter on data leaves its check to the executor.
    pub fn removed_genes(&self) -> RemovedGenes {
        let removed = RemovedGenes::new();
        for node in self.nodes() {
            match node {
                // Which genes a filter on data keeps is known only when the plan runs.
                Self::Scan(_) | Self::WithDefaultLayer(_) | Self::FilterGenes(_) => {}
            }
        }
        removed
    }

    /// This plan with its default layer set to `layer` (a new plan; `self` is unchanged).
    ///
    /// # Errors
    /// `KY-E109` if the schema has no layer named `layer` (names are case-sensitive).
    pub fn with_default_layer(self: &Arc<Self>, layer: &str) -> Result<Arc<Self>, KytoError> {
        if !self
            .schema()
            .layers
            .iter()
            .any(|stored| stored.name == layer)
        {
            return Err(KytoError::MissingMember {
                kind: "layer".to_owned(),
                name: layer.to_owned(),
                store: self.root().source().to_owned(),
            });
        }
        Ok(Arc::new(Self::WithDefaultLayer(WithDefaultLayer::new(
            Arc::clone(self),
            layer.to_owned(),
        ))))
    }

    /// This plan keeping only the genes where `predicate` is true (a new plan; `self`
    /// is unchanged). The predicate is resolved against this plan now (R6) and must be
    /// boolean. Which genes it keeps is known only when the plan runs, but every
    /// stored derived cell column read downstream is stale from here on (`KY-E301`).
    ///
    /// # Errors
    /// Any error of [`Plan::resolve`] for `predicate`; `KY-E205` if the resolved
    /// predicate is not boolean.
    pub fn filter_genes(self: &Arc<Self>, predicate: &GeneExpr) -> Result<Arc<Self>, KytoError> {
        let predicate = self.resolve(predicate)?;
        if predicate.dtype() != ExprDtype::Boolean {
            return Err(KytoError::ExpressionOperand {
                operation: "filter_genes".to_owned(),
                accepted: "a boolean gene expression".to_owned(),
                got: format!("a {} expression", predicate.dtype().name()),
                operand: predicate.to_string(),
            });
        }
        Ok(Arc::new(Self::FilterGenes(FilterGenes::new(
            Arc::clone(self),
            predicate,
        ))))
    }

    /// The node's name, as `explain` will show it.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Scan(_) => "SCAN",
            Self::WithDefaultLayer(_) => "WITH_DEFAULT_LAYER",
            Self::FilterGenes(_) => "FILTER_GENES",
        }
    }

    /// The node's input, or `None` for a scan.
    pub fn input(&self) -> Option<&Arc<Self>> {
        match self {
            Self::Scan(_) => None,
            Self::WithDefaultLayer(node) => node.input.get(),
            Self::FilterGenes(node) => node.input.get(),
        }
    }

    /// This node, then its input, and so on down to the scan.
    pub fn nodes(&self) -> PlanNodes<'_> {
        PlanNodes { next: Some(self) }
    }

    /// The plan's output schema. Each node keeps its own (shared, not copied), so this
    /// does not walk the chain. No node changes the schema yet (4.1 adds those; a gene
    /// filter keeps its input's schema, see [`FilterGenes`]).
    pub fn schema(&self) -> &Arc<AtlasSchema> {
        match self {
            Self::Scan(scan) => scan.schema(),
            Self::WithDefaultLayer(node) => node.schema(),
            Self::FilterGenes(node) => node.schema(),
        }
    }

    /// The default layer set nearest to this node, or `None` if no node sets one.
    pub fn default_layer(&self) -> Option<&str> {
        self.nodes().find_map(|node| match node {
            Self::WithDefaultLayer(node) => Some(node.layer()),
            Self::Scan(_) | Self::FilterGenes(_) => None,
        })
    }

    /// The scan at the root of the plan (walks the chain).
    pub fn root(&self) -> &Scan {
        let mut node = self;
        loop {
            match node {
                Self::Scan(scan) => return scan,
                Self::WithDefaultLayer(_) | Self::FilterGenes(_) => match node.input() {
                    Some(input) => node = input,
                    // Only a plan being dropped has no input; it is never reachable.
                    None => unreachable!("plan node without input"),
                },
            }
        }
    }

    /// Whether the two nodes are equal, ignoring their inputs.
    fn same_node(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Scan(left), Self::Scan(right)) => left == right,
            (Self::WithDefaultLayer(left), Self::WithDefaultLayer(right)) => {
                left.layer() == right.layer()
            }
            (Self::FilterGenes(left), Self::FilterGenes(right)) => {
                left.predicate() == right.predicate()
            }
            _ => false,
        }
    }

    /// One line describing this node alone.
    fn describe(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scan(scan) => {
                let schema = scan.schema();
                write!(
                    formatter,
                    "{} {:?} ({} cells x {} genes)",
                    self.name(),
                    scan.source(),
                    schema.n_cells,
                    schema.n_genes
                )
            }
            Self::WithDefaultLayer(node) => {
                write!(formatter, "{} {:?}", self.name(), node.layer())
            }
            Self::FilterGenes(node) => {
                write!(formatter, "{} {}", self.name(), node.predicate())
            }
        }
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

/// One line per node, from this node down to the scan.
impl fmt::Display for Plan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (position, node) in self.nodes().enumerate() {
            if position > 0 {
                formatter.write_str("\n")?;
            }
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
        let mut next = self.take_input();
        while let Some(input) = next {
            // Shared inputs stay alive for their other owners; stop there. `into_inner`
            // (not `try_unwrap`) hands the node to exactly one of two threads dropping a
            // shared tail at once, so neither falls back to a recursive drop.
            next = Arc::into_inner(input).and_then(|mut plan| plan.take_input());
        }
    }
}

impl Plan {
    fn take_input(&mut self) -> Option<Arc<Self>> {
        match self {
            Self::Scan(_) => None,
            Self::WithDefaultLayer(node) => node.input.take(),
            Self::FilterGenes(node) => node.input.take(),
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

/// A node's input. Always set, except while the node is being dropped (see `Drop`).
#[derive(Clone)]
pub(crate) struct Input(Option<Arc<Plan>>);

impl Input {
    pub(crate) fn new(plan: Arc<Plan>) -> Self {
        Self(Some(plan))
    }

    pub(crate) fn get(&self) -> Option<&Arc<Plan>> {
        self.0.as_ref()
    }

    fn take(&mut self) -> Option<Arc<Plan>> {
        self.0.take()
    }
}

/// The input is shown by the plan's own `Debug`, one line per node; not here.
impl fmt::Debug for Input {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Input(..)")
    }
}
