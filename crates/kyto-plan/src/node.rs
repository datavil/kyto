//! What every logical node declares, and the state each node caches when built.
//!
//! Does: define [`LogicalNode`] (name, the axis a node changes, per-cell or
//! cross-cell) and [`Transform`] (the output-schema function of a node with an input),
//! the generic [`Node`] that holds a node's input, parameters and cached state, its
//! stable [`NodeId`], and the [`NodeState`] derived once per node from its input's:
//! output schema, genes removed, which nodes changed an axis or wrote a layer again,
//! the default layer. Caching these makes building a chain of N nodes O(N), not
//! O(N^2) (task 3.8's open item).
//! Does not: validate parameters against the schema (each builder does, before the
//! node exists) or serialize plans (`plan_json.rs`, task 4.3).

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::sync::Arc;

use kyto_core::{Axis, GeneIds, RemovedGenes};
use serde::{Deserialize, Serialize};

use crate::plan::Plan;
use crate::plan_schema::PlanSchema;

/// A node's position in its plan: 0 for the scan, then one more per node above it.
/// Stable: it depends only on the chain below the node, never on what is built on
/// top, so the same plan always numbers its nodes the same way (the plan JSON of 4.3
/// and a column's `lineage.node` refer to nodes by it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub u64);

impl fmt::Display for NodeId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// How a node uses cells: one at a time, or all of them before it outputs anything.
/// Predicate pushdown (4.4) moves cell filters across per-cell nodes only, and the
/// physical plan (4.5) gives a cross-cell node its own passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CellScope {
    /// Each cell's output depends on that cell alone: streams block by block.
    PerCell,
    /// The output depends on statistics over all cells (a pass before the output).
    CrossCell,
}

/// What every logical node declares (task 4.1).
pub trait LogicalNode {
    /// The node's name in `explain` (`FILTER_CELLS`, `PCA`).
    fn name(&self) -> &'static str;

    /// The plan method that builds it, as error messages name the node
    /// (`filter_cells`, `pca`).
    fn method(&self) -> &'static str;

    /// The axis whose length or order the node changes, if any.
    fn changes_axis(&self) -> Option<Axis>;

    /// Whether the node works per cell or across cells.
    fn cell_scope(&self) -> CellScope;

    /// The node's parameters, written after its name on its `explain` line.
    fn describe(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result;
}

/// A node with an input: its output schema follows from the input's and its own
/// parameters alone.
pub trait Transform: LogicalNode {
    /// The schema the node outputs, given its input's.
    fn output_schema(&self, input: &PlanSchema) -> PlanSchema;

    /// The layer the node writes, if it maps to one (stored columns computed from that
    /// layer become stale when it already existed).
    fn writes_layer(&self) -> Option<&str> {
        None
    }

    /// The default layer the node sets for everything downstream.
    fn sets_default_layer(&self) -> Option<&str> {
        None
    }

    /// The genes the node keeps by ID, in output order; every other gene is removed
    /// (`KY-E203` on a later read).
    fn selects_genes(&self) -> Option<&Arc<[String]>> {
        None
    }
}

/// A plan node: its input (none for a scan), its parameters `P` and the state cached
/// when it was built.
#[derive(Debug, Clone)]
pub struct Node<P> {
    pub(crate) input: Input,
    params: P,
    pub(crate) state: NodeState,
}

impl<P> Node<P> {
    pub(crate) fn new(input: Input, params: P, state: NodeState) -> Self {
        Self {
            input,
            params,
            state,
        }
    }

    /// The node's parameters.
    pub fn params(&self) -> &P {
        &self.params
    }

    /// The node's input, or `None` for a scan.
    pub fn input(&self) -> Option<&Arc<Plan>> {
        self.input.get()
    }

    /// The node's stable ID.
    pub fn id(&self) -> NodeId {
        self.state.id
    }

    /// The node's output schema.
    pub fn schema(&self) -> &PlanSchema {
        &self.state.schema
    }
}

/// A node's input. Always set, except for a scan and while the node is being dropped
/// (see `Plan`'s `Drop`).
#[derive(Clone)]
pub(crate) struct Input(Option<Arc<Plan>>);

impl Input {
    pub(crate) fn new(plan: Arc<Plan>) -> Self {
        Self(Some(plan))
    }

    pub(crate) fn none() -> Self {
        Self(None)
    }

    pub(crate) fn get(&self) -> Option<&Arc<Plan>> {
        self.0.as_ref()
    }

    pub(crate) fn take(&mut self) -> Option<Arc<Plan>> {
        self.0.take()
    }
}

/// The input is shown by the plan's own `Debug`, one line per node; not here.
impl fmt::Debug for Input {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Input(..)")
    }
}

/// What a node knows about the plan below it, derived once when the node is built.
/// Shared parts sit behind `Arc`, so a node that does not change them clones pointers.
#[derive(Debug, Clone)]
pub(crate) struct NodeState {
    pub(crate) id: NodeId,
    pub(crate) schema: Arc<PlanSchema>,
    /// The store location, as given to the scan.
    pub(crate) source: Arc<str>,
    /// The store's gene IDs, as read by the scan.
    pub(crate) gene_ids: Arc<GeneIds>,
    /// The default layer set nearest to this node.
    pub(crate) default_layer: Option<Arc<str>>,
    /// The genes that may remain, in order, when a selection set them; `None` means
    /// every gene of `gene_ids` (in table order).
    pub(crate) kept_genes: Option<Arc<[String]>>,
    pub(crate) removed_genes: Arc<RemovedGenes>,
    /// The first node from the scan that changed the gene axis.
    pub(crate) gene_axis_changed_by: Option<&'static str>,
    /// The first node from the scan that changed the cell axis.
    pub(crate) cell_axis_changed_by: Option<&'static str>,
    /// Each layer that existed and was written again, with the first node that did.
    pub(crate) rewritten_layers: Arc<BTreeMap<String, String>>,
}

impl NodeState {
    /// The state of a scan of `source`.
    pub(crate) fn root(source: Arc<str>, schema: PlanSchema, gene_ids: Arc<GeneIds>) -> Self {
        Self {
            id: NodeId(0),
            schema: Arc::new(schema),
            source,
            gene_ids,
            default_layer: None,
            kept_genes: None,
            removed_genes: Arc::new(RemovedGenes::new()),
            gene_axis_changed_by: None,
            cell_axis_changed_by: None,
            rewritten_layers: Arc::new(BTreeMap::new()),
        }
    }

    /// The state of `node` built on a node whose state is `input`.
    pub(crate) fn child(input: &Self, node: &impl Transform) -> Self {
        let mut state = Self {
            id: NodeId(input.id.0 + 1),
            schema: Arc::new(node.output_schema(&input.schema)),
            ..input.clone()
        };
        let method = node.method();
        match node.changes_axis() {
            Some(Axis::Cells) => {
                state.cell_axis_changed_by.get_or_insert(method);
            }
            Some(Axis::Genes) => {
                state.gene_axis_changed_by.get_or_insert(method);
            }
            None => {}
        }
        if let Some(layer) = node.writes_layer()
            && input.schema.layer(layer).is_some()
            && !state.rewritten_layers.contains_key(layer)
        {
            Arc::make_mut(&mut state.rewritten_layers).insert(layer.to_owned(), method.to_owned());
        }
        if let Some(layer) = node.sets_default_layer() {
            state.default_layer = Some(Arc::from(layer));
        }
        if let Some(selected) = node.selects_genes() {
            let keep: HashSet<&str> = selected.iter().map(String::as_str).collect();
            let removed = input
                .genes()
                .iter()
                .filter(|gene_id| !keep.contains(gene_id.as_str()))
                .cloned();
            Arc::make_mut(&mut state.removed_genes).insert(removed, method);
            state.kept_genes = Some(Arc::clone(selected));
        }
        state
    }

    /// The genes that may remain, in order (exactly these when the gene count is
    /// known).
    pub(crate) fn genes(&self) -> &[String] {
        match &self.kept_genes {
            Some(kept) => kept,
            None => self.gene_ids.ordered(),
        }
    }
}
