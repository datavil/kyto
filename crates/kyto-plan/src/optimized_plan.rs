//! The optimized plan: what the optimizer (task 4.4) outputs and the physical plan
//! (4.5) reads.
//!
//! Does: hold the optimized nodes in an arena, inputs before the nodes that read them,
//! as a DAG: one node may feed several (common sub-plans are computed once), and each
//! [`OptimizedOutput`] names the node a terminal reads and the members it reads. A node
//! is a scan reading only the members it must ([`Operation::Scan`]), a logical node
//! whose parameters are used as they are ([`Operation::Node`]; its own input is the
//! logical plan's and is not used here), or value-by-value maps run in one pass
//! ([`Operation::Fused`]). Each node keeps the [`NodeId`]s of the logical nodes it
//! came from. Prints one line per node, outputs first, then from the top down.
//! Does not: rewrite anything (the rule files `optimize_*.rs`), serialize (the plan
//! JSON writes the logical plan, 4.3) or choose devices and passes (4.5).

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use kyto_core::Axis;

use crate::map_elementwise::MapElementwise;
use crate::members::{Member, Members, node_reads, node_writes};
use crate::node::{CellScope, LogicalNode, NodeId};
use crate::plan::Plan;

/// What an optimized node does.
#[derive(Debug, Clone)]
pub enum Operation {
    /// The scan of the logical plan `plan`, reading only `reads` from its store.
    Scan {
        /// The logical scan.
        plan: Arc<Plan>,
        /// The members read (the ID columns are always read).
        reads: Members,
    },
    /// The logical node `plan`, with its parameters as they are. Its input here is
    /// [`OptimizedNode::input`], not the logical plan's.
    Node(Arc<Plan>),
    /// Value-by-value maps over layers, run in this order in one pass over each block
    /// (task 4.4's elementwise fusion: `normalize_total` then `log1p`). Only the layers
    /// in `writes` are kept; the others live only inside the pass.
    Fused {
        /// The maps, in order; only the first may be cross-cell (a statistics pass
        /// over its input layer, then the one pass).
        maps: Vec<MapElementwise>,
        /// The layers kept for the nodes above (a subset of the maps' outputs).
        writes: BTreeSet<String>,
    },
}

/// One node of an [`OptimizedPlan`].
#[derive(Debug, Clone)]
pub struct OptimizedNode {
    pub(crate) operation: Operation,
    pub(crate) input: Option<usize>,
    pub(crate) origin: Vec<NodeId>,
}

impl OptimizedNode {
    /// What the node does.
    pub fn operation(&self) -> &Operation {
        &self.operation
    }

    /// The position of its input in [`OptimizedPlan::nodes`] (`None` for a scan).
    pub fn input(&self) -> Option<usize> {
        self.input
    }

    /// The IDs of the logical nodes it came from, in plan order.
    pub fn origin(&self) -> &[NodeId] {
        &self.origin
    }

    /// The name `explain` shows.
    pub fn name(&self) -> &'static str {
        match &self.operation {
            Operation::Scan { plan, .. } | Operation::Node(plan) => plan.name(),
            Operation::Fused { .. } => "FUSED_ELEMENTWISE",
        }
    }

    /// The axis it changes, if any.
    pub fn changes_axis(&self) -> Option<Axis> {
        match &self.operation {
            Operation::Scan { .. } | Operation::Fused { .. } => None,
            Operation::Node(plan) => plan.changes_axis(),
        }
    }

    /// Per-cell or cross-cell. A fused node is cross-cell when its first map is.
    pub fn cell_scope(&self) -> CellScope {
        match &self.operation {
            Operation::Scan { plan, .. } | Operation::Node(plan) => plan.cell_scope(),
            Operation::Fused { maps, .. } => maps
                .first()
                .map_or(CellScope::PerCell, LogicalNode::cell_scope),
        }
    }

    /// The members it reads from its input. A fused node reads each map's input
    /// layer unless an earlier map of the same pass wrote it.
    pub fn reads(&self) -> Members {
        match &self.operation {
            Operation::Scan { .. } => Members::new(),
            Operation::Node(plan) => node_reads(plan),
            Operation::Fused { maps, .. } => {
                let mut written = BTreeSet::new();
                let mut reads = Members::new();
                for map in maps {
                    if !written.contains(map.layer()) {
                        reads.insert(Member::Layer(map.layer().to_owned()));
                    }
                    written.insert(map.output_layer());
                }
                reads
            }
        }
    }

    /// The members it adds or replaces (a fused node: every map's output, kept or
    /// not, since each one hides the input's layer of that name).
    pub fn writes(&self) -> Members {
        match &self.operation {
            Operation::Scan { reads, .. } => reads.clone(),
            Operation::Node(plan) => node_writes(plan),
            Operation::Fused { maps, .. } => maps
                .iter()
                .map(|map| Member::Layer(map.output_layer().to_owned()))
                .collect(),
        }
    }
}

/// What a terminal reads: the node and its members.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizedOutput {
    pub(crate) node: usize,
    pub(crate) reads: Members,
}

impl OptimizedOutput {
    /// The position of the node in [`OptimizedPlan::nodes`].
    pub fn node(&self) -> usize {
        self.node
    }

    /// The members the terminal reads (the ID columns always come along).
    pub fn reads(&self) -> &Members {
        &self.reads
    }
}

/// An optimized plan: nodes in an arena (inputs first) and the outputs read from it.
#[derive(Debug, Clone)]
pub struct OptimizedPlan {
    pub(crate) nodes: Vec<OptimizedNode>,
    pub(crate) outputs: Vec<OptimizedOutput>,
}

impl OptimizedPlan {
    /// The nodes; each one's input comes before it.
    pub fn nodes(&self) -> &[OptimizedNode] {
        &self.nodes
    }

    /// The outputs, in the order they were asked for.
    pub fn outputs(&self) -> &[OptimizedOutput] {
        &self.outputs
    }

    /// How many nodes and outputs read each node.
    pub(crate) fn consumer_counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.nodes.len()];
        for node in &self.nodes {
            if let Some(input) = node.input {
                counts[input] += 1;
            }
        }
        for output in &self.outputs {
            counts[output.node] += 1;
        }
        counts
    }

    /// The same plan keeping only the nodes the outputs reach, renumbered so each
    /// input comes before its readers (rules leave unreachable nodes and any order).
    pub(crate) fn compact(self) -> Self {
        let mut position: Vec<Option<usize>> = vec![None; self.nodes.len()];
        let mut walked = vec![false; self.nodes.len()];
        let mut order = Vec::with_capacity(self.nodes.len());
        for output in &self.outputs {
            // Walk down to the first node already placed, then place the walk's nodes
            // from the bottom up (a loop: chains are long).
            let mut walk = Vec::new();
            let mut next = Some(output.node);
            while let Some(index) = next {
                if position[index].is_some() || walked[index] {
                    break;
                }
                walked[index] = true;
                walk.push(index);
                next = self.nodes[index].input;
            }
            for &index in walk.iter().rev() {
                position[index] = Some(order.len());
                order.push(index);
            }
        }
        let renumber = |index: usize| position[index].unwrap_or(index);
        let mut slots: Vec<Option<OptimizedNode>> = self.nodes.into_iter().map(Some).collect();
        let nodes = order
            .iter()
            .filter_map(|&index| slots[index].take())
            .map(|mut node| {
                node.input = node.input.map(renumber);
                node
            })
            .collect();
        let outputs = self
            .outputs
            .into_iter()
            .map(|output| OptimizedOutput {
                node: renumber(output.node),
                reads: output.reads,
            })
            .collect();
        Self { nodes, outputs }
    }
}

/// A node's line: its name, parameters, input and origin.
struct NodeLine<'a>(usize, &'a OptimizedNode);

impl fmt::Display for NodeLine<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let NodeLine(index, node) = *self;
        write!(formatter, "#{index} {}", node.name())?;
        match &node.operation {
            Operation::Scan { plan, reads } => {
                formatter.write_str(" ")?;
                plan.node().describe(formatter)?;
                write!(formatter, " reads {}", MemberList(reads))?;
            }
            Operation::Node(plan) => {
                formatter.write_str(" ")?;
                plan.node().describe(formatter)?;
            }
            Operation::Fused { maps, writes } => {
                formatter.write_str(" [")?;
                for (position, map) in maps.iter().enumerate() {
                    if position > 0 {
                        formatter.write_str("; ")?;
                    }
                    write!(formatter, "{} ", map.name())?;
                    map.describe(formatter)?;
                }
                let kept: Vec<String> = writes.iter().map(|layer| format!("{layer:?}")).collect();
                write!(formatter, "] keeps [{}]", kept.join(", "))?;
            }
        }
        if let Some(input) = node.input {
            write!(formatter, " <- #{input}")?;
        }
        let origin: Vec<String> = node.origin.iter().map(ToString::to_string).collect();
        write!(formatter, " (node {})", origin.join(", "))
    }
}

/// `[layer:counts, cells:batch]`.
struct MemberList<'a>(&'a Members);

impl fmt::Display for MemberList<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("[")?;
        for (position, member) in self.0.iter().enumerate() {
            if position > 0 {
                formatter.write_str(", ")?;
            }
            write!(formatter, "{member}")?;
        }
        formatter.write_str("]")
    }
}

/// One line per output (`output #3 reads [layer:log1p]`), then one per node from the
/// last (top) to the first (a scan).
impl fmt::Display for OptimizedPlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for output in &self.outputs {
            writeln!(
                formatter,
                "output #{} reads {}",
                output.node,
                MemberList(&output.reads)
            )?;
        }
        for (index, node) in self.nodes.iter().enumerate().rev() {
            write!(formatter, "{}", NodeLine(index, node))?;
            if index > 0 {
                formatter.write_str("\n")?;
            }
        }
        Ok(())
    }
}
