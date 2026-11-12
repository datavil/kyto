//! `explain` (task 4.7): a plan's logical and optimized trees, as text or JSON.
//!
//! Does: gather, from a logical plan and its [`PhysicalPlan`], one entry per logical
//! node (from the last down to the scan: its ID, name, parameters with every name
//! resolved in full (R6), input, and a note in brackets when the step has a cheaper
//! alternative: a centred `scale` densifies, task 6.4) and the optimized plan's outputs and nodes (from
//! the top down: arena position, name, parameters, what a scan reads, input, the
//! logical nodes it came from, and how it runs: device, passes, blocks). [`Explain`]
//! prints the text form and writes the JSON form from the same entries, so the two
//! always say the same. Text:
//!
//! ```text
//! logical plan
//!   node 1 FILTER_GENES gene_column('feature_type') == 'Gene Expression' <- node 0
//!   node 0 SCAN "pbmc.kyto" (2 cells x 3 genes)
//! optimized plan
//!   output #1 reads [layer:counts, genes:feature_type, genes:symbol]
//!   #1 FILTER_GENES gene_column('feature_type') == 'Gene Expression' <- #0 (node 1) [device=cpu passes=1 blocks=4096 cells]
//!   #0 SCAN "pbmc.kyto" (2 cells x 3 genes) reads [layer:counts, genes:feature_type, genes:symbol] (node 0) [device=cpu passes=1 blocks=4096 cells]
//! ```
//!
//! JSON: `{"logical": [...], "optimized": {"outputs": [...], "nodes": [...]}}`, the
//! entries in the text's order, with the keys of [`LogicalEntry`], [`OutputEntry`] and
//! [`OptimizedEntry`].
//! Does not: optimize or choose devices (4.4, 4.5), estimate costs (4.6: `estimate()`
//! is its own call), or serialize a plan to read back (the plan JSON of 4.3; this JSON
//! is for reading only and has no version).

use std::fmt;
use std::sync::Arc;

use serde::Serialize;

use crate::cross_cell::CrossCellOp;
use crate::optimized_plan::{NodeParameters, Operation};
use crate::physical_plan::{Blocks, PhysicalOptions, PhysicalPlan};
use crate::plan::Plan;

/// A plan's logical and optimized trees, for `explain`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Explain {
    logical: Vec<LogicalEntry>,
    optimized: OptimizedTree,
}

/// One logical node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LogicalEntry {
    /// Its stable node ID (position from the scan).
    pub id: u64,
    /// Its name (`FILTER_GENES`).
    pub name: &'static str,
    /// Its parameters, names resolved in full.
    pub parameters: String,
    /// Its input's ID (`None` for the scan).
    pub input: Option<u64>,
    /// A warning about the step and what to do instead (`None` for most steps): a
    /// centred `scale` densifies its layer (task 6.4).
    pub note: Option<String>,
}

/// One output of the optimized plan: the node a terminal reads and what it reads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OutputEntry {
    /// The node's position in the optimized plan.
    pub node: usize,
    /// The members read (`layer:counts`, `cells:batch`).
    pub reads: Vec<String>,
}

/// The optimized plan's outputs and nodes (from the top down).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct OptimizedTree {
    outputs: Vec<OutputEntry>,
    nodes: Vec<OptimizedEntry>,
}

/// One optimized node and how it runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OptimizedEntry {
    /// Its position in the optimized plan (inputs come first).
    pub index: usize,
    /// Its name (`FUSED_ELEMENTWISE` for a fused pass).
    pub name: &'static str,
    /// Its parameters, names resolved in full (a fused pass: its maps and kept layers).
    pub parameters: String,
    /// The members a scan reads from its store (`None` for every other node).
    pub reads: Option<Vec<String>>,
    /// Its input's position (`None` for a scan).
    pub input: Option<usize>,
    /// The IDs of the logical nodes it came from.
    pub origin: Vec<u64>,
    /// The device it runs on (`cpu`).
    pub device: &'static str,
    /// Its passes over its input's blocks (statistics passes plus the apply pass).
    pub passes: u32,
    /// Passes over all of its input's blocks before it outputs anything.
    pub statistics_passes: u32,
    /// Whether it maps the stream of blocks.
    pub apply_pass: bool,
    /// How it meets the data: `cells` (streamed), `in_memory` or `none`.
    pub blocks: &'static str,
    /// The cells in each block when streamed (`None` otherwise).
    pub block_cells: Option<u64>,
}

impl Explain {
    /// The explanation of `plan`, whose physical plan is `physical`.
    pub fn new(plan: &Plan, physical: &PhysicalPlan) -> Self {
        let logical = plan
            .nodes()
            .map(|node| LogicalEntry {
                id: node.id().0,
                name: node.name(),
                parameters: Described(node).to_string(),
                input: node.input().map(|input| input.id().0),
                note: note(node),
            })
            .collect();
        let optimized = physical.optimized();
        let outputs = optimized
            .outputs()
            .iter()
            .map(|output| OutputEntry {
                node: output.node(),
                reads: output.reads().iter().map(ToString::to_string).collect(),
            })
            .collect();
        let nodes = optimized
            .nodes()
            .iter()
            .zip(physical.executions())
            .enumerate()
            .rev()
            .map(|(index, (node, execution))| {
                let (blocks, block_cells) = match execution.blocks() {
                    Blocks::Cells(cells) => ("cells", Some(cells)),
                    Blocks::InMemory => ("in_memory", None),
                    Blocks::None => ("none", None),
                };
                OptimizedEntry {
                    index,
                    name: node.name(),
                    parameters: NodeParameters(node).to_string(),
                    reads: match node.operation() {
                        Operation::Scan { reads, .. } => {
                            Some(reads.iter().map(ToString::to_string).collect())
                        }
                        Operation::Node(_) | Operation::Fused { .. } => None,
                    },
                    input: node.input(),
                    origin: node.origin().iter().map(|id| id.0).collect(),
                    device: execution.device().name(),
                    passes: execution.passes(),
                    statistics_passes: execution.statistics_passes(),
                    apply_pass: execution.apply_pass(),
                    blocks,
                    block_cells,
                }
            })
            .collect();
        Self {
            logical,
            optimized: OptimizedTree { outputs, nodes },
        }
    }

    /// The logical nodes, from the last down to the scan.
    pub fn logical(&self) -> &[LogicalEntry] {
        &self.logical
    }

    /// The optimized plan's outputs, in the order they were asked for.
    pub fn outputs(&self) -> &[OutputEntry] {
        &self.optimized.outputs
    }

    /// The optimized nodes, from the top (last) down to the first.
    pub fn optimized(&self) -> &[OptimizedEntry] {
        &self.optimized.nodes
    }

    /// The JSON form: the same entries as the text, indented.
    pub fn to_json(&self) -> String {
        // Every key is a string and there are no floats, so this cannot fail.
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|error| unreachable!("an explanation always serializes: {error}"))
    }
}

impl Plan {
    /// This plan's explanation: its logical tree and its physical plan
    /// ([`Plan::physical`]: every optimizer rule, for a terminal reading all of its
    /// output, on the CPU). Reads nothing; the plan is unchanged.
    pub fn explanation(self: &Arc<Self>) -> Explain {
        Explain::new(self, &self.physical(PhysicalOptions::default()))
    }

    /// How this plan would run, as text: [`Plan::explanation`]'s logical tree and
    /// optimized plan (see the `explain` module for the form).
    pub fn explain(self: &Arc<Self>) -> String {
        self.explanation().to_string()
    }

    /// The same as [`Plan::explain`], as indented JSON text.
    pub fn explain_json(self: &Arc<Self>) -> String {
        self.explanation().to_json()
    }
}

/// A logical node's parameters (its `describe`).
struct Described<'a>(&'a Plan);

impl fmt::Display for Described<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.node().describe(formatter)
    }
}

/// What `explain` warns about `node`, if anything: a centred `scale` stores every gene
/// of every cell, where PCA would centre the sparse layer implicitly (task 6.4).
fn note(node: &Plan) -> Option<String> {
    match node {
        Plan::CrossCell(cross_cell) => match cross_cell.params().op() {
            CrossCellOp::Scale {
                zero_center: true,
                into,
                ..
            } => Some(format!(
                "densifies: zero_center=true stores every gene of every cell in {into:?} \
                 (see estimate()); for PCA, give pca the unscaled layer, which it centres \
                 implicitly"
            )),
            _ => None,
        },
        _ => None,
    }
}

/// `[layer:counts, cells:batch]`.
fn member_list(members: &[String]) -> String {
    format!("[{}]", members.join(", "))
}

/// The text form (see the module documentation).
impl fmt::Display for Explain {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("logical plan")?;
        for entry in &self.logical {
            write!(
                formatter,
                "\n  node {} {} {}",
                entry.id, entry.name, entry.parameters
            )?;
            if let Some(input) = entry.input {
                write!(formatter, " <- node {input}")?;
            }
            if let Some(note) = &entry.note {
                write!(formatter, " [{note}]")?;
            }
        }
        formatter.write_str("\noptimized plan")?;
        for output in &self.optimized.outputs {
            let reads = member_list(&output.reads);
            write!(formatter, "\n  output #{} reads {reads}", output.node)?;
        }
        for entry in &self.optimized.nodes {
            write!(
                formatter,
                "\n  #{} {} {}",
                entry.index, entry.name, entry.parameters
            )?;
            if let Some(reads) = &entry.reads {
                write!(formatter, " reads {}", member_list(reads))?;
            }
            if let Some(input) = entry.input {
                write!(formatter, " <- #{input}")?;
            }
            let origin: Vec<String> = entry.origin.iter().map(ToString::to_string).collect();
            write!(
                formatter,
                " (node {}) [device={} passes={}",
                origin.join(", "),
                entry.device,
                entry.passes
            )?;
            if entry.statistics_passes > 0 {
                write!(
                    formatter,
                    " (statistics={} apply={})",
                    entry.statistics_passes,
                    u32::from(entry.apply_pass)
                )?;
            }
            match entry.block_cells {
                Some(cells) => write!(formatter, " blocks={cells} cells]")?,
                None if entry.blocks == "in_memory" => formatter.write_str(" in memory]")?,
                None => formatter.write_str(" no data]")?,
            }
        }
        Ok(())
    }
}
