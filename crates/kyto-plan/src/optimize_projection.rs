//! The projection pushdown rule (task 4.4): compute and read only what is needed.
//!
//! Does: walk the optimized arena from the top down, carrying the members needed
//! above each node (what the outputs read, plus what each kept node reads, less what
//! it writes). A node that writes members, changes no axis, and writes none that is
//! needed is left out (its readers read its input). A fused pass keeps only the layers
//! needed above. The scan reads only the members needed (the ID columns always).
//! Does not: split a node (a `with_cells` with one needed column of three computes
//! all three), or leave out a filter or a selection (they read no member, but change
//! every member's rows), a sink or a `with_default_layer`.

use crate::members::{Member, Members};
use crate::optimized_plan::{Operation, OptimizedNode, OptimizedPlan};
use crate::plan::Plan;

/// `plan` without the nodes nothing needs, each scan and fused pass keeping only what
/// is needed.
pub(crate) fn push_down(mut plan: OptimizedPlan) -> OptimizedPlan {
    let count = plan.nodes.len();
    let mut needed: Vec<Members> = vec![Members::new(); count];
    for output in &plan.outputs {
        needed[output.node].extend(output.reads.iter().cloned());
    }
    let mut dead = vec![false; count];
    // Inputs come first, so every reader of a node is visited before the node.
    for index in (0..count).rev() {
        let here = std::mem::take(&mut needed[index]);
        let node = &mut plan.nodes[index];
        let writes = node.writes();
        let below = if unused(node, &writes, &here) {
            dead[index] = true;
            here
        } else {
            match &mut node.operation {
                Operation::Scan { reads, .. } => {
                    *reads = reads.intersection(&here).cloned().collect();
                }
                Operation::Fused { maps, writes: kept } => {
                    *kept = maps
                        .iter()
                        .map(|map| map.output_layer().to_owned())
                        .filter(|layer| here.contains(&Member::Layer(layer.clone())))
                        .collect();
                }
                Operation::Node(_) => {}
            }
            // Updated in place, not rebuilt: a long chain carrying many members stays
            // linear (bug-hunter: rebuilding the set at each node was quadratic).
            let mut below = here;
            for member in &writes {
                below.remove(member);
            }
            below.extend(node.reads());
            below
        };
        if let Some(input) = plan.nodes[index].input {
            if needed[input].is_empty() {
                needed[input] = below;
            } else {
                needed[input].extend(below);
            }
        }
    }
    // A dead node is never a scan, so it has an input to read instead.
    let live = |mut index: usize, nodes: &[OptimizedNode]| {
        while dead[index] {
            match nodes[index].input {
                Some(input) => index = input,
                None => break,
            }
        }
        index
    };
    let inputs: Vec<Option<usize>> = plan
        .nodes
        .iter()
        .map(|node| node.input.map(|input| live(input, &plan.nodes)))
        .collect();
    for (node, input) in plan.nodes.iter_mut().zip(inputs) {
        node.input = input;
    }
    let outputs: Vec<usize> = plan
        .outputs
        .iter()
        .map(|output| live(output.node, &plan.nodes))
        .collect();
    for (output, node) in plan.outputs.iter_mut().zip(outputs) {
        output.node = node;
    }
    plan.compact()
}

/// Whether `node`, writing `writes`, can be left out when `needed` is needed above.
fn unused(node: &OptimizedNode, writes: &Members, needed: &Members) -> bool {
    let removable = match &node.operation {
        Operation::Scan { .. } => false,
        Operation::Fused { .. } => true,
        Operation::Node(plan) => {
            plan.changes_axis().is_none()
                && !matches!(**plan, Plan::Sink(_) | Plan::WithDefaultLayer(_))
        }
    };
    removable && !writes.is_empty() && writes.is_disjoint(needed)
}
