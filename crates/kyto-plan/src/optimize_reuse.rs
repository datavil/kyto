//! Lowering logical plans to the optimized arena, and the common sub-plan reuse rule
//! (task 4.4).
//!
//! Does: walk each output's logical chain from its scan up and add one optimized
//! node per logical node. With reuse on, a node equal to one already added over the
//! same input (the same `Arc`, or equal parameters: two plans built the same way from
//! equal scans) is not added again, so a sub-plan several outputs share is computed
//! once and read by each of them (the arena becomes a DAG). Two equal nodes in one
//! chain are not merged: they have different inputs.
//! Does not: compare nodes whose inputs differ, or merge outputs that only become
//! equal after another rule (the reuse runs once, before the others).

use std::collections::HashMap;
use std::sync::Arc;

use crate::members::all_members;
use crate::optimize::Output;
use crate::optimized_plan::{Operation, OptimizedNode, OptimizedOutput, OptimizedPlan};
use crate::plan::Plan;

/// The optimized arena of `outputs`: one chain per output, or with `reuse`, one node
/// per distinct sub-plan.
pub(crate) fn lower(outputs: &[Output], reuse: bool) -> OptimizedPlan {
    let mut nodes: Vec<OptimizedNode> = Vec::new();
    // With reuse: the nodes added over each input, and each logical node's index.
    let mut over: HashMap<Option<usize>, Vec<usize>> = HashMap::new();
    let mut by_pointer: HashMap<*const Plan, usize> = HashMap::new();
    let mut lowered = Vec::with_capacity(outputs.len());
    for output in outputs {
        let mut below: Option<usize> = None;
        for plan in chain(&output.plan) {
            let pointer = Arc::as_ptr(&plan);
            if reuse {
                let found = by_pointer.get(&pointer).copied().or_else(|| {
                    over.get(&below)?.iter().copied().find(|&index| {
                        logical(&nodes[index]).is_some_and(|node| node.same_node(&plan))
                    })
                });
                if let Some(index) = found {
                    by_pointer.insert(pointer, index);
                    below = Some(index);
                    continue;
                }
            }
            let index = nodes.len();
            let origin = vec![plan.id()];
            let operation = match &*plan {
                Plan::Scan(scan) => Operation::Scan {
                    reads: all_members(scan.params().schema()),
                    plan: Arc::clone(&plan),
                },
                _ => Operation::Node(Arc::clone(&plan)),
            };
            nodes.push(OptimizedNode {
                operation,
                input: below,
                origin,
            });
            if reuse {
                over.entry(below).or_default().push(index);
                by_pointer.insert(pointer, index);
            }
            below = Some(index);
        }
        lowered.push(OptimizedOutput {
            // Every chain starts with a scan, so `below` is set.
            node: below.unwrap_or_default(),
            reads: output.reads.clone(),
        });
    }
    OptimizedPlan {
        nodes,
        outputs: lowered,
    }
}

/// The logical nodes of `plan`, from its scan up.
fn chain(plan: &Arc<Plan>) -> Vec<Arc<Plan>> {
    let mut chain = vec![Arc::clone(plan)];
    let mut next = plan.input();
    while let Some(input) = next {
        chain.push(Arc::clone(input));
        next = input.input();
    }
    chain.reverse();
    chain
}

/// The logical node an optimized node holds (`None` for a fused one, which lowering
/// never adds).
fn logical(node: &OptimizedNode) -> Option<&Plan> {
    match &node.operation {
        Operation::Scan { plan, .. } | Operation::Node(plan) => Some(plan),
        Operation::Fused { .. } => None,
    }
}
