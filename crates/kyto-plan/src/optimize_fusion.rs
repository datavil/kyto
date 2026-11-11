//! The elementwise fusion rule (task 4.4): consecutive value-by-value maps run in one
//! pass (`normalize_total` then `log1p`, task 6.2).
//!
//! Does: merge a per-cell `normalize_total` or `log1p` into the map or fused pass it
//! reads when nothing else reads that one, so each block is read once and the maps run
//! one after another on it; the intermediate layers stay inside the pass unless
//! something above needs them (projection pushdown decides). A cross-cell map
//! (`normalize_total` to the median total) can only start a pass: its statistics come
//! from a pass over its input first, which a map before it in the same pass would have
//! to compute too.
//! Does not: fuse across any other node (a filter between two maps keeps them apart,
//! unless predicate pushdown later moves it), or reorder maps.

use std::collections::BTreeSet;

use crate::node::{CellScope, LogicalNode};
use crate::optimized_plan::{Operation, OptimizedNode, OptimizedPlan};
use crate::plan::Plan;

/// `plan` with each run of value-by-value maps merged into one fused pass.
pub(crate) fn fuse(mut plan: OptimizedPlan) -> OptimizedPlan {
    let readers = plan.consumer_counts();
    for index in 0..plan.nodes.len() {
        let map = match &plan.nodes[index].operation {
            Operation::Node(node) => match &**node {
                Plan::MapElementwise(map) if map.params().cell_scope() == CellScope::PerCell => {
                    map.params().clone()
                }
                _ => continue,
            },
            _ => continue,
        };
        let Some(input) = plan.nodes[index].input else {
            continue;
        };
        if readers[input] != 1 {
            continue;
        }
        let below = &mut plan.nodes[input];
        let mut maps = match &mut below.operation {
            Operation::Fused { maps, .. } => std::mem::take(maps),
            Operation::Node(node) => match &**node {
                Plan::MapElementwise(first) => vec![first.params().clone()],
                _ => continue,
            },
            Operation::Scan { .. } => continue,
        };
        maps.push(map);
        // Every output kept until projection pushdown says otherwise.
        let writes: BTreeSet<String> = maps
            .iter()
            .map(|map| map.output_layer().to_owned())
            .collect();
        let mut origin = std::mem::take(&mut below.origin);
        let bottom = below.input;
        origin.extend_from_slice(&plan.nodes[index].origin);
        // The input node is now unreachable; `compact` drops it.
        plan.nodes[index] = OptimizedNode {
            operation: Operation::Fused { maps, writes },
            input: bottom,
            origin,
        };
    }
    plan.compact()
}
