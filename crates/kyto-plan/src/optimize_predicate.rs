//! The predicate pushdown rule (task 4.4): cell filters move down, towards the scan,
//! so fewer cells flow through the nodes below them.
//!
//! Does: move each `filter_cells` below the node it reads while that node is per-cell
//! (each cell's output depends on that cell alone, so filtering before or after gives
//! the same rows and values), changes no cells (a `take_cells` or another cell filter:
//! keeps their order and their errors), writes nothing the predicate reads, and is
//! read by the filter alone (a shared sub-plan stays whole for its other readers). A
//! predicate that reads one gene's values or recomputes a derived column
//! (`.recompute()`) also stops at a node that changes the genes: a filter on data may
//! remove that gene (`KY-E203` when run), and a recomputed column sums over the genes
//! present (bug-hunter).
//! Does not: stop below a cross-cell node (`scale`, `pca`, `normalize_total` to the
//! median, ...: their statistics would change), move gene filters (their pushdown needs
//! per-gene scopes: `normalize_total` sums over genes), or merge filters.

use crate::members::{Members, reads_genes};
use crate::node::CellScope;
use crate::optimized_plan::{Operation, OptimizedNode, OptimizedPlan};
use crate::plan::Plan;
use kyto_core::Axis;

/// `plan` with each cell filter moved as far down as it can go.
pub(crate) fn push_down(mut plan: OptimizedPlan) -> OptimizedPlan {
    let count = plan.nodes.len();
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (index, node) in plan.nodes.iter().enumerate() {
        if let Some(input) = node.input {
            readers[input].push(index);
        }
    }
    let mut outputs_reading = vec![0_usize; count];
    for output in &plan.outputs {
        outputs_reading[output.node] += 1;
    }
    // Lower filters first, so a higher one can then move down to it.
    for filter in 0..count {
        let Some(reads_genes) = cell_filter(&plan.nodes[filter]) else {
            continue;
        };
        let reads = plan.nodes[filter].reads();
        while let Some(below) = plan.nodes[filter].input {
            let only_reader = readers[below].len() == 1 && outputs_reading[below] == 0;
            if !only_reader || !passes(&plan.nodes[below], &reads, reads_genes) {
                break;
            }
            // Swap: the filter reads what `below` read, `below` reads the filter, and
            // the filter's readers read `below`.
            let bottom = plan.nodes[below].input;
            plan.nodes[filter].input = bottom;
            plan.nodes[below].input = Some(filter);
            let above = std::mem::replace(&mut readers[filter], vec![below]);
            for &reader in &above {
                plan.nodes[reader].input = Some(below);
            }
            readers[below] = above;
            if let Some(bottom) = bottom {
                for reader in &mut readers[bottom] {
                    if *reader == below {
                        *reader = filter;
                    }
                }
            }
            if outputs_reading[filter] > 0 {
                for output in &mut plan.outputs {
                    if output.node == filter {
                        output.node = below;
                    }
                }
                outputs_reading[below] = std::mem::take(&mut outputs_reading[filter]);
            }
        }
    }
    plan.compact()
}

/// `Some(whether it depends on the genes present)` for a cell filter, `None` otherwise.
fn cell_filter(node: &OptimizedNode) -> Option<bool> {
    match &node.operation {
        Operation::Node(plan) => match &**plan {
            Plan::Filter(filter) if filter.params().axis() == Axis::Cells => {
                Some(reads_genes(plan))
            }
            _ => None,
        },
        _ => None,
    }
}

/// Whether a cell filter reading `reads` can move below `node`.
fn passes(node: &OptimizedNode, reads: &Members, reads_genes: bool) -> bool {
    if matches!(node.operation, Operation::Scan { .. }) || node.cell_scope() != CellScope::PerCell {
        return false;
    }
    match node.changes_axis() {
        Some(Axis::Cells) => return false,
        Some(Axis::Genes) if reads_genes => return false,
        _ => {}
    }
    node.writes().is_disjoint(reads)
}
