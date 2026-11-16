//! Task 7.7: `Plan::summary()` is the plan in one line (node count, sizes, layers),
//! built from what the plan holds. A size only the data can tell is `?`.
#![allow(clippy::unwrap_used)]

use crate::support;

use kyto_core::{BinaryOp, CellExpr, Expr, GeneExpr, GeneIds, Literal};
use kyto_plan::Plan;
use support::scan;

#[test]
fn a_scan_has_one_node_and_exact_sizes() {
    assert_eq!(
        scan().summary(),
        "plan: 1 node · cells: 24 · genes: 6 · layers: counts, log1p"
    );
}

#[test]
fn a_filter_makes_its_axis_unknown_and_a_new_layer_is_listed() {
    let cells = Expr::binary(CellExpr::column("score"), BinaryOp::Gt, Literal::Int(1)).unwrap();
    let genes = Expr::binary(GeneExpr::column("mean"), BinaryOp::Gt, Literal::Float(0.5)).unwrap();
    let filtered = scan().filter_cells(&cells).unwrap();
    assert_eq!(
        filtered.summary(),
        "plan: 2 nodes · cells: ? · genes: 6 · layers: counts, log1p"
    );
    let plan = filtered
        .filter_genes(&genes)
        .unwrap()
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap();
    assert_eq!(
        plan.summary(),
        "plan: 4 nodes · cells: ? · genes: ? · layers: counts, log1p, norm"
    );
    // The earlier plans are unchanged (R2).
    assert_eq!(
        filtered.summary(),
        "plan: 2 nodes · cells: ? · genes: 6 · layers: counts, log1p"
    );
}

#[test]
fn chosen_cells_and_genes_have_exact_sizes() {
    let plan = scan()
        .select_genes(["G1".to_owned(), "G3".to_owned()])
        .unwrap()
        .take_cells(["c0".to_owned(), "c5".to_owned(), "c7".to_owned()])
        .unwrap();
    assert_eq!(
        plan.summary(),
        "plan: 3 nodes · cells: 3 · genes: 2 · layers: counts, log1p"
    );
}

#[test]
fn a_store_without_layers_says_none() {
    let base = scan();
    let mut schema = base.schema().clone();
    schema.layers.clear();
    let plan = Plan::scan("empty.kyto", schema, GeneIds::new(support::gene_id_list()));
    assert_eq!(
        plan.summary(),
        "plan: 1 node · cells: 24 · genes: 6 · layers: none"
    );
}

#[test]
fn a_control_character_in_a_layer_name_is_escaped() {
    // A scan's schema is taken as given (the plan JSON replays it unchecked).
    let mut schema = scan().schema().clone();
    schema.layers[1].name = "log\n<Atlas evil>".to_owned();
    let plan = Plan::scan("odd.kyto", schema, GeneIds::new(support::gene_id_list()));
    assert_eq!(
        plan.summary(),
        "plan: 1 node · cells: 24 · genes: 6 · layers: counts, log\\n<Atlas evil>"
    );
}

#[test]
fn the_summary_is_one_line_at_any_depth() {
    let mut plan = scan();
    for _ in 0..1000 {
        plan = plan.with_default_layer("counts").unwrap();
    }
    assert_eq!(
        plan.summary(),
        "plan: 1001 nodes · cells: 24 · genes: 6 · layers: counts, log1p"
    );
}
