//! Task 8.3: `Plan::peek_output()` is the plan's cell table as an output that reads
//! the store's tables only, and `KY-E219` when a step it depends on reads data.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use crate::support;

use kyto_core::{BinaryOp, CellExpr, Expr, GeneExpr, Literal, QcMetric};
use kyto_error::KytoError;
use kyto_plan::{Member, Operation, OptimizeOptions, Output, Plan, optimize};
use support::scan;

/// Every member the optimized `output` reads from the store.
fn scan_reads(output: &Output) -> Vec<String> {
    let optimized = optimize(std::slice::from_ref(output), OptimizeOptions::ALL);
    optimized
        .nodes()
        .iter()
        .filter_map(|node| match node.operation() {
            Operation::Scan { reads, .. } => Some(reads.iter().map(Member::to_string)),
            _ => None,
        })
        .flatten()
        .collect()
}

fn refused(plan: &Arc<Plan>) -> (String, String) {
    match plan.peek_output().unwrap_err() {
        KytoError::PeekNeedsData { step, member } => (step, member),
        other => panic!("expected KY-E219, got {other}"),
    }
}

fn high_score() -> CellExpr {
    Expr::binary(CellExpr::column("score"), BinaryOp::Gt, Literal::Int(1)).unwrap()
}

#[test]
fn a_scan_reads_its_cell_columns_only() {
    let output = scan().peek_output().unwrap();
    assert_eq!(
        scan_reads(&output),
        ["cells:batch", "cells:score", "cells:total_counts"]
    );
}

#[test]
fn metadata_steps_read_tables_only() {
    let doubled = Expr::binary(CellExpr::column("score"), BinaryOp::Mul, Literal::Int(2)).unwrap();
    let variable = GeneExpr::column("highly_variable");
    let plan = scan()
        .filter_cells(&high_score())
        .unwrap()
        .with_cells([("doubled".to_owned(), doubled)])
        .unwrap()
        .filter_genes(&variable)
        .unwrap()
        .take_cells(["c3".to_owned(), "c1".to_owned()])
        .unwrap();
    let output = plan.peek_output().unwrap();
    assert!(
        scan_reads(&output)
            .iter()
            .all(|member| member.starts_with("cells:") || member.starts_with("genes:")),
        "{:?}",
        scan_reads(&output)
    );
}

#[test]
fn steps_the_cell_table_does_not_depend_on_are_left_out() {
    // A layer map, a PCA and a gene filter on data change no cell column.
    let plan = scan()
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .pca(Some("log1p"), "X_new", 2, 0, None)
        .unwrap();
    let output = plan.peek_output().unwrap();
    assert_eq!(
        scan_reads(&output),
        ["cells:batch", "cells:score", "cells:total_counts"]
    );
}

#[test]
fn a_column_computed_from_a_layer_is_refused_naming_the_step() {
    let total = CellExpr::qc(Some("counts".to_owned()), QcMetric::TotalCounts).unwrap();
    let plan = scan().with_cells([("total".to_owned(), total)]).unwrap();
    assert_eq!(
        refused(&plan),
        ("with_cells".to_owned(), "layer:counts".to_owned())
    );
    let error = plan.peek_output().unwrap_err();
    assert_eq!(error.code(), "KY-E219");
    let text = error.to_string();
    assert!(
        text.contains("'layer:counts', which 'with_cells' reads"),
        "{text}"
    );
    // The plan before the step still answers.
    assert!(scan().peek_output().is_ok());
}

#[test]
fn a_filter_on_expression_values_is_refused() {
    let expressed = Expr::binary(
        CellExpr::expression(Some("counts".to_owned()), "G0"),
        BinaryOp::Gt,
        Literal::Int(0),
    )
    .unwrap();
    // Metadata steps after the filter do not hide it.
    let plan = scan()
        .filter_cells(&expressed)
        .unwrap()
        .filter_cells(&high_score())
        .unwrap();
    assert_eq!(
        refused(&plan),
        ("filter_cells".to_owned(), "layer:counts".to_owned())
    );
}

#[test]
fn a_layer_written_by_the_plan_is_refused_at_the_step_that_reads_it() {
    let total = CellExpr::qc(Some("norm".to_owned()), QcMetric::TotalCounts).unwrap();
    let plan = scan()
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .with_cells([("total".to_owned(), total)])
        .unwrap();
    assert_eq!(
        refused(&plan),
        ("with_cells".to_owned(), "layer:norm".to_owned())
    );
}

#[test]
fn a_column_read_from_an_embedding_is_refused() {
    let component = CellExpr::embedding("X_pca", 0);
    let plan = scan().with_cells([("pc1".to_owned(), component)]).unwrap();
    assert_eq!(
        refused(&plan),
        ("with_cells".to_owned(), "embedding:X_pca".to_owned())
    );
}
