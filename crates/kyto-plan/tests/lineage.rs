//! Task 3.8 Done when: `n_genes` read after `filter_genes` raises `KY-E301`.
//!
//! A stored derived cell column (its `schema.json` entry records `lineage`) was computed
//! over every gene; after a node that changes the gene axis it is stale. The check runs
//! when the plan is built (R6), whichever genes the filter keeps; `.recompute()` lets the
//! read through, and needs the lineage (`KY-E302`).
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use kyto_core::{BinaryOp, CellExpr, ExprDtype, GeneExpr, GeneIds, Literal, Source};
use kyto_plan::Plan;

fn plan() -> Arc<Plan> {
    let schema = serde_json::from_str(
        r#"{
  "n_cells": 2,
  "n_genes": 3,
  "tiling": {"cell_block_size": 4096, "gene_block_size": 1024},
  "cells": {"columns": [
    {"name": "cell_id", "dtype": "String"},
    {"name": "n_genes", "dtype": "UInt32", "lineage": {"layer": "counts", "node": "n3"}},
    {"name": "louvain", "dtype": "Enum", "n_categories": 2, "ordered": false, "lineage": {"layer": "log1p", "node": "n9"}},
    {"name": "batch", "dtype": "String"}
  ]},
  "genes": {"columns": [
    {"name": "gene_id", "dtype": "String"},
    {"name": "highly_variable", "dtype": "Boolean"},
    {"name": "n_cells", "dtype": "UInt32", "lineage": {"layer": "counts", "node": "n4"}}
  ]},
  "layers": [
    {"name": "counts", "dtype": "uint32", "nnz": 3},
    {"name": "log1p", "dtype": "float32", "nnz": 3}
  ],
  "embeddings": [],
  "graphs": []
}"#,
    )
    .unwrap();
    Plan::scan(
        "pbmc.kyto",
        schema,
        GeneIds::new(["CD3E", "CD4", "MS4A1"].map(str::to_owned)),
    )
}

fn highly_variable() -> GeneExpr {
    GeneExpr::column("highly_variable")
}

fn n_genes() -> CellExpr {
    CellExpr::column("n_genes")
}

#[test]
fn n_genes_read_after_filter_genes_is_ky_e301() {
    let scan = plan();
    // Before the filter the stored column is current.
    assert_eq!(
        scan.resolve(&n_genes()).unwrap().dtype(),
        ExprDtype::Numeric
    );

    let filtered = scan.filter_genes(&highly_variable()).unwrap();
    let error = filtered.resolve(&n_genes()).unwrap_err();
    assert_eq!(error.code(), "KY-E301");
    assert_eq!(
        error.to_string(),
        "KY-E301: column 'n_genes' was computed before 'filter_genes' changed its inputs"
    );
    // Inside a larger expression, and through `feature(...)`, too.
    let larger = CellExpr::binary(n_genes(), BinaryOp::Gt, Literal::Int(200)).unwrap();
    assert_eq!(filtered.resolve(&larger).unwrap_err().code(), "KY-E301");
    for name in ["n_genes", "meta:n_genes"] {
        let feature = CellExpr::feature(name).unwrap();
        assert_eq!(filtered.resolve(&feature).unwrap_err().code(), "KY-E301");
    }
    // Also further downstream, past other nodes.
    let downstream = filtered.with_default_layer("counts").unwrap();
    assert_eq!(
        downstream.resolve(&n_genes()).unwrap_err().code(),
        "KY-E301"
    );
    // The plan below the filter is unchanged (R2).
    assert!(scan.resolve(&n_genes()).is_ok());
}

#[test]
fn recompute_lets_the_read_through() {
    let filtered = plan().filter_genes(&highly_variable()).unwrap();
    let recomputed = n_genes().recompute().unwrap();
    assert_eq!(recomputed.to_string(), "cell_column('n_genes').recompute()");
    let resolved = filtered.resolve(&recomputed).unwrap();
    assert_eq!(resolved.to_string(), "cell_column('n_genes').recompute()");
    assert_eq!(resolved.dtype(), ExprDtype::Numeric);
    assert!(matches!(
        resolved.node().as_ref(),
        kyto_core::ExprNode::Source(Source::Recompute { name }) if name == "n_genes"
    ));
    // It reads the matrix, so it never lowers to a Polars column read of the stale value.
    assert!(!resolved.is_metadata_only());
    assert!(resolved.to_polars().is_none());
    // Resolving it again gives it back; recompute is idempotent.
    assert_eq!(filtered.resolve(&resolved).unwrap(), resolved);
    assert_eq!(recomputed.recompute().unwrap(), recomputed);
    // The stored categorical keeps its rules: an unordered Enum cannot be ordered.
    let louvain = CellExpr::column("louvain").recompute().unwrap();
    let ordered = CellExpr::binary(louvain, BinaryOp::Lt, Literal::String("a".into())).unwrap();
    assert_eq!(filtered.resolve(&ordered).unwrap_err().code(), "KY-E205");
}

#[test]
fn columns_without_lineage_are_not_stale() {
    let filtered = plan().filter_genes(&highly_variable()).unwrap();
    assert!(filtered.resolve(&CellExpr::column("batch")).is_ok());
    assert!(filtered.resolve(&CellExpr::column("cell_id")).is_ok());
    // Gene table columns are per gene: a gene filter does not change a kept gene's value.
    assert!(filtered.resolve(&GeneExpr::column("n_cells")).is_ok());
    // A second gene filter on the first one resolves its predicate normally.
    let twice = filtered.filter_genes(&highly_variable()).unwrap();
    // The first gene filter is the one named.
    assert_eq!(twice.gene_axis_changed_by(), Some("filter_genes"));
    assert_eq!(plan().gene_axis_changed_by(), None);
}

#[test]
fn recompute_needs_lineage_and_a_plain_column() {
    let scan = plan();
    let error = scan
        .resolve(&CellExpr::column("batch").recompute().unwrap())
        .unwrap_err();
    assert_eq!(error.code(), "KY-E302");
    assert_eq!(
        error.to_string(),
        "KY-E302: cell_column('batch').recompute() asks to recompute cell column 'batch', \
         but the store records no lineage for it"
    );
    let missing = scan
        .resolve(&CellExpr::column("n_gene").recompute().unwrap())
        .unwrap_err();
    assert_eq!(missing.code(), "KY-E210");

    let sum = CellExpr::binary(n_genes(), BinaryOp::Add, Literal::Int(1)).unwrap();
    let error = sum.recompute().unwrap_err();
    assert_eq!(error.code(), "KY-E213");
    assert_eq!(
        error.to_string(),
        "KY-E213: '.recompute()' needs a plain column read such as \
         cell_column('n_genes').recompute(), got cell_column('n_genes') + 1"
    );
    let profile = GeneExpr::profile(None, "AAAC-1").recompute().unwrap_err();
    assert_eq!(profile.code(), "KY-E213");
    assert!(
        profile
            .to_string()
            .contains("gene_column('n_cells').recompute()")
    );
    // A gene column with lineage can be recomputed too.
    let n_cells = GeneExpr::column("n_cells").recompute().unwrap();
    assert_eq!(
        scan.resolve(&n_cells).unwrap().to_string(),
        n_cells.to_string()
    );
}

#[test]
fn filter_genes_resolves_its_predicate_and_needs_a_boolean() {
    let scan = plan();
    let filtered = scan.filter_genes(&highly_variable()).unwrap();
    assert_eq!(
        filtered.to_string(),
        "FILTER_GENES gene_column('highly_variable')\nSCAN \"pbmc.kyto\" (2 cells x 3 genes)"
    );
    // Same predicate, same plan; another predicate, another plan.
    assert_eq!(*filtered, *scan.filter_genes(&highly_variable()).unwrap());
    let other = scan
        .filter_genes(
            &GeneExpr::binary(GeneExpr::column("n_cells"), BinaryOp::Gt, Literal::Int(0)).unwrap(),
        )
        .unwrap();
    assert_ne!(*filtered, *other);
    // The schema is shared, not changed; no gene is known to be removed.
    assert!(Arc::ptr_eq(filtered.schema(), scan.schema()));
    assert!(filtered.removed_genes().is_empty());
    assert_eq!(filtered.root().source(), "pbmc.kyto");

    let error = scan.filter_genes(&GeneExpr::column("n_cells")).unwrap_err();
    assert_eq!(error.code(), "KY-E205");
    assert_eq!(
        error.to_string(),
        "KY-E205: 'filter_genes' needs a boolean gene expression, got a numeric expression: \
         gene_column('n_cells')"
    );
    let unknown = scan.filter_genes(&GeneExpr::column("hvg")).unwrap_err();
    assert_eq!(unknown.code(), "KY-E210");
}
