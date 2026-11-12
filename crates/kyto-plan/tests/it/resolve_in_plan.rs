//! Task 3.5 Done when: adding a column named `CD3E` after a plan is built does not
//! change that plan's output. A plan resolves names against the schema and gene IDs its
//! scan read, and the default layer of its own chain (R6), so neither a changed store
//! nor a later `with_default_layer` changes what it resolved.
#![allow(clippy::unwrap_used)]

use kyto_core::{AtlasSchema, BinaryOp, CellExpr, GeneIds, Literal};
use kyto_plan::Plan;

fn schema(extra_cell_column: &str) -> AtlasSchema {
    let text = format!(
        r#"{{
  "n_cells": 2,
  "n_genes": 3,
  "tiling": {{"cell_block_size": 4096, "gene_block_size": 1024}},
  "cells": {{"columns": [{{"name": "cell_id", "dtype": "String"}}{extra_cell_column}]}},
  "genes": {{"columns": [{{"name": "gene_id", "dtype": "String"}}]}},
  "layers": [
    {{"name": "counts", "dtype": "uint32", "nnz": 3}},
    {{"name": "log1p", "dtype": "float32", "nnz": 3}}
  ],
  "embeddings": [],
  "graphs": []
}}"#
    );
    serde_json::from_str(&text).unwrap()
}

fn genes() -> GeneIds {
    GeneIds::new(["CD3E", "CD4", "MS4A1"].map(str::to_owned))
}

fn high_cd3e() -> CellExpr {
    CellExpr::binary(
        CellExpr::feature("CD3E").unwrap(),
        BinaryOp::Gt,
        Literal::Int(1),
    )
    .unwrap()
}

#[test]
fn adding_a_column_named_cd3e_does_not_change_a_built_plan() {
    let plan = Plan::scan("pbmc.kyto", schema(""), genes())
        .with_default_layer("log1p")
        .unwrap();
    let resolved = plan.resolve(&high_cd3e()).unwrap();
    assert_eq!(
        resolved.to_string(),
        "gene_expression('CD3E', layer='log1p') > 1"
    );
    let plan_text = plan.to_string();

    // The store gains a column `CD3E`: a plan built now sees two features.
    let rebuilt = Plan::scan(
        "pbmc.kyto",
        schema(r#", {"name": "CD3E", "dtype": "Float32"}"#),
        genes(),
    )
    .with_default_layer("log1p")
    .unwrap();
    let error = rebuilt.resolve(&high_cd3e()).unwrap_err();
    assert_eq!(error.code(), "KY-E201");
    assert_eq!(
        error.to_string(),
        "KY-E201: ambiguous name 'CD3E': it matches 'meta:CD3E', 'log1p:CD3E'"
    );

    // The plan built before is unchanged and still resolves the same way.
    assert_eq!(plan.to_string(), plan_text);
    assert_eq!(plan.resolve(&high_cd3e()).unwrap(), resolved);
    // The resolved expression names its layer, so resolving it on the new plan, or under
    // another default layer, keeps its meaning.
    assert_eq!(rebuilt.resolve(&resolved).unwrap(), resolved);
    let counts = plan.with_default_layer("counts").unwrap();
    assert_eq!(counts.resolve(&resolved).unwrap(), resolved);
    assert_eq!(
        counts.resolve(&high_cd3e()).unwrap().to_string(),
        "gene_expression('CD3E', layer='counts') > 1"
    );
}

#[test]
fn the_nearest_default_layer_is_used_and_a_scan_has_none() {
    let scan = Plan::scan("pbmc.kyto", schema(""), genes());
    assert_eq!(scan.resolve(&high_cd3e()).unwrap_err().code(), "KY-E211");
    let plan = scan
        .with_default_layer("counts")
        .unwrap()
        .with_default_layer("log1p")
        .unwrap();
    assert_eq!(
        plan.resolve(&high_cd3e()).unwrap().to_string(),
        "gene_expression('CD3E', layer='log1p') > 1"
    );
    // Unknown layers name the plan's store.
    let error = scan
        .resolve(&CellExpr::expression(Some("raw".to_owned()), "CD3E"))
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "KY-E109: no layer named 'raw' in the store at 'pbmc.kyto'"
    );
}
