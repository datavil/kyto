//! Task 3.5: the feature resolver. Short and full names resolve by kind against the
//! schema with no priority order (`KY-E201`), every constructor's names are checked
//! (`KY-E210`, `KY-E211`, `KY-E109`), column kinds come from the schema and operations
//! are checked again (`KY-E205`). The grammar is docs/feature_names.md.
#![allow(clippy::unwrap_used)]

use kyto_core::{
    AtlasSchema, BinaryOp, CellExpr, ExprDtype, ExprNode, GeneExpr, GeneIds, Literal, Resolver,
    Source, StrFunction,
};
use kyto_error::KytoError;

/// A store with a layer named `meta`, a layer and an embedding both named `log1p`, a
/// cell column `CD4` that is also a gene ID, and a gene `cluster` that is also a column.
const SCHEMA: &str = r#"{
  "n_cells": 4,
  "n_genes": 6,
  "tiling": {"cell_block_size": 4096, "gene_block_size": 1024},
  "cells": {"columns": [
    {"name": "cell_id", "dtype": "String"},
    {"name": "n_genes", "dtype": "Int32"},
    {"name": "cluster", "dtype": "Enum", "n_categories": 3, "ordered": false},
    {"name": "stage", "dtype": "Enum", "n_categories": 3, "ordered": true},
    {"name": "is_doublet", "dtype": "Boolean"},
    {"name": "CD4", "dtype": "Float32"}
  ]},
  "genes": {"columns": [
    {"name": "gene_id", "dtype": "String"},
    {"name": "highly_variable", "dtype": "Boolean"}
  ]},
  "layers": [
    {"name": "counts", "dtype": "uint32", "nnz": 10},
    {"name": "log1p", "dtype": "float32", "nnz": 10},
    {"name": "meta", "dtype": "float32", "nnz": 10}
  ],
  "embeddings": [
    {"name": "loadings", "axis": "genes", "width": 3, "dtype": "float32"},
    {"name": "log1p", "axis": "cells", "width": 2, "dtype": "float32"},
    {"name": "pca", "axis": "cells", "width": 3, "dtype": "float32"}
  ],
  "graphs": []
}"#;

fn schema() -> AtlasSchema {
    serde_json::from_str(SCHEMA).unwrap()
}

fn genes() -> GeneIds {
    GeneIds::new(
        ["CD3E", "CD4", "MS4A1", "cluster", "0", "a:b"]
            .into_iter()
            .map(str::to_owned),
    )
}

/// Resolve `expr` with default layer `layer`.
fn resolve_cell(expr: &CellExpr, layer: Option<&str>) -> Result<CellExpr, KytoError> {
    let (schema, genes) = (schema(), genes());
    Resolver::new(&schema, &genes, layer, "pbmc.kyto").resolve(expr)
}

fn resolve_gene(expr: &GeneExpr, layer: Option<&str>) -> Result<GeneExpr, KytoError> {
    let (schema, genes) = (schema(), genes());
    Resolver::new(&schema, &genes, layer, "pbmc.kyto").resolve(expr)
}

fn feature(name: &str, layer: Option<&str>) -> Result<CellExpr, KytoError> {
    resolve_cell(&CellExpr::feature(name).unwrap(), layer)
}

fn resolved(name: &str) -> String {
    feature(name, Some("log1p")).unwrap().to_string()
}

#[test]
fn short_names_resolve_to_a_column_or_a_gene_of_the_default_layer() {
    assert_eq!(resolved("CD3E"), "gene_expression('CD3E', layer='log1p')");
    assert_eq!(resolved("n_genes"), "cell_column('n_genes')");
    assert_eq!(
        feature("CD3E", Some("counts")).unwrap().to_string(),
        "gene_expression('CD3E', layer='counts')"
    );
}

#[test]
fn full_names_resolve_by_kind() {
    assert_eq!(resolved("meta:n_genes"), "cell_column('n_genes')");
    assert_eq!(
        resolved("counts:CD3E"),
        "gene_expression('CD3E', layer='counts')"
    );
    assert_eq!(resolved("pca:2"), "cell_embedding('pca', 2)");
    // A layer named `meta` holds gene `CD3E`; there is no column `CD3E`.
    assert_eq!(
        resolved("meta:CD3E"),
        "gene_expression('CD3E', layer='meta')"
    );
    // `log1p` is a layer and an embedding; gene `1` does not exist, component 1 does.
    assert_eq!(resolved("log1p:1"), "cell_embedding('log1p', 1)");
    // The name splits at the first ':'.
    assert_eq!(
        resolved("counts:a:b"),
        "gene_expression('a:b', layer='counts')"
    );
}

#[test]
fn resolved_kinds_come_from_the_schema() {
    let kind = |name: &str| feature(name, Some("log1p")).unwrap().dtype();
    assert_eq!(kind("n_genes"), ExprDtype::Numeric);
    assert_eq!(kind("stage"), ExprDtype::String);
    assert_eq!(kind("meta:cell_id"), ExprDtype::String);
    assert_eq!(kind("is_doublet"), ExprDtype::Boolean);
    assert_eq!(kind("CD3E"), ExprDtype::Numeric);
    let column = resolve_gene(&GeneExpr::column("highly_variable"), None).unwrap();
    assert_eq!(column.dtype(), ExprDtype::Boolean);
}

fn ambiguous(error: KytoError) -> (String, Vec<String>) {
    match error {
        KytoError::AmbiguousName { name, candidates } => (name, candidates),
        other => panic!("expected KY-E201, got {other}"),
    }
}

#[test]
fn two_matches_raise_ky_e201_listing_both() {
    // Column `CD4` and gene `CD4`: no priority order.
    let error = feature("CD4", Some("log1p")).unwrap_err();
    assert_eq!(error.code(), "KY-E201");
    assert!(
        error.to_string().contains("'meta:CD4', 'log1p:CD4'"),
        "{error}"
    );
    // Column `cluster` and gene `cluster` of layer `meta` share the full name.
    let (name, candidates) = ambiguous(feature("meta:cluster", Some("log1p")).unwrap_err());
    assert_eq!(name, "meta:cluster");
    assert_eq!(
        candidates,
        ["meta:cluster (cell column)", "meta:cluster (gene)"]
    );
    // Gene `0` of layer `log1p` and component 0 of embedding `log1p`.
    let (_, candidates) = ambiguous(feature("log1p:0", None).unwrap_err());
    assert_eq!(
        candidates,
        ["log1p:0 (gene)", "log1p:0 (embedding component)"]
    );
    // Short `cluster`: a column and a gene of the default layer.
    let (_, candidates) = ambiguous(feature("cluster", Some("counts")).unwrap_err());
    assert_eq!(candidates, ["meta:cluster", "counts:cluster"]);
}

#[test]
fn short_gene_names_need_a_default_layer() {
    let error = feature("CD3E", None).unwrap_err();
    assert_eq!(error.code(), "KY-E211");
    assert_eq!(
        error.to_string(),
        "KY-E211: feature('CD3E') reads the default layer, but the plan has none"
    );
    assert!(
        error.hint().contains("'counts', 'log1p', 'meta'"),
        "{}",
        error.hint()
    );
    let error = resolve_cell(&CellExpr::expression(None, "CD3E"), None).unwrap_err();
    assert_eq!(error.code(), "KY-E211");
    let error = resolve_gene(&GeneExpr::profile(None, "AAAC-1"), None).unwrap_err();
    assert_eq!(error.code(), "KY-E211");
    // A full name needs no default layer.
    assert!(feature("counts:CD3E", None).is_ok());
}

#[test]
fn ambiguous_without_default_layer_marks_shared_full_names() {
    // Column `CD4` (`meta:CD4`) and gene `CD4` in layer `meta` (`meta:CD4`) collide.
    let (_, candidates) = ambiguous(feature("CD4", None).unwrap_err());
    assert_eq!(
        candidates,
        [
            "meta:CD4 (cell column)",
            "counts:CD4",
            "log1p:CD4",
            "meta:CD4 (gene)"
        ]
    );
}

fn unknown(error: &KytoError) -> (String, String, String, String) {
    match error {
        KytoError::UnknownName {
            kind,
            name,
            expression,
            problem,
        } => (
            kind.clone(),
            name.clone(),
            expression.clone(),
            problem.clone(),
        ),
        other => panic!("expected KY-E210, got {other}"),
    }
}

#[test]
fn unknown_feature_names_say_why() {
    let problem = |name: &str| unknown(&feature(name, Some("log1p")).unwrap_err()).3;
    assert_eq!(
        problem("CD3"),
        "it is not a column of the cell table or a gene ID"
    );
    assert_eq!(
        problem("pca"),
        "'pca' is an embedding; name one component, such as 'pca:0'"
    );
    assert_eq!(
        problem("meta:louvain"),
        "the cell table has no column 'louvain'; layer 'meta' has no gene 'louvain'"
    );
    assert_eq!(problem("counts:CD3"), "layer 'counts' has no gene 'CD3'");
    assert_eq!(problem("pca:3"), "embedding 'pca' has 3 components, 0 to 2");
    assert_eq!(
        problem("pca:01"),
        "'01' is not a component of embedding 'pca' (components are written 0, 1, 2, ...)"
    );
    assert_eq!(
        problem("loadings:0"),
        "'loadings' is not 'meta', a layer or a cell embedding"
    );
    assert_eq!(
        problem("raw:CD3E"),
        "'raw' is not 'meta', a layer or a cell embedding"
    );
    let error = feature("CD3", Some("log1p")).unwrap_err();
    assert_eq!(error.code(), "KY-E210");
    assert_eq!(
        error.to_string(),
        "KY-E210: unknown feature 'CD3' in feature('CD3'): it is not a column of the cell \
         table or a gene ID"
    );
}

#[test]
fn constructor_names_are_checked() {
    let error = resolve_cell(&CellExpr::column("n_gene"), None).unwrap_err();
    assert_eq!(
        unknown(&error),
        (
            "cell column".to_owned(),
            "n_gene".to_owned(),
            "cell_column('n_gene')".to_owned(),
            "the cell table has no column with this name".to_owned()
        )
    );
    let error = resolve_cell(&CellExpr::column(""), None).unwrap_err();
    assert_eq!(unknown(&error).0, "cell column");
    let error = resolve_gene(&GeneExpr::column("n_genes"), None).unwrap_err();
    assert_eq!(unknown(&error).0, "gene column");
    // Genes: unknown and empty IDs.
    for gene in ["CD3", ""] {
        let error = resolve_cell(&CellExpr::expression(None, gene), Some("log1p")).unwrap_err();
        assert_eq!(unknown(&error).0, "gene");
    }
    // Cells: empty IDs.
    let error = resolve_gene(&GeneExpr::profile(None, ""), Some("log1p")).unwrap_err();
    assert_eq!(unknown(&error).0, "cell");
    // Layers named in full.
    let error =
        resolve_cell(&CellExpr::expression(Some("raw".to_owned()), "CD3E"), None).unwrap_err();
    assert_eq!(
        error.to_string(),
        "KY-E109: no layer named 'raw' in the store at 'pbmc.kyto'"
    );
    let error = resolve_gene(&GeneExpr::profile(Some(String::new()), "AAAC-1"), None).unwrap_err();
    assert_eq!(error.code(), "KY-E109");
    // Embeddings: missing, other axis, component out of range.
    let error = resolve_cell(&CellExpr::embedding("umap", 0), None).unwrap_err();
    assert_eq!(error.code(), "KY-E109");
    let error = resolve_cell(&CellExpr::embedding("loadings", 0), None).unwrap_err();
    assert_eq!(
        unknown(&error).3,
        "'loadings' is a gene embedding; read it with gene_embedding('loadings', 0)"
    );
    let error = resolve_cell(&CellExpr::embedding("pca", 3), None).unwrap_err();
    assert_eq!(
        unknown(&error),
        (
            "embedding component".to_owned(),
            "3".to_owned(),
            "cell_embedding('pca', 3)".to_owned(),
            "'pca' has 3 components, 0 to 2".to_owned()
        )
    );
    assert!(resolve_gene(&GeneExpr::embedding("loadings", 2), None).is_ok());
}

#[test]
fn layers_are_written_in_full() {
    let expr = resolve_cell(&CellExpr::expression(None, "CD3E"), Some("counts")).unwrap();
    assert_eq!(
        expr.node().as_ref(),
        &ExprNode::Source(Source::Expression {
            layer: Some("counts".to_owned()),
            gene_id: "CD3E".to_owned()
        })
    );
    let expr = resolve_gene(&GeneExpr::profile(None, "AAAC-1"), Some("log1p")).unwrap();
    assert_eq!(expr.to_string(), "cell_profile('AAAC-1', layer='log1p')");
    // An explicit layer wins over the default; it is not a short name.
    let expr = resolve_cell(
        &CellExpr::expression(Some("meta".to_owned()), "CD3E"),
        Some("counts"),
    )
    .unwrap();
    assert_eq!(expr.to_string(), "gene_expression('CD3E', layer='meta')");
}

#[test]
fn operations_are_rechecked_with_resolved_kinds() {
    let column = |name: &str| CellExpr::column(name);
    // `~` on an integer column (Polars would do a bitwise not).
    let error = resolve_cell(&column("n_genes").not().unwrap(), None).unwrap_err();
    assert_eq!(error.code(), "KY-E205");
    assert!(
        error.to_string().contains("a numeric expression"),
        "{error}"
    );
    // Arithmetic on a string column, `&` on a number, `.str` on a number.
    let plus = CellExpr::binary(column("cluster"), BinaryOp::Add, Literal::Int(1)).unwrap();
    assert_eq!(resolve_cell(&plus, None).unwrap_err().code(), "KY-E205");
    let and = CellExpr::binary(column("n_genes"), BinaryOp::And, column("is_doublet")).unwrap();
    assert_eq!(resolve_cell(&and, None).unwrap_err().code(), "KY-E205");
    let starts = column("n_genes")
        .str(StrFunction::StartsWith("1".to_owned()))
        .unwrap();
    assert_eq!(resolve_cell(&starts, None).unwrap_err().code(), "KY-E205");
    // Comparing a number with a string.
    let compare = CellExpr::binary(
        column("n_genes"),
        BinaryOp::Eq,
        Literal::String("a".to_owned()),
    )
    .unwrap();
    assert_eq!(resolve_cell(&compare, None).unwrap_err().code(), "KY-E205");
    // Unknown on both sides, resolved to different kinds.
    let mixed = CellExpr::binary(column("n_genes"), BinaryOp::Lt, column("cluster")).unwrap();
    assert_eq!(resolve_cell(&mixed, None).unwrap_err().code(), "KY-E205");
    // `feature` operands are rechecked the same way.
    let feature_not = CellExpr::feature("n_genes").unwrap().not().unwrap();
    assert_eq!(
        resolve_cell(&feature_not, None).unwrap_err().code(),
        "KY-E205"
    );
    // Kinds that fit still resolve, nested.
    let fine = CellExpr::binary(
        CellExpr::binary(column("n_genes"), BinaryOp::Gt, Literal::Int(200)).unwrap(),
        BinaryOp::And,
        column("is_doublet").not().unwrap(),
    )
    .unwrap();
    let resolved = resolve_cell(&fine, None).unwrap();
    assert_eq!(resolved, fine);
    assert_eq!(resolved.dtype(), ExprDtype::Boolean);
}

#[test]
fn ordering_needs_an_ordered_categorical() {
    let column = |name: &str| CellExpr::column(name);
    let literal = || Literal::String("b".to_owned());
    for op in [BinaryOp::Lt, BinaryOp::LtEq, BinaryOp::Gt, BinaryOp::GtEq] {
        let unordered = CellExpr::binary(column("cluster"), op, literal()).unwrap();
        let error = resolve_cell(&unordered, None).unwrap_err();
        assert_eq!(error.code(), "KY-E205");
        assert!(
            error
                .to_string()
                .contains("got an unordered categorical column: cell_column('cluster')"),
            "{error}"
        );
        let reflected = CellExpr::binary(literal(), op, column("cluster")).unwrap();
        assert_eq!(
            resolve_cell(&reflected, None).unwrap_err().code(),
            "KY-E205"
        );
        assert!(
            resolve_cell(
                &CellExpr::binary(column("stage"), op, literal()).unwrap(),
                None
            )
            .is_ok()
        );
        assert!(
            resolve_cell(
                &CellExpr::binary(column("cell_id"), op, literal()).unwrap(),
                None
            )
            .is_ok()
        );
    }
    for op in [BinaryOp::Eq, BinaryOp::NotEq] {
        assert!(
            resolve_cell(
                &CellExpr::binary(column("cluster"), op, literal()).unwrap(),
                None
            )
            .is_ok()
        );
    }
    assert!(
        resolve_cell(
            &column("cluster")
                .str(StrFunction::StartsWith("T".to_owned()))
                .unwrap(),
            None
        )
        .is_ok()
    );
}

#[test]
fn resolving_twice_changes_nothing() {
    let expr = CellExpr::binary(
        CellExpr::feature("CD3E").unwrap(),
        BinaryOp::Add,
        CellExpr::feature("pca:0").unwrap(),
    )
    .unwrap();
    let once = resolve_cell(&expr, Some("log1p")).unwrap();
    assert_eq!(
        once.to_string(),
        "gene_expression('CD3E', layer='log1p') + cell_embedding('pca', 0)"
    );
    // The input is unchanged; a second pass (any default layer) gives the same.
    assert_eq!(expr.to_string(), "feature('CD3E') + feature('pca:0')");
    assert_eq!(resolve_cell(&once, Some("counts")).unwrap(), once);
    assert_eq!(resolve_cell(&once, None).unwrap(), once);
}

#[test]
fn deep_trees_resolve_without_overflow() {
    let mut expr = CellExpr::column("n_genes");
    for _ in 1..kyto_core::MAX_EXPR_DEPTH {
        expr = CellExpr::binary(expr, BinaryOp::Add, Literal::Int(1)).unwrap();
    }
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || resolve_cell(&expr, None).map(|resolved| resolved.depth()))
        .unwrap();
    assert_eq!(handle.join().unwrap().unwrap(), kyto_core::MAX_EXPR_DEPTH);
}
