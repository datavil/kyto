//! Task 6.1: QC metrics as expression leaves. `CellExpr::qc` builds them, `Display`
//! writes the Python code, the resolver names the layer and checks the gene set (a
//! boolean predicate on gene-table columns, R6), and the JSON form round-trips.
#![allow(clippy::unwrap_used)]

use kyto_core::{
    AtlasSchema, BinaryOp, CellExpr, ExprJson, GeneExpr, GeneIds, Literal, Operand, QcMetric,
    Resolver, StrFunction,
};
use kyto_error::KytoError;

fn schema() -> AtlasSchema {
    serde_json::from_str(
        r#"{
  "n_cells": 4,
  "n_genes": 3,
  "tiling": {"cell_block_size": 4096, "gene_block_size": 1024},
  "cells": {"columns": [{"name": "cell_id", "dtype": "String"}]},
  "genes": {"columns": [
    {"name": "gene_id", "dtype": "String"},
    {"name": "symbol", "dtype": "Enum", "n_categories": 3, "ordered": false},
    {"name": "n_cells", "dtype": "Int64"}
  ]},
  "layers": [
    {"name": "counts", "dtype": "uint32", "nnz": 6},
    {"name": "log1p", "dtype": "float32", "nnz": 6}
  ],
  "embeddings": [],
  "graphs": []
}"#,
    )
    .unwrap()
}

fn mitochondrial() -> GeneExpr {
    GeneExpr::column("symbol")
        .str(StrFunction::StartsWith("MT-".to_owned()))
        .unwrap()
}

fn resolve(expr: &CellExpr, default_layer: Option<&str>) -> Result<CellExpr, KytoError> {
    let schema = schema();
    let genes = GeneIds::new(["G1", "G2", "G3"].map(str::to_owned));
    Resolver::new(&schema, &genes, default_layer, "pbmc.kyto").resolve(expr)
}

fn qc(layer: Option<&str>, metric: QcMetric) -> CellExpr {
    CellExpr::qc(layer.map(str::to_owned), metric).unwrap()
}

#[test]
fn qc_leaves_display_as_their_python_constructors() {
    assert_eq!(
        qc(None, QcMetric::TotalCounts).to_string(),
        "total_counts()"
    );
    assert_eq!(
        qc(Some("counts"), QcMetric::GenesDetected).to_string(),
        "genes_detected(layer='counts')"
    );
    let percent = qc(
        Some("counts"),
        QcMetric::PercentCountsIn {
            genes: mitochondrial(),
        },
    );
    assert_eq!(
        percent.to_string(),
        "percent_counts_in(gene_column('symbol').str.starts_with('MT-'), layer='counts')"
    );
    let total_in = qc(
        None,
        QcMetric::TotalCountsIn {
            genes: mitochondrial(),
        },
    );
    assert_eq!(
        total_in.to_string(),
        "total_counts_in(gene_column('symbol').str.starts_with('MT-'))"
    );
    assert!(!percent.is_metadata_only());
    // A QC leaf is numeric: arithmetic and comparisons build on it.
    let filter = CellExpr::binary(
        Operand::Expr(percent.clone()),
        BinaryOp::Lt,
        Operand::Literal(Literal::Float(5.0)),
    )
    .unwrap();
    assert_eq!(
        filter.to_string(),
        "percent_counts_in(gene_column('symbol').str.starts_with('MT-'), layer='counts') < 5.0"
    );
}

#[test]
fn resolving_names_the_layer_and_resolves_the_gene_set() {
    let resolved = resolve(&qc(None, QcMetric::TotalCounts), Some("counts")).unwrap();
    assert_eq!(resolved.to_string(), "total_counts(layer='counts')");
    let missing = resolve(&qc(None, QcMetric::GenesDetected), None).unwrap_err();
    assert_eq!(missing.code(), "KY-E211");
    let unknown_layer = resolve(&qc(Some("raw"), QcMetric::TotalCounts), None).unwrap_err();
    assert_eq!(unknown_layer.code(), "KY-E109");
    let percent = qc(
        Some("log1p"),
        QcMetric::PercentCountsIn {
            genes: mitochondrial(),
        },
    );
    assert_eq!(
        resolve(&percent, None).unwrap().to_string(),
        "percent_counts_in(gene_column('symbol').str.starts_with('MT-'), layer='log1p')"
    );
}

#[test]
fn a_gene_set_must_be_a_boolean_predicate_on_gene_columns() {
    // Numeric at construction: refused at once.
    let numeric = GeneExpr::profile(None, "AAAC-1");
    let error = CellExpr::qc(None, QcMetric::TotalCountsIn { genes: numeric }).unwrap_err();
    assert_eq!(error.code(), "KY-E205");
    assert!(error.to_string().contains("total_counts_in"), "{error}");
    // A column of unknown dtype is accepted, then refused when it resolves to an integer.
    let integer = qc(
        Some("counts"),
        QcMetric::TotalCountsIn {
            genes: GeneExpr::column("n_cells"),
        },
    );
    let error = resolve(&integer, None).unwrap_err();
    assert_eq!(error.code(), "KY-E205");
    assert!(
        error.to_string().contains("gene_column('n_cells')"),
        "{error}"
    );
    // A boolean predicate reading matrix values is not a gene set.
    let profile = GeneExpr::profile(Some("counts".to_owned()), "AAAC-1");
    let reads_matrix = GeneExpr::binary(
        Operand::Expr(profile),
        BinaryOp::Gt,
        Operand::Literal(Literal::Int(0)),
    )
    .unwrap();
    let error = resolve(
        &qc(
            Some("counts"),
            QcMetric::PercentCountsIn {
                genes: reads_matrix,
            },
        ),
        None,
    )
    .unwrap_err();
    assert_eq!(error.code(), "KY-E205");
    assert!(error.to_string().contains("matrix"), "{error}");
    // An unknown gene column is the usual KY-E210.
    let unknown = qc(
        Some("counts"),
        QcMetric::TotalCountsIn {
            genes: GeneExpr::column("mito"),
        },
    );
    assert_eq!(resolve(&unknown, None).unwrap_err().code(), "KY-E210");
}

#[test]
fn qc_leaves_round_trip_through_json() {
    for expr in [
        qc(None, QcMetric::TotalCounts),
        qc(Some("counts"), QcMetric::GenesDetected),
        qc(
            Some("counts"),
            QcMetric::TotalCountsIn {
                genes: mitochondrial(),
            },
        ),
        qc(
            None,
            QcMetric::PercentCountsIn {
                genes: mitochondrial(),
            },
        ),
    ] {
        let json = ExprJson::from_expr(&expr);
        let text = serde_json::to_string(&json).unwrap();
        let back: ExprJson = serde_json::from_str(&text).unwrap();
        let rebuilt: CellExpr = back.to_expr("test").unwrap();
        assert_eq!(rebuilt, expr, "{text}");
        assert_eq!(rebuilt.depth(), expr.depth());
    }
    let text = r#"[{"op": "qc", "metric": "total_counts", "layer": null, "genes": null}]"#;
    let json: ExprJson = serde_json::from_str(text).unwrap();
    assert_eq!(
        json.to_expr::<kyto_core::CellAxis>("test")
            .unwrap()
            .to_string(),
        "total_counts()"
    );
    // A metric and its gene set must agree; QC is cell-axis only.
    let without_genes =
        r#"[{"op": "qc", "metric": "percent_counts_in", "layer": null, "genes": null}]"#;
    let json: ExprJson = serde_json::from_str(without_genes).unwrap();
    assert_eq!(
        json.to_expr::<kyto_core::CellAxis>("test")
            .unwrap_err()
            .code(),
        "KY-E217"
    );
    let json: ExprJson = serde_json::from_str(text).unwrap();
    assert_eq!(
        json.to_expr::<kyto_core::GeneAxis>("test")
            .unwrap_err()
            .code(),
        "KY-E217"
    );
}

#[test]
fn a_nested_gene_set_counts_against_the_expansion_budget() {
    // A gene set of 2^19 nodes once expanded (each `&` uses the entry before twice),
    // under a cell expression that uses the QC leaf 2^19 times: refused.
    let mut genes = vec![r#"{"op": "column", "name": "flag"}"#.to_owned()];
    for level in 0..18 {
        genes.push(format!(
            r#"{{"op": "binary", "operator": "&", "left": {level}, "right": {level}}}"#
        ));
    }
    let mut entries = vec![format!(
        r#"{{"op": "qc", "metric": "total_counts_in", "layer": null, "genes": [{}]}}"#,
        genes.join(", ")
    )];
    for level in 0..18 {
        entries.push(format!(
            r#"{{"op": "binary", "operator": "+", "left": {level}, "right": {level}}}"#
        ));
    }
    let json: ExprJson = serde_json::from_str(&format!("[{}]", entries.join(", "))).unwrap();
    let error = json.to_expr::<kyto_core::CellAxis>("test").unwrap_err();
    assert_eq!(error.code(), "KY-E217");
    assert!(error.to_string().contains("expands"), "{error}");
}
