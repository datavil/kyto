//! Task 6.13: per-gene QC metrics as gene expression leaves, and `log1p` on numeric
//! expressions. `GeneExpr::qc` builds the leaves, `Display` writes the Python code, the
//! resolver names the layer (R6) and refuses them on the cell axis, `log1p` accepts
//! numbers only and lowers to Polars, and the JSON forms round-trip.
#![allow(clippy::unwrap_used)]

use kyto_core::{
    AtlasSchema, BinaryOp, CellAxis, CellExpr, ExprJson, GeneAxis, GeneExpr, GeneIds, GeneQcMetric,
    Literal, Operand, QcMetric, Resolver,
};
use kyto_error::KytoError;
use polars::prelude::{DataFrame, IntoLazy, NamedFrom, Series};

fn schema() -> AtlasSchema {
    serde_json::from_str(
        r#"{
  "n_cells": 4,
  "n_genes": 3,
  "tiling": {"cell_block_size": 4096, "gene_block_size": 1024},
  "cells": {"columns": [
    {"name": "cell_id", "dtype": "String"},
    {"name": "depth", "dtype": "Float64"}
  ]},
  "genes": {"columns": [
    {"name": "gene_id", "dtype": "String"},
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

fn resolve_genes(expr: &GeneExpr, default_layer: Option<&str>) -> Result<GeneExpr, KytoError> {
    let schema = schema();
    let genes = GeneIds::new(["G1", "G2", "G3"].map(str::to_owned));
    Resolver::new(&schema, &genes, default_layer, "pbmc.kyto").resolve(expr)
}

fn qc(layer: Option<&str>, metric: GeneQcMetric) -> GeneExpr {
    GeneExpr::qc(layer.map(str::to_owned), metric)
}

#[test]
fn display_writes_the_python_call() {
    let expected = [
        "cells_detected",
        "gene_total_counts",
        "gene_mean_counts",
        "percent_dropout",
    ];
    for (metric, function) in GeneQcMetric::ALL.into_iter().zip(expected) {
        assert_eq!(metric.function(), function);
        assert_eq!(GeneQcMetric::from_function(function), Some(metric));
        assert_eq!(qc(None, metric).to_string(), format!("{function}()"));
        assert_eq!(
            qc(Some("counts"), metric).to_string(),
            format!("{function}(layer='counts')")
        );
    }
    assert_eq!(GeneQcMetric::from_function("total_counts"), None);
    let filter = GeneExpr::binary(
        Operand::Expr(qc(None, GeneQcMetric::CellsDetected)),
        BinaryOp::GtEq,
        Operand::Literal(Literal::Int(3)),
    )
    .unwrap();
    assert_eq!(filter.to_string(), "cells_detected() >= 3");
    assert!(!filter.is_metadata_only());
}

#[test]
fn the_resolver_names_the_layer() {
    let resolved = resolve_genes(&qc(None, GeneQcMetric::MeanCounts), Some("counts")).unwrap();
    assert_eq!(resolved.to_string(), "gene_mean_counts(layer='counts')");
    let missing = resolve_genes(&qc(None, GeneQcMetric::CellsDetected), None).unwrap_err();
    assert_eq!(missing.code(), "KY-E211");
    let unknown = resolve_genes(&qc(Some("raw"), GeneQcMetric::TotalCounts), None).unwrap_err();
    assert_eq!(unknown.code(), "KY-E109");
    assert!(unknown.to_string().contains("raw"), "{unknown}");
}

#[test]
fn gene_qc_is_refused_on_the_cell_axis() {
    // Only plan JSON can put the leaf on the other axis: the constructors are typed.
    let text = r#"[{"op": "gene_qc", "metric": "cells_detected", "layer": null}]"#;
    let json: ExprJson = serde_json::from_str(text).unwrap();
    let error = json.to_expr::<CellAxis>("test").unwrap_err();
    assert_eq!(error.code(), "KY-E217");
    assert!(error.to_string().contains("gene_qc"), "{error}");
    assert_eq!(
        json.to_expr::<GeneAxis>("test").unwrap().to_string(),
        "cells_detected()"
    );
    let unknown = r#"[{"op": "gene_qc", "metric": "total_counts", "layer": null}]"#;
    assert!(serde_json::from_str::<ExprJson>(unknown).is_err());
}

#[test]
fn log1p_takes_numbers_only() {
    let total = CellExpr::qc(None, QcMetric::TotalCounts).unwrap();
    let logged = total.log1p().unwrap();
    assert_eq!(logged.to_string(), "total_counts().log1p()");
    assert_eq!(logged.depth(), total.depth() + 1);
    let sum = CellExpr::binary(
        Operand::Expr(CellExpr::column("depth")),
        BinaryOp::Add,
        Operand::Literal(Literal::Int(1)),
    )
    .unwrap();
    assert_eq!(
        sum.log1p().unwrap().to_string(),
        "(cell_column('depth') + 1).log1p()"
    );
    let flag = CellExpr::binary(
        Operand::Expr(CellExpr::column("depth")),
        BinaryOp::Gt,
        Operand::Literal(Literal::Int(1)),
    )
    .unwrap();
    let error = flag.log1p().unwrap_err();
    assert_eq!(error.code(), "KY-E205");
    assert!(error.to_string().contains("log1p"), "{error}");
    // A string column is known only once resolved: refused then.
    let schema = schema();
    let genes = GeneIds::new(["G1", "G2", "G3"].map(str::to_owned));
    let resolver = Resolver::new(&schema, &genes, None, "pbmc.kyto");
    let text = CellExpr::column("cell_id").log1p().unwrap();
    assert_eq!(resolver.resolve(&text).unwrap_err().code(), "KY-E205");
    let depth = CellExpr::column("depth").log1p().unwrap();
    assert_eq!(
        resolver.resolve(&depth).unwrap().to_string(),
        "cell_column('depth').log1p()"
    );
}

#[test]
fn log1p_lowers_to_polars_as_numpy() {
    let expr = GeneExpr::column("n_cells").log1p().unwrap();
    assert!(expr.is_metadata_only());
    let frame = DataFrame::new(
        4,
        vec![Series::new("n_cells".into(), [Some(0_i64), Some(9), Some(-1), None]).into()],
    )
    .unwrap();
    let result = frame
        .lazy()
        .select([expr.to_polars().unwrap().alias("out")])
        .collect()
        .unwrap();
    let values: Vec<Option<f64>> = result
        .column("out")
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .collect();
    assert_eq!(
        values,
        [
            Some(0.0),
            Some(10.0_f64.ln()),
            Some(f64::NEG_INFINITY),
            None
        ]
    );
}

#[test]
fn leaves_and_log1p_round_trip_through_json() {
    let mut exprs: Vec<GeneExpr> = GeneQcMetric::ALL
        .into_iter()
        .flat_map(|metric| [qc(None, metric), qc(Some("counts"), metric)])
        .collect();
    exprs.push(qc(None, GeneQcMetric::TotalCounts).log1p().unwrap());
    exprs.push(
        GeneExpr::column("n_cells")
            .log1p()
            .unwrap()
            .log1p()
            .unwrap(),
    );
    for expr in exprs {
        let json = ExprJson::from_expr(&expr);
        let text = serde_json::to_string(&json).unwrap();
        let back: ExprJson = serde_json::from_str(&text).unwrap();
        let rebuilt: GeneExpr = back.to_expr("test").unwrap();
        assert_eq!(rebuilt, expr, "{text}");
        assert_eq!(rebuilt.depth(), expr.depth());
    }
    let text = serde_json::to_string(&ExprJson::from_expr(
        &qc(Some("counts"), GeneQcMetric::PercentDropout)
            .log1p()
            .unwrap(),
    ))
    .unwrap();
    assert_eq!(
        text,
        r#"[{"op":"gene_qc","metric":"percent_dropout","layer":"counts"},{"op":"log1p","input":0}]"#
    );
}
