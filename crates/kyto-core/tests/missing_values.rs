//! Task 3.7, the missing-value rules of R9 at the level of expressions and the resolver:
//! missing metadata is null, tested with `is_null` / `is_not_null` and propagated by
//! Polars' rules when an expression is lowered (`NaN` is a value, not null, format.md
//! §5.5); a gene removed upstream is refused with `KY-E203` when the plan is built.
//! The densify-to-0 rule is tested in kyto-store (`tests/missing_values.rs`), where
//! `CellBlock` lives.
#![allow(clippy::unwrap_used)]

use kyto_core::{
    AtlasSchema, BinaryOp, CellExpr, ExprDtype, GeneExpr, GeneIds, Literal, MAX_EXPR_DEPTH,
    NullTest, RemovedGenes, Resolver, StrFunction,
};
use kyto_error::KytoError;
use polars::prelude::{
    BooleanChunked, DataFrame, DataType, FrozenCategories, IntoLazy, NamedFrom, Series, col,
};

const SCHEMA: &str = r#"{
  "n_cells": 4,
  "n_genes": 3,
  "tiling": {"cell_block_size": 4096, "gene_block_size": 1024},
  "cells": {"columns": [
    {"name": "cell_id", "dtype": "String"},
    {"name": "score", "dtype": "Float64"},
    {"name": "n_genes", "dtype": "Int32"},
    {"name": "is_doublet", "dtype": "Boolean"},
    {"name": "cell_type", "dtype": "Enum", "n_categories": 2, "ordered": false},
    {"name": "CD4", "dtype": "Float32"}
  ]},
  "genes": {"columns": [{"name": "gene_id", "dtype": "String"}]},
  "layers": [
    {"name": "counts", "dtype": "uint32", "nnz": 5},
    {"name": "log1p", "dtype": "float32", "nnz": 5}
  ],
  "embeddings": [],
  "graphs": []
}"#;

fn schema() -> AtlasSchema {
    serde_json::from_str(SCHEMA).unwrap()
}

fn genes() -> GeneIds {
    GeneIds::new(["CD3E", "CD4", "MS4A1"].into_iter().map(str::to_owned))
}

/// `CD3E` and `CD4` removed by `select_genes`, `MS4A1` kept.
fn removed() -> RemovedGenes {
    let mut removed = RemovedGenes::new();
    removed.insert(["CD3E".to_owned(), "CD4".to_owned()], "select_genes");
    removed
}

fn resolve(expr: &CellExpr, layer: Option<&str>) -> Result<CellExpr, KytoError> {
    let (schema, genes, removed) = (schema(), genes(), removed());
    Resolver::new(&schema, &genes, layer, "pbmc.kyto")
        .with_removed_genes(&removed)
        .resolve(expr)
}

/// The cell table of four cells: row 0 is complete, row 1 has nulls in every column,
/// row 2 has `NaN` (a value, not null) and row 3 is complete.
fn cells() -> DataFrame {
    let categories = FrozenCategories::new(["B", "T"]).unwrap();
    let cell_type = Series::new("cell_type".into(), &[Some("B"), None, Some("T"), Some("T")])
        .cast(&DataType::from_frozen_categories(categories))
        .unwrap();
    DataFrame::new_infer_height(vec![
        Series::new(
            "score".into(),
            &[Some(0.5), None, Some(f64::NAN), Some(3.0)],
        )
        .into(),
        Series::new(
            "n_genes".into(),
            &[Some(100_i32), None, Some(300), Some(900)],
        )
        .into(),
        Series::new(
            "is_doublet".into(),
            &[Some(false), None, Some(true), Some(false)],
        )
        .into(),
        cell_type.into(),
    ])
    .unwrap()
}

/// `expr` evaluated on [`cells`], as one optional bool per row.
fn evaluate(expr: &CellExpr) -> Vec<Option<bool>> {
    let frame = cells()
        .lazy()
        .select([expr.to_polars().unwrap().alias("out")])
        .collect()
        .unwrap();
    let values: &BooleanChunked = frame.column("out").unwrap().bool().unwrap();
    values.iter().collect()
}

/// The rows of [`cells`] a filter on `expr` keeps.
fn kept(expr: &CellExpr) -> Vec<i32> {
    let frame = cells()
        .lazy()
        .with_row_index("row", None)
        .filter(expr.to_polars().unwrap())
        .select([col("row").cast(DataType::Int32)])
        .collect()
        .unwrap();
    let rows = frame.column("row").unwrap().i32().unwrap();
    rows.iter().map(Option::unwrap).collect()
}

fn column(name: &str) -> CellExpr {
    resolve(&CellExpr::column(name), None).unwrap()
}

fn compare(name: &str, op: BinaryOp, literal: Literal) -> CellExpr {
    CellExpr::binary(column(name), op, literal).unwrap()
}

#[test]
fn null_tests_are_boolean_on_every_kind_and_written_as_python() {
    let tests = [
        (NullTest::IsNull, "is_null"),
        (NullTest::IsNotNull, "is_not_null"),
    ];
    for (test, name) in tests {
        assert_eq!(test.name(), name);
        for expr in [
            CellExpr::column("score"),
            CellExpr::column("cell_type"),
            CellExpr::column("is_doublet"),
            CellExpr::expression(None, "CD3E"),
        ] {
            let tested = expr.null_test(test).unwrap();
            assert_eq!(tested.dtype(), ExprDtype::Boolean);
            assert_eq!(tested.to_string(), format!("{expr}.{name}()"));
        }
    }
    // Methods bind tighter than operators and `~`: those inputs get parentheses.
    let below = compare("score", BinaryOp::Lt, Literal::Int(1));
    assert_eq!(
        below.null_test(NullTest::IsNull).unwrap().to_string(),
        "(cell_column('score') < 1).is_null()"
    );
    let negated = column("is_doublet").not().unwrap();
    assert_eq!(
        negated.null_test(NullTest::IsNotNull).unwrap().to_string(),
        "(~cell_column('is_doublet')).is_not_null()"
    );
    // And `~x.is_null()` is `~(x.is_null())` in Python, so no parentheses there.
    let not_null = column("score")
        .null_test(NullTest::IsNull)
        .unwrap()
        .not()
        .unwrap();
    assert_eq!(not_null.to_string(), "~cell_column('score').is_null()");
    // The result is boolean, so it combines with & and ~ and has no str namespace.
    let error = column("score")
        .null_test(NullTest::IsNull)
        .unwrap()
        .str(StrFunction::StartsWith("a".to_owned()))
        .unwrap_err();
    assert_eq!(error.code(), "KY-E205");
    let gene = GeneExpr::column("highly_variable")
        .null_test(NullTest::IsNull)
        .unwrap();
    assert_eq!(gene.to_string(), "gene_column('highly_variable').is_null()");
}

#[test]
fn null_tests_count_toward_the_depth_limit() {
    let mut expr = CellExpr::column("score");
    for _ in 1..MAX_EXPR_DEPTH {
        expr = expr.null_test(NullTest::IsNull).unwrap();
    }
    assert_eq!(expr.depth(), MAX_EXPR_DEPTH);
    let error = expr.null_test(NullTest::IsNotNull).unwrap_err();
    assert_eq!(error.code(), "KY-E207");
    assert!(error.to_string().contains("'is_not_null'"), "{error}");
}

#[test]
fn missing_metadata_is_null_and_nan_is_a_value() {
    for name in ["score", "n_genes", "is_doublet", "cell_type"] {
        let is_null = column(name).null_test(NullTest::IsNull).unwrap();
        assert!(is_null.is_metadata_only());
        // Only row 1 is missing; the NaN score of row 2 is a value (format.md §5.5).
        assert_eq!(
            evaluate(&is_null),
            [Some(false), Some(true), Some(false), Some(false)],
            "{name}"
        );
        let is_not_null = column(name).null_test(NullTest::IsNotNull).unwrap();
        assert_eq!(kept(&is_not_null), [0, 2, 3], "{name}");
    }
}

#[test]
fn nulls_propagate_through_operations_and_filters_drop_them() {
    // A comparison with a null is null, never true or false.
    let high = compare("n_genes", BinaryOp::Gt, Literal::Int(200));
    assert_eq!(evaluate(&high), [Some(false), None, Some(true), Some(true)]);
    assert_eq!(kept(&high), [2, 3]);
    // ~null is null: the missing row is dropped by both a filter and its negation
    // (pandas keeps it in the negation of a float or categorical comparison, where a
    // missing value compares False; deviations.md).
    let low = high.not().unwrap();
    assert_eq!(evaluate(&low), [Some(true), None, Some(false), Some(false)]);
    assert_eq!(kept(&low), [0]);
    let not_b = compare(
        "cell_type",
        BinaryOp::NotEq,
        Literal::String("B".to_owned()),
    );
    assert_eq!(kept(&not_b), [2, 3]);
    // Arithmetic keeps the null.
    let doubled = CellExpr::binary(
        CellExpr::binary(column("n_genes"), BinaryOp::Mul, Literal::Int(2)).unwrap(),
        BinaryOp::GtEq,
        Literal::Int(0),
    )
    .unwrap();
    assert_eq!(
        evaluate(&doubled),
        [Some(true), None, Some(true), Some(true)]
    );
    // & and | are three-valued: null | true is true, null & false is false.
    let either = CellExpr::binary(
        column("is_doublet"),
        BinaryOp::Or,
        compare("n_genes", BinaryOp::Gt, Literal::Int(-1))
            .null_test(NullTest::IsNull)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        evaluate(&either),
        [Some(false), Some(true), Some(true), Some(false)]
    );
    let both =
        CellExpr::binary(column("is_doublet"), BinaryOp::And, Literal::Boolean(false)).unwrap();
    assert_eq!(evaluate(&both), [Some(false); 4]);
    // To keep missing rows, say so.
    let low_or_missing = CellExpr::binary(
        low,
        BinaryOp::Or,
        column("n_genes").null_test(NullTest::IsNull).unwrap(),
    )
    .unwrap();
    assert_eq!(kept(&low_or_missing), [0, 1]);
}

#[test]
fn nan_follows_polars_ordering() {
    // Polars orders NaN above every number, so `NaN > 1` is true (NumPy and pandas
    // say false; deviations.md, an open question of task 3.7).
    let high = compare("score", BinaryOp::Gt, Literal::Int(1));
    assert_eq!(evaluate(&high), [Some(false), None, Some(true), Some(true)]);
}

#[test]
fn null_tests_resolve_with_their_input() {
    let expr = CellExpr::feature("cell_type")
        .unwrap()
        .null_test(NullTest::IsNull)
        .unwrap();
    let resolved = resolve(&expr, None).unwrap();
    assert_eq!(resolved.to_string(), "cell_column('cell_type').is_null()");
    assert_eq!(resolve(&resolved, None).unwrap(), resolved);
    // The input is still checked.
    let error = resolve(
        &CellExpr::column("missing")
            .null_test(NullTest::IsNull)
            .unwrap(),
        None,
    )
    .unwrap_err();
    assert_eq!(error.code(), "KY-E210");
}

fn removed_gene(error: &KytoError) -> (&str, &str) {
    match error {
        KytoError::RemovedGene { gene, removed_by } => (gene, removed_by),
        other => panic!("expected KY-E203, got {other}"),
    }
}

#[test]
fn reading_a_removed_gene_raises_ky_e203() {
    let reads = [
        CellExpr::expression(Some("log1p".to_owned()), "CD3E"),
        CellExpr::expression(None, "CD3E"),
        CellExpr::feature("CD3E").unwrap(),
        CellExpr::feature("counts:CD3E").unwrap(),
        // Inside an operation and a null test too.
        CellExpr::binary(
            CellExpr::expression(None, "CD3E"),
            BinaryOp::Gt,
            Literal::Int(0),
        )
        .unwrap()
        .null_test(NullTest::IsNull)
        .unwrap(),
    ];
    for read in reads {
        let error = resolve(&read, Some("log1p")).unwrap_err();
        assert_eq!(error.code(), "KY-E203", "{read}");
        assert_eq!(removed_gene(&error), ("CD3E", "select_genes"), "{read}");
        let message = error.to_string();
        assert!(
            message.contains("gene 'CD3E' was removed by 'select_genes'"),
            "{message}"
        );
    }
    // A kept gene reads as before.
    let kept = resolve(&CellExpr::feature("MS4A1").unwrap(), Some("log1p")).unwrap();
    assert_eq!(kept.to_string(), "gene_expression('MS4A1', layer='log1p')");
    // An unknown gene is still unknown, not removed.
    let error = resolve(&CellExpr::expression(None, "NOPE"), Some("log1p")).unwrap_err();
    assert_eq!(error.code(), "KY-E210");
    // A cell profile reads the remaining genes; removing some is not an error.
    let profile = GeneExpr::profile(None, "AAAC-1");
    let (schema, genes, removed) = (schema(), genes(), removed());
    let resolver =
        Resolver::new(&schema, &genes, Some("log1p"), "pbmc.kyto").with_removed_genes(&removed);
    assert!(resolver.resolve(&profile).is_ok());
}

#[test]
fn a_removed_gene_keeps_its_name_so_resolution_does_not_change() {
    // `CD4` is a column and a removed gene: still ambiguous (KY-E201), as before the
    // removal, so removing a gene never makes a short name mean something else.
    let error = resolve(&CellExpr::feature("CD4").unwrap(), Some("log1p")).unwrap_err();
    assert_eq!(error.code(), "KY-E201", "{error}");
    let column = resolve(&CellExpr::feature("meta:CD4").unwrap(), Some("log1p")).unwrap();
    assert_eq!(column.to_string(), "cell_column('CD4')");
    // Without a default layer, a short removed gene asks for a layer first (KY-E211),
    // as for any short gene name.
    let error = resolve(&CellExpr::feature("CD3E").unwrap(), None).unwrap_err();
    assert_eq!(error.code(), "KY-E211", "{error}");
}

#[test]
fn removed_genes_keep_the_upstream_node() {
    let mut removed = RemovedGenes::new();
    assert!(removed.is_empty());
    removed.insert(["CD3E".to_owned()], "select_genes");
    removed.insert(["CD3E".to_owned(), "CD4".to_owned()], "filter_genes");
    assert_eq!(removed.len(), 2);
    assert_eq!(removed.removed_by("CD3E"), Some("select_genes"));
    assert_eq!(removed.removed_by("CD4"), Some("filter_genes"));
    assert_eq!(removed.removed_by("MS4A1"), None);
    assert_eq!(format!("{removed:?}"), "RemovedGenes(2 genes)");
}

#[test]
fn a_resolver_without_removed_genes_reads_every_gene() {
    let (schema, genes) = (schema(), genes());
    let resolver = Resolver::new(&schema, &genes, Some("log1p"), "pbmc.kyto");
    assert!(
        resolver
            .resolve(&CellExpr::expression(None, "CD3E"))
            .is_ok()
    );
    let empty = RemovedGenes::new();
    assert!(
        resolver
            .with_removed_genes(&empty)
            .resolve(&CellExpr::expression(None, "CD3E"))
            .is_ok()
    );
}
