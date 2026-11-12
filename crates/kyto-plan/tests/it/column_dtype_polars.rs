//! Task 4.2: the dtype `column_dtype` declares for a computed column against the dtype
//! Polars infers for the same expression (4.1 Open (4)).
#![allow(clippy::unwrap_used)]

use kyto_core::{Axis, BinaryOp, CellExpr, ColumnDtype, GeneIds, Literal};
use kyto_plan::{Plan, column_dtype};
use polars::prelude::{DataFrame, DataType, IntoLazy, NamedFrom, Series};

const NUMERIC: [(&str, ColumnDtype); 10] = [
    ("i8", ColumnDtype::Int8),
    ("i16", ColumnDtype::Int16),
    ("i32", ColumnDtype::Int32),
    ("i64", ColumnDtype::Int64),
    ("u8", ColumnDtype::UInt8),
    ("u16", ColumnDtype::UInt16),
    ("u32", ColumnDtype::UInt32),
    ("u64", ColumnDtype::UInt64),
    ("f32", ColumnDtype::Float32),
    ("f64", ColumnDtype::Float64),
];

fn polars_dtype(dtype: ColumnDtype) -> DataType {
    match dtype {
        ColumnDtype::Int8 => DataType::Int8,
        ColumnDtype::Int16 => DataType::Int16,
        ColumnDtype::Int32 => DataType::Int32,
        ColumnDtype::Int64 => DataType::Int64,
        ColumnDtype::UInt8 => DataType::UInt8,
        ColumnDtype::UInt16 => DataType::UInt16,
        ColumnDtype::UInt32 => DataType::UInt32,
        ColumnDtype::UInt64 => DataType::UInt64,
        ColumnDtype::Float32 => DataType::Float32,
        ColumnDtype::Float64 => DataType::Float64,
        ColumnDtype::Boolean => DataType::Boolean,
        other => panic!("not a numeric dtype: {other:?}"),
    }
}

fn plan() -> std::sync::Arc<Plan> {
    let columns: Vec<String> = NUMERIC
        .iter()
        .map(|(name, dtype)| format!(r#"{{"name": "{name}", "dtype": "{dtype:?}"}}"#))
        .collect();
    let schema = format!(
        r#"{{
  "n_cells": 2, "n_genes": 1,
  "tiling": {{"cell_block_size": 4096, "gene_block_size": 1024}},
  "cells": {{"columns": [{{"name": "cell_id", "dtype": "String"}}, {}]}},
  "genes": {{"columns": [{{"name": "gene_id", "dtype": "String"}}]}},
  "layers": [], "embeddings": [], "graphs": []
}}"#,
        columns.join(", ")
    );
    Plan::scan(
        "dtypes.kyto",
        serde_json::from_str(&schema).unwrap(),
        GeneIds::new(["G1".to_owned()]),
    )
}

fn frame() -> DataFrame {
    let columns = NUMERIC
        .iter()
        .map(|(name, dtype)| {
            Series::new((*name).into(), [1i64, 2])
                .cast(&polars_dtype(*dtype))
                .unwrap()
                .into()
        })
        .collect();
    DataFrame::new(2, columns).unwrap()
}

/// The dtype Polars gives `expr` on the cell table.
fn inferred(expr: &CellExpr) -> DataType {
    let lowered = expr.to_polars().unwrap().alias("out");
    let mut lazy = frame().lazy().select([lowered]);
    lazy.collect_schema().unwrap().get("out").unwrap().clone()
}

#[test]
fn declared_dtypes_match_polars() {
    let plan = plan();
    let schema = plan.output_schema();
    let column = CellExpr::column;
    let ops = [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div];
    let mut cases: Vec<CellExpr> = Vec::new();
    for (left, _) in NUMERIC {
        for (right, _) in NUMERIC {
            for op in ops {
                cases.push(CellExpr::binary(column(left), op, column(right)).unwrap());
            }
        }
        for op in ops {
            // Literals that fit every dtype, and ones that do not (Polars widens).
            for literal in [
                Literal::Int(2),
                Literal::Int(-1),
                Literal::Int(300),
                Literal::Int(70_000),
                Literal::Int(3_000_000_000),
                Literal::Int(i64::MIN),
                Literal::Float(0.5),
            ] {
                cases.push(CellExpr::binary(column(left), op, literal.clone()).unwrap());
                cases.push(CellExpr::binary(literal, op, column(left)).unwrap());
            }
        }
    }
    // Nested: a literal adapts to the declared dtype of the expression beside it.
    let pair = |left, op, right| CellExpr::binary(column(left), op, column(right)).unwrap();
    for (inner, op, literal) in [
        (
            pair("i8", BinaryOp::Add, "u8"),
            BinaryOp::Mul,
            Literal::Int(2),
        ),
        (
            pair("f32", BinaryOp::Add, "i16"),
            BinaryOp::Div,
            Literal::Int(2),
        ),
        (
            pair("u8", BinaryOp::Sub, "u16"),
            BinaryOp::Add,
            Literal::Float(0.5),
        ),
        (
            pair("i32", BinaryOp::Div, "u8"),
            BinaryOp::Sub,
            Literal::Int(1),
        ),
        (
            pair("u64", BinaryOp::Mul, "i8"),
            BinaryOp::Add,
            Literal::Int(1),
        ),
    ] {
        cases.push(CellExpr::binary(inner.clone(), op, literal.clone()).unwrap());
        cases.push(CellExpr::binary(literal, op, inner.clone()).unwrap());
        cases.push(CellExpr::binary(inner.clone(), op, column("f32")).unwrap());
    }
    // Literal-only subtrees stay literals in Polars.
    let literals = |left: i64, op, right: i64| {
        CellExpr::binary(Literal::Int(left), op, Literal::Int(right)).unwrap()
    };
    for (name, _) in NUMERIC {
        for op in ops {
            let inner = literals(-1, BinaryOp::Mul, 1);
            cases.push(CellExpr::binary(inner.clone(), op, column(name)).unwrap());
            cases.push(CellExpr::binary(column(name), op, inner).unwrap());
            let big = literals(3_000_000_000, BinaryOp::Add, 1);
            cases.push(CellExpr::binary(column(name), op, big).unwrap());
        }
    }
    cases.push(literals(1, BinaryOp::Add, 1));
    cases.push(literals(1, BinaryOp::Div, 2));
    let mut mismatches = Vec::new();
    for expr in &cases {
        let resolved = plan.resolve(expr).unwrap();
        let declared = polars_dtype(column_dtype(schema, Axis::Cells, resolved.node()));
        let polars = inferred(&resolved);
        if declared != polars {
            mismatches.push(format!("{resolved}: declared {declared}, Polars {polars}"));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {}:\n{}",
        mismatches.len(),
        cases.len(),
        mismatches.join("\n")
    );
}
