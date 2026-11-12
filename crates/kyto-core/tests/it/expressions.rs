//! Task 3.3: typed expressions. Dtype kinds through operations, `KY-E205` on wrong
//! operands, `KY-E202` text for mixed axes, Python code rendering, and lowering of
//! metadata-only expressions to Polars (evaluated on a frame, not just built).
#![allow(clippy::unwrap_used)]

use kyto_core::{
    BinaryOp, CellExpr, ExprDtype, ExprNode, GeneExpr, Literal, Source, StrFunction, mixed_axes,
};
use kyto_error::KytoError;
use polars::prelude::{DataFrame, IntoLazy, NamedFrom, Series, col};

fn n_genes() -> CellExpr {
    CellExpr::column("n_genes")
}

fn gt(expr: &CellExpr, value: i64) -> CellExpr {
    CellExpr::binary(expr.clone(), BinaryOp::Gt, Literal::Int(value)).unwrap()
}

#[test]
fn sources_carry_their_tag_and_dtype() {
    let column = n_genes();
    assert_eq!(column.dtype(), ExprDtype::Unknown);
    assert_eq!(
        **column.node(),
        ExprNode::Source(Source::Column {
            name: "n_genes".to_owned()
        })
    );
    let expression = CellExpr::expression(Some("log1p".to_owned()), "CD3E");
    assert_eq!(expression.dtype(), ExprDtype::Numeric);
    assert!(!expression.is_metadata_only());
    assert_eq!(
        expression.to_string(),
        "gene_expression('CD3E', layer='log1p')"
    );
    let profile = GeneExpr::profile(None, "AAAC-1");
    assert_eq!(profile.dtype(), ExprDtype::Numeric);
    assert_eq!(profile.to_string(), "cell_profile('AAAC-1')");
    let embedding = GeneExpr::embedding("pca", 3);
    assert_eq!(embedding.dtype(), ExprDtype::Numeric);
    assert_eq!(embedding.to_string(), "gene_embedding('pca', 3)");
    assert_eq!(
        CellExpr::embedding("umap", 0).to_string(),
        "cell_embedding('umap', 0)"
    );
    assert!(embedding.to_polars().is_none());
}

#[test]
fn operations_set_the_result_dtype() {
    let numeric = CellExpr::binary(n_genes(), BinaryOp::Add, Literal::Int(1)).unwrap();
    assert_eq!(numeric.dtype(), ExprDtype::Numeric);
    let boolean = gt(&n_genes(), 200);
    assert_eq!(boolean.dtype(), ExprDtype::Boolean);
    let both = CellExpr::binary(boolean.clone(), BinaryOp::And, boolean.clone()).unwrap();
    assert_eq!(both.dtype(), ExprDtype::Boolean);
    assert_eq!(both.not().unwrap().dtype(), ExprDtype::Boolean);
    let prefix = CellExpr::column("symbol")
        .str(StrFunction::StartsWith("MT-".to_owned()))
        .unwrap();
    assert_eq!(prefix.dtype(), ExprDtype::Boolean);
}

#[test]
fn operations_leave_their_inputs_unchanged() {
    let column = n_genes();
    let before = column.clone();
    let _ = gt(&column, 1);
    assert_eq!(column, before);
}

fn assert_operand_error(error: KytoError, operation: &str, got: &str, operand: &str) {
    assert_eq!(error.code(), "KY-E205", "{error}");
    match error {
        KytoError::ExpressionOperand {
            operation: actual_operation,
            got: actual_got,
            operand: actual_operand,
            ..
        } => {
            assert_eq!(actual_operation, operation);
            assert_eq!(actual_got, got);
            assert_eq!(actual_operand, operand);
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn str_on_a_numeric_expression_is_ky_e205() {
    let numeric = CellExpr::binary(n_genes(), BinaryOp::Add, Literal::Int(1)).unwrap();
    let error = numeric
        .str(StrFunction::StartsWith("MT-".to_owned()))
        .unwrap_err();
    assert_operand_error(
        error,
        "str.starts_with",
        "a numeric expression",
        "cell_column('n_genes') + 1",
    );
}

#[test]
fn wrong_operand_kinds_are_ky_e205() {
    let boolean = gt(&n_genes(), 200);
    assert_operand_error(
        CellExpr::binary(boolean.clone(), BinaryOp::Add, Literal::Int(1)).unwrap_err(),
        "+",
        "a boolean expression",
        "cell_column('n_genes') > 200",
    );
    assert_operand_error(
        CellExpr::binary(n_genes(), BinaryOp::Mul, Literal::String("a".to_owned())).unwrap_err(),
        "*",
        "a string value",
        "'a'",
    );
    let numeric = CellExpr::expression(None, "CD3E");
    assert_operand_error(
        CellExpr::binary(numeric.clone(), BinaryOp::And, boolean.clone()).unwrap_err(),
        "&",
        "a numeric expression",
        "gene_expression('CD3E')",
    );
    assert_operand_error(
        numeric.not().unwrap_err(),
        "~",
        "a numeric expression",
        "gene_expression('CD3E')",
    );
    // Ordering booleans is not accepted.
    assert_operand_error(
        CellExpr::binary(boolean.clone(), BinaryOp::Lt, Literal::Boolean(true)).unwrap_err(),
        "<",
        "a boolean expression",
        "cell_column('n_genes') > 200",
    );
}

#[test]
fn comparing_different_kinds_is_ky_e205_and_blames_the_literal() {
    let numeric = CellExpr::expression(None, "CD3E");
    let error = CellExpr::binary(
        Literal::String("x".to_owned()),
        BinaryOp::Eq,
        numeric.clone(),
    )
    .unwrap_err();
    assert_operand_error(error, "==", "a string value", "'x'");
    let error = CellExpr::binary(numeric, BinaryOp::Eq, Literal::Boolean(false)).unwrap_err();
    match &error {
        KytoError::ExpressionOperand { accepted, .. } => {
            assert_eq!(accepted, "a numeric operand, like the other side");
        }
        other => panic!("unexpected {other:?}"),
    }
    // An unknown column is checked when the plan resolves it, not here.
    assert!(CellExpr::binary(n_genes(), BinaryOp::Eq, Literal::String("x".to_owned())).is_ok());
}

#[test]
fn mixed_axes_is_ky_e202_with_both_sides() {
    let cell = gt(&n_genes(), 200);
    let gene = GeneExpr::column("highly_variable");
    let error = mixed_axes("&", &cell, &gene);
    assert_eq!(error.code(), "KY-E202");
    assert_eq!(
        error.to_string(),
        "KY-E202: '&' mixes a cell expression (cell_column('n_genes') > 200) with a gene \
         expression (gene_column('highly_variable'))"
    );
}

#[test]
fn display_parenthesizes_nested_operations_and_escapes_strings() {
    let left = gt(&n_genes(), 200);
    let right = CellExpr::column("it's \\ odd")
        .str(StrFunction::EndsWith("x\n".to_owned()))
        .unwrap();
    let both = CellExpr::binary(left, BinaryOp::And, right).unwrap();
    assert_eq!(
        both.not().unwrap().to_string(),
        "~((cell_column('n_genes') > 200) & cell_column('it\\'s \\\\ odd').str.ends_with('x\\n'))"
    );
    let reversed = CellExpr::binary(Literal::Float(1.5), BinaryOp::Sub, n_genes()).unwrap();
    assert_eq!(reversed.to_string(), "1.5 - cell_column('n_genes')");
    let nan = CellExpr::binary(n_genes(), BinaryOp::Lt, Literal::Float(f64::NAN)).unwrap();
    assert_eq!(nan.to_string(), "cell_column('n_genes') < float('nan')");
    let negated = CellExpr::column("flag").not().unwrap();
    assert_eq!(negated.to_string(), "~cell_column('flag')");
}

#[test]
fn metadata_expressions_lower_to_polars_and_evaluate() {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("n_genes".into(), &[100_i64, 250, 300, 900]).into(),
        Series::new("symbol".into(), &["MT-CO1", "CD3E", "MT-ND1", "CD4"]).into(),
        Series::new("doublet".into(), &[false, false, true, false]).into(),
    ])
    .unwrap();
    // (n_genes / 2 > 120) & ~doublet & ~symbol.str.starts_with("MT-"), plus n_genes - 1 >= 899
    let half = CellExpr::binary(n_genes(), BinaryOp::Div, Literal::Int(2)).unwrap();
    let many = CellExpr::binary(half, BinaryOp::Gt, Literal::Int(120)).unwrap();
    let single = CellExpr::column("doublet").not().unwrap();
    let mito = CellExpr::column("symbol")
        .str(StrFunction::StartsWith("MT-".to_owned()))
        .unwrap()
        .not()
        .unwrap();
    let keep = CellExpr::binary(
        CellExpr::binary(many, BinaryOp::And, single).unwrap(),
        BinaryOp::And,
        mito,
    )
    .unwrap();
    let last = CellExpr::binary(
        CellExpr::binary(n_genes(), BinaryOp::Sub, Literal::Int(1)).unwrap(),
        BinaryOp::GtEq,
        Literal::Int(899),
    )
    .unwrap();
    let either = CellExpr::binary(keep.clone(), BinaryOp::Or, last).unwrap();
    assert!(either.is_metadata_only());

    let kept = frame
        .clone()
        .lazy()
        .filter(keep.to_polars().unwrap())
        .select([col("symbol")])
        .collect()
        .unwrap();
    let symbols: Vec<Option<&str>> = kept
        .column("symbol")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .collect();
    assert_eq!(symbols, [Some("CD3E"), Some("CD4")]);

    let ends = CellExpr::column("symbol")
        .str(StrFunction::EndsWith("1".to_owned()))
        .unwrap();
    let not_equal = CellExpr::binary(
        CellExpr::column("symbol"),
        BinaryOp::NotEq,
        Literal::String("CD4".to_owned()),
    )
    .unwrap();
    let both = CellExpr::binary(ends, BinaryOp::And, not_equal).unwrap();
    let kept = frame
        .lazy()
        .filter(both.to_polars().unwrap())
        .collect()
        .unwrap();
    assert_eq!(kept.height(), 2);
}

#[test]
fn true_division_of_integers_gives_floats() {
    let frame =
        DataFrame::new_infer_height(vec![Series::new("n".into(), &[3_i64, 4]).into()]).unwrap();
    let half = CellExpr::binary(CellExpr::column("n"), BinaryOp::Div, Literal::Int(2)).unwrap();
    let out = frame
        .lazy()
        .select([half.to_polars().unwrap()])
        .collect()
        .unwrap();
    let values: Vec<Option<f64>> = out.column("n").unwrap().f64().unwrap().iter().collect();
    assert_eq!(values, [Some(1.5), Some(2.0)]);
}

#[test]
fn matrix_sources_inside_an_operation_do_not_lower() {
    let mixed = CellExpr::binary(
        CellExpr::expression(None, "CD3E"),
        BinaryOp::Gt,
        CellExpr::column("threshold"),
    )
    .unwrap();
    assert!(!mixed.is_metadata_only());
    assert!(mixed.to_polars().is_none());
}

#[test]
fn column_dtypes_map_to_kinds() {
    use kyto_core::ColumnDtype;
    assert_eq!(ExprDtype::of_column(ColumnDtype::UInt8), ExprDtype::Numeric);
    assert_eq!(
        ExprDtype::of_column(ColumnDtype::Float64),
        ExprDtype::Numeric
    );
    assert_eq!(
        ExprDtype::of_column(ColumnDtype::Boolean),
        ExprDtype::Boolean
    );
    assert_eq!(ExprDtype::of_column(ColumnDtype::String), ExprDtype::String);
    assert_eq!(
        ExprDtype::of_column(ColumnDtype::Enum {
            n_categories: 3,
            ordered: false
        }),
        ExprDtype::String
    );
}

#[test]
fn trees_deeper_than_the_limit_are_ky_e207() {
    use kyto_core::MAX_EXPR_DEPTH;
    let mut expr = n_genes();
    for _ in 1..MAX_EXPR_DEPTH {
        expr = CellExpr::binary(expr, BinaryOp::Add, Literal::Int(1)).unwrap();
    }
    assert_eq!(expr.depth(), MAX_EXPR_DEPTH);
    let error = CellExpr::binary(Literal::Int(1), BinaryOp::Sub, expr.clone()).unwrap_err();
    assert_eq!(error.code(), "KY-E207");
    assert_eq!(
        error.to_string(),
        "KY-E207: '-' would make an expression 1001 levels deep; the limit is 1000"
    );
    let boolean = CellExpr::binary(expr.clone(), BinaryOp::Gt, Literal::Int(0));
    assert_eq!(boolean.unwrap_err().code(), "KY-E207");
    // The deepest tree renders, lowers and drops on a small (512 KiB) thread stack.
    let handle = std::thread::Builder::new()
        .stack_size(512 * 1024)
        .spawn(move || {
            let text = expr.to_string();
            let lowered = expr.to_polars().is_some();
            drop(expr);
            (text.len(), lowered)
        })
        .unwrap();
    let (length, lowered) = handle.join().unwrap();
    assert!(length > 4000);
    assert!(lowered);
}

#[test]
fn non_printable_characters_are_escaped() {
    let column = CellExpr::column("a\0b\u{1b}\u{2028}\u{a0}\u{200b}\u{7f}é\u{1F600} z");
    assert_eq!(
        column.to_string(),
        "cell_column('a\\x00b\\x1b\\u2028\\xa0\\u200b\\x7fé\u{1F600} z')"
    );
}

#[test]
fn str_functions_work_on_categorical_columns() {
    use polars::prelude::{DataType, FrozenCategories, NamedFrom as _};
    let symbols = Series::new("symbol".into(), &["MT-CO1", "CD3E", "MT-ND1"]);
    let categories = FrozenCategories::new(["CD3E", "MT-CO1", "MT-ND1"]).unwrap();
    let enum_symbols = symbols
        .cast(&DataType::from_frozen_categories(categories))
        .unwrap();
    let frame = DataFrame::new_infer_height(vec![enum_symbols.into()]).unwrap();
    let mito = CellExpr::column("symbol")
        .str(StrFunction::StartsWith("MT-".to_owned()))
        .unwrap();
    let kept = frame
        .lazy()
        .filter(mito.to_polars().unwrap())
        .collect()
        .unwrap();
    assert_eq!(kept.height(), 2);
}
