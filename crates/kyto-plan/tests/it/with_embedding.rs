//! `with_embedding` (task 6.11): the node, its checks, its plan JSON (bit for bit) and
//! the optimizer's rules around it (no cell filter moves across it).

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use kyto_core::{Axis, BinaryOp, CellExpr, Expr, FloatDtype, Literal};
use kyto_error::KytoError;
use kyto_plan::{GivenValues, OptimizeOptions, Output, Plan, WithEmbedding, optimize};

use crate::support::{N_CELLS, check_equal, scan};

/// `N_CELLS` rows of `width` awkward `float32` values: signed zeros, subnormals, the
/// extremes, values with no short decimal form.
fn awkward_f32(width: usize) -> Vec<f32> {
    let specials = [
        0.0_f32,
        -0.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(0x007f_ffff),
        f32::MAX,
        f32::MIN,
        0.1,
        1.0 / 3.0,
        f32::EPSILON,
    ];
    (0..N_CELLS * width)
        .map(|at| specials[at % specials.len()] * if at % 7 == 0 { 1.0 } else { 0.75 })
        .collect()
}

fn embedding(name: &str, width: u64, values: GivenValues) -> WithEmbedding {
    WithEmbedding::new(name, width, values)
}

fn code(error: &KytoError) -> &'static str {
    error.code()
}

fn score_filter() -> CellExpr {
    Expr::binary(CellExpr::column("score"), BinaryOp::Gt, Literal::Float(0.0)).unwrap()
}

#[test]
fn adds_a_cell_embedding_in_its_dtype() {
    let plan = scan()
        .with_embedding(embedding("X_scVI", 3, GivenValues::Float32(awkward_f32(3))))
        .unwrap();
    let schema = plan.output_schema();
    let found = schema.embedding("X_scVI").unwrap();
    assert_eq!(
        (found.axis, found.width, found.dtype),
        (Axis::Cells, 3, FloatDtype::Float32)
    );
    let line = plan.to_string();
    assert!(
        line.starts_with("WITH_EMBEDDING \"X_scVI\" (24 cells x 3 float32)"),
        "{line}"
    );
    // Replaces an embedding of that name, dtype included.
    let replaced = plan
        .with_embedding(embedding(
            "X_scVI",
            1,
            GivenValues::Float64(vec![2.5; N_CELLS]),
        ))
        .unwrap();
    let found = replaced.output_schema().embedding("X_scVI").unwrap();
    assert_eq!((found.width, found.dtype), (1, FloatDtype::Float64));
}

#[test]
fn refuses_bad_values_and_names_when_built() {
    let cases = [
        (
            embedding("X", 0, GivenValues::Float32(Vec::new())),
            "rows of at least one",
        ),
        (
            embedding("X", 2, GivenValues::Float32(vec![0.0; 3])),
            "rows of at least one",
        ),
        (
            embedding("X", 1, GivenValues::Float32(vec![0.0; N_CELLS - 1])),
            "the plan has 24 cells",
        ),
        (
            embedding("X", 2, {
                let mut values = vec![0.0; 2 * N_CELLS];
                values[5] = f64::NAN;
                GivenValues::Float64(values)
            }),
            "row 2, column 1",
        ),
        (
            embedding("X", 1, {
                let mut values = vec![0.0; N_CELLS];
                values[0] = f32::INFINITY;
                GivenValues::Float32(values)
            }),
            "NaN or infinite",
        ),
        (
            embedding("a/b", 1, GivenValues::Float32(vec![0.0; N_CELLS])),
            "a/b",
        ),
        // Differs from the stored `X_pca` only in case.
        (
            embedding("x_PCA", 1, GivenValues::Float32(vec![0.0; N_CELLS])),
            "x_PCA",
        ),
    ];
    for (params, part) in cases {
        let error = scan().with_embedding(params).unwrap_err();
        assert_eq!(code(&error), "KY-E215", "{error}");
        assert!(error.to_string().contains(part), "{error}");
    }
    // The store's gene embedding `PCs` cannot become a cell embedding.
    let error = scan()
        .with_embedding(embedding(
            "PCs",
            1,
            GivenValues::Float32(vec![0.0; N_CELLS]),
        ))
        .unwrap_err();
    assert!(error.to_string().contains("gene embedding"), "{error}");
    let sunk = scan()
        .sink(kyto_plan::Sink::Zarr {
            path: "out.kyto".to_owned(),
        })
        .unwrap();
    let error = sunk
        .with_embedding(embedding("X", 1, GivenValues::Float32(vec![0.0; N_CELLS])))
        .unwrap_err();
    assert_eq!(code(&error), "KY-E216");
}

#[test]
fn after_a_cell_filter_the_row_count_is_left_to_the_run() {
    // The cells kept are known only when the plan runs: any row count is accepted.
    let filtered = scan().filter_cells(&score_filter()).unwrap();
    assert_eq!(filtered.output_schema().n_cells(), None);
    filtered
        .with_embedding(embedding("X", 1, GivenValues::Float32(vec![1.0; 5])))
        .unwrap();
}

#[test]
fn plan_json_round_trips_every_bit() {
    let f32_plan = scan()
        .with_embedding(embedding("X_scVI", 4, GivenValues::Float32(awkward_f32(4))))
        .unwrap();
    let f64_values: Vec<f64> = (0..N_CELLS * 2)
        .map(|at| match at % 5 {
            0 => -0.0,
            1 => f64::from_bits(1),
            2 => f64::MAX,
            3 => 0.1 + 0.2,
            _ => -1.0 / 3.0,
        })
        .collect();
    let f64_plan = f32_plan
        .filter_cells(&score_filter())
        .unwrap()
        .with_embedding(embedding("X_harmony", 2, GivenValues::Float64(f64_values)))
        .unwrap();
    for plan in [f32_plan, f64_plan] {
        let text = plan.to_json();
        let back = Plan::from_json(&text).unwrap();
        assert_eq!(*back, *plan);
        assert_eq!(back.to_json(), text);
        for (left, right) in back.nodes().zip(plan.nodes()) {
            if let (Plan::WithEmbedding(left), Plan::WithEmbedding(right)) = (left, right) {
                assert_eq!(left.params(), right.params());
            }
        }
    }
}

#[test]
fn equal_only_when_every_bit_is() {
    let plan = |zero: f32| {
        scan()
            .with_embedding(embedding("X", 1, GivenValues::Float32(vec![zero; N_CELLS])))
            .unwrap()
    };
    assert_eq!(*plan(0.0), *plan(0.0));
    assert_ne!(*plan(0.0), *plan(-0.0));
    let float64 = scan()
        .with_embedding(embedding("X", 1, GivenValues::Float64(vec![0.0; N_CELLS])))
        .unwrap();
    assert_ne!(*plan(0.0), *float64);
}

#[test]
fn no_cell_filter_moves_across_it() {
    let values: Vec<f32> = (0..N_CELLS).map(|cell| cell as f32).collect();
    let plan = scan()
        .with_embedding(embedding("X", 1, GivenValues::Float32(values)))
        .unwrap()
        .filter_cells(&score_filter())
        .unwrap();
    let outputs = [Output::all(&plan)];
    let optimized = optimize(&outputs, OptimizeOptions::ALL);
    check_equal(&outputs, &optimized, OptimizeOptions::ALL).unwrap();
    let text = optimized.to_string();
    let filter = text.find("FILTER_CELLS").unwrap();
    let given = text.find("WITH_EMBEDDING").unwrap();
    // Printed from the top: the filter stays above the embedding.
    assert!(filter < given, "{text}");
    let _: &Arc<Plan> = &plan;
}
