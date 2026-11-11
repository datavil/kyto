//! Task 5.8: the executor runs filters, `select_genes`, `take_cells` and computed
//! columns.
//!
//! The 4.4 reference evaluator (`kyto-plan/tests/support`, shared through `#[path]`)
//! is the oracle: its seeded dataset is written as a real store (24 cells x 6 genes in
//! 8 x 4 tiles, so 3 cell blocks and 2 gene blocks), random chains of the steps the
//! executor runs (cell and gene filters over table columns, gene values and cell
//! profiles, `select_genes`, `take_cells`, `with_cells`, `with_genes`, per-cell and
//! median `normalize_total`, `log1p`, `with_default_layer`) are optimized with random
//! rules and run by the executor, and every cell, gene, layer value and table column
//! must equal the reference's run of the logical plan (4.4 Open (1): the optimizer's
//! rewrites checked against the real executor). Values are compared to 1e-5 relative
//! (the executor computes in `float32`, the reference in `f64`); everything else
//! exactly. A plan the reference cannot run (a cell or gene an earlier step removed)
//! must fail in the executor too (`KY-E210`). Each plan also runs at 4 threads with 3
//! blocks in flight and blocks of 3 cells, with the same bits (R11).
//!
//! `uint32` gene values (the `counts` layer) take part in arithmetic as `Float64`, as
//! the reference reads every layer (user decision, Nov 12, 2026; deviations.md).
//!
//! Not covered here: QC leaves (the reference has none; `kyto-algo`'s `qc_parity.rs`
//! and this crate's `filter_qc_parity.rs` compare them with scanpy), and `take_cells`
//! order: the executor keeps store order, so the
//! reference's rows are put in store order before comparing (plan 5.8, deviations.md).

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "../../kyto-plan/tests/support/mod.rs"]
mod support;

use std::collections::{BTreeMap, HashMap};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use kyto_core::{BinaryOp, CellExpr, Expr, GeneExpr, GeneIds, Literal, QcMetric, Tiling};
use kyto_error::KytoError;
use kyto_exec::{CellBlock, CellValues, ExecuteOptions, Pipeline};
use kyto_plan::{OptimizeOptions, Output, PhysicalOptions, Plan, physical_plan};
use kyto_store::{
    LayerSpec, StoreWriter, TableAxis, ValueDtype, open_atlas_schema, read_ids, read_symbols,
    read_table,
};
use polars::prelude::{AnyValue, Column, DataFrame, DataType, IdxCa, IdxSize};
use proptest::prelude::*;
use support::Data;

/// The reference dataset written as a store, and the data the reference runs on (the
/// `log1p` layer rounded to the `float32` the store keeps).
struct Fixture {
    _dir: tempfile::TempDir,
    path: PathBuf,
    data: Data,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reference.kyto");
        let mut data = support::dataset();
        for row in data.layers.get_mut("log1p").unwrap() {
            for value in row {
                *value = f64::from(*value as f32);
            }
        }
        write_store(&data, &path);
        Fixture {
            _dir: dir,
            path,
            data,
        }
    })
}

/// `rows` (dense) as a CSR block of its non-zero values.
fn block(rows: &[Vec<f64>], dtype: ValueDtype) -> CellBlock {
    let mut indptr = vec![0_u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in rows {
        for (gene, value) in row.iter().enumerate() {
            if *value != 0.0 {
                indices.push(gene as u32);
                values.push(*value);
            }
        }
        indptr.push(indices.len() as u64);
    }
    let n_genes = rows.first().map_or(0, Vec::len) as u64;
    match dtype {
        ValueDtype::U32 => {
            let values = values.iter().map(|value| *value as u32).collect();
            CellBlock::from_u32(indptr, indices, values, n_genes).unwrap()
        }
        ValueDtype::F32 => {
            let values = values.iter().map(|value| *value as f32).collect();
            CellBlock::from_f32(indptr, indices, values, n_genes).unwrap()
        }
    }
}

fn write_store(data: &Data, path: &Path) {
    let spec = |name: &str, dtype| LayerSpec {
        name: name.to_owned(),
        dtype,
        options: Default::default(),
    };
    let tiling = Tiling {
        cell_block_size: 8,
        gene_block_size: 4,
    };
    let mut writer = StoreWriter::create(
        path,
        &data.genes,
        &[
            spec("counts", ValueDtype::U32),
            spec("log1p", ValueDtype::F32),
        ],
        tiling,
        false,
    )
    .unwrap();
    writer
        .push(&[
            block(&data.layers["counts"], ValueDtype::U32),
            block(&data.layers["log1p"], ValueDtype::F32),
        ])
        .unwrap();
    writer.finish(&data.cells, None).unwrap();
}

fn scan(path: &Path) -> Arc<Plan> {
    let schema = open_atlas_schema(path).unwrap();
    let ids = read_ids(path, TableAxis::Genes, schema.n_genes).unwrap();
    let symbols = read_symbols(path, schema.n_genes).unwrap();
    Plan::scan(
        path.display().to_string(),
        schema,
        GeneIds::new(ids).with_symbols(symbols),
    )
}

// --- Running a plan in the executor -------------------------------------------------

/// What the executor gives for an output reading everything: the cells (store
/// positions), the cell and gene tables of the result, and each layer's dense rows.
#[derive(Debug)]
struct Run {
    cells: Vec<u64>,
    cell_table: DataFrame,
    gene_table: DataFrame,
    layers: BTreeMap<String, Vec<Vec<f64>>>,
    /// Every value, exactly, for comparing runs (R11).
    fingerprint: String,
}

#[derive(Clone, Copy)]
struct How {
    options: OptimizeOptions,
    threads: usize,
    blocks_in_flight: u64,
    max_block_cells: Option<u64>,
}

fn execute(plan: &Arc<Plan>, how: How) -> Result<Run, KytoError> {
    let output = Output::all(plan);
    let options = PhysicalOptions {
        blocks_in_flight: NonZeroU64::new(how.blocks_in_flight).unwrap(),
        max_block_cells: how.max_block_cells.and_then(NonZeroU64::new),
        ..PhysicalOptions::default()
    };
    let physical = physical_plan(std::slice::from_ref(&output), how.options, options);
    let pipeline = Pipeline::new(&physical, 0)?;
    let stream = pipeline.run(ExecuteOptions::new(NonZeroUsize::new(how.threads).unwrap()))?;
    let mut cells = Vec::new();
    let mut computed: Option<DataFrame> = None;
    let mut layers: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
    let mut blocks = Vec::new();
    let mut stream = stream;
    for block in stream.by_ref() {
        let block = block?;
        assert!(!block.cells.is_empty(), "an empty block is not handed over");
        assert!(block.cells.windows(2).all(|pair| pair[0] < pair[1]));
        cells.extend(&block.cells);
        if block.columns.width() > 0 {
            assert_eq!(block.columns.height(), block.cells.len());
            match &mut computed {
                Some(frame) => {
                    frame.vstack_mut(&block.columns).unwrap();
                }
                None => computed = Some(block.columns.clone()),
            }
        }
        for (name, layer) in &block.layers {
            assert_eq!(layer.n_rows(), block.cells.len());
            layers.entry(name.clone()).or_default().extend(dense(layer));
        }
        blocks.push(block.cell_block);
    }
    let path = &fixture().path;
    let store_cells = read_table(path, TableAxis::Cells).unwrap();
    let indices: Vec<IdxSize> = cells.iter().map(|cell| *cell as IdxSize).collect();
    let mut cell_table = store_cells
        .take(&IdxCa::from_vec("cells".into(), indices))
        .unwrap();
    if let Some(computed) = computed {
        for column in computed.columns() {
            cell_table.with_column(column.clone()).unwrap();
        }
    }
    let gene_table = match stream.genes() {
        Some(genes) => {
            assert_eq!(genes.positions().len(), genes.table().height());
            genes.table().clone()
        }
        None => read_table(path, TableAxis::Genes).unwrap(),
    };
    let fingerprint = format!(
        "{cells:?}\n{}\n{}\n{:?}",
        frame_text(&cell_table),
        frame_text(&gene_table),
        layers
            .iter()
            .map(|(name, rows)| (
                name,
                rows.iter()
                    .flatten()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            ))
            .collect::<Vec<_>>()
    );
    Ok(Run {
        cells,
        cell_table,
        gene_table,
        layers,
        fingerprint,
    })
}

/// Every value of `frame`, exactly (`f32`/`f64` print their shortest round trip).
fn frame_text(frame: &DataFrame) -> String {
    frame
        .columns()
        .iter()
        .map(|column| {
            let values: Vec<String> = column
                .as_materialized_series()
                .iter()
                .map(|value| format!("{value:?}"))
                .collect();
            format!("{} {:?} {values:?}", column.name(), column.dtype())
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn dense(layer: &CellBlock) -> Vec<Vec<f64>> {
    let n_genes = layer.n_genes() as usize;
    (0..layer.n_rows())
        .map(|row| {
            let range = layer.indptr()[row] as usize..layer.indptr()[row + 1] as usize;
            let mut values = vec![0.0; n_genes];
            for entry in range {
                let gene = layer.indices()[entry] as usize;
                values[gene] = match layer.data() {
                    CellValues::F32(data) => f64::from(data[entry]),
                    CellValues::U32(data) => f64::from(data[entry]),
                };
            }
            values
        })
        .collect()
}

// --- Comparing with the reference ----------------------------------------------------

fn close(left: f64, right: f64) -> bool {
    (left.is_nan() && right.is_nan())
        || left == right
        || (left - right).abs() <= 1e-5 * left.abs().max(right.abs()) + 1e-6
}

/// Two columns hold the same values: numbers to 1e-5, everything else as text.
fn same_column(name: &str, executor: &Column, reference: &Column) -> Result<(), String> {
    if executor.len() != reference.len() {
        return Err(format!(
            "column {name}: {} vs {} rows",
            executor.len(),
            reference.len()
        ));
    }
    let numeric =
        |column: &Column| column.dtype().is_primitive_numeric() || column.dtype().is_bool();
    if numeric(executor) && numeric(reference) {
        let as_f64 = |column: &Column| {
            column
                .cast(&DataType::Float64)
                .unwrap()
                .f64()
                .unwrap()
                .iter()
                .collect::<Vec<_>>()
        };
        for (row, (left, right)) in as_f64(executor)
            .into_iter()
            .zip(as_f64(reference))
            .enumerate()
        {
            let same = match (left, right) {
                (None, None) => true,
                (Some(left), Some(right)) => close(left, right),
                _ => false,
            };
            if !same {
                return Err(format!(
                    "column {name} row {row}: executor {left:?}, reference {right:?}"
                ));
            }
        }
        return Ok(());
    }
    let as_text = |column: &Column| -> Vec<String> {
        column
            .as_materialized_series()
            .iter()
            .map(|value| match value {
                AnyValue::Null => "null".to_owned(),
                value => value.str_value().into_owned(),
            })
            .collect()
    };
    if as_text(executor) == as_text(reference) {
        Ok(())
    } else {
        Err(format!(
            "column {name}: executor {:?}, reference {:?}",
            as_text(executor),
            as_text(reference)
        ))
    }
}

/// The reference's result with its cells in store order (the executor keeps it for
/// `take_cells`; every other step keeps the input's order).
fn in_store_order(mut data: Data) -> Data {
    let store_ids = support::cell_ids();
    let position: HashMap<&str, usize> = store_ids
        .iter()
        .enumerate()
        .map(|(position, id)| (id.as_str(), position))
        .collect();
    let ids = support::ids(&data.cells, "cell_id");
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by_key(|row| position[ids[*row].as_str()]);
    let indices: Vec<IdxSize> = order.iter().map(|row| *row as IdxSize).collect();
    data.cells = data
        .cells
        .take(&IdxCa::from_vec("rows".into(), indices))
        .unwrap();
    for rows in data.layers.values_mut() {
        *rows = order.iter().map(|row| rows[*row].clone()).collect();
    }
    data
}

fn compare(run: &Run, reference: &Data) -> Result<(), String> {
    let store_ids = support::cell_ids();
    let executor_ids: Vec<String> = run
        .cells
        .iter()
        .map(|cell| store_ids[*cell as usize].clone())
        .collect();
    let reference_ids = support::ids(&reference.cells, "cell_id");
    if executor_ids != reference_ids {
        return Err(format!(
            "cells: executor {executor_ids:?}, reference {reference_ids:?}"
        ));
    }
    let executor_genes = support::ids(&run.gene_table, "gene_id");
    let reference_genes = support::ids(&reference.genes, "gene_id");
    if executor_genes != reference_genes {
        return Err(format!(
            "genes: executor {executor_genes:?}, reference {reference_genes:?}"
        ));
    }
    for (frame, executor_frame, table) in [
        (&reference.cells, &run.cell_table, "cell"),
        (&reference.genes, &run.gene_table, "gene"),
    ] {
        for column in frame.columns() {
            let name = column.name().as_str();
            // With no cells left no block is handed over, so no block carries the
            // computed cell columns (a terminal takes their dtypes from the plan).
            if table == "cell" && frame.height() == 0 && executor_frame.column(name).is_err() {
                continue;
            }
            let executor = executor_frame
                .column(name)
                .map_err(|_| format!("{table} column {name} is missing"))?;
            same_column(name, executor, column)?;
        }
    }
    for (name, rows) in &reference.layers {
        let executor = run.layers.get(name);
        // A layer with no cells left has no rows in either run.
        if rows.is_empty() && executor.is_none() {
            continue;
        }
        let executor = executor.ok_or_else(|| format!("layer {name} is missing"))?;
        if executor.len() != rows.len() {
            return Err(format!(
                "layer {name}: {} vs {} rows",
                executor.len(),
                rows.len()
            ));
        }
        for (row, (left, right)) in executor.iter().zip(rows).enumerate() {
            if left.len() != right.len() || left.iter().zip(right).any(|(l, r)| !close(*l, *r)) {
                return Err(format!(
                    "layer {name} row {row}: executor {left:?}, reference {right:?}"
                ));
            }
        }
    }
    Ok(())
}

/// Runs `plan` in the reference and in the executor (two ways) and compares.
fn check(plan: &Arc<Plan>, options: OptimizeOptions) -> Result<(), String> {
    let reference = support::run_logical(plan, &fixture().data);
    let first = execute(
        plan,
        How {
            options,
            threads: 1,
            blocks_in_flight: 2,
            max_block_cells: None,
        },
    );
    let second = execute(
        plan,
        How {
            options,
            threads: 4,
            blocks_in_flight: 3,
            max_block_cells: Some(3),
        },
    );
    match (reference, first, second) {
        (Ok(reference), Ok(first), Ok(second)) => {
            if first.fingerprint != second.fingerprint {
                return Err("1 thread and 4 threads with smaller blocks differ".to_owned());
            }
            compare(&first, &in_store_order(reference))
        }
        (Err(_), Err(first), Err(second)) => {
            if first.code() != "KY-E210" || second.code() != first.code() {
                return Err(format!("the executor failed with {first} and {second}"));
            }
            Ok(())
        }
        (reference, first, second) => Err(format!(
            "reference {:?}, executor {:?} / {:?}",
            reference.map(|_| "ok"),
            first.map(|_| "ok"),
            second.map(|_| "ok"),
        )),
    }
}

// --- Plans ----------------------------------------------------------------------------

fn cell(name: &str) -> CellExpr {
    CellExpr::column(name)
}

fn cell_expr() -> impl Strategy<Value = CellExpr> {
    let leaf = prop_oneof![
        prop::sample::select(vec!["score", "batch", "total_counts", "double"]).prop_map(cell),
        (
            prop::sample::select(vec!["G0", "G1", "G2", "G5"]),
            prop::sample::select(vec![
                None,
                Some("counts"),
                Some("log1p"),
                Some("a"),
                Some("b")
            ])
        )
            .prop_map(|(gene_id, layer)| CellExpr::expression(layer.map(str::to_owned), gene_id)),
    ];
    leaf.prop_recursive(3, 12, 2, |inner| {
        prop_oneof![
            (inner.clone(), operator(), inner.clone())
                .prop_filter_map("kinds", |(left, op, right)| Expr::binary(left, op, right)
                    .ok()),
            (inner.clone(), operator(), literal())
                .prop_filter_map("kinds", |(left, op, right)| Expr::binary(left, op, right)
                    .ok()),
            inner.prop_filter_map("kinds", |expr| expr.not().ok()),
        ]
    })
}

fn gene_expr() -> impl Strategy<Value = GeneExpr> {
    prop_oneof![
        Just(GeneExpr::column("highly_variable")),
        (-1.0..3.0_f64).prop_map(|value| Expr::binary(
            GeneExpr::column("mean"),
            BinaryOp::Gt,
            Literal::Float(value)
        )
        .unwrap()),
        (
            prop::sample::select(vec!["c0", "c1", "c7"]),
            prop::sample::select(vec!["counts", "log1p", "a"])
        )
            .prop_map(|(cell_id, layer)| {
                Expr::binary(
                    GeneExpr::profile(Some(layer.to_owned()), cell_id),
                    BinaryOp::GtEq,
                    Literal::Float(1.0),
                )
                .unwrap()
            }),
    ]
}

fn operator() -> impl Strategy<Value = BinaryOp> {
    prop::sample::select(vec![
        BinaryOp::Eq,
        BinaryOp::Lt,
        BinaryOp::Gt,
        BinaryOp::Add,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::And,
        BinaryOp::Or,
    ])
}

fn literal() -> impl Strategy<Value = Literal> {
    prop_oneof![
        any::<bool>().prop_map(Literal::Boolean),
        (-5..5_i64).prop_map(Literal::Int),
        (-3.0..3.0_f64).prop_map(Literal::Float),
        prop::sample::select(vec!["b0", "b1"]).prop_map(|value| Literal::String(value.to_owned())),
    ]
}

#[derive(Debug, Clone)]
enum Step {
    DefaultLayer(&'static str),
    FilterCells(CellExpr),
    FilterGenes(GeneExpr),
    SelectGenes(Vec<String>),
    TakeCells(Vec<String>),
    NormalizeTotal(Option<f64>, &'static str, &'static str),
    Log1p(&'static str, &'static str),
    WithCells(&'static str, CellExpr),
    WithGenes(&'static str, GeneExpr),
}

fn step() -> impl Strategy<Value = Step> {
    let layers = || prop::sample::select(vec!["counts", "log1p", "a", "b"]);
    let ids = |pool: Vec<&'static str>| {
        prop::sample::subsequence(pool.clone(), 1..=pool.len())
            .prop_shuffle()
            .prop_map(|ids| ids.into_iter().map(str::to_owned).collect::<Vec<_>>())
    };
    prop_oneof![
        1 => layers().prop_map(Step::DefaultLayer),
        4 => cell_expr().prop_map(Step::FilterCells),
        2 => gene_expr().prop_map(Step::FilterGenes),
        1 => ids(vec!["G0", "G1", "G2", "G3", "G5"]).prop_map(Step::SelectGenes),
        1 => ids(vec!["c0", "c2", "c5", "c9", "c3", "c17", "c23"]).prop_map(Step::TakeCells),
        3 => (proptest::option::of(1.0..1e4_f64), layers(), layers())
            .prop_map(|(sum, layer, into)| Step::NormalizeTotal(sum, layer, into)),
        3 => (layers(), layers()).prop_map(|(layer, into)| Step::Log1p(layer, into)),
        3 => (prop::sample::select(vec!["double", "score"]), cell_expr()).prop_map(|(name, expr)| Step::WithCells(name, expr)),
        2 => (prop::sample::select(vec!["flag", "mean"]), gene_expr()).prop_map(|(name, expr)| Step::WithGenes(name, expr)),
    ]
}

fn apply(plan: &Arc<Plan>, step: Step) -> Option<Arc<Plan>> {
    match step {
        Step::DefaultLayer(layer) => plan.with_default_layer(layer),
        Step::FilterCells(predicate) => plan.filter_cells(&predicate),
        Step::FilterGenes(predicate) => plan.filter_genes(&predicate),
        Step::SelectGenes(ids) => plan.select_genes(ids),
        Step::TakeCells(ids) => plan.take_cells(ids),
        Step::NormalizeTotal(sum, layer, into) => plan.normalize_total(sum, Some(layer), into),
        Step::Log1p(layer, into) => plan.log1p(Some(layer), into),
        Step::WithCells(name, expr) => plan.with_cells([(name.to_owned(), expr)]),
        Step::WithGenes(name, expr) => plan.with_genes([(name.to_owned(), expr)]),
    }
    .ok()
}

fn chain(steps: Vec<Step>) -> Arc<Plan> {
    steps.into_iter().fold(scan(&fixture().path), |plan, step| {
        apply(&plan, step).unwrap_or(plan)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Random chains, random optimizer rules: the executor gives the reference's
    /// result, the same bits on 1 and 4 threads and any block size.
    #[test]
    fn random_plans_match_the_reference_evaluator(
        steps in proptest::collection::vec(step(), 0..9),
        rules in (any::<bool>(), any::<bool>(), any::<bool>(), any::<bool>()),
    ) {
        let plan = chain(steps);
        let options = OptimizeOptions {
            reuse_sub_plans: rules.0,
            projection_pushdown: rules.1,
            fuse_elementwise: rules.2,
            predicate_pushdown: rules.3,
        };
        if let Err(message) = check(&plan, options) {
            panic!("{message}\nplan:\n{plan}");
        }
    }
}

// --- Fixed cases ------------------------------------------------------------------------

fn all_rules() -> How {
    How {
        options: OptimizeOptions::ALL,
        threads: 2,
        blocks_in_flight: 2,
        max_block_cells: None,
    }
}

/// A standard chain: filter cells on a stored column, keep three genes in a new order,
/// compute a gene column from a cell's profile, normalize, log, compute a cell column
/// and filter on it.
#[test]
fn a_standard_chain_matches_the_reference() {
    let plan = scan(&fixture().path)
        .filter_cells(&Expr::binary(cell("score"), BinaryOp::Gt, Literal::Int(-2)).unwrap())
        .unwrap()
        .select_genes(["G5".to_owned(), "G0".to_owned(), "G2".to_owned()])
        .unwrap()
        .with_genes([(
            "seen".to_owned(),
            Expr::binary(
                GeneExpr::profile(Some("counts".to_owned()), "c1"),
                BinaryOp::Gt,
                Literal::Int(0),
            )
            .unwrap(),
        )])
        .unwrap()
        .normalize_total(Some(1e4), Some("counts"), "normalized")
        .unwrap()
        .log1p(Some("normalized"), "normalized")
        .unwrap()
        .with_cells([(
            "g0".to_owned(),
            CellExpr::expression(Some("normalized".to_owned()), "G0"),
        )])
        .unwrap()
        .filter_cells(&Expr::binary(cell("g0"), BinaryOp::Gt, Literal::Float(0.0)).unwrap())
        .unwrap();
    // The reference runs it (c1 passes the score filter: score 4).
    support::run_logical(&plan, &fixture().data).unwrap();
    check(&plan, OptimizeOptions::ALL).unwrap();
    let run = execute(&plan, all_rules()).unwrap();
    assert!(
        !run.cells.is_empty() && run.cells.len() < 24,
        "{:?}",
        run.cells
    );
    assert_eq!(
        support::ids(&run.gene_table, "gene_id"),
        ["G5", "G0", "G2"],
        "select_genes keeps its order"
    );
    // The computed columns have the dtypes the plan declares.
    assert_eq!(
        run.cell_table.column("g0").unwrap().dtype(),
        &DataType::Float32
    );
    assert_eq!(
        run.gene_table.column("seen").unwrap().dtype(),
        &DataType::Boolean
    );
}

/// `uint32` counts are `Float64` in arithmetic (no wrapping below zero); a plain read
/// keeps `UInt32`, a QC count of detected genes too.
#[test]
fn uint32_arithmetic_does_not_wrap() {
    let counts = || CellExpr::expression(Some("counts".to_owned()), "G0");
    let plan = scan(&fixture().path)
        .with_cells([
            (
                "below".to_owned(),
                Expr::binary(counts(), BinaryOp::Sub, Literal::Int(1_000)).unwrap(),
            ),
            ("read".to_owned(), counts()),
            (
                "detected".to_owned(),
                Expr::binary(
                    CellExpr::qc(Some("counts".to_owned()), QcMetric::GenesDetected).unwrap(),
                    BinaryOp::Sub,
                    Literal::Int(100),
                )
                .unwrap(),
            ),
            (
                "n_genes".to_owned(),
                CellExpr::qc(Some("counts".to_owned()), QcMetric::GenesDetected).unwrap(),
            ),
        ])
        .unwrap();
    let run = execute(&plan, all_rules()).unwrap();
    let column = |name: &str| run.cell_table.column(name).unwrap();
    assert_eq!(column("below").dtype(), &DataType::Float64);
    assert_eq!(column("read").dtype(), &DataType::UInt32);
    assert_eq!(column("detected").dtype(), &DataType::Float64);
    assert_eq!(column("n_genes").dtype(), &DataType::UInt32);
    let values = |name: &str| -> Vec<f64> {
        column(name)
            .cast(&DataType::Float64)
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect()
    };
    let counts = &fixture().data.layers["counts"];
    assert_eq!(
        values("read"),
        counts.iter().map(|row| row[0]).collect::<Vec<_>>()
    );
    assert_eq!(
        values("below"),
        counts
            .iter()
            .map(|row| row[0] - 1_000.0)
            .collect::<Vec<_>>()
    );
    let detected = counts
        .iter()
        .map(|row| row.iter().filter(|value| **value != 0.0).count() as f64);
    assert_eq!(values("n_genes"), detected.clone().collect::<Vec<_>>());
    assert_eq!(
        values("detected"),
        detected.map(|count| count - 100.0).collect::<Vec<_>>()
    );
}

/// `take_cells` keeps store order; an ID the store lacks fails before reading; an ID an
/// earlier filter removed fails at the stream's end, after the blocks before it.
#[test]
fn take_cells_order_and_missing_cells() {
    let base = scan(&fixture().path);
    let taken = base
        .take_cells(["c9".to_owned(), "c2".to_owned(), "c17".to_owned()])
        .unwrap();
    let run = execute(&taken, all_rules()).unwrap();
    assert_eq!(run.cells, [2, 9, 17]);

    // The plan bounds IDs by count only; the store does not have c99.
    let missing = base.take_cells(["c99".to_owned()]).unwrap();
    let error = execute(&missing, all_rules()).unwrap_err();
    assert_eq!(error.code(), "KY-E210");
    assert!(error.to_string().contains("c99"), "{error}");

    let removed = base
        .filter_cells(
            &Expr::binary(
                cell("batch"),
                BinaryOp::Eq,
                Literal::String("b0".to_owned()),
            )
            .unwrap(),
        )
        .unwrap()
        .take_cells(["c0".to_owned(), "c1".to_owned()])
        .unwrap();
    let error = execute(&removed, all_rules()).unwrap_err();
    assert_eq!(error.code(), "KY-E210");
    assert!(error.to_string().contains("removed"), "{error}");
}

/// A gene an earlier filter removed, read by `select_genes`, a cell expression or a
/// profile of a cell a filter removed: `KY-E210` naming it.
#[test]
fn removed_genes_and_cells_are_coded_errors() {
    let base = scan(&fixture().path);
    let hvg = base
        .filter_genes(&GeneExpr::column("highly_variable"))
        .unwrap();
    // G1 is not highly variable in the dataset (odd genes are false).
    let selected = hvg.select_genes(["G1".to_owned()]).unwrap();
    let error = execute(&selected, all_rules()).unwrap_err();
    assert_eq!(error.code(), "KY-E210");
    assert!(error.to_string().contains("'G1'"), "{error}");

    let read = hvg
        .with_cells([(
            "g1".to_owned(),
            CellExpr::expression(Some("counts".to_owned()), "G1"),
        )])
        .unwrap();
    let error = execute(&read, all_rules()).unwrap_err();
    assert_eq!(error.code(), "KY-E210");

    let profile = base
        .take_cells(["c2".to_owned()])
        .unwrap()
        .filter_genes(
            &Expr::binary(
                GeneExpr::profile(Some("counts".to_owned()), "c5"),
                BinaryOp::Gt,
                Literal::Int(0),
            )
            .unwrap(),
        )
        .unwrap();
    let error = execute(&profile, all_rules()).unwrap_err();
    assert_eq!(error.code(), "KY-E210");
    assert!(error.to_string().contains("c5"), "{error}");
}

/// Embedding leaves and recomputed columns wait for 5.9; sinks too.
#[test]
fn what_waits_for_5_9_is_refused_before_reading() {
    let base = scan(&fixture().path);
    let plan = base
        .sink(kyto_plan::Sink::Zarr {
            path: "out.kyto".to_owned(),
        })
        .unwrap();
    let physical = physical_plan(
        &[Output::all(&plan)],
        OptimizeOptions::ALL,
        PhysicalOptions::default(),
    );
    let error = Pipeline::new(&physical, 0).unwrap_err();
    assert_eq!(error.code(), "KY-E601");
    assert!(error.to_string().contains("5.9"), "{error}");
}

/// Polars runs a one-row column as a scalar, through another float kernel (`x / c`, not
/// `x * (1 / c)`): blocks of one row must still give the bits of larger blocks (R11;
/// found by the random plans).
#[test]
fn one_row_blocks_give_the_same_bits() {
    let ratio = Expr::binary(
        Expr::binary(cell("score"), BinaryOp::Div, Literal::Int(-2)).unwrap(),
        BinaryOp::Div,
        Literal::Float(1.432_328_992_236_826_3),
    )
    .unwrap();
    let plan = scan(&fixture().path)
        .with_cells([("ratio".to_owned(), ratio)])
        .unwrap();
    let whole = execute(&plan, all_rules()).unwrap();
    for max_block_cells in [1, 3] {
        let split = execute(
            &plan,
            How {
                max_block_cells: Some(max_block_cells),
                ..all_rules()
            },
        )
        .unwrap();
        assert_eq!(
            split.fingerprint, whole.fingerprint,
            "blocks of {max_block_cells}"
        );
    }
}

/// A step reading a gene column `highly_variable_genes` writes is refused before
/// reading (6.3 writes those columns): the stored `highly_variable` of the same name
/// would be read stale, a new one like `dispersions_norm` is not there (bug-hunter).
#[test]
fn columns_hvg_writes_wait_for_6_3() {
    let hvg = scan(&fixture().path)
        .highly_variable_genes(Some("log1p"), kyto_plan::HvgFlavor::Seurat, Some(1), None)
        .unwrap();
    for column in ["highly_variable", "dispersions_norm"] {
        let predicate = match column {
            "highly_variable" => GeneExpr::column(column),
            _ => Expr::binary(GeneExpr::column(column), BinaryOp::Gt, Literal::Float(0.0)).unwrap(),
        };
        let plan = hvg.filter_genes(&predicate).unwrap();
        let error = execute(&plan, all_rules()).unwrap_err();
        assert_eq!(error.code(), "KY-E601", "{column}");
        assert!(error.to_string().contains("6.3"), "{error}");
    }
    // A with_genes column of that name replaces it; reading it then runs.
    let replaced = hvg
        .with_genes([(
            "highly_variable".to_owned(),
            Expr::binary(GeneExpr::column("mean"), BinaryOp::Gt, Literal::Float(1.0)).unwrap(),
        )])
        .unwrap()
        .filter_genes(&GeneExpr::column("highly_variable"))
        .unwrap();
    execute(&replaced, all_rules()).unwrap();
}
