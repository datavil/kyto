//! Task 4.4 Done when: each rule has before/after plan snapshot tests, and optimized
//! results equal unoptimized results on fixtures.
//!
//! One section per rule: the plan with every rule off (before) and with only that rule
//! on (after), as text, then the results of both compared by the reference evaluator
//! (`support/`: no executor before 5.x) on the seeded dataset with the synthetic
//! fixture's edge cases. Then the rules together, and random chains and outputs
//! (proptest) with random rules on, each compared the same way.
#![allow(clippy::unwrap_used)]

mod support;

use std::sync::Arc;

use kyto_core::{BinaryOp, CellExpr, Expr, GeneExpr, Literal};
use kyto_plan::{
    Cluster, ClusterMethod, HvgFlavor, Member, Metric, NeighborSearch, Neighbors, OptimizeOptions,
    OptimizedPlan, Output, Plan, Sink, all_members, optimize,
};
use proptest::prelude::*;

use support::{check_equal, scan};

fn cell(name: &str) -> CellExpr {
    CellExpr::column(name)
}

fn gt(expr: CellExpr, value: f64) -> CellExpr {
    Expr::binary(expr, BinaryOp::Gt, Literal::Float(value)).unwrap()
}

fn eq(expr: CellExpr, value: &str) -> CellExpr {
    Expr::binary(expr, BinaryOp::Eq, Literal::String(value.to_owned())).unwrap()
}

fn neighbors() -> Neighbors {
    Neighbors {
        embedding: "X_pca".to_owned(),
        n_components: None,
        n_neighbors: 15,
        metric: Metric::Euclidean,
        search: NeighborSearch::Exact,
        seed: 0,
        key: None,
    }
}

fn leiden(into: &str) -> Cluster {
    Cluster {
        method: ClusterMethod::Leiden,
        graph: "connectivities".to_owned(),
        resolution: 1.0,
        seed: 0,
        into: into.to_owned(),
    }
}

fn layer(name: &str) -> Member {
    Member::Layer(name.to_owned())
}

/// Only `rule` on.
fn only(rule: fn(&mut OptimizeOptions)) -> OptimizeOptions {
    let mut options = OptimizeOptions::NONE;
    rule(&mut options);
    options
}

/// The plan of `outputs` with `options`, checked to give the logical plan's results.
fn optimized(outputs: &[Output], options: OptimizeOptions) -> OptimizedPlan {
    let plan = optimize(outputs, options);
    check_equal(outputs, &plan, options).unwrap();
    plan
}

fn assert_snapshot(plan: &OptimizedPlan, expected: &str) {
    if expected.trim().is_empty() {
        println!("SNAPSHOT>>>\n{plan}\n<<<SNAPSHOT");
        return;
    }
    assert_eq!(plan.to_string(), expected.trim(), "\n{plan}");
}

fn predicate_pushdown() -> OptimizeOptions {
    only(|options| options.predicate_pushdown = true)
}

fn projection_pushdown() -> OptimizeOptions {
    only(|options| options.projection_pushdown = true)
}

fn fusion() -> OptimizeOptions {
    only(|options| options.fuse_elementwise = true)
}

fn reuse() -> OptimizeOptions {
    only(|options| options.reuse_sub_plans = true)
}

/// Every member of the store, as a scan with projection pushdown off reads it.
const ALL: &str = "[layer:counts, layer:log1p, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]";

// --- Predicate pushdown -----------------------------------------------------------

#[test]
fn predicate_pushdown_moves_a_cell_filter_below_per_cell_nodes() {
    let plan = scan()
        .log1p(Some("counts"), "logc")
        .unwrap()
        .with_cells([(
            "double".to_owned(),
            Expr::binary(cell("score"), BinaryOp::Mul, Literal::Int(2)).unwrap(),
        )])
        .unwrap()
        .filter_cells(&eq(cell("batch"), "b1"))
        .unwrap();
    let outputs = [Output::all(&plan)];
    assert_snapshot(
        &optimized(&outputs, OptimizeOptions::NONE),
        &format!(
            r#"
output #3 reads [layer:counts, layer:log1p, layer:logc, cells:batch, cells:double, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
#3 FILTER_CELLS cell_column('batch') == 'b1' <- #2 (node 3)
#2 WITH_CELLS "double" = cell_column('score') * 2 <- #1 (node 2)
#1 LOG1P "counts" -> "logc" <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
    assert_snapshot(
        &optimized(&outputs, predicate_pushdown()),
        &format!(
            r#"
output #3 reads [layer:counts, layer:log1p, layer:logc, cells:batch, cells:double, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
#3 WITH_CELLS "double" = cell_column('score') * 2 <- #2 (node 2)
#2 LOG1P "counts" -> "logc" <- #1 (node 1)
#1 FILTER_CELLS cell_column('batch') == 'b1' <- #0 (node 3)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
}

#[test]
fn predicate_pushdown_stops_at_cross_cell_nodes() {
    // `normalize_total` to the median total, `scale`, `pca`, `neighbors` and a gene
    // filter on a cell's profile all need every cell first.
    let median = scan()
        .normalize_total(None, Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "logn")
        .unwrap()
        .filter_cells(&gt(cell("score"), 0.0))
        .unwrap();
    assert_snapshot(
        &optimized(&[Output::all(&median)], predicate_pushdown()),
        &format!(
            r#"
output #3 reads [layer:counts, layer:log1p, layer:logn, layer:norm, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
#3 LOG1P "norm" -> "logn" <- #2 (node 2)
#2 FILTER_CELLS cell_column('score') > 0.0 <- #1 (node 3)
#1 NORMALIZE_TOTAL "counts" -> "norm" target_sum=median <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
    let pca = scan()
        .pca(Some("log1p"), "X_new", 2, 0)
        .unwrap()
        .filter_cells(&gt(cell("score"), 0.0))
        .unwrap();
    assert_snapshot(
        &optimized(&[Output::all(&pca)], predicate_pushdown()),
        &format!(
            r#"
output #2 reads [layer:counts, layer:log1p, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_new, embedding:X_new_loadings, embedding:X_pca, graph:connectivities]
#2 FILTER_CELLS cell_column('score') > 0.0 <- #1 (node 2)
#1 PCA "log1p" -> "X_new" n_components=2 seed=0 <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
    let profile = Expr::binary(
        GeneExpr::profile(Some("counts".to_owned()), "c1"),
        BinaryOp::Gt,
        Literal::Float(0.0),
    )
    .unwrap();
    let gene_filter = scan()
        .filter_genes(&profile)
        .unwrap()
        .filter_cells(&gt(cell("score"), 0.0))
        .unwrap();
    assert_snapshot(
        &optimized(&[Output::all(&gene_filter)], predicate_pushdown()),
        &format!(
            r#"
output #2 reads {ALL}
#2 FILTER_CELLS cell_column('score') > 0.0 <- #1 (node 2)
#1 FILTER_GENES cell_profile('c1', layer='counts') > 0.0 <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
}

#[test]
fn predicate_pushdown_stops_where_the_predicate_reads_what_a_node_writes() {
    let plan = scan()
        .with_default_layer("counts")
        .unwrap()
        .log1p(None, "logc")
        .unwrap()
        .filter_cells(&gt(
            CellExpr::expression(Some("logc".to_owned()), "G1"),
            0.5,
        ))
        .unwrap();
    assert_snapshot(
        &optimized(&[Output::all(&plan)], predicate_pushdown()),
        &format!(
            r#"
output #3 reads [layer:counts, layer:log1p, layer:logc, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
#3 FILTER_CELLS gene_expression('G1', layer='logc') > 0.5 <- #2 (node 3)
#2 LOG1P "counts" -> "logc" <- #1 (node 2)
#1 WITH_DEFAULT_LAYER "counts" <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
}

#[test]
fn predicate_pushdown_keeps_cell_order_and_gene_dependent_predicates() {
    // Below a `take_cells` the filter would drop a taken cell before it is taken.
    let taken = scan()
        .take_cells(["c5", "c1", "c4"].map(str::to_owned))
        .unwrap()
        .log1p(Some("counts"), "logc")
        .unwrap()
        .filter_cells(&gt(cell("score"), 0.0))
        .unwrap();
    assert_snapshot(
        &optimized(&[Output::all(&taken)], predicate_pushdown()),
        &format!(
            r#"
output #3 reads [layer:counts, layer:log1p, layer:logc, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
#3 LOG1P "counts" -> "logc" <- #2 (node 2)
#2 FILTER_CELLS cell_column('score') > 0.0 <- #1 (node 3)
#1 TAKE_CELLS ['c5', 'c1', 'c4'] <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
    // A recomputed total sums over the genes present, so it stays above a selection;
    // a gene's values stay above a gene filter, which may remove that gene.
    let recomputed = scan()
        .select_genes(["G0", "G1", "G5"].map(str::to_owned))
        .unwrap()
        .log1p(Some("counts"), "logc")
        .unwrap()
        .filter_cells(&gt(cell("total_counts").recompute().unwrap(), 4.0))
        .unwrap();
    assert_snapshot(
        &optimized(&[Output::all(&recomputed)], predicate_pushdown()),
        &format!(
            r#"
output #3 reads [layer:counts, layer:log1p, layer:logc, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
#3 LOG1P "counts" -> "logc" <- #2 (node 2)
#2 FILTER_CELLS cell_column('total_counts').recompute() > 4.0 <- #1 (node 3)
#1 SELECT_GENES ['G0', 'G1', 'G5'] <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
    const ALL_LOGC: &str = "[layer:counts, layer:log1p, layer:logc, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]";
    let gene_read = scan()
        .filter_genes(&GeneExpr::column("highly_variable"))
        .unwrap()
        .log1p(Some("counts"), "logc")
        .unwrap()
        .filter_cells(&gt(
            CellExpr::expression(Some("counts".to_owned()), "G1"),
            0.0,
        ))
        .unwrap();
    assert_snapshot(
        &optimized(&[Output::all(&gene_read)], predicate_pushdown()),
        &format!(
            r#"
output #3 reads {ALL_LOGC}
#3 LOG1P "counts" -> "logc" <- #2 (node 2)
#2 FILTER_CELLS gene_expression('G1', layer='counts') > 0.0 <- #1 (node 3)
#1 FILTER_GENES gene_column('highly_variable') <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
}

// --- Projection pushdown ----------------------------------------------------------

#[test]
fn projection_pushdown_reads_only_what_the_output_needs() {
    let plan = scan()
        .log1p(Some("counts"), "logc")
        .unwrap()
        .pca(Some("logc"), "X_new", 2, 0)
        .unwrap()
        .with_cells([(
            "double".to_owned(),
            Expr::binary(cell("score"), BinaryOp::Mul, Literal::Int(2)).unwrap(),
        )])
        .unwrap();
    let outputs = [Output::members(&plan, [Member::CellColumn("double".to_owned())]).unwrap()];
    assert_snapshot(
        &optimized(&outputs, OptimizeOptions::NONE),
        &format!(
            r#"
output #3 reads [cells:double]
#3 WITH_CELLS "double" = cell_column('score') * 2 <- #2 (node 3)
#2 PCA "logc" -> "X_new" n_components=2 seed=0 <- #1 (node 2)
#1 LOG1P "counts" -> "logc" <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
    assert_snapshot(
        &optimized(&outputs, projection_pushdown()),
        r#"
output #1 reads [cells:double]
#1 WITH_CELLS "double" = cell_column('score') * 2 <- #0 (node 3)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads [cells:score] (node 0)
"#,
    );
}

#[test]
fn projection_pushdown_keeps_what_a_sink_writes_and_every_filter() {
    let plan = scan()
        .log1p(Some("counts"), "logc")
        .unwrap()
        .filter_cells(&gt(CellExpr::embedding("X_pca", 0), 0.0))
        .unwrap()
        .sink(Sink::Zarr {
            path: "out.kyto".to_owned(),
        })
        .unwrap();
    assert_snapshot(
        &optimized(&[Output::all(&plan)], projection_pushdown()),
        &format!(
            r#"
output #3 reads [layer:counts, layer:log1p, layer:logc, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
#3 SINK_ZARR "out.kyto" <- #2 (node 3)
#2 FILTER_CELLS cell_embedding('X_pca', 0) > 0.0 <- #1 (node 2)
#1 LOG1P "counts" -> "logc" <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
    // A layer written twice: only the second write is read.
    let twice = scan()
        .log1p(Some("counts"), "x")
        .unwrap()
        .filter_cells(&gt(cell("score"), 0.0))
        .unwrap()
        .normalize_total(Some(10.0), Some("log1p"), "x")
        .unwrap();
    let outputs = [Output::members(&twice, [layer("x")]).unwrap()];
    assert_snapshot(
        &optimized(&outputs, projection_pushdown()),
        r#"
output #2 reads [layer:x]
#2 NORMALIZE_TOTAL "log1p" -> "x" target_sum=10 <- #1 (node 3)
#1 FILTER_CELLS cell_column('score') > 0.0 <- #0 (node 2)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads [layer:log1p, cells:score] (node 0)
"#,
    );
}

#[test]
fn outputs_refuse_members_the_plan_does_not_have() {
    let error = Output::members(&scan(), [layer("logc")]).unwrap_err();
    assert_eq!(error.code(), "KY-E109", "{error}");
    assert!(error.to_string().contains("logc"), "{error}");
}

// --- Elementwise fusion -----------------------------------------------------------

#[test]
fn fusion_runs_normalize_total_then_log1p_in_one_pass() {
    let plan = scan()
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "logn")
        .unwrap();
    let outputs = [Output::all(&plan)];
    assert_snapshot(
        &optimized(&outputs, OptimizeOptions::NONE),
        &format!(
            r#"
output #2 reads [layer:counts, layer:log1p, layer:logn, layer:norm, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
#2 LOG1P "norm" -> "logn" <- #1 (node 2)
#1 NORMALIZE_TOTAL "counts" -> "norm" target_sum=10000 <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
    assert_snapshot(
        &optimized(&outputs, fusion()),
        &format!(
            r#"
output #1 reads [layer:counts, layer:log1p, layer:logn, layer:norm, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
#1 FUSED_ELEMENTWISE [NORMALIZE_TOTAL "counts" -> "norm" target_sum=10000; LOG1P "norm" -> "logn"] keeps ["logn", "norm"] <- #0 (node 1, 2)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
    // With projection pushdown the intermediate layer stays inside the pass.
    let outputs = [Output::members(&plan, [layer("logn")]).unwrap()];
    let options = OptimizeOptions {
        fuse_elementwise: true,
        projection_pushdown: true,
        ..OptimizeOptions::NONE
    };
    assert_snapshot(
        &optimized(&outputs, options),
        r#"
output #1 reads [layer:logn]
#1 FUSED_ELEMENTWISE [NORMALIZE_TOTAL "counts" -> "norm" target_sum=10000; LOG1P "norm" -> "logn"] keeps ["logn"] <- #0 (node 1, 2)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads [layer:counts] (node 0)
"#,
    );
}

#[test]
fn fusion_lets_a_median_normalize_total_start_a_pass_only() {
    let median_first = scan()
        .normalize_total(None, Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "logn")
        .unwrap();
    assert_snapshot(
        &optimized(&[Output::all(&median_first)], fusion()),
        &format!(
            r#"
output #1 reads [layer:counts, layer:log1p, layer:logn, layer:norm, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
#1 FUSED_ELEMENTWISE [NORMALIZE_TOTAL "counts" -> "norm" target_sum=median; LOG1P "norm" -> "logn"] keeps ["logn", "norm"] <- #0 (node 1, 2)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
    let median_second = scan()
        .log1p(Some("counts"), "logc")
        .unwrap()
        .normalize_total(None, Some("logc"), "norm")
        .unwrap();
    assert_snapshot(
        &optimized(&[Output::all(&median_second)], fusion()),
        &format!(
            r#"
output #2 reads [layer:counts, layer:log1p, layer:logc, layer:norm, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
#2 NORMALIZE_TOTAL "logc" -> "norm" target_sum=median <- #1 (node 2)
#1 LOG1P "counts" -> "logc" <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
}

#[test]
fn fusion_keeps_a_map_another_output_reads() {
    let norm = scan()
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap();
    let logn = norm.log1p(Some("norm"), "logn").unwrap();
    let outputs = [Output::all(&norm), Output::all(&logn)];
    let options = OptimizeOptions {
        fuse_elementwise: true,
        reuse_sub_plans: true,
        ..OptimizeOptions::NONE
    };
    assert_snapshot(
        &optimized(&outputs, options),
        &format!(
            r#"
output #1 reads [layer:counts, layer:log1p, layer:norm, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
output #2 reads [layer:counts, layer:log1p, layer:logn, layer:norm, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
#2 LOG1P "norm" -> "logn" <- #1 (node 2)
#1 NORMALIZE_TOTAL "counts" -> "norm" target_sum=10000 <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
}

// --- Common sub-plan reuse --------------------------------------------------------

#[test]
fn reuse_computes_a_shared_sub_plan_once() {
    // Built twice from separate scans: equal parameters, not the same `Arc`.
    let left = scan()
        .log1p(Some("counts"), "logc")
        .unwrap()
        .pca(Some("logc"), "X_new", 2, 7)
        .unwrap();
    let right = scan()
        .log1p(Some("counts"), "logc")
        .unwrap()
        .highly_variable_genes(Some("logc"), HvgFlavor::Seurat, Some(2), Some("batch"))
        .unwrap();
    let outputs = [Output::all(&left), Output::all(&right)];
    assert_snapshot(
        &optimized(&outputs, OptimizeOptions::NONE),
        &format!(
            r#"
output #2 reads [layer:counts, layer:log1p, layer:logc, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_new, embedding:X_new_loadings, embedding:X_pca, graph:connectivities]
output #5 reads [layer:counts, layer:log1p, layer:logc, cells:batch, cells:score, cells:total_counts, genes:dispersions, genes:dispersions_norm, genes:highly_variable, genes:highly_variable_intersection, genes:highly_variable_nbatches, genes:mean, genes:means, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
#5 HIGHLY_VARIABLE_GENES "logc" flavor=seurat n_top_genes=2 batch_key="batch" <- #4 (node 2)
#4 LOG1P "counts" -> "logc" <- #3 (node 1)
#3 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
#2 PCA "logc" -> "X_new" n_components=2 seed=7 <- #1 (node 2)
#1 LOG1P "counts" -> "logc" <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
    assert_snapshot(
        &optimized(&outputs, reuse()),
        &format!(
            r#"
output #2 reads [layer:counts, layer:log1p, layer:logc, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_new, embedding:X_new_loadings, embedding:X_pca, graph:connectivities]
output #3 reads [layer:counts, layer:log1p, layer:logc, cells:batch, cells:score, cells:total_counts, genes:dispersions, genes:dispersions_norm, genes:highly_variable, genes:highly_variable_intersection, genes:highly_variable_nbatches, genes:mean, genes:means, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
#3 HIGHLY_VARIABLE_GENES "logc" flavor=seurat n_top_genes=2 batch_key="batch" <- #1 (node 2)
#2 PCA "logc" -> "X_new" n_components=2 seed=7 <- #1 (node 2)
#1 LOG1P "counts" -> "logc" <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
}

#[test]
fn reuse_does_not_merge_different_parameters_or_inputs() {
    let left = scan().log1p(Some("counts"), "logc").unwrap();
    let right = scan().log1p(Some("log1p"), "logc").unwrap();
    let filtered = scan()
        .filter_cells(&gt(cell("score"), 0.0))
        .unwrap()
        .log1p(Some("counts"), "logc")
        .unwrap();
    let outputs = [
        Output::all(&left),
        Output::all(&right),
        Output::all(&filtered),
    ];
    assert_snapshot(
        &optimized(&outputs, reuse()),
        &format!(
            r#"
output #1 reads [layer:counts, layer:log1p, layer:logc, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
output #2 reads [layer:counts, layer:log1p, layer:logc, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
output #4 reads [layer:counts, layer:log1p, layer:logc, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities]
#4 LOG1P "counts" -> "logc" <- #3 (node 2)
#3 FILTER_CELLS cell_column('score') > 0.0 <- #0 (node 1)
#2 LOG1P "log1p" -> "logc" <- #0 (node 1)
#1 LOG1P "counts" -> "logc" <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
}

// --- All rules --------------------------------------------------------------------

#[test]
fn all_rules_on_a_standard_pipeline() {
    let normalized = scan()
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "logn")
        .unwrap()
        .filter_cells(
            &Expr::binary(
                gt(cell("score"), -2.0),
                BinaryOp::And,
                eq(cell("batch"), "b0"),
            )
            .unwrap(),
        )
        .unwrap();
    let clustered = normalized
        .highly_variable_genes(Some("logn"), HvgFlavor::CellRanger, Some(3), None)
        .unwrap()
        .scale(Some("logn"), "scaled", true, Some(10.0))
        .unwrap()
        .pca(Some("scaled"), "X_new", 2, 0)
        .unwrap()
        .neighbors(Neighbors {
            embedding: "X_new".to_owned(),
            ..neighbors()
        })
        .unwrap()
        .cluster(leiden("leiden"))
        .unwrap();
    let outputs = [
        Output::members(&normalized, [layer("logn")]).unwrap(),
        Output::members(
            &clustered,
            [
                Member::CellColumn("leiden".to_owned()),
                Member::GeneColumn("highly_variable".to_owned()),
            ],
        )
        .unwrap(),
    ];
    assert_snapshot(
        &optimized(&outputs, OptimizeOptions::NONE),
        &format!(
            r#"
output #3 reads [layer:logn]
output #12 reads [cells:leiden, genes:highly_variable]
#12 LEIDEN "connectivities" -> "leiden" resolution=1 seed=0 <- #11 (node 8)
#11 NEIGHBORS "X_new" -> "distances", "connectivities" n_neighbors=15 metric=Euclidean search=Exact seed=0 <- #10 (node 7)
#10 PCA "scaled" -> "X_new" n_components=2 seed=0 <- #9 (node 6)
#9 SCALE "logn" -> "scaled" zero_center=true max_value=10 <- #8 (node 5)
#8 HIGHLY_VARIABLE_GENES "logn" flavor=cell_ranger n_top_genes=3 <- #7 (node 4)
#7 FILTER_CELLS (cell_column('score') > -2.0) & (cell_column('batch') == 'b0') <- #6 (node 3)
#6 LOG1P "norm" -> "logn" <- #5 (node 2)
#5 NORMALIZE_TOTAL "counts" -> "norm" target_sum=10000 <- #4 (node 1)
#4 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
#3 FILTER_CELLS (cell_column('score') > -2.0) & (cell_column('batch') == 'b0') <- #2 (node 3)
#2 LOG1P "norm" -> "logn" <- #1 (node 2)
#1 NORMALIZE_TOTAL "counts" -> "norm" target_sum=10000 <- #0 (node 1)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads {ALL} (node 0)
"#
        ),
    );
    assert_snapshot(
        &optimized(&outputs, OptimizeOptions::ALL),
        r#"
output #2 reads [layer:logn]
output #7 reads [cells:leiden, genes:highly_variable]
#7 LEIDEN "connectivities" -> "leiden" resolution=1 seed=0 <- #6 (node 8)
#6 NEIGHBORS "X_new" -> "distances", "connectivities" n_neighbors=15 metric=Euclidean search=Exact seed=0 <- #5 (node 7)
#5 PCA "scaled" -> "X_new" n_components=2 seed=0 <- #4 (node 6)
#4 SCALE "logn" -> "scaled" zero_center=true max_value=10 <- #3 (node 5)
#3 HIGHLY_VARIABLE_GENES "logn" flavor=cell_ranger n_top_genes=3 <- #2 (node 4)
#2 FUSED_ELEMENTWISE [NORMALIZE_TOTAL "counts" -> "norm" target_sum=10000; LOG1P "norm" -> "logn"] keeps ["logn"] <- #1 (node 1, 2)
#1 FILTER_CELLS (cell_column('score') > -2.0) & (cell_column('batch') == 'b0') <- #0 (node 3)
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads [layer:counts, cells:batch, cells:score] (node 0)
"#,
    );
}

#[test]
fn plan_optimize_runs_every_rule_and_leaves_the_plan_unchanged() {
    let plan = scan()
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "logn")
        .unwrap()
        .filter_cells(&gt(cell("score"), 0.0))
        .unwrap();
    let before = plan.to_string();
    let optimized = plan.optimize();
    assert_eq!(plan.to_string(), before);
    assert_eq!(
        optimized.to_string(),
        optimize(&[Output::all(&plan)], OptimizeOptions::ALL).to_string()
    );
    check_equal(&[Output::all(&plan)], &optimized, OptimizeOptions::ALL).unwrap();
}

#[test]
fn long_chains_of_distinct_layers_optimize_in_linear_time() {
    // Bug-hunter: projection pushdown rebuilt the needed set at every node, quadratic
    // in chain length when every map writes a new layer (16 s for 4000 nodes).
    let mut plan: Arc<Plan> = scan();
    for step in 0..4000 {
        plan = if step % 2 == 0 {
            plan.log1p(Some("counts"), &format!("l{step}")).unwrap()
        } else {
            plan.filter_cells(&gt(cell("score"), -5.0)).unwrap()
        };
    }
    let start = std::time::Instant::now();
    let optimized = plan.optimize();
    assert!(start.elapsed().as_secs_f64() < 5.0, "{:?}", start.elapsed());
    assert_eq!(optimized.nodes().len(), 2002);
}

#[test]
fn long_chains_optimize_without_deep_recursion() {
    let mut plan: Arc<Plan> = scan();
    for step in 0..2000 {
        plan = if step % 2 == 0 {
            plan.log1p(Some(if step == 0 { "counts" } else { "logc" }), "logc")
                .unwrap()
        } else {
            plan.filter_cells(&gt(cell("score"), -5.0)).unwrap()
        };
    }
    let optimized = plan.optimize();
    // Every filter moves to the scan; the 1000 log1p maps fuse into one pass.
    assert_eq!(optimized.nodes().len(), 1002, "{}", optimized.nodes().len());
    assert_eq!(optimized.nodes()[1001].name(), "FUSED_ELEMENTWISE");
    check_equal(&[Output::all(&plan)], &optimized, OptimizeOptions::ALL).unwrap();
}

// --- Random plans -----------------------------------------------------------------

/// A random cell expression over the dataset's names (may be ill-typed or stale: the
/// builder then refuses the step, and the chain goes on without it).
fn cell_expr() -> impl Strategy<Value = CellExpr> {
    let leaf = prop_oneof![
        prop::sample::select(vec!["score", "batch", "total_counts", "double", "leiden"])
            .prop_map(cell),
        Just(cell("total_counts").recompute().unwrap()),
        (
            prop::sample::select(vec!["G0", "G1", "G2", "G5"]),
            prop::sample::select(vec![None, Some("counts"), Some("a"), Some("b")])
        )
            .prop_map(|(gene_id, layer)| CellExpr::expression(layer.map(str::to_owned), gene_id)),
        (0..3_u64).prop_map(|component| CellExpr::embedding("X_pca", component)),
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
        prop::sample::select(vec!["c0", "c1", "c7"]).prop_map(|cell_id| {
            Expr::binary(
                GeneExpr::profile(Some("counts".to_owned()), cell_id),
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

/// One step of a random chain.
#[derive(Debug, Clone)]
enum Step {
    DefaultLayer(&'static str),
    FilterCells(CellExpr),
    FilterGenes(GeneExpr),
    SelectGenes(Vec<String>),
    TakeCells(Vec<String>),
    NormalizeTotal(Option<f64>, &'static str, &'static str),
    Log1p(&'static str, &'static str),
    Scale(bool, Option<f64>, &'static str, &'static str),
    Hvg(HvgFlavor, Option<u64>, bool),
    Pca(u64, &'static str),
    WithCells(&'static str, CellExpr),
    WithGenes(&'static str, GeneExpr),
    Neighbors(&'static str, Option<u64>),
    Cluster(&'static str, &'static str),
    Sink,
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
        1 => gene_expr().prop_map(Step::FilterGenes),
        1 => ids(vec!["G0", "G1", "G2", "G3", "G5"]).prop_map(Step::SelectGenes),
        1 => ids(vec!["c0", "c2", "c5", "c9", "c3"]).prop_map(Step::TakeCells),
        3 => (proptest::option::of(1.0..1e4_f64), layers(), layers())
            .prop_map(|(sum, layer, into)| Step::NormalizeTotal(sum, layer, into)),
        3 => (layers(), layers()).prop_map(|(layer, into)| Step::Log1p(layer, into)),
        1 => (any::<bool>(), proptest::option::of(0.5..3.0_f64), layers(), layers())
            .prop_map(|(center, max, layer, into)| Step::Scale(center, max, layer, into)),
        1 => (
            prop::sample::select(vec![HvgFlavor::Seurat, HvgFlavor::CellRanger, HvgFlavor::SeuratV3]),
            proptest::option::of(1..5_u64),
            any::<bool>()
        )
            .prop_map(|(flavor, n, batch)| Step::Hvg(flavor, n, batch)),
        1 => (1..4_u64, prop::sample::select(vec!["X_pca", "X_new"])).prop_map(|(n, into)| Step::Pca(n, into)),
        2 => (prop::sample::select(vec!["double", "score"]), cell_expr()).prop_map(|(name, expr)| Step::WithCells(name, expr)),
        1 => (prop::sample::select(vec!["flag", "mean"]), gene_expr()).prop_map(|(name, expr)| Step::WithGenes(name, expr)),
        1 => (prop::sample::select(vec!["X_pca", "X_new"]), proptest::option::of(1..3_u64))
            .prop_map(|(embedding, n)| Step::Neighbors(embedding, n)),
        1 => (prop::sample::select(vec!["connectivities", "distances"]), prop::sample::select(vec!["leiden", "batch"]))
            .prop_map(|(graph, into)| Step::Cluster(graph, into)),
        1 => Just(Step::Sink),
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
        Step::Scale(center, max, layer, into) => plan.scale(Some(layer), into, center, max),
        Step::Hvg(flavor, n, batch) => {
            plan.highly_variable_genes(Some("counts"), flavor, n, batch.then_some("batch"))
        }
        Step::Pca(n, into) => plan.pca(Some("counts"), into, n, 3),
        Step::WithCells(name, expr) => plan.with_cells([(name.to_owned(), expr)]),
        Step::WithGenes(name, expr) => plan.with_genes([(name.to_owned(), expr)]),
        Step::Neighbors(embedding, n_components) => plan.neighbors(Neighbors {
            embedding: embedding.to_owned(),
            n_components,
            ..neighbors()
        }),
        Step::Cluster(graph, into) => plan.cluster(Cluster {
            graph: graph.to_owned(),
            ..leiden(into)
        }),
        Step::Sink => plan.sink(Sink::H5ad {
            path: "out.h5ad".to_owned(),
        }),
    }
    .ok()
}

fn chain(start: &Arc<Plan>, steps: Vec<Step>) -> Arc<Plan> {
    steps.into_iter().fold(Arc::clone(start), |plan, step| {
        apply(&plan, step).unwrap_or(plan)
    })
}

/// All of `plan`'s output, or the members `mask` picks.
fn output(plan: &Arc<Plan>, mask: Option<u64>) -> Output {
    match mask {
        None => Output::all(plan),
        Some(mask) => {
            let members = all_members(plan.schema())
                .into_iter()
                .enumerate()
                .filter(|(position, _)| mask >> (position % 64) & 1 == 1)
                .map(|(_, member)| member);
            Output::members(plan, members).unwrap()
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    /// Two outputs sharing a prefix (the second built on the first, or rebuilt from a
    /// fresh scan), random rules on: every output sees what the logical plan gives.
    #[test]
    fn random_plans_give_the_logical_results(
        prefix in proptest::collection::vec(step(), 0..10),
        suffix in proptest::collection::vec(step(), 0..6),
        rebuilt in any::<bool>(),
        masks in (proptest::option::of(any::<u64>()), proptest::option::of(any::<u64>())),
        rules in (any::<bool>(), any::<bool>(), any::<bool>(), any::<bool>()),
    ) {
        let first = chain(&scan(), prefix.clone());
        let base = if rebuilt { chain(&scan(), prefix) } else { Arc::clone(&first) };
        let second = chain(&base, suffix);
        let outputs = [output(&first, masks.0), output(&second, masks.1)];
        let options = OptimizeOptions {
            reuse_sub_plans: rules.0,
            projection_pushdown: rules.1,
            fuse_elementwise: rules.2,
            predicate_pushdown: rules.3,
        };
        let plan = optimize(&outputs, options);
        if let Err(message) = check_equal(&outputs, &plan, options) {
            panic!("{message}\nlogical:\n{}\n--\n{}", outputs[0].plan(), outputs[1].plan());
        }
    }
}
