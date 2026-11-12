//! Task 4.5 Done when: `explain()` shows device and pass count for every node.
//!
//! One plan per node kind and operation variant (every rule off, so no node is left
//! out or fused): each node's line in the physical plan names its device and passes,
//! with the passes its rule gives. Then snapshots: a standard pipeline with every
//! rule off and on (the fused pass), `Plan::physical` (the optimized section of
//! `explain()`, task 4.7: `tests/explain.rs`), and outputs sharing a sub-plan.
#![allow(clippy::unwrap_used)]

mod support;

use std::sync::Arc;

use kyto_core::{BinaryOp, CellExpr, Expr, GeneExpr, Literal};
use kyto_plan::{
    Blocks, Cluster, ClusterMethod, Device, HvgFlavor, Member, Metric, NeighborSearch, Neighbors,
    OptimizeOptions, Output, PCA_POWER_ITERATIONS, PhysicalOptions, PhysicalPlan, Plan, Sink,
    physical_plan,
};

use support::scan;

fn gt_cell(expr: CellExpr, value: f64) -> CellExpr {
    Expr::binary(expr, BinaryOp::Gt, Literal::Float(value)).unwrap()
}

fn gt_gene(expr: GeneExpr, value: f64) -> GeneExpr {
    Expr::binary(expr, BinaryOp::Gt, Literal::Float(value)).unwrap()
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

fn leiden() -> Cluster {
    Cluster {
        method: ClusterMethod::Leiden,
        graph: "connectivities".to_owned(),
        resolution: 1.0,
        seed: 0,
        into: "leiden".to_owned(),
    }
}

/// The physical plan of `plan` with every optimizer rule off: one node per logical node.
fn unoptimized(plan: &Arc<Plan>) -> PhysicalPlan {
    physical_plan(
        &[Output::all(plan)],
        OptimizeOptions::NONE,
        PhysicalOptions::default(),
    )
}

/// The device, passes (statistics, apply) and blocks of the last node of `plan`.
fn last(plan: &Arc<Plan>) -> (Device, u32, u32, bool, Blocks) {
    let physical = unoptimized(plan);
    let top = physical.optimized().outputs()[0].node();
    let execution = physical.executions()[top];
    (
        execution.device(),
        execution.passes(),
        execution.statistics_passes(),
        execution.apply_pass(),
        execution.blocks(),
    )
}

const C: Blocks = Blocks::Cells(4096);

/// One plan per node kind and variant, with the passes its last node must show.
fn cases() -> Vec<(&'static str, Arc<Plan>, u32, u32, Blocks)> {
    let base = scan();
    let pca = base.pca(Some("log1p"), "X_new", 2, 0).unwrap();
    let graphs = base.neighbors(neighbors()).unwrap();
    let hvg = |flavor| {
        base.highly_variable_genes(Some("log1p"), flavor, Some(3), None)
            .unwrap()
    };
    vec![
        ("scan", Arc::clone(&base), 1, 0, C),
        (
            "with_default_layer",
            base.with_default_layer("log1p").unwrap(),
            0,
            0,
            Blocks::None,
        ),
        (
            "filter_cells on metadata",
            base.filter_cells(&gt_cell(CellExpr::column("score"), 0.0))
                .unwrap(),
            1,
            0,
            C,
        ),
        (
            "filter_cells on data",
            base.filter_cells(&gt_cell(
                CellExpr::expression(Some("counts".to_owned()), "G1"),
                0.0,
            ))
            .unwrap(),
            1,
            0,
            C,
        ),
        (
            "filter_genes on metadata",
            base.filter_genes(&gt_gene(GeneExpr::column("mean"), 0.0))
                .unwrap(),
            1,
            0,
            C,
        ),
        (
            "filter_genes on data",
            base.filter_genes(&gt_gene(
                GeneExpr::profile(Some("counts".to_owned()), "c1"),
                0.0,
            ))
            .unwrap(),
            2,
            1,
            C,
        ),
        (
            "select_genes",
            base.select_genes(["G3".to_owned(), "G1".to_owned()])
                .unwrap(),
            1,
            0,
            C,
        ),
        (
            "take_cells",
            base.take_cells(["c2".to_owned(), "c1".to_owned()]).unwrap(),
            1,
            0,
            C,
        ),
        (
            "normalize_total with target_sum",
            base.normalize_total(Some(1e4), Some("counts"), "norm")
                .unwrap(),
            1,
            0,
            C,
        ),
        (
            "normalize_total median",
            base.normalize_total(None, Some("counts"), "norm").unwrap(),
            2,
            1,
            C,
        ),
        (
            "log1p",
            base.log1p(Some("counts"), "logn").unwrap(),
            1,
            0,
            C,
        ),
        (
            "scale",
            base.scale(Some("log1p"), "scaled", true, None).unwrap(),
            2,
            1,
            C,
        ),
        ("hvg seurat", hvg(HvgFlavor::Seurat), 1, 1, C),
        ("hvg cell_ranger", hvg(HvgFlavor::CellRanger), 1, 1, C),
        ("hvg seurat_v3", hvg(HvgFlavor::SeuratV3), 2, 2, C),
        (
            "hvg seurat_v3 with batch_key",
            base.highly_variable_genes(Some("counts"), HvgFlavor::SeuratV3, Some(3), Some("batch"))
                .unwrap(),
            2,
            2,
            C,
        ),
        (
            "pca",
            Arc::clone(&pca),
            2 + PCA_POWER_ITERATIONS,
            2 + PCA_POWER_ITERATIONS,
            C,
        ),
        (
            "with_cells on metadata",
            base.with_cells([("twice".to_owned(), CellExpr::column("score"))])
                .unwrap(),
            0,
            0,
            Blocks::InMemory,
        ),
        (
            "with_cells on data",
            base.with_cells([(
                "g1".to_owned(),
                CellExpr::expression(Some("counts".to_owned()), "G1"),
            )])
            .unwrap(),
            1,
            0,
            C,
        ),
        (
            "with_genes on metadata",
            base.with_genes([("m".to_owned(), GeneExpr::column("mean"))])
                .unwrap(),
            0,
            0,
            Blocks::InMemory,
        ),
        (
            "with_genes on data",
            base.with_genes([(
                "in_c1".to_owned(),
                GeneExpr::profile(Some("counts".to_owned()), "c1"),
            )])
            .unwrap(),
            1,
            1,
            C,
        ),
        ("neighbors", Arc::clone(&graphs), 0, 0, Blocks::InMemory),
        (
            "leiden",
            graphs.cluster(leiden()).unwrap(),
            0,
            0,
            Blocks::InMemory,
        ),
        (
            "sink",
            base.sink(Sink::Zarr {
                path: "out.kyto".to_owned(),
            })
            .unwrap(),
            1,
            0,
            C,
        ),
    ]
}

#[test]
fn every_node_kind_has_a_device_and_passes() {
    let mut names = std::collections::BTreeSet::new();
    for (case, plan, passes, statistics, blocks) in cases() {
        names.insert(plan.name());
        let (device, got_passes, got_statistics, apply, got_blocks) = last(&plan);
        assert_eq!(device, Device::Cpu, "{case}");
        assert_eq!(got_passes, passes, "{case}");
        assert_eq!(got_statistics, statistics, "{case}");
        assert_eq!(u32::from(apply), passes - statistics, "{case}");
        assert_eq!(got_blocks, blocks, "{case}");
        // The physical plan (`explain()`'s optimized section) shows them on every line.
        let text = plan.physical(PhysicalOptions::default()).to_string();
        assert_eq!(text.lines().count(), plan.nodes().count() + 1, "{case}");
        for line in text.lines().skip(1) {
            assert!(
                line.contains(" [device=cpu passes="),
                "{case}: no device or passes in {line:?}"
            );
        }
    }
    // Every logical node kind is covered (names as `explain` shows them).
    let expected = [
        "SCAN",
        "WITH_DEFAULT_LAYER",
        "FILTER_CELLS",
        "FILTER_GENES",
        "SELECT_GENES",
        "TAKE_CELLS",
        "NORMALIZE_TOTAL",
        "LOG1P",
        "SCALE",
        "HIGHLY_VARIABLE_GENES",
        "PCA",
        "WITH_CELLS",
        "WITH_GENES",
        "NEIGHBORS",
        "LEIDEN",
        "SINK_ZARR",
    ];
    for name in expected {
        assert!(names.contains(name), "no case for {name}: {names:?}");
    }
}

#[test]
fn passes_and_blocks_print_on_each_line() {
    let plan = scan()
        .normalize_total(None, Some("counts"), "norm")
        .unwrap()
        .with_default_layer("norm")
        .unwrap()
        .neighbors(neighbors())
        .unwrap();
    assert_eq!(
        plan.physical(PhysicalOptions::default()).to_string(),
        r#"output #3 reads [layer:counts, layer:log1p, layer:norm, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities, graph:distances]
#3 NEIGHBORS "X_pca" -> "distances", "connectivities" n_neighbors=15 metric=Euclidean search=Exact seed=0 <- #2 (node 3) [device=cpu passes=0 in memory]
#2 WITH_DEFAULT_LAYER "norm" <- #1 (node 2) [device=cpu passes=0 no data]
#1 NORMALIZE_TOTAL "counts" -> "norm" target_sum=median <- #0 (node 1) [device=cpu passes=2 (statistics=1 apply=1) blocks=4096 cells]
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads [layer:counts, layer:log1p, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca] (node 0) [device=cpu passes=1 blocks=4096 cells]"#
    );
}

/// The standard pipeline of the optimizer tests: every rule off, then on (one fused
/// pass with the median's statistics pass first).
#[test]
fn standard_pipeline_snapshots() {
    let normalized = scan()
        .normalize_total(None, Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "logn")
        .unwrap()
        .filter_cells(&gt_cell(CellExpr::column("score"), -2.0))
        .unwrap();
    let clustered = normalized
        .highly_variable_genes(Some("logn"), HvgFlavor::Seurat, Some(3), None)
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
        .cluster(leiden())
        .unwrap();
    let outputs = [
        Output::members(&normalized, [Member::Layer("logn".to_owned())]).unwrap(),
        Output::members(&clustered, [Member::CellColumn("leiden".to_owned())]).unwrap(),
    ];
    let before = physical_plan(&outputs, OptimizeOptions::NONE, PhysicalOptions::default());
    assert_eq!(
        before.to_string(),
        r#"output #3 reads [layer:logn]
output #12 reads [cells:leiden]
#12 LEIDEN "connectivities" -> "leiden" resolution=1 seed=0 <- #11 (node 8) [device=cpu passes=0 in memory]
#11 NEIGHBORS "X_new" -> "distances", "connectivities" n_neighbors=15 metric=Euclidean search=Exact seed=0 <- #10 (node 7) [device=cpu passes=0 in memory]
#10 PCA "scaled" -> "X_new" n_components=2 seed=0 <- #9 (node 6) [device=cpu passes=11 (statistics=11 apply=0) blocks=4096 cells]
#9 SCALE "logn" -> "scaled" zero_center=true max_value=10 <- #8 (node 5) [device=cpu passes=2 (statistics=1 apply=1) blocks=4096 cells]
#8 HIGHLY_VARIABLE_GENES "logn" flavor=seurat n_top_genes=3 <- #7 (node 4) [device=cpu passes=1 (statistics=1 apply=0) blocks=4096 cells]
#7 FILTER_CELLS cell_column('score') > -2.0 <- #6 (node 3) [device=cpu passes=1 blocks=4096 cells]
#6 LOG1P "norm" -> "logn" <- #5 (node 2) [device=cpu passes=1 blocks=4096 cells]
#5 NORMALIZE_TOTAL "counts" -> "norm" target_sum=median <- #4 (node 1) [device=cpu passes=2 (statistics=1 apply=1) blocks=4096 cells]
#4 SCAN "synthetic.kyto" (24 cells x 6 genes) reads [layer:counts, layer:log1p, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities] (node 0) [device=cpu passes=1 blocks=4096 cells]
#3 FILTER_CELLS cell_column('score') > -2.0 <- #2 (node 3) [device=cpu passes=1 blocks=4096 cells]
#2 LOG1P "norm" -> "logn" <- #1 (node 2) [device=cpu passes=1 blocks=4096 cells]
#1 NORMALIZE_TOTAL "counts" -> "norm" target_sum=median <- #0 (node 1) [device=cpu passes=2 (statistics=1 apply=1) blocks=4096 cells]
#0 SCAN "synthetic.kyto" (24 cells x 6 genes) reads [layer:counts, layer:log1p, cells:batch, cells:score, cells:total_counts, genes:highly_variable, genes:mean, genes:symbol, embedding:PCs, embedding:X_pca, graph:connectivities] (node 0) [device=cpu passes=1 blocks=4096 cells]"#
    );
    let after = physical_plan(&outputs, OptimizeOptions::ALL, PhysicalOptions::default());
    let text = after.to_string();
    // Every node shows its device and passes; the fused pass keeps the median's
    // statistics pass.
    for line in text.lines().filter(|line| line.starts_with('#')) {
        assert!(line.contains(" [device=cpu passes="), "{line}");
    }
    let fused = text
        .lines()
        .find(|line| line.contains("FUSED_ELEMENTWISE"))
        .unwrap();
    assert!(
        fused.ends_with("[device=cpu passes=2 (statistics=1 apply=1) blocks=4096 cells]"),
        "{fused}"
    );
    assert_eq!(after.executions().len(), after.optimized().nodes().len());
}

#[test]
fn fused_pass_without_median_is_one_pass() {
    let plan = scan()
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "logn")
        .unwrap();
    let text = plan.physical(PhysicalOptions::default()).to_string();
    let fused = text
        .lines()
        .find(|line| line.contains("FUSED_ELEMENTWISE"))
        .unwrap();
    assert!(
        fused.ends_with("[device=cpu passes=1 blocks=4096 cells]"),
        "{fused}"
    );
}

/// Blocks follow the store's tiling, through every node above the scan.
#[test]
fn blocks_follow_the_store_tiling() {
    let mut schema = scan().schema().clone();
    schema.tiling.cell_block_size = 512;
    let plan = Plan::scan(
        "small_blocks.kyto",
        schema,
        scan().root().gene_ids().as_ref().clone(),
    )
    .log1p(Some("counts"), "logn")
    .unwrap()
    .scale(Some("logn"), "scaled", false, None)
    .unwrap();
    let physical = unoptimized(&plan);
    for execution in physical.executions() {
        assert_eq!(execution.blocks(), Blocks::Cells(512));
    }
}
