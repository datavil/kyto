//! Task 4.3 Done when: the plan JSON round trip is exact for every plan in the test suite.
//!
//! Exact means: `Plan::from_json(plan.to_json())` equals `plan` node for node, has the
//! same node IDs and output schemas, and writes the same text again. Checked on
//! (1) one corpus holding every plan shape the kyto-plan tests build: every node kind,
//! every operation variant and optional argument, every expression form (leaves of each
//! source, every operator, literals of each kind including non-finite and extreme
//! floats, shared and 1000-level-deep trees, literal-only subtrees), stale-column
//! recomputes, removed genes, long chains, Unicode names; (2) random chains of random
//! steps with random expressions (proptest), each step kept only when its builder
//! accepts it. Then the format (version, IDs, `method` names) and every refusal:
//! `KY-E218` for another version, `KY-E217` for broken text, and the builders' own
//! errors for text edited into an invalid plan.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use kyto_core::{
    BinaryOp, CellExpr, Expr, ExprAxis, GeneExpr, GeneIds, Literal, NullTest, StrFunction,
};
use kyto_error::KytoError;
use kyto_plan::{
    Cluster, ClusterMethod, HvgFlavor, Metric, NeighborSearch, Neighbors, PLAN_JSON_VERSION, Plan,
    Scrublet, Sink, Umap, UmapInit,
};
use proptest::prelude::*;
use serde_json::Value;

const SCHEMA: &str = r#"{
  "n_cells": 4,
  "n_genes": 3,
  "tiling": {"cell_block_size": 4096, "gene_block_size": 1024},
  "cells": {"columns": [
    {"name": "cell_id", "dtype": "String"},
    {"name": "n_genes", "dtype": "UInt32", "lineage": {"layer": "counts", "node": "n3"}},
    {"name": "total", "dtype": "Float32", "lineage": {"layer": "log1p", "node": "n4"}},
    {"name": "batch", "dtype": "Enum", "n_categories": 2, "ordered": false},
    {"name": "stage", "dtype": "Enum", "n_categories": 3, "ordered": true},
    {"name": "score", "dtype": "Int32"},
    {"name": "flag", "dtype": "Boolean"}
  ]},
  "genes": {"columns": [
    {"name": "gene_id", "dtype": "String"},
    {"name": "symbol", "dtype": "Enum", "n_categories": 2, "ordered": false},
    {"name": "highly_variable", "dtype": "Boolean"},
    {"name": "n_cells", "dtype": "UInt32", "lineage": {"layer": "counts", "node": "n5"}},
    {"name": "mean", "dtype": "Float64"}
  ]},
  "layers": [
    {"name": "counts", "dtype": "uint32", "nnz": 7},
    {"name": "log1p", "dtype": "float32", "nnz": 18446744073709551615}
  ],
  "embeddings": [
    {"name": "X_pca", "axis": "cells", "width": 3, "dtype": "float32"},
    {"name": "PCs", "axis": "genes", "width": 3, "dtype": "float32"}
  ],
  "graphs": [
    {"name": "connectivities", "axis": "cells", "dtype": "float32", "nnz": 6},
    {"name": "coexpression", "axis": "genes", "dtype": "float32", "nnz": 2}
  ]
}"#;

/// 4 cells; genes `G1` and `G2` share the symbol `TBCE`, `G3` is `CD3E`.
fn scan() -> Arc<Plan> {
    let gene_ids = GeneIds::new(["G1", "G2", "G3"].map(str::to_owned)).with_symbols([
        Some("TBCE".to_owned()),
        Some("TBCE".to_owned()),
        Some("CD3E".to_owned()),
    ]);
    Plan::scan(
        "s3://bucket/pbmc \"3k\".kyto",
        serde_json::from_str(SCHEMA).unwrap(),
        gene_ids,
    )
}

/// The same store without symbols, and one gene without a symbol.
fn scans() -> Vec<Arc<Plan>> {
    let schema = || serde_json::from_str(SCHEMA).unwrap();
    let ids = || GeneIds::new(["G1", "G2", "G3"].map(str::to_owned));
    vec![
        scan(),
        Plan::scan("pbmc.kyto", schema(), ids()),
        Plan::scan(
            "ünïcode/路径.kyto",
            schema(),
            ids().with_symbols([None, Some("TBCE".to_owned()), None]),
        ),
    ]
}

fn layered() -> Arc<Plan> {
    scan().with_default_layer("log1p").unwrap()
}

fn cell(name: &str) -> CellExpr {
    CellExpr::column(name)
}

fn gene(name: &str) -> GeneExpr {
    GeneExpr::column(name)
}

fn bin<A: ExprAxis>(
    left: impl Into<kyto_core::Operand<A>>,
    op: BinaryOp,
    right: impl Into<kyto_core::Operand<A>>,
) -> Expr<A> {
    Expr::binary(left, op, right).unwrap()
}

fn neighbors(key: Option<&str>) -> Neighbors {
    Neighbors {
        embedding: "X_pca".to_owned(),
        n_components: None,
        n_neighbors: 15,
        metric: Metric::Euclidean,
        search: NeighborSearch::Exact,
        seed: 0,
        key: key.map(str::to_owned),
    }
}

fn cluster(method: ClusterMethod, graph: &str, into: &str) -> Cluster {
    Cluster {
        method,
        graph: graph.to_owned(),
        resolution: 1.0,
        seed: 0,
        into: into.to_owned(),
    }
}

/// Cell predicates covering every leaf, operator and literal form.
fn cell_predicates() -> Vec<CellExpr> {
    let floats = [
        0.0,
        -0.0,
        0.1 + 0.2,
        1e-308,
        5e-324,
        f64::MAX,
        f64::MIN,
        f64::EPSILON,
        std::f64::consts::PI,
        1e21,
        123_456_789.123_456_79,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut predicates = vec![
        bin(cell("score"), BinaryOp::Gt, Literal::Int(1)),
        bin(cell("score"), BinaryOp::GtEq, Literal::Int(i64::MIN)),
        bin(cell("score"), BinaryOp::LtEq, Literal::Int(i64::MAX)),
        bin(cell("flag"), BinaryOp::Eq, Literal::Boolean(false)),
        cell("flag"),
        bin(
            cell("batch"),
            BinaryOp::NotEq,
            Literal::String("b \"1\"\n\\ ✓".to_owned()),
        ),
        bin(
            cell("stage"),
            BinaryOp::Lt,
            Literal::String("late".to_owned()),
        ),
        cell("cell_id")
            .str(StrFunction::StartsWith("AAA".to_owned()))
            .unwrap(),
        cell("cell_id")
            .str(StrFunction::EndsWith("-1".to_owned()))
            .unwrap(),
        cell("batch").null_test(NullTest::IsNull).unwrap(),
        cell("score").null_test(NullTest::IsNotNull).unwrap(),
        cell("flag").not().unwrap(),
        bin(
            bin(cell("flag"), BinaryOp::And, cell("flag").not().unwrap()),
            BinaryOp::Or,
            bin(cell("score"), BinaryOp::Lt, Literal::Int(3)),
        ),
        bin(
            bin(
                bin(cell("score"), BinaryOp::Add, Literal::Int(1)),
                BinaryOp::Mul,
                bin(cell("total"), BinaryOp::Sub, Literal::Float(0.5)),
            ),
            BinaryOp::Div,
            Literal::Int(2),
        )
        .null_test(NullTest::IsNotNull)
        .unwrap(),
        bin(
            CellExpr::expression(None, "G3"),
            BinaryOp::Gt,
            Literal::Float(0.5),
        ),
        bin(
            CellExpr::expression(Some("counts".to_owned()), "G1"),
            BinaryOp::Gt,
            Literal::Int(0),
        ),
        bin(
            CellExpr::expression_by_symbol(None, "CD3E"),
            BinaryOp::Gt,
            Literal::Int(0),
        ),
        bin(
            CellExpr::feature("G3").unwrap(),
            BinaryOp::Gt,
            Literal::Int(0),
        ),
        bin(
            CellExpr::feature("counts:G2").unwrap(),
            BinaryOp::Gt,
            Literal::Int(0),
        ),
        bin(
            CellExpr::feature("meta:score").unwrap(),
            BinaryOp::Gt,
            Literal::Int(0),
        ),
        bin(
            CellExpr::feature("X_pca:2").unwrap(),
            BinaryOp::Gt,
            Literal::Int(0),
        ),
        bin(
            CellExpr::embedding("X_pca", 0),
            BinaryOp::Lt,
            Literal::Float(-1.5),
        ),
        // A literal-only subtree on the left.
        bin(
            bin(Literal::Int(1), BinaryOp::Add, Literal::Float(2.5)),
            BinaryOp::Lt,
            cell("total"),
        ),
    ];
    predicates.extend(
        floats
            .into_iter()
            .map(|value| bin(cell("total"), BinaryOp::Gt, Literal::Float(value))),
    );
    // A subtree shared by both sides, and a tree 1000 levels deep.
    let shared = bin(cell("score"), BinaryOp::Gt, Literal::Int(1));
    predicates.push(bin(shared.clone(), BinaryOp::And, shared));
    let mut deep = cell("score");
    while deep.depth() < kyto_core::MAX_EXPR_DEPTH - 1 {
        deep = bin(deep, BinaryOp::Add, Literal::Int(1));
    }
    predicates.push(bin(deep, BinaryOp::Gt, Literal::Int(0)));
    predicates
}

fn gene_predicates() -> Vec<GeneExpr> {
    vec![
        gene("highly_variable"),
        bin(gene("mean"), BinaryOp::GtEq, Literal::Float(0.0125)),
        bin(
            gene("symbol"),
            BinaryOp::Eq,
            Literal::String("CD3E".to_owned()),
        ),
        gene("gene_id")
            .str(StrFunction::StartsWith("MT-".to_owned()))
            .unwrap()
            .not()
            .unwrap(),
        bin(
            GeneExpr::profile(None, "AAAC-1"),
            BinaryOp::Gt,
            Literal::Int(0),
        ),
        bin(
            GeneExpr::profile(Some("counts".to_owned()), "細胞"),
            BinaryOp::Gt,
            Literal::Int(0),
        ),
        bin(
            GeneExpr::embedding("PCs", 1),
            BinaryOp::Gt,
            Literal::Float(0.0),
        ),
        gene("mean").null_test(NullTest::IsNull).unwrap(),
    ]
}

/// Every plan shape the kyto-plan tests build, and more.
fn corpus() -> Vec<Arc<Plan>> {
    let mut plans = scans();
    let base = layered();
    plans.push(Arc::clone(&base));
    for predicate in cell_predicates() {
        plans.push(base.filter_cells(&predicate).unwrap());
    }
    for predicate in gene_predicates() {
        plans.push(base.filter_genes(&predicate).unwrap());
    }
    // Stale columns recomputed, removed genes, both axes changed.
    let filtered = base.filter_genes(&gene("highly_variable")).unwrap();
    plans.push(
        filtered
            .filter_cells(&bin(
                cell("n_genes").recompute().unwrap(),
                BinaryOp::Gt,
                Literal::Int(1),
            ))
            .unwrap(),
    );
    let selected = base
        .select_genes(["G3".to_owned(), "G1".to_owned()])
        .unwrap();
    plans.push(Arc::clone(&selected));
    plans.push(
        selected
            .filter_cells(&bin(
                CellExpr::expression_by_symbol(None, "TBCE"),
                BinaryOp::Gt,
                Literal::Int(0),
            ))
            .unwrap(),
    );
    plans.push(
        base.take_cells(["AAAC-1".to_owned(), "c\"2\"".to_owned()])
            .unwrap(),
    );
    plans.push(
        base.take_cells(["c1".to_owned()])
            .unwrap()
            .with_genes([("n_cells".to_owned(), gene("n_cells").recompute().unwrap())])
            .unwrap(),
    );
    // Layers.
    for target_sum in [
        None,
        Some(1e4),
        Some(0.1 + 0.2),
        Some(f64::from(f32::MIN_POSITIVE)),
    ] {
        plans.push(
            scan()
                .normalize_total(target_sum, Some("counts"), "normalized")
                .unwrap(),
        );
    }
    plans.push(base.normalize_total(Some(1e4), None, "log1p").unwrap());
    plans.push(base.log1p(Some("counts"), "logged").unwrap());
    plans.push(base.log1p(None, "log1p").unwrap());
    for zero_center in [true, false] {
        for max_value in [None, Some(10.0), Some(1e-300)] {
            plans.push(base.scale(None, "scaled", zero_center, max_value).unwrap());
        }
    }
    // Cross-cell steps.
    for flavor in [
        HvgFlavor::Seurat,
        HvgFlavor::CellRanger,
        HvgFlavor::SeuratV3,
    ] {
        for n_top_genes in [None, Some(2), Some(u64::MAX)] {
            for batch_key in [None, Some("batch")] {
                plans.push(
                    base.highly_variable_genes(None, flavor, n_top_genes, batch_key)
                        .unwrap(),
                );
            }
        }
    }
    plans.push(
        base.highly_variable_genes(Some("counts"), HvgFlavor::SeuratV3, Some(2), Some("batch"))
            .unwrap(),
    );
    plans.push(base.pca(None, "X_new", 2, u64::MAX).unwrap());
    plans.push(base.pca(Some("counts"), "X_pca", 3, 0).unwrap());
    // Columns.
    plans.push(
        base.with_cells([
            (
                "ratio".to_owned(),
                bin(cell("total"), BinaryOp::Div, cell("score")),
            ),
            (
                "big".to_owned(),
                bin(cell("score"), BinaryOp::Gt, Literal::Int(2)),
            ),
            (
                "constant".to_owned(),
                bin(Literal::Int(1), BinaryOp::Add, Literal::Int(2)),
            ),
            (
                "score".to_owned(),
                bin(cell("score"), BinaryOp::Mul, Literal::Float(f64::NAN)),
            ),
            ("Ünï".to_owned(), CellExpr::expression(None, "G1")),
        ])
        .unwrap(),
    );
    plans.push(
        base.with_genes([
            (
                "dispersion".to_owned(),
                bin(gene("mean"), BinaryOp::Mul, Literal::Int(2)),
            ),
            ("in_cell".to_owned(), GeneExpr::profile(None, "AAAC-1")),
        ])
        .unwrap(),
    );
    plans.push(base.with_cells(Vec::<(String, CellExpr)>::new()).unwrap());
    // Graphs and clusters.
    let pca = base.pca(None, "X_new", 3, 0).unwrap();
    for metric in [Metric::Euclidean, Metric::Cosine] {
        for search in [NeighborSearch::Exact, NeighborSearch::Approximate] {
            plans.push(
                pca.neighbors(Neighbors {
                    embedding: "X_new".to_owned(),
                    n_components: Some(2),
                    metric,
                    search,
                    seed: 7,
                    ..neighbors(Some("new"))
                })
                .unwrap(),
            );
        }
    }
    let graphs = base.neighbors(neighbors(None)).unwrap();
    plans.push(Arc::clone(&graphs));
    for method in [ClusterMethod::Leiden, ClusterMethod::Louvain] {
        plans.push(
            graphs
                .cluster(cluster(method, "connectivities", method.name()))
                .unwrap(),
        );
        plans.push(
            graphs
                .cluster(Cluster {
                    resolution: 0.1 + 0.7,
                    seed: u64::MAX,
                    ..cluster(method, "connectivities", "cluster")
                })
                .unwrap(),
        );
    }
    plans.push(graphs.umap(Umap::default()).unwrap());
    plans.push(
        graphs
            .umap(Umap {
                n_components: 3,
                min_dist: 0.1 + 0.2,
                spread: 1.5,
                n_epochs: Some(200),
                alpha: 0.25,
                gamma: 0.0,
                negative_sample_rate: 0,
                init: UmapInit::Random,
                seed: u64::MAX,
                into: "X_new".to_owned(),
                ..Umap::default()
            })
            .unwrap(),
    );
    // Doublets.
    plans.push(base.scrublet(Some("counts"), Scrublet::default()).unwrap());
    plans.push(
        base.scrublet(
            None,
            Scrublet {
                batch_key: Some("batch".to_owned()),
                sim_doublet_ratio: 0.1 + 0.2,
                expected_doublet_rate: 0.06,
                stdev_doublet_rate: 0.0,
                n_prin_comps: 2,
                n_neighbors: Some(5),
                threshold: Some(-0.25),
                seed: u64::MAX,
                score_into: "Ünï score".to_owned(),
                predicted_into: "doublet".to_owned(),
            },
        )
        .unwrap(),
    );
    // Sinks end plans.
    for sink in [
        Sink::Zarr {
            path: "s3://out/pbmc.kyto".to_owned(),
        },
        Sink::H5ad {
            path: "out/pbmc.h5ad".to_owned(),
        },
    ] {
        plans.push(scan().sink(sink.clone()).unwrap());
        plans.push(graphs.sink(sink).unwrap());
    }
    // A whole pipeline, and a long chain.
    let pipeline = scan()
        .filter_cells(&bin(cell("score"), BinaryOp::GtEq, Literal::Int(1)))
        .unwrap()
        .filter_genes(&bin(
            gene("n_cells").recompute().unwrap(),
            BinaryOp::GtEq,
            Literal::Int(1),
        ))
        .unwrap()
        .normalize_total(Some(1e4), Some("counts"), "normalized")
        .unwrap()
        .log1p(Some("normalized"), "log1p")
        .unwrap()
        .with_default_layer("log1p")
        .unwrap()
        .highly_variable_genes(None, HvgFlavor::Seurat, None, None)
        .unwrap()
        .filter_genes(&gene("highly_variable"))
        .unwrap()
        .scale(None, "scaled", true, Some(10.0))
        .unwrap()
        .pca(Some("scaled"), "X_pca", 2, 0)
        .unwrap()
        .neighbors(neighbors(None))
        .unwrap()
        .cluster(cluster(ClusterMethod::Leiden, "connectivities", "leiden"))
        .unwrap()
        .sink(Sink::Zarr {
            path: "out.kyto".to_owned(),
        })
        .unwrap();
    plans.push(pipeline);
    let mut long = scan();
    for _ in 0..2_000 {
        long = long.filter_genes(&gene("highly_variable")).unwrap();
    }
    plans.push(long);
    plans
}

/// The round trip of `plan` is exact; returns its text.
fn assert_round_trip(plan: &Arc<Plan>) -> String {
    let json = plan.to_json();
    let back = Plan::from_json(&json).unwrap_or_else(|error| panic!("{error}\n{plan}"));
    // Task 8.4: the text matches the published JSON Schema and has no diagnostics.
    let diagnostics = kyto_plan::validate_plan_json(&json);
    assert!(diagnostics.is_empty(), "{diagnostics:?}\n{plan}");
    assert_eq!(back.to_json(), json, "{plan}");
    // A NaN literal is never equal to itself; the text above is the check then.
    if !json.contains("\"NaN\"") {
        assert_eq!(*back, **plan);
    }
    for (left, right) in back.nodes().zip(plan.nodes()) {
        assert_eq!(left.id(), right.id());
        assert_eq!(left.name(), right.name());
        assert_eq!(left.output_schema(), right.output_schema());
        assert_eq!(left.genes(), right.genes());
        assert_eq!(left.default_layer(), right.default_layer());
        assert_eq!(left.removed_genes(), right.removed_genes());
    }
    assert_eq!(back.nodes().count(), plan.nodes().count());
    json
}

#[test]
fn every_plan_of_the_corpus_round_trips_exactly() {
    let plans = corpus();
    assert!(plans.len() > 100, "{}", plans.len());
    for plan in &plans {
        assert_round_trip(plan);
    }
}

#[test]
fn equal_plans_write_equal_text() {
    let left = corpus();
    let right = corpus();
    for (left, right) in left.iter().zip(&right) {
        assert_eq!(left.to_json(), right.to_json());
    }
}

#[test]
fn the_text_has_a_version_stable_ids_and_the_methods() {
    let plan = layered()
        .filter_cells(&bin(cell("score"), BinaryOp::Gt, Literal::Int(1)))
        .unwrap()
        .cluster(cluster(ClusterMethod::Louvain, "connectivities", "louvain"))
        .unwrap()
        .sink(Sink::H5ad {
            path: "out.h5ad".to_owned(),
        })
        .unwrap();
    let json: Value = serde_json::from_str(&plan.to_json()).unwrap();
    assert_eq!(json["format"], "kyto-plan");
    assert_eq!(json["version"], PLAN_JSON_VERSION);
    assert_eq!(PLAN_JSON_VERSION, 1);
    let nodes = json["nodes"].as_array().unwrap();
    let mut chain: Vec<&Plan> = plan.nodes().collect();
    chain.reverse();
    assert_eq!(nodes.len(), chain.len());
    for (position, (node, built)) in nodes.iter().zip(chain).enumerate() {
        assert_eq!(node["id"], position);
        assert_eq!(node["id"], built.id().0);
        let input = if position == 0 {
            Value::Null
        } else {
            Value::from(position - 1)
        };
        assert_eq!(node["input"], input);
        assert_eq!(node["step"]["method"], built.node().method());
    }
    // The predicate is resolved (R6) and flat, the root last.
    assert_eq!(
        nodes[2]["step"]["predicate"],
        serde_json::json!([
            {"op": "column", "name": "score"},
            {"op": "int", "value": 1},
            {"op": "binary", "operator": ">", "left": 0, "right": 1}
        ])
    );
    // The scan carries its snapshot: schema, gene IDs and symbols.
    let step = &nodes[0]["step"];
    assert_eq!(step["gene_ids"], serde_json::json!(["G1", "G2", "G3"]));
    assert_eq!(step["symbols"], serde_json::json!(["TBCE", "TBCE", "CD3E"]));
    assert_eq!(step["schema"]["n_genes"], 3);
}

#[test]
fn every_method_name_of_the_corpus_is_the_nodes_method() {
    for plan in corpus() {
        let json: Value = serde_json::from_str(&plan.to_json()).unwrap();
        let mut chain: Vec<&Plan> = plan.nodes().collect();
        chain.reverse();
        for (node, built) in json["nodes"].as_array().unwrap().iter().zip(chain) {
            assert_eq!(node["step"]["method"], built.node().method());
        }
    }
}

#[test]
fn reading_needs_no_store_and_names_stay_resolved() {
    // The predicate names a gene by symbol and the default layer; the text holds the
    // gene ID and the layer in full, and so does the plan read back.
    let plan = layered()
        .filter_cells(&bin(
            CellExpr::expression_by_symbol(None, "CD3E"),
            BinaryOp::Gt,
            Literal::Int(0),
        ))
        .unwrap();
    let json = assert_round_trip(&plan);
    assert!(
        json.contains(r#"{"op":"gene_expression","layer":"log1p","gene_id":"G3"}"#),
        "{json}"
    );
}

// --- Refusals ------------------------------------------------------------------

fn edited(plan: &Plan, edit: impl FnOnce(&mut Value)) -> String {
    let mut json: Value = serde_json::from_str(&plan.to_json()).unwrap();
    edit(&mut json);
    json.to_string()
}

fn refused(text: &str) -> KytoError {
    let error = Plan::from_json(text).unwrap_err();
    // Task 8.4: `validate_plan_json` reports it too, first, with the same code.
    let diagnostics = kyto_plan::validate_plan_json(text);
    assert_eq!(
        diagnostics.first().map(kyto_plan::PlanDiagnostic::code),
        Some(error.code()),
        "{error}\n{diagnostics:?}"
    );
    error
}

#[test]
fn another_version_is_ky_e218_before_anything_else() {
    let plan = layered();
    let text = edited(&plan, |json| {
        json["version"] = Value::from(2);
        json["nodes"] = Value::from("whatever version 2 holds");
    });
    let error = refused(&text);
    assert_eq!(error.code(), "KY-E218");
    assert!(error.to_string().contains("version 2"), "{error}");
    assert!(error.to_string().contains("only version 1"), "{error}");
}

#[test]
fn broken_text_is_ky_e217_with_where_and_what() {
    let plan = layered()
        .filter_cells(&bin(cell("score"), BinaryOp::Gt, Literal::Int(1)))
        .unwrap();
    let cases: Vec<(String, &str)> = vec![
        ("not json".to_owned(), "line 1 column"),
        ("{}".to_owned(), "key 'version': it is missing"),
        (
            r#"{"format":"other","version":1,"nodes":[]}"#.to_owned(),
            r#"it is "other""#,
        ),
        (
            r#"{"format":"kyto-plan","version":1,"nodes":[]}"#.to_owned(),
            "at least one node",
        ),
        (
            edited(&plan, |json| json["extra"] = Value::from(1)),
            "unknown field `extra`",
        ),
        (
            edited(&plan, |json| {
                json["nodes"][1]["step"]["method"] = Value::from("explode")
            }),
            "unknown variant `explode`",
        ),
        (
            edited(&plan, |json| {
                json["nodes"][1]["step"]["colour"] = Value::from("red")
            }),
            "unknown field `colour`",
        ),
        (
            edited(&plan, |json| {
                json["nodes"][1]["step"]
                    .as_object_mut()
                    .unwrap()
                    .remove("layer");
            }),
            "missing field `layer`",
        ),
        (
            edited(&plan, |json| json["nodes"][1]["id"] = Value::from(5)),
            "node 1 (with_default_layer)",
        ),
        (
            edited(&plan, |json| json["nodes"][2]["input"] = Value::from(0)),
            "chains only",
        ),
        (
            edited(&plan, |json| json["nodes"][0]["input"] = Value::from(0)),
            "node 0 (scan)",
        ),
        (
            edited(&plan, |json| {
                let nodes = json["nodes"].as_array_mut().unwrap();
                nodes.swap(0, 1);
                nodes[0]["id"] = Value::from(0);
                nodes[0]["input"] = Value::Null;
                nodes[1]["id"] = Value::from(1);
                nodes[1]["input"] = Value::from(0);
            }),
            "the first node of a plan is a scan",
        ),
        (
            edited(&plan, |json| {
                let scan = json["nodes"][0].clone();
                json["nodes"][1]["step"] = scan["step"].clone();
            }),
            "only the first node is a scan",
        ),
        (
            edited(&plan, |json| {
                json["nodes"][0]["step"]["gene_ids"] = serde_json::json!(["G1", "G2"]);
            }),
            "2 gene IDs",
        ),
        (
            edited(&plan, |json| {
                json["nodes"][0]["step"]["gene_ids"] = serde_json::json!(["G1", "G1", "G2"]);
            }),
            "not unique",
        ),
        (
            edited(&plan, |json| {
                json["nodes"][0]["step"]["symbols"] = serde_json::json!(["A"]);
            }),
            "1 symbols for 3 genes",
        ),
        (
            edited(&plan, |json| {
                json["nodes"][2]["step"]["predicate"][2]["left"] = Value::from(2);
            }),
            "node 2 (filter_cells), predicate entry 2",
        ),
        (
            edited(&plan, |json| {
                json["nodes"][2]["step"]["predicate"] = serde_json::json!([]);
            }),
            "no entries",
        ),
        (
            edited(&plan, |json| {
                json["nodes"][2]["step"]["predicate"][0] =
                    serde_json::json!({"op": "cell_profile", "layer": null, "cell_id": "c"});
            }),
            "'cell_profile' makes a gene expression",
        ),
        (
            edited(&plan, |json| {
                json["nodes"][2]["step"]["predicate"][1] =
                    serde_json::json!({"op": "float", "value": "NotANumber"});
            }),
            "line 1 column",
        ),
        (
            edited(&plan, |json| {
                json["nodes"][2]["step"]["predicate"][2]["operator"] = Value::from("**");
            }),
            "unknown variant `**`",
        ),
    ];
    for (text, expected) in cases {
        let error = refused(&text);
        assert_eq!(error.code(), "KY-E217", "{expected}: {error}");
        assert!(error.to_string().contains(expected), "{expected}: {error}");
    }
}

#[test]
fn the_version_is_checked_first_and_must_be_an_integer() {
    for (text, code, expected) in [
        (
            r#"{"version":3,"format":"something else","nodes":7}"#,
            "KY-E218",
            "version 3",
        ),
        (
            r#"{"format":"kyto-plan","version":-1,"nodes":[]}"#,
            "KY-E217",
            "non-negative integer",
        ),
        (
            r#"{"format":"kyto-plan","version":"2","nodes":[]}"#,
            "KY-E217",
            "non-negative integer",
        ),
        (
            r#"{"format":"kyto-plan","version":1.0,"nodes":[]}"#,
            "KY-E217",
            "non-negative integer",
        ),
        (
            r#"{"format":"kyto-plan","nodes":[]}"#,
            "KY-E217",
            "key 'version'",
        ),
        (r#"{"version":1,"nodes":[]}"#, "KY-E217", "key 'format'"),
    ] {
        let error = refused(text);
        assert_eq!(error.code(), code, "{text}: {error}");
        assert!(error.to_string().contains(expected), "{text}: {error}");
    }
}

#[test]
fn a_scan_schema_no_store_has_is_refused() {
    let plan = layered();
    let cases: Vec<(String, &str)> = vec![
        (
            edited(&plan, |json| {
                json["nodes"][0]["step"]["schema"]["tiling"]["cell_block_size"] = Value::from(0);
            }),
            "cell_block_size = 0",
        ),
        (
            edited(&plan, |json| {
                let layers = json["nodes"][0]["step"]["schema"]["layers"]
                    .as_array_mut()
                    .unwrap();
                let first = layers[0].clone();
                layers.push(first);
            }),
            "the layer 'counts' twice",
        ),
        (
            edited(&plan, |json| {
                let columns = json["nodes"][0]["step"]["schema"]["cells"]["columns"]
                    .as_array_mut()
                    .unwrap();
                let first = columns[1].clone();
                columns.push(first);
            }),
            "the cell column 'n_genes' twice",
        ),
        (
            edited(&plan, |json| {
                json["nodes"][0]["step"]["schema"]["modalities"] =
                    serde_json::json!([{"name": "adt", "n_features": 1}]);
            }),
            "line 1 column",
        ),
    ];
    for (text, expected) in cases {
        let error = refused(&text);
        assert_eq!(error.code(), "KY-E217", "{expected}: {error}");
        assert!(error.to_string().contains(expected), "{expected}: {error}");
    }
}

#[test]
fn shared_entries_cannot_expand_without_bound() {
    // 30 entries, each `previous & previous`: 2^30 nodes once expanded.
    let plan = scan().filter_cells(&cell("flag")).unwrap();
    let text = edited(&plan, |json| {
        let mut entries = vec![serde_json::json!({"op": "column", "name": "flag"})];
        for position in 0..30 {
            entries.push(serde_json::json!(
                {"op": "binary", "operator": "&", "left": position, "right": position}
            ));
        }
        json["nodes"][1]["step"]["predicate"] = Value::from(entries);
    });
    let started = std::time::Instant::now();
    let error = refused(&text);
    assert!(started.elapsed().as_secs() < 5);
    assert_eq!(error.code(), "KY-E217");
    assert!(error.to_string().contains("predicate entry 19"), "{error}");
    assert!(error.to_string().contains("1000000"), "{error}");
}

#[test]
fn text_edited_into_an_invalid_plan_is_refused_by_the_builders() {
    let plan = layered()
        .filter_cells(&bin(cell("score"), BinaryOp::Gt, Literal::Int(1)))
        .unwrap()
        .pca(None, "X_new", 2, 0)
        .unwrap();
    let cases: Vec<(String, &str)> = vec![
        (
            edited(&plan, |json| {
                json["nodes"][1]["step"]["layer"] = Value::from("raw")
            }),
            "KY-E109",
        ),
        (
            edited(&plan, |json| {
                json["nodes"][2]["step"]["predicate"][0]["name"] = Value::from("scores");
            }),
            "KY-E210",
        ),
        (
            edited(&plan, |json| {
                json["nodes"][2]["step"]["predicate"][1] =
                    serde_json::json!({"op": "string", "value": "x"});
            }),
            "KY-E205",
        ),
        (
            edited(&plan, |json| {
                json["nodes"][3]["step"]["n_components"] = Value::from(0)
            }),
            "KY-E215",
        ),
        (
            edited(&plan, |json| {
                json["nodes"][2]["step"]["predicate"][0] =
                    serde_json::json!({"op": "feature", "name": "log1p:"});
            }),
            "KY-E208",
        ),
        (
            edited(&plan, |json| {
                let sink = serde_json::json!({"id": 1, "input": 0, "step": {"method": "sink_zarr", "path": "a"}});
                let nodes = json["nodes"].as_array_mut().unwrap();
                nodes.insert(1, sink);
                for (position, node) in nodes.iter_mut().enumerate().skip(1) {
                    node["id"] = Value::from(position);
                    node["input"] = Value::from(position - 1);
                }
            }),
            "KY-E216",
        ),
    ];
    for (text, code) in cases {
        let error = refused(&text);
        assert_eq!(error.code(), code, "{error}");
    }
}

// --- Random chains --------------------------------------------------------------

/// A random cell expression over the corpus schema's names (may be ill-typed: the
/// builders then refuse the step, and the chain goes on without it).
fn cell_expr() -> impl Strategy<Value = CellExpr> {
    let leaf = prop_oneof![
        prop::sample::select(vec!["score", "flag", "batch", "stage", "total", "cell_id"])
            .prop_map(cell),
        prop::sample::select(vec!["G1", "G2", "G3"])
            .prop_map(|gene_id| CellExpr::expression(None, gene_id)),
        (0..4_u64).prop_map(|component| CellExpr::embedding("X_pca", component)),
    ];
    leaf.prop_recursive(4, 24, 2, |inner| {
        prop_oneof![
            (inner.clone(), operator(), inner.clone())
                .prop_filter_map("kinds", |(left, op, right)| Expr::binary(left, op, right)
                    .ok()),
            (inner.clone(), operator(), literal())
                .prop_filter_map("kinds", |(left, op, right)| Expr::binary(left, op, right)
                    .ok()),
            inner
                .clone()
                .prop_filter_map("kinds", |expr| expr.not().ok()),
            inner.clone().prop_filter_map("kinds", |expr| expr
                .str(StrFunction::StartsWith("A".to_owned()))
                .ok()),
            inner.prop_filter_map("kinds", |expr| expr.null_test(NullTest::IsNull).ok()),
        ]
    })
}

fn operator() -> impl Strategy<Value = BinaryOp> {
    prop::sample::select(vec![
        BinaryOp::Eq,
        BinaryOp::NotEq,
        BinaryOp::Lt,
        BinaryOp::LtEq,
        BinaryOp::Gt,
        BinaryOp::GtEq,
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::And,
        BinaryOp::Or,
    ])
}

fn literal() -> impl Strategy<Value = Literal> {
    prop_oneof![
        any::<bool>().prop_map(Literal::Boolean),
        any::<i64>().prop_map(Literal::Int),
        any::<f64>().prop_map(Literal::Float),
        "\\PC{0,6}".prop_map(Literal::String),
    ]
}

/// One step of a random chain.
#[derive(Debug, Clone)]
enum Step {
    DefaultLayer(&'static str),
    FilterCells(CellExpr),
    FilterGenes(&'static str),
    SelectGenes(Vec<String>),
    TakeCells(Vec<String>),
    NormalizeTotal(Option<f64>, String),
    Log1p(String),
    Scale(bool, Option<f64>, String),
    Hvg(HvgFlavor, Option<u64>, bool),
    Pca(u64, u64, String),
    WithCells(String, CellExpr),
    Neighbors(Option<u64>, u64, Option<String>),
    Cluster(ClusterMethod, f64, String),
    Sink(bool, String),
}

fn name() -> impl Strategy<Value = String> {
    // Mostly valid names; a few that some builders refuse.
    prop::sample::select(vec![
        "a",
        "b",
        "c",
        "log1p",
        "X_new",
        "out",
        "Ünï code",
        "a b",
        "a:b",
        "",
    ])
    .prop_map(str::to_owned)
}

/// Mostly positive finite floats, sometimes anything (refused where not accepted).
fn positive() -> impl Strategy<Value = f64> {
    prop_oneof![4 => 1e-6..1e6_f64, 1 => any::<f64>()]
}

fn step() -> impl Strategy<Value = Step> {
    let ids = |pool: Vec<&'static str>| {
        prop::sample::subsequence(pool.clone(), 0..=pool.len())
            .prop_shuffle()
            .prop_map(|ids| ids.into_iter().map(str::to_owned).collect::<Vec<_>>())
    };
    prop_oneof![
        3 => prop::sample::select(vec!["counts", "log1p", "a", "missing"]).prop_map(Step::DefaultLayer),
        3 => cell_expr().prop_map(Step::FilterCells),
        2 => prop::sample::select(vec!["highly_variable", "mean", "n_cells"]).prop_map(Step::FilterGenes),
        1 => ids(vec!["G1", "G2", "G3"]).prop_map(Step::SelectGenes),
        1 => ids(vec!["c1", "c2", "c3", "c4", "c5"]).prop_map(Step::TakeCells),
        2 => (proptest::option::of(positive()), name()).prop_map(|(sum, into)| Step::NormalizeTotal(sum, into)),
        2 => name().prop_map(Step::Log1p),
        2 => (any::<bool>(), proptest::option::of(positive()), name())
            .prop_map(|(center, max, into)| Step::Scale(center, max, into)),
        2 => (
            prop::sample::select(vec![HvgFlavor::Seurat, HvgFlavor::CellRanger, HvgFlavor::SeuratV3]),
            proptest::option::of(0..5_u64),
            any::<bool>()
        )
            .prop_map(|(flavor, n, batch)| Step::Hvg(flavor, n, batch)),
        2 => (0..4_u64, any::<u64>(), name()).prop_map(|(n, seed, into)| Step::Pca(n, seed, into)),
        2 => (name(), cell_expr()).prop_map(|(name, expr)| Step::WithCells(name, expr)),
        2 => (proptest::option::of(0..4_u64), 0..20_u64, proptest::option::of(name()))
            .prop_map(|(n, k, key)| Step::Neighbors(n, k, key)),
        2 => (
            prop::sample::select(vec![ClusterMethod::Leiden, ClusterMethod::Louvain]),
            positive(),
            name()
        )
            .prop_map(|(method, resolution, into)| Step::Cluster(method, resolution, into)),
        // Rare: a sink ends the chain.
        1 => (any::<bool>(), name()).prop_map(|(zarr, path)| Step::Sink(zarr, path)),
    ]
}

fn apply(plan: &Arc<Plan>, step: Step) -> Result<Arc<Plan>, KytoError> {
    match step {
        Step::DefaultLayer(layer) => plan.with_default_layer(layer),
        Step::FilterCells(predicate) => plan.filter_cells(&predicate),
        Step::FilterGenes(column) => plan.filter_genes(&gene(column)),
        Step::SelectGenes(ids) => plan.select_genes(ids),
        Step::TakeCells(ids) => plan.take_cells(ids),
        Step::NormalizeTotal(sum, into) => plan.normalize_total(sum, None, &into),
        Step::Log1p(into) => plan.log1p(None, &into),
        Step::Scale(center, max, into) => plan.scale(None, &into, center, max),
        Step::Hvg(flavor, n, batch) => {
            plan.highly_variable_genes(None, flavor, n, batch.then_some("batch"))
        }
        Step::Pca(n, seed, into) => plan.pca(None, &into, n, seed),
        Step::WithCells(name, expr) => plan.with_cells([(name, expr)]),
        Step::Neighbors(n_components, n_neighbors, key) => plan.neighbors(Neighbors {
            n_components,
            n_neighbors,
            key,
            ..neighbors(None)
        }),
        Step::Cluster(method, resolution, into) => plan.cluster(Cluster {
            resolution,
            ..cluster(method, "connectivities", &into)
        }),
        Step::Sink(zarr, path) => plan.sink(if zarr {
            Sink::Zarr { path }
        } else {
            Sink::H5ad { path }
        }),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn random_chains_round_trip_exactly(start in 0..4_usize, steps in proptest::collection::vec(step(), 0..16)) {
        let mut plan = match start {
            3 => layered(),
            start => Arc::clone(&scans()[start]),
        };
        for step in steps {
            if let Ok(next) = apply(&plan, step) {
                plan = next;
            }
        }
        assert_round_trip(&plan);
    }

    #[test]
    fn random_float_literals_keep_their_bits(value in any::<f64>()) {
        let plan = scan()
            .filter_cells(&bin(cell("total"), BinaryOp::Gt, Literal::Float(value)))
            .unwrap();
        let back = Plan::from_json(&plan.to_json()).unwrap();
        let kyto_plan::Plan::Filter(node) = &*back else { panic!("not a filter") };
        let kyto_plan::Predicate::Cells(predicate) = node.params().predicate() else { panic!("not cells") };
        let kyto_core::ExprNode::Binary { right, .. } = predicate.node().as_ref() else { panic!("not binary") };
        let kyto_core::ExprNode::Literal(Literal::Float(read)) = right.as_ref() else { panic!("not a float") };
        prop_assert_eq!(read.to_bits(), value.to_bits());
    }
}
