//! Task 4.2 Done when: every `KY-E2xx` error fires from a plan that has read no data.
//!
//! The plan cannot read data: kyto-plan does not depend on kyto-store, and a scan holds
//! only the schema and gene IDs the caller read for it (`schema.json` and the ID column
//! of `genes.parquet`, metadata). So every error below is raised with zero matrix reads.
//! `tests/test_build_validation.py` shows the same from Python on a real store that is
//! deleted right after the scan.
//!
//! Each `KY-E2xx` code of the registry has a case here, raised while a plan is built
//! (a step's arguments, the names and types its expressions resolve to), or, for the
//! few that only Python can make (a Rust type rules them out), a reason and the Python
//! case. A new `KY-E2xx` code fails this test until it gets one. Then the name rules of
//! format.md that steps now check (§2.1 members, §5.6 columns).
#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;
use std::sync::Arc;

use kyto_core::{BinaryOp, CellExpr, GeneExpr, GeneIds, Literal};
use kyto_error::KytoError;
use kyto_plan::{Cluster, ClusterMethod, HvgFlavor, Metric, NeighborSearch, Neighbors, Plan, Sink};

const SCHEMA: &str = r#"{
  "n_cells": 4,
  "n_genes": 3,
  "tiling": {"cell_block_size": 4096, "gene_block_size": 1024},
  "cells": {"columns": [
    {"name": "cell_id", "dtype": "String"},
    {"name": "n_genes", "dtype": "UInt32", "lineage": {"layer": "counts", "node": "n3"}},
    {"name": "CD3E", "dtype": "Float32"},
    {"name": "batch", "dtype": "Enum", "n_categories": 2, "ordered": false},
    {"name": "score", "dtype": "Int32"}
  ]},
  "genes": {"columns": [
    {"name": "gene_id", "dtype": "String"},
    {"name": "symbol", "dtype": "Enum", "n_categories": 2, "ordered": false},
    {"name": "highly_variable", "dtype": "Boolean"}
  ]},
  "layers": [
    {"name": "counts", "dtype": "uint32", "nnz": 7},
    {"name": "log1p", "dtype": "float32", "nnz": 7}
  ],
  "embeddings": [
    {"name": "X_pca", "axis": "cells", "width": 3, "dtype": "float32"},
    {"name": "PCs", "axis": "genes", "width": 3, "dtype": "float32"}
  ],
  "graphs": [
    {"name": "connectivities", "axis": "cells", "dtype": "float32", "nnz": 6}
  ]
}"#;

/// 4 cells, genes `G1`, `G2` (both symbol `TBCE`) and `CD3E`; cell column `CD3E` too.
fn scan() -> Arc<Plan> {
    let gene_ids = GeneIds::new(["G1", "G2", "CD3E"].map(str::to_owned)).with_symbols([
        Some("TBCE".to_owned()),
        Some("TBCE".to_owned()),
        Some("CD3".to_owned()),
    ]);
    Plan::scan("pbmc.kyto", serde_json::from_str(SCHEMA).unwrap(), gene_ids)
}

fn layered() -> Arc<Plan> {
    scan().with_default_layer("log1p").unwrap()
}

fn neighbors(key: Option<&str>) -> Neighbors {
    Neighbors {
        embedding: "X_pca".to_owned(),
        n_components: None,
        n_neighbors: 3,
        metric: Metric::Euclidean,
        search: NeighborSearch::Exact,
        seed: 0,
        key: key.map(str::to_owned),
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

fn error<T: std::fmt::Debug>(result: Result<T, KytoError>) -> KytoError {
    result.unwrap_err()
}

/// One `KY-E2xx` error per code, each from building a plan or an expression for one.
fn plan_build_errors() -> Vec<KytoError> {
    let column = CellExpr::column;
    vec![
        // A short name that is a cell column and a gene of the default layer.
        error(
            layered().filter_cells(
                &CellExpr::binary(
                    CellExpr::feature("CD3E").unwrap(),
                    BinaryOp::Gt,
                    Literal::Float(1.0),
                )
                .unwrap(),
            ),
        ),
        // A gene left out by a selection, read later.
        error(
            layered()
                .select_genes(["G1".to_owned()])
                .unwrap()
                .with_cells([("x".to_owned(), CellExpr::expression(None, "CD3E"))]),
        ),
        // A symbol two genes carry.
        error(
            layered().with_cells([("x".to_owned(), CellExpr::expression_by_symbol(None, "TBCE"))]),
        ),
        // A filter predicate that resolves to a number, not a boolean.
        error(scan().filter_cells(&column("score"))),
        // An expression deeper than kyto builds.
        error((0..1000).try_fold(column("score"), |expr, _| {
            CellExpr::binary(expr, BinaryOp::Add, Literal::Int(1))
        })),
        // A feature name that cannot name anything.
        error(CellExpr::feature("log1p:")),
        // A name the plan does not have.
        error(scan().filter_genes(&GeneExpr::column("highly_variabel"))),
        // A gene read from the default layer of a plan that has none.
        error(scan().with_cells([("x".to_owned(), CellExpr::expression(None, "G1"))])),
        // `.recompute()` on something other than a column read.
        error(
            CellExpr::binary(column("n_genes"), BinaryOp::Add, Literal::Int(1))
                .and_then(|expr| expr.recompute()),
        ),
        // A step argument out of range.
        error(layered().pca(None, "pca", 0, 0)),
        // A step after the sink.
        error(
            layered()
                .sink(Sink::Zarr {
                    path: "out.kyto".to_owned(),
                })
                .unwrap()
                .log1p(None, "log1p"),
        ),
    ]
}

/// Codes only the Python layer raises: the Rust types rule the mistake out.
const PYTHON_ONLY: [(&str, &str); 5] = [
    (
        "KY-E202",
        "CellExpr and GeneExpr are distinct Rust types; Python: cell_column(...) & gene_column(...)",
    ),
    (
        "KY-E206",
        "Rust has no truth value of an expression; Python: bool(cell_column(...) > 1)",
    ),
    (
        "KY-E209",
        "a component is a u64 in Rust; Python: cell_embedding('X_pca', -1)",
    ),
    (
        "KY-E212",
        "Rust has no None literal; Python: cell_column(...) == None",
    ),
    (
        "KY-E214",
        "`by` is a Rust function choice; Python: gene_expression(..., by='name')",
    ),
];

#[test]
fn every_ky_e2xx_code_fires_while_the_plan_is_built() {
    let registered: BTreeSet<&str> = KytoError::CODES
        .into_iter()
        .filter(|code| code.starts_with("KY-E2"))
        .collect();
    let mut covered = BTreeSet::new();
    for error in plan_build_errors() {
        let code = error.code();
        assert!(code.starts_with("KY-E2"), "{code}: {error}");
        assert!(covered.insert(code), "{code} has two cases");
    }
    for (code, reason) in PYTHON_ONLY {
        assert!(!reason.is_empty());
        assert!(covered.insert(code), "{code} is raised from Rust too");
    }
    assert_eq!(covered, registered, "every KY-E2xx code needs a case");
}

#[test]
fn errors_name_the_step_and_the_fix() {
    let errors = plan_build_errors();
    let text = |code: &str| {
        errors
            .iter()
            .find(|error| error.code() == code)
            .unwrap()
            .to_string()
    };
    assert!(text("KY-E201").contains("'meta:CD3E'"));
    assert!(text("KY-E203").contains("'select_genes'"));
    assert!(text("KY-E204").contains("'G1', 'G2'"));
    assert!(text("KY-E205").contains("'filter_cells' needs"));
    assert!(text("KY-E210").contains("'highly_variabel'"));
    assert!(text("KY-E215").contains("pca(..., n_components=0)"));
    assert!(text("KY-E216").contains("'log1p' cannot follow 'sink_zarr'"));
}

// --- Name rules of format.md, checked when a step names what it writes -----------

fn refused(result: Result<Arc<Plan>, KytoError>) -> String {
    let error = result.unwrap_err();
    assert_eq!(error.code(), "KY-E215", "{error}");
    format!("{error}\n{}", error.hint())
}

#[test]
fn new_layer_names_follow_the_member_rules() {
    let plan = layered();
    let message = refused(plan.log1p(None, "log/1p"));
    assert!(message.contains("log1p(..., into='log/1p') is invalid: character '/'"));
    assert!(message.contains("1 to 64 characters"));
    assert!(refused(plan.log1p(None, "")).contains("the name is empty"));
    assert!(refused(plan.log1p(None, &"a".repeat(65))).contains("65 bytes"));
    assert!(refused(plan.normalize_total(None, None, ".hidden")).contains("starts with"));
    assert!(refused(plan.scale(None, "__x", true, None)).contains("starts with"));
    assert!(refused(plan.scale(None, "Con.txt", true, None)).contains("Windows device"));
    // Case-insensitive clash with another layer; an equal name replaces it.
    let message = refused(plan.log1p(None, "Log1P"));
    assert!(message.contains("layer 'Log1P' differs from the plan's layer 'log1p' only in case"));
    assert!(message.contains("pass 'log1p' (to replace that layer) or a name"));
    plan.log1p(None, "log1p").unwrap();
    plan.log1p(None, "a".repeat(64).as_str()).unwrap();
    // A layer may share a name with an embedding or a graph (format.md §2.1).
    plan.log1p(None, "x_pca").unwrap();
    // A layer added upstream counts.
    let added = plan.log1p(None, "norm").unwrap();
    assert!(refused(added.scale(None, "NORM", true, None)).contains("'norm'"));
}

#[test]
fn derived_member_names_are_checked_too() {
    let plan = layered();
    // `<into>_loadings` would be 66 bytes.
    let message = refused(plan.pca(None, &"p".repeat(57), 2, 0));
    assert!(message.contains(&format!("the embedding name '{}_loadings'", "p".repeat(57))));
    plan.pca(None, &"p".repeat(55), 2, 0).unwrap();
    assert!(refused(plan.pca(None, "x_PCA", 2, 0)).contains("'X_pca'"));
    // Cell and gene embeddings share one group.
    assert!(refused(plan.pca(None, "pcs", 2, 0)).contains("embedding 'pcs' differs"));
    let graphs = scan().neighbors(neighbors(None)).unwrap();
    assert!(refused(graphs.neighbors(neighbors(Some("a b")))).contains("'a b_distances'"));
    assert!(refused(scan().neighbors(neighbors(Some("")))).contains("never empty"));
    let keyed = scan().neighbors(neighbors(Some("umap"))).unwrap();
    let message = refused(keyed.neighbors(neighbors(Some("UMAP"))));
    assert!(
        message.contains("graph 'UMAP_distances' differs from the plan's graph 'umap_distances'")
    );
}

#[test]
fn new_column_names_follow_the_column_rules() {
    let plan = layered();
    let score = CellExpr::column("score");
    let message = refused(plan.with_cells([("meta:x".to_owned(), score.clone())]));
    assert!(message.contains("with_cells(..., name='meta:x') is invalid: ':' is not allowed"));
    assert!(refused(plan.with_cells([(String::new(), score.clone())])).contains("never empty"));
    assert!(refused(plan.with_cells([("cell_id".to_owned(), score.clone())])).contains("reserved"));
    // Either ID name in either table: the store writer refuses both (format.md §5.2).
    assert!(refused(plan.with_cells([("gene_id".to_owned(), score.clone())])).contains("reserved"));
    let hvg = GeneExpr::column("highly_variable");
    assert!(refused(plan.with_genes([("symbol".to_owned(), hvg.clone())])).contains("reserved"));
    assert!(
        refused(plan.with_cells([("x".to_owned(), score.clone()), ("x".to_owned(), score)]))
            .contains("written twice")
    );
    let graphs = plan.neighbors(neighbors(None)).unwrap();
    assert!(refused(graphs.cluster(leiden("leiden:1"))).contains("leiden(..., into='leiden:1')"));
    assert!(refused(graphs.cluster(leiden("cell_id"))).contains("reserved"));
    // Column names are free text otherwise (format.md §5.6), unlike member names.
    graphs.cluster(leiden("leiden res 1.0/é")).unwrap();
    plan.highly_variable_genes(None, HvgFlavor::Seurat, None, None)
        .unwrap();
}
