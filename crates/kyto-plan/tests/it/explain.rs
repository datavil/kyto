//! Task 4.7 Done when: snapshot tests cover text and JSON for 5 reference pipelines.
//!
//! Each reference pipeline's `explain()` text and `explain_json()` JSON is compared
//! with its snapshot in `tests/snapshots/explain/` (`<name>.txt`, `<name>.json`). To
//! update them on purpose (say so in the commit message), run
//! `KYTO_UPDATE_SNAPSHOTS=1 uv run cargo test -p kyto-plan --test it explain::` and read the
//! diff. Then: the text and the JSON say the same, the optimized section is the
//! physical plan's own text, short names are shown resolved, and nothing changes the
//! plan.
#![allow(clippy::unwrap_used)]

use crate::support;

use std::path::PathBuf;
use std::sync::Arc;

use kyto_core::{BinaryOp, CellExpr, Expr, GeneExpr, Literal};
use kyto_plan::{
    Cluster, ClusterMethod, HvgFlavor, Metric, NeighborSearch, Neighbors, PhysicalOptions, Plan,
    Sink,
};
use support::scan;

fn gt_cell(expr: CellExpr, value: f64) -> CellExpr {
    Expr::binary(expr, BinaryOp::Gt, Literal::Float(value)).unwrap()
}

fn gt_gene(expr: GeneExpr, value: f64) -> GeneExpr {
    Expr::binary(expr, BinaryOp::Gt, Literal::Float(value)).unwrap()
}

/// The five reference pipelines.
fn pipelines() -> Vec<(&'static str, Arc<Plan>)> {
    let base = scan();
    // 1. The scan alone.
    let scan_only = Arc::clone(&base);
    // 2. Quality control on metadata and data, with short names resolved: a default
    // layer, a gene filter on a column, a cell filter reading a gene through `feature`.
    let qc = base
        .with_default_layer("counts")
        .unwrap()
        .filter_genes(&gt_gene(GeneExpr::column("mean"), 0.5))
        .unwrap()
        .filter_cells(&gt_cell(CellExpr::feature("G1").unwrap(), 0.0))
        .unwrap();
    // 3. Standard preprocessing: normalize and log in one fused pass, HVG
    // (`seurat_v3`, 2 statistics passes), scale, PCA.
    let preprocessing = base
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "logn")
        .unwrap()
        .highly_variable_genes(Some("counts"), HvgFlavor::SeuratV3, Some(3), Some("batch"))
        .unwrap()
        .scale(Some("logn"), "scaled", true, Some(10.0))
        .unwrap()
        .pca(Some("scaled"), "X_new", 2, 0)
        .unwrap();
    // 4. Clustering: median normalization fused with log1p (a statistics pass), a
    // cell filter that stays above it (pushdown stops at cross-cell statistics), the
    // neighbor graph and Leiden in memory.
    let clustering = base
        .normalize_total(None, Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "logn")
        .unwrap()
        .filter_cells(&gt_cell(CellExpr::column("score"), -2.0))
        .unwrap()
        .pca(Some("logn"), "X_new", 2, 0)
        .unwrap()
        .neighbors(Neighbors {
            embedding: "X_new".to_owned(),
            n_components: None,
            n_neighbors: 15,
            metric: Metric::Euclidean,
            search: NeighborSearch::Exact,
            seed: 0,
            key: None,
        })
        .unwrap()
        .cluster(Cluster {
            method: ClusterMethod::Leiden,
            graph: "connectivities".to_owned(),
            resolution: 1.0,
            seed: 0,
            into: "leiden".to_owned(),
        })
        .unwrap();
    // 5. A subset written out: genes and cells chosen by ID, columns computed from the
    // data and metadata, a sink (reads everything).
    let subset = base
        .select_genes(["G3".to_owned(), "G1".to_owned()])
        .unwrap()
        .take_cells(["c2".to_owned(), "c1".to_owned()])
        .unwrap()
        .with_cells([(
            "g1".to_owned(),
            CellExpr::expression_by_symbol(Some("counts".to_owned()), "S1"),
        )])
        .unwrap()
        .with_genes([("in_c1".to_owned(), GeneExpr::column("mean"))])
        .unwrap()
        .sink(Sink::H5ad {
            path: "subset.h5ad".to_owned(),
        })
        .unwrap();
    vec![
        ("scan", scan_only),
        ("qc", qc),
        ("preprocessing", preprocessing),
        ("clustering", clustering),
        ("subset", subset),
    ]
}

fn snapshot_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/snapshots/explain")
        .join(name)
}

/// Compare `actual` with the snapshot file `name`, or write it when
/// `KYTO_UPDATE_SNAPSHOTS` is set.
fn check_snapshot(name: &str, actual: &str) {
    let path = snapshot_path(name);
    if std::env::var_os("KYTO_UPDATE_SNAPSHOTS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, format!("{actual}\n")).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "no snapshot {} ({error}); write it with KYTO_UPDATE_SNAPSHOTS=1",
            path.display()
        )
    });
    assert_eq!(
        actual,
        expected.strip_suffix('\n').unwrap_or(&expected),
        "{name} differs from its snapshot"
    );
}

#[test]
fn reference_pipelines_match_their_snapshots() {
    for (name, plan) in pipelines() {
        check_snapshot(&format!("{name}.txt"), &plan.explain());
        check_snapshot(&format!("{name}.json"), &plan.explain_json());
    }
}

/// The JSON says what the text says: every entry's values appear on its line, in order.
#[test]
fn text_and_json_say_the_same() {
    for (name, plan) in pipelines() {
        let json: serde_json::Value = serde_json::from_str(&plan.explain_json()).unwrap();
        let text = plan.explain();
        let lines: Vec<&str> = text.lines().collect();
        let logical = json["logical"].as_array().unwrap();
        let outputs = json["optimized"]["outputs"].as_array().unwrap();
        let nodes = json["optimized"]["nodes"].as_array().unwrap();
        assert_eq!(
            lines.len(),
            2 + logical.len() + outputs.len() + nodes.len(),
            "{name}"
        );
        assert_eq!(lines[0], "logical plan", "{name}");
        assert_eq!(lines[1 + logical.len()], "optimized plan", "{name}");
        for (entry, line) in logical.iter().zip(&lines[1..]) {
            let start = format!(
                "  node {} {} {}",
                entry["id"],
                entry["name"].as_str().unwrap(),
                entry["parameters"].as_str().unwrap()
            );
            assert!(line.starts_with(&start), "{name}: {line:?} vs {start:?}");
        }
        for (entry, line) in nodes
            .iter()
            .zip(&lines[2 + logical.len() + outputs.len()..])
        {
            let start = format!(
                "  #{} {} {}",
                entry["index"],
                entry["name"].as_str().unwrap(),
                entry["parameters"].as_str().unwrap()
            );
            assert!(line.starts_with(&start), "{name}: {line:?} vs {start:?}");
            let passes = format!("[device=cpu passes={}", entry["passes"]);
            assert!(line.contains(&passes), "{name}: {line:?}");
        }
    }
}

/// The optimized section is the physical plan's own text, indented.
#[test]
fn optimized_section_is_the_physical_plan() {
    for (name, plan) in pipelines() {
        let text = plan.explain();
        let (_, optimized) = text.split_once("\noptimized plan\n").unwrap();
        let physical = plan.physical(PhysicalOptions::default()).to_string();
        let indented: Vec<String> = physical.lines().map(|line| format!("  {line}")).collect();
        assert_eq!(optimized, indented.join("\n"), "{name}");
    }
}

/// The logical section lists every logical node, from the last down to the scan,
/// with the plan's own `Display` after its ID.
#[test]
fn logical_section_is_every_node() {
    for (name, plan) in pipelines() {
        let explanation = plan.explanation();
        let shown: Vec<String> = explanation
            .logical()
            .iter()
            .map(|entry| format!("{} {}", entry.name, entry.parameters))
            .collect();
        let display: Vec<String> = plan.to_string().lines().map(str::to_owned).collect();
        assert_eq!(shown, display, "{name}");
        let ids: Vec<u64> = explanation.logical().iter().map(|entry| entry.id).collect();
        let expected: Vec<u64> = (0..ids.len() as u64).rev().collect();
        assert_eq!(ids, expected, "{name}");
    }
}

/// Short names are shown as the plan stored them, in full (R6).
#[test]
fn names_are_resolved() {
    let pipelines = pipelines();
    let qc = &pipelines[1].1.explain();
    assert!(
        qc.contains("gene_expression('G1', layer='counts') > 0.0"),
        "{qc}"
    );
    assert!(!qc.contains("feature("), "{qc}");
    let subset = &pipelines[4].1.explain();
    // Read by its symbol `S1`, stored by its ID.
    assert!(
        subset.contains("gene_expression('G1', layer='counts')"),
        "{subset}"
    );
    assert!(!subset.contains("S1"), "{subset}");
}

#[test]
fn explain_leaves_the_plan_unchanged() {
    for (name, plan) in pipelines() {
        let before = plan.to_json();
        assert_eq!(plan.explain(), plan.explain(), "{name}");
        assert_eq!(plan.explain_json(), plan.explain_json(), "{name}");
        assert_eq!(plan.to_json(), before, "{name}");
    }
}

/// A centred `scale` carries a note that it densifies and what to do instead (task
/// 6.4); an uncentred one, and every other step, none.
#[test]
fn a_centred_scale_notes_that_it_densifies() {
    let centred = scan().scale(Some("counts"), "scaled", true, None).unwrap();
    let explanation = centred.explanation();
    let note = explanation.logical()[0].note.as_deref().unwrap();
    assert!(note.starts_with("densifies"), "{note}");
    assert!(
        note.contains("pca") && note.contains("implicitly"),
        "{note}"
    );
    assert!(
        explanation.logical()[1..]
            .iter()
            .all(|entry| entry.note.is_none())
    );
    let text = centred.explain();
    assert!(
        text.lines()
            .nth(1)
            .unwrap()
            .ends_with(&format!(" [{note}]")),
        "{text}"
    );
    let json: serde_json::Value = serde_json::from_str(&centred.explain_json()).unwrap();
    assert_eq!(json["logical"][0]["note"].as_str(), Some(note));
    assert!(json["logical"][1]["note"].is_null());

    let uncentred = scan().scale(Some("counts"), "scaled", false, None).unwrap();
    assert!(uncentred.explanation().logical()[0].note.is_none());
    assert!(!uncentred.explain().contains("densifies"));
}
