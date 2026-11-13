//! Task 8.4: the published JSON Schema of the plan JSON and `validate_plan_json`.
//!
//! Done when: deliberately broken plans each return the expected diagnostic (code and
//! JSON Pointer path). Also: a valid plan has none, the schema is a valid draft 2020-12
//! schema naming every method and expression `op`, every diagnostic agrees with
//! `from_json` (no diagnostics exactly when it reads the text; the first has its code),
//! and shape problems are all reported, not only the first.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use kyto_core::{BinaryOp, CellExpr, Expr, GeneIds, Literal};
use kyto_plan::{
    Cluster, ClusterMethod, Metric, NeighborSearch, Neighbors, PLAN_JSON_SCHEMA, Plan,
    PlanDiagnostic, Sink, validate_plan_json,
};
use serde_json::{Value, json};

const SCHEMA: &str = r#"{
  "n_cells": 4,
  "n_genes": 3,
  "tiling": {"cell_block_size": 4096, "gene_block_size": 1024},
  "cells": {"columns": [
    {"name": "cell_id", "dtype": "String"},
    {"name": "total", "dtype": "Float32"},
    {"name": "batch", "dtype": "Enum", "n_categories": 2, "ordered": false}
  ]},
  "genes": {"columns": [{"name": "gene_id", "dtype": "String"}]},
  "layers": [{"name": "counts", "dtype": "uint32", "nnz": 7}],
  "embeddings": [],
  "graphs": []
}"#;

/// Nodes: 0 scan, 1 with_default_layer, 2 filter_cells (`total > 1.0`: entries column,
/// float, binary), 3 log1p, 4 pca, 5 neighbors, 6 leiden, 7 sink_zarr.
fn plan() -> Arc<Plan> {
    let ids = GeneIds::new(["G1", "G2", "G3"].map(str::to_owned));
    let predicate: CellExpr =
        Expr::binary(CellExpr::column("total"), BinaryOp::Gt, Literal::Float(1.0)).unwrap();
    Plan::scan("pbmc.kyto", serde_json::from_str(SCHEMA).unwrap(), ids)
        .with_default_layer("counts")
        .unwrap()
        .filter_cells(&predicate)
        .unwrap()
        .log1p(None, "log1p")
        .unwrap()
        .pca(Some("log1p"), "X_pca", 2, 0)
        .unwrap()
        .neighbors(Neighbors {
            embedding: "X_pca".to_owned(),
            n_components: None,
            n_neighbors: 3,
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
        .unwrap()
        .sink(Sink::Zarr {
            path: "out.kyto".to_owned(),
        })
        .unwrap()
}

fn edited(edit: impl FnOnce(&mut Value)) -> String {
    let mut json: Value = serde_json::from_str(&plan().to_json()).unwrap();
    edit(&mut json);
    json.to_string()
}

fn step(json: &mut Value, node: usize) -> &mut Value {
    &mut json["nodes"][node]["step"]
}

/// `text` has a diagnostic with `code` at `path`; `from_json` refuses it with the
/// first diagnostic's code. Returns the diagnostics.
fn assert_diagnostic(text: &str, code: &str, path: &str) -> Vec<PlanDiagnostic> {
    let diagnostics = validate_plan_json(text);
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code() == code && diagnostic.path == path),
        "expected {code} at {path:?}, got {diagnostics:#?}"
    );
    let error = Plan::from_json(text).unwrap_err();
    assert_eq!(diagnostics[0].code(), error.code(), "{error}");
    for diagnostic in &diagnostics {
        assert_eq!(
            format!("{}: {}", diagnostic.code(), diagnostic.message()),
            diagnostic.error.to_string()
        );
        assert_eq!(diagnostic.hint(), diagnostic.error.hint());
        assert!(!diagnostic.hint().is_empty());
    }
    diagnostics
}

/// As [`assert_diagnostic`], and it is the only one.
fn assert_only(text: &str, code: &str, path: &str) -> PlanDiagnostic {
    let diagnostics = assert_diagnostic(text, code, path);
    assert_eq!(diagnostics.len(), 1, "{diagnostics:#?}");
    diagnostics.into_iter().next().unwrap()
}

#[test]
fn a_valid_plan_has_no_diagnostics() {
    let text = plan().to_json();
    assert_eq!(validate_plan_json(&text), Vec::new());
    // Pretty-printed and with keys reordered, it is still the same plan.
    let value: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        validate_plan_json(&serde_json::to_string_pretty(&value).unwrap()),
        Vec::new()
    );
}

#[test]
fn the_schema_is_a_draft_2020_12_schema_naming_every_method_and_op() {
    let schema: Value = serde_json::from_str(PLAN_JSON_SCHEMA).unwrap();
    assert_eq!(
        schema["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
    assert!(jsonschema::meta::is_valid(&schema));
    // The first node is a scan, every later node any other method.
    let mut methods: Vec<&str> = schema["$defs"]["later_node"]["allOf"][1]["properties"]
        ["step"]["properties"]["method"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .map(|method| method.as_str().unwrap())
        .collect();
    assert!(!methods.contains(&"scan"));
    assert_eq!(
        schema["$defs"]["scan_node"]["allOf"][1]["properties"]["step"]["properties"]["method"]["const"],
        "scan"
    );
    methods.push("scan");
    // Every method the plan JSON reads (4.3's `Step`), no more: an unknown one is refused
    // by both the schema and `from_json`.
    for method in &methods {
        let text = edited(|json| {
            *step(json, 1) = json!({"method": method});
        });
        let diagnostics = validate_plan_json(&text);
        assert!(
            diagnostics
                .iter()
                .all(|diagnostic| diagnostic.path != "/nodes/1/step/method" || *method == "scan"),
            "{method}: {diagnostics:#?}"
        );
    }
    assert_eq!(methods.len(), 20);
    let ops = schema["$defs"]["expr_entry"]["properties"]["op"]["enum"]
        .as_array()
        .unwrap();
    assert_eq!(ops.len(), 18);
}

#[test]
fn deliberately_broken_plans_each_return_the_expected_diagnostic() {
    // 1. Not JSON: one diagnostic for the whole text, naming the line and column.
    let diagnostic = assert_only(r#"{"format": "kyto-plan", "version": 1, "#, "KY-E217", "");
    assert!(
        diagnostic.message().contains("line 1 column"),
        "{diagnostic:?}"
    );

    // 2. Another version: only that, whatever the rest holds.
    assert_only(
        &edited(|json| {
            json["version"] = json!(2);
            json["nodes"] = json!("anything");
        }),
        "KY-E218",
        "/version",
    );

    // 3. A version that is not an integer.
    assert_only(
        &edited(|json| json["version"] = json!("1")),
        "KY-E217",
        "/version",
    );

    // 4. Another format.
    assert_only(
        &edited(|json| json["format"] = json!("kyto")),
        "KY-E217",
        "/format",
    );

    // 5. An unknown top-level key: reported on the object that holds it.
    let diagnostic = assert_only(
        &edited(|json| json["comment"] = json!("mine")),
        "KY-E217",
        "",
    );
    assert!(diagnostic.message().contains("comment"), "{diagnostic:?}");

    // 6. No nodes.
    assert_only(
        &edited(|json| json["nodes"] = json!([])),
        "KY-E217",
        "/nodes",
    );

    // 7. An unknown method.
    assert_only(
        &edited(|json| step(json, 3)["method"] = json!("log2p")),
        "KY-E217",
        "/nodes/3/step/method",
    );

    // 8. A missing argument: on the step that lacks it, naming it.
    let diagnostic = assert_only(
        &edited(|json| {
            step(json, 3).as_object_mut().unwrap().remove("into");
        }),
        "KY-E217",
        "/nodes/3/step",
    );
    assert!(diagnostic.message().contains("into"), "{diagnostic:?}");

    // 9. An argument of the wrong type.
    assert_only(
        &edited(|json| step(json, 4)["n_components"] = json!("2")),
        "KY-E217",
        "/nodes/4/step/n_components",
    );

    // 10. A negative seed.
    assert_only(
        &edited(|json| step(json, 4)["seed"] = json!(-1)),
        "KY-E217",
        "/nodes/4/step/seed",
    );

    // 11. A value outside an enumeration.
    assert_only(
        &edited(|json| step(json, 5)["metric"] = json!("manhattan")),
        "KY-E217",
        "/nodes/5/step/metric",
    );

    // 12. An unknown expression entry.
    assert_only(
        &edited(|json| step(json, 2)["predicate"][0]["op"] = json!("columns")),
        "KY-E217",
        "/nodes/2/step/predicate/0/op",
    );

    // 13. An unknown operator.
    assert_only(
        &edited(|json| step(json, 2)["predicate"][2]["operator"] = json!("**")),
        "KY-E217",
        "/nodes/2/step/predicate/2/operator",
    );

    // 14. A scan after the first node.
    assert_diagnostic(
        &edited(|json| {
            let scan = step(json, 0).clone();
            *step(json, 2) = scan;
        }),
        "KY-E217",
        "/nodes/2/step/method",
    );

    // 15. A tiling block size of 0.
    assert_only(
        &edited(|json| step(json, 0)["schema"]["tiling"]["cell_block_size"] = json!(0)),
        "KY-E217",
        "/nodes/0/step/schema/tiling/cell_block_size",
    );

    // 16. An `Enum` column without its category count.
    assert_only(
        &edited(|json| {
            step(json, 0)["schema"]["cells"]["columns"][2]
                .as_object_mut()
                .unwrap()
                .remove("n_categories");
        }),
        "KY-E217",
        "/nodes/0/step/schema/cells/columns/2",
    );

    // 17. Several shape problems: every one is reported, in one call.
    let diagnostics = assert_diagnostic(
        &edited(|json| {
            step(json, 3)["layer"] = json!(7);
            step(json, 6)["resolution"] = json!("high");
        }),
        "KY-E217",
        "/nodes/3/step/layer",
    );
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.path == "/nodes/6/step/resolution"),
        "{diagnostics:#?}"
    );

    // Past the shape, the replay: one diagnostic, the first step that does not build.

    // 18. IDs out of order.
    assert_only(
        &edited(|json| json["nodes"][2]["id"] = json!(5)),
        "KY-E217",
        "/nodes/2/id",
    );

    // 19. An input that is not the node before.
    assert_only(
        &edited(|json| json["nodes"][3]["input"] = json!(0)),
        "KY-E217",
        "/nodes/3/input",
    );

    // 20. Gene IDs that do not match the scan's schema.
    assert_only(
        &edited(|json| step(json, 0)["gene_ids"] = json!(["G1", "G2"])),
        "KY-E217",
        "/nodes/0/step/gene_ids",
    );

    // 21. A layer listed twice in the scan's schema.
    assert_only(
        &edited(|json| {
            let layer = step(json, 0)["schema"]["layers"][0].clone();
            step(json, 0)["schema"]["layers"]
                .as_array_mut()
                .unwrap()
                .push(layer);
        }),
        "KY-E217",
        "/nodes/0/step/schema/layers/1/name",
    );

    // 22. An operand written after its operation: pointed at the entry.
    assert_only(
        &edited(|json| step(json, 2)["predicate"][2]["left"] = json!(5)),
        "KY-E217",
        "/nodes/2/step/predicate/2",
    );

    // 23. Operands of kinds that do not mix (a string plus a float).
    assert_only(
        &edited(|json| {
            step(json, 2)["predicate"][0] = json!({"op": "string", "value": "a"});
            step(json, 2)["predicate"][2]["operator"] = json!("+");
        }),
        "KY-E205",
        "/nodes/2/step/predicate",
    );

    // 24. An unknown column in an expression.
    assert_only(
        &edited(|json| step(json, 2)["predicate"][0]["name"] = json!("totl")),
        "KY-E210",
        "/nodes/2/step",
    );

    // 25. A layer the plan does not have.
    assert_only(
        &edited(|json| step(json, 3)["layer"] = json!("spliced")),
        "KY-E109",
        "/nodes/3/step",
    );

    // 26. An argument value a method refuses: pointed at that argument.
    assert_only(
        &edited(|json| step(json, 4)["n_components"] = json!(0)),
        "KY-E215",
        "/nodes/4/step/n_components",
    );

    // 27. A step after the sink.
    assert_only(
        &edited(|json| {
            let log1p = json["nodes"][3].clone();
            let nodes = json["nodes"].as_array_mut().unwrap();
            let mut extra = log1p;
            extra["id"] = json!(8);
            extra["input"] = json!(7);
            extra["step"]["into"] = json!("again");
            nodes.push(extra);
        }),
        "KY-E216",
        "/nodes/8/step",
    );
}

#[test]
fn diagnostics_say_where_and_how_to_fix() {
    let diagnostic = assert_only(
        &edited(|json| step(json, 4)["seed"] = json!(-1)),
        "KY-E217",
        "/nodes/4/step/seed",
    );
    let message = diagnostic.message();
    assert!(message.contains("/nodes/4/step/seed"), "{message}");
    assert!(message.contains("-1"), "{message}");
    assert!(
        diagnostic.hint().contains("Atlas.to_json()"),
        "{}",
        diagnostic.hint()
    );
}

#[test]
fn expression_errors_point_at_their_entry() {
    // An entry of a QC metric's gene set, in a computed column.
    let scan = Plan::scan(
        "pbmc.kyto",
        serde_json::from_str(SCHEMA).unwrap(),
        GeneIds::new(["G1", "G2", "G3"].map(str::to_owned)),
    );
    let genes = kyto_core::GeneExpr::column("gene_id")
        .str(kyto_core::StrFunction::StartsWith("MT-".to_owned()))
        .unwrap();
    let metric = CellExpr::qc(
        Some("counts".to_owned()),
        kyto_core::QcMetric::PercentCountsIn { genes },
    )
    .unwrap();
    let plan = scan
        .with_cells([("percent_mt".to_owned(), metric)])
        .unwrap();
    let mut json: Value = serde_json::from_str(&plan.to_json()).unwrap();
    let qc = json["nodes"][1]["step"]["columns"][0]["expr"]
        .as_array()
        .unwrap()
        .iter()
        .position(|entry| entry["op"] == "qc")
        .unwrap();
    // `starts_with` reads an entry written after it.
    json["nodes"][1]["step"]["columns"][0]["expr"][qc]["genes"][1]["input"] = json!(9);
    assert_only(
        &json.to_string(),
        "KY-E217",
        &format!("/nodes/1/step/columns/0/expr/{qc}/genes/1"),
    );
}

#[test]
fn text_that_is_not_an_object_is_one_diagnostic() {
    for text in [
        "",
        "null",
        "[]",
        "42",
        "\"plan\"",
        "{\"format\": \"kyto-plan\"} trailing",
    ] {
        let diagnostics = validate_plan_json(text);
        assert_eq!(diagnostics.len(), 1, "{text}: {diagnostics:#?}");
        assert_eq!(diagnostics[0].code(), "KY-E217", "{text}");
        assert!(Plan::from_json(text).is_err(), "{text}");
    }
}

#[test]
fn bug_hunter_regressions() {
    // An empty expression after a step that does not build: the schema does not refuse
    // it (serde reads it), so the replay's earlier error comes first, as in `from_json`.
    let diagnostics = assert_diagnostic(
        &edited(|json| {
            step(json, 1)["layer"] = json!("nope");
            step(json, 2)["predicate"] = json!([]);
        }),
        "KY-E109",
        "/nodes/1/step",
    );
    assert_eq!(diagnostics.len(), 1, "{diagnostics:#?}");

    // A wrong value inside a nullable list is pointed at, and the message stays short.
    let mut symbols: Vec<Value> = vec![json!("A"), json!("B"), json!("C")];
    symbols[1] = json!(7);
    let diagnostic = assert_only(
        &edited(|json| step(json, 0)["symbols"] = Value::from(symbols)),
        "KY-E217",
        "/nodes/0/step/symbols/1",
    );
    assert!(diagnostic.message().len() < 200, "{}", diagnostic.message());
    assert_only(
        &edited(|json| json["nodes"][3]["input"] = json!("2")),
        "KY-E217",
        "/nodes/3/input",
    );

    // Diagnostics come in the order of the text.
    let diagnostics = validate_plan_json(&edited(|json| {
        json["comment"] = json!(1);
        step(json, 7)["path"] = json!(1);
        step(json, 2)["predicate"][0]["op"] = json!(1.5);
        step(json, 0)["source"] = json!(1);
    }));
    let paths: Vec<&str> = diagnostics
        .iter()
        .map(|diagnostic| diagnostic.path.as_str())
        .collect();
    assert_eq!(
        paths,
        [
            "",
            "/nodes/0/step/source",
            "/nodes/2/step/predicate/0/op",
            "/nodes/7/step/path"
        ],
        "{diagnostics:#?}"
    );
}
