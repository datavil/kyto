//! Task 4.1 Done when: every node has an output-schema function with unit tests.
//!
//! One section per node: its output schema (members, columns, exact sizes and bounds),
//! the axis it changes, per-cell or cross-cell, its `explain` line and its build-time
//! errors. Then what the cached node state feeds: removed genes, stale columns after a
//! cell filter or a layer rewrite, symbols after a selection, node IDs, long chains.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;
use std::time::Instant;

use kyto_core::{
    Axis, BinaryOp, CellExpr, ColumnDtype, ColumnSchema, FloatDtype, GeneExpr, GeneIds, LayerDtype,
    Literal, NullTest,
};
use kyto_plan::{
    CellScope, Cluster, ClusterMethod, CrossCellOp, ElementwiseOp, HvgFlavor, Metric,
    NeighborSearch, Neighbors, NodeId, Plan, PlanSchema, Sink, SparseMember, column_dtype,
    hvg_columns,
};

const SCHEMA: &str = r#"{
  "n_cells": 4,
  "n_genes": 3,
  "tiling": {"cell_block_size": 4096, "gene_block_size": 1024},
  "cells": {"columns": [
    {"name": "cell_id", "dtype": "String"},
    {"name": "n_genes", "dtype": "UInt32", "lineage": {"layer": "counts", "node": "n3"}},
    {"name": "total", "dtype": "Float32", "lineage": {"layer": "log1p", "node": "n4"}},
    {"name": "batch", "dtype": "Enum", "n_categories": 2, "ordered": false},
    {"name": "score", "dtype": "Int32"}
  ]},
  "genes": {"columns": [
    {"name": "gene_id", "dtype": "String"},
    {"name": "symbol", "dtype": "Enum", "n_categories": 2, "ordered": false},
    {"name": "highly_variable", "dtype": "Boolean"},
    {"name": "n_cells", "dtype": "UInt32", "lineage": {"layer": "counts", "node": "n5"}}
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
    {"name": "connectivities", "axis": "cells", "dtype": "float32", "nnz": 6},
    {"name": "coexpression", "axis": "genes", "dtype": "float32", "nnz": 2}
  ]
}"#;

/// A store of 4 cells and 3 genes; genes `G1` and `G2` share the symbol `TBCE`.
fn scan() -> Arc<Plan> {
    let gene_ids = GeneIds::new(["G1", "G2", "G3"].map(str::to_owned)).with_symbols([
        Some("TBCE".to_owned()),
        Some("TBCE".to_owned()),
        Some("CD3E".to_owned()),
    ]);
    Plan::scan("pbmc.kyto", serde_json::from_str(SCHEMA).unwrap(), gene_ids)
}

fn layered() -> Arc<Plan> {
    scan().with_default_layer("log1p").unwrap()
}

fn first_line(plan: &Plan) -> String {
    plan.to_string().lines().next().unwrap().to_owned()
}

fn hvg() -> GeneExpr {
    GeneExpr::column("highly_variable")
}

fn layer_nnz(schema: &PlanSchema, name: &str) -> Option<u64> {
    schema.nnz(&SparseMember::Layer(name.to_owned()))
}

fn graph_nnz(schema: &PlanSchema, name: &str) -> Option<u64> {
    schema.nnz(&SparseMember::Graph(name.to_owned()))
}

// --- Scan -----------------------------------------------------------------------

#[test]
fn scan_outputs_the_store_schema_with_exact_sizes() {
    let plan = scan();
    let schema = plan.output_schema();
    assert_eq!(*schema.atlas(), serde_json::from_str(SCHEMA).unwrap());
    assert_eq!((schema.n_cells(), schema.n_genes()), (Some(4), Some(3)));
    assert_eq!(layer_nnz(schema, "counts"), Some(7));
    assert_eq!(graph_nnz(schema, "connectivities"), Some(6));
    assert_eq!(plan.changes_axis(), None);
    assert_eq!(plan.cell_scope(), CellScope::PerCell);
    assert_eq!(plan.id(), NodeId(0));
    assert_eq!(plan.to_string(), "SCAN \"pbmc.kyto\" (4 cells x 3 genes)");
    assert_eq!(plan.genes(), ["G1", "G2", "G3"]);
}

// --- WithDefaultLayer ----------------------------------------------------------

#[test]
fn with_default_layer_keeps_the_schema() {
    let plan = layered();
    assert_eq!(plan.output_schema(), scan().output_schema());
    assert_eq!(plan.default_layer(), Some("log1p"));
    assert_eq!(plan.changes_axis(), None);
    assert_eq!(plan.cell_scope(), CellScope::PerCell);
    assert_eq!(plan.id(), NodeId(1));
    assert_eq!(first_line(&plan), "WITH_DEFAULT_LAYER \"log1p\"");
    assert_eq!(
        scan().with_default_layer("raw").unwrap_err().code(),
        "KY-E109"
    );
}

// --- Filter ---------------------------------------------------------------------

#[test]
fn filter_cells_bounds_the_cell_axis() {
    let predicate =
        CellExpr::binary(CellExpr::column("score"), BinaryOp::Gt, Literal::Int(1)).unwrap();
    let plan = scan().filter_cells(&predicate).unwrap();
    let schema = plan.output_schema();
    assert_eq!(schema.n_cells(), None);
    assert_eq!(schema.atlas().n_cells, 4, "the bound stays the input's");
    assert_eq!(schema.n_genes(), Some(3));
    assert_eq!(layer_nnz(schema, "counts"), None);
    assert_eq!(schema.atlas().layers[0].nnz, 7, "the bound stays");
    assert_eq!(
        graph_nnz(schema, "connectivities"),
        None,
        "a cell graph is filtered"
    );
    assert_eq!(
        graph_nnz(schema, "coexpression"),
        Some(2),
        "a gene graph is not"
    );
    assert_eq!(schema.atlas().cells, scan().schema().cells);
    assert_eq!(plan.changes_axis(), Some(Axis::Cells));
    assert_eq!(plan.cell_scope(), CellScope::PerCell);
    assert_eq!(first_line(&plan), "FILTER_CELLS cell_column('score') > 1");
    assert_eq!(plan.cell_axis_changed_by(), Some("filter_cells"));
    assert_eq!(plan.gene_axis_changed_by(), None);

    let numeric = scan().filter_cells(&CellExpr::column("score")).unwrap_err();
    assert_eq!(numeric.code(), "KY-E205");
    assert!(
        numeric
            .to_string()
            .contains("'filter_cells' needs a boolean cell expression")
    );
}

#[test]
fn filter_genes_bounds_the_gene_axis_and_reads_data_across_cells() {
    let plan = scan().filter_genes(&hvg()).unwrap();
    let schema = plan.output_schema();
    assert_eq!((schema.n_cells(), schema.n_genes()), (Some(4), None));
    assert_eq!(graph_nnz(schema, "connectivities"), Some(6));
    assert_eq!(graph_nnz(schema, "coexpression"), None);
    assert_eq!(plan.changes_axis(), Some(Axis::Genes));
    assert_eq!(plan.cell_scope(), CellScope::PerCell);
    assert!(
        plan.removed_genes().is_empty(),
        "known only when the plan runs"
    );

    // A gene predicate on one cell's profile needs that cell before any gene.
    let profile = GeneExpr::binary(
        GeneExpr::profile(Some("counts".to_owned()), "AAAC-1"),
        BinaryOp::Gt,
        Literal::Int(0),
    )
    .unwrap();
    let by_profile = scan().filter_genes(&profile).unwrap();
    assert_eq!(by_profile.cell_scope(), CellScope::CrossCell);
}

// --- SelectGenes ----------------------------------------------------------------

#[test]
fn select_genes_sets_the_genes_exactly_and_removes_the_others() {
    let plan = scan()
        .select_genes(["G3", "G1"].map(str::to_owned))
        .unwrap();
    let schema = plan.output_schema();
    assert_eq!(schema.n_genes(), Some(2));
    assert_eq!(schema.atlas().n_genes, 2);
    assert_eq!(layer_nnz(schema, "counts"), None);
    assert_eq!(plan.genes(), ["G3", "G1"], "in the order given");
    assert_eq!(plan.removed_genes().removed_by("G2"), Some("select_genes"));
    assert_eq!(plan.removed_genes().len(), 1);
    assert_eq!(plan.changes_axis(), Some(Axis::Genes));
    assert_eq!(plan.cell_scope(), CellScope::PerCell);
    assert_eq!(plan.gene_axis_changed_by(), Some("select_genes"));
    assert_eq!(first_line(&plan), "SELECT_GENES ['G3', 'G1']");

    // Reading a removed gene, or selecting it again, is KY-E203.
    let read = plan
        .resolve(&CellExpr::expression(Some("counts".to_owned()), "G2"))
        .unwrap_err();
    assert_eq!(read.code(), "KY-E203");
    assert_eq!(
        read.to_string(),
        "KY-E203: gene 'G2' was removed by 'select_genes' earlier in the plan"
    );
    let again = plan.select_genes(["G2".to_owned()]).unwrap_err();
    assert_eq!(again.code(), "KY-E203");
    // A second selection removes more, the first remover is kept.
    let narrower = plan.select_genes(["G1".to_owned()]).unwrap();
    assert_eq!(
        narrower.removed_genes().removed_by("G3"),
        Some("select_genes")
    );
    assert_eq!(narrower.removed_genes().len(), 2);
    assert_eq!(narrower.genes(), ["G1"]);
}

#[test]
fn select_genes_refuses_unknown_and_repeated_ids() {
    let unknown = scan().select_genes(["G9".to_owned()]).unwrap_err();
    assert_eq!(unknown.code(), "KY-E210");
    assert!(unknown.to_string().contains("unknown gene 'G9'"));
    let repeated = scan()
        .select_genes(["G1", "G1"].map(str::to_owned))
        .unwrap_err();
    assert_eq!(repeated.code(), "KY-E215");
    assert_eq!(
        repeated.to_string(),
        "KY-E215: select_genes(..., gene_ids=[..., 'G1', ...]) is invalid: 'G1' is listed \
         more than once"
    );
    let many = scan().select_genes(Vec::<String>::new()).unwrap();
    assert_eq!(many.output_schema().n_genes(), Some(0));
}

#[test]
fn explain_lists_at_most_five_ids() {
    let mut schema: serde_json::Value = serde_json::from_str(SCHEMA).unwrap();
    schema["n_cells"] = 100.into();
    let big = Plan::scan(
        "big.kyto",
        serde_json::from_value(schema).unwrap(),
        GeneIds::new(["G1".to_owned()]),
    );
    let ids: Vec<String> = (0..8).map(|i| format!("C{i}")).collect();
    let plan = big.take_cells(ids).unwrap();
    assert_eq!(
        first_line(&plan),
        "TAKE_CELLS ['C0', 'C1', 'C2', 'C3', 'C4', ... 3 more]"
    );
}

// --- TakeCells ------------------------------------------------------------------

#[test]
fn take_cells_sets_the_cells_exactly() {
    let plan = scan().take_cells(["c2", "c1"].map(str::to_owned)).unwrap();
    let schema = plan.output_schema();
    assert_eq!((schema.n_cells(), schema.n_genes()), (Some(2), Some(3)));
    assert_eq!(schema.atlas().n_cells, 2);
    assert_eq!(graph_nnz(schema, "connectivities"), None);
    assert_eq!(graph_nnz(schema, "coexpression"), Some(2));
    assert_eq!(plan.changes_axis(), Some(Axis::Cells));
    assert_eq!(plan.cell_scope(), CellScope::PerCell);
    assert_eq!(plan.cell_axis_changed_by(), Some("take_cells"));
    assert_eq!(
        scan().take_cells([String::new()]).unwrap_err().code(),
        "KY-E210"
    );
    assert_eq!(
        scan()
            .take_cells(["a", "a"].map(str::to_owned))
            .unwrap_err()
            .code(),
        "KY-E215"
    );
}

// --- MapElementwise -------------------------------------------------------------

#[test]
fn log1p_adds_a_float_layer_with_the_input_nnz() {
    let plan = scan().log1p(Some("counts"), "logged").unwrap();
    let schema = plan.output_schema();
    let layer = schema.layer("logged").unwrap();
    assert_eq!((layer.dtype, layer.nnz), (LayerDtype::Float32, 7));
    assert_eq!(layer_nnz(schema, "logged"), Some(7));
    assert_eq!(schema.atlas().layers.len(), 3);
    assert_eq!(plan.changes_axis(), None);
    assert_eq!(plan.cell_scope(), CellScope::PerCell);
    assert_eq!(first_line(&plan), "LOG1P \"counts\" -> \"logged\"");

    // After a filter the new layer's nnz is a bound too.
    let filtered = scan()
        .filter_cells(
            &CellExpr::column("batch")
                .null_test(NullTest::IsNull)
                .unwrap(),
        )
        .unwrap()
        .log1p(Some("counts"), "logged")
        .unwrap();
    assert_eq!(layer_nnz(filtered.output_schema(), "logged"), None);
}

#[test]
fn normalize_total_is_cross_cell_without_a_target() {
    let median = layered().normalize_total(None, None, "normalized").unwrap();
    assert_eq!(median.cell_scope(), CellScope::CrossCell);
    assert_eq!(
        first_line(&median),
        "NORMALIZE_TOTAL \"log1p\" -> \"normalized\" target_sum=median"
    );
    let fixed = scan()
        .normalize_total(Some(1e4), Some("counts"), "counts")
        .unwrap();
    assert_eq!(fixed.cell_scope(), CellScope::PerCell);
    let layer = fixed.output_schema().layer("counts").unwrap();
    assert_eq!(layer.dtype, LayerDtype::Float32, "replaced in place");
    assert_eq!(fixed.schema().layers[0].name, "counts");
    assert_eq!(fixed.schema().layers.len(), 2);

    assert_eq!(
        scan()
            .normalize_total(Some(0.0), Some("counts"), "x")
            .unwrap_err()
            .code(),
        "KY-E215"
    );
    assert_eq!(
        scan()
            .normalize_total(Some(f64::NAN), Some("counts"), "x")
            .unwrap_err()
            .code(),
        "KY-E215"
    );
    // Task 6.2: the target is used as a float32, so it must be a normal float32.
    for target_sum in [1e-50, 1e-40, 1e39, f64::MAX] {
        let error = scan()
            .normalize_total(Some(target_sum), Some("counts"), "x")
            .unwrap_err();
        assert_eq!(error.code(), "KY-E215", "{target_sum}");
        let message = error.to_string();
        assert!(message.contains("target_sum"), "{message}");
        assert!(message.contains("float32's normal range"), "{message}");
        assert!(error.hint().contains("1.1754944e-38"), "{}", error.hint());
    }
    for target_sum in [
        f64::from(f32::MIN_POSITIVE),
        1e-30,
        1e30,
        f64::from(f32::MAX),
    ] {
        assert!(
            scan()
                .normalize_total(Some(target_sum), Some("counts"), "x")
                .is_ok(),
            "{target_sum}"
        );
    }
    let no_default = scan().log1p(None, "x").unwrap_err();
    assert_eq!(no_default.code(), "KY-E211");
    assert!(no_default.to_string().contains("log1p(layer=None)"));
    assert_eq!(
        scan().log1p(Some("raw"), "x").unwrap_err().code(),
        "KY-E109"
    );
    assert_eq!(
        scan().log1p(Some("counts"), "").unwrap_err().code(),
        "KY-E215"
    );
}

// --- CrossCell ------------------------------------------------------------------

#[test]
fn scale_densifies_when_centred() {
    let centred = layered().scale(None, "scaled", true, Some(10.0)).unwrap();
    let schema = centred.output_schema();
    assert_eq!(schema.layer("scaled").unwrap().nnz, 12, "4 cells x 3 genes");
    assert_eq!(
        layer_nnz(schema, "scaled"),
        None,
        "a bound: a value at the mean is 0"
    );
    assert_eq!(centred.cell_scope(), CellScope::CrossCell);
    assert_eq!(centred.changes_axis(), None);
    assert_eq!(
        first_line(&centred),
        "SCALE \"log1p\" -> \"scaled\" zero_center=true max_value=10"
    );
    let sparse = layered().scale(None, "scaled", false, None).unwrap();
    assert_eq!(layer_nnz(sparse.output_schema(), "scaled"), Some(7));
    let filtered = layered()
        .filter_genes(&hvg())
        .unwrap()
        .scale(None, "scaled", true, None)
        .unwrap();
    assert_eq!(layer_nnz(filtered.output_schema(), "scaled"), None);
    assert_eq!(filtered.schema().layers[2].nnz, 12, "the bound");
    assert_eq!(
        layered()
            .scale(None, "s", true, Some(-1.0))
            .unwrap_err()
            .code(),
        "KY-E215"
    );
}

#[test]
fn highly_variable_genes_adds_the_flavours_columns() {
    let plan = layered()
        .highly_variable_genes(None, HvgFlavor::Seurat, Some(2), None)
        .unwrap();
    let names: Vec<&str> = plan
        .schema()
        .genes
        .columns
        .iter()
        .map(|column| column.name.as_str())
        .collect();
    assert_eq!(
        names,
        [
            "gene_id",
            "symbol",
            "highly_variable",
            "n_cells",
            "means",
            "dispersions",
            "dispersions_norm"
        ],
        "highly_variable is replaced in place, the others appended"
    );
    let replaced = plan
        .output_schema()
        .column(Axis::Genes, "highly_variable")
        .unwrap();
    assert_eq!(
        *replaced,
        ColumnSchema::new("highly_variable", ColumnDtype::Boolean)
    );
    assert_eq!(plan.cell_scope(), CellScope::CrossCell);
    assert_eq!(plan.changes_axis(), None);
    assert_eq!(plan.output_schema().n_genes(), Some(3));
    assert_eq!(
        first_line(&plan),
        "HIGHLY_VARIABLE_GENES \"log1p\" flavor=seurat n_top_genes=2"
    );

    let batched = layered()
        .highly_variable_genes(Some("counts"), HvgFlavor::SeuratV3, Some(2), Some("batch"))
        .unwrap();
    let column = batched
        .output_schema()
        .column(Axis::Genes, "highly_variable_nbatches")
        .unwrap();
    assert_eq!(column.dtype, ColumnDtype::Int64);
    assert!(
        batched
            .output_schema()
            .column(Axis::Genes, "highly_variable_intersection")
            .is_none()
    );
    assert_eq!(
        hvg_columns(HvgFlavor::CellRanger, true)
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        [
            "highly_variable",
            "means",
            "dispersions",
            "dispersions_norm",
            "highly_variable_nbatches",
            "highly_variable_intersection"
        ]
    );
    assert_eq!(
        hvg_columns(HvgFlavor::SeuratV3, false)
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        [
            "highly_variable",
            "highly_variable_rank",
            "means",
            "variances",
            "variances_norm"
        ]
    );

    let unknown = layered()
        .highly_variable_genes(None, HvgFlavor::Seurat, None, Some("donor"))
        .unwrap_err();
    assert_eq!(unknown.code(), "KY-E210");
    let zero = layered()
        .highly_variable_genes(None, HvgFlavor::Seurat, Some(0), None)
        .unwrap_err();
    assert_eq!(zero.code(), "KY-E215");
}

#[test]
fn pca_adds_a_cell_embedding_and_gene_loadings() {
    let plan = layered().pca(None, "X_pca", 2, 0).unwrap();
    let schema = plan.output_schema();
    let cells = schema.embedding("X_pca").unwrap();
    assert_eq!(
        (cells.axis, cells.width, cells.dtype),
        (Axis::Cells, 2, FloatDtype::Float32)
    );
    assert_eq!(
        schema.atlas().embeddings[0].name,
        "X_pca",
        "replaced in place"
    );
    let loadings = schema.embedding("X_pca_loadings").unwrap();
    assert_eq!((loadings.axis, loadings.width), (Axis::Genes, 2));
    assert_eq!(schema.atlas().embeddings.len(), 3);
    assert_eq!(plan.cell_scope(), CellScope::CrossCell);
    assert_eq!(plan.changes_axis(), None);
    assert_eq!(
        first_line(&plan),
        "PCA \"log1p\" -> \"X_pca\" n_components=2 seed=0"
    );

    for n_components in [0, 4] {
        let error = layered().pca(None, "X_pca", n_components, 0).unwrap_err();
        assert_eq!(error.code(), "KY-E215");
    }
    assert!(
        layered()
            .pca(None, "X_pca", 4, 0)
            .unwrap_err()
            .to_string()
            .contains("the plan has at most 3 cells or genes")
    );
    // The PCA's embedding is readable downstream (R6 against the new schema).
    let component = plan.resolve(&CellExpr::embedding("X_pca", 1)).unwrap();
    assert_eq!(component.to_string(), "cell_embedding('X_pca', 1)");
}

// --- WithColumns ----------------------------------------------------------------

#[test]
fn with_cells_adds_and_replaces_columns_with_declared_dtypes() {
    let doubled =
        CellExpr::binary(CellExpr::column("score"), BinaryOp::Mul, Literal::Int(2)).unwrap();
    let ratio = CellExpr::binary(
        CellExpr::column("total"),
        BinaryOp::Div,
        Literal::Float(2.0),
    )
    .unwrap();
    let flag = CellExpr::binary(CellExpr::column("score"), BinaryOp::Gt, Literal::Int(1)).unwrap();
    let gene = CellExpr::expression(None, "G1");
    let plan = layered()
        .with_cells([
            ("score".to_owned(), doubled),
            ("ratio".to_owned(), ratio),
            ("flag".to_owned(), flag),
            ("g1".to_owned(), gene.clone()),
            ("batch_copy".to_owned(), CellExpr::column("batch")),
        ])
        .unwrap();
    let schema = plan.output_schema();
    let dtype = |name: &str| schema.column(Axis::Cells, name).unwrap().dtype;
    assert_eq!(dtype("score"), ColumnDtype::Int32, "an int literal adapts");
    assert_eq!(
        dtype("ratio"),
        ColumnDtype::Float32,
        "a float literal adapts"
    );
    assert_eq!(dtype("flag"), ColumnDtype::Boolean);
    assert_eq!(dtype("g1"), ColumnDtype::Float32, "log1p is float32");
    assert_eq!(
        dtype("batch_copy"),
        ColumnDtype::Enum {
            n_categories: 2,
            ordered: false
        }
    );
    assert_eq!(
        schema.atlas().cells.columns[4].name,
        "score",
        "replaced in place"
    );
    assert_eq!(schema.atlas().cells.columns.len(), 9);
    assert!(
        schema
            .atlas()
            .cells
            .columns
            .iter()
            .skip(5)
            .all(|c| c.lineage.is_none())
    );
    assert_eq!(plan.changes_axis(), None);
    assert_eq!(plan.cell_scope(), CellScope::PerCell);
    assert!(first_line(&plan).starts_with(
        "WITH_CELLS \"score\" = cell_column('score') * 2, \"ratio\" = cell_column('total') / 2.0"
    ));
    // The expressions are stored resolved (R6): the default layer is written in.
    let kyto_plan::Plan::WithColumns(node) = &*plan else {
        panic!("not a WithColumns node")
    };
    let kyto_plan::Columns::Cells(columns) = node.params().columns() else {
        panic!("not cell columns")
    };
    assert_eq!(
        columns[3].1.to_string(),
        "gene_expression('G1', layer='log1p')"
    );
    // A column it writes is read like any other.
    assert_eq!(
        plan.resolve(&CellExpr::column("ratio"))
            .unwrap()
            .to_string(),
        "cell_column('ratio')"
    );
}

#[test]
fn with_columns_refuses_bad_names_and_reads_its_input_only() {
    let score = || CellExpr::column("score");
    // Reserved names are per table: a cell column may be called `symbol`.
    assert!(scan().with_cells([("symbol".to_owned(), score())]).is_ok());
    assert_eq!(
        scan()
            .with_genes([("symbol".to_owned(), hvg())])
            .unwrap_err()
            .code(),
        "KY-E215"
    );
    for name in ["", "cell_id"] {
        let error = scan().with_cells([(name.to_owned(), score())]).unwrap_err();
        assert_eq!(error.code(), "KY-E215", "{name:?}");
    }
    let twice = scan()
        .with_cells([("a".to_owned(), score()), ("a".to_owned(), score())])
        .unwrap_err();
    assert!(twice.to_string().contains("'a' is written twice"));
    // Siblings are not visible to each other (Polars' with_columns).
    let sibling = scan()
        .with_cells([
            ("a".to_owned(), score()),
            ("b".to_owned(), CellExpr::column("a")),
        ])
        .unwrap_err();
    assert_eq!(sibling.code(), "KY-E210");
    assert_eq!(
        scan()
            .with_genes([("gene_id".to_owned(), hvg())])
            .unwrap_err()
            .code(),
        "KY-E215"
    );
}

#[test]
fn with_genes_is_cross_cell_when_it_reads_a_cell() {
    let plan = scan()
        .with_genes([("hv".to_owned(), hvg().not().unwrap())])
        .unwrap();
    assert_eq!(
        first_line(&plan),
        "WITH_GENES \"hv\" = ~gene_column('highly_variable')"
    );
    assert_eq!(plan.cell_scope(), CellScope::PerCell);
    assert_eq!(
        plan.output_schema()
            .column(Axis::Genes, "hv")
            .unwrap()
            .dtype,
        ColumnDtype::Boolean
    );
    let profile = scan()
        .with_genes([(
            "first_cell".to_owned(),
            GeneExpr::profile(Some("counts".to_owned()), "AAAC-1"),
        )])
        .unwrap();
    assert_eq!(profile.cell_scope(), CellScope::CrossCell);
    assert_eq!(
        profile
            .output_schema()
            .column(Axis::Genes, "first_cell")
            .unwrap()
            .dtype,
        ColumnDtype::UInt32
    );
}

#[test]
fn column_dtype_follows_the_declared_rules() {
    let schema = scan().output_schema().clone();
    let cell = |expr: CellExpr| column_dtype(&schema, Axis::Cells, expr.node());
    let binary = |left: CellExpr, op, right: Literal| CellExpr::binary(left, op, right).unwrap();
    let column = CellExpr::column;
    assert_eq!(
        cell(binary(column("score"), BinaryOp::Add, Literal::Float(0.5))),
        ColumnDtype::Float64
    );
    assert_eq!(
        cell(binary(column("total"), BinaryOp::Add, Literal::Float(0.5))),
        ColumnDtype::Float32
    );
    assert_eq!(
        cell(binary(column("total"), BinaryOp::Div, Literal::Int(2))),
        ColumnDtype::Float32
    );
    assert_eq!(
        cell(binary(column("total"), BinaryOp::Div, Literal::Float(2.0))),
        ColumnDtype::Float32
    );
    assert_eq!(
        cell(binary(column("score"), BinaryOp::Div, Literal::Int(2))),
        ColumnDtype::Float64
    );
    let total_half = CellExpr::binary(column("total"), BinaryOp::Div, column("total")).unwrap();
    assert_eq!(cell(total_half), ColumnDtype::Float32);
    let mixed_ints = CellExpr::binary(column("score"), BinaryOp::Add, column("n_genes")).unwrap();
    assert_eq!(cell(mixed_ints), ColumnDtype::Int64);
    let mixed = CellExpr::binary(column("score"), BinaryOp::Sub, column("total")).unwrap();
    assert_eq!(cell(mixed), ColumnDtype::Float64);
    assert_eq!(
        cell(column("batch").null_test(NullTest::IsNull).unwrap()),
        ColumnDtype::Boolean
    );
    assert_eq!(cell(CellExpr::embedding("X_pca", 0)), ColumnDtype::Float32);
}

// --- Neighbors ------------------------------------------------------------------

fn neighbors() -> Neighbors {
    Neighbors {
        embedding: "X_pca".to_owned(),
        n_components: Some(2),
        n_neighbors: 3,
        metric: Metric::Euclidean,
        search: NeighborSearch::Exact,
        seed: 0,
        key: None,
    }
}

#[test]
fn neighbors_adds_two_bounded_cell_graphs() {
    let plan = scan().neighbors(neighbors()).unwrap();
    let schema = plan.output_schema();
    let distances = schema.graph("distances").unwrap();
    assert_eq!(
        (distances.axis, distances.nnz),
        (Axis::Cells, 8),
        "4 cells x 2"
    );
    assert_eq!(schema.graph("connectivities").unwrap().nnz, 16);
    assert_eq!(
        schema.atlas().graphs[0].name,
        "connectivities",
        "replaced in place"
    );
    assert_eq!(graph_nnz(schema, "distances"), None);
    assert_eq!(graph_nnz(schema, "connectivities"), None);
    assert_eq!(graph_nnz(schema, "coexpression"), Some(2));
    assert_eq!(plan.cell_scope(), CellScope::CrossCell);
    assert_eq!(plan.changes_axis(), None);
    assert_eq!(
        first_line(&plan),
        "NEIGHBORS \"X_pca\"[:2] -> \"distances\", \"connectivities\" n_neighbors=3 \
         metric=Euclidean search=Exact seed=0"
    );
    let keyed = scan()
        .neighbors(Neighbors {
            key: Some("rna".to_owned()),
            ..neighbors()
        })
        .unwrap();
    assert!(keyed.output_schema().graph("rna_connectivities").is_some());
    assert!(keyed.output_schema().graph("rna_distances").is_some());
}

#[test]
fn neighbors_checks_the_embedding_and_its_parameters() {
    let gene_embedding = scan()
        .neighbors(Neighbors {
            embedding: "PCs".to_owned(),
            ..neighbors()
        })
        .unwrap_err();
    assert_eq!(gene_embedding.code(), "KY-E109");
    assert!(
        gene_embedding
            .to_string()
            .contains("no cell embedding named 'PCs'")
    );
    for bad in [
        Neighbors {
            n_neighbors: 1,
            ..neighbors()
        },
        Neighbors {
            n_components: Some(4),
            ..neighbors()
        },
        Neighbors {
            n_components: Some(0),
            ..neighbors()
        },
        Neighbors {
            key: Some(String::new()),
            ..neighbors()
        },
    ] {
        assert_eq!(scan().neighbors(bad).unwrap_err().code(), "KY-E215");
    }
    // An embedding made by the plan is found.
    let after_pca = layered()
        .pca(None, "X_new", 2, 1)
        .unwrap()
        .neighbors(Neighbors {
            embedding: "X_new".to_owned(),
            ..neighbors()
        });
    assert!(after_pca.is_ok());
}

// --- Cluster --------------------------------------------------------------------

fn leiden() -> Cluster {
    Cluster {
        method: ClusterMethod::Leiden,
        graph: "connectivities".to_owned(),
        resolution: 1.0,
        seed: 0,
        into: "leiden".to_owned(),
    }
}

#[test]
fn cluster_adds_a_string_cell_column() {
    let plan = scan().cluster(leiden()).unwrap();
    let column = plan.output_schema().column(Axis::Cells, "leiden").unwrap();
    assert_eq!(*column, ColumnSchema::new("leiden", ColumnDtype::String));
    assert_eq!(plan.cell_scope(), CellScope::CrossCell);
    assert_eq!(plan.changes_axis(), None);
    assert_eq!(
        first_line(&plan),
        "LEIDEN \"connectivities\" -> \"leiden\" resolution=1 seed=0"
    );
    let louvain = scan()
        .cluster(Cluster {
            method: ClusterMethod::Louvain,
            ..leiden()
        })
        .unwrap();
    assert_eq!(louvain.name(), "LOUVAIN");
    // Replacing a stored column drops its lineage: the plan computes it.
    let replaced = scan()
        .cluster(Cluster {
            into: "total".to_owned(),
            ..leiden()
        })
        .unwrap();
    assert!(replaced.resolve(&CellExpr::column("total")).is_ok());

    let missing = scan()
        .cluster(Cluster {
            graph: "coexpression".to_owned(),
            ..leiden()
        })
        .unwrap_err();
    assert_eq!(missing.code(), "KY-E109");
    for bad in [
        Cluster {
            resolution: 0.0,
            ..leiden()
        },
        Cluster {
            resolution: f64::INFINITY,
            ..leiden()
        },
        Cluster {
            into: String::new(),
            ..leiden()
        },
        Cluster {
            into: "cell_id".to_owned(),
            ..leiden()
        },
    ] {
        assert_eq!(scan().cluster(bad).unwrap_err().code(), "KY-E215");
    }
}

// --- Sink -----------------------------------------------------------------------

#[test]
fn sink_keeps_the_schema_and_ends_the_plan() {
    let input = layered().log1p(None, "again").unwrap();
    let plan = input
        .sink(Sink::Zarr {
            path: "out.kyto".to_owned(),
        })
        .unwrap();
    assert_eq!(plan.output_schema(), input.output_schema());
    assert_eq!(plan.cell_scope(), CellScope::PerCell);
    assert_eq!(plan.changes_axis(), None);
    assert_eq!(first_line(&plan), "SINK_ZARR \"out.kyto\"");
    assert_eq!(
        plan.normalize_total(Some(-1.0), Some("counts"), "n")
            .unwrap_err()
            .code(),
        "KY-E216",
        "checked before the arguments"
    );
    let ended = plan.log1p(None, "x").unwrap_err();
    assert_eq!(ended.code(), "KY-E216");
    assert_eq!(
        ended.to_string(),
        "KY-E216: 'log1p' cannot follow 'sink_zarr': a plan ends at its sink"
    );
    assert_eq!(
        plan.sink(Sink::H5ad {
            path: "out.h5ad".to_owned()
        })
        .unwrap_err()
        .code(),
        "KY-E216"
    );
    assert_eq!(
        plan.filter_genes(&hvg()).unwrap_err().code(),
        "KY-E216",
        "checked before the predicate is resolved"
    );
    assert_eq!(
        input
            .sink(Sink::H5ad {
                path: String::new()
            })
            .unwrap_err()
            .code(),
        "KY-E215"
    );
}

// --- What the cached state feeds -----------------------------------------------

#[test]
fn a_cell_filter_makes_stored_gene_columns_stale() {
    let filtered = scan()
        .filter_cells(
            &CellExpr::column("batch")
                .null_test(NullTest::IsNull)
                .unwrap(),
        )
        .unwrap();
    let error = filtered.resolve(&GeneExpr::column("n_cells")).unwrap_err();
    assert_eq!(error.code(), "KY-E301");
    assert_eq!(
        error.to_string(),
        "KY-E301: column 'n_cells' was computed before 'filter_cells' changed its inputs"
    );
    assert!(
        filtered
            .resolve(&GeneExpr::column("n_cells").recompute().unwrap())
            .is_ok()
    );
    // Gene columns without lineage, and cell columns, are read as stored.
    assert!(filtered.resolve(&hvg()).is_ok());
    assert!(filtered.resolve(&CellExpr::column("n_genes")).is_ok());
}

#[test]
fn rewriting_a_layer_makes_columns_computed_from_it_stale() {
    let rewritten = scan().log1p(Some("counts"), "counts").unwrap();
    for (error, column) in [
        (
            rewritten.resolve(&CellExpr::column("n_genes")).unwrap_err(),
            "n_genes",
        ),
        (
            rewritten.resolve(&GeneExpr::column("n_cells")).unwrap_err(),
            "n_cells",
        ),
    ] {
        assert_eq!(error.code(), "KY-E301");
        assert_eq!(
            error.to_string(),
            format!("KY-E301: column '{column}' was computed before 'log1p' changed its inputs")
        );
    }
    // `total` comes from log1p, which is not rewritten; a new layer stales nothing.
    assert!(rewritten.resolve(&CellExpr::column("total")).is_ok());
    let new_layer = scan().log1p(Some("counts"), "fresh").unwrap();
    assert!(new_layer.resolve(&CellExpr::column("n_genes")).is_ok());
    // Scale rewrites too, and the first rewriter is named.
    let scaled = rewritten.scale(Some("log1p"), "log1p", true, None).unwrap();
    let error = scaled.resolve(&CellExpr::column("total")).unwrap_err();
    assert!(error.to_string().contains("before 'scale'"));
    let error = scaled.resolve(&CellExpr::column("n_genes")).unwrap_err();
    assert!(error.to_string().contains("before 'log1p'"));
}

#[test]
fn a_symbol_shared_with_a_removed_gene_names_the_kept_one() {
    let by_symbol = CellExpr::expression_by_symbol(Some("counts".to_owned()), "TBCE");
    assert_eq!(scan().resolve(&by_symbol).unwrap_err().code(), "KY-E204");
    let kept = scan()
        .select_genes(["G2", "G3"].map(str::to_owned))
        .unwrap();
    assert_eq!(
        kept.resolve(&by_symbol).unwrap().to_string(),
        "gene_expression('G2', layer='counts')"
    );
    let neither = scan().select_genes(["G3".to_owned()]).unwrap();
    let error = neither.resolve(&by_symbol).unwrap_err();
    assert_eq!(error.code(), "KY-E203");
    assert!(
        error
            .to_string()
            .contains("gene 'G1' was removed by 'select_genes'")
    );
    // A filter on data removes nothing at build: the symbol stays ambiguous.
    let filtered = scan().filter_genes(&hvg()).unwrap();
    assert_eq!(filtered.resolve(&by_symbol).unwrap_err().code(), "KY-E204");
}

#[test]
fn node_ids_count_from_the_scan_and_ignore_what_is_built_on_top() {
    let base = layered().filter_genes(&hvg()).unwrap();
    let left = base.log1p(None, "a").unwrap();
    let right = base.pca(None, "X", 1, 0).unwrap();
    let ids: Vec<u64> = left.nodes().map(|node| node.id().0).collect();
    assert_eq!(ids, [3, 2, 1, 0]);
    assert_eq!(right.id(), NodeId(3));
    assert_eq!(base.id(), NodeId(2));
}

#[test]
fn every_node_kind_compares_by_its_parameters() {
    let a = layered().pca(None, "X", 2, 0).unwrap();
    assert_eq!(*a, *layered().pca(None, "X", 2, 0).unwrap());
    assert_ne!(*a, *layered().pca(None, "X", 2, 1).unwrap());
    assert_ne!(*a, *layered().scale(None, "X", true, None).unwrap());
    let b = scan().neighbors(neighbors()).unwrap();
    assert_eq!(*b, *scan().neighbors(neighbors()).unwrap());
    assert_ne!(
        *b,
        *scan()
            .neighbors(Neighbors {
                seed: 9,
                ..neighbors()
            })
            .unwrap()
    );
}

#[test]
fn parameters_round_trip_through_serde() {
    let op = CrossCellOp::Pca {
        layer: "log1p".to_owned(),
        into: "X_pca".to_owned(),
        n_components: 50,
        seed: 0,
    };
    let json = serde_json::to_string(&op).unwrap();
    assert_eq!(
        json,
        r#"{"op":"pca","layer":"log1p","into":"X_pca","n_components":50,"seed":0}"#
    );
    assert_eq!(serde_json::from_str::<CrossCellOp>(&json).unwrap(), op);
    let elementwise = ElementwiseOp::NormalizeTotal {
        target_sum: Some(1e4),
    };
    let json = serde_json::to_string(&elementwise).unwrap();
    assert_eq!(
        serde_json::from_str::<ElementwiseOp>(&json).unwrap(),
        elementwise
    );
    let neighbors = neighbors();
    let json = serde_json::to_string(&neighbors).unwrap();
    assert_eq!(serde_json::from_str::<Neighbors>(&json).unwrap(), neighbors);
    let schema = scan().select_genes(["G1".to_owned()]).unwrap();
    let json = serde_json::to_string(schema.output_schema()).unwrap();
    assert_eq!(
        serde_json::from_str::<PlanSchema>(&json).unwrap(),
        *schema.output_schema()
    );
}

#[test]
fn building_a_long_chain_is_linear() {
    // Task 3.8 measured 1.5 s for 3 000 chained gene filters, each walking the chain to
    // resolve its predicate; with the state cached per node, 20 000 take far less.
    let started = Instant::now();
    let mut plan = scan();
    for _ in 0..20_000 {
        plan = plan.filter_genes(&hvg()).unwrap();
    }
    assert_eq!(plan.id(), NodeId(20_000));
    assert!(
        started.elapsed().as_secs() < 10,
        "took {:?}",
        started.elapsed()
    );
}
