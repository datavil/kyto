//! Task 3.6: the read-only handles answer `names()`, `schema()` and `estimate()` from the
//! plan's schema and gene IDs, check the member they name (`KY-E109`), and share the
//! plan they were taken from without changing it (R2).
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use kyto_core::{
    AtlasSchema, Axis, ColumnDtype, ColumnSchema, GeneIds, LayerDtype, LayerSchema, ModalitySchema,
    TableSchema,
};
use kyto_plan::{EDGE_COLUMNS, Embedding, Estimate, Graph, Matrix, Modality, Plan, Table};

fn schema() -> AtlasSchema {
    serde_json::from_str(
        r#"{
  "n_cells": 4,
  "n_genes": 3,
  "tiling": {"cell_block_size": 4096, "gene_block_size": 1024},
  "cells": {"columns": [
    {"name": "cell_id", "dtype": "String"},
    {"name": "louvain", "dtype": "Enum", "n_categories": 3, "ordered": false}
  ]},
  "genes": {"columns": [{"name": "gene_id", "dtype": "String"}]},
  "layers": [
    {"name": "counts", "dtype": "uint32", "nnz": 5},
    {"name": "log1p", "dtype": "float32", "nnz": 5}
  ],
  "embeddings": [
    {"name": "X_pca", "axis": "cells", "width": 3, "dtype": "float64"},
    {"name": "loadings", "axis": "genes", "width": 2, "dtype": "float32"}
  ],
  "graphs": [{"name": "connectivities", "axis": "cells", "dtype": "float32", "nnz": 7}]
}"#,
    )
    .unwrap()
}

/// Gene IDs out of byte order, to show `Matrix::names` keeps table order.
fn plan() -> Arc<Plan> {
    Plan::scan(
        "pbmc.kyto",
        schema(),
        GeneIds::new(["MS4A1", "CD3E", "CD4"].map(str::to_owned)),
    )
}

#[test]
fn tables() {
    let plan = plan();
    let cells = Table::new(&plan, Axis::Cells);
    assert_eq!(cells.names(), ["cell_id", "louvain"]);
    assert_eq!(cells.schema(), &plan.schema().cells);
    assert_eq!(
        cells.estimate(),
        Estimate {
            n_rows: Some(4),
            n_columns: Some(2),
            nnz: None,
            dense_bytes: None
        }
    );
    let genes = Table::new(&plan, Axis::Genes);
    assert_eq!(genes.axis(), Axis::Genes);
    assert_eq!(genes.names(), ["gene_id"]);
    assert_eq!(genes.estimate().n_rows, Some(3));
}

#[test]
fn matrix() {
    let plan = plan();
    let matrix = Matrix::new(&plan, "log1p").unwrap();
    assert_eq!(matrix.names(), ["MS4A1", "CD3E", "CD4"]);
    assert_eq!(matrix.schema().dtype, LayerDtype::Float32);
    assert_eq!(
        matrix.estimate(),
        Estimate {
            n_rows: Some(4),
            n_columns: Some(3),
            nnz: Some(5),
            dense_bytes: Some(4 * 3 * 4)
        }
    );
}

#[test]
fn embeddings_follow_their_axis() {
    let plan = plan();
    let pca = Embedding::new(&plan, "X_pca").unwrap();
    assert_eq!(pca.names(), ["X_pca:0", "X_pca:1", "X_pca:2"]);
    assert_eq!(pca.estimate().n_rows, Some(4));
    assert_eq!(pca.estimate().dense_bytes, Some(4 * 3 * 8));
    let loadings = Embedding::new(&plan, "loadings").unwrap();
    assert_eq!(loadings.schema().axis, Axis::Genes);
    assert_eq!(
        loadings.estimate(),
        Estimate {
            n_rows: Some(3),
            n_columns: Some(2),
            nnz: None,
            dense_bytes: Some(3 * 2 * 4)
        }
    );
}

#[test]
fn graph() {
    let plan = plan();
    let graph = Graph::new(&plan, "connectivities").unwrap();
    assert_eq!(graph.names(), EDGE_COLUMNS);
    assert_eq!(graph.schema().nnz, 7);
    assert_eq!(
        graph.estimate(),
        Estimate {
            n_rows: Some(4),
            n_columns: Some(4),
            nnz: Some(7),
            dense_bytes: Some(4 * 4 * 4)
        }
    );
}

#[test]
fn modality() {
    let mut schema = schema();
    schema.modalities.push(ModalitySchema {
        name: "adt".to_owned(),
        n_features: 2,
        features: TableSchema {
            columns: vec![ColumnSchema::new("feature_id", ColumnDtype::String)],
        },
        layers: vec![LayerSchema {
            name: "counts".to_owned(),
            dtype: LayerDtype::Uint32,
            nnz: 6,
        }],
    });
    let plan = Plan::scan("pbmc.kyto", schema, GeneIds::default());
    let modality = Modality::new(&plan, "adt").unwrap();
    assert_eq!(modality.names(), ["counts"]);
    assert_eq!(modality.schema().n_features, 2);
    assert_eq!(modality.estimate().n_rows, Some(4));
    assert_eq!(modality.estimate().n_columns, Some(2));
}

#[test]
fn missing_members_are_ky_e109() {
    let plan = plan();
    let errors = [
        Matrix::new(&plan, "Counts").err().unwrap(),
        Embedding::new(&plan, "connectivities").err().unwrap(),
        Graph::new(&plan, "X_pca").err().unwrap(),
        Modality::new(&plan, "adt").err().unwrap(),
    ];
    let texts: Vec<String> = errors.iter().map(ToString::to_string).collect();
    assert_eq!(
        texts,
        [
            "KY-E109: no layer named 'Counts' in the store at 'pbmc.kyto'",
            "KY-E109: no embedding named 'connectivities' in the store at 'pbmc.kyto'",
            "KY-E109: no graph named 'X_pca' in the store at 'pbmc.kyto'",
            "KY-E109: no modality named 'adt' in the store at 'pbmc.kyto'",
        ]
    );
    assert!(errors.iter().all(|error| error.code() == "KY-E109"));
}

#[test]
fn handles_share_the_plan_and_leave_it_unchanged() {
    let plan = plan().with_default_layer("counts").unwrap();
    let before = plan.to_string();
    let matrix = Matrix::new(&plan, "counts").unwrap();
    let cells = Table::new(&plan, Axis::Cells);
    assert!(Arc::ptr_eq(matrix.plan(), &plan));
    assert!(Arc::ptr_eq(cells.plan(), &plan));
    assert_eq!(plan.to_string(), before);
    assert_eq!(Arc::strong_count(&plan), 3);
    drop((matrix, cells));
    assert_eq!(Arc::strong_count(&plan), 1);
}

#[test]
fn dense_bytes_never_overflow() {
    let mut schema = schema();
    schema.n_cells = u64::MAX;
    schema.n_genes = u64::MAX;
    let plan = Plan::scan("huge.kyto", schema, GeneIds::default());
    // u64::MAX squared fits a u128; times 4 bytes it does not, and saturates.
    let expected = u128::MAX;
    assert_eq!(
        Matrix::new(&plan, "counts").unwrap().estimate().dense_bytes,
        Some(expected)
    );
    assert_eq!(
        Graph::new(&plan, "connectivities")
            .unwrap()
            .estimate()
            .dense_bytes,
        Some(expected)
    );
}
