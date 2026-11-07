//! `schema.json` through `kyto_core::AtlasSchema` (task 3.1): the schema read from a
//! store equals the schema written, field by field, and serializing it gives the stored
//! file back byte for byte. Imported stores are checked the same way in
//! `import_parity.rs`.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used)]

use kyto_core::{
    AtlasSchema, ColumnDtype, ColumnSchema, LayerDtype, LayerSchema, TableSchema, Tiling,
};
use kyto_store::{
    CellBlock, LayerSpec, LayerWriteOptions, StoreWriter, TableAxis, ValueDtype, read_atlas_schema,
};
use polars::prelude::{
    Column, DataFrame, DataType, FrozenCategories, IntoColumn, NamedFrom, Series,
};

const TILING: Tiling = Tiling {
    cell_block_size: 2,
    gene_block_size: 2,
};

fn enum_column(name: &str, categories: &[&str], values: &[Option<&str>]) -> Column {
    let frozen = FrozenCategories::new(categories.iter().copied()).unwrap();
    Series::new(name.into(), values)
        .strict_cast(&DataType::from_frozen_categories(frozen))
        .unwrap()
        .into_column()
}

#[test]
fn schema_read_from_a_store_equals_the_schema_written() {
    let genes = DataFrame::new(
        3,
        vec![
            Series::new("gene_id".into(), ["g0", "g1", "g2"]).into_column(),
            enum_column("symbol", &["B", "A"], &[Some("A"), None, Some("B")]),
        ],
    )
    .unwrap();
    let cells = DataFrame::new(
        3,
        vec![
            Series::new("cell_id".into(), ["c0", "c1", "c2"]).into_column(),
            enum_column(
                "stage",
                &["early", "late", "unused"],
                &[Some("late"), None, Some("early")],
            ),
            Series::new("plain".into(), ["x", "y", "x"]).into_column(),
            Series::new("score".into(), [0.5_f64, f64::NAN, -1.0]).into_column(),
            Series::new("n".into(), [Some(1_i16), None, Some(3)]).into_column(),
        ],
    )
    .unwrap();
    // Given out of name order: `schema.json` sorts layers by name.
    let specs = [
        LayerSpec {
            name: "scaled".to_owned(),
            dtype: ValueDtype::F32,
            options: LayerWriteOptions::default(),
        },
        LayerSpec {
            name: "counts".to_owned(),
            dtype: ValueDtype::U32,
            options: LayerWriteOptions::default(),
        },
    ];
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("atlas.kyto");
    let mut writer = StoreWriter::create(&target, &genes, &specs, TILING, false).unwrap();
    writer.mark_ordered(TableAxis::Cells, "stage");
    let (indptr, indices) = (vec![0, 2, 2, 3], vec![0, 2, 1]);
    writer
        .push(&[
            CellBlock::from_f32(indptr.clone(), indices.clone(), vec![0.5, -1.0, 2.0], 3).unwrap(),
            CellBlock::from_u32(indptr, indices, vec![1, 4, 9], 3).unwrap(),
        ])
        .unwrap();
    writer.finish(&cells, None).unwrap();

    let enumeration = |n_categories, ordered| ColumnDtype::Enum {
        n_categories,
        ordered,
    };
    let written = AtlasSchema {
        n_cells: 3,
        n_genes: 3,
        tiling: TILING,
        cells: TableSchema {
            columns: vec![
                ColumnSchema::new("cell_id", ColumnDtype::String),
                ColumnSchema::new("stage", enumeration(3, true)),
                ColumnSchema::new("plain", enumeration(2, false)),
                ColumnSchema::new("score", ColumnDtype::Float64),
                ColumnSchema::new("n", ColumnDtype::Int16),
            ],
        },
        genes: TableSchema {
            columns: vec![
                ColumnSchema::new("gene_id", ColumnDtype::String),
                ColumnSchema::new("symbol", enumeration(2, false)),
            ],
        },
        layers: vec![
            LayerSchema {
                name: "counts".to_owned(),
                dtype: LayerDtype::Uint32,
                nnz: 3,
            },
            LayerSchema {
                name: "scaled".to_owned(),
                dtype: LayerDtype::Float32,
                nnz: 3,
            },
        ],
        embeddings: vec![],
        graphs: vec![],
        modalities: vec![],
    };
    let read = read_atlas_schema(&target).unwrap();
    assert_eq!(read, written);

    let mut bytes = serde_json::to_vec_pretty(&read).unwrap();
    bytes.push(b'\n');
    assert_eq!(
        String::from_utf8(bytes).unwrap(),
        std::fs::read_to_string(target.join("schema.json")).unwrap()
    );
}

#[test]
fn bad_tiling_and_missing_store_are_coded() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(read_atlas_schema(dir.path()).unwrap_err().code(), "KY-E110");
    let text = r#"{"n_cells": 1, "n_genes": 1,
        "tiling": {"cell_block_size": 0, "gene_block_size": 1},
        "cells": {"columns": []}, "genes": {"columns": []},
        "layers": [], "embeddings": [], "graphs": []}"#;
    std::fs::write(dir.path().join("schema.json"), text).unwrap();
    assert_eq!(read_atlas_schema(dir.path()).unwrap_err().code(), "KY-E104");
}
