//! `AtlasSchema` JSON round trip (task 3.1): any schema serializes and parses back to
//! an equal value, and re-serializing gives the same bytes. Also checks the spec example
//! of format.md §8 parses and writes back byte for byte.

use kyto_core::{
    AtlasSchema, Axis, ColumnDtype, ColumnSchema, EmbeddingSchema, FloatDtype, GraphSchema,
    LayerDtype, LayerSchema, Lineage, ModalitySchema, TableSchema, Tiling,
};
use proptest::prelude::*;

fn name() -> impl Strategy<Value = String> {
    // Any Unicode, quotes and escapes included: JSON must carry them exactly.
    "\\PC{0,8}"
}

fn column_dtype() -> impl Strategy<Value = ColumnDtype> {
    prop_oneof![
        Just(ColumnDtype::Boolean),
        Just(ColumnDtype::Int8),
        Just(ColumnDtype::Int16),
        Just(ColumnDtype::Int32),
        Just(ColumnDtype::Int64),
        Just(ColumnDtype::UInt8),
        Just(ColumnDtype::UInt16),
        Just(ColumnDtype::UInt32),
        Just(ColumnDtype::UInt64),
        Just(ColumnDtype::Float32),
        Just(ColumnDtype::Float64),
        Just(ColumnDtype::String),
        (any::<u64>(), any::<bool>()).prop_map(|(n_categories, ordered)| ColumnDtype::Enum {
            n_categories,
            ordered
        }),
    ]
}

fn table() -> impl Strategy<Value = TableSchema> {
    let lineage =
        proptest::option::of((name(), name()).prop_map(|(layer, node)| Lineage { layer, node }));
    proptest::collection::vec(
        (name(), column_dtype(), lineage).prop_map(|(name, dtype, lineage)| ColumnSchema {
            name,
            dtype,
            lineage,
        }),
        0..4,
    )
    .prop_map(|columns| TableSchema { columns })
}

fn axis() -> impl Strategy<Value = Axis> {
    prop_oneof![Just(Axis::Cells), Just(Axis::Genes)]
}

fn float_dtype() -> impl Strategy<Value = FloatDtype> {
    prop_oneof![Just(FloatDtype::Float32), Just(FloatDtype::Float64)]
}

fn layers() -> impl Strategy<Value = Vec<LayerSchema>> {
    let dtype = prop_oneof![Just(LayerDtype::Float32), Just(LayerDtype::Uint32)];
    proptest::collection::vec(
        (name(), dtype, any::<u64>()).prop_map(|(name, dtype, nnz)| LayerSchema {
            name,
            dtype,
            nnz,
        }),
        0..3,
    )
}

fn schema() -> impl Strategy<Value = AtlasSchema> {
    let embeddings = proptest::collection::vec(
        (name(), axis(), any::<u64>(), float_dtype()).prop_map(|(name, axis, width, dtype)| {
            EmbeddingSchema {
                name,
                axis,
                width,
                dtype,
            }
        }),
        0..3,
    );
    let graphs = proptest::collection::vec(
        (name(), axis(), float_dtype(), any::<u64>()).prop_map(|(name, axis, dtype, nnz)| {
            GraphSchema {
                name,
                axis,
                dtype,
                nnz,
            }
        }),
        0..3,
    );
    let modalities = proptest::collection::vec(
        (name(), any::<u64>(), table(), layers()).prop_map(
            |(name, n_features, features, layers)| ModalitySchema {
                name,
                n_features,
                features,
                layers,
            },
        ),
        0..2,
    );
    (
        (any::<u64>(), any::<u64>(), any::<u64>(), any::<u64>()),
        (table(), table(), layers()),
        (embeddings, graphs, modalities),
    )
        .prop_map(
            |(
                (n_cells, n_genes, cell_block_size, gene_block_size),
                (cells, genes, layers),
                (embeddings, graphs, modalities),
            )| AtlasSchema {
                n_cells,
                n_genes,
                tiling: Tiling {
                    cell_block_size,
                    gene_block_size,
                },
                cells,
                genes,
                layers,
                embeddings,
                graphs,
                modalities,
            },
        )
}

proptest! {
    #[test]
    fn json_round_trip_is_exact(schema in schema()) {
        for text in [
            serde_json::to_string(&schema).unwrap(),
            serde_json::to_string_pretty(&schema).unwrap(),
        ] {
            let parsed: AtlasSchema = serde_json::from_str(&text).unwrap();
            prop_assert_eq!(&parsed, &schema);
            let again = if text.contains('\n') {
                serde_json::to_string_pretty(&parsed).unwrap()
            } else {
                serde_json::to_string(&parsed).unwrap()
            };
            prop_assert_eq!(again, text);
        }
    }
}

/// The example of format.md §8, pretty-printed as kyto-store writes it.
const SPEC_EXAMPLE: &str = r#"{
  "n_cells": 2700,
  "n_genes": 32738,
  "tiling": {
    "cell_block_size": 4096,
    "gene_block_size": 1024
  },
  "cells": {
    "columns": [
      {
        "name": "cell_id",
        "dtype": "String"
      },
      {
        "name": "louvain",
        "dtype": "Enum",
        "n_categories": 3,
        "ordered": false
      },
      {
        "name": "n_genes",
        "dtype": "UInt32",
        "lineage": {
          "layer": "counts",
          "node": "n3"
        }
      }
    ]
  },
  "genes": {
    "columns": [
      {
        "name": "gene_id",
        "dtype": "String"
      },
      {
        "name": "symbol",
        "dtype": "Enum",
        "n_categories": 32738,
        "ordered": false
      }
    ]
  },
  "layers": [
    {
      "name": "counts",
      "dtype": "uint32",
      "nnz": 18446744073709551615
    }
  ],
  "embeddings": [
    {
      "name": "X_pca",
      "axis": "cells",
      "width": 50,
      "dtype": "float32"
    }
  ],
  "graphs": [
    {
      "name": "connectivities",
      "axis": "cells",
      "dtype": "float32",
      "nnz": 37134
    }
  ]
}"#;

#[test]
fn spec_example_round_trips_byte_for_byte() {
    let schema: AtlasSchema = serde_json::from_str(SPEC_EXAMPLE).unwrap();
    // nnz above 2^53 survives as u64 (format.md §8).
    assert_eq!(schema.layers[0].nnz, u64::MAX);
    assert_eq!(schema.embeddings[0].axis, Axis::Cells);
    assert_eq!(
        schema.cells.columns[2].lineage,
        Some(Lineage {
            layer: "counts".to_owned(),
            node: "n3".to_owned()
        })
    );
    assert!(schema.modalities.is_empty());
    assert_eq!(serde_json::to_string_pretty(&schema).unwrap(), SPEC_EXAMPLE);
}

#[test]
fn rejects_unknown_keys_wrong_types_and_missing_keys() {
    for (from, to, needle) in [
        (
            "\"n_genes\": 32738,",
            "\"n_genes\": 32738, \"extra\": 1,",
            "unknown field `extra`",
        ),
        ("\"n_genes\": 32738,", "", "missing field `n_genes`"),
        ("\"uint32\"", "\"int64\"", "unknown variant `int64`"),
        (
            "\"axis\": \"cells\",\n      \"width\"",
            "\"axis\": \"rows\",\n      \"width\"",
            "unknown variant `rows`",
        ),
        ("\"width\": 50", "\"width\": -1", "invalid value"),
        ("\"nnz\": 37134", "\"nnz\": 1.5", "invalid type"),
        (
            "\"gene_block_size\": 1024",
            "\"gene_block_size\": 1024, \"x\": 0",
            "unknown field `x`",
        ),
    ] {
        assert!(SPEC_EXAMPLE.contains(from), "{from}");
        let text = SPEC_EXAMPLE.replacen(from, to, 1);
        let error = serde_json::from_str::<AtlasSchema>(&text)
            .unwrap_err()
            .to_string();
        assert!(error.contains(needle), "{to}: {error}");
    }
}
