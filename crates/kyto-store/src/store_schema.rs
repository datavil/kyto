//! Read a store's `schema.json` (format.md §8) as a `kyto_core::AtlasSchema`, and the
//! parts of it a reader needs before it opens a layer.
//!
//! Does: parse `schema.json` with serde into an [`AtlasSchema`] (unknown keys and wrong
//! types are `KY-E122`), check the tiling and that a v0 store has no modalities, and
//! derive the [`StoreSchema`] view (tile grid, layers with their value dtype, the
//! `ordered` flag of every `Enum` column).
//! Does not: check that `schema.json` agrees with the members it describes (each reader
//! checks what it opens: the tile reader the layer arrays, `read_table` the tables).

use std::path::Path;

use kyto_core::{AtlasSchema, ColumnDtype};
use kyto_error::KytoError;
use zarrs::filesystem::FilesystemStore;

use crate::layer_open::ValueDtype;
use crate::store_version::check_format_version;
use crate::table_axis::TableAxis;
use crate::tiling::TileGrid;
use crate::zarr_arrays::store_io;

/// One `layers` entry of `schema.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SchemaLayer {
    pub name: String,
    pub dtype: ValueDtype,
    pub nnz: u64,
}

/// What [`read_schema`] reads from `schema.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoreSchema {
    /// The tile grid: `n_cells`, `n_genes` and `tiling`.
    pub grid: TileGrid,
    /// The layers, in `schema.json` order (sorted by name, §8).
    pub layers: Vec<SchemaLayer>,
    /// `Enum` columns with `ordered: true`.
    pub ordered: Vec<(TableAxis, String)>,
}

/// Read `schema.json` of the local store at `store`.
///
/// The result equals the schema the store's writer wrote, and serializing it with
/// `serde_json::to_vec_pretty` gives the file's bytes back (without the final newline).
///
/// # Errors
/// - `KY-E110` if the file cannot be read.
/// - `KY-E122` if it is not JSON, a key is missing, unknown or of the wrong type, or it
///   lists modalities (format v0 has none).
/// - `KY-E104`/`KY-E105` (from [`TileGrid::new`]) if the tiling is invalid.
pub fn read_atlas_schema(store: &Path) -> Result<AtlasSchema, KytoError> {
    let store_text = store.display().to_string();
    let corrupt = |problem: String| KytoError::CorruptSchema {
        store: store_text.clone(),
        problem,
    };
    let bytes = std::fs::read(store.join("schema.json")).map_err(|error| KytoError::StoreIo {
        operation: "read schema.json of".to_owned(),
        path: store_text.clone(),
        detail: error.to_string(),
    })?;
    let schema: AtlasSchema = serde_json::from_slice(&bytes).map_err(|error| {
        let problem = if error.is_syntax() || error.is_eof() {
            format!("it is not valid JSON ({error})")
        } else {
            format!("it does not match the format ({error})")
        };
        corrupt(problem)
    })?;
    if !schema.modalities.is_empty() {
        return Err(corrupt(
            "it lists modalities, which format 0.1 stores do not have".to_owned(),
        ));
    }
    TileGrid::new(schema.n_cells, schema.n_genes, schema.tiling)?;
    Ok(schema)
}

/// Check that `store` is a local kyto store of a supported format version, then read
/// its `schema.json` (what a scan needs; reads no table or matrix).
///
/// # Errors
/// - `KY-E116`, `KY-E115` if `store` is not a kyto store of a supported version.
/// - As [`read_atlas_schema`] otherwise.
pub fn open_atlas_schema(store: &Path) -> Result<AtlasSchema, KytoError> {
    let store_text = store.display().to_string();
    let storage =
        FilesystemStore::new(store).map_err(|error| store_io("open store", &store_text, error))?;
    check_format_version(&storage, &store_text)?;
    read_atlas_schema(store)
}

/// Read `schema.json` of the store at `store` and derive the reader's view of it.
///
/// # Errors
/// As [`read_atlas_schema`].
pub(crate) fn read_schema(store: &Path) -> Result<StoreSchema, KytoError> {
    let schema = read_atlas_schema(store)?;
    let grid = TileGrid::new(schema.n_cells, schema.n_genes, schema.tiling)?;
    let layers = schema
        .layers
        .into_iter()
        .map(|layer| SchemaLayer {
            name: layer.name,
            dtype: layer.dtype.into(),
            nnz: layer.nnz,
        })
        .collect();
    let mut ordered = Vec::new();
    for (axis, table) in [
        (TableAxis::Cells, &schema.cells),
        (TableAxis::Genes, &schema.genes),
    ] {
        for column in &table.columns {
            if let ColumnDtype::Enum { ordered: true, .. } = column.dtype {
                ordered.push((axis, column.name.clone()));
            }
        }
    }
    Ok(StoreSchema {
        grid,
        layers,
        ordered,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_schema(text: &str) -> (tempfile::TempDir, Result<StoreSchema, KytoError>) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("schema.json"), text).unwrap();
        let result = read_schema(dir.path());
        (dir, result)
    }

    const VALID: &str = r#"{"n_cells": 3, "n_genes": 2,
        "tiling": {"cell_block_size": 4, "gene_block_size": 2},
        "cells": {"columns": [{"name": "cell_id", "dtype": "String"},
            {"name": "stage", "dtype": "Enum", "n_categories": 2, "ordered": true}]},
        "genes": {"columns": [{"name": "gene_id", "dtype": "String"},
            {"name": "symbol", "dtype": "Enum", "n_categories": 2, "ordered": false}]},
        "layers": [{"name": "counts", "dtype": "uint32", "nnz": 4}],
        "embeddings": [], "graphs": []}"#;

    #[test]
    fn reads_grid_layers_and_ordered_flags() {
        let (_dir, schema) = write_schema(VALID);
        let schema = schema.unwrap();
        assert_eq!(schema.grid.n_cells(), 3);
        assert_eq!(schema.grid.n_genes(), 2);
        assert_eq!(
            schema.layers,
            vec![SchemaLayer {
                name: "counts".to_owned(),
                dtype: ValueDtype::U32,
                nnz: 4
            }]
        );
        assert_eq!(schema.ordered, vec![(TableAxis::Cells, "stage".to_owned())]);
    }

    #[test]
    fn bad_schemas_are_e122() {
        for (text, needle) in [
            ("not json", "not valid JSON"),
            (&VALID.replace("\"n_genes\": 2,", ""), "missing field `n_genes`"),
            (&VALID.replace("\"uint32\"", "\"int64\""), "unknown variant `int64`"),
            (&VALID.replace("\"nnz\": 4", "\"nnz\": -1"), "invalid value: integer `-1`"),
            (&VALID.replace("\"ordered\": true", "\"ordered\": 1"), "invalid type"),
            (
                &VALID.replace("\"cell_block_size\": 4,", ""),
                "missing field `cell_block_size`",
            ),
            (
                &VALID.replace("\"n_genes\": 2,", "\"n_genes\": 2, \"extra\": 0,"),
                "unknown field `extra`",
            ),
            (
                &VALID.replace("\"graphs\": []", "\"graphs\": [], \"modalities\": [{\"name\": \"adt\", \"n_features\": 1, \"features\": {\"columns\": []}, \"layers\": []}]"),
                "lists modalities",
            ),
        ] {
            let (_dir, result) = write_schema(text);
            let error = result.unwrap_err();
            assert_eq!(error.code(), "KY-E122", "{error}");
            assert!(error.to_string().contains(needle), "{error}");
        }
    }

    #[test]
    fn missing_schema_is_e110() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_schema(dir.path()).unwrap_err().code(), "KY-E110");
    }
}
