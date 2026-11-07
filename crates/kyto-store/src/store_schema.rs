//! Read the parts of a store's `schema.json` a reader needs before it opens a layer
//! (format.md §8).
//!
//! Does: parse `n_cells`, `n_genes`, `tiling`, the `layers` list (name, dtype, nnz) and
//! the `ordered` flag of every `Enum` column, and check their types.
//! Does not: check that `schema.json` agrees with the members it describes (each reader
//! checks what it opens: the tile reader the layer arrays, `read_table` the tables), or
//! read `lineage`, `embeddings` or `graphs` (not used yet).

use std::path::Path;

use kyto_error::KytoError;
use serde_json::Value;

use crate::layer_open::ValueDtype;
use crate::table_axis::TableAxis;
use crate::tiling::{TileGrid, Tiling};

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

/// Read `schema.json` of the store at `store`.
///
/// # Errors
/// - `KY-E110` if the file cannot be read.
/// - `KY-E122` if it is not JSON or a key is missing or of the wrong type.
/// - `KY-E104`/`KY-E105` (from [`TileGrid::new`]) if the tiling is invalid.
pub(crate) fn read_schema(store: &Path) -> Result<StoreSchema, KytoError> {
    let path = store.join("schema.json");
    let store_text = store.display().to_string();
    let corrupt = |problem: String| KytoError::CorruptSchema {
        store: store_text.clone(),
        problem,
    };
    let bytes = std::fs::read(&path).map_err(|error| KytoError::StoreIo {
        operation: "read schema.json of".to_owned(),
        path: store_text.clone(),
        detail: error.to_string(),
    })?;
    let root: Value = serde_json::from_slice(&bytes)
        .map_err(|error| corrupt(format!("it is not valid JSON ({error})")))?;
    let integer = |value: &Value, key: &str| {
        value
            .as_u64()
            .ok_or_else(|| corrupt(format!("{key} is not a non-negative integer")))
    };
    // `key` is the full path for messages (`layers[0].name`); its last part is looked up.
    let field = |value: &'_ Value, key: &str| -> Result<Value, KytoError> {
        value
            .get(key.rsplit('.').next().unwrap_or(key))
            .cloned()
            .ok_or_else(|| corrupt(format!("{key} is missing")))
    };
    let n_cells = integer(&field(&root, "n_cells")?, "n_cells")?;
    let n_genes = integer(&field(&root, "n_genes")?, "n_genes")?;
    let tiling_value = field(&root, "tiling")?;
    let tiling = Tiling {
        cell_block_size: integer(
            &field(&tiling_value, "tiling.cell_block_size")?,
            "tiling.cell_block_size",
        )?,
        gene_block_size: integer(
            &field(&tiling_value, "tiling.gene_block_size")?,
            "tiling.gene_block_size",
        )?,
    };
    let grid = TileGrid::new(n_cells, n_genes, tiling)?;

    let layers_value = field(&root, "layers")?;
    let entries = layers_value
        .as_array()
        .ok_or_else(|| corrupt("layers is not a list".to_owned()))?;
    let mut layers = Vec::with_capacity(entries.len());
    for (position, entry) in entries.iter().enumerate() {
        let key = format!("layers[{position}]");
        let name = field(entry, &format!("{key}.name"))?
            .as_str()
            .ok_or_else(|| corrupt(format!("{key}.name is not a string")))?
            .to_owned();
        let dtype_value = field(entry, &format!("{key}.dtype"))?;
        let dtype = match dtype_value.as_str() {
            Some("float32") => ValueDtype::F32,
            Some("uint32") => ValueDtype::U32,
            _ => {
                return Err(corrupt(format!(
                    "{key}.dtype is {dtype_value}, not 'float32' or 'uint32'"
                )));
            }
        };
        let nnz = integer(&field(entry, &format!("{key}.nnz"))?, &format!("{key}.nnz"))?;
        layers.push(SchemaLayer { name, dtype, nnz });
    }

    let mut ordered = Vec::new();
    for axis in [TableAxis::Cells, TableAxis::Genes] {
        let key = axis.as_str();
        let columns_value = field(&field(&root, key)?, &format!("{key}.columns"))?;
        let columns = columns_value
            .as_array()
            .ok_or_else(|| corrupt(format!("{key}.columns is not a list")))?;
        for (position, column) in columns.iter().enumerate() {
            let column_key = format!("{key}.columns[{position}]");
            if column.get("dtype").and_then(Value::as_str) != Some("Enum") {
                continue;
            }
            let name = field(column, &format!("{column_key}.name"))?;
            let name = name
                .as_str()
                .ok_or_else(|| corrupt(format!("{column_key}.name is not a string")))?;
            match column.get("ordered") {
                Some(Value::Bool(true)) => ordered.push((axis, name.to_owned())),
                Some(Value::Bool(false)) => {}
                _ => {
                    return Err(corrupt(format!(
                        "{column_key}.ordered is missing or not a boolean"
                    )));
                }
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
        "layers": [{"name": "counts", "dtype": "uint32", "nnz": 4}]}"#;

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
            (&VALID.replace("\"n_genes\": 2,", ""), "n_genes is missing"),
            (&VALID.replace("\"uint32\"", "\"int64\""), "layers[0].dtype"),
            (&VALID.replace("\"nnz\": 4", "\"nnz\": -1"), "layers[0].nnz"),
            (
                &VALID.replace("\"ordered\": true", "\"ordered\": 1"),
                "cells.columns[1].ordered",
            ),
            (
                &VALID.replace("\"cell_block_size\": 4,", ""),
                "tiling.cell_block_size is missing",
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
