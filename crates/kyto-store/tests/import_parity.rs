//! Task 2.6 Done when: every fixture converts and matches anndata's values exactly.
//!
//! `tests/fixtures/import_inputs.py` writes each fixture (synthetic, pbmc3k) as `.h5ad`
//! (CSR, CSC with gzip, dense with lzf), 10x `.h5` (v2, v3) and 10x Matrix Market
//! directories (v2 plain, v3 gzipped), and what anndata or scanpy read from each. This
//! test converts every case and reads the store back with the 2.4 reader: layer values
//! bit for bit, tables cell for cell, `schema.json` columns (dtypes, Enum category
//! counts, `ordered`) and Enum category lists. Needs Python with anndata and scanpy
//! (`uv run cargo test`); skips where they have no wheels (exit code 77).

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used)]

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use kyto_store::{
    AxisSelection, CellValues, ConvertOptions, LayerReader, TableAxis, TileGrid, Tiling, convert,
    read_table,
};
use polars::prelude::{DataFrame, DataType, ParquetReader, SerReader};
use serde_json::Value;
use zarrs::filesystem::FilesystemStore;

/// A Python interpreter with anndata: `PYTHON`, else `python` on `PATH` (under
/// `uv run cargo test`, the project's `.venv`).
fn python() -> Command {
    Command::new(std::env::var("PYTHON").unwrap_or_else(|_| "python".to_owned()))
}

fn repo_tests() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests")
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Little-endian words of a file.
fn words<const N: usize>(path: &Path) -> Vec<[u8; N]> {
    std::fs::read(path).unwrap().as_chunks::<N>().0.to_vec()
}

/// A layer read back whole: indptr, indices, and data as little-endian bytes.
fn read_layer_back(store: &Path, layer: &str, grid: TileGrid) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let storage = Arc::new(FilesystemStore::new(store).unwrap());
    let reader = LayerReader::open(storage, "store", layer, grid).unwrap();
    let (mut indptr, mut indices, mut data) = (vec![0_u64], Vec::new(), Vec::new());
    for selected in reader
        .cell_blocks(
            &AxisSelection::All,
            &AxisSelection::All,
            NonZeroUsize::new(4).unwrap(),
        )
        .unwrap()
    {
        let block = selected.unwrap().block;
        let base = indptr[indptr.len() - 1];
        indptr.extend(block.indptr()[1..].iter().map(|offset| offset + base));
        indices.extend_from_slice(block.indices());
        match block.data() {
            CellValues::F32(values) => {
                data.extend(values.iter().flat_map(|value| value.to_le_bytes()));
            }
            CellValues::U32(values) => {
                data.extend(values.iter().flat_map(|value| value.to_le_bytes()));
            }
        }
    }
    (indptr, indices, data)
}

/// Enum columns as strings, so stored and expected frames compare by value.
fn enums_as_strings(frame: &DataFrame) -> DataFrame {
    let columns = frame
        .columns()
        .iter()
        .map(|column| match column.dtype() {
            DataType::Enum(..) => column.cast(&DataType::String).unwrap(),
            _ => column.clone(),
        })
        .collect();
    DataFrame::new(frame.height(), columns).unwrap()
}

/// The category list of an Enum column of a stored table.
fn categories(frame: &DataFrame, name: &str) -> Vec<String> {
    match frame.column(name).unwrap().dtype() {
        DataType::Enum(frozen, _) => frozen
            .categories()
            .values_iter()
            .map(str::to_owned)
            .collect(),
        other => panic!("column {name} is {other}, not an Enum"),
    }
}

fn check_case(case: &Value, expected_root: &Path, work: &Path) {
    let name = case["case"].as_str().unwrap();
    let input = Path::new(case["input"].as_str().unwrap());
    let expected = expected_root.join(name);
    let target = work.join(format!("{name}.kyto"));
    // Small tiles and, for the synthetic fixture, a tiny budget: many cell blocks, gene
    // blocks and transposition ranges (one per cell block for CSC and mtx inputs).
    let synthetic = name.starts_with("synthetic");
    let options = ConvertOptions {
        tiling: Tiling {
            cell_block_size: if synthetic { 64 } else { 512 },
            gene_block_size: if synthetic { 48 } else { 4096 },
        },
        memory_budget_bytes: if synthetic {
            1 << 10
        } else {
            ConvertOptions::default().memory_budget_bytes
        },
        ..ConvertOptions::default()
    };
    convert(input, &target, &options).unwrap_or_else(|error| panic!("{name}: {error}"));

    let schema = read_json(&target.join("schema.json"));
    let want = read_json(&expected.join("schema.json"));
    for (axis, key, file) in [
        (TableAxis::Cells, "cells", "cells.parquet"),
        (TableAxis::Genes, "genes", "genes.parquet"),
    ] {
        let stored = read_table(&target, axis).unwrap();
        let mut want_columns = want[key].as_array().unwrap().clone();
        for column in &mut want_columns {
            if let Some(list) = column.as_object_mut().unwrap().remove("categories") {
                let column_name = column["name"].as_str().unwrap();
                let list: Vec<String> = serde_json::from_value(list).unwrap();
                assert_eq!(
                    categories(&stored, column_name),
                    list,
                    "{name} {key}.{column_name}"
                );
            }
        }
        assert_eq!(
            schema[key]["columns"],
            Value::Array(want_columns),
            "{name} {key}"
        );
        let reference = ParquetReader::new(std::fs::File::open(expected.join(file)).unwrap())
            .finish()
            .unwrap();
        let stored = enums_as_strings(&stored);
        assert!(
            stored.equals_missing(&reference),
            "{name} {key}: stored {stored} expected {reference}"
        );
    }

    let n_cells = schema["n_cells"].as_u64().unwrap();
    let n_genes = schema["n_genes"].as_u64().unwrap();
    let grid = TileGrid::new(n_cells, n_genes, options.tiling).unwrap();
    let mut stored_layers: Vec<(String, String)> = schema["layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|layer| {
            (
                layer["name"].as_str().unwrap().to_owned(),
                layer["dtype"].as_str().unwrap().to_owned(),
            )
        })
        .collect();
    let mut want_layers: Vec<(String, String)> = want["layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|layer| {
            (
                layer["name"].as_str().unwrap().to_owned(),
                layer["dtype"].as_str().unwrap().to_owned(),
            )
        })
        .collect();
    stored_layers.sort();
    want_layers.sort();
    assert_eq!(stored_layers, want_layers, "{name} layers");
    for (layer, _) in &want_layers {
        let (indptr, indices, data) = read_layer_back(&target, layer, grid);
        let want_indptr: Vec<u64> = words::<8>(&expected.join(format!("{layer}.indptr")))
            .into_iter()
            .map(u64::from_le_bytes)
            .collect();
        let want_indices: Vec<u32> = words::<4>(&expected.join(format!("{layer}.indices")))
            .into_iter()
            .map(u32::from_le_bytes)
            .collect();
        let want_data = std::fs::read(expected.join(format!("{layer}.data"))).unwrap();
        assert_eq!(indptr, want_indptr, "{name} {layer} indptr");
        assert_eq!(indices, want_indices, "{name} {layer} indices");
        assert!(data == want_data, "{name} {layer} data differs");
    }

    let provenance = read_json(&target.join("provenance.json"));
    assert_eq!(provenance["source"]["kind"], case["kind"], "{name}");
    assert_eq!(provenance["source"]["sha256"].as_str().unwrap().len(), 64);
}

#[test]
fn every_fixture_matches_anndata() {
    let work = tempfile::tempdir().unwrap();
    let cases_dir = work.path().join("cases");
    let status = python()
        .args(["-m", "fixtures.import_inputs"])
        .arg(&cases_dir)
        .current_dir(repo_tests())
        .status()
        .expect("run python (use `uv run cargo test`, or set PYTHON)");
    if status.code() == Some(77) {
        eprintln!("skipped: anndata or scanpy is not installed on this platform");
        return;
    }
    assert!(status.success(), "fixtures.import_inputs failed: {status}");
    let manifest = read_json(&cases_dir.join("manifest.json"));
    let cases = manifest.as_array().unwrap();
    assert_eq!(cases.len(), 14, "7 inputs per fixture, 2 fixtures");
    for case in cases {
        check_case(case, &cases_dir.join("expected"), work.path());
    }
}
