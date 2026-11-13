//! Combining stores (task 2.9): genes remapped onto the joined axis (also when a
//! source's genes come out of order), layers joined with `uint32` + `float32` giving
//! `float32` and a missing layer giving no values, cells in source order across
//! several cell blocks, the label column, and the target check. Parity with
//! `anndata.concat` is `tests/test_concat.py`; memory is `tests/concat_memory.rs`.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use kyto_store::{
    CellBlock, ConcatOptions, ConvertOptions, CsrMatrix, Join, LayerSpec, LayerWriteOptions,
    StoreWriter, TableAxis, Tiling, ValueDtype, concat, read_atlas_schema, read_layer, read_table,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};

const TILING: Tiling = Tiling {
    cell_block_size: 2,
    gene_block_size: 2,
};

/// One layer of a test store: name, dtype and dense rows.
struct Layer<'a> {
    name: &'a str,
    dtype: ValueDtype,
    rows: Vec<Vec<f32>>,
}

/// Write a store with these cells, genes and layers (cell blocks of 2).
fn write_store(path: &Path, cells: &[&str], genes: &[&str], layers: &[Layer<'_>]) {
    let gene_table = DataFrame::new(
        genes.len(),
        vec![Series::new("gene_id".into(), genes).into_column()],
    )
    .unwrap();
    let specs: Vec<LayerSpec> = layers
        .iter()
        .map(|layer| LayerSpec {
            name: layer.name.to_owned(),
            dtype: layer.dtype,
            options: LayerWriteOptions::default(),
        })
        .collect();
    let mut writer = StoreWriter::create(path, &gene_table, &specs, TILING, false).unwrap();
    for row in 0..cells.len() {
        let blocks: Vec<CellBlock> = layers
            .iter()
            .map(|layer| {
                let (indices, values): (Vec<u32>, Vec<f32>) = layer.rows[row]
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| **value != 0.0)
                    .map(|(gene, value)| (gene as u32, *value))
                    .unzip();
                let indptr = vec![0, indices.len() as u64];
                let n_genes = genes.len() as u64;
                match layer.dtype {
                    ValueDtype::F32 => CellBlock::from_f32(indptr, indices, values, n_genes),
                    ValueDtype::U32 => CellBlock::from_u32(
                        indptr,
                        indices,
                        values.iter().map(|value| *value as u32).collect(),
                        n_genes,
                    ),
                }
                .unwrap()
            })
            .collect();
        writer.push(&blocks).unwrap();
    }
    let cell_table = DataFrame::new(
        cells.len(),
        vec![Series::new("cell_id".into(), cells).into_column()],
    )
    .unwrap();
    writer.finish(&cell_table, None).unwrap();
}

/// A layer of a store as dense rows of `f32`, with its dtype.
fn dense(store: &Path, name: &str) -> (ValueDtype, Vec<Vec<f32>>) {
    let schema = read_atlas_schema(store).unwrap();
    let layer = schema
        .layers
        .iter()
        .find(|layer| layer.name == name)
        .unwrap();
    let (n_cells, n_genes, tiling) = (schema.n_cells, schema.n_genes, schema.tiling);
    let to_rows = |matrix: CsrMatrix<f32>| -> Vec<Vec<f32>> {
        (0..n_cells as usize)
            .map(|row| {
                let mut values = vec![0.0; n_genes as usize];
                let (start, end) = (matrix.indptr[row] as usize, matrix.indptr[row + 1] as usize);
                for entry in start..end {
                    values[matrix.indices[entry] as usize] = matrix.data[entry];
                }
                values
            })
            .collect()
    };
    if layer.dtype == ValueDtype::U32.into() {
        let matrix = read_layer::<u32>(store, name, n_cells, n_genes, tiling).unwrap();
        let floats = CsrMatrix {
            n_cells,
            n_genes,
            indptr: matrix.indptr,
            indices: matrix.indices,
            data: matrix.data.iter().map(|value| *value as f32).collect(),
        };
        (ValueDtype::U32, to_rows(floats))
    } else {
        let matrix = read_layer::<f32>(store, name, n_cells, n_genes, tiling).unwrap();
        (ValueDtype::F32, to_rows(matrix))
    }
}

fn strings(frame: &DataFrame, name: &str) -> Vec<String> {
    let column = frame
        .column(name)
        .unwrap()
        .cast(&polars::prelude::DataType::String)
        .unwrap();
    column
        .str()
        .unwrap()
        .iter()
        .map(|value| value.unwrap().to_owned())
        .collect()
}

/// Store `a`: 5 cells (3 cell blocks), genes g3, g1, g2, layers `counts` (uint32) and
/// `extra` (float32). Store `b`: 2 cells, genes g2, g4, layer `counts` (float32).
fn two_stores(folder: &Path) -> (PathBuf, PathBuf) {
    let a = folder.join("a.kyto");
    let rows: Vec<Vec<f32>> = (0..5)
        .map(|cell| vec![cell as f32 + 1.0, 0.0, 10.0 + cell as f32])
        .collect();
    write_store(
        &a,
        &["a0", "a1", "a2", "a3", "a4"],
        &["g3", "g1", "g2"],
        &[
            Layer {
                name: "counts",
                dtype: ValueDtype::U32,
                rows: rows.clone(),
            },
            Layer {
                name: "extra",
                dtype: ValueDtype::F32,
                rows: rows
                    .iter()
                    .map(|row| vec![row[0] / 2.0, 0.5, 0.0])
                    .collect(),
            },
        ],
    );
    let b = folder.join("b.kyto");
    write_store(
        &b,
        &["b0", "b1"],
        &["g2", "g4"],
        &[Layer {
            name: "counts",
            dtype: ValueDtype::F32,
            rows: vec![vec![0.25, 3.0], vec![7.0, 0.0]],
        }],
    );
    (a, b)
}

fn options(join: Join) -> ConcatOptions {
    ConcatOptions {
        join,
        convert: ConvertOptions {
            tiling: TILING,
            ..ConvertOptions::default()
        },
        ..ConcatOptions::default()
    }
}

#[test]
fn outer_join_remaps_genes_and_layers() {
    let folder = tempfile::tempdir().unwrap();
    let (a, b) = two_stores(folder.path());
    let target = folder.path().join("both.kyto");
    concat(&[a, b], &target, &options(Join::Outer)).unwrap();

    let genes = read_table(&target, TableAxis::Genes).unwrap();
    assert_eq!(strings(&genes, "gene_id"), ["g1", "g2", "g3", "g4"]);
    let cells = read_table(&target, TableAxis::Cells).unwrap();
    assert_eq!(
        strings(&cells, "cell_id"),
        ["a0", "a1", "a2", "a3", "a4", "b0", "b1"]
    );
    assert_eq!(
        strings(&cells, "sample"),
        ["a", "a", "a", "a", "a", "b", "b"]
    );

    // a's genes g3, g1, g2 land at 2, 0, 1; b's g2, g4 at 1, 3.
    let (dtype, counts) = dense(&target, "counts");
    assert_eq!(dtype, ValueDtype::F32, "uint32 + float32 gives float32");
    let mut expected: Vec<Vec<f32>> = (0..5)
        .map(|cell| vec![0.0, 10.0 + cell as f32, cell as f32 + 1.0, 0.0])
        .collect();
    expected.push(vec![0.0, 0.25, 0.0, 3.0]);
    expected.push(vec![0.0, 7.0, 0.0, 0.0]);
    assert_eq!(counts, expected);

    let (dtype, extra) = dense(&target, "extra");
    assert_eq!(dtype, ValueDtype::F32);
    let mut expected: Vec<Vec<f32>> = (0..5)
        .map(|cell| vec![0.5, 0.0, (cell as f32 + 1.0) / 2.0, 0.0])
        .collect();
    expected.extend([vec![0.0; 4], vec![0.0; 4]]);
    assert_eq!(extra, expected, "b has no `extra`: absent values (R9)");
}

#[test]
fn inner_join_keeps_first_order_and_common_layers() {
    let folder = tempfile::tempdir().unwrap();
    let (a, b) = two_stores(folder.path());
    let target = folder.path().join("both.kyto");
    let options = ConcatOptions {
        keys: Some(vec!["first".to_owned(), "second".to_owned()]),
        label: "batch".to_owned(),
        ..options(Join::Inner)
    };
    concat(&[b, a], &target, &options).unwrap();

    let genes = read_table(&target, TableAxis::Genes).unwrap();
    assert_eq!(strings(&genes, "gene_id"), ["g2"]);
    let schema = read_atlas_schema(&target).unwrap();
    let layers: Vec<&str> = schema
        .layers
        .iter()
        .map(|layer| layer.name.as_str())
        .collect();
    assert_eq!(layers, ["counts"], "`extra` is not in every source");
    let cells = read_table(&target, TableAxis::Cells).unwrap();
    assert_eq!(
        strings(&cells, "batch"),
        [
            "first", "first", "second", "second", "second", "second", "second"
        ]
    );
    let (_, counts) = dense(&target, "counts");
    let column: Vec<f32> = counts.iter().map(|row| row[0]).collect();
    assert_eq!(column, [0.25, 7.0, 10.0, 11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn same_genes_keep_uint32() {
    let folder = tempfile::tempdir().unwrap();
    let (a, _) = two_stores(folder.path());
    let target = folder.path().join("twice.kyto");
    let options = ConcatOptions {
        keys: Some(vec!["x".to_owned(), "y".to_owned()]),
        on_duplicate_ids: kyto_store::OnDuplicateIds::Suffix,
        ..options(Join::Inner)
    };
    concat(&[a.clone(), a.clone()], &target, &options).unwrap();
    let (dtype, counts) = dense(&target, "counts");
    assert_eq!(dtype, ValueDtype::U32);
    let (_, once) = dense(&a, "counts");
    assert_eq!(counts, [once.clone(), once].concat());
    let cells = read_table(&target, TableAxis::Cells).unwrap();
    assert_eq!(
        strings(&cells, "cell_id")[5..],
        ["a0-1", "a1-1", "a2-1", "a3-1", "a4-1"]
    );

    // Without suffixes the repeated cells are refused before anything is written.
    let refused = folder.path().join("refused.kyto");
    let error = concat(&[a.clone(), a], &refused, &options_with_keys()).unwrap_err();
    assert_eq!(error.code(), "KY-E101");
    assert!(!refused.exists());
}

fn options_with_keys() -> ConcatOptions {
    ConcatOptions {
        keys: Some(vec!["x".to_owned(), "y".to_owned()]),
        ..options(Join::Inner)
    }
}

#[test]
fn target_may_not_be_a_source() {
    let folder = tempfile::tempdir().unwrap();
    let (a, b) = two_stores(folder.path());
    let options = ConcatOptions {
        overwrite: true,
        ..options(Join::Inner)
    };
    let error = concat(&[a.clone(), b], &a, &options).unwrap_err();
    assert_eq!(error.code(), "KY-E118");
    assert!(error.to_string().contains("source"));
    assert!(a.join("zarr.json").is_file(), "the source is untouched");
}
