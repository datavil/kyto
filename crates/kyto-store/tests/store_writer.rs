//! The streaming store writer (task 2.5, format.md §10): a round trip through the tile
//! reader is exact for any batching of rows (proptest), bytes do not depend on batching
//! or thread count and equal `write_layer`'s (§10.3, R11), `schema.json` and
//! `provenance.json` follow §8 and §9, the target appears only on success (§10.1),
//! stale temporary directories are deleted by the next write, and bad input gets coded
//! errors (R10). The kill test is `tests/store_writer_kill.rs`.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use kyto_store::{
    AxisSelection, CellBlock, CellValues, CsrMatrix, ImportSource, LayerReader, LayerSpec,
    LayerWriteOptions, StoreWriter, TableAxis, TileGrid, Tiling, ValueDtype, read_table,
    write_layer,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};
use proptest::prelude::*;
use serde_json::Value;
use zarrs::filesystem::FilesystemStore;

/// A small seeded generator, local to each call (memory/test_seed_isolation.md).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next() % n }
    }
}

/// A random `n_cells x n_genes` pair of layers (float32 and uint32, same pattern) with
/// stored zeros (and `-0.0`) that the writer must drop.
fn matrices(n_cells: u64, n_genes: u64, seed: u64) -> (CsrMatrix<f32>, CsrMatrix<u32>) {
    let mut rng = Lcg(seed);
    let density = rng.below(101);
    let mut floats = CsrMatrix {
        n_cells,
        n_genes,
        indptr: vec![0],
        indices: Vec::new(),
        data: Vec::new(),
    };
    let mut counts: CsrMatrix<u32> = CsrMatrix {
        n_cells,
        n_genes,
        indptr: Vec::new(),
        indices: Vec::new(),
        data: Vec::new(),
    };
    for _ in 0..n_cells {
        for gene in 0..n_genes {
            if rng.below(100) < density {
                let value = rng.below(6);
                floats.indices.push(gene as u32);
                floats.data.push(match value {
                    0 => 0.0,
                    1 => -0.0,
                    _ => value as f32 - 2.5,
                });
                counts.data.push(value as u32);
            }
        }
        floats.indptr.push(floats.indices.len() as u64);
    }
    counts.indptr = floats.indptr.clone();
    counts.indices = floats.indices.clone();
    (floats, counts)
}

/// `matrix` without its zeros: what the store must give back (format.md §4.6).
fn canonical<T: Copy>(matrix: &CsrMatrix<T>, is_zero: impl Fn(T) -> bool) -> CsrMatrix<T> {
    let mut out = CsrMatrix {
        n_cells: matrix.n_cells,
        n_genes: matrix.n_genes,
        indptr: vec![0],
        indices: Vec::new(),
        data: Vec::new(),
    };
    for row in matrix.indptr.windows(2) {
        for at in row[0] as usize..row[1] as usize {
            if !is_zero(matrix.data[at]) {
                out.indices.push(matrix.indices[at]);
                out.data.push(matrix.data[at]);
            }
        }
        out.indptr.push(out.indices.len() as u64);
    }
    out
}

/// Rows `start..end` of `matrix` as CSR parts with rebased `indptr`.
fn rows<T: Copy>(matrix: &CsrMatrix<T>, start: u64, end: u64) -> (Vec<u64>, Vec<u32>, Vec<T>) {
    let first = matrix.indptr[start as usize];
    let last = matrix.indptr[end as usize];
    let indptr = matrix.indptr[start as usize..=end as usize]
        .iter()
        .map(|at| at - first)
        .collect();
    let range = first as usize..last as usize;
    (
        indptr,
        matrix.indices[range.clone()].to_vec(),
        matrix.data[range].to_vec(),
    )
}

fn table(id_column: &str, prefix: &str, n: u64) -> DataFrame {
    let ids: Vec<String> = (0..n).map(|row| format!("{prefix}{row}")).collect();
    let groups: Vec<&str> = (0..n)
        .map(|row| if row % 3 == 0 { "b" } else { "a" })
        .collect();
    DataFrame::new(
        n as usize,
        vec![
            Series::new(id_column.into(), ids).into_column(),
            Series::new("group".into(), groups).into_column(),
        ],
    )
    .unwrap()
}

fn layers(options: LayerWriteOptions) -> Vec<LayerSpec> {
    vec![
        LayerSpec {
            name: "norm".to_owned(),
            dtype: ValueDtype::F32,
            options,
        },
        LayerSpec {
            name: "counts".to_owned(),
            dtype: ValueDtype::U32,
            options,
        },
    ]
}

/// Write both layers to `target`, pushing rows in batches of `splits` (cycled).
fn write_store(
    target: &Path,
    floats: &CsrMatrix<f32>,
    counts: &CsrMatrix<u32>,
    tiling: Tiling,
    options: LayerWriteOptions,
    splits: &[u64],
) {
    let n_cells = floats.n_cells;
    let genes = table("gene_id", "g", floats.n_genes);
    let mut writer = StoreWriter::create(target, &genes, &layers(options), tiling, false).unwrap();
    let mut start = 0;
    let mut split = splits.iter().cycle();
    while start < n_cells {
        let end = (start + split.next().unwrap()).min(n_cells);
        let (indptr, indices, data) = rows(floats, start, end);
        let norm = CellBlock::from_f32(indptr, indices, data, floats.n_genes).unwrap();
        let (indptr, indices, data) = rows(counts, start, end);
        let count = CellBlock::from_u32(indptr, indices, data, counts.n_genes).unwrap();
        writer.push(&[norm, count]).unwrap();
        start = end;
    }
    assert_eq!(writer.n_cells(), n_cells);
    writer
        .finish(&table("cell_id", "c", n_cells), None)
        .unwrap();
}

/// Read layer `name` whole through the tile reader (task 2.4).
fn read_back(store: &Path, name: &str, grid: TileGrid) -> (Vec<u64>, Vec<u32>, Vec<CellData>) {
    let storage = Arc::new(FilesystemStore::new(store).unwrap());
    let reader = LayerReader::open(storage, "store", name, grid).unwrap();
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for selected in reader
        .cell_blocks(
            &AxisSelection::All,
            &AxisSelection::All,
            NonZeroUsize::new(2).unwrap(),
        )
        .unwrap()
    {
        let block = selected.unwrap().block;
        let offset = indices.len() as u64;
        indptr.extend(block.indptr()[1..].iter().map(|end| offset + end));
        indices.extend_from_slice(block.indices());
        match block.data() {
            CellValues::F32(values) => data.extend(values.iter().map(|&v| CellData::F32(v))),
            CellValues::U32(values) => data.extend(values.iter().map(|&v| CellData::U32(v))),
        }
    }
    (indptr, indices, data)
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CellData {
    F32(f32),
    U32(u32),
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Every file below `root`, by relative path, `provenance.json` without its `run`
/// object (the only part that may vary, format.md §10.3).
fn store_bytes(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                let relative = path
                    .strip_prefix(root)
                    .unwrap()
                    .components()
                    .map(|part| part.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                let mut bytes = std::fs::read(&path).unwrap();
                if relative == "provenance.json" {
                    let mut value: Value = serde_json::from_slice(&bytes).unwrap();
                    value.as_object_mut().unwrap().remove("run");
                    bytes = serde_json::to_vec(&value).unwrap();
                }
                out.insert(relative, bytes);
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// Entries of `dir`, by name.
fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Done when (part): any batching of rows reads back exactly, zeros dropped.
    #[test]
    fn round_trip_through_the_tile_reader_is_exact(
        n_cells in 0_u64..30,
        n_genes in 0_u64..10,
        cell_block_size in 1_u64..6,
        gene_block_size in 1_u64..5,
        nnz_chunk_size in 1_u64..4,
        chunks_per_shard in 1_u64..4,
        splits in prop::collection::vec(1_u64..12, 1..5),
        seed in any::<u64>(),
    ) {
        let (floats, counts) = matrices(n_cells, n_genes, seed);
        let tiling = Tiling { cell_block_size, gene_block_size };
        let options = LayerWriteOptions {
            nnz_chunk_size,
            nnz_shard_size: nnz_chunk_size * chunks_per_shard,
            zstd_level: 3,
        };
        let parent = tempfile::tempdir().unwrap();
        let target = parent.path().join("x.kyto");
        write_store(&target, &floats, &counts, tiling, options, &splits);
        prop_assert_eq!(entries(parent.path()), vec!["x.kyto".to_owned()]);

        let grid = TileGrid::new(n_cells, n_genes, tiling).unwrap();
        let expected_floats = canonical(&floats, |value| value == 0.0);
        let expected_counts = canonical(&counts, |value| value == 0);
        let (indptr, indices, data) = read_back(&target, "norm", grid);
        prop_assert_eq!(&indptr, &expected_floats.indptr);
        prop_assert_eq!(&indices, &expected_floats.indices);
        let floats_read: Vec<CellData> =
            expected_floats.data.iter().map(|&v| CellData::F32(v)).collect();
        prop_assert_eq!(&data, &floats_read);
        let (indptr, indices, data) = read_back(&target, "counts", grid);
        prop_assert_eq!(&indptr, &expected_counts.indptr);
        prop_assert_eq!(&indices, &expected_counts.indices);
        let counts_read: Vec<CellData> =
            expected_counts.data.iter().map(|&v| CellData::U32(v)).collect();
        prop_assert_eq!(&data, &counts_read);

        let cells = read_table(&target, TableAxis::Cells).unwrap();
        prop_assert_eq!(cells.height() as u64, n_cells);
        let genes = read_table(&target, TableAxis::Genes).unwrap();
        prop_assert_eq!(genes.height() as u64, n_genes);
        let schema = read_json(&target.join("schema.json"));
        prop_assert_eq!(&schema["n_cells"], &Value::from(n_cells));
        prop_assert_eq!(&schema["layers"][0]["nnz"], &Value::from(expected_counts.nnz()));
        prop_assert_eq!(&schema["layers"][1]["nnz"], &Value::from(expected_floats.nnz()));
    }

    /// The layer files are byte-identical to `write_layer`'s for the same matrix, and
    /// do not depend on batching or thread count (format.md §10.3, R11).
    #[test]
    fn bytes_do_not_depend_on_batching_or_threads(
        n_cells in 0_u64..25,
        n_genes in 0_u64..9,
        cell_block_size in 1_u64..5,
        gene_block_size in 1_u64..4,
        splits_a in prop::collection::vec(1_u64..10, 1..4),
        splits_b in prop::collection::vec(1_u64..10, 1..4),
        seed in any::<u64>(),
    ) {
        let (floats, counts) = matrices(n_cells, n_genes, seed);
        let tiling = Tiling { cell_block_size, gene_block_size };
        let options = LayerWriteOptions { nnz_chunk_size: 2, nnz_shard_size: 6, zstd_level: 3 };
        let parent = tempfile::tempdir().unwrap();
        let one = parent.path().join("one.kyto");
        let four = parent.path().join("four.kyto");
        rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap()
            .install(|| write_store(&one, &floats, &counts, tiling, options, &splits_a));
        rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap()
            .install(|| write_store(&four, &floats, &counts, tiling, options, &splits_b));
        let streamed = store_bytes(&one);
        prop_assert_eq!(&streamed, &store_bytes(&four));

        let whole = parent.path().join("whole");
        std::fs::create_dir(&whole).unwrap();
        write_layer(&whole, "norm", &floats, tiling, &options).unwrap();
        write_layer(&whole, "counts", &counts, tiling, &options).unwrap();
        let whole_bytes = store_bytes(&whole);
        for (path, bytes) in &whole_bytes {
            prop_assert_eq!(streamed.get(path), Some(bytes), "{}", path);
        }
        let streamed_layer_files = streamed
            .keys()
            .filter(|path| path.starts_with("layers/") && path.matches('/').count() > 2)
            .count();
        prop_assert_eq!(streamed_layer_files, whole_bytes.len() - 2);
    }
}

fn small_store(target: &Path, overwrite: bool) -> Result<(), kyto_error::KytoError> {
    let genes = table("gene_id", "g", 3);
    let options = LayerWriteOptions::default();
    let mut writer = StoreWriter::create(
        target,
        &genes,
        &layers(options),
        Tiling::default(),
        overwrite,
    )?;
    let norm = CellBlock::from_f32(vec![0, 1, 3], vec![1, 0, 2], vec![1.5, 2.0, 3.0], 3)?;
    let count = CellBlock::from_u32(vec![0, 1, 3], vec![1, 0, 2], vec![1, 2, 3], 3)?;
    writer.push(&[norm, count])?;
    let source = ImportSource {
        kind: "h5ad".to_owned(),
        path: "in.h5ad".to_owned(),
        sha256: "00".repeat(32),
    };
    writer.finish(&table("cell_id", "c", 2), Some(&source))
}

#[test]
fn schema_and_provenance_follow_the_spec() {
    let parent = tempfile::tempdir().unwrap();
    let target = parent.path().join("x.kyto");
    small_store(&target, false).unwrap();
    assert_eq!(
        entries(&target),
        [
            "cells.parquet",
            "embeddings",
            "genes.parquet",
            "graphs",
            "layers",
            "provenance.json",
            "schema.json",
            "zarr.json"
        ]
    );
    assert_eq!(entries(&target.join("embeddings")), ["zarr.json"]);
    assert_eq!(
        entries(&target.join("layers")),
        ["counts", "norm", "zarr.json"]
    );
    let root = read_json(&target.join("zarr.json"));
    assert_eq!(
        root["attributes"],
        serde_json::json!({"kyto": {"format_version": "0.1"}})
    );
    assert_eq!(root["node_type"], "group");
    // An empty group has no attributes key (zarrs omits it; Zarr v3 allows that).
    let graphs = read_json(&target.join("graphs/zarr.json"));
    assert_eq!(graphs["node_type"], "group");
    assert_eq!(graphs.get("attributes"), None);

    let text = std::fs::read_to_string(target.join("schema.json")).unwrap();
    let schema: Value = serde_json::from_str(&text).unwrap();
    let keys: Vec<&String> = schema.as_object().unwrap().keys().collect();
    assert_eq!(
        keys,
        [
            "n_cells",
            "n_genes",
            "tiling",
            "cells",
            "genes",
            "layers",
            "embeddings",
            "graphs"
        ]
    );
    assert_eq!(
        schema["tiling"],
        serde_json::json!({"cell_block_size": 4096, "gene_block_size": 1024})
    );
    assert_eq!(
        schema["genes"]["columns"],
        serde_json::json!([
            {"name": "gene_id", "dtype": "String"},
            {"name": "group", "dtype": "Enum", "n_categories": 2, "ordered": false}
        ])
    );
    // Sorted by name, not in layer order (format.md §8).
    assert_eq!(
        schema["layers"],
        serde_json::json!([
            {"name": "counts", "dtype": "uint32", "nnz": 3},
            {"name": "norm", "dtype": "float32", "nnz": 3}
        ])
    );
    assert!(text.find("\"name\"").unwrap() < text.find("\"dtype\"").unwrap());

    let provenance = read_json(&target.join("provenance.json"));
    let keys: Vec<&String> = provenance.as_object().unwrap().keys().collect();
    assert_eq!(
        keys,
        [
            "kyto_version",
            "dependencies",
            "plan",
            "source",
            "seeds",
            "device",
            "run"
        ]
    );
    assert_eq!(provenance["plan"], Value::Null);
    assert_eq!(provenance["source"]["kind"], "h5ad");
    assert_eq!(provenance["device"], "cpu");
    assert_eq!(provenance["dependencies"]["zarrs"], "0.22.10");
    assert!(
        provenance["dependencies"]["zstd"]
            .as_str()
            .unwrap()
            .starts_with("1.")
    );
    let started = provenance["run"]["started_at"].as_str().unwrap();
    assert_eq!(started.len(), "2026-11-07T10:00:00Z".len());
    assert!(started.ends_with('Z'));
}

#[test]
fn an_existing_target_is_kept_unless_overwriting_a_store() {
    let parent = tempfile::tempdir().unwrap();
    let target = parent.path().join("x.kyto");
    small_store(&target, false).unwrap();
    let error = small_store(&target, false).unwrap_err();
    assert_eq!(error.code(), "KY-E118");
    assert!(
        error
            .to_string()
            .contains("a kyto store already exists there")
    );

    // Overwrite replaces a store, after the new one is complete, and leaves no sibling.
    let before = std::fs::read(target.join("schema.json")).unwrap();
    std::fs::write(target.join("marker"), b"old").unwrap();
    small_store(&target, true).unwrap();
    assert!(!target.join("marker").exists());
    assert_eq!(std::fs::read(target.join("schema.json")).unwrap(), before);
    assert_eq!(entries(parent.path()), ["x.kyto"]);

    // Overwrite never replaces something that is not a kyto store.
    let other = parent.path().join("photos");
    std::fs::create_dir(&other).unwrap();
    std::fs::write(other.join("a.jpg"), b"jpg").unwrap();
    let error = small_store(&other, true).unwrap_err();
    assert_eq!(error.code(), "KY-E118");
    assert!(error.to_string().contains("not a kyto store"));
    assert!(other.join("a.jpg").exists());
    let file = parent.path().join("file.kyto");
    std::fs::write(&file, b"x").unwrap();
    assert_eq!(small_store(&file, false).unwrap_err().code(), "KY-E118");
    assert_eq!(entries(parent.path()), ["file.kyto", "photos", "x.kyto"]);
}

#[test]
fn a_failed_or_dropped_write_leaves_nothing() {
    let parent = tempfile::tempdir().unwrap();
    let target = parent.path().join("x.kyto");
    let genes = table("gene_id", "g", 3);
    let specs = layers(LayerWriteOptions::default());

    let writer = StoreWriter::create(&target, &genes, &specs, Tiling::default(), false).unwrap();
    assert_eq!(
        entries(parent.path()).len(),
        2,
        "a temporary directory and its lock"
    );
    drop(writer);
    assert!(entries(parent.path()).is_empty());

    // A cell table of the wrong length fails in finish; the temporary is deleted.
    let mut writer =
        StoreWriter::create(&target, &genes, &specs, Tiling::default(), false).unwrap();
    let norm = CellBlock::from_f32(vec![0, 1], vec![0], vec![1.0], 3).unwrap();
    let count = CellBlock::from_u32(vec![0, 1], vec![0], vec![1], 3).unwrap();
    writer.push(&[norm, count]).unwrap();
    let error = writer.finish(&table("cell_id", "c", 2), None).unwrap_err();
    assert_eq!(error.code(), "KY-E119");
    assert!(
        error
            .to_string()
            .contains("the cells table has 2 rows, but the matrix has 1 cells")
    );
    assert!(entries(parent.path()).is_empty());
}

#[test]
fn bad_blocks_and_layers_get_coded_errors() {
    let parent = tempfile::tempdir().unwrap();
    let target = parent.path().join("x.kyto");
    let genes = table("gene_id", "g", 3);
    let specs = layers(LayerWriteOptions::default());
    let mut writer =
        StoreWriter::create(&target, &genes, &specs, Tiling::default(), false).unwrap();
    let norm = |n_rows: u64, n_genes: u64| {
        CellBlock::from_f32(
            (0..=n_rows).map(|_| 0).collect(),
            Vec::new(),
            Vec::new(),
            n_genes,
        )
        .unwrap()
    };
    let count = |n_rows: u64| {
        CellBlock::from_u32((0..=n_rows).map(|_| 0).collect(), Vec::new(), Vec::new(), 3).unwrap()
    };
    let problem = |error: kyto_error::KytoError| {
        assert_eq!(error.code(), "KY-E117");
        error.to_string()
    };
    assert!(problem(writer.push(&[norm(1, 3)]).unwrap_err()).contains("got 1 blocks for 2 layers"));
    assert!(
        problem(writer.push(&[norm(1, 2), count(1)]).unwrap_err()).contains(
            "(StoreWriter::push, layer 'norm'): the block has 2 genes but the store has 3"
        )
    );
    assert!(
        problem(writer.push(&[norm(1, 3), count(2)]).unwrap_err())
            .contains("(StoreWriter::push, layer 'counts'): the block has 2 rows but the block of layer 'norm' has 1")
    );
    assert!(
        problem(writer.push(&[norm(1, 3), norm(1, 3)]).unwrap_err())
            .contains("(StoreWriter::push, layer 'counts'): the layer stores uint32 but the block holds float32")
    );
    // Rejected pushes write nothing; the writer is still usable.
    writer.push(&[norm(2, 3), count(2)]).unwrap();
    assert_eq!(writer.n_cells(), 2);
    writer.finish(&table("cell_id", "c", 2), None).unwrap();

    let twins = vec![
        LayerSpec {
            name: "Counts".to_owned(),
            dtype: ValueDtype::U32,
            options: LayerWriteOptions::default(),
        },
        specs[1].clone(),
    ];
    let other = parent.path().join("y.kyto");
    let error = StoreWriter::create(&other, &genes, &twins, Tiling::default(), false)
        .err()
        .unwrap();
    assert_eq!(error.code(), "KY-E103");
    let mut bad_name = specs.clone();
    bad_name[0].name = "a/b".to_owned();
    let error = StoreWriter::create(&other, &genes, &bad_name, Tiling::default(), false)
        .err()
        .unwrap();
    assert_eq!(error.code(), "KY-E102");
    let duplicate_genes = table("gene_id", "g", 3)
        .vstack(&table("gene_id", "g", 1))
        .unwrap();
    let error = StoreWriter::create(&other, &duplicate_genes, &specs, Tiling::default(), false)
        .err()
        .unwrap();
    assert_eq!(error.code(), "KY-E101");
    let tiling = Tiling {
        cell_block_size: 0,
        gene_block_size: 1,
    };
    let error = StoreWriter::create(&other, &genes, &specs, tiling, false)
        .err()
        .unwrap();
    assert_eq!(error.code(), "KY-E104");
    // Input errors are found before anything is created.
    assert_eq!(entries(parent.path()), ["x.kyto"]);
}

#[test]
fn a_store_without_layers_takes_n_cells_from_the_cell_table() {
    let parent = tempfile::tempdir().unwrap();
    let target = parent.path().join("x.kyto");
    let mut writer = StoreWriter::create(
        &target,
        &table("gene_id", "g", 2),
        &[],
        Tiling::default(),
        false,
    )
    .unwrap();
    writer.push(&[]).unwrap();
    writer.finish(&table("cell_id", "c", 5), None).unwrap();
    let schema = read_json(&target.join("schema.json"));
    assert_eq!(schema["n_cells"], 5);
    assert_eq!(schema["layers"], serde_json::json!([]));
    assert_eq!(entries(&target.join("layers")), ["zarr.json"]);
}

/// A stale temporary directory (its writer died, so its lock is free) of the same target
/// is deleted by the next write; live ones, other targets and look-alikes are kept.
#[test]
fn the_next_write_deletes_stale_temporaries_of_its_target() {
    let parent = tempfile::tempdir().unwrap();
    let target = parent.path().join("x.kyto");
    let make = |name: &str| -> PathBuf {
        let path = parent.path().join(name);
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("zarr.json"), b"{}").unwrap();
        path
    };
    let stale = make(".x.kyto.kyto-tmp-0123456789abcdef");
    std::fs::write(
        parent.path().join(".x.kyto.kyto-tmp-0123456789abcdef.lock"),
        b"",
    )
    .unwrap();
    let stale_without_lock = make(".x.kyto.kyto-tmp-fedcba9876543210");
    let orphan_lock = parent.path().join(".x.kyto.kyto-tmp-00000000000000aa.lock");
    std::fs::write(&orphan_lock, b"").unwrap();
    let other_target = make(".y.kyto.kyto-tmp-0123456789abcdef");
    let look_alike = make(".x.kyto.kyto-tmp-not-random");
    let old_store = make(".x.kyto.kyto-old-0123456789abcdef");

    // A live writer of the same target holds its lock; its directory must survive.
    let genes = table("gene_id", "g", 3);
    let specs = layers(LayerWriteOptions::default());
    let live = StoreWriter::create(&target, &genes, &specs, Tiling::default(), false).unwrap();
    let live_names: Vec<String> = entries(parent.path())
        .into_iter()
        .filter(|name| {
            name.starts_with(".x.kyto.kyto-tmp-")
                && !name.ends_with(".lock")
                && name != ".x.kyto.kyto-tmp-not-random"
        })
        .collect();
    assert_eq!(live_names.len(), 1, "{live_names:?}");
    assert!(!stale.exists() && !stale_without_lock.exists() && !orphan_lock.exists());

    small_store(&target, false).unwrap();
    assert!(
        parent.path().join(&live_names[0]).is_dir(),
        "a live temporary was deleted"
    );
    drop(live);
    assert!(other_target.exists() && look_alike.exists() && old_store.exists());
    assert_eq!(
        entries(parent.path()),
        [
            ".x.kyto.kyto-old-0123456789abcdef",
            ".x.kyto.kyto-tmp-not-random",
            ".y.kyto.kyto-tmp-0123456789abcdef",
            "x.kyto"
        ]
    );
}
