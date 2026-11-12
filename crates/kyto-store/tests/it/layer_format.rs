//! A written layer follows format.md §4 byte for byte where the spec fixes it:
//! `zarr.json` of the group and each array (§4.2, §4.4), no stored chunks or inner
//! chunks of fill values (§4.4), byte-identical rewrites (§10.3), and coded errors for
//! bad input and bad stores (§1.2, §2.1, R10).

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::Path;

use kyto_store::{CsrMatrix, LayerWriteOptions, Tiling, read_layer, write_layer};
use serde_json::{Value, json};

const TILING: Tiling = Tiling {
    cell_block_size: 2,
    gene_block_size: 2,
};
const OPTIONS: LayerWriteOptions = LayerWriteOptions {
    nnz_chunk_size: 2,
    nnz_shard_size: 4,
    zstd_level: 3,
};

/// Rows `[1 0 2 0 3]`, `[0 0 0 0 0]`, `[0 4 0 0 0]`: tiles (1, 1) and (1, 2) are empty.
fn matrix() -> CsrMatrix<f32> {
    CsrMatrix {
        n_cells: 3,
        n_genes: 5,
        indptr: vec![0, 3, 3, 4],
        indices: vec![0, 2, 4, 1],
        data: vec![1.0, 2.0, 3.0, 4.0],
    }
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn sharding(inner_chunk_shape: Value) -> Value {
    json!([{
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": inner_chunk_shape,
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "zstd", "configuration": {"level": 3, "checksum": false}}
            ],
            "index_codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "crc32c"}
            ],
            "index_location": "end"
        }
    }])
}

fn array_json(
    shape: Value,
    dtype: &str,
    chunk: Value,
    fill: Value,
    codecs: Value,
    names: Value,
) -> Value {
    json!({
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": dtype,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": fill,
        "codecs": codecs,
        "dimension_names": names
    })
}

#[test]
fn metadata_matches_format_4_2_and_4_4() {
    let store = tempfile::tempdir().unwrap();
    write_layer(store.path(), "counts", &matrix(), TILING, &OPTIONS).unwrap();
    let layer = store.path().join("layers/counts");
    assert_eq!(
        read_json(&layer.join("zarr.json")),
        json!({"zarr_format": 3, "node_type": "group", "attributes": {"kyto": {"kind": "tiled_csr"}}})
    );
    assert_eq!(
        read_json(&layer.join("indptr/zarr.json")),
        array_json(
            json!([2, 3, 3]),
            "uint32",
            json!([1, 3, 3]),
            json!(0),
            sharding(json!([1, 1, 3])),
            json!(["cell_block", "gene_block", "row"])
        )
    );
    let plain = json!([
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "zstd", "configuration": {"level": 3, "checksum": false}}
    ]);
    assert_eq!(
        read_json(&layer.join("tile_offsets/zarr.json")),
        array_json(
            json!([7]),
            "uint64",
            json!([65536]),
            json!(0),
            plain,
            json!(["tile"])
        )
    );
    assert_eq!(
        read_json(&layer.join("indices/zarr.json")),
        array_json(
            json!([4]),
            "uint32",
            json!([4]),
            json!(0),
            sharding(json!([2])),
            json!(["entry"])
        )
    );
    assert_eq!(
        read_json(&layer.join("data/zarr.json")),
        array_json(
            json!([4]),
            "float32",
            json!([4]),
            json!(0.0),
            sharding(json!([2])),
            json!(["entry"])
        )
    );
}

/// Every file under `root`, by relative path, with its bytes.
fn files(root: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut found = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                let name = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                found.insert(name, std::fs::read(&path).unwrap());
            }
        }
    }
    found
}

#[test]
fn fill_value_chunks_are_not_stored() {
    let store = tempfile::tempdir().unwrap();
    write_layer(store.path(), "counts", &matrix(), TILING, &OPTIONS).unwrap();
    // Shard of cell block 1: three inner chunks (tiles), then a 16-byte (offset, nbytes)
    // entry per inner chunk and a 4-byte crc32c. Empty tiles (1, 1), (1, 2) are absent.
    let shard = std::fs::read(store.path().join("layers/counts/indptr/c/1/0/0")).unwrap();
    let index = &shard[shard.len() - 3 * 16 - 4..shard.len() - 4];
    let entry = |inner: usize| {
        let word = |at: usize| u64::from_le_bytes(index[at..at + 8].try_into().unwrap());
        (word(inner * 16), word(inner * 16 + 8))
    };
    assert_ne!(entry(0), (u64::MAX, u64::MAX));
    assert_eq!(entry(1), (u64::MAX, u64::MAX));
    assert_eq!(entry(2), (u64::MAX, u64::MAX));

    // An all-zero layer stores metadata only.
    let zeros = CsrMatrix {
        data: vec![0.0, -0.0, 0.0, 0.0],
        ..matrix()
    };
    write_layer(store.path(), "zeros", &zeros, TILING, &OPTIONS).unwrap();
    let names: Vec<String> = files(&store.path().join("layers/zeros"))
        .into_keys()
        .collect();
    assert_eq!(
        names,
        [
            "data/zarr.json",
            "indices/zarr.json",
            "indptr/zarr.json",
            "tile_offsets/zarr.json",
            "zarr.json"
        ]
    );
    let read = read_layer::<f32>(store.path(), "zeros", 3, 5, TILING).unwrap();
    assert_eq!((read.nnz(), read.indptr), (0, vec![0, 0, 0, 0]));
}

#[test]
fn rewrites_are_byte_identical() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    // A larger layer too, so shards hold many inner chunks for parallel encoders to race on.
    let (n_cells, n_genes) = (60_u64, 50_u64);
    let mut large = CsrMatrix {
        n_cells,
        n_genes,
        indptr: vec![0],
        indices: Vec::new(),
        data: Vec::new(),
    };
    for cell in 0..n_cells {
        for gene in (cell % 3..n_genes).step_by(2) {
            large.indices.push(gene as u32);
            large.data.push((cell * n_genes + gene) as f32);
        }
        large.indptr.push(large.indices.len() as u64);
    }
    for store in [first.path(), second.path()] {
        write_layer(store, "counts", &matrix(), TILING, &OPTIONS).unwrap();
        write_layer(store, "large", &large, TILING, &OPTIONS).unwrap();
    }
    let written = files(first.path());
    let rewritten = files(second.path());
    assert_eq!(
        written
            .keys()
            .filter(|name| name.starts_with("layers/counts/"))
            .count(),
        10
    );
    assert!(written.keys().eq(rewritten.keys()));
    let differing: Vec<&String> = written
        .iter()
        .filter(|(name, bytes)| rewritten.get(*name) != Some(bytes))
        .map(|(name, _)| name)
        .collect();
    assert!(differing.is_empty(), "files differ: {differing:?}");
}

fn code<T: std::fmt::Debug>(result: Result<T, kyto_error::KytoError>) -> (String, String) {
    let error = result.unwrap_err();
    (error.code().to_owned(), error.to_string())
}

#[test]
fn writer_errors_are_coded() {
    let store = tempfile::tempdir().unwrap();
    let write = |name: &str, matrix: &CsrMatrix<f32>, options: &LayerWriteOptions| {
        write_layer(store.path(), name, matrix, TILING, options)
    };
    let (found, message) = code(write("../escape", &matrix(), &OPTIONS));
    assert_eq!(found, "KY-E102", "{message}");
    write("counts", &matrix(), &OPTIONS).unwrap();
    let (found, message) = code(write("COUNTS", &matrix(), &OPTIONS));
    assert_eq!(found, "KY-E103");
    assert!(
        message.contains("already exists in the store as 'counts'"),
        "{message}"
    );
    let bad_options = LayerWriteOptions {
        nnz_shard_size: 3,
        ..OPTIONS
    };
    assert_eq!(code(write("other", &matrix(), &bad_options)).0, "KY-E104");
    let unsorted = CsrMatrix {
        indices: vec![2, 0, 4, 1],
        ..matrix()
    };
    let (found, message) = code(write("other", &unsorted, &OPTIONS));
    assert_eq!(found, "KY-E106");
    assert!(
        message.contains("layer 'other': row 0: gene index 0 follows 2"),
        "{message}"
    );
    // Nothing was written for the rejected layers.
    assert!(!store.path().join("layers/other").exists());
}

#[test]
fn reader_errors_are_coded() {
    let store = tempfile::tempdir().unwrap();
    write_layer(store.path(), "counts", &matrix(), TILING, &OPTIONS).unwrap();
    let (found, message) = code(read_layer::<u32>(store.path(), "counts", 3, 5, TILING));
    assert_eq!(found, "KY-E108");
    assert!(
        message.contains("stores float32 values, not uint32"),
        "{message}"
    );
    let (found, message) = code(read_layer::<f32>(store.path(), "count", 3, 5, TILING));
    assert_eq!(found, "KY-E109");
    assert!(message.contains("no layer named 'count'"), "{message}");
    let (found, message) = code(read_layer::<f32>(store.path(), "counts", 5, 5, TILING));
    assert_eq!(found, "KY-E107");
    assert!(
        message.contains("array indptr has shape [2, 3, 3], expected [3, 3, 3]"),
        "{message}"
    );

    let group = store.path().join("layers/counts/zarr.json");
    let text = std::fs::read_to_string(&group)
        .unwrap()
        .replace("tiled_csr", "dense");
    std::fs::write(&group, text).unwrap();
    let (found, message) = code(read_layer::<f32>(store.path(), "counts", 3, 5, TILING));
    assert_eq!(found, "KY-E107");
    assert!(message.contains("attributes.kyto.kind"), "{message}");
}
