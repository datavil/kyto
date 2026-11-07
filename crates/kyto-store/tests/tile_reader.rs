//! The tile reader reads exactly the tiles a selection touches (format.md §4.7), joins
//! them into CellBlocks over the selected genes (§4.8), gives the same stream on any
//! thread count (R11), checks `format_version` on open (§3, §11) and rejects bad
//! selections with coded errors (R10).
//!
//! Reads are counted by [`CountingStore`], a storage wrapper passed to the reader (no
//! global state, R7), and mapped back to inner chunks through each shard's index.
//!
//! Every test runs on each backend of `support/backends.rs` (task 2.8): the local
//! filesystem, HTTP, and S3 (MinIO) when configured. The reads counted are the same on
//! all of them, since remote storage reads the byte ranges zarrs asks for.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used)]

#[path = "support/backends.rs"]
mod backends;

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, Mutex};

use backends::{Backend, Storage, backends, storage};
use kyto_store::{
    AxisSelection, CellValues, CsrMatrix, LayerReader, LayerValue, LayerWriteOptions,
    SelectedBlock, TileGrid, Tiling, read_layer, write_layer,
};
use proptest::prelude::*;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::{MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey};

/// A readable store that records every byte range read, per key.
struct CountingStore<S> {
    inner: S,
    reads: Mutex<Vec<(String, ByteRange)>>,
}

impl<S> CountingStore<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            reads: Mutex::new(Vec::new()),
        }
    }

    /// The reads so far, emptying the record.
    fn take(&self) -> Vec<(String, ByteRange)> {
        std::mem::take(&mut *self.reads.lock().unwrap())
    }
}

impl<S: ReadableStorageTraits> ReadableStorageTraits for CountingStore<S> {
    // `get` and `get_partial` default to `get_partial_many`, so every read passes here.
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let ranges: Vec<ByteRange> = byte_ranges.collect();
        let mut reads = self.reads.lock().unwrap();
        reads.extend(ranges.iter().map(|range| (key.as_str().to_owned(), *range)));
        drop(reads);
        self.inner
            .get_partial_many(key, Box::new(ranges.into_iter()))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.inner.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.inner.supports_get_partial()
    }
}

const LAYER: &str = "x";

fn root_json(version: &str) -> String {
    format!(
        r#"{{"zarr_format":3,"node_type":"group","attributes":{{"kyto":{{"format_version":"{version}"}}}}}}"#
    )
}

/// Write `matrix` as layer `x` of a new store with a v0.1 root `zarr.json`.
fn write_store<T: LayerValue>(
    dir: &Path,
    matrix: &CsrMatrix<T>,
    tiling: Tiling,
    options: &LayerWriteOptions,
) {
    write_layer(dir, LAYER, matrix, tiling, options).unwrap();
    std::fs::write(dir.join("zarr.json"), root_json("0.1")).unwrap();
}

/// A dense `n_cells x n_genes` matrix with value `1 + i * n_genes + j` (no zeros), so
/// every tile is non-empty.
fn dense(n_cells: u64, n_genes: u64) -> CsrMatrix<f32> {
    let mut matrix = CsrMatrix {
        n_cells,
        n_genes,
        indptr: vec![0],
        indices: Vec::new(),
        data: Vec::new(),
    };
    for cell in 0..n_cells {
        for gene in 0..n_genes {
            matrix.indices.push(gene as u32);
            matrix.data.push((1 + cell * n_genes + gene) as f32);
        }
        matrix.indptr.push(matrix.indices.len() as u64);
    }
    matrix
}

fn open<S: ?Sized + ReadableStorageTraits + 'static>(
    storage: Arc<S>,
    n_cells: u64,
    n_genes: u64,
    tiling: Tiling,
) -> LayerReader<S> {
    let grid = TileGrid::new(n_cells, n_genes, tiling).unwrap();
    LayerReader::open(storage, "test.kyto", LAYER, grid).unwrap()
}

fn collect<S: ?Sized + ReadableStorageTraits + 'static>(
    reader: &LayerReader<S>,
    cells: &AxisSelection,
    genes: &AxisSelection,
    blocks_in_flight: usize,
) -> Vec<SelectedBlock> {
    reader
        .cell_blocks(cells, genes, NonZeroUsize::new(blocks_in_flight).unwrap())
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

/// The reference: rows `cells` and columns `genes` of `matrix`, as (cells, CSR with
/// selection positions).
fn reference<T: Copy>(
    matrix: &CsrMatrix<T>,
    cells: &[u64],
    genes: &[u64],
) -> (Vec<u64>, Vec<u64>, Vec<u32>, Vec<T>) {
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for &cell in cells {
        let row = matrix.indptr[cell as usize] as usize..matrix.indptr[cell as usize + 1] as usize;
        for at in row {
            let gene = u64::from(matrix.indices[at]);
            if let Ok(position) = genes.binary_search(&gene) {
                indices.push(position as u32);
                data.push(matrix.data[at]);
            }
        }
        indptr.push(indices.len() as u64);
    }
    (cells.to_vec(), indptr, indices, data)
}

/// Join a stream into one (cells, indptr, indices, data), checking each block's shape.
fn join(blocks: &[SelectedBlock], n_genes: u64) -> (Vec<u64>, Vec<u64>, Vec<u32>, Vec<f32>) {
    let mut cells = Vec::new();
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    let mut last_block = None;
    for SelectedBlock {
        cell_block,
        cells: block_cells,
        block,
    } in blocks
    {
        assert!(last_block < Some(*cell_block), "blocks out of cell order");
        last_block = Some(*cell_block);
        assert_eq!(block.n_genes(), n_genes);
        assert_eq!(block.n_rows(), block_cells.len());
        assert!(!block_cells.is_empty(), "a block without selected cells");
        assert_eq!(block.indptr()[0], 0);
        let offset = indices.len() as u64;
        indptr.extend(block.indptr()[1..].iter().map(|end| offset + end));
        indices.extend_from_slice(block.indices());
        match block.data() {
            CellValues::F32(values) => data.extend_from_slice(values),
            CellValues::U32(values) => data.extend(values.iter().map(|&value| value as f32)),
        }
        cells.extend_from_slice(block_cells);
    }
    (cells, indptr, indices, data)
}

/// Which part of a sharded array's shard a byte range read: its index, one inner chunk,
/// or nothing because the shard is not stored (all fill values, format.md §4.4). A
/// whole-shard read is a read of each inner chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ShardPart {
    Index,
    Inner(usize),
    Missing,
    /// Not stored in the shard (all fill values), seen in a shard index read.
    Absent(usize),
}

/// Map a read of shard file `path` to the part it reads, using the shard index at the
/// end (`[offset, nbytes]` u64 pairs, then a crc32c, format.md §4.4).
fn shard_part(path: &Path, range: ByteRange, n_inner: usize) -> Vec<ShardPart> {
    let Ok(metadata) = std::fs::metadata(path) else {
        return vec![ShardPart::Missing];
    };
    let size = metadata.len();
    let (offset, length) = match range {
        ByteRange::FromStart(offset, Some(length)) => (offset, length),
        ByteRange::FromStart(offset, None) => (offset, size - offset),
        ByteRange::Suffix(length) => (size - length, length),
    };
    if (offset, length) == (0, size) {
        return (0..n_inner).map(ShardPart::Inner).collect();
    }
    let index_len = n_inner as u64 * 16 + 4;
    let bytes = std::fs::read(path).unwrap();
    let index = &bytes[(size - index_len) as usize..];
    let word = |at: usize| u64::from_le_bytes(index[at * 8..at * 8 + 8].try_into().unwrap());
    if (offset, length) == (size - index_len, index_len) {
        let absent = (0..n_inner).filter(|&inner| word(2 * inner) == u64::MAX);
        return std::iter::once(ShardPart::Index)
            .chain(absent.map(ShardPart::Absent))
            .collect();
    }
    (0..n_inner)
        .find(|&inner| (word(2 * inner), word(2 * inner + 1)) == (offset, length))
        .map(|inner| vec![ShardPart::Inner(inner)])
        .unwrap_or_else(|| panic!("read {range:?} of {path:?} is no index or inner chunk"))
}

/// Reads per array, mapped to (shard key, part) with their counts.
type Parts = BTreeMap<String, BTreeMap<(String, ShardPart), usize>>;

fn classify(
    dir: &Path,
    reads: &[(String, ByteRange)],
    n_gene_blocks: usize,
    inner_per_shard: usize,
) -> Parts {
    let mut parts = Parts::new();
    for (key, range) in reads {
        let rest = key
            .strip_prefix(&format!("layers/{LAYER}/"))
            .unwrap_or_else(|| panic!("read outside the layer: {key}"));
        let (array, shard) = rest
            .split_once("/c/")
            .unwrap_or_else(|| panic!("not a chunk read: {key}"));
        let n_inner = match array {
            "indptr" => n_gene_blocks,
            "indices" | "data" => inner_per_shard,
            other => panic!("array {other} read while streaming"),
        };
        let found = shard_part(&dir.join(key), *range, n_inner);
        let array_parts = parts.entry(array.to_owned()).or_default();
        for part in found {
            *array_parts.entry((shard.to_owned(), part)).or_default() += 1;
        }
    }
    parts
}

/// The inner chunks of `K` entries, as `(shard, inner)` with `S / K` per shard, that
/// overlap the entry ranges `[start, end)`.
fn entry_chunks(
    ranges: &[(u64, u64)],
    k: u64,
    inner_per_shard: u64,
) -> BTreeSet<(String, ShardPart)> {
    let mut chunks = BTreeSet::new();
    for &(start, end) in ranges {
        for chunk in start / k..=(end - 1) / k {
            let shard = (chunk / inner_per_shard).to_string();
            chunks.insert((shard, ShardPart::Inner((chunk % inner_per_shard) as usize)));
        }
    }
    chunks
}

// 10 cells in cell blocks of 4 (last block 2 rows), 10 genes in gene blocks of 3 (last
// block 1 gene). Inner chunks of 2 entries, 2 inner chunks per shard.
const N_CELLS: u64 = 10;
const N_GENES: u64 = 10;
const TILING: Tiling = Tiling {
    cell_block_size: 4,
    gene_block_size: 3,
};
const OPTIONS: LayerWriteOptions = LayerWriteOptions {
    nnz_chunk_size: 2,
    nnz_shard_size: 4,
    zstd_level: 3,
};

/// Entry range `[start, end)` of tile `(b, g)` of the dense matrix.
fn dense_tile_range(block: u64, gene_block: u64) -> (u64, u64) {
    let grid = TileGrid::new(N_CELLS, N_GENES, TILING).unwrap();
    let mut start = 0;
    for b in 0..grid.n_cell_blocks() {
        for g in 0..grid.n_gene_blocks() {
            let nnz = grid.rows(b) * grid.width(g);
            if (b, g) == (block, gene_block) {
                return (start, start + nnz);
            }
            start += nnz;
        }
    }
    unreachable!()
}

/// Any backend's storage, with its reads counted.
type Counted = CountingStore<Arc<Storage>>;

fn counted_store(backend: Backend) -> (tempfile::TempDir, Arc<Counted>, LayerReader<Counted>) {
    let dir = tempfile::tempdir().unwrap();
    write_store(dir.path(), &dense(N_CELLS, N_GENES), TILING, &OPTIONS);
    let storage = Arc::new(CountingStore::new(storage(backend, dir.path())));
    let reader = open(storage.clone(), N_CELLS, N_GENES, TILING);
    let opened = storage.take();
    assert!(
        opened.iter().any(|(key, _)| key == "zarr.json"),
        "open reads the root zarr.json (format_version)"
    );
    (dir, storage, reader)
}

/// Check that streaming read exactly the tiles `cell_blocks x gene_blocks` of the
/// dense store: per tile one `indptr` shard index lookup plus its inner chunk (exact
/// counts), and for `indices` and `data` exactly the inner chunks overlapping those
/// tiles' entries, with shard index lookups only on shards read for entries.
fn assert_read_tiles(parts: &Parts, cell_blocks: &[u64], gene_blocks: &[u64]) {
    let mut indptr = BTreeMap::new();
    let mut ranges = Vec::new();
    for &b in cell_blocks {
        for &g in gene_blocks {
            let shard = format!("{b}/0/0");
            *indptr.entry((shard.clone(), ShardPart::Index)).or_default() += 1;
            *indptr
                .entry((shard, ShardPart::Inner(g as usize)))
                .or_default() += 1;
            ranges.push(dense_tile_range(b, g));
        }
    }
    let mut indptr_read = parts.get("indptr").cloned().unwrap_or_default();
    indptr_read.retain(|(_, part), _| !matches!(part, ShardPart::Absent(_)));
    assert_eq!(indptr_read, indptr, "indptr reads: {parts:#?}");
    let expected = if ranges.is_empty() {
        BTreeSet::new()
    } else {
        entry_chunks(&ranges, 2, 2)
    };
    for array in ["indices", "data"] {
        let found = parts.get(array).cloned().unwrap_or_default();
        let missing: BTreeSet<&String> = found
            .keys()
            .filter(|(_, part)| *part == ShardPart::Missing)
            .map(|(shard, _)| shard)
            .collect();
        // Inner chunks read, plus expected ones that need no read: not stored in their
        // shard (seen in its index) or in a shard that is not stored (all fill values).
        let inner: BTreeSet<_> = found
            .keys()
            .filter_map(|(shard, part)| match part {
                ShardPart::Inner(inner) => Some((shard.clone(), ShardPart::Inner(*inner))),
                ShardPart::Absent(inner) => Some((shard.clone(), ShardPart::Inner(*inner)))
                    .filter(|chunk| expected.contains(chunk)),
                _ => None,
            })
            .chain(
                expected
                    .iter()
                    .filter(|(shard, _)| missing.contains(shard))
                    .cloned(),
            )
            .collect();
        assert_eq!(inner, expected, "{array} inner chunks read");
        for (shard, part) in found.keys() {
            if !matches!(part, ShardPart::Inner(_)) {
                assert!(
                    expected.iter().any(|(read, _)| read == shard),
                    "{array}: shard {shard} read but holds no selected tile"
                );
            }
        }
    }
    let arrays: Vec<&String> = parts.keys().collect();
    assert!(
        arrays
            .iter()
            .all(|array| ["indptr", "indices", "data"].contains(&array.as_str()))
    );
}

#[test]
fn three_genes_read_one_gene_block_per_cell_block() {
    for backend in backends() {
        let (dir, storage, reader) = counted_store(backend);
        // Genes 3, 4 and 5 are gene block 1.
        let genes = AxisSelection::Indices(vec![3, 4, 5]);
        let blocks = collect(&reader, &AxisSelection::All, &genes, 2);
        assert_eq!(blocks.len(), 3);
        let parts = classify(dir.path(), &storage.take(), 4, 2);
        // Per cell block exactly one indptr shard index lookup and inner chunk (b, 1), and
        // only the entry chunks of tiles (b, 1).
        assert_read_tiles(&parts, &[0, 1, 2], &[1]);
        let full = dense(N_CELLS, N_GENES);
        let all_cells: Vec<u64> = (0..N_CELLS).collect();
        let expected = reference(&full, &all_cells, &[3, 4, 5]);
        assert_eq!(join(&blocks, 3), expected);
    }
}

#[test]
fn cell_selection_skips_cell_blocks_and_genes_in_two_blocks_read_two() {
    for backend in backends() {
        let (dir, storage, reader) = counted_store(backend);
        // Cells 1 and 9: blocks 0 and 2, block 1 is skipped. Genes 2 and 9: blocks 0 and 3.
        let cells = AxisSelection::Indices(vec![1, 9]);
        let genes = AxisSelection::Indices(vec![2, 9]);
        let blocks = collect(&reader, &cells, &genes, 1);
        assert_eq!(
            blocks
                .iter()
                .map(|block| block.cell_block)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        let parts = classify(dir.path(), &storage.take(), 4, 2);
        assert_read_tiles(&parts, &[0, 2], &[0, 3]);
        let expected = reference(&dense(N_CELLS, N_GENES), &[1, 9], &[2, 9]);
        assert_eq!(join(&blocks, 2), expected);
        // Gene 9 is position 1 of the selection.
        assert_eq!(blocks[1].block.indices(), &[0, 1]);
    }
}

#[test]
fn every_tile_is_read_once_for_a_full_read() {
    for backend in backends() {
        let (dir, storage, reader) = counted_store(backend);
        let blocks = collect(&reader, &AxisSelection::All, &AxisSelection::All, 3);
        let parts = classify(dir.path(), &storage.take(), 4, 2);
        assert_read_tiles(&parts, &[0, 1, 2], &[0, 1, 2, 3]);
        let full = dense(N_CELLS, N_GENES);
        let (cells, indptr, indices, data) = join(&blocks, N_GENES);
        assert_eq!(cells, (0..N_CELLS).collect::<Vec<_>>());
        assert_eq!(
            (indptr, indices, data),
            (full.indptr, full.indices, full.data)
        );
    }
}

#[test]
fn empty_selections_read_no_tiles() {
    for backend in backends() {
        let (_dir, storage, reader) = counted_store(backend);
        let none = AxisSelection::Indices(Vec::new());
        assert!(collect(&reader, &none, &AxisSelection::All, 2).is_empty());
        assert!(collect(&reader, &AxisSelection::Range(3..3), &AxisSelection::All, 2).is_empty());
        // No genes: one block per selected cell block, rows without entries.
        let blocks = collect(&reader, &AxisSelection::Range(2..6), &none, 2);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].cells, vec![2, 3]);
        assert_eq!(blocks[1].cells, vec![4, 5]);
        assert_eq!(blocks[0].block.indptr(), &[0, 0, 0]);
        assert_eq!(blocks[0].block.n_genes(), 0);
        assert!(blocks[1].block.data().is_empty());
        assert!(storage.take().is_empty(), "an empty selection read storage");
    }
}

#[test]
fn stream_is_identical_on_any_thread_count() {
    let dir = tempfile::tempdir().unwrap();
    let matrix = random_matrix(37, 23, 0.3, 7);
    let tiling = Tiling {
        cell_block_size: 5,
        gene_block_size: 4,
    };
    write_store(dir.path(), &matrix, tiling, &OPTIONS);
    for backend in backends() {
        let reader = open(storage(backend, dir.path()), 37, 23, tiling);
        let cells = AxisSelection::Indices((0..37).filter(|cell| cell % 3 != 1).collect());
        let genes = AxisSelection::Indices(vec![0, 3, 4, 5, 9, 16, 22]);
        let run = |threads: usize, in_flight: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| collect(&reader, &cells, &genes, in_flight))
        };
        let single = run(1, 1);
        assert_eq!(single.len(), 8);
        for (threads, in_flight) in [(1, 4), (4, 1), (4, 3), (8, 8), (3, 100)] {
            assert_eq!(
                run(threads, in_flight),
                single,
                "{backend:?}: {threads} threads, {in_flight} in flight"
            );
        }
    }
}

#[test]
fn zero_cell_and_zero_gene_stores() {
    for (n_cells, n_genes) in [(0, 5), (6, 0), (0, 0)] {
        let dir = tempfile::tempdir().unwrap();
        write_store(dir.path(), &dense(n_cells, n_genes), TILING, &OPTIONS);
        for backend in backends() {
            let reader = open(storage(backend, dir.path()), n_cells, n_genes, TILING);
            let blocks = collect(&reader, &AxisSelection::All, &AxisSelection::All, 2);
            assert_eq!(
                blocks.len() as u64,
                n_cells.div_ceil(4),
                "{backend:?}: {n_cells} x {n_genes}"
            );
            for block in &blocks {
                assert_eq!(block.block.nnz(), 0);
                assert_eq!(block.block.n_genes(), 0);
            }
        }
    }
}

#[test]
fn uint32_layers_stream_uint32_values() {
    let dir = tempfile::tempdir().unwrap();
    let matrix = CsrMatrix::<u32> {
        n_cells: 3,
        n_genes: 4,
        indptr: vec![0, 2, 2, 3],
        indices: vec![0, 3, 2],
        data: vec![5, 6, 7],
    };
    write_store(dir.path(), &matrix, TILING, &OPTIONS);
    for backend in backends() {
        let reader = open(storage(backend, dir.path()), 3, 4, TILING);
        let blocks = collect(&reader, &AxisSelection::All, &AxisSelection::Range(2..4), 1);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].block.indptr(), &[0, 1, 1, 2]);
        assert_eq!(blocks[0].block.indices(), &[1, 0]);
        assert_eq!(blocks[0].block.data(), CellValues::U32(&[6, 7]));
    }
}

fn error_of<T>(result: Result<T, kyto_error::KytoError>) -> kyto_error::KytoError {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(error) => error,
    }
}

#[test]
fn bad_selections_fail_before_reading() {
    for backend in backends() {
        let (_dir, storage, reader) = counted_store(backend);
        let one = NonZeroUsize::new(1).unwrap();
        let cases = [
            (
                AxisSelection::All,
                AxisSelection::Indices(vec![2, 10]),
                "invalid genes selection: position 1 is 10, not below the axis length 10",
            ),
            (
                AxisSelection::All,
                AxisSelection::Indices(vec![5, 3]),
                "position 1 is 3, which does not follow 5",
            ),
            (
                AxisSelection::All,
                AxisSelection::Indices(vec![4, 4]),
                "position 1 repeats 4",
            ),
            (
                AxisSelection::Range(0..11),
                AxisSelection::All,
                "invalid cells selection: range 0..11 ends after the axis length 10",
            ),
            (
                AxisSelection::Indices(vec![9, 0]),
                AxisSelection::All,
                "invalid cells selection: position 1 is 0, which does not follow 9",
            ),
        ];
        for (cells, genes, message) in cases {
            let error = error_of(reader.cell_blocks(&cells, &genes, one));
            assert_eq!(error.code(), "KY-E114");
            assert!(error.to_string().contains(message), "{error}");
            assert!(error.hint().contains("sorted, unique"), "{}", error.hint());
        }
        assert!(storage.take().is_empty());
    }
}

#[test]
fn open_checks_format_version_and_layer() {
    for backend in backends() {
        let dir = tempfile::tempdir().unwrap();
        write_layer(dir.path(), LAYER, &dense(3, 3), TILING, &OPTIONS).unwrap();
        let grid = TileGrid::new(3, 3, TILING).unwrap();
        let try_open = |layer: &str| {
            LayerReader::open(storage(backend, dir.path()), "test.kyto", layer, grid).map(|_| ())
        };
        let error = error_of(try_open(LAYER));
        assert_eq!(error.code(), "KY-E116");
        assert!(
            error
                .to_string()
                .contains("'test.kyto' is not a kyto store: no root zarr.json")
        );
        let root = dir.path().join("zarr.json");
        std::fs::write(
            &root,
            r#"{"zarr_format":3,"node_type":"group","attributes":{}}"#,
        )
        .unwrap();
        assert!(
            error_of(try_open(LAYER))
                .to_string()
                .contains("no string attributes.kyto.format_version")
        );
        std::fs::write(&root, "not json").unwrap();
        assert!(
            error_of(try_open(LAYER))
                .to_string()
                .contains("not valid JSON")
        );
        for version in ["0.2", "1.0", "0.10", ""] {
            std::fs::write(&root, root_json(version)).unwrap();
            let error = error_of(try_open(LAYER));
            assert_eq!(error.code(), "KY-E115", "{version}");
            assert!(
                error.to_string().contains(&format!(
                    "has format_version '{version}', but this kyto reads only '0.1'"
                )),
                "{error}"
            );
        }
        std::fs::write(&root, root_json("0.1")).unwrap();
        try_open(LAYER).unwrap();
        assert_eq!(error_of(try_open("y")).code(), "KY-E109");
        assert_eq!(error_of(try_open("a/b")).code(), "KY-E102");
    }
}

#[test]
fn corrupt_tile_is_reported_and_ends_the_stream() {
    let dir = tempfile::tempdir().unwrap();
    write_store(dir.path(), &dense(N_CELLS, N_GENES), TILING, &OPTIONS);
    // Replace tile (1, 0)'s indptr inner chunk with that of tile (1, 3) (narrower, fewer
    // entries) by swapping shards: cell block 1's indptr shard becomes block 2's.
    let indptr = dir.path().join(format!("layers/{LAYER}/indptr/c"));
    std::fs::copy(indptr.join("2/0/0"), indptr.join("1/0/0")).unwrap();
    for backend in backends() {
        let reader = open(storage(backend, dir.path()), N_CELLS, N_GENES, TILING);
        let stream: Vec<_> = reader
            .cell_blocks(
                &AxisSelection::All,
                &AxisSelection::Indices(vec![0]),
                NonZeroUsize::new(3).unwrap(),
            )
            .unwrap()
            .collect();
        assert_eq!(stream.len(), 2, "the stream ends after the error");
        assert!(stream[0].is_ok());
        let error = stream[1].as_ref().unwrap_err();
        assert_eq!(error.code(), "KY-E107");
        assert!(
            error.to_string().contains("tile (1, 0)"),
            "{backend:?}: {error}"
        );
    }
}

/// A random matrix of the given density, from its own seeded generator (memory/test_seed_isolation.md).
fn random_matrix(n_cells: u64, n_genes: u64, density: f64, seed: u64) -> CsrMatrix<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut matrix = CsrMatrix {
        n_cells,
        n_genes,
        indptr: vec![0],
        indices: Vec::new(),
        data: Vec::new(),
    };
    for _ in 0..n_cells {
        for gene in 0..n_genes {
            if next() < density {
                matrix.indices.push(gene as u32);
                matrix.data.push((next() * 100.0) as f32 + 1.0);
            }
        }
        matrix.indptr.push(matrix.indices.len() as u64);
    }
    matrix
}

fn subset(n: u64, mask: &[bool]) -> Vec<u64> {
    (0..n)
        .filter(|&at| mask[at as usize % mask.len()])
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Any canonical selection gives the rows and columns of the whole layer.
    #[test]
    fn selections_match_the_whole_layer(
        n_cells in 0_u64..30,
        n_genes in 0_u64..20,
        cell_block_size in 1_u64..8,
        gene_block_size in 1_u64..7,
        density in 0.0_f64..0.8,
        seed in any::<u64>(),
        cell_mask in proptest::collection::vec(any::<bool>(), 1..9),
        gene_mask in proptest::collection::vec(any::<bool>(), 1..9),
        as_range in any::<bool>(),
        in_flight in 1_usize..5,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let tiling = Tiling { cell_block_size, gene_block_size };
        let matrix = random_matrix(n_cells, n_genes, density, seed);
        write_store(dir.path(), &matrix, tiling, &OPTIONS);
        let whole: CsrMatrix<f32> = read_layer(dir.path(), LAYER, n_cells, n_genes, tiling).unwrap();
        let cells = subset(n_cells, &cell_mask);
        let genes = subset(n_genes, &gene_mask);
        let (cell_selection, cell_list) = if as_range {
            let end = n_cells.min(cell_mask.len() as u64 * 3);
            let start = end / 3;
            (AxisSelection::Range(start..end), (start..end).collect())
        } else {
            (AxisSelection::Indices(cells.clone()), cells)
        };
        let expected = reference(&whole, &cell_list, &genes);
        for backend in backends() {
            let reader = open(storage(backend, dir.path()), n_cells, n_genes, tiling);
            let blocks = collect(&reader, &cell_selection, &AxisSelection::Indices(genes.clone()), in_flight);
            prop_assert_eq!(join(&blocks, genes.len() as u64), expected.clone(), "{:?}", backend);
        }
    }
}

/// Two corrupt tiles in one cell block: the error names the first one on any thread
/// count (R11).
#[test]
fn first_corrupt_tile_is_reported_on_any_thread_count() {
    let dir = tempfile::tempdir().unwrap();
    let tiling = Tiling {
        cell_block_size: 1,
        gene_block_size: 1,
    };
    write_store(dir.path(), &dense(1, 64), tiling, &OPTIONS);
    // Tiles 5 and 40 claim 2 entries, tiles 6 and 41 none: still monotonic, ends at nnz.
    let mut offsets: Vec<u64> = (0..=64).collect();
    offsets[6] = 7;
    offsets[41] = 42;
    let local = Arc::new(FilesystemStore::new(dir.path()).unwrap());
    let array = zarrs::array::Array::open(local, &format!("/layers/{LAYER}/tile_offsets")).unwrap();
    let mut chunk = offsets.clone();
    chunk.resize(65536, 0);
    array.store_chunk_elements(&[0], &chunk).unwrap();
    for backend in backends() {
        let reader = open(storage(backend, dir.path()), 1, 64, tiling);
        for threads in [1, 2, 8, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for _ in 0..20 {
                let first = pool.install(|| {
                    reader
                        .cell_blocks(
                            &AxisSelection::All,
                            &AxisSelection::All,
                            NonZeroUsize::new(1).unwrap(),
                        )
                        .unwrap()
                        .next()
                        .unwrap()
                });
                let error = first.unwrap_err();
                assert_eq!(error.code(), "KY-E107");
                assert!(
                    error.to_string().contains("tile (0, 5)"),
                    "{backend:?}, {threads} threads: {error}"
                );
            }
        }
    }
}
