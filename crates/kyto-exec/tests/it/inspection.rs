//! Task 8.3: inspection without reading the matrix.
//!
//! Done when: the I/O counter shows zero matrix-tile reads for each call. `schema()`,
//! `features()`, `estimate()` and `explain()` are given the plan alone (kyto-plan
//! cannot open a store), so the one call that touches the store is `peek(n)`. Here its
//! output (`Plan::peek_output`) runs through the executor over a storage wrapper that
//! records every key read (memory/counting_tile_reads.md): a plan with a metadata
//! filter and a computed column reads nothing under `layers/`, `embeddings/` or
//! `graphs/`, while the same plan's layer output does (so the counter sees tile
//! reads). The Python calls are checked in `tests/test_inspection.py`.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, Mutex};

use kyto_core::{BinaryOp, CellExpr, Expr, GeneIds, Literal, Tiling};
use kyto_exec::{ExecuteOptions, Pipeline, PipelineBlock};
use kyto_plan::{Member, OptimizeOptions, Output, PhysicalOptions, Plan, physical_plan};
use kyto_store::{
    CellBlock, LayerSpec, LayerWriteOptions, StoreWriter, TableAxis, ValueDtype, open_atlas_schema,
    read_ids,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::byte_range::ByteRangeIterator;
use zarrs::storage::{MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey};

const N_CELLS: usize = 90;
const N_GENES: usize = 12;
/// Small tiles, so the store has several cell blocks.
const TILING: Tiling = Tiling {
    cell_block_size: 16,
    gene_block_size: 8,
};

/// The I/O counter: every key read through the storage.
struct CountingStore<S> {
    inner: S,
    keys: Mutex<Vec<String>>,
}

impl<S> CountingStore<S> {
    fn new(inner: S) -> Arc<Self> {
        Arc::new(Self {
            inner,
            keys: Mutex::new(Vec::new()),
        })
    }

    /// The keys read since the last call that hold matrix data: tiles of a layer, or
    /// chunks of an embedding or a graph.
    fn take_data_reads(&self) -> BTreeSet<String> {
        std::mem::take(&mut *self.keys.lock().unwrap())
            .into_iter()
            .filter(|key| {
                ["layers/", "embeddings/", "graphs/"]
                    .iter()
                    .any(|group| key.starts_with(group))
            })
            .collect()
    }
}

impl<S: ReadableStorageTraits> ReadableStorageTraits for CountingStore<S> {
    // `get` and `get_partial` default to `get_partial_many`, so every read passes here.
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.keys.lock().unwrap().push(key.as_str().to_owned());
        self.inner.get_partial_many(key, byte_ranges)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.keys.lock().unwrap().push(key.as_str().to_owned());
        self.inner.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.inner.supports_get_partial()
    }
}

/// Cell `cell`'s `score`: 0, 1 or 2.
fn score(cell: usize) -> i32 {
    (cell % 3) as i32
}

/// A store of `N_CELLS` x `N_GENES` with the layer `counts` and the cell column `score`.
fn write_store(target: &Path) {
    let gene_ids: Vec<String> = (0..N_GENES).map(|gene| format!("G{gene}")).collect();
    let cell_ids: Vec<String> = (0..N_CELLS).map(|cell| format!("c{cell}")).collect();
    let genes = DataFrame::new(
        N_GENES,
        vec![Series::new("gene_id".into(), gene_ids).into_column()],
    )
    .unwrap();
    let cells = DataFrame::new(
        N_CELLS,
        vec![
            Series::new("cell_id".into(), cell_ids).into_column(),
            Series::new("score".into(), (0..N_CELLS).map(score).collect::<Vec<_>>()).into_column(),
        ],
    )
    .unwrap();
    let specs = [LayerSpec {
        name: "counts".to_owned(),
        dtype: ValueDtype::U32,
        options: LayerWriteOptions::default(),
    }];
    let mut writer = StoreWriter::create(target, &genes, &specs, TILING, false).unwrap();
    let mut indptr = vec![0_u64];
    let (mut indices, mut data) = (Vec::new(), Vec::new());
    for cell in 0..N_CELLS {
        for gene in 0..N_GENES {
            if (cell + gene) % 4 == 0 {
                indices.push(gene as u32);
                data.push(1 + (cell % 5) as u32);
            }
        }
        indptr.push(indices.len() as u64);
    }
    writer
        .push(&[CellBlock::from_u32(indptr, indices, data, N_GENES as u64).unwrap()])
        .unwrap();
    writer.finish(&cells, None).unwrap();
}

fn scan(store: &Path) -> Arc<Plan> {
    let schema = open_atlas_schema(store).unwrap();
    let gene_ids = GeneIds::new(read_ids(store, TableAxis::Genes, schema.n_genes).unwrap());
    Plan::scan(store.display().to_string(), schema, gene_ids)
}

/// Run `output` over `storage`; its blocks and the matrix data keys it read.
fn run(
    output: Output,
    storage: &Arc<CountingStore<FilesystemStore>>,
    threads: usize,
) -> (Vec<PipelineBlock>, BTreeSet<String>) {
    let physical = physical_plan(&[output], OptimizeOptions::ALL, PhysicalOptions::default());
    storage.take_data_reads();
    let stream = Pipeline::new(&physical, 0)
        .unwrap()
        .run_on(
            Arc::clone(storage),
            ExecuteOptions::new(NonZeroUsize::new(threads).unwrap()),
        )
        .unwrap();
    let blocks = stream.collect::<Result<Vec<_>, _>>().unwrap();
    (blocks, storage.take_data_reads())
}

#[test]
fn peek_reads_no_matrix_tile() {
    let folder = tempfile::tempdir().unwrap();
    let store = folder.path().join("store.kyto");
    write_store(&store);
    let storage = CountingStore::new(FilesystemStore::new(&store).unwrap());

    let high = Expr::binary(CellExpr::column("score"), BinaryOp::Gt, Literal::Int(0)).unwrap();
    let doubled = Expr::binary(CellExpr::column("score"), BinaryOp::Mul, Literal::Int(2)).unwrap();
    // The layer steps change no cell column: `peek` leaves them out.
    let plan = scan(&store)
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .filter_cells(&high)
        .unwrap()
        .with_cells([("doubled".to_owned(), doubled)])
        .unwrap()
        .log1p(Some("norm"), "log1p")
        .unwrap();

    for (threads, peeked) in [(1, &scan(&store)), (1, &plan), (4, &plan)] {
        let (blocks, data_reads) = run(peeked.peek_output().unwrap(), &storage, threads);
        assert_eq!(
            data_reads,
            BTreeSet::new(),
            "peek read matrix data on {threads} threads"
        );
        let cells: Vec<u64> = blocks
            .iter()
            .flat_map(|block| block.cells.iter().copied())
            .collect();
        let kept = if Arc::ptr_eq(peeked, &plan) {
            (0..N_CELLS as u64)
                .filter(|&cell| score(cell as usize) > 0)
                .collect()
        } else {
            (0..N_CELLS as u64).collect::<Vec<_>>()
        };
        assert_eq!(cells, kept, "the cells of the plan, in store order");
    }

    // The counter does see tile reads: the same plan's layer output reads the layer.
    let layer = Output::members(&plan, [Member::Layer("log1p".to_owned())]).unwrap();
    let (_, data_reads) = run(layer, &storage, 1);
    assert!(
        data_reads
            .iter()
            .any(|key| key.starts_with("layers/counts/data/c/")),
        "{data_reads:?}"
    );
}
