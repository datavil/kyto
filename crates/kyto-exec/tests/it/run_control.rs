//! Task 5.7: cancelling a run and following its progress through a `RunControl`.
//!
//! On a small store split into 32 blocks of 64 cells: a full run of a median
//! `normalize_total` then `log1p` (one statistics pass, then the output's stream)
//! reports every cell twice; a run cancelled while the output streams ends with one
//! `KY-E503` after at most a window of blocks; a run cancelled in its statistics pass
//! (deterministically, by a storage wrapper that cancels at its Nth tile read) returns
//! `KY-E503` from `run_on` and reads nothing more; a run cancelled before it starts
//! reads nothing. After each cancel, a new run over the same store streams every cell.
//! The Python side (Ctrl-C, the progress callback) is `tests/test_run_control.py`.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use kyto_core::{GeneIds, Tiling};
use kyto_error::KytoError;
use kyto_exec::{ExecuteOptions, Pipeline, Progress, RunControl};
use kyto_plan::{
    Member, OptimizeOptions, Output, PhysicalOptions, PhysicalPlan, Plan, physical_plan,
};
use kyto_store::{
    CellBlock, LayerSpec, LayerWriteOptions, StoreWriter, TableAxis, ValueDtype, open_atlas_schema,
    read_ids,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::byte_range::ByteRangeIterator;
use zarrs::storage::{MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey};

const N_CELLS: u64 = 2_048;
const N_GENES: usize = 64;
const BLOCK_CELLS: u64 = 64;
const N_BLOCKS: u64 = N_CELLS / BLOCK_CELLS;
const IN_FLIGHT: u64 = 2;

/// Cells `c` hold gene `g` with value `1 + (c + g) % 7` where `(c * 3 + g) % 5 == 0`.
fn write_store(target: &Path) {
    let gene_ids: Vec<String> = (0..N_GENES).map(|gene| format!("G{gene}")).collect();
    let cell_ids: Vec<String> = (0..N_CELLS).map(|cell| format!("c{cell}")).collect();
    let genes = DataFrame::new(
        N_GENES,
        vec![Series::new("gene_id".into(), gene_ids).into_column()],
    )
    .unwrap();
    let cells = DataFrame::new(
        N_CELLS as usize,
        vec![Series::new("cell_id".into(), cell_ids).into_column()],
    )
    .unwrap();
    let specs = [LayerSpec {
        name: "counts".to_owned(),
        dtype: ValueDtype::U32,
        options: LayerWriteOptions::default(),
    }];
    let mut writer = StoreWriter::create(target, &genes, &specs, Tiling::default(), false).unwrap();
    let mut indptr = vec![0_u64];
    let (mut indices, mut data) = (Vec::new(), Vec::new());
    for cell in 0..N_CELLS {
        for gene in 0..N_GENES as u64 {
            if (cell * 3 + gene) % 5 == 0 {
                indices.push(gene as u32);
                data.push(1 + ((cell + gene) % 7) as u32);
            }
        }
        indptr.push(indices.len() as u64);
    }
    let block = CellBlock::from_u32(indptr, indices, data, N_GENES as u64).unwrap();
    writer.push(&[block]).unwrap();
    writer.finish(&cells, None).unwrap();
}

/// A median `normalize_total` then `log1p`, reading `log1p`, in blocks of 64 cells:
/// one statistics pass, then the output's stream.
fn physical(store: &Path) -> PhysicalPlan {
    let schema = open_atlas_schema(store).unwrap();
    let gene_ids = GeneIds::new(read_ids(store, TableAxis::Genes, schema.n_genes).unwrap());
    let plan = Plan::scan(store.display().to_string(), schema, gene_ids)
        .normalize_total(None, Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "log1p")
        .unwrap();
    let output = Output::members(&plan, [Member::Layer("log1p".to_owned())]).unwrap();
    let options = PhysicalOptions {
        blocks_in_flight: NonZeroU64::new(IN_FLIGHT).unwrap(),
        max_block_cells: NonZeroU64::new(BLOCK_CELLS),
        ..PhysicalOptions::default()
    };
    physical_plan(&[output], OptimizeOptions::ALL, options)
}

fn options(control: &RunControl) -> ExecuteOptions {
    ExecuteOptions::new(NonZeroUsize::new(2).unwrap()).with_control(control.clone())
}

/// Cancels `control` at its `cancel_at`th read of a tile's `data` chunk.
struct CancellingStore {
    inner: FilesystemStore,
    control: RunControl,
    cancel_at: u64,
    data_reads: AtomicU64,
}

impl ReadableStorageTraits for CancellingStore {
    // `get` and `get_partial` default to `get_partial_many`, so every read passes here.
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let name = key.as_str();
        if name.starts_with("layers/") && name.contains("/data/c/") {
            let reads = self.data_reads.fetch_add(1, Ordering::SeqCst) + 1;
            if reads == self.cancel_at {
                self.control.cancel();
            }
        }
        self.inner.get_partial_many(key, byte_ranges)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.inner.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.inner.supports_get_partial()
    }
}

/// Drain a stream: its blocks' cells, then its error, if any.
fn drain(stream: kyto_exec::BlockStream) -> (u64, Option<KytoError>) {
    let mut stream = stream;
    let mut cells = 0;
    while let Some(item) = stream.next() {
        match item {
            Ok(block) => {
                cells += block.cells.len() as u64;
                stream.recycle(block);
            }
            Err(error) => {
                assert!(stream.next().is_none(), "the stream ends after an error");
                return (cells, Some(error));
            }
        }
    }
    (cells, None)
}

/// A new run of `physical` on its own control streams every cell, with both passes in
/// its progress.
fn assert_runs_to_the_end(physical: &PhysicalPlan) {
    let control = RunControl::new();
    let stream = Pipeline::new(physical, 0)
        .unwrap()
        .run(options(&control))
        .unwrap();
    let (cells, error) = drain(stream);
    assert!(error.is_none(), "{error:?}");
    assert_eq!(cells, N_CELLS);
    assert_eq!(
        control.progress(),
        Progress {
            done: 2 * N_CELLS,
            total: 2 * N_CELLS
        }
    );
    assert!(!control.is_cancelled());
}

#[test]
fn a_full_run_reports_every_cell_of_every_pass() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    assert_eq!(
        Pipeline::new(&physical(&store), 0)
            .unwrap()
            .block_cells()
            .get(),
        BLOCK_CELLS
    );
    assert_runs_to_the_end(&physical(&store));
}

#[test]
fn cancelling_the_output_stream_ends_it_with_ky_e503() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let physical = physical(&store);
    for threads in [1, 2, 4] {
        let control = RunControl::new();
        let options =
            ExecuteOptions::new(NonZeroUsize::new(threads).unwrap()).with_control(control.clone());
        let mut stream = Pipeline::new(&physical, 0).unwrap().run(options).unwrap();
        let first = stream.next().unwrap().unwrap();
        assert_eq!(first.cells.len() as u64, BLOCK_CELLS);
        stream.recycle(first);
        control.cancel();
        let (cells, error) = drain(stream);
        // At most the window computed before the cancel was seen, and the one after.
        assert!(
            cells <= (IN_FLIGHT + 1) * BLOCK_CELLS,
            "{cells} cells after cancel"
        );
        let Some(KytoError::Cancelled { done, total }) = error else {
            panic!("expected KY-E503, got {error:?}")
        };
        assert_eq!(total, 2 * N_CELLS);
        assert!((N_CELLS..2 * N_CELLS).contains(&done), "{done}");
        assert!(control.progress().done < 2 * N_CELLS);
        assert_runs_to_the_end(&physical);
    }
}

#[test]
fn cancelling_a_statistics_pass_stops_the_run_before_its_output_stream() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let physical = physical(&store);
    // Count the `data` reads of one 64-cell block: a whole run reads every block twice.
    let counting = Arc::new(CancellingStore {
        inner: FilesystemStore::new(&store).unwrap(),
        control: RunControl::new(),
        cancel_at: u64::MAX,
        data_reads: AtomicU64::new(0),
    });
    let stream = Pipeline::new(&physical, 0)
        .unwrap()
        .run_on(Arc::clone(&counting), options(&RunControl::new()))
        .unwrap();
    assert_eq!(drain(stream).0, N_CELLS);
    let whole_run = counting.data_reads.load(Ordering::SeqCst);
    assert_eq!(whole_run % (2 * N_BLOCKS), 0, "{whole_run} data reads");
    let per_block = whole_run / (2 * N_BLOCKS);
    // Cancel while the statistics pass reads its 5th block.
    let control = RunControl::new();
    let storage = Arc::new(CancellingStore {
        inner: FilesystemStore::new(&store).unwrap(),
        control: control.clone(),
        cancel_at: 4 * per_block + 1,
        data_reads: AtomicU64::new(0),
    });
    let result = Pipeline::new(&physical, 0)
        .unwrap()
        .run_on(Arc::clone(&storage), options(&control));
    let Err(KytoError::Cancelled { done, total }) = result else {
        panic!("expected KY-E503 from the statistics pass")
    };
    assert_eq!(total, 2 * N_CELLS);
    // The cancel is seen at the next block read or computed: at most the window.
    assert!(done <= 4 * BLOCK_CELLS, "{done}");
    // The reader reads a window of blocks at once (in parallel): the cancel lets the
    // window it is in finish, and no other.
    let reads = storage.data_reads.load(Ordering::SeqCst);
    assert!(
        reads <= (4 + IN_FLIGHT) * per_block,
        "{reads} data reads: the pass read on after the cancel"
    );
    assert_runs_to_the_end(&physical);
}

#[test]
fn a_run_cancelled_before_it_starts_reads_nothing() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let physical = physical(&store);
    let control = RunControl::new();
    control.cancel();
    let storage = Arc::new(CancellingStore {
        inner: FilesystemStore::new(&store).unwrap(),
        control: control.clone(),
        cancel_at: u64::MAX,
        data_reads: AtomicU64::new(0),
    });
    let result = Pipeline::new(&physical, 0)
        .unwrap()
        .run_on(Arc::clone(&storage), options(&control));
    assert!(matches!(result, Err(KytoError::Cancelled { done: 0, total }) if total == 2 * N_CELLS));
    assert_eq!(storage.data_reads.load(Ordering::SeqCst), 0);
    assert_runs_to_the_end(&physical);
}
