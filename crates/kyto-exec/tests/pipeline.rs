//! Task 5.2: the pipeline executor on a store written for the test.
//!
//! The store has 1000 cells x 300 genes of seeded `uint32` counts (an all-zero cell
//! included) in 64 x 128 tiles, so 16 cell blocks of 3 tiles each. Checked: the stream
//! of `normalize_total(1e4)` then `log1p` equals the same maps computed by hand, in cell
//! order, bit for bit, on 1 to 8 threads and 1 to 5 blocks in flight (R11); the fused
//! pass (every optimizer rule) equals the maps run one by one (no rule), bit for bit
//! (task 4.4 Open (1) and (2) for these nodes); nodes the executor does not run yet give
//! `KY-E601` before anything is read; a slow consumer stops the reading (backpressure);
//! dropping the stream early stops the producer.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use kyto_core::{GeneIds, Tiling};
use kyto_exec::{CellValues, ExecuteOptions, Pipeline, PipelineBlock};
use kyto_plan::{
    HvgFlavor, Member, Operation, OptimizeOptions, Output, PhysicalOptions, PhysicalPlan, Plan,
    physical_plan,
};
use kyto_store::{
    CellBlock, LayerSpec, LayerWriteOptions, StoreWriter, TableAxis, ValueDtype, open_atlas_schema,
    read_ids,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::{MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey};

const N_CELLS: usize = 1000;
const N_GENES: usize = 300;
const ZERO_CELL: usize = 5;
const TILING: Tiling = Tiling {
    cell_block_size: 64,
    gene_block_size: 128,
};
const TARGET_SUM: f64 = 1e4;

/// A small deterministic generator (memory/test_seed_isolation.md).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

/// The counts, one sparse row per cell: (gene, count) with increasing genes.
fn counts() -> Vec<Vec<(u32, u32)>> {
    let mut random = Lcg(7);
    (0..N_CELLS)
        .map(|cell| {
            (0..N_GENES as u32)
                .filter_map(|gene| {
                    let draw = random.next();
                    (cell != ZERO_CELL && draw.is_multiple_of(10))
                        .then_some((gene, 1 + (draw / 10 % 40) as u32))
                })
                .collect()
        })
        .collect()
}

/// Write the counts as a store at `target`.
fn write_store(target: &Path) {
    let rows = counts();
    let gene_ids: Vec<String> = (0..N_GENES).map(|gene| format!("G{gene}")).collect();
    let cell_ids: Vec<String> = (0..N_CELLS).map(|cell| format!("c{cell}")).collect();
    let genes = DataFrame::new(
        N_GENES,
        vec![Series::new("gene_id".into(), gene_ids).into_column()],
    )
    .unwrap();
    let cells = DataFrame::new(
        N_CELLS,
        vec![Series::new("cell_id".into(), cell_ids).into_column()],
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
    for row in &rows {
        for &(gene, count) in row {
            indices.push(gene);
            data.push(count);
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

/// `normalize_total(1e4)` of `counts` into `norm`, then `log1p` of `norm` into `log1p`.
fn preprocess(store: &Path) -> Arc<Plan> {
    scan(store)
        .normalize_total(Some(TARGET_SUM), Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "log1p")
        .unwrap()
}

/// The physical plan of `plan` reading only layer `log1p`.
fn physical(plan: &Arc<Plan>, rules: OptimizeOptions, blocks_in_flight: u64) -> PhysicalPlan {
    let output = Output::members(plan, [Member::Layer("log1p".to_owned())]).unwrap();
    physical_plan(
        &[output],
        rules,
        PhysicalOptions {
            blocks_in_flight: NonZeroU64::new(blocks_in_flight).unwrap(),
            ..PhysicalOptions::default()
        },
    )
}

fn threads(threads: usize) -> ExecuteOptions {
    ExecuteOptions::new(NonZeroUsize::new(threads).unwrap())
}

/// One row of a layer: (gene, value bits).
type Row = Vec<(u32, u32)>;

/// Every row of `layer`, in stream order, and the cell positions and block numbers.
fn rows_of(blocks: &[PipelineBlock], layer: &str) -> (Vec<u64>, Vec<u64>, Vec<Row>) {
    let mut cells = Vec::new();
    let mut numbers = Vec::new();
    let mut rows = Vec::new();
    for block in blocks {
        numbers.push(block.cell_block);
        cells.extend(&block.cells);
        let layer = &block.layers[layer];
        let CellValues::F32(values) = layer.data() else {
            panic!("{layer:?} is not float32")
        };
        for row in layer.indptr().windows(2) {
            let range = row[0] as usize..row[1] as usize;
            rows.push(
                layer.indices()[range.clone()]
                    .iter()
                    .zip(&values[range])
                    .map(|(&gene, value)| (gene, value.to_bits()))
                    .collect(),
            );
        }
    }
    (cells, numbers, rows)
}

/// `log1p(normalize_total(counts, 1e4))` by hand, each map's output rounded to f32,
/// `normalize_total` in scanpy's float32 steps (task 6.2).
fn expected_rows() -> Vec<Row> {
    counts()
        .iter()
        .map(|row| {
            let total: f64 = row.iter().map(|&(_, count)| f64::from(count)).sum();
            let divisor = total as f32 / TARGET_SUM as f32;
            let divisor = if divisor == 0.0 { 1.0 } else { divisor };
            row.iter()
                .map(|&(gene, count)| {
                    let normalized = count as f32 / divisor;
                    (gene, kyto_exec::log1p(normalized).to_bits())
                })
                .collect()
        })
        .collect()
}

fn collect(plan: &PhysicalPlan, options: ExecuteOptions) -> Vec<PipelineBlock> {
    Pipeline::new(plan, 0)
        .unwrap()
        .run(options)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

#[test]
fn stream_equals_the_maps_by_hand_on_any_thread_count() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let plan = preprocess(&store);
    let expected = expected_rows();
    let n_blocks = N_CELLS.div_ceil(64) as u64;
    for blocks_in_flight in [1, 2, 5] {
        let physical = physical(&plan, OptimizeOptions::ALL, blocks_in_flight);
        for thread_count in [1, 2, 4, 8] {
            let blocks = collect(&physical, threads(thread_count));
            let (cells, numbers, rows) = rows_of(&blocks, "log1p");
            assert_eq!(numbers, (0..n_blocks).collect::<Vec<_>>());
            assert_eq!(cells, (0..N_CELLS as u64).collect::<Vec<_>>());
            assert!(
                rows == expected,
                "threads={thread_count} in_flight={blocks_in_flight}"
            );
            // Only what the output reads comes out.
            assert!(blocks.iter().all(|block| block.layers.keys().eq(["log1p"])));
            assert!(rows[ZERO_CELL].is_empty());
        }
    }
}

#[test]
fn fused_pass_equals_the_maps_one_by_one() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let plan = preprocess(&store);
    let fused = physical(&plan, OptimizeOptions::ALL, 2);
    let unfused = physical(&plan, OptimizeOptions::NONE, 2);
    let is_fused = |plan: &PhysicalPlan| {
        plan.optimized()
            .nodes()
            .iter()
            .any(|node| matches!(node.operation(), Operation::Fused { .. }))
    };
    assert!(is_fused(&fused) && !is_fused(&unfused));
    let fused_blocks = collect(&fused, threads(4));
    let unfused_blocks = collect(&unfused, threads(3));
    assert_eq!(
        rows_of(&fused_blocks, "log1p"),
        rows_of(&unfused_blocks, "log1p")
    );
    // Only what the output reads comes out, fused or not.
    assert!(
        unfused_blocks
            .iter()
            .all(|block| block.layers.keys().eq(["log1p"]))
    );
}

#[test]
fn a_scan_streams_the_stored_counts() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let plan = scan(&store);
    let output = Output::members(&plan, [Member::Layer("counts".to_owned())]).unwrap();
    let physical = physical_plan(&[output], OptimizeOptions::ALL, PhysicalOptions::default());
    let pipeline = Pipeline::new(&physical, 0).unwrap();
    assert_eq!(pipeline.layers(), ["counts"]);
    let mut stream = pipeline.run(threads(2)).unwrap();
    let mut rows = Vec::new();
    while let Some(block) = stream.next() {
        let block = block.unwrap();
        let counts = &block.layers["counts"];
        let CellValues::U32(values) = counts.data() else {
            panic!("counts are uint32")
        };
        for row in counts.indptr().windows(2) {
            let range = row[0] as usize..row[1] as usize;
            rows.push(
                counts.indices()[range.clone()]
                    .iter()
                    .copied()
                    .zip(values[range].iter().copied())
                    .collect::<Vec<_>>(),
            );
        }
        stream.recycle(block);
    }
    assert_eq!(rows, counts());
    // Recycled blocks are kept for later ones.
    assert!(stream.pool().free_blocks() > 0);
}

#[test]
fn nodes_not_run_yet_are_refused_before_reading() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let base = scan(&store);
    let scale = base.scale(Some("counts"), "scaled", true, None).unwrap();
    let batched = base
        .highly_variable_genes(Some("counts"), HvgFlavor::SeuratV3, None, Some("cell_id"))
        .unwrap();
    // `select_genes` and the other 5.8 steps run now; sinks wait for 5.9.
    let sunk = base
        .sink(kyto_plan::Sink::Zarr {
            path: work.path().join("out.kyto").display().to_string(),
        })
        .unwrap();
    let pca = base.pca(Some("counts"), "X_pca", 2, 0).unwrap();
    for (plan, method, task) in [
        (&scale, "scale", "6.4"),
        (&batched, "batch_key", "6.3"),
        (&sunk, "sink_zarr", "5.9"),
        (&pca, "pca", "6.5"),
    ] {
        let physical = PhysicalPlan::new(plan.optimize(), PhysicalOptions::default());
        let error = Pipeline::new(&physical, 0).unwrap_err();
        assert_eq!(error.code(), "KY-E601");
        let message = error.to_string();
        assert!(
            message.contains(method) && message.contains(task),
            "{message}"
        );
    }
    let physical = PhysicalPlan::new(base.optimize(), PhysicalOptions::default());
    let error = Pipeline::new(&physical, 1).unwrap_err();
    assert_eq!(error.code(), "KY-E215");
}

#[test]
fn a_missing_store_is_an_error_from_run() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let physical = physical(&preprocess(&store), OptimizeOptions::ALL, 2);
    let pipeline = Pipeline::new(&physical, 0).unwrap();
    std::fs::remove_dir_all(&store).unwrap();
    let error = pipeline.run(threads(2)).unwrap_err();
    assert!(error.code().starts_with("KY-E1"), "{error}");
}

/// Counts the storage requests, as `crates/kyto-store/tests/tile_reader.rs` does.
struct CountingStore<S> {
    inner: S,
    reads: AtomicUsize,
}

impl<S: ReadableStorageTraits> ReadableStorageTraits for CountingStore<S> {
    // `get` and `get_partial` default to `get_partial_many`, so every read passes here.
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let ranges: Vec<ByteRange> = byte_ranges.collect();
        self.reads.fetch_add(ranges.len(), Ordering::SeqCst);
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

#[test]
fn a_slow_consumer_stops_the_reading() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let physical = physical(&preprocess(&store), OptimizeOptions::ALL, 1);
    let pipeline = Pipeline::new(&physical, 0).unwrap();
    let storage = Arc::new(CountingStore {
        inner: FilesystemStore::new(&store).unwrap(),
        reads: AtomicUsize::new(0),
    });
    let mut stream = pipeline.run_on(Arc::clone(&storage), threads(4)).unwrap();
    let opened = storage.reads.load(Ordering::SeqCst);
    let first = stream.next().unwrap().unwrap();
    assert_eq!(first.cell_block, 0);
    std::thread::sleep(Duration::from_millis(300));
    let while_waiting = storage.reads.load(Ordering::SeqCst) - opened;
    let blocks = 1 + stream.count();
    let total = storage.reads.load(Ordering::SeqCst) - opened;
    assert_eq!(blocks, 16);
    // One block in flight: the producer reads block 1, then waits to hand it over;
    // it reads block 2 only after the consumer takes block 1.
    let per_block = total / blocks;
    assert!(
        while_waiting <= 2 * per_block + per_block / 2,
        "read {while_waiting} of {total} requests while the consumer held block 0"
    );
}

#[test]
fn dropping_the_stream_stops_the_producer() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let physical = physical(&preprocess(&store), OptimizeOptions::ALL, 1);
    let pipeline = Pipeline::new(&physical, 0).unwrap();
    let storage = Arc::new(CountingStore {
        inner: FilesystemStore::new(&store).unwrap(),
        reads: AtomicUsize::new(0),
    });
    let mut stream = pipeline.run_on(Arc::clone(&storage), threads(2)).unwrap();
    stream.next().unwrap().unwrap();
    // Drop joins the producer: it must return, having read only a few blocks.
    drop(stream);
    let after_drop = storage.reads.load(Ordering::SeqCst);
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(storage.reads.load(Ordering::SeqCst), after_drop);
}

#[test]
fn blocks_in_flight_is_capped_by_the_cell_blocks() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let physical = physical(&preprocess(&store), OptimizeOptions::ALL, u64::MAX);
    let pipeline = Pipeline::new(&physical, 0).unwrap();
    assert_eq!(pipeline.blocks_in_flight().get(), 16);
    let (_, _, rows) = rows_of(&collect(&physical, threads(2)), "log1p");
    assert!(rows == expected_rows());
}
