//! Task 5.9: sinks in the executor.
//!
//! On a small store (64 genes, 2 048 cells in blocks of 64; a `counts` layer; a stored
//! cell column `n_genes` and gene column `n_cells` with lineage, an ordered `Enum` cell
//! column `group`): `sink_zarr` writes the stream's layers, the kept cells' stored
//! columns plus the computed ones and the genes present, bit-identical to the stream;
//! `sink_h5ad` round-trips through 2.6's importer to the same store contents; lineage
//! is carried for stored derived columns still valid and the stale ones are dropped
//! (3.8), and recorded for a column computed from one layer (5.11); an error or a cancel (`KY-E503`, 5.7), midway or after the last block while
//! the target is finished, leaves no target and no temporary; a target that exists
//! (without `overwrite`) or is the scanned store is refused before anything is read.
//! The scanpy parity of a sunk filter -> QC -> normalize/log1p plan is
//! `filter_qc_parity.rs`.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use kyto_core::{BinaryOp, CellExpr, Expr, GeneExpr, GeneIds, Lineage, Literal, Tiling};
use kyto_error::KytoError;
use kyto_exec::{ExecuteOptions, Pipeline, RunControl};
use kyto_plan::{
    OptimizeOptions, Output, PhysicalOptions, PhysicalPlan, Plan, Sink, physical_plan,
};
use kyto_store::{
    CellBlock, ColumnDtype, ConvertOptions, CsrMatrix, LayerSpec, LayerWriteOptions, StoreWriter,
    TableAxis, ValueDtype, convert, open_atlas_schema, read_ids, read_layer, read_table,
};
use polars::prelude::{DataFrame, DataType, FrozenCategories, IntoColumn, NamedFrom, Series};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::byte_range::ByteRangeIterator;
use zarrs::storage::{MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey};

const N_CELLS: u64 = 2_048;
const N_GENES: usize = 64;
const BLOCK_CELLS: u64 = 64;

fn lineage(node: &str) -> Lineage {
    Lineage {
        layer: "counts".to_owned(),
        node: node.to_owned(),
    }
}

/// Cells `c` hold gene `g` with value `1 + (c + g) % 7` where `(c * 3 + g) % 5 == 0`.
fn write_store(target: &Path) {
    let gene_ids: Vec<String> = (0..N_GENES).map(|gene| format!("G{gene}")).collect();
    let cell_ids: Vec<String> = (0..N_CELLS).map(|cell| format!("c{cell}")).collect();
    let n_cells_column: Vec<u32> = (0..N_GENES as u32).collect();
    let genes = DataFrame::new(
        N_GENES,
        vec![
            Series::new("gene_id".into(), gene_ids).into_column(),
            Series::new("n_cells".into(), n_cells_column).into_column(),
        ],
    )
    .unwrap();
    let groups: Vec<&str> = (0..N_CELLS)
        .map(|cell| if cell % 3 == 0 { "high" } else { "low" })
        .collect();
    let categories = FrozenCategories::new(["low", "high"]).unwrap();
    let group = Series::new("group".into(), groups)
        .cast(&DataType::from_frozen_categories(categories))
        .unwrap();
    let n_genes_column: Vec<u32> = (0..N_CELLS as u32).map(|cell| cell % 13).collect();
    let cells = DataFrame::new(
        N_CELLS as usize,
        vec![
            Series::new("cell_id".into(), cell_ids).into_column(),
            Series::new("n_genes".into(), n_genes_column).into_column(),
            group.into_column(),
        ],
    )
    .unwrap();
    let specs = [LayerSpec {
        name: "counts".to_owned(),
        dtype: ValueDtype::U32,
        options: LayerWriteOptions::default(),
    }];
    let mut writer = StoreWriter::create(target, &genes, &specs, Tiling::default(), false).unwrap();
    writer.mark_ordered(TableAxis::Cells, "group");
    writer.set_lineage(TableAxis::Cells, "n_genes", lineage("n3"));
    writer.set_lineage(TableAxis::Genes, "n_cells", lineage("n4"));
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

fn scan(store: &Path) -> Arc<Plan> {
    let schema = open_atlas_schema(store).unwrap();
    let gene_ids = GeneIds::new(read_ids(store, TableAxis::Genes, schema.n_genes).unwrap());
    Plan::scan(store.display().to_string(), schema, gene_ids)
}

/// `with_cells` adds `total` (G0 + G5), cells with a zero total are dropped (those
/// kept are the cells `c` with `c % 5 == 0`), and `log1p` of the counts is the default
/// layer.
fn cell_steps(plan: &Arc<Plan>) -> Arc<Plan> {
    let gene = |id: &str| CellExpr::expression(Some("counts".to_owned()), id);
    let total = Expr::binary(gene("G0"), BinaryOp::Add, gene("G5")).unwrap();
    let keep = Expr::binary(
        CellExpr::column("total"),
        BinaryOp::NotEq,
        Literal::Float(0.0),
    )
    .unwrap();
    plan.with_cells([("total".to_owned(), total)])
        .unwrap()
        .filter_cells(&keep)
        .unwrap()
        .log1p(Some("counts"), "log1p")
        .unwrap()
        .with_default_layer("log1p")
        .unwrap()
}

fn physical(plan: &Arc<Plan>) -> PhysicalPlan {
    let options = PhysicalOptions {
        max_block_cells: NonZeroU64::new(BLOCK_CELLS),
        blocks_in_flight: NonZeroU64::new(2).unwrap(),
        ..PhysicalOptions::default()
    };
    physical_plan(&[Output::all(plan)], OptimizeOptions::ALL, options)
}

fn options(threads: usize, control: &RunControl) -> ExecuteOptions {
    ExecuteOptions::new(NonZeroUsize::new(threads).unwrap()).with_control(control.clone())
}

fn sink(plan: &Arc<Plan>, target: Sink, overwrite: bool) -> Result<u64, KytoError> {
    Pipeline::new(&physical(&plan.sink(target).unwrap()), 0)?
        .sink(options(2, &RunControl::new()), overwrite)
}

/// What a store holds: cell and gene tables, the schema's columns and every layer.
#[derive(Debug, PartialEq)]
struct Contents {
    cells: DataFrame,
    genes: DataFrame,
    cell_columns: Vec<kyto_store::ColumnSchema>,
    gene_columns: Vec<kyto_store::ColumnSchema>,
    counts: CsrMatrix<u32>,
    log1p: CsrMatrix<f32>,
}

fn contents(store: &Path) -> Contents {
    let schema = open_atlas_schema(store).unwrap();
    let (n_cells, n_genes, tiling) = (schema.n_cells, schema.n_genes, schema.tiling);
    let counts = read_layer::<u32>(store, "counts", n_cells, n_genes, tiling).unwrap();
    let log1p = read_layer::<f32>(store, "log1p", n_cells, n_genes, tiling).unwrap();
    Contents {
        cells: read_table(store, TableAxis::Cells).unwrap(),
        genes: read_table(store, TableAxis::Genes).unwrap(),
        cell_columns: schema.cells.columns,
        gene_columns: schema.genes.columns,
        counts,
        log1p,
    }
}

/// What the stream gives for the same plan, as a store would hold it.
fn streamed(
    plan: &Arc<Plan>,
    threads: usize,
) -> (Vec<u64>, CsrMatrix<u32>, CsrMatrix<f32>, Vec<f64>) {
    let mut stream = Pipeline::new(&physical(plan), 0)
        .unwrap()
        .run(options(threads, &RunControl::new()))
        .unwrap();
    let mut cells = Vec::new();
    let mut totals = Vec::new();
    let mut counts = CsrMatrix {
        n_cells: 0,
        n_genes: N_GENES as u64,
        indptr: vec![0],
        indices: Vec::new(),
        data: Vec::new(),
    };
    let mut log1p = CsrMatrix {
        n_cells: 0,
        n_genes: N_GENES as u64,
        indptr: vec![0],
        indices: Vec::new(),
        data: Vec::<f32>::new(),
    };
    while let Some(block) = stream.next() {
        let block = block.unwrap();
        cells.extend_from_slice(&block.cells);
        let total = block.columns.column("total").unwrap();
        totals.extend(
            total
                .cast(&DataType::Float64)
                .unwrap()
                .f64()
                .unwrap()
                .iter()
                .map(Option::unwrap),
        );
        let append_u32 = |matrix: &mut CsrMatrix<u32>, layer: &CellBlock| {
            let offset = *matrix.indptr.last().unwrap();
            matrix
                .indptr
                .extend(layer.indptr()[1..].iter().map(|end| offset + end));
            matrix.indices.extend_from_slice(layer.indices());
            let kyto_exec::CellValues::U32(values) = layer.data() else {
                panic!("counts are uint32")
            };
            matrix.data.extend_from_slice(values);
            matrix.n_cells += layer.n_rows() as u64;
        };
        append_u32(&mut counts, &block.layers["counts"]);
        let layer = &block.layers["log1p"];
        let offset = *log1p.indptr.last().unwrap();
        log1p
            .indptr
            .extend(layer.indptr()[1..].iter().map(|end| offset + end));
        log1p.indices.extend_from_slice(layer.indices());
        let kyto_exec::CellValues::F32(values) = layer.data() else {
            panic!("log1p is float32")
        };
        log1p.data.extend_from_slice(values);
        log1p.n_cells += layer.n_rows() as u64;
        stream.recycle(block);
    }
    (cells, counts, log1p, totals)
}

#[test]
fn sink_zarr_writes_the_stream() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let plan = cell_steps(&scan(&store));
    let target = work.path().join("out.kyto");
    let written = sink(
        &plan,
        Sink::Zarr {
            path: target.display().to_string(),
        },
        false,
    )
    .unwrap();
    let (cells, counts, log1p, totals) = streamed(&plan, 1);
    assert!(cells.len() < N_CELLS as usize && !cells.is_empty());
    assert_eq!(written, cells.len() as u64);
    let out = contents(&target);
    assert_eq!(out.counts, counts);
    assert_eq!(out.log1p, log1p);
    // The kept cells' stored columns, in the store's order, then the computed one.
    let source = read_table(&store, TableAxis::Cells).unwrap();
    let positions: Vec<u32> = cells.iter().map(|cell| *cell as u32).collect();
    let kept = source
        .take(&polars::prelude::IdxCa::new("cells".into(), positions))
        .unwrap();
    for name in ["cell_id", "n_genes", "group"] {
        assert_eq!(
            out.cells.column(name).unwrap(),
            kept.column(name).unwrap(),
            "{name}"
        );
    }
    let total: Vec<f64> = out
        .cells
        .column("total")
        .unwrap()
        .cast(&DataType::Float64)
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .map(Option::unwrap)
        .collect();
    assert_eq!(total, totals);
    assert_eq!(
        out.genes,
        read_table(&store, TableAxis::Genes)
            .unwrap()
            .select(["gene_id"])
            .unwrap()
    );
    // Lineage: n_genes (cells) is still valid (no gene changed); n_cells (genes) is
    // stale after the cell filter and dropped. `group` stays ordered.
    let names: Vec<&str> = out.cell_columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["cell_id", "n_genes", "group", "total"]);
    assert_eq!(out.cell_columns[1].lineage, Some(lineage("n3")));
    assert!(matches!(
        out.cell_columns[2].dtype,
        ColumnDtype::Enum { ordered: true, .. }
    ));
    // `total` is computed from counts by node 1 and no later node changes the genes or
    // rewrites counts: it records that lineage (task 5.11).
    assert_eq!(out.cell_columns[3].lineage, Some(lineage("1")));
    let gene_names: Vec<&str> = out.gene_columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(gene_names, ["gene_id"]);
    // The same bytes at other thread counts (R11): the store's contents are equal.
    let again = work.path().join("again.kyto");
    Pipeline::new(
        &physical(
            &plan
                .sink(Sink::Zarr {
                    path: again.display().to_string(),
                })
                .unwrap(),
        ),
        0,
    )
    .unwrap()
    .sink(options(4, &RunControl::new()), false)
    .unwrap();
    assert_eq!(contents(&again), out);
    // No temporaries are left next to the targets.
    let mut entries: Vec<String> = std::fs::read_dir(work.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    entries.sort();
    assert_eq!(entries, ["again.kyto", "atlas.kyto", "out.kyto"]);
}

#[test]
fn a_gene_step_drops_stale_cell_columns_and_writes_the_genes_present() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let keep = Expr::binary(GeneExpr::column("n_cells"), BinaryOp::Lt, Literal::Int(10)).unwrap();
    let plan = scan(&store)
        .select_genes([
            "G7".to_owned(),
            "G2".to_owned(),
            "G40".to_owned(),
            "G3".to_owned(),
        ])
        .unwrap();
    let target = work.path().join("out.kyto");
    sink(
        &plan,
        Sink::Zarr {
            path: target.display().to_string(),
        },
        false,
    )
    .unwrap();
    let schema = open_atlas_schema(&target).unwrap();
    assert_eq!(schema.n_cells, N_CELLS);
    assert_eq!(
        read_ids(&target, TableAxis::Genes, schema.n_genes).unwrap(),
        ["G7", "G2", "G40", "G3"]
    );
    // n_genes (cells) was computed over every gene: stale, dropped. n_cells (genes) is
    // still valid (no cell changed) and keeps its lineage.
    let cell_names: Vec<&str> = schema
        .cells
        .columns
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    assert_eq!(cell_names, ["cell_id", "group"]);
    assert_eq!(schema.genes.columns[1].lineage, Some(lineage("n4")));
    let counts = read_layer::<u32>(&target, "counts", N_CELLS, 4, schema.tiling).unwrap();
    let source =
        read_layer::<u32>(&store, "counts", N_CELLS, N_GENES as u64, Tiling::default()).unwrap();
    for cell in 0..N_CELLS as usize {
        let row = |matrix: &CsrMatrix<u32>, gene: u32| {
            let range = matrix.indptr[cell] as usize..matrix.indptr[cell + 1] as usize;
            matrix.indices[range.clone()]
                .iter()
                .zip(&matrix.data[range])
                .find(|(index, _)| **index == gene)
                .map_or(0, |(_, value)| *value)
        };
        for (position, gene) in [7, 2, 40, 3].into_iter().enumerate() {
            assert_eq!(row(&counts, position as u32), row(&source, gene));
        }
    }
    // A gene filter on a stored column also changes the gene axis.
    let filtered = scan(&store).filter_genes(&keep).unwrap();
    let target = work.path().join("filtered.kyto");
    sink(
        &filtered,
        Sink::Zarr {
            path: target.display().to_string(),
        },
        false,
    )
    .unwrap();
    let schema = open_atlas_schema(&target).unwrap();
    assert_eq!(schema.n_genes, 10);
    assert!(schema.cells.columns.iter().all(|c| c.name != "n_genes"));
}

#[test]
fn sink_h5ad_round_trips_through_the_importer() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let plan = cell_steps(&scan(&store));
    let zarr = work.path().join("out.kyto");
    let h5ad = work.path().join("out.h5ad");
    sink(
        &plan,
        Sink::Zarr {
            path: zarr.display().to_string(),
        },
        false,
    )
    .unwrap();
    let written = sink(
        &plan,
        Sink::H5ad {
            path: h5ad.display().to_string(),
        },
        false,
    )
    .unwrap();
    assert_eq!(written, open_atlas_schema(&zarr).unwrap().n_cells);
    // The default layer (log1p) is X; convert names it back.
    let back = work.path().join("back.kyto");
    let options = ConvertOptions {
        layer: "log1p".to_owned(),
        ..ConvertOptions::default()
    };
    convert(&h5ad, &back, &options).unwrap();
    let (sunk, round_trip) = (contents(&zarr), contents(&back));
    assert_eq!(round_trip.counts, sunk.counts);
    assert_eq!(round_trip.log1p, sunk.log1p);
    assert_eq!(round_trip.cells, sunk.cells);
    assert_eq!(round_trip.genes, sunk.genes);
    // No temporary store is left next to the file.
    let mut entries: Vec<String> = std::fs::read_dir(work.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    entries.sort();
    assert_eq!(entries, ["atlas.kyto", "back.kyto", "out.h5ad", "out.kyto"]);
}

/// Cancels `control` at its `cancel_at`th read of a tile's `data` chunk.
struct CancellingStore {
    inner: FilesystemStore,
    control: RunControl,
    cancel_at: u64,
    data_reads: AtomicU64,
}

impl ReadableStorageTraits for CancellingStore {
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

/// The names in `directory`, sorted.
fn entries(directory: &Path) -> Vec<String> {
    let mut entries: Vec<String> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    entries.sort();
    entries
}

#[test]
fn a_cancelled_or_failing_sink_leaves_no_target() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let out = work.path().join("out");
    std::fs::create_dir(&out).unwrap();
    for target in [
        Sink::Zarr {
            path: out.join("cancelled.kyto").display().to_string(),
        },
        Sink::H5ad {
            path: out.join("cancelled.h5ad").display().to_string(),
        },
    ] {
        let control = RunControl::new();
        let storage = Arc::new(CancellingStore {
            inner: FilesystemStore::new(&store).unwrap(),
            control: control.clone(),
            cancel_at: 5,
            data_reads: AtomicU64::new(0),
        });
        let pipeline = Pipeline::new(
            &physical(&cell_steps(&scan(&store)).sink(target).unwrap()),
            0,
        )
        .unwrap();
        let error = pipeline
            .sink_on(storage, options(2, &control), false)
            .unwrap_err();
        assert!(matches!(error, KytoError::Cancelled { .. }), "{error}");
        assert!(control.progress().done < N_CELLS);
        assert!(entries(&out).is_empty(), "{:?}", entries(&out));
    }
    // An error at the stream's end (take_cells of a cell the filter removed).
    let plan = cell_steps(&scan(&store))
        .take_cells(["c1".to_owned(), "c0".to_owned()])
        .unwrap();
    for target in [
        Sink::Zarr {
            path: out.join("failed.kyto").display().to_string(),
        },
        Sink::H5ad {
            path: out.join("failed.h5ad").display().to_string(),
        },
    ] {
        let error = sink(&plan, target, false).unwrap_err();
        assert_eq!(error.code(), "KY-E210", "{error}");
        assert!(entries(&out).is_empty(), "{:?}", entries(&out));
    }
}

/// Whether `directory` holds an entry whose name starts with `prefix`, does not end
/// with `.lock`, and (with `member`) holds `member`.
fn appeared(directory: &Path, prefix: &str, member: Option<&str>) -> bool {
    entries(directory).iter().any(|name| {
        name.starts_with(prefix)
            && !name.ends_with(".lock")
            && member.is_none_or(|member| directory.join(name).join(member).exists())
    })
}

/// A cancel that comes after the last block, while the sink finishes its target (the
/// tables, the sync, the `.h5ad` export), still leaves no target: the writers check
/// the run's flag right before their rename (found by running Ctrl-C late in a sink).
#[test]
fn a_cancel_after_the_last_block_leaves_no_target() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let out = work.path().join("out");
    std::fs::create_dir(&out).unwrap();
    // When the sink is past its stream: `sink_zarr`'s temporary store has its root
    // `zarr.json` (written last, before the tree is synced and renamed); `sink_h5ad`'s
    // export has started its temporary file (its store is written by then).
    let cases: [(Sink, &str, &'static str, Option<&'static str>); 2] = [
        (
            Sink::Zarr {
                path: out.join("late.kyto").display().to_string(),
            },
            "late.kyto",
            ".late.kyto.kyto-tmp-",
            Some("zarr.json"),
        ),
        (
            Sink::H5ad {
                path: out.join("late.h5ad").display().to_string(),
            },
            "late.h5ad",
            ".kyto-part-",
            None,
        ),
    ];
    for (target, name, prefix, member) in cases {
        let pipeline = Pipeline::new(
            &physical(&cell_steps(&scan(&store)).sink(target).unwrap()),
            0,
        )
        .unwrap();
        // The watcher races the rename; it wins by a wide margin (a sync and more are
        // left), and an attempt it loses is a finished sink: try again.
        let mut cancelled_late = false;
        for _ in 0..20 {
            let control = RunControl::new();
            let finished = Arc::new(AtomicBool::new(false));
            let watcher = std::thread::spawn({
                let (control, finished, out) = (control.clone(), finished.clone(), out.clone());
                move || {
                    while !finished.load(Ordering::SeqCst) {
                        if appeared(&out, prefix, member) {
                            control.cancel();
                            break;
                        }
                        std::thread::yield_now();
                    }
                }
            });
            let result = pipeline.sink(options(2, &control), false);
            finished.store(true, Ordering::SeqCst);
            watcher.join().unwrap();
            match result {
                Err(error) => {
                    assert!(matches!(error, KytoError::Cancelled { .. }), "{error}");
                    let progress = control.progress();
                    assert_eq!(progress.done, progress.total, "every block was computed");
                    assert!(entries(&out).is_empty(), "{:?}", entries(&out));
                    cancelled_late = true;
                    break;
                }
                Ok(_) => {
                    // The cancel came after the rename: the sink had finished.
                    assert_eq!(entries(&out), [name]);
                    let written = out.join(name);
                    if written.is_dir() {
                        std::fs::remove_dir_all(written).unwrap();
                    } else {
                        std::fs::remove_file(written).unwrap();
                    }
                }
            }
        }
        assert!(
            cancelled_late,
            "{name}: no cancel after the stream stopped the sink"
        );
    }
}

#[test]
fn targets_are_checked_before_reading() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let plan = scan(&store);
    let zarr = work.path().join("out.kyto");
    let h5ad = work.path().join("out.h5ad");
    let zarr_sink = || Sink::Zarr {
        path: zarr.display().to_string(),
    };
    let h5ad_sink = || Sink::H5ad {
        path: h5ad.display().to_string(),
    };
    sink(&plan, zarr_sink(), false).unwrap();
    sink(&plan, h5ad_sink(), false).unwrap();
    // A target that exists, without overwrite: refused before reading (a cancelled
    // control would otherwise give KY-E503).
    let control = RunControl::new();
    control.cancel();
    for (target, code) in [(zarr_sink(), "KY-E118"), (h5ad_sink(), "KY-E123")] {
        let error = Pipeline::new(&physical(&plan.sink(target).unwrap()), 0)
            .unwrap()
            .sink(options(1, &control), false)
            .unwrap_err();
        assert_eq!(error.code(), code, "{error}");
        assert!(error.to_string().contains("overwrite"), "{error}");
    }
    // With overwrite, both are replaced by the new output.
    let filtered = cell_steps(&plan);
    sink(&filtered, zarr_sink(), true).unwrap();
    sink(&filtered, h5ad_sink(), true).unwrap();
    assert!(open_atlas_schema(&zarr).unwrap().n_cells < N_CELLS);
    // The scanned store itself, or a path inside it, never.
    for (path, code) in [
        (store.clone(), "KY-E118"),
        (store.join("inner.kyto"), "KY-E118"),
    ] {
        let target = Sink::Zarr {
            path: path.display().to_string(),
        };
        let error = sink(&plan, target, true).unwrap_err();
        assert_eq!(error.code(), code, "{error}");
    }
    let error = sink(
        &plan,
        Sink::H5ad {
            path: store.join("x.h5ad").display().to_string(),
        },
        true,
    )
    .unwrap_err();
    assert_eq!(error.code(), "KY-E123", "{error}");
    assert_eq!(open_atlas_schema(&store).unwrap().n_cells, N_CELLS);
    // Pipeline::sink on an output that is not a sink.
    let error = Pipeline::new(&physical(&plan), 0)
        .unwrap()
        .sink(options(1, &RunControl::new()), false)
        .unwrap_err();
    assert_eq!(error.code(), "KY-E215", "{error}");
}

/// A recomputed column (3.8, 5.11) needs the plan that wrote the store: this store's
/// lineage is written by hand and its `provenance.json` holds no plan, so it is
/// refused (`KY-E303`) before reading. Recomputing is `embeddings_graphs.rs`'.
#[test]
fn a_recomputed_column_without_a_recipe_is_refused_before_reading() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let recomputed = CellExpr::column("n_genes").recompute().unwrap();
    let plan = scan(&store)
        .select_genes(["G1".to_owned()])
        .unwrap()
        .with_cells([("n_genes".to_owned(), recomputed)])
        .unwrap();
    let target = Sink::Zarr {
        path: work.path().join("out.kyto").display().to_string(),
    };
    let error = Pipeline::new(&physical(&plan.sink(target).unwrap()), 0).unwrap_err();
    assert_eq!(error.code(), "KY-E303", "{error}");
    assert!(error.to_string().contains("records no plan"), "{error}");
    assert_eq!(entries(work.path()), ["atlas.kyto"]);
}
