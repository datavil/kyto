//! Task 5.20 Done when: a sink run with `threads = n` does its writes on the run's pool
//! of n threads, never on rayon's global pool (R7), and writes the same bytes at any n.
//!
//! The check: every thread of rayon's global pool is held by a job that waits for the
//! end of the test, so any rayon work that reaches the global pool never runs and its
//! caller waits forever. A sink that writes on the global pool would therefore hang; one
//! that writes on the run's pool finishes. Each sink runs on a thread of its own and must
//! finish within [`DEADLINE`]. Rayon work from outside any pool goes to the global
//! pool, so a finished sink did all its rayon work (the layers in parallel, the shards'
//! inner chunks, the `.h5ad` export's reads) on the run's pool of exactly the threads
//! asked for. A control job spawned on the held global pool must not run. The global
//! pool is process-wide, so this check stays a binary of its own (AGENTS.md).
//!
//! The plan: `log1p` of the counts as a second layer, a computed cell column and a PCA
//! embedding (read by the sink after its stream), sunk to a Zarr store and to an `.h5ad`
//! file at 1 and 4 threads. The stores' files are byte-identical across thread counts
//! (except `provenance.json`'s target path and timestamps). The `.h5ad` files are not
//! compared as bytes (HDF5 stamps its objects with the time they were written): each is
//! converted back to a store once the global pool is released, and those stores' files
//! are compared.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::Duration;

use kyto_core::{BinaryOp, CellExpr, Expr, GeneIds, Tiling};
use kyto_exec::{ExecuteOptions, Pipeline};
use kyto_plan::{OptimizeOptions, Output, PhysicalOptions, Plan, Sink, physical_plan};
use kyto_store::{
    CellBlock, ConvertOptions, LayerSpec, LayerWriteOptions, StoreWriter, TableAxis, ValueDtype,
    convert, open_atlas_schema, read_ids,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};

const N_CELLS: u64 = 3_000;
const N_GENES: usize = 300;
const BLOCK_CELLS: u64 = 256;
const GLOBAL_THREADS: usize = 2;
/// How long one sink may take before it counts as stuck on the global pool.
const DEADLINE: Duration = Duration::from_secs(120);

/// Cells `c` hold gene `g` with value `1 + (c + g) % 7` where `(c * 3 + g) % 5 == 0`,
/// on tiles of 1024 cells x 128 genes (several gene tiles and cell blocks).
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
    let tiling = Tiling {
        cell_block_size: 1024,
        gene_block_size: 128,
    };
    let mut writer = StoreWriter::create(target, &genes, &specs, tiling, false).unwrap();
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

/// `with_cells` adds `total` (G0 + G5), then `log1p` of the counts as a second layer and
/// its PCA (`X_pca`, 4 components).
fn plan(store: &Path) -> Arc<Plan> {
    let schema = open_atlas_schema(store).unwrap();
    let gene_ids = GeneIds::new(read_ids(store, TableAxis::Genes, schema.n_genes).unwrap());
    let scan = Plan::scan(store.display().to_string(), schema, gene_ids);
    let gene = |id: &str| CellExpr::expression(Some("counts".to_owned()), id);
    let total = Expr::binary(gene("G0"), BinaryOp::Add, gene("G5")).unwrap();
    scan.with_cells([("total".to_owned(), total)])
        .unwrap()
        .log1p(Some("counts"), "log1p")
        .unwrap()
        .with_default_layer("log1p")
        .unwrap()
        .pca(Some("log1p"), "X_pca", 4, 0, None)
        .unwrap()
}

/// Sink `plan` to `target` on `threads` threads, on a thread of its own; fails the test
/// when it takes longer than [`DEADLINE`] (stuck on the held global pool).
fn sink_within_deadline(plan: &Arc<Plan>, target: Sink, threads: usize) -> u64 {
    let physical = physical_plan(
        &[Output::all(&plan.sink(target.clone()).unwrap())],
        OptimizeOptions::ALL,
        PhysicalOptions {
            max_block_cells: NonZeroU64::new(BLOCK_CELLS),
            blocks_in_flight: NonZeroU64::new(threads as u64).unwrap(),
            ..PhysicalOptions::default()
        },
    );
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let options = ExecuteOptions::new(NonZeroUsize::new(threads).unwrap());
        let written = Pipeline::new(&physical, 0)
            .and_then(|pipeline| pipeline.sink(options, false))
            .map_err(|error| error.to_string());
        let _ = sender.send(written);
    });
    match receiver.recv_timeout(DEADLINE) {
        Ok(written) => written.unwrap(),
        Err(_) => panic!(
            "the sink to {} on {threads} threads did not finish in {DEADLINE:?}: \
             it waits for rayon's global pool, outside the run's pool (R7)",
            target.path()
        ),
    }
}

/// Every file under `root` (relative path to bytes), `provenance.json` left out.
fn files(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut found = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else if path.file_name().unwrap() != "provenance.json" {
                let relative = path.strip_prefix(root).unwrap().to_path_buf();
                found.insert(relative, std::fs::read(&path).unwrap());
            }
        }
    }
    found
}

/// Holds every thread of rayon's global pool until dropped.
struct HeldGlobalPool {
    released: Arc<(Mutex<bool>, Condvar)>,
}

impl HeldGlobalPool {
    fn hold() -> Self {
        let released = Arc::new((Mutex::new(false), Condvar::new()));
        let (started_sender, started) = mpsc::channel();
        let held = Arc::clone(&released);
        rayon::spawn_broadcast(move |_| {
            started_sender.send(()).unwrap();
            let (lock, condvar) = &*held;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = condvar.wait(released).unwrap();
            }
        });
        for _ in 0..GLOBAL_THREADS {
            started.recv_timeout(DEADLINE).unwrap();
        }
        Self { released }
    }
}

impl Drop for HeldGlobalPool {
    fn drop(&mut self) {
        let (lock, condvar) = &*self.released;
        *lock.lock().unwrap() = true;
        condvar.notify_all();
    }
}

#[test]
fn a_sink_writes_on_the_runs_pool_only() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    // Sized before anything uses it (the first use builds it at one thread per core).
    rayon::ThreadPoolBuilder::new()
        .num_threads(GLOBAL_THREADS)
        .build_global()
        .unwrap();
    // Written before the global pool is held: this writer runs outside any pool.
    write_store(&store);
    let plan = plan(&store);

    let held = HeldGlobalPool::hold();
    // The control: work spawned on the held global pool does not run.
    let ran = Arc::new(AtomicBool::new(false));
    let control = Arc::clone(&ran);
    rayon::spawn(move || control.store(true, Ordering::SeqCst));
    std::thread::sleep(Duration::from_millis(200));
    assert!(
        !ran.load(Ordering::SeqCst),
        "the global pool is not held, so the check proves nothing"
    );

    let mut stores = Vec::new();
    let mut h5ads = Vec::new();
    for threads in [1, 4] {
        let zarr = work.path().join(format!("out-{threads}.kyto"));
        let h5ad = work.path().join(format!("out-{threads}.h5ad"));
        let written = sink_within_deadline(
            &plan,
            Sink::Zarr {
                path: zarr.display().to_string(),
            },
            threads,
        );
        assert_eq!(written, N_CELLS);
        let written = sink_within_deadline(
            &plan,
            Sink::H5ad {
                path: h5ad.display().to_string(),
            },
            threads,
        );
        assert_eq!(written, N_CELLS);
        stores.push(files(&zarr));
        h5ads.push(h5ad);
    }
    drop(held);
    assert!(
        stores[0]
            .keys()
            .any(|path| path.starts_with("layers/log1p"))
    );
    assert!(
        stores[0]
            .keys()
            .any(|path| path.starts_with("embeddings/X_pca"))
    );
    assert_eq!(
        stores[0], stores[1],
        "the store differs between 1 and 4 threads"
    );
    // Converted after the global pool is released (the importer runs outside any pool).
    let converted: Vec<_> = h5ads
        .iter()
        .map(|h5ad| {
            let back = h5ad.with_extension("back.kyto");
            let options = ConvertOptions {
                layer: "log1p".to_owned(),
                ..ConvertOptions::default()
            };
            convert(h5ad, &back, &options).unwrap();
            files(&back)
        })
        .collect();
    assert!(
        converted[0]
            .keys()
            .any(|path| path.starts_with("layers/log1p"))
    );
    assert_eq!(
        converted[0], converted[1],
        "the .h5ad's contents differ between 1 and 4 threads"
    );
}
