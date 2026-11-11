//! Task 4.6 Done when: on fixtures, estimated peak memory is within 25% of measured.
//!
//! There is no executor yet (5.x), so "measured" is what runs a plan's matrix work
//! today: a scan read whole by the tile reader (`kyto_store::LayerReader`), every block
//! streamed and dropped, as a terminal reading the plan's layer would. A peak-tracking
//! global allocator (`peak_alloc`: a realloc allocates, copies, then frees, so a moving
//! buffer counts twice while it moves) measures the heap bytes the reader holds at
//! most, opening the layer included. `tests/fixtures` writes each fixture (synthetic,
//! pbmc3k) as `.h5ad`; it is converted at the writer's tiling and at a tiling with
//! several cell blocks, read with 1 and 2 blocks in flight, and compared with
//! `PhysicalPlan::estimate` of `Plan::scan` on the store's schema. Needs Python with
//! anndata (`uv run cargo test`); skips where it has no wheels (exit code 77). The
//! only test in this binary, so no other test allocates while it measures.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use kyto_core::{GeneIds, Tiling};
use kyto_plan::{PhysicalOptions, Plan};
use kyto_store::{
    AxisSelection, ConvertOptions, LayerReader, TableAxis, TileGrid, convert, open_atlas_schema,
    read_ids,
};
use peak_alloc::PeakAlloc;
use zarrs::filesystem::FilesystemStore;

#[global_allocator]
static ALLOCATOR: PeakAlloc = PeakAlloc;

/// The Done-when tolerance.
const TOLERANCE: f64 = 0.25;

/// Writes `load_fixture(name)` as `.h5ad` (anndata), or exits 77 without anndata.
const WRITE_H5AD: &str = "\
import sys
try:
    from fixtures.interop import to_anndata
except ImportError:
    sys.exit(77)
from fixtures.registry import load_fixture
to_anndata(load_fixture(sys.argv[1])).write_h5ad(sys.argv[2])
";

/// Write fixture `name` to `target`; `false` when anndata is not installed.
fn write_h5ad(name: &str, target: &Path) -> bool {
    let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests");
    let status = Command::new(std::env::var("PYTHON").unwrap_or_else(|_| "python".to_owned()))
        .current_dir(tests)
        .args(["-c", WRITE_H5AD, name])
        .arg(target)
        .status()
        .expect("run python (use `uv run cargo test`, or set PYTHON)");
    if status.code() == Some(77) {
        return false;
    }
    assert!(status.success(), "writing fixture {name} failed: {status}");
    true
}

/// Heap bytes held at most while opening `layer` and streaming all of it.
fn measured_peak(store: &Path, layer: &str, grid: TileGrid, blocks_in_flight: u64) -> u64 {
    let storage = Arc::new(FilesystemStore::new(store).unwrap());
    let in_flight = NonZeroUsize::new(usize::try_from(blocks_in_flight).unwrap()).unwrap();
    let before = ALLOCATOR.current_usage();
    ALLOCATOR.reset_peak_usage();
    let reader = LayerReader::open(storage, "fixture", layer, grid).unwrap();
    let mut entries = 0;
    for selected in reader
        .cell_blocks(&AxisSelection::All, &AxisSelection::All, in_flight)
        .unwrap()
    {
        entries += selected.unwrap().block.nnz();
    }
    let peak = ALLOCATOR.peak_usage() - before;
    drop(reader);
    assert!(entries > 0);
    u64::try_from(peak).unwrap()
}

#[test]
fn estimated_peak_memory_is_within_25_percent_of_measured() {
    let work = tempfile::tempdir().unwrap();
    let mut report = Vec::new();
    for name in ["synthetic", "pbmc3k"] {
        let h5ad: PathBuf = work.path().join(format!("{name}.h5ad"));
        if !write_h5ad(name, &h5ad) {
            eprintln!("skipped: anndata is not installed on this platform");
            return;
        }
        // The writer's tiling (one cell block on both fixtures), then several blocks.
        for tiling in [
            Tiling::default(),
            Tiling {
                cell_block_size: 1024,
                gene_block_size: 4096,
            },
        ] {
            let store = work.path().join(format!(
                "{name}-{}x{}",
                tiling.cell_block_size, tiling.gene_block_size
            ));
            let options = ConvertOptions {
                tiling,
                ..ConvertOptions::default()
            };
            convert(&h5ad, &store, &options).unwrap();
            let schema = open_atlas_schema(&store).unwrap();
            let grid = TileGrid::new(schema.n_cells, schema.n_genes, schema.tiling).unwrap();
            let gene_ids =
                GeneIds::new(read_ids(&store, TableAxis::Genes, schema.n_genes).unwrap());
            let layer = schema.layers[0].name.clone();
            let plan = Plan::scan(store.display().to_string(), schema, gene_ids);
            for blocks_in_flight in [1, 2] {
                let options = PhysicalOptions {
                    blocks_in_flight: NonZeroU64::new(blocks_in_flight).unwrap(),
                    ..PhysicalOptions::default()
                };
                let estimate = plan.physical(options).estimate();
                assert_eq!(estimate.passes(), 1, "{name}: one pass reads the layer");
                let measured = measured_peak(&store, &layer, grid, blocks_in_flight);
                let estimated = estimate.peak_memory();
                let error = (estimated as f64 - measured as f64) / measured as f64;
                report.push(format!(
                    "{name} C={} G={} in_flight={blocks_in_flight}: estimated {estimated} B, measured {measured} B ({:+.1}%)",
                    tiling.cell_block_size,
                    tiling.gene_block_size,
                    error * 100.0
                ));
                assert!(
                    error.abs() <= TOLERANCE,
                    "estimate off by more than 25%:\n{}",
                    report.join("\n")
                );
            }
        }
    }
    // The table the plan's 4.6 entry quotes (`cargo test -- --nocapture`).
    println!("{}", report.join("\n"));
}
