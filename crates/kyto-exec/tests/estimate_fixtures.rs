//! Task 4.6 Done when: on fixtures, estimated peak memory is within 25% of measured,
//! measured with the executor (moved here from kyto-plan by task 5.5, as 4.6 Open (1)
//! asked).
//!
//! "Measured" is a whole run of a plan's stream: `Pipeline::run` (task 5.2) opening the
//! layer, reading every block and, for the second plan, mapping it
//! (`normalize_total(1e4)` then `log1p`, one fused pass), with a consumer that recycles
//! each block as a terminal would (so the stream's pool fills). A peak-tracking global
//! allocator (`peak_alloc`: a realloc allocates, copies, then frees, so a moving buffer
//! counts twice while it moves) measures the heap bytes held at most. `tests/fixtures`
//! writes each fixture (synthetic, pbmc3k) as `.h5ad`; it is converted at the writer's
//! tiling and at a tiling with several cell blocks, run with 1 and 2 blocks in flight,
//! whole blocks and blocks split to 256 cells (task 5.5), on 2 threads, and compared
//! with `PhysicalPlan::estimate` of the same physical plan: within 25% for whole
//! blocks; for split blocks at most 25% below (above is the decode term's upper bound,
//! 4.6 Open (3): every tile of a block decoding at once, 32 tiles on pbmc3k at 1024
//! genes a block, while 2 threads decode 2). Needs Python with anndata
//! (`uv run cargo test`); skips where it has no wheels (exit code 77). The only test in
//! this binary, so no other test allocates while it measures.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use kyto_core::{GeneIds, Tiling};
use kyto_exec::{ExecuteOptions, Pipeline};
use kyto_plan::{Member, Output, PhysicalOptions, PhysicalPlan, Plan, physical_plan};
use kyto_store::{ConvertOptions, TableAxis, convert, open_atlas_schema, read_ids};
use peak_alloc::PeakAlloc;

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

/// Heap bytes held at most while running `plan`'s output 0 and consuming its stream.
fn measured_peak(plan: &PhysicalPlan) -> u64 {
    let pipeline = Pipeline::new(plan, 0).unwrap();
    let options = ExecuteOptions::new(NonZeroUsize::new(2).unwrap());
    let before = ALLOCATOR.current_usage();
    ALLOCATOR.reset_peak_usage();
    let mut stream = pipeline.run(options).unwrap();
    let mut entries = 0;
    while let Some(block) = stream.next() {
        let block = block.unwrap();
        entries += block.layers.values().map(|layer| layer.nnz()).sum::<u64>();
        stream.recycle(block);
    }
    drop(stream);
    let peak = ALLOCATOR.peak_usage() - before;
    assert!(entries > 0);
    u64::try_from(peak).unwrap()
}

/// The physical plan of `plan` reading layer `layer`.
fn physical(plan: &Arc<Plan>, layer: &str, options: PhysicalOptions) -> PhysicalPlan {
    let output = Output::members(plan, [Member::Layer(layer.to_owned())]).unwrap();
    physical_plan(&[output], kyto_plan::OptimizeOptions::default(), options)
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
            let gene_ids =
                GeneIds::new(read_ids(&store, TableAxis::Genes, schema.n_genes).unwrap());
            let layer = schema.layers[0].name.clone();
            let scan = Plan::scan(store.display().to_string(), schema, gene_ids);
            let mapped = scan
                .normalize_total(Some(1e4), Some(&layer), "norm")
                .unwrap()
                .log1p(Some("norm"), "log1p")
                .unwrap();
            let plans = [("scan", &scan, layer.as_str()), ("maps", &mapped, "log1p")];
            for ((kind, plan, read), blocks_in_flight, block_cells) in plans
                .into_iter()
                .flat_map(|plan| [1, 2].map(|in_flight| (plan, in_flight)))
                .flat_map(|(plan, in_flight)| {
                    [None, Some(256)].map(|cells| (plan, in_flight, cells))
                })
            {
                let options = PhysicalOptions {
                    blocks_in_flight: NonZeroU64::new(blocks_in_flight).unwrap(),
                    max_block_cells: block_cells.and_then(NonZeroU64::new),
                    ..PhysicalOptions::default()
                };
                let plan = physical(plan, read, options);
                let estimate = plan.estimate();
                assert_eq!(estimate.passes(), 1, "{name}: one pass reads the layer");
                let measured = measured_peak(&plan);
                let estimated = estimate.peak_memory();
                let error = (estimated as f64 - measured as f64) / measured as f64;
                report.push(format!(
                    "{name} {kind} C={} G={} in_flight={blocks_in_flight} block_cells={}: estimated {estimated} B, measured {measured} B ({:+.1}%)",
                    tiling.cell_block_size,
                    tiling.gene_block_size,
                    block_cells.map_or("C".to_owned(), |cells| cells.to_string()),
                    error * 100.0
                ));
                // Whole blocks: the 4.6 Done-when. Split blocks (5.5) on many gene blocks
                // are dominated by the decode term, which assumes every tile of a block
                // decodes at once (4.6 Open (3)): too high, never far too low.
                let off = if block_cells.is_none() {
                    error.abs() > TOLERANCE
                } else {
                    error < -TOLERANCE
                };
                if off {
                    report.push("  ^ off by more than 25%".to_owned());
                }
            }
        }
    }
    // The table the plan's 4.6 and 5.5 entries quote (`cargo test -- --nocapture`).
    println!("{}", report.join("\n"));
    assert!(
        !report.iter().any(|line| line.contains("off by more")),
        "estimate off by more than 25%:\n{}",
        report.join("\n")
    );
}
