//! Task 5.2 Done when: CPU use scales with threads on a fixture pipeline (the bench
//! proper is task 10.3).
//!
//! On purpose only, in release (timings): `uv run cargo test --release -p kyto-exec
//! --test it -- thread_scaling:: --ignored --nocapture`. Writes the pbmc3k fixture as
//! `.h5ad` (`tests/fixtures`, needs anndata; skips with exit code 77 without it),
//! converts it at 128 x 4096 tiles (22 cell blocks of 8 tiles), and runs `scan ->
//! normalize_total(1e4) -> log1p` (one fused pass) to the end of the stream on 1, 2, 4
//! and 8 threads, with the default 2 blocks in flight and with as many blocks in flight
//! as threads. Same data, same work and the same output bits on every thread count
//! (checked); the time is the median of 7 runs, the store's files in the OS cache
//! after a warm-up run. Asserts a speedup of at least 2 at 4 threads with 4 blocks in
//! flight.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use kyto_core::{GeneIds, Tiling};
use kyto_exec::{CellValues, ExecuteOptions, Pipeline};
use kyto_plan::{Member, OptimizeOptions, Output, PhysicalOptions, Plan, physical_plan};
use kyto_store::{ConvertOptions, TableAxis, convert, open_atlas_schema, read_ids};

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

const RUNS: usize = 7;

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

/// Run the pipeline once; the time and a digest of every output value's bits.
fn run_once(pipeline: &Pipeline, threads: usize) -> (Duration, u64) {
    let options = ExecuteOptions::new(NonZeroUsize::new(threads).unwrap());
    let start = Instant::now();
    let mut stream = pipeline.run(options).unwrap();
    let mut digest = 0_u64;
    while let Some(block) = stream.next() {
        let block = block.unwrap();
        let CellValues::F32(values) = block.layers["log1p"].data() else {
            panic!("log1p is float32")
        };
        for value in values {
            digest = digest.rotate_left(5) ^ u64::from(value.to_bits());
        }
        stream.recycle(block);
    }
    (start.elapsed(), digest)
}

#[test]
#[ignore = "timings: run in release on purpose (see the module doc)"]
fn cpu_use_scales_with_threads_on_pbmc3k() {
    let work = tempfile::tempdir().unwrap();
    let h5ad = work.path().join("pbmc3k.h5ad");
    if !write_h5ad("pbmc3k", &h5ad) {
        eprintln!("skipped: anndata is not installed on this platform");
        return;
    }
    let store = work.path().join("pbmc3k.kyto");
    let options = ConvertOptions {
        tiling: Tiling {
            cell_block_size: 128,
            gene_block_size: 4096,
        },
        ..ConvertOptions::default()
    };
    convert(&h5ad, &store, &options).unwrap();
    let schema = open_atlas_schema(&store).unwrap();
    let layer = schema.layers[0].name.clone();
    let gene_ids = GeneIds::new(read_ids(&store, TableAxis::Genes, schema.n_genes).unwrap());
    let plan = Plan::scan(store.display().to_string(), schema, gene_ids)
        .normalize_total(Some(1e4), Some(&layer), "norm")
        .unwrap()
        .log1p(Some("norm"), "log1p")
        .unwrap();
    let output = Output::members(&plan, [Member::Layer("log1p".to_owned())]).unwrap();
    let mut report = Vec::new();
    let mut speedup_at_4 = 0.0;
    let mut digests = Vec::new();
    for in_flight_is_threads in [false, true] {
        let mut single = Duration::ZERO;
        for threads in [1, 2, 4, 8] {
            let blocks_in_flight = if in_flight_is_threads {
                threads as u64
            } else {
                2
            };
            let physical = physical_plan(
                std::slice::from_ref(&output),
                OptimizeOptions::ALL,
                PhysicalOptions {
                    blocks_in_flight: NonZeroU64::new(blocks_in_flight).unwrap(),
                    ..PhysicalOptions::default()
                },
            );
            let pipeline = Pipeline::new(&physical, 0).unwrap();
            run_once(&pipeline, threads); // warm-up: the store's files in the OS cache
            let mut times = Vec::new();
            for _ in 0..RUNS {
                let (time, digest) = run_once(&pipeline, threads);
                times.push(time);
                digests.push(digest);
            }
            times.sort();
            let median = times[RUNS / 2];
            if threads == 1 {
                single = median;
            }
            let speedup = single.as_secs_f64() / median.as_secs_f64();
            if in_flight_is_threads && threads == 4 {
                speedup_at_4 = speedup;
            }
            report.push(format!(
                "threads={threads} blocks_in_flight={blocks_in_flight}: median {:.1} ms, speedup {speedup:.2}x",
                median.as_secs_f64() * 1e3
            ));
        }
    }
    eprintln!("{}", report.join("\n"));
    // R11: the same bits on every thread count and window.
    assert!(digests.windows(2).all(|pair| pair[0] == pair[1]));
    assert!(
        speedup_at_4 >= 2.0,
        "4 threads are only {speedup_at_4:.2}x faster than 1:\n{}",
        report.join("\n")
    );
}
