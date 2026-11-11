//! Task 5.5 Done when: peak RSS stays under the limit on a dataset 4x larger than the
//! limit; impossible plans raise `KY-E401` before running.
//!
//! The dataset: [`N_CELLS`] x [`N_GENES`] seeded `uint32` counts (about 20% of genes
//! per cell) at the writer's default tiling (4096 x 1024), at least 4 x [`LIMIT`] bytes
//! as CSR in memory (`nnz x 8 + (n_cells + 1) x 8`). The plan is `normalize_total(1e4)`
//! then `log1p`, reading `log1p`, on 4 threads with the default 2 blocks in flight.
//! [`kyto_exec::fit_memory_limit`] must shrink its blocks so the plan's estimate, with
//! the allocator's overhead, fits beside what the process already holds. The
//! limited run happens in a child process (this binary, running only
//! [`child_runs_the_plan_within_the_limit`]), so writing the store and the unlimited
//! reference run do not count: on macOS the parent runs it under `/usr/bin/time -l`
//! (its maximum resident set size), on Linux the child reads `VmHWM` from
//! `/proc/self/status`; Windows has neither and skips the check. The child's whole
//! process counts (the test harness, threads, allocator), so the RSS bound is stricter
//! than the estimate's. The limited stream must give the unlimited stream's rows, bit
//! for bit. The child also reports its peak heap bytes (`peak_alloc`) next to the
//! estimate (task 4.6 Open (1), 5.2 Open (2)); `cargo test -- --nocapture` prints them.
//!
//! `KY-E401`: on a store whose matrix arrays are deleted (any read fails), a limit below
//! the plan's smallest peak gives `KY-E401` with that peak, and that peak is exactly
//! the smallest limit that fits (one cell a block, one block in flight).

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;

use kyto_core::{GeneIds, Tiling};
use kyto_exec::{
    ALLOCATOR_OVERHEAD_PERCENT, CellValues, ExecuteOptions, Pipeline, fit_memory_limit,
    fit_memory_limit_with, process_resident_bytes,
};
use kyto_plan::{
    Blocks, Member, MemoryLimit, OptimizeOptions, Output, PhysicalOptions, PhysicalPlan, Plan,
    optimize,
};
use kyto_store::{
    CellBlock, LayerSpec, LayerWriteOptions, StoreWriter, TableAxis, ValueDtype, open_atlas_schema,
    read_ids,
};
use peak_alloc::PeakAlloc;
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};

#[global_allocator]
static ALLOCATOR: PeakAlloc = PeakAlloc;

/// The memory limit of the Done-when check.
const LIMIT: u64 = 64 << 20;
const N_CELLS: usize = 84_000;
const N_GENES: usize = 2_000;
/// Cells written per `StoreWriter::push`.
const WRITE_CELLS: usize = 4_096;
const THREADS: usize = 4;
/// Where the parent tells the child its store is.
const CHILD_STORE: &str = "KYTO_TEST_MEMORY_LIMIT_STORE";
/// The line the child prints its results on.
const REPORT: &str = "kyto-memory-limit:";

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

/// Write the counts as a store at `target` (genes 1 to 9 apart: mean density 20%).
/// Returns the dataset's bytes as CSR in memory.
fn write_store(target: &Path, n_cells: usize) -> u64 {
    let gene_ids: Vec<String> = (0..N_GENES).map(|gene| format!("G{gene}")).collect();
    let cell_ids: Vec<String> = (0..n_cells).map(|cell| format!("c{cell}")).collect();
    let genes = DataFrame::new(
        N_GENES,
        vec![Series::new("gene_id".into(), gene_ids).into_column()],
    )
    .unwrap();
    let cells = DataFrame::new(
        n_cells,
        vec![Series::new("cell_id".into(), cell_ids).into_column()],
    )
    .unwrap();
    let specs = [LayerSpec {
        name: "counts".to_owned(),
        dtype: ValueDtype::U32,
        options: LayerWriteOptions::default(),
    }];
    let mut writer = StoreWriter::create(target, &genes, &specs, Tiling::default(), false).unwrap();
    let mut random = Lcg(11);
    let mut nnz = 0_u64;
    for first in (0..n_cells).step_by(WRITE_CELLS) {
        let mut indptr = vec![0_u64];
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        for _ in first..(first + WRITE_CELLS).min(n_cells) {
            let mut gene = random.next() % 9;
            while gene < N_GENES as u64 {
                let draw = random.next();
                indices.push(gene as u32);
                data.push(1 + (draw % 50) as u32);
                gene += 1 + draw / 50 % 9;
            }
            indptr.push(indices.len() as u64);
        }
        nnz += indices.len() as u64;
        let block = CellBlock::from_u32(indptr, indices, data, N_GENES as u64).unwrap();
        writer.push(&[block]).unwrap();
    }
    writer.finish(&cells, None).unwrap();
    nnz * 8 + (n_cells as u64 + 1) * 8
}

fn scan(store: &Path) -> Arc<Plan> {
    let schema = open_atlas_schema(store).unwrap();
    let gene_ids = GeneIds::new(read_ids(store, TableAxis::Genes, schema.n_genes).unwrap());
    Plan::scan(store.display().to_string(), schema, gene_ids)
}

/// `normalize_total(1e4)` then `log1p`, reading `log1p` (all optimizer rules).
fn output(store: &Path) -> Output {
    let plan = scan(store)
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "log1p")
        .unwrap();
    Output::members(&plan, [Member::Layer("log1p".to_owned())]).unwrap()
}

fn limit(bytes: u64) -> MemoryLimit {
    MemoryLimit::new(NonZeroU64::new(bytes).unwrap())
}

/// The plan fitted to `bytes` as if the process held `resident` bytes already.
fn fitted(store: &Path, bytes: u64, resident: u64) -> Result<PhysicalPlan, kyto_error::KytoError> {
    let optimized = optimize(&[output(store)], OptimizeOptions::default());
    fit_memory_limit_with(
        optimized,
        PhysicalOptions::default(),
        limit(bytes),
        resident,
    )
}

/// What a run streamed: blocks, rows and a hash of every row's genes and value bits.
#[derive(Debug, PartialEq, Eq)]
struct Streamed {
    blocks: u64,
    rows: u64,
    hash: u64,
}

/// Run `plan`, consuming each block (hashing its rows) and recycling it.
fn run(plan: &PhysicalPlan) -> Streamed {
    let pipeline = Pipeline::new(plan, 0).unwrap();
    let options = ExecuteOptions::new(NonZeroUsize::new(THREADS).unwrap());
    let mut stream = pipeline.run(options).unwrap();
    let mut streamed = Streamed {
        blocks: 0,
        rows: 0,
        hash: 0xcbf2_9ce4_8422_2325,
    };
    let mut mix = |value: u64| {
        streamed.hash = (streamed.hash ^ value).wrapping_mul(0x0100_0000_01b3);
    };
    let (mut blocks, mut rows) = (0, 0);
    while let Some(block) = stream.next() {
        let block = block.unwrap();
        let layer = &block.layers["log1p"];
        let CellValues::F32(values) = layer.data() else {
            panic!("log1p is not float32")
        };
        for (&cell, row) in block.cells.iter().zip(layer.indptr().windows(2)) {
            mix(cell);
            let range = row[0] as usize..row[1] as usize;
            for (&gene, value) in layer.indices()[range.clone()].iter().zip(&values[range]) {
                mix(u64::from(gene) << 32 | u64::from(value.to_bits()));
            }
        }
        blocks += 1;
        rows += block.cells.len() as u64;
        stream.recycle(block);
    }
    streamed.blocks = blocks;
    streamed.rows = rows;
    streamed
}

/// The scan's cells per block and the blocks in flight of `plan`.
fn blocks_of(plan: &PhysicalPlan) -> (u64, u64) {
    let Blocks::Cells(cells) = plan.executions()[0].blocks() else {
        panic!("the scan streams")
    };
    (cells, plan.options().blocks_in_flight.get())
}

/// The child: run the fitted plan on the parent's store and report.
#[test]
#[ignore = "run by peak_rss_stays_under_the_limit in a child process"]
fn child_runs_the_plan_within_the_limit() {
    let Ok(store) = std::env::var(CHILD_STORE) else {
        return;
    };
    let store = Path::new(&store);
    let optimized = optimize(&[output(store)], OptimizeOptions::default());
    let plan = fit_memory_limit(optimized, PhysicalOptions::default(), limit(LIMIT)).unwrap();
    let estimate = plan.estimate().peak_memory();
    let baseline_rss = process_resident_bytes().unwrap_or(0);
    ALLOCATOR.reset_peak_usage();
    let before = ALLOCATOR.current_usage();
    let streamed = run(&plan);
    let peak_heap = ALLOCATOR.peak_usage() - before;
    let (cells, in_flight) = blocks_of(&plan);
    println!(
        "{REPORT} blocks={} rows={} hash={} block_cells={cells} in_flight={in_flight} estimate={estimate} peak_heap={peak_heap} baseline_rss={baseline_rss} vm_hwm={}",
        streamed.blocks,
        streamed.rows,
        streamed.hash,
        linux_peak_rss().unwrap_or(0),
    );
}

/// `VmHWM` (peak resident set) of this process in bytes, on Linux.
fn linux_peak_rss() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

/// The `key=value` of the child's report line.
fn field(report: &str, key: &str) -> u64 {
    report
        .split_whitespace()
        .find_map(|item| item.strip_prefix(&format!("{key}=")))
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| panic!("no {key} in {report}"))
}

#[test]
fn impossible_limits_raise_ky_e401_before_reading() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("small.kyto");
    write_store(&store, 5_000);
    // Without its matrix arrays, any read of the layer fails.
    for array in ["indices", "data", "indptr"] {
        std::fs::remove_dir_all(store.join("layers/counts").join(array)).unwrap();
    }
    // As if the process held 20 MB: the plan gets half of what is left
    // (`ALLOCATOR_OVERHEAD_PERCENT`).
    let resident = 20_000_000;
    let error = fitted(&store, resident + 100_000, resident).unwrap_err();
    assert_eq!(error.code(), "KY-E401");
    let kyto_error::KytoError::MemoryLimit {
        required_bytes,
        limit_bytes,
    } = error
    else {
        panic!("{error}")
    };
    assert_eq!(limit_bytes, resident + 100_000);
    assert!(required_bytes > limit_bytes);
    let message = error.to_string();
    assert!(
        message.contains(&format!(
            "the plan needs at least {required_bytes} bytes but memory_limit is {limit_bytes} bytes"
        )),
        "{message}"
    );
    // The smallest peak is a limit that fits, with the smallest blocks; one byte less
    // does not.
    let smallest = fitted(&store, required_bytes, resident).unwrap();
    let plan_peak = smallest.estimate().peak_memory();
    assert_eq!(
        required_bytes,
        resident + (plan_peak * (100 + ALLOCATOR_OVERHEAD_PERCENT)).div_ceil(100)
    );
    assert_eq!(blocks_of(&smallest), (1, 1));
    let error = fitted(&store, required_bytes - 1, resident).unwrap_err();
    assert_eq!(error.code(), "KY-E401");
    assert!(
        error
            .to_string()
            .contains(&format!("at least {required_bytes} bytes"))
    );
    // A limit below what the process holds fits nothing.
    let error = fitted(&store, resident, resident).unwrap_err();
    assert!(
        error
            .to_string()
            .contains(&format!("at least {required_bytes} bytes"))
    );
    // Running would have read the store, and failed.
    let pipeline = Pipeline::new(&smallest, 0).unwrap();
    let options = ExecuteOptions::new(NonZeroUsize::new(1).unwrap());
    assert!(pipeline.run(options).is_err());
}

#[test]
fn peak_rss_stays_under_the_limit() {
    if cfg!(not(any(target_os = "linux", target_os = "macos"))) {
        eprintln!("peak RSS is read on Linux and macOS only; skipped");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("large.kyto");
    let dataset = write_store(&store, N_CELLS);
    assert!(
        dataset >= 4 * LIMIT,
        "dataset {dataset} B is not 4x {LIMIT} B"
    );

    // Unlimited, the plan does not fit; the limit shrinks its blocks.
    let unlimited = PhysicalPlan::new(
        optimize(&[output(&store)], OptimizeOptions::default()),
        PhysicalOptions::default(),
    );
    assert!(unlimited.estimate().peak_memory() > LIMIT);
    let expected = run(&unlimited);
    assert_eq!(expected.rows, N_CELLS as u64);

    let test = std::env::current_exe().unwrap();
    let arguments = [
        "child_runs_the_plan_within_the_limit",
        "--exact",
        "--ignored",
        "--nocapture",
        "--test-threads=1",
    ];
    let mut command = if cfg!(target_os = "macos") {
        let mut command = std::process::Command::new("/usr/bin/time");
        command.arg("-l").arg(&test);
        command
    } else {
        std::process::Command::new(&test)
    };
    let child = command
        .args(arguments)
        .env(CHILD_STORE, &store)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&child.stdout);
    let stderr = String::from_utf8_lossy(&child.stderr);
    assert!(child.status.success(), "child failed:\n{stdout}\n{stderr}");
    let report = stdout
        .lines()
        .find_map(|line| line.split_once(REPORT).map(|(_, rest)| rest.to_owned()))
        .unwrap_or_else(|| panic!("no report from the child:\n{stdout}\n{stderr}"));
    let peak_rss = if cfg!(target_os = "macos") {
        // `/usr/bin/time -l`: "  123456  maximum resident set size" (bytes).
        stderr
            .lines()
            .find(|line| line.contains("maximum resident set size"))
            .and_then(|line| line.split_whitespace().next()?.parse().ok())
            .unwrap_or_else(|| panic!("no maximum resident set size:\n{stderr}"))
    } else {
        field(&report, "vm_hwm")
    };
    println!(
        "dataset {dataset} B, limit {LIMIT} B, peak RSS {peak_rss} B; child:{report}; unlimited estimate {} B",
        unlimited.estimate().peak_memory()
    );
    assert_eq!(field(&report, "rows"), expected.rows);
    assert_eq!(field(&report, "hash"), expected.hash, "limited rows differ");
    assert!(field(&report, "blocks") > expected.blocks);
    assert!(field(&report, "block_cells") < Tiling::default().cell_block_size);
    assert!(peak_rss > 0);
    assert!(
        peak_rss < LIMIT,
        "peak RSS {peak_rss} B is not under the limit {LIMIT} B: {report}"
    );
}
