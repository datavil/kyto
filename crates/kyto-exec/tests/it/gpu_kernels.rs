//! Task 9.2 Done when: each CubeCL kernel matches the CPU output to 1e-6 relative.
//!
//! Runs on every GPU this build and machine have: wgpu (Metal, Vulkan, DX12) with
//! `--features wgpu`, CUDA with `--features cuda`. A runtime that is not built in, or
//! finds no GPU, prints a `SKIPPED` line naming why and checks nothing (CI's machines
//! have no GPU; 9.7 adds a GPU runner). Run on purpose:
//! `uv run cargo test -p kyto-exec --features wgpu --test it -- gpu_kernels:: --nocapture`.
//!
//! Checked per GPU, against the 5.3 CPU kernels (`kyto_exec::Kernels`): row sums of
//! `f32` values over many orders of magnitude and of `u32` counts (exact: the same bits),
//! row nnz (the same counts, stored zeros, `-0.0` and NaN included), the division by a
//! row's divisor and `log1p` over a walk of the `f32` range, all within 1e-6 relative;
//! the same bits on a second run (no atomics, a fixed order per thread); a dispatch past
//! 65535 workgroups; and the pipeline: `normalize_total` + `log1p` placed on the GPU
//! (`on(device=...)`) against the CPU run on pbmc3k, both target sums, with one timing
//! of each for the record (not a benchmark: M6). Task 9.9: a store sunk with
//! `on(device="auto")` records the request, the devices found and each step's device
//! (the GPU's for `normalize_total` + `log1p`) in `provenance.json`, the same bytes at
//! 1 and 4 threads, `run` aside.
//!
//! Task 9.10 Done when: `with_cells(total_counts=...)` on a GPU equals the CPU column
//! (counts: the same bits; `float32` layers: 1e-6 relative), and `explain()` shows the
//! step's device. Checked per GPU: each QC metric (`total_counts`, `genes_detected`,
//! `total_counts_in`, `percent_counts_in`) of [`CellQc`] on random blocks, counts with
//! the CPU's bits, `float32` within 1e-6 (rows whose `float32` sum overflows, NaN,
//! infinities and subnormals with the CPU's values); and on pbmc3k, a `with_cells` of every metric and
//! a `filter_cells` on one placed on the GPU (`explain` names it) against the CPU run:
//! the same cells and the same bits on the counts, 1e-6 relative on a `float32` layer
//! (`log1p`, written by a CPU run first), the same bits run to run.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::NonZeroUsize;
use std::time::Instant;

use std::sync::Arc;

use kyto_core::{BinaryOp, CellExpr, Expr, GeneExpr, Literal, QcMetric, StrFunction};
use kyto_exec::{CellQc, ExecuteOptions, GeneSet, Kernels, Pipeline, QcValues};
use kyto_gpu::{Backend, Cuda, Device, DeviceKernels, RowValues, Wgpu, probe};
use kyto_plan::{
    DeviceRequest, Devices, Explain, Member, OptimizeOptions, Output, PhysicalOptions, Plan, Sink,
    physical_plan,
};
use kyto_store::{CellBlock, CellBlockParts, CellData, CellValues, ConvertOptions};
use polars::prelude::DataType;

use crate::normalize_log1p_parity::{Csr, scan, write_fixture};

/// The GPUs to check: each built-in runtime's device, if found. Prints why one is
/// skipped.
pub(crate) fn gpus() -> Vec<DeviceKernels> {
    let found = probe();
    let mut gpus = Vec::new();
    for (backend, device) in [
        (&Cuda as &dyn Backend, found.cuda().then_some(Device::Cuda)),
        (&Wgpu, found.wgpu().map(Device::Wgpu)),
    ] {
        let name = backend.runtime().name();
        match device {
            _ if !backend.built() => eprintln!(
                "SKIPPED {name}: kyto built without the '{name}' feature (--features {name})"
            ),
            None => eprintln!("SKIPPED {name}: no {name} GPU found on this machine"),
            Some(device) => gpus.push(DeviceKernels::open(device, "test").unwrap()),
        }
    }
    gpus
}

/// `|gpu - cpu| <= 1e-6 * |cpu|`, NaN for NaN, infinities equal, a subnormal CPU value
/// within 2^-126 (a GPU may flush it); the worst relative error.
fn assert_close<T: Into<f64> + Copy>(what: &str, gpu: &[T], cpu: &[T]) -> f64 {
    assert_eq!(gpu.len(), cpu.len(), "{what}: length");
    let mut worst = 0.0_f64;
    for (index, (&gpu, &cpu)) in gpu.iter().zip(cpu).enumerate() {
        let (gpu, cpu): (f64, f64) = (gpu.into(), cpu.into());
        if cpu.is_nan() || !cpu.is_finite() {
            assert!(
                gpu.to_bits() == cpu.to_bits() || (gpu.is_nan() && cpu.is_nan()),
                "{what} [{index}]: gpu {gpu} vs cpu {cpu}"
            );
            continue;
        }
        let error = (gpu - cpu).abs();
        if cpu.abs() < f64::from(f32::MIN_POSITIVE) {
            // A subnormal result: a GPU may flush it to zero (deviations.md).
            assert!(
                error < f64::from(f32::MIN_POSITIVE),
                "{what} [{index}]: gpu {gpu:e} vs cpu {cpu:e}"
            );
            continue;
        }
        assert!(
            error <= 1e-6 * cpu.abs(),
            "{what} [{index}]: gpu {gpu:e} vs cpu {cpu:e}"
        );
        if cpu != 0.0 {
            worst = worst.max(error / cpu.abs());
        }
    }
    worst
}

/// A splitmix64 stream: test data without a dependency.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Rows of random length (empty rows included) and values: `f32` over ten orders of
/// magnitude with stored zeros, `-0.0`; and `u32` counts, some above 2^24.
fn rows(seed: u64, n_rows: usize) -> (Vec<u64>, Vec<f32>, Vec<u32>) {
    let mut random = Random(seed);
    let mut indptr = vec![0_u64];
    let mut values = Vec::new();
    let mut counts = Vec::new();
    for _ in 0..n_rows {
        let len = match random.below(10) {
            0 => 0,
            1 => random.below(5000),
            _ => random.below(300),
        };
        for _ in 0..len {
            let value = match random.below(20) {
                0 => 0.0,
                1 => -0.0,
                _ => {
                    let mantissa = (random.below(1 << 24) as f32) / (1 << 24) as f32 + 0.5;
                    let exponent = random.below(10) as i32 - 3;
                    mantissa * 10_f32.powi(exponent)
                }
            };
            values.push(value);
            counts.push(match random.below(1000) {
                0 => (1 << 24) + random.below(1 << 30) as u32,
                _ => random.below(40) as u32,
            });
        }
        indptr.push(values.len() as u64);
    }
    (indptr, values, counts)
}

#[test]
fn each_kernel_matches_the_cpu_and_repeats_its_bits() {
    let cpu = Kernels::detected();
    for gpu in gpus() {
        let device = gpu.device();
        let (indptr, mut values, counts) = rows(7, 3000);
        // Non-finite values in their own rows: a NaN, an infinity, and both signs; a
        // subnormal with the NaN (nnz counts it; GPUs may flush it in arithmetic).
        let n = values.len();
        values.extend([
            f32::NAN,
            f32::from_bits(1000),
            f32::INFINITY,
            2.0,
            f32::INFINITY,
            -f32::INFINITY,
        ]);
        let indptr: Vec<u64> = indptr
            .into_iter()
            .chain([n as u64 + 2, n as u64 + 4, n as u64 + 6])
            .collect();
        let counts: Vec<u32> = counts.into_iter().chain([0, 3, 1, 0, 9, 1]).collect();

        // Row sums: f32 within 1e-6 (in fact ~1e-14); u32 counts exact.
        let mut expected = Vec::new();
        cpu.row_sums(&indptr, &values, &mut expected);
        let sums = gpu
            .row_sums(&indptr, RowValues::F32(&values), "test")
            .unwrap();
        let worst = assert_close(&format!("{device} row sums f32"), &sums, &expected);
        cpu.row_sums(&indptr, &counts, &mut expected);
        let count_sums = gpu
            .row_sums(&indptr, RowValues::U32(&counts), "test")
            .unwrap();
        assert_eq!(count_sums, expected, "{device}: u32 sums are exact");

        // Row nnz: the same counts.
        let mut expected_nnz = Vec::new();
        for (values, name) in [
            (RowValues::F32(&values), "f32"),
            (RowValues::U32(&counts), "u32"),
        ] {
            match values {
                RowValues::F32(values) => cpu.row_nnz(&indptr, values, &mut expected_nnz),
                RowValues::U32(values) => cpu.row_nnz(&indptr, values, &mut expected_nnz),
            }
            let nnz = gpu.row_nnz(&indptr, values, "test").unwrap();
            assert_eq!(nnz, expected_nnz, "{device}: nnz of {name}");
        }

        // Division by each row's divisor, f32 and u32 inputs.
        let rows = indptr.len() - 1;
        let divisors: Vec<f32> = (0..rows).map(|row| 0.37 + row as f32 * 1.3).collect();
        let mut divided = vec![0.0; values.len()];
        let mut expected_divided = vec![0.0; values.len()];
        gpu.divide_rows(
            &indptr,
            RowValues::F32(&values),
            &divisors,
            &mut divided,
            "test",
        )
        .unwrap();
        cpu.divide_rows(&indptr, &values, &divisors, &mut expected_divided);
        let worst_divide =
            assert_close(&format!("{device} divide f32"), &divided, &expected_divided);
        gpu.divide_rows(
            &indptr,
            RowValues::U32(&counts),
            &divisors,
            &mut divided,
            "test",
        )
        .unwrap();
        cpu.divide_rows(&indptr, &counts, &divisors, &mut expected_divided);
        assert_close(&format!("{device} divide u32"), &divided, &expected_divided);

        // Determinism: a second run gives the same bits.
        let again = gpu
            .row_sums(&indptr, RowValues::F32(&values), "test")
            .unwrap();
        let bits = |sums: &[f64]| sums.iter().map(|sum| sum.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&again), bits(&sums), "{device}: row sums repeat");
        let mut divided_again = vec![0.0; values.len()];
        gpu.divide_rows(
            &indptr,
            RowValues::U32(&counts),
            &divisors,
            &mut divided_again,
            "test",
        )
        .unwrap();
        assert!(
            divided_again
                .iter()
                .zip(&divided)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{device}: division repeats"
        );
        let same = sums
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        eprintln!(
            "{device}: row sums f32 worst relative error {worst:e}; division {worst_divide:e}; \
             f32 sums with the CPU's bits: {same} of {rows}"
        );
    }
}

/// Every 127th `f32` from 0 up to infinity, and from -0 down to -1 and past it.
fn log1p_inputs() -> Vec<f32> {
    let mut inputs: Vec<f32> = (0..=f32::INFINITY.to_bits())
        .step_by(127)
        .map(f32::from_bits)
        .collect();
    inputs.extend(
        ((-0.0_f32).to_bits()..=(-1.0_f32).to_bits())
            .step_by(127)
            .map(f32::from_bits),
    );
    inputs.extend([
        -1.0,
        -1.5,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
        f32::MAX,
        0.0,
        -0.0,
        1.0,
    ]);
    inputs
}

#[test]
fn log1p_matches_the_cpu_over_the_f32_range() {
    let cpu = Kernels::detected();
    for gpu in gpus() {
        let device = gpu.device();
        let inputs = log1p_inputs();
        // More than 65535 workgroups of 256 threads: a dispatch along two axes.
        assert!(inputs.len() > 65_535 * 256, "{}", inputs.len());
        let mut out = vec![0.0; inputs.len()];
        gpu.log1p(RowValues::F32(&inputs), &mut out, "test")
            .unwrap();
        let mut expected = vec![0.0; inputs.len()];
        cpu.log1p(&inputs, &mut expected);
        // Subnormal inputs (log1p(x) = x) may be flushed to zero by a GPU: compared
        // apart, against an absolute 2^-126.
        let normal = |x: &f32| x.is_nan() || x.abs() >= f32::MIN_POSITIVE || *x == 0.0;
        let (mut gpu_normal, mut cpu_normal) = (Vec::new(), Vec::new());
        let mut flushed = 0;
        for ((input, &gpu_value), &cpu_value) in inputs.iter().zip(&out).zip(&expected) {
            if normal(input) {
                gpu_normal.push(gpu_value);
                cpu_normal.push(cpu_value);
            } else {
                assert!(
                    (gpu_value - cpu_value).abs() < f32::MIN_POSITIVE,
                    "{input:e}"
                );
                flushed += usize::from(gpu_value != cpu_value);
            }
        }
        let worst = assert_close(&format!("{device} log1p"), &gpu_normal, &cpu_normal);
        let same = out
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        // u32 counts, rounded to f32 first.
        let counts: Vec<u32> = (0..100_000).map(|x| x * 977).collect();
        let mut out_counts = vec![0.0; counts.len()];
        let mut expected_counts = vec![0.0; counts.len()];
        gpu.log1p(RowValues::U32(&counts), &mut out_counts, "test")
            .unwrap();
        cpu.log1p(&counts, &mut expected_counts);
        assert_close(
            &format!("{device} log1p u32"),
            &out_counts,
            &expected_counts,
        );
        let mut again = vec![0.0; inputs.len()];
        gpu.log1p(RowValues::F32(&inputs), &mut again, "test")
            .unwrap();
        assert!(
            again
                .iter()
                .zip(&out)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{device}: log1p repeats"
        );
        eprintln!(
            "{device}: log1p over {} inputs, worst relative error {worst:e}, the CPU's bits \
             for {same}, subnormal outputs flushed: {flushed}",
            inputs.len()
        );
    }
}

/// `normalize_total(target_sum)` then `log1p` of pbmc3k's counts, on `request` with
/// `devices`, collected as CSR, and the seconds the run took.
fn run(
    store: &std::path::Path,
    target_sum: Option<f64>,
    request: DeviceRequest,
    devices: Devices,
) -> ([Csr; 2], f64) {
    let plan = scan(store)
        .normalize_total(target_sum, Some("counts"), "normalized")
        .unwrap()
        .log1p(Some("normalized"), "log1p")
        .unwrap()
        .on_device(request)
        .unwrap();
    let layers = ["normalized", "log1p"];
    let output =
        Output::members(&plan, layers.map(|layer| Member::Layer(layer.to_owned()))).unwrap();
    let options = PhysicalOptions {
        devices,
        ..PhysicalOptions::default()
    };
    let physical = physical_plan(std::slice::from_ref(&output), OptimizeOptions::ALL, options);
    let started = Instant::now();
    let threads = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    let stream = Pipeline::new(&physical, 0)
        .unwrap()
        .run(ExecuteOptions::new(threads))
        .unwrap();
    let mut csr = layers.map(|_| Csr {
        indptr: vec![0],
        indices: Vec::new(),
        values: Vec::new(),
    });
    for block in stream {
        let block = block.unwrap();
        for (layer, csr) in layers.iter().zip(&mut csr) {
            let cells = &block.layers[*layer];
            let CellValues::F32(values) = cells.data() else {
                panic!("{layer} is float32")
            };
            let offset = csr.values.len() as u64;
            csr.indptr
                .extend(cells.indptr()[1..].iter().map(|&end| offset + end));
            csr.indices.extend_from_slice(cells.indices());
            csr.values.extend_from_slice(values);
        }
    }
    (csr, started.elapsed().as_secs_f64())
}

#[test]
fn normalize_and_log1p_on_the_gpu_match_the_cpu_on_pbmc3k() {
    let gpus = gpus();
    if gpus.is_empty() {
        return;
    }
    let work = tempfile::tempdir().unwrap();
    let h5ad = work.path().join("pbmc3k.h5ad");
    let reference = work.path().join("reference");
    std::fs::create_dir(&reference).unwrap();
    if !write_fixture("pbmc3k", &h5ad, &reference) {
        eprintln!("SKIPPED pbmc3k: anndata is not installed on this platform");
        return;
    }
    let store = work.path().join("pbmc3k.kyto");
    kyto_store::convert(&h5ad, &store, &ConvertOptions::default()).unwrap();
    for gpu in gpus {
        let device = gpu.device();
        let (request, devices) = match device {
            Device::Cuda => (DeviceRequest::Cuda, Devices::new(true, None)),
            Device::Wgpu(backend) => (DeviceRequest::Wgpu, Devices::new(false, Some(backend))),
            Device::Cpu => unreachable!("gpus() holds GPUs"),
        };
        for target_sum in [Some(1e4), None] {
            let (cpu, cpu_seconds) = run(&store, target_sum, DeviceRequest::Cpu, devices);
            let (on_gpu, gpu_seconds) = run(&store, target_sum, request, devices);
            let (again, _) = run(&store, target_sum, request, devices);
            for ((layer, cpu), (on_gpu, again)) in ["normalized", "log1p"]
                .iter()
                .zip(&cpu)
                .zip(on_gpu.iter().zip(&again))
            {
                let what = format!("{device} {layer} target_sum={target_sum:?}");
                assert_eq!(on_gpu.indptr, cpu.indptr, "{what}: structure");
                assert_eq!(on_gpu.indices, cpu.indices, "{what}: structure");
                let worst = assert_close(&what, &on_gpu.values, &cpu.values);
                assert_eq!(
                    again.bits(),
                    on_gpu.bits(),
                    "{what}: the same bits run to run"
                );
                let same = on_gpu
                    .bits()
                    .iter()
                    .zip(cpu.bits())
                    .filter(|(a, b)| **a == *b)
                    .count();
                eprintln!(
                    "{what}: worst relative error {worst:e}, the CPU's bits for {same} of {} values",
                    cpu.values.len()
                );
            }
            eprintln!(
                "{device} pbmc3k normalize_total + log1p (target_sum={target_sum:?}, all \
                 threads, one run): cpu {cpu_seconds:.3} s, {device} {gpu_seconds:.3} s"
            );
        }
    }
}

/// `provenance.json` of `store` without `run`, as a JSON value.
fn provenance_without_run(store: &std::path::Path) -> serde_json::Value {
    let text = std::fs::read_to_string(store.join("provenance.json")).unwrap();
    let mut provenance: serde_json::Value = serde_json::from_str(&text).unwrap();
    provenance.as_object_mut().unwrap().remove("run");
    provenance
}

#[test]
fn provenance_records_the_gpu_steps_of_an_auto_sink() {
    let found = probe();
    let gpus = gpus();
    if gpus.is_empty() {
        return;
    }
    let work = tempfile::tempdir().unwrap();
    let h5ad = work.path().join("synthetic.h5ad");
    let reference = work.path().join("reference");
    std::fs::create_dir(&reference).unwrap();
    if !write_fixture("synthetic", &h5ad, &reference) {
        eprintln!("SKIPPED synthetic: anndata is not installed on this platform");
        return;
    }
    let store = work.path().join("synthetic.kyto");
    kyto_store::convert(&h5ad, &store, &ConvertOptions::default()).unwrap();
    // Nodes: scan 0, normalize_total 1, log1p 2, on 3, sink 4.
    let plan = scan(&store)
        .normalize_total(Some(1e4), Some("counts"), "normalized")
        .unwrap()
        .log1p(Some("normalized"), "log1p")
        .unwrap()
        .on_device(DeviceRequest::Auto)
        .unwrap();
    let mut written = Vec::new();
    for threads in [1, 4] {
        let target = work.path().join(format!("auto-{threads}.kyto"));
        let sunk = plan
            .sink(Sink::Zarr {
                path: target.display().to_string(),
            })
            .unwrap();
        let options = PhysicalOptions {
            devices: found,
            ..PhysicalOptions::default()
        };
        let physical = physical_plan(&[Output::all(&sunk)], OptimizeOptions::ALL, options);
        Pipeline::new(&physical, 0)
            .unwrap()
            .sink(
                ExecuteOptions::new(NonZeroUsize::new(threads).unwrap()),
                false,
            )
            .unwrap();
        written.push(provenance_without_run(&target));
    }
    // The same bytes on any thread count (R11); the files are checked in sink_threads.
    assert_eq!(written[0], written[1]);
    // auto takes the first GPU found with a kernel: CUDA before wgpu.
    let gpu = gpus[0].device().name();
    let provenance = &written[0];
    assert_eq!(provenance["device"], gpu);
    let cpu = serde_json::json!({"device": "cpu", "device_note": null});
    let on_gpu = serde_json::json!({"device": gpu, "device_note": "within 1e-6 of cpu"});
    assert_eq!(
        provenance["devices"],
        serde_json::json!({
            "requested": "auto",
            "found": found.to_string(),
            "nodes": {"n0": cpu, "n1": on_gpu, "n2": on_gpu, "n4": cpu},
        })
    );
    eprintln!("{gpu}: provenance devices {}", provenance["devices"]);
}

/// The four QC metrics (the gene-set ones over any gene set: [`CellQc::new`] takes it).
fn qc_metrics() -> [QcMetric; 4] {
    let genes = || GeneExpr::column("mt");
    [
        QcMetric::TotalCounts,
        QcMetric::GenesDetected,
        QcMetric::TotalCountsIn { genes: genes() },
        QcMetric::PercentCountsIn { genes: genes() },
    ]
}

/// A block of the rows `indptr` cuts from `data`, each row's genes `0, 1, ...`.
fn block(indptr: &[u64], data: CellData, n_genes: u64) -> CellBlock {
    let indices: Vec<u32> = indptr
        .windows(2)
        .flat_map(|row| 0..(row[1] - row[0]) as u32)
        .collect();
    let parts = CellBlockParts {
        indptr: indptr.into(),
        indices: indices.as_slice().into(),
        data,
    };
    CellBlock::from_parts(parts, n_genes).unwrap()
}

#[test]
fn qc_metrics_on_the_gpu_match_the_cpu() {
    let n_genes = 5000_u64;
    let set = GeneSet::from_mask((0..n_genes).map(|gene| gene % 3 == 0).collect());
    for gpu in gpus() {
        let device = gpu.device();
        let (mut indptr, mut values, counts) = rows(11, 2000);
        let counts = block(&indptr, CellData::U32(counts.as_slice().into()), n_genes);
        // Rows whose float32 sum overflows, or holding NaN or infinities: summed again
        // on the CPU.
        for row in [
            [f32::MAX, f32::MAX, 1.0],
            [f32::NAN, 1.0, 2.0],
            [f32::INFINITY, f32::NEG_INFINITY, 0.0],
            [f32::INFINITY, 3.0, f32::MAX],
            // Subnormals, which a GPU may flush to zero.
            [1e-40, 1e-40, 1e-40],
        ] {
            values.extend(row);
            indptr.push(values.len() as u64);
        }
        let floats = block(&indptr, CellData::F32(values.as_slice().into()), n_genes);
        for metric in qc_metrics() {
            let cpu = CellQc::new(&metric, Some(set.clone()));
            let on_gpu = cpu.clone().on_gpu(Some(gpu.clone()));
            let what = format!("{device} {}", metric.function());
            // Counts: the CPU's bits.
            let (expected, got) = (
                cpu.evaluate(&counts).unwrap(),
                on_gpu.evaluate(&counts).unwrap(),
            );
            match (&expected, &got) {
                (QcValues::Float64(expected), QcValues::Float64(got)) => {
                    let bits =
                        |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(got), bits(expected), "{what} counts");
                }
                _ => assert_eq!(got, expected, "{what} counts"),
            }
            // float32: within 1e-6 relative, non-finite rows as the CPU.
            let (expected, got) = (
                cpu.evaluate(&floats).unwrap(),
                on_gpu.evaluate(&floats).unwrap(),
            );
            let worst = match (&expected, &got) {
                (QcValues::Float64(expected), QcValues::Float64(got)) => {
                    let tail = expected.len() - 5;
                    for (index, (got, expected)) in
                        got[tail..].iter().zip(&expected[tail..]).enumerate()
                    {
                        assert!(
                            got.to_bits() == expected.to_bits()
                                || (got.is_nan() && expected.is_nan()),
                            "{what} overflow row {index}: gpu {got} vs cpu {expected}"
                        );
                    }
                    assert_close(&format!("{what} float32"), got, expected)
                }
                _ => {
                    assert_eq!(got, expected, "{what} float32");
                    0.0
                }
            };
            let again = on_gpu.evaluate(&floats).unwrap();
            match (&again, &got) {
                (QcValues::Float64(again), QcValues::Float64(got)) => assert!(
                    again
                        .iter()
                        .zip(got)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{what}: the same bits run to run"
                ),
                _ => assert_eq!(again, got),
            }
            eprintln!("{what}: float32 worst relative error {worst:e}");
        }
    }
}

/// The kept cells (store positions) and each QC column, as `f64`.
type QcColumns = (Vec<u64>, Vec<Vec<f64>>);

/// The QC columns of [`qc_plan`].
const QC_COLUMNS: [&str; 4] = [
    "total_counts",
    "n_genes_by_counts",
    "total_counts_mt",
    "pct_counts_mt",
];

/// `with_cells` of every QC metric of `layer` (the gene set: symbols starting `MT-`),
/// and a `filter_cells` on one, on `request`.
fn qc_plan(store: &std::path::Path, layer: &str, request: DeviceRequest) -> Arc<Plan> {
    let qc = |metric| CellExpr::qc(Some(layer.to_owned()), metric).unwrap();
    let mitochondrial = || {
        GeneExpr::column("symbol")
            .str(StrFunction::StartsWith("MT-".to_owned()))
            .unwrap()
    };
    let detected = Expr::binary(
        qc(QcMetric::GenesDetected),
        BinaryOp::GtEq,
        Literal::Int(1000),
    )
    .unwrap();
    scan(store)
        .with_cells([
            ("total_counts".to_owned(), qc(QcMetric::TotalCounts)),
            ("n_genes_by_counts".to_owned(), qc(QcMetric::GenesDetected)),
            (
                "total_counts_mt".to_owned(),
                qc(QcMetric::TotalCountsIn {
                    genes: mitochondrial(),
                }),
            ),
            (
                "pct_counts_mt".to_owned(),
                qc(QcMetric::PercentCountsIn {
                    genes: mitochondrial(),
                }),
            ),
        ])
        .unwrap()
        .filter_cells(&detected)
        .unwrap()
        .on_device(request)
        .unwrap()
}

/// The kept cells and each QC column (as `f64`) of [`qc_plan`] on `request`, the
/// optimized plan's explain text, and the seconds the run took.
fn run_qc(
    store: &std::path::Path,
    layer: &str,
    request: DeviceRequest,
    devices: Devices,
) -> (QcColumns, String, f64) {
    let plan = qc_plan(store, layer, request);
    let output = Output::members(
        &plan,
        QC_COLUMNS.map(|name| Member::CellColumn(name.to_owned())),
    )
    .unwrap();
    let options = PhysicalOptions {
        devices,
        ..PhysicalOptions::default()
    };
    let physical = physical_plan(std::slice::from_ref(&output), OptimizeOptions::ALL, options);
    let explained = Explain::new(&plan, &physical).to_string();
    let started = Instant::now();
    let threads = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    let stream = Pipeline::new(&physical, 0)
        .unwrap()
        .run(ExecuteOptions::new(threads))
        .unwrap();
    let mut cells = Vec::new();
    let mut columns = vec![Vec::new(); QC_COLUMNS.len()];
    for block in stream {
        let block = block.unwrap();
        cells.extend_from_slice(&block.cells);
        for (values, name) in columns.iter_mut().zip(QC_COLUMNS) {
            let column = block
                .columns
                .column(name)
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap();
            values.extend(column.f64().unwrap().iter().map(Option::unwrap));
        }
    }
    ((cells, columns), explained, started.elapsed().as_secs_f64())
}

#[test]
fn qc_steps_on_the_gpu_match_the_cpu_on_pbmc3k() {
    let gpus = gpus();
    if gpus.is_empty() {
        return;
    }
    let work = tempfile::tempdir().unwrap();
    let h5ad = work.path().join("pbmc3k.h5ad");
    let reference = work.path().join("reference");
    std::fs::create_dir(&reference).unwrap();
    if !write_fixture("pbmc3k", &h5ad, &reference) {
        eprintln!("SKIPPED pbmc3k: anndata is not installed on this platform");
        return;
    }
    let store = work.path().join("pbmc3k.kyto");
    kyto_store::convert(&h5ad, &store, &ConvertOptions::default()).unwrap();
    // A float32 layer, written by a CPU run.
    let floats = work.path().join("log1p.kyto");
    let sunk = scan(&store)
        .normalize_total(Some(1e4), Some("counts"), "normalized")
        .unwrap()
        .log1p(Some("normalized"), "log1p")
        .unwrap()
        .sink(Sink::Zarr {
            path: floats.display().to_string(),
        })
        .unwrap();
    let physical = physical_plan(
        &[Output::all(&sunk)],
        OptimizeOptions::ALL,
        PhysicalOptions::default(),
    );
    Pipeline::new(&physical, 0)
        .unwrap()
        .sink(ExecuteOptions::new(NonZeroUsize::new(4).unwrap()), false)
        .unwrap();
    for gpu in gpus {
        let device = gpu.device();
        let (request, devices) = match device {
            Device::Cuda => (DeviceRequest::Cuda, Devices::new(true, None)),
            Device::Wgpu(backend) => (DeviceRequest::Wgpu, Devices::new(false, Some(backend))),
            Device::Cpu => unreachable!("gpus() holds GPUs"),
        };
        for (store, layer) in [(&store, "counts"), (&floats, "log1p")] {
            let what = format!("{device} pbmc3k {layer}");
            let (cpu, _, cpu_seconds) = run_qc(store, layer, DeviceRequest::Cpu, devices);
            let (on_gpu, explained, gpu_seconds) = run_qc(store, layer, request, devices);
            let (again, _, _) = run_qc(store, layer, request, devices);
            // explain() names the GPU on both QC steps.
            for step in ["WITH_CELLS", "FILTER_CELLS"] {
                let line = explained
                    .lines()
                    .find(|line| line.contains(step) && line.contains("[device="))
                    .unwrap_or_else(|| panic!("{what}: no {step} in {explained}"));
                assert!(
                    line.contains(&format!("[device={} (within 1e-6 of cpu)", device.name())),
                    "{what}: {line}"
                );
            }
            assert!(
                cpu.0.len() > 100 && cpu.0.len() < 2700,
                "{what}: {} cells",
                cpu.0.len()
            );
            assert_eq!(on_gpu.0, cpu.0, "{what}: the same cells");
            let bits = |columns: &[Vec<f64>]| -> Vec<Vec<u64>> {
                columns
                    .iter()
                    .map(|values| values.iter().map(|v| v.to_bits()).collect())
                    .collect()
            };
            assert_eq!(
                bits(&again.1),
                bits(&on_gpu.1),
                "{what}: the same bits run to run"
            );
            for ((name, gpu_values), cpu_values) in QC_COLUMNS.iter().zip(&on_gpu.1).zip(&cpu.1) {
                if layer == "counts" {
                    assert_eq!(
                        bits(std::slice::from_ref(gpu_values)),
                        bits(std::slice::from_ref(cpu_values)),
                        "{what} {name}: the CPU's bits"
                    );
                    eprintln!(
                        "{what} {name}: the CPU's bits for all {} cells",
                        cpu_values.len()
                    );
                } else {
                    let worst = assert_close(&format!("{what} {name}"), gpu_values, cpu_values);
                    eprintln!("{what} {name}: worst relative error {worst:e}");
                }
            }
            eprintln!(
                "{what} QC (with_cells + filter_cells, all threads, one run): cpu \
                 {cpu_seconds:.3} s, {device} {gpu_seconds:.3} s"
            );
        }
    }
}
