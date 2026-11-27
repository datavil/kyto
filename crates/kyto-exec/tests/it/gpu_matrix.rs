//! Task 9.8 Done when: on a Metal device, PCA matches CPU PCA within the 6.5 tolerances
//! and neighbours equal CPU exact mode.
//!
//! Runs on every GPU this build and machine have (wgpu with `--features wgpu`, CUDA with
//! `--features cuda`, through CubeCL); without one each test prints `SKIPPED` and checks
//! nothing (as `gpu_kernels.rs`). Run on purpose:
//! `uv run cargo test -p kyto-exec --features wgpu --test it -- gpu_matrix:: --nocapture`.
//!
//! Checked per GPU: the matrix kernels on random blocks against `f64` sums of the same
//! products (`X D` and `X^T D` within 1e-10 of the terms' magnitude: exact `f32`
//! products, compensated sums), `X^T D`
//! folded in two parts with the bits of one call, the same bits on a second run, and
//! the candidates of exact neighbours (with ties and duplicate rows, both metrics, two
//! tiles of queries) giving the CPU's graph. On pbmc3k (6.5's steps): `pca` placed on
//! the GPU against the CPU's `pca` and against scanpy's reference by the 6.5
//! tolerances (`tests/parity/tolerances.toml`, not loosened), the same bits at 1 and
//! all threads, with blocks of 37 cells and streamed instead of held (R11), and on a
//! rerun; `neighbors` on the GPU, Euclidean and cosine, on the CPU's embedding (handed
//! over with `with_embedding`), equal to the CPU's exact graph and connectivities;
//! `explain()` shows each step's device and note. Large gene means (offsets up to 1e6
//! times the spread) stay within the tolerances of the CPU, and values outside 1e-10 to
//! 1e10 are refused with `KY-E215` (bug-hunter). One timing of each for the record
//! (not a benchmark: M6).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use kyto_core::Tiling;
use kyto_exec::{ExecuteOptions, NeighborStatistics, NodeStatistics, PcaStatistics, Pipeline};
use kyto_gpu::{Device, DeviceKernels, PairSums, PairwiseKey, SparseRows};
use kyto_plan::{
    DeviceRequest, Devices, Explain, GivenValues, Member, Metric, NeighborSearch, Neighbors,
    OptimizeOptions, Output, PhysicalOptions, Plan, WithEmbedding, physical_plan,
};
use kyto_store::ConvertOptions;

use crate::gpu_kernels::gpus;
use crate::pca_parity::{
    column, column_difference, largest_angle, read_f64, read_lines, relative, scan, write_fixture,
};
use crate::tolerances::tolerance;

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

    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The request and devices that place a plan on `gpu`.
fn request(gpu: &DeviceKernels) -> (DeviceRequest, Devices) {
    match gpu.device() {
        Device::Cuda => (DeviceRequest::Cuda, Devices::new(true, None)),
        Device::Wgpu(backend) => (DeviceRequest::Wgpu, Devices::new(false, Some(backend))),
        Device::Cpu => unreachable!("gpus() holds GPUs"),
    }
}

/// A random CSR block: `n_rows` rows over `n_columns` columns, about one value in 12,
/// over several orders of magnitude.
fn random_rows(seed: u64, n_rows: usize, n_columns: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let mut random = Random(seed);
    let (mut indptr, mut columns, mut values) = (vec![0_u32], Vec::new(), Vec::new());
    for _ in 0..n_rows {
        for column in 0..n_columns {
            if random.below(12) == 0 {
                columns.push(column as u32);
                let scale = [1e-3_f32, 1.0, 1e3][random.below(3) as usize];
                values.push(random.unit() * scale);
            }
        }
        indptr.push(values.len() as u32);
    }
    (indptr, columns, values)
}

/// `|gpu - exact| <= 1e-10 * magnitude` for each value (exact products, compensated
/// sums); the worst ratio.
fn assert_within(what: &str, gpu: &[f64], exact: &[f64], magnitude: &[f64]) -> f64 {
    assert_eq!(gpu.len(), exact.len(), "{what}: length");
    let mut worst = 0.0_f64;
    for (index, ((&gpu, &exact), &magnitude)) in gpu.iter().zip(exact).zip(magnitude).enumerate() {
        let error = (gpu - exact).abs();
        assert!(
            error <= 1e-10 * magnitude.max(f64::MIN_POSITIVE),
            "{what} [{index}]: gpu {gpu:e} vs {exact:e} (terms {magnitude:e})"
        );
        if magnitude > 0.0 {
            worst = worst.max(error / magnitude);
        }
    }
    worst
}

#[test]
fn sparse_products_match_f64_sums_and_continue_across_calls() {
    for gpu in gpus() {
        let device = gpu.device();
        let (n_rows, n_columns, width) = (300, 500, 7);
        let (indptr, columns, values) = random_rows(1, n_rows, n_columns);
        let rows = SparseRows::new(&indptr, &columns, &values, n_columns);
        let mut random = Random(2);
        let dense: Vec<f64> = (0..n_columns * width)
            .map(|_| f64::from(random.unit()) * (1.0 + 1e-9))
            .collect();
        let start: Vec<f64> = (0..width).map(|j| j as f64 * 0.25 - 1.0).collect();

        // X D against f64 sums of the same f32 products.
        let product = gpu
            .sparse_dense(&rows, &dense, width, &start, "test")
            .unwrap();
        let (mut exact, mut magnitude) = (Vec::new(), Vec::new());
        for row in 0..n_rows {
            for j in 0..width {
                let (mut sum, mut terms) = (start[j], start[j].abs());
                for entry in indptr[row] as usize..indptr[row + 1] as usize {
                    let term =
                        f64::from(values[entry]) * dense[columns[entry] as usize * width + j];
                    sum += term;
                    terms += term.abs();
                }
                exact.push(sum);
                magnitude.push(terms);
            }
        }
        let worst = assert_within(&format!("{device} X D"), &product, &exact, &magnitude);
        let again = gpu
            .sparse_dense(&rows, &dense, width, &start, "test")
            .unwrap();
        assert_eq!(again, product, "{device} X D: the same bits run to run");

        // X^T D, in one call and in two (the rows split at 137): the same bits.
        let tall: Vec<f32> = (0..n_rows * width).map(|_| random.unit()).collect();
        let mut whole = PairSums::zeros(n_columns * width);
        gpu.sparse_transposed_dense(&rows, &tall, width, &mut whole, "test")
            .unwrap();
        let split = 137;
        let first = indptr[split] as usize;
        let head_indptr = &indptr[..=split];
        let tail_indptr: Vec<u32> = indptr[split..]
            .iter()
            .map(|&end| end - first as u32)
            .collect();
        let mut parts = PairSums::zeros(n_columns * width);
        let head = SparseRows::new(head_indptr, &columns[..first], &values[..first], n_columns);
        gpu.sparse_transposed_dense(&head, &tall[..split * width], width, &mut parts, "test")
            .unwrap();
        let tail = SparseRows::new(&tail_indptr, &columns[first..], &values[first..], n_columns);
        gpu.sparse_transposed_dense(&tail, &tall[split * width..], width, &mut parts, "test")
            .unwrap();
        assert_eq!(
            parts, whole,
            "{device} X^T D: two calls give one call's bits"
        );
        let (mut exact, mut magnitude) = (
            vec![0.0_f64; n_columns * width],
            vec![0.0_f64; n_columns * width],
        );
        for row in 0..n_rows {
            for entry in indptr[row] as usize..indptr[row + 1] as usize {
                let column = columns[entry] as usize;
                for j in 0..width {
                    let term = f64::from(values[entry]) * f64::from(tall[row * width + j]);
                    exact[column * width + j] += term;
                    magnitude[column * width + j] += term.abs();
                }
            }
        }
        let worst_transposed = assert_within(
            &format!("{device} X^T D"),
            &whole.to_f64(),
            &exact,
            &magnitude,
        );
        eprintln!(
            "{device}: X D worst {worst:.1e}, X^T D worst {worst_transposed:.1e} of the \
             terms' magnitude"
        );
    }
}

/// `n` rows of `width` small integers (many ties), the last `duplicates` rows copies
/// of earlier ones, and two zero rows (cosine distance 1 to everything).
fn tied_points(n: usize, width: usize, duplicates: usize) -> Vec<f32> {
    let mut random = Random(3);
    let mut points: Vec<f32> = (0..n * width)
        .map(|_| random.below(5) as f32 - 2.0)
        .collect();
    for row in n - duplicates..n {
        let source = random.below((n - duplicates) as u64) as usize;
        let copied = points[source * width..(source + 1) * width].to_vec();
        points[row * width..(row + 1) * width].copy_from_slice(&copied);
    }
    for row in [3, 11] {
        points[row * width..(row + 1) * width].fill(0.0);
    }
    points
}

/// `neighbors` of the embedding `values` (`width` a row) handed to `base` with
/// `with_embedding`, on `request`; its statistics, the explained plan and the seconds.
fn run_neighbors(
    base: &Arc<Plan>,
    values: &[f32],
    width: u64,
    metric: Metric,
    request: DeviceRequest,
    devices: Devices,
    threads: usize,
) -> (NeighborStatistics, String, f64) {
    let plan = base
        .with_embedding(WithEmbedding::new(
            "X_given",
            width,
            GivenValues::Float32(values.to_vec()),
        ))
        .unwrap()
        .neighbors(Neighbors {
            embedding: "X_given".to_owned(),
            n_components: None,
            n_neighbors: 15,
            metric,
            search: NeighborSearch::Exact,
            seed: 0,
            key: None,
        })
        .unwrap()
        .on_device(request)
        .unwrap();
    let output = Output::members(&plan, [Member::Graph("distances".to_owned())]).unwrap();
    let options = PhysicalOptions {
        devices,
        ..PhysicalOptions::default()
    };
    let physical = physical_plan(std::slice::from_ref(&output), OptimizeOptions::ALL, options);
    let explained = Explain::new(&plan, &physical).to_string();
    let started = Instant::now();
    let stream = Pipeline::new(&physical, 0)
        .unwrap()
        .run(ExecuteOptions::new(NonZeroUsize::new(threads).unwrap()))
        .unwrap();
    let found = stream
        .statistics()
        .iter()
        .find_map(|(_, statistics)| match statistics {
            NodeStatistics::Neighbors(neighbors) => Some((**neighbors).clone()),
            _ => None,
        })
        .expect("neighbors' statistics");
    (found, explained, started.elapsed().as_secs_f64())
}

/// The optimized plan's line of `explained` with `method`'s node.
fn explain_line<'a>(explained: &'a str, method: &str) -> &'a str {
    explained
        .lines()
        .find(|line| line.contains(method) && line.contains("[device="))
        .unwrap_or_else(|| panic!("{method} in {explained}"))
}

/// A store of pbmc3k (written by `tests/fixtures`) in `work` with its scanpy PCA
/// reference; `None` without anndata.
fn pbmc3k(work: &Path) -> Option<(std::path::PathBuf, std::path::PathBuf, (u64, f64, u64))> {
    let h5ad = work.join("pbmc3k.h5ad");
    let reference = work.join("reference");
    std::fs::create_dir(&reference).unwrap();
    let fixture = write_fixture("pbmc3k", &h5ad, &reference)?;
    let store = work.join("pbmc3k.kyto");
    let options = ConvertOptions {
        tiling: Tiling {
            cell_block_size: 256,
            gene_block_size: 4096,
        },
        ..ConvertOptions::default()
    };
    kyto_store::convert(&h5ad, &store, &options).unwrap();
    Some((store, reference, fixture))
}

#[test]
fn neighbors_on_the_gpu_equal_the_cpu_with_ties_duplicates_and_tiles() {
    let gpus = gpus();
    if gpus.is_empty() {
        return;
    }
    let work = tempfile::tempdir().unwrap();
    let Some((store, _, _)) = pbmc3k(work.path()) else {
        eprintln!("SKIPPED pbmc3k: anndata is not installed on this platform");
        return;
    };
    let base = scan(&store);
    let n_cells = kyto_store::open_atlas_schema(&store).unwrap().n_cells as usize;
    let all = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let width = 6;
    let points = tied_points(n_cells, width, 200);
    for gpu in &gpus {
        let (request, devices) = request(gpu);
        let device = gpu.device();
        for metric in [Metric::Euclidean, Metric::Cosine] {
            let (cpu, _, _) = run_neighbors(
                &base,
                &points,
                width as u64,
                metric,
                DeviceRequest::Cpu,
                devices,
                all,
            );
            for threads in [1, all] {
                let (on_gpu, explained, _) = run_neighbors(
                    &base,
                    &points,
                    width as u64,
                    metric,
                    request,
                    devices,
                    threads,
                );
                assert_eq!(
                    on_gpu, cpu,
                    "{device} {metric:?} at {threads} threads: the CPU's exact graph"
                );
                let line = explain_line(&explained, "NEIGHBORS");
                assert!(
                    line.contains(&format!("[device={} (equal to cpu)", device.name())),
                    "{line}"
                );
            }
        }
        // Two tiles of queries on the device: 20,000 points, 1,000 queries.
        let mut random = Random(5);
        let points: Vec<f32> = (0..20_000 * 4).map(|_| random.unit()).collect();
        let queries = &points[..1_000 * 4];
        let skip: Vec<u32> = (0..1_000).collect();
        let (keys, found) = gpu
            .nearest_keys(
                queries,
                &points,
                4,
                PairwiseKey::SquaredEuclidean,
                &skip,
                20,
                "test",
            )
            .unwrap();
        for query in 0..1_000 {
            let mut exact: Vec<(f64, u32)> = (0..20_000_u32)
                .filter(|&point| point != query as u32)
                .map(|point| {
                    let key = (0..4)
                        .map(|c| {
                            let d = f64::from(queries[query * 4 + c])
                                - f64::from(points[point as usize * 4 + c]);
                            d * d
                        })
                        .sum::<f64>();
                    (key, point)
                })
                .collect();
            exact.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let row = &found[query * 20..(query + 1) * 20];
            let row_keys = &keys[query * 20..(query + 1) * 20];
            assert!(row_keys.windows(2).all(|pair| pair[0] <= pair[1]), "sorted");
            // The 19 nearest by f64 are among the 20 (no near-ties at this scale).
            for (_, point) in &exact[..19] {
                assert!(row.contains(point), "{device} query {query}: point {point}");
            }
        }
    }
}

/// `pca` of 6.5's pbmc3k steps on `request`: its statistics, the explained plan and the
/// seconds.
fn run_pca(
    plan: &Arc<Plan>,
    request: DeviceRequest,
    devices: Devices,
    threads: usize,
    block_cells: Option<u64>,
    hold_inputs: bool,
) -> (PcaStatistics, String, f64) {
    let plan = plan.on_device(request).unwrap();
    let output = Output::members(&plan, [Member::Embedding("X_pca".to_owned())]).unwrap();
    let options = PhysicalOptions {
        devices,
        max_block_cells: block_cells.and_then(NonZeroU64::new),
        hold_inputs,
        ..PhysicalOptions::default()
    };
    let physical = physical_plan(std::slice::from_ref(&output), OptimizeOptions::ALL, options);
    let explained = Explain::new(&plan, &physical).to_string();
    let started = Instant::now();
    let stream = Pipeline::new(&physical, 0)
        .unwrap()
        .run(ExecuteOptions::new(NonZeroUsize::new(threads).unwrap()))
        .unwrap();
    let found = stream
        .statistics()
        .iter()
        .find_map(|(_, statistics)| match statistics {
            NodeStatistics::Pca(pca) => Some((**pca).clone()),
            _ => None,
        })
        .expect("pca's statistics");
    (found, explained, started.elapsed().as_secs_f64())
}

/// 6.5's comparisons of `pca` with `reference` (variances, loadings genes x k and
/// embedding cells x k), each asserted against its tolerance: `(variance, ratio,
/// leading angle, all angle, loadings, embedding)`.
fn compare_pca(
    what: &str,
    pca: &PcaStatistics,
    variance: &[f64],
    variance_ratio: &[f64],
    loadings: &[f64],
    embedding: &[f64],
) -> [f64; 6] {
    let k = pca.n_components();
    let compared = 10;
    let kyto_loadings: Vec<f64> = pca.loadings().iter().map(|&value| value.into()).collect();
    let mut worst_loading: f64 = 0.0;
    let mut worst_embedding: f64 = 0.0;
    for component in 0..compared {
        worst_loading = worst_loading.max(column_difference(
            &column(&kyto_loadings, k, component),
            &column(loadings, k, component),
        ));
        worst_embedding = worst_embedding.max(column_difference(
            &column(pca.embedding(), k, component),
            &column(embedding, k, component),
        ));
    }
    let measured = [
        relative(pca.variance(), variance),
        relative(pca.variance_ratio(), variance_ratio),
        largest_angle(&kyto_loadings, loadings, k, compared),
        largest_angle(&kyto_loadings, loadings, k, k),
        worst_loading,
        worst_embedding,
    ];
    let limits = [
        ("pca.variance", tolerance("pca.variance")),
        ("pca.variance", tolerance("pca.variance")),
        ("pca.leading_angle", tolerance("pca.leading_angle")),
        ("pca.all_angle", tolerance("pca.all_angle")),
        ("pca.loadings", tolerance("pca.loadings")),
        ("pca.embedding", tolerance("pca.embedding")),
    ];
    eprintln!(
        "{what}: variance {:.1e}, ratio {:.1e}, angle {:.1e} rad (first {compared}), \
         {:.1e} (all {k}), loadings {:.1e}, embedding {:.1e}",
        measured[0], measured[1], measured[2], measured[3], measured[4], measured[5]
    );
    for (value, (name, limit)) in measured.iter().zip(limits) {
        assert!(*value <= limit, "{what}: {name} {value:e}");
    }
    measured
}

#[test]
fn pca_on_the_gpu_matches_the_cpu_and_neighbors_equal_it_on_pbmc3k() {
    let gpus = gpus();
    if gpus.is_empty() {
        return;
    }
    let work = tempfile::tempdir().unwrap();
    let Some((store, reference, (min_cells, target_sum, n_comps))) = pbmc3k(work.path()) else {
        eprintln!("SKIPPED pbmc3k: anndata is not installed on this platform");
        return;
    };
    let mut base = scan(&store);
    if min_cells > 0 {
        base = base
            .select_genes(read_lines(&reference.join("filtered")))
            .unwrap();
    }
    let plan = base
        .normalize_total(Some(target_sum), Some("counts"), "normalized")
        .unwrap()
        .log1p(Some("normalized"), "log1p")
        .unwrap()
        .select_genes(read_lines(&reference.join("hvg")))
        .unwrap()
        .pca(Some("log1p"), "X_pca", n_comps, 0, None)
        .unwrap();
    let all = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let scanpy = ["variance", "variance_ratio", "components", "embedding"]
        .map(|name| read_f64(&reference.join(name)));
    for gpu in &gpus {
        let (request, devices) = request(gpu);
        let device = gpu.device();
        let (cpu, _, cpu_seconds) = run_pca(&plan, DeviceRequest::Cpu, devices, all, None, true);
        let (on_gpu, explained, gpu_seconds) = run_pca(&plan, request, devices, all, None, true);
        let line = explain_line(&explained, "PCA");
        assert!(
            line.contains(&format!("[device={} (within 1e-3 of cpu)", device.name())),
            "{line}"
        );
        let cpu_loadings: Vec<f64> = cpu.loadings().iter().map(|&value| value.into()).collect();
        let cpu_embedding: Vec<f64> = cpu.embedding().iter().map(|&value| value.into()).collect();
        compare_pca(
            &format!("pbmc3k {device} pca vs cpu"),
            &on_gpu,
            cpu.variance(),
            cpu.variance_ratio(),
            &cpu_loadings,
            &cpu_embedding,
        );
        compare_pca(
            &format!("pbmc3k {device} pca vs scanpy"),
            &on_gpu,
            &scanpy[0],
            &scanpy[1],
            &scanpy[2],
            &scanpy[3],
        );
        assert_eq!(on_gpu.cells(), cpu.cells());
        // R11: the same bits on a rerun, at 1 thread, with split blocks, and streamed.
        for (threads, block_cells, hold) in [
            (all, None, true),
            (1, None, true),
            (all, Some(37), true),
            (4, Some(37), false),
        ] {
            let (again, _, _) = run_pca(&plan, request, devices, threads, block_cells, hold);
            assert_eq!(
                again, on_gpu,
                "{device} pca at {threads} threads, {block_cells:?} cells, held {hold}"
            );
        }
        eprintln!(
            "{device} pbmc3k pca ({n_comps} components, all threads, one run): cpu \
             {cpu_seconds:.3} s, {device} {gpu_seconds:.3} s"
        );

        // Neighbours on the GPU, on the CPU's embedding: the CPU's exact graph.
        let width = cpu.n_components() as u64;
        for metric in [Metric::Euclidean, Metric::Cosine] {
            let scan = scan(&store);
            let (cpu_graph, _, cpu_seconds) = run_neighbors(
                &scan,
                cpu.embedding(),
                width,
                metric,
                DeviceRequest::Cpu,
                devices,
                all,
            );
            let (gpu_graph, _, gpu_seconds) =
                run_neighbors(&scan, cpu.embedding(), width, metric, request, devices, all);
            assert_eq!(
                gpu_graph, cpu_graph,
                "{device} {metric:?}: the CPU's exact graph"
            );
            eprintln!(
                "{device} pbmc3k neighbors exact {metric:?} (15, {width} components, all \
                 threads, one run): equal to the cpu; cpu {cpu_seconds:.3} s, {device} \
                 {gpu_seconds:.3} s"
            );
        }
    }
}

/// `pca(n_components)` of the store's `values` layer on `request`.
fn pca_of(
    store: &Path,
    n_components: u64,
    request: DeviceRequest,
    devices: Devices,
) -> Result<PcaStatistics, kyto_error::KytoError> {
    let plan = scan(store)
        .pca(Some("values"), "X_pca", n_components, 0, None)?
        .on_device(request)?;
    let output = Output::members(&plan, [Member::Embedding("X_pca".to_owned())])?;
    let options = PhysicalOptions {
        devices,
        ..PhysicalOptions::default()
    };
    let physical = physical_plan(std::slice::from_ref(&output), OptimizeOptions::ALL, options);
    let stream = Pipeline::new(&physical, 0)?.run(ExecuteOptions::new(NonZeroUsize::MIN))?;
    Ok(stream
        .statistics()
        .iter()
        .find_map(|(_, statistics)| match statistics {
            NodeStatistics::Pca(pca) => Some((**pca).clone()),
            _ => None,
        })
        .expect("pca's statistics"))
}

#[test]
fn pca_on_the_gpu_keeps_large_means_and_refuses_values_its_products_cannot_hold() {
    // Bug-hunter: f32 products before the centring lost 4.2e-3 of the embedding at
    // means 1e6 times the spread; values of 1e-20 underflowed to a silently wrong PCA.
    let gpus = gpus();
    if gpus.is_empty() {
        return;
    }
    let work = tempfile::tempdir().unwrap();
    for gpu in &gpus {
        let (request, devices) = request(gpu);
        let device = gpu.device();
        for offset in [0.0_f32, 1e3, 1e5, 1e6] {
            let store = work.path().join(format!("{device}-offset{offset}.kyto"));
            let rows: Vec<Vec<f32>> = crate::pca::data(200, 100, 4, 1.0, 6)
                .into_iter()
                .map(|row| row.into_iter().map(|value| value + offset).collect())
                .collect();
            crate::pca::write_store(&store, &rows, false);
            let cpu = pca_of(&store, 5, DeviceRequest::Cpu, devices).unwrap();
            let on_gpu = pca_of(&store, 5, request, devices).unwrap();
            let loadings: Vec<f64> = cpu.loadings().iter().map(|&value| value.into()).collect();
            let embedding: Vec<f64> = cpu.embedding().iter().map(|&value| value.into()).collect();
            let k = cpu.n_components();
            let mut worst = relative(on_gpu.variance(), cpu.variance());
            for component in 0..k {
                worst = worst.max(column_difference(
                    &column(on_gpu.embedding(), k, component),
                    &column(&embedding, k, component),
                ));
                worst = worst.max(column_difference(
                    &column(on_gpu.loadings(), k, component),
                    &column(&loadings, k, component),
                ));
            }
            eprintln!("{device} offset {offset:e}: worst {worst:.1e} from the cpu");
            assert!(
                worst <= tolerance("pca.embedding"),
                "{device} offset {offset}: {worst:e}"
            );
        }
        for scale in [1e-20_f32, 1e30] {
            let store = work.path().join(format!("{device}-scale{scale}.kyto"));
            let rows: Vec<Vec<f32>> = crate::pca::data(100, 20, 3, 1.0, 7)
                .into_iter()
                .map(|row| row.into_iter().map(|value| value * scale).collect())
                .collect();
            crate::pca::write_store(&store, &rows, false);
            assert!(pca_of(&store, 3, DeviceRequest::Cpu, devices).is_ok());
            let error = pca_of(&store, 3, request, devices).unwrap_err();
            assert_eq!(error.code(), "KY-E215", "{device} scale {scale}: {error}");
            assert!(error.to_string().contains("1e-10 to 1e10"), "{error}");
        }
    }
}
