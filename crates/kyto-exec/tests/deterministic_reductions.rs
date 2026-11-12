//! Task 5.6: deterministic reductions.
//!
//! Done when: outputs are bit-identical at 1, 4 and all threads. Each store runs
//! plans with every statistics pass the executor has (a median `normalize_total`,
//! `highly_variable_genes` of every flavour, `seurat_v3`'s two passes) and an output
//! layer after a median, at 1, 4 and all threads, and also at other
//! `blocks_in_flight` and smaller blocks (`max_block_cells`, as a memory limit splits
//! them, task 5.5): the statistics and the output's values must be the same bits in
//! every run (compared by a fingerprint of every bit, independent of how the rows
//! are split into blocks). The stores: the synthetic and pbmc3k fixtures (written as
//! `.h5ad` by `tests/fixtures`, converted at 128-cell blocks so the tree has 4 and 22
//! leaves; skipped without anndata, exit code 77), and a small store written
//! here whose `float32` values span 13 orders of magnitude (so any change in the
//! order of additions changes the sums), read through a storage wrapper that delays
//! each tile read by a different amount per run, so threads finish in other orders.
//! A proptest draws threads, `blocks_in_flight` and block sizes on that store.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hash::{DefaultHasher, Hash, Hasher};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use kyto_core::{GeneIds, Tiling};
use kyto_exec::{CellValues, ExecuteOptions, GeneMoments, NodeStatistics, Pipeline};
use kyto_plan::{
    HvgFlavor, Member, OptimizeOptions, Output, PhysicalOptions, PhysicalPlan, Plan, physical_plan,
};
use kyto_store::{
    CellBlock, ConvertOptions, LayerSpec, LayerWriteOptions, StoreWriter, TableAxis, ValueDtype,
    convert, open_atlas_schema, read_ids,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};
use proptest::prelude::*;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::{MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey};

/// Delays every read by 0 to 3 ms, by the key and the run's `seed`, so blocks of a
/// window finish in an order that changes from run to run.
struct DelayingStore<S> {
    inner: S,
    seed: u64,
}

impl<S: ReadableStorageTraits> ReadableStorageTraits for DelayingStore<S> {
    // `get` and `get_partial` default to `get_partial_many`, so every read passes here.
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let ranges: Vec<ByteRange> = byte_ranges.collect();
        if self.seed > 0 {
            let mut hasher = DefaultHasher::new();
            (key.as_str(), self.seed).hash(&mut hasher);
            std::thread::sleep(Duration::from_micros(hasher.finish() % 3000));
        }
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

/// How one run is executed.
#[derive(Debug, Clone, Copy)]
struct RunOptions {
    threads: usize,
    blocks_in_flight: u64,
    max_block_cells: Option<u64>,
    /// 0: no delays; otherwise the [`DelayingStore`]'s seed.
    delay_seed: u64,
}

impl RunOptions {
    fn threads(threads: usize) -> Self {
        Self {
            threads,
            blocks_in_flight: 2,
            max_block_cells: None,
            delay_seed: 0,
        }
    }
}

fn all_threads() -> usize {
    std::thread::available_parallelism().map_or(1, NonZeroUsize::get)
}

fn scan(store: &Path) -> Arc<Plan> {
    let schema = open_atlas_schema(store).unwrap();
    let gene_ids = GeneIds::new(read_ids(store, TableAxis::Genes, schema.n_genes).unwrap());
    Plan::scan(store.display().to_string(), schema, gene_ids)
}

/// The plans of every statistics pass the executor runs, by name: HVG gene columns
/// (statistics only; `cell_ranger` on the wide-ranging store, `seurat_v3` on counts) and a layer after a median
/// `normalize_total`.
fn cases(base: &Arc<Plan>, layer: &str) -> Vec<(&'static str, Output)> {
    let median = base
        .normalize_total(None, Some(layer), "norm")
        .unwrap()
        .log1p(Some("norm"), "log1p")
        .unwrap();
    let hvg = |input: &Arc<Plan>, hvg_layer: &str, flavor: HvgFlavor| {
        let plan = input
            .highly_variable_genes(Some(hvg_layer), flavor, Some(50), None)
            .unwrap();
        Output::members(&plan, [Member::GeneColumn("highly_variable".to_owned())]).unwrap()
    };
    let mut cases = vec![
        (
            "median normalize_total, log1p",
            Output::members(&median, [Member::Layer("log1p".to_owned())]).unwrap(),
        ),
        (
            "seurat after a median",
            hvg(&median, "log1p", HvgFlavor::Seurat),
        ),
    ];
    // 6.3's refusals: seurat_v3 refuses non-counts (the wide-ranging store's `values`),
    // cell_ranger refuses equal percentile edges (the fixtures' many genes seen in no
    // cell, as scanpy); each runs where its columns can be computed.
    if layer == "values" {
        cases.push(("cell_ranger", hvg(base, layer, HvgFlavor::CellRanger)));
    } else {
        cases.push(("seurat_v3", hvg(base, layer, HvgFlavor::SeuratV3)));
    }
    cases
}

fn moments_bits(moments: &GeneMoments, hasher: &mut DefaultHasher) {
    moments.n_cells().hash(hasher);
    for value in moments.sums().iter().chain(moments.sums_of_squares()) {
        value.to_bits().hash(hasher);
    }
}

/// Run `output` and fingerprint every bit of its statistics and of its rows (cell,
/// genes, value bits; the same however the rows are split into blocks), with the
/// number of statistics values seen.
fn fingerprint(store: &Path, output: &Output, options: RunOptions) -> (u64, usize) {
    let physical_options = PhysicalOptions {
        blocks_in_flight: NonZeroU64::new(options.blocks_in_flight).unwrap(),
        max_block_cells: options
            .max_block_cells
            .map(|cells| NonZeroU64::new(cells).unwrap()),
        ..PhysicalOptions::default()
    };
    let physical: PhysicalPlan = physical_plan(
        std::slice::from_ref(output),
        OptimizeOptions::ALL,
        physical_options,
    );
    let storage = Arc::new(DelayingStore {
        inner: FilesystemStore::new(store).unwrap(),
        seed: options.delay_seed,
    });
    let stream = Pipeline::new(&physical, 0)
        .unwrap()
        .run_on(
            storage,
            ExecuteOptions::new(NonZeroUsize::new(options.threads).unwrap()),
        )
        .unwrap();
    let mut hasher = DefaultHasher::new();
    let mut n_statistics = 0;
    for (node, statistics) in stream.statistics().iter() {
        node.hash(&mut hasher);
        match statistics {
            NodeStatistics::MedianTotal(median) => {
                median.to_bits().hash(&mut hasher);
                n_statistics += 1;
            }
            NodeStatistics::HighlyVariableGenes(hvg) => {
                moments_bits(hvg.moments(), &mut hasher);
                n_statistics += hvg.moments().sums().len();
                if let Some(clipped) = hvg.clipped() {
                    for clip in clipped.clip() {
                        clip.to_bits().hash(&mut hasher);
                    }
                    moments_bits(clipped.moments(), &mut hasher);
                    n_statistics += clipped.clip().len();
                }
            }
            _ => panic!("a statistics kind this test does not know"),
        }
    }
    for block in stream {
        let block = block.unwrap();
        for (name, layer) in &block.layers {
            let values: Vec<u32> = match layer.data() {
                CellValues::F32(values) => values.iter().map(|value| value.to_bits()).collect(),
                CellValues::U32(values) => values.to_vec(),
            };
            for (row, cell) in block.cells.iter().enumerate() {
                let (start, end) = (
                    layer.indptr()[row] as usize,
                    layer.indptr()[row + 1] as usize,
                );
                (name, cell).hash(&mut hasher);
                layer.indices()[start..end].hash(&mut hasher);
                values[start..end].hash(&mut hasher);
            }
        }
    }
    (hasher.finish(), n_statistics)
}

/// Every case on `store` gives the same bits at every run option.
fn assert_bit_identical(name: &str, store: &Path, runs: &[RunOptions]) {
    let schema = open_atlas_schema(store).unwrap();
    let layer = schema.layers[0].name.clone();
    let base = scan(store);
    for (case, output) in cases(&base, &layer) {
        let (expected, n_statistics) = fingerprint(store, &output, RunOptions::threads(1));
        assert!(n_statistics > 0, "{name} {case}: no statistics computed");
        for &options in runs {
            assert_eq!(
                fingerprint(store, &output, options),
                (expected, n_statistics),
                "{name} {case}: other bits at {options:?} than on 1 thread"
            );
        }
    }
}

/// The runs every store is checked at: 1, 4 and all threads, then other windows,
/// smaller blocks and delayed reads.
fn runs() -> Vec<RunOptions> {
    let all = all_threads();
    vec![
        RunOptions::threads(4),
        RunOptions::threads(all),
        RunOptions {
            blocks_in_flight: 1,
            ..RunOptions::threads(all)
        },
        RunOptions {
            blocks_in_flight: 7,
            ..RunOptions::threads(all)
        },
        RunOptions {
            max_block_cells: Some(100),
            ..RunOptions::threads(4)
        },
        RunOptions {
            blocks_in_flight: 3,
            max_block_cells: Some(37),
            delay_seed: 1,
            ..RunOptions::threads(all)
        },
    ]
}

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

#[test]
fn reductions_on_fixtures_are_bit_identical_at_1_4_and_all_threads() {
    let work = tempfile::tempdir().unwrap();
    for name in ["synthetic", "pbmc3k"] {
        let h5ad = work.path().join(format!("{name}.h5ad"));
        if !write_h5ad(name, &h5ad) {
            eprintln!("skipped: anndata is not installed on this platform");
            return;
        }
        let store = work.path().join(format!("{name}.kyto"));
        let convert_options = ConvertOptions {
            tiling: Tiling {
                cell_block_size: 128,
                gene_block_size: 4096,
            },
            ..ConvertOptions::default()
        };
        convert(&h5ad, &store, &convert_options).unwrap();
        assert_bit_identical(name, &store, &runs());
    }
}

const N_CELLS: usize = 700;
const N_GENES: usize = 200;
const TILING: Tiling = Tiling {
    cell_block_size: 64,
    gene_block_size: 128,
};

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

/// A store of `float32` values from 1e-6 to 1e7, about one entry in four stored.
fn write_store(target: &Path) {
    let mut random = Lcg(5);
    let mut indptr = vec![0_u64];
    let (mut indices, mut data) = (Vec::new(), Vec::new());
    for _ in 0..N_CELLS {
        for gene in 0..N_GENES as u32 {
            let draw = random.next();
            if draw.is_multiple_of(4) {
                let exponent = (draw / 4 % 14) as i32 - 6;
                let mantissa = 1.0 + (draw / 64 % 1000) as f32 / 1000.0;
                indices.push(gene);
                data.push(mantissa * 10_f32.powi(exponent));
            }
        }
        indptr.push(indices.len() as u64);
    }
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
        name: "values".to_owned(),
        dtype: ValueDtype::F32,
        options: LayerWriteOptions::default(),
    }];
    let mut writer = StoreWriter::create(target, &genes, &specs, TILING, false).unwrap();
    writer
        .push(&[CellBlock::from_f32(indptr, indices, data, N_GENES as u64).unwrap()])
        .unwrap();
    writer.finish(&cells, None).unwrap();
}

#[test]
fn reductions_of_wide_ranging_values_are_bit_identical_however_threads_finish() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let mut runs = runs();
    for seed in 2..5 {
        runs.push(RunOptions {
            delay_seed: seed,
            ..RunOptions::threads(all_threads().max(4))
        });
        runs.push(RunOptions {
            max_block_cells: Some(1 + seed * 9),
            delay_seed: seed,
            ..RunOptions::threads(4)
        });
    }
    assert_bit_identical("wide-ranging store", &store, &runs);
}

/// A sequential sum over every cell (no leaves, no tree) differs in the last bits from
/// the tree's on the wide-ranging store: the test above can tell orders apart.
#[test]
fn the_wide_ranging_store_tells_summation_orders_apart() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let base = scan(&store);
    let plan = base
        .highly_variable_genes(Some("values"), HvgFlavor::CellRanger, Some(50), None)
        .unwrap();
    let output =
        Output::members(&plan, [Member::GeneColumn("highly_variable".to_owned())]).unwrap();
    let physical = physical_plan(
        std::slice::from_ref(&output),
        OptimizeOptions::ALL,
        PhysicalOptions::default(),
    );
    let mut stream = Pipeline::new(&physical, 0)
        .unwrap()
        .run(ExecuteOptions::new(NonZeroUsize::new(2).unwrap()))
        .unwrap();
    let Some((_, NodeStatistics::HighlyVariableGenes(hvg))) = stream.statistics().iter().next()
    else {
        panic!("expected HVG statistics")
    };
    let tree_sums = hvg.moments().sums().to_vec();
    let mut sequential = vec![0.0_f64; N_GENES];
    let scan_output = Output::members(&base, [Member::Layer("values".to_owned())]).unwrap();
    let scan_physical = physical_plan(
        std::slice::from_ref(&scan_output),
        OptimizeOptions::ALL,
        PhysicalOptions::default(),
    );
    stream = Pipeline::new(&scan_physical, 0)
        .unwrap()
        .run(ExecuteOptions::new(NonZeroUsize::new(2).unwrap()))
        .unwrap();
    for block in stream {
        let block = block.unwrap();
        let layer = &block.layers["values"];
        let CellValues::F32(values) = layer.data() else {
            panic!("a float32 layer")
        };
        for (&gene, &value) in layer.indices().iter().zip(values) {
            sequential[gene as usize] += f64::from(value);
        }
    }
    let differing = tree_sums
        .iter()
        .zip(&sequential)
        .filter(|(tree, sequential)| tree.to_bits() != sequential.to_bits())
        .count();
    assert!(
        differing > N_GENES / 10,
        "only {differing} of {N_GENES} sums depend on the order"
    );
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 12, failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn any_threads_window_and_block_size_give_the_same_bits(
        threads in 1_usize..=8,
        blocks_in_flight in 1_u64..=6,
        max_block_cells in proptest::option::of(1_u64..=80),
        delay_seed in 0_u64..4,
    ) {
        let work = tempfile::tempdir().unwrap();
        let store = work.path().join("atlas.kyto");
        write_store(&store);
        let base = scan(&store);
        for (case, output) in cases(&base, "values") {
            let expected = fingerprint(&store, &output, RunOptions::threads(1));
            let options = RunOptions { threads, blocks_in_flight, max_block_cells, delay_seed };
            prop_assert_eq!(fingerprint(&store, &output, options), expected, "{}", case);
        }
    }
}
