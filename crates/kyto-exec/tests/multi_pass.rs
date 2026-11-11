//! Task 5.4: multi-pass operators.
//!
//! Done when: HVG on fixtures needs exactly the passes `explain()` reports. Each
//! fixture (synthetic, pbmc3k; written as `.h5ad` by `tests/fixtures`, converted at
//! 1024 x 4096 tiles) runs `highly_variable_genes` of every flavour through the
//! executor over a storage wrapper that records every tile read (`indptr`, `indices`
//! and `data` chunk keys and byte ranges, memory/counting_tile_reads.md). One pass is
//! what a plain scan of the layer reads; the HVG run must read exactly `P` times every
//! range of that pass, with `P` the statistics passes `explain()` reports for the HVG
//! node (1 for `seurat`/`cell_ranger`, 2 for `seurat_v3`), which is also
//! `estimate().passes()`. With a median `normalize_total` below HVG, one pass more;
//! with an output reading a layer, a final pass more. The statistics equal the same
//! sums by hand over the stream, on 1 and 4 threads. Needs Python with anndata
//! (`uv run cargo test`); skips where it has no wheels (exit code 77).
//!
//! On a small store written here: a median `normalize_total` (alone, fused with
//! `log1p`, and without fusion) equals `normalize_total(median)` by hand, bit for bit,
//! in two passes over the store.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};

use kyto_core::{GeneIds, Tiling};
use kyto_exec::{
    CellValues, ExecuteOptions, GeneMoments, NodeStatistics, Pipeline, PipelineBlock, Statistics,
};
use kyto_plan::{
    Explain, HvgFlavor, Member, OptimizeOptions, Output, PhysicalOptions, PhysicalPlan, Plan,
    physical_plan,
};
use kyto_store::{
    CellBlock, ConvertOptions, LayerSpec, LayerWriteOptions, StoreWriter, TableAxis, ValueDtype,
    convert, open_atlas_schema, read_ids,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::{MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey};

/// Every tile read: (chunk key, byte range) -> times read.
type Reads = BTreeMap<(String, String), u64>;

/// Records the reads of tile chunks (`layers/<layer>/{indptr,indices,data}/c/...`).
struct RecordingStore<S> {
    inner: S,
    reads: Mutex<Reads>,
}

impl<S> RecordingStore<S> {
    fn new(inner: S) -> Arc<Self> {
        Arc::new(Self {
            inner,
            reads: Mutex::new(Reads::new()),
        })
    }

    fn take(&self) -> Reads {
        std::mem::take(&mut *self.reads.lock().unwrap())
    }
}

impl<S: ReadableStorageTraits> ReadableStorageTraits for RecordingStore<S> {
    // `get` and `get_partial` default to `get_partial_many`, so every read passes here.
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let ranges: Vec<ByteRange> = byte_ranges.collect();
        let name = key.as_str();
        let tile = name.starts_with("layers/")
            && ["/indptr/c/", "/indices/c/", "/data/c/"]
                .iter()
                .any(|array| name.contains(array));
        if tile {
            let mut reads = self.reads.lock().unwrap();
            for range in &ranges {
                *reads
                    .entry((name.to_owned(), format!("{range:?}")))
                    .or_default() += 1;
            }
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

/// How many passes `reads` makes, each reading exactly what `one_pass` reads.
fn passes(reads: &Reads, one_pass: &Reads) -> u64 {
    assert!(!one_pass.is_empty(), "a pass reads tiles");
    let (key, once) = one_pass.iter().next().unwrap();
    let passes = reads.get(key).copied().unwrap_or(0) / once;
    let expected: Reads = one_pass
        .iter()
        .map(|(key, count)| (key.clone(), count * passes))
        .filter(|(_, count)| *count > 0)
        .collect();
    assert_eq!(
        reads, &expected,
        "the run is not {passes} whole passes over the store"
    );
    passes
}

fn threads(threads: usize) -> ExecuteOptions {
    ExecuteOptions::new(NonZeroUsize::new(threads).unwrap())
}

fn options() -> PhysicalOptions {
    PhysicalOptions {
        blocks_in_flight: NonZeroU64::new(2).unwrap(),
        ..PhysicalOptions::default()
    }
}

/// Run output 0 of `physical` over `storage`; the blocks, the statistics and the tile
/// reads.
fn run(
    physical: &PhysicalPlan,
    storage: &Arc<RecordingStore<FilesystemStore>>,
    options: ExecuteOptions,
) -> (Vec<PipelineBlock>, Statistics, Reads) {
    storage.take();
    let stream = Pipeline::new(physical, 0)
        .unwrap()
        .run_on(Arc::clone(storage), options)
        .unwrap();
    let statistics = stream.statistics().clone();
    let blocks = stream.collect::<Result<Vec<_>, _>>().unwrap();
    (blocks, statistics, storage.take())
}

fn scan(store: &Path) -> Arc<Plan> {
    let schema = open_atlas_schema(store).unwrap();
    let gene_ids = GeneIds::new(read_ids(store, TableAxis::Genes, schema.n_genes).unwrap());
    Plan::scan(store.display().to_string(), schema, gene_ids)
}

fn layer_output(plan: &Arc<Plan>, layer: &str) -> PhysicalPlan {
    let output = Output::members(plan, [Member::Layer(layer.to_owned())]).unwrap();
    physical_plan(&[output], OptimizeOptions::ALL, options())
}

/// The statistics passes `explain()` reports for each node, by name.
fn explained_statistics_passes(plan: &Plan, physical: &PhysicalPlan) -> Vec<(&'static str, u32)> {
    Explain::new(plan, physical)
        .optimized()
        .iter()
        .map(|node| (node.name, node.statistics_passes))
        .collect()
}

/// Per-gene moments of `layer` over `blocks` by hand, each value through `transform`.
fn moments_by_hand(
    blocks: &[PipelineBlock],
    layer: &str,
    n_genes: usize,
    transform: impl Fn(usize, f64) -> f64,
) -> (u64, Vec<f64>, Vec<f64>) {
    let mut sums = vec![0.0; n_genes];
    let mut squares = vec![0.0; n_genes];
    let mut cells = 0;
    for block in blocks {
        let layer = &block.layers[layer];
        cells += block.cells.len() as u64;
        let values: Vec<f64> = match layer.data() {
            CellValues::F32(values) => values.iter().map(|&v| f64::from(v)).collect(),
            CellValues::U32(values) => values.iter().map(|&v| f64::from(v)).collect(),
        };
        for (&gene, value) in layer.indices().iter().zip(values) {
            let value = transform(gene as usize, value);
            sums[gene as usize] += value;
            squares[gene as usize] += value * value;
        }
    }
    (cells, sums, squares)
}

fn assert_moments(moments: &GeneMoments, by_hand: &(u64, Vec<f64>, Vec<f64>)) {
    assert_eq!(moments.n_cells(), by_hand.0);
    assert!(moments.sums() == by_hand.1.as_slice(), "sums differ");
    assert!(
        moments.sums_of_squares() == by_hand.2.as_slice(),
        "sums of squares differ"
    );
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
fn hvg_on_fixtures_needs_exactly_the_passes_explain_reports() {
    let work = tempfile::tempdir().unwrap();
    let mut report = Vec::new();
    for name in ["synthetic", "pbmc3k"] {
        let h5ad = work.path().join(format!("{name}.h5ad"));
        if !write_h5ad(name, &h5ad) {
            eprintln!("skipped: anndata is not installed on this platform");
            return;
        }
        let store = work.path().join(format!("{name}.kyto"));
        let convert_options = ConvertOptions {
            tiling: Tiling {
                cell_block_size: 1024,
                gene_block_size: 4096,
            },
            ..ConvertOptions::default()
        };
        convert(&h5ad, &store, &convert_options).unwrap();
        let storage = RecordingStore::new(FilesystemStore::new(&store).unwrap());
        let schema = open_atlas_schema(&store).unwrap();
        let layer = schema.layers[0].name.clone();
        let n_genes = usize::try_from(schema.n_genes).unwrap();
        let base = scan(&store);

        // One pass: the scan of the layer alone (task 5.2), as explain() says.
        let scan_physical = layer_output(&base, &layer);
        assert_eq!(
            explained_statistics_passes(&base, &scan_physical),
            [("SCAN", 0)]
        );
        let (counts, _, one_pass) = run(&scan_physical, &storage, threads(2));
        let log_normalized = base
            .normalize_total(Some(1e4), Some(&layer), "norm")
            .unwrap()
            .log1p(Some("norm"), "log1p")
            .unwrap();
        let (logged, _, logged_reads) = run(
            &layer_output(&log_normalized, "log1p"),
            &storage,
            threads(2),
        );
        assert_eq!(passes(&logged_reads, &one_pass), 1);

        let median = base
            .normalize_total(None, Some(&layer), "norm")
            .unwrap()
            .log1p(Some("norm"), "log1p")
            .unwrap();
        let cases = [
            ("seurat", &log_normalized, HvgFlavor::Seurat, "log1p"),
            (
                "cell_ranger",
                &log_normalized,
                HvgFlavor::CellRanger,
                "log1p",
            ),
            ("seurat_v3", &base, HvgFlavor::SeuratV3, layer.as_str()),
            ("seurat after a median", &median, HvgFlavor::Seurat, "log1p"),
        ];
        for (case, input, flavor, hvg_layer) in cases {
            let plan = input
                .highly_variable_genes(Some(hvg_layer), flavor, Some(100), None)
                .unwrap();
            // An output of HVG's gene columns: no final pass.
            let columns = Output::members(
                &plan,
                [
                    Member::GeneColumn("highly_variable".to_owned()),
                    Member::GeneColumn("means".to_owned()),
                ],
            )
            .unwrap();
            let physical = physical_plan(&[columns], OptimizeOptions::ALL, options());
            let explained = explained_statistics_passes(&plan, &physical);
            let hvg_passes = explained
                .iter()
                .find(|(node, _)| *node == "HIGHLY_VARIABLE_GENES")
                .unwrap()
                .1;
            let explained_total: u32 = explained.iter().map(|(_, passes)| passes).sum();
            let (blocks, statistics, reads) = run(&physical, &storage, threads(1));
            let measured = passes(&reads, &one_pass);
            report.push(format!(
                "{name} {case}: explain() HVG statistics passes {hvg_passes}, all nodes \
                 {explained_total}; estimate() {}; measured {measured}",
                physical.estimate().passes()
            ));
            let expected_hvg = if flavor == HvgFlavor::SeuratV3 { 2 } else { 1 };
            assert_eq!(hvg_passes, expected_hvg, "{case}");
            assert_eq!(measured, u64::from(explained_total), "{case}");
            assert_eq!(measured, u64::from(physical.estimate().passes()), "{case}");
            if case != "seurat after a median" {
                assert_eq!(measured, u64::from(hvg_passes), "{case}");
            }
            // The output's stream read nothing: cells only.
            assert!(blocks.iter().all(|block| block.layers.is_empty()));

            // The statistics are the sums by hand, on any thread count.
            let hvg_node = physical.optimized().outputs()[0].node();
            let Some(NodeStatistics::HighlyVariableGenes(hvg)) = statistics.get(hvg_node) else {
                panic!("{case}: no HVG statistics")
            };
            assert_eq!(hvg.flavor(), flavor);
            match flavor {
                HvgFlavor::Seurat if case == "seurat" => {
                    let by_hand =
                        moments_by_hand(&logged, "log1p", n_genes, |_, value| value.exp_m1());
                    assert_moments(hvg.moments(), &by_hand);
                }
                HvgFlavor::CellRanger => {
                    let by_hand = moments_by_hand(&logged, "log1p", n_genes, |_, value| value);
                    assert_moments(hvg.moments(), &by_hand);
                }
                HvgFlavor::SeuratV3 => {
                    let by_hand = moments_by_hand(&counts, &layer, n_genes, |_, value| value);
                    assert_moments(hvg.moments(), &by_hand);
                    let clipped = hvg.clipped().unwrap();
                    let clip = clipped.clip();
                    let by_hand = moments_by_hand(&counts, &layer, n_genes, |gene, value| {
                        value.min(clip[gene])
                    });
                    assert_moments(clipped.moments(), &by_hand);
                }
                HvgFlavor::Seurat => {
                    let Some(NodeStatistics::MedianTotal(target)) =
                        statistics.iter().find_map(|(_, s)| {
                            matches!(s, NodeStatistics::MedianTotal(_)).then_some(s)
                        })
                    else {
                        panic!("no median")
                    };
                    assert!(target.is_finite() && *target > 0.0);
                }
            }
            let (_, on_four, _) = run(&physical, &storage, threads(4));
            assert_eq!(on_four, statistics, "{case}: same statistics on 4 threads");

            // An output reading a layer adds the final pass.
            let all = physical_plan(&[Output::all(&plan)], OptimizeOptions::ALL, options());
            let (blocks, _, reads) = run(&all, &storage, threads(2));
            assert_eq!(
                passes(&reads, &one_pass),
                u64::from(explained_total) + 1,
                "{case}: final pass"
            );
            assert_eq!(
                passes(&reads, &one_pass),
                u64::from(all.estimate().passes())
            );
            assert!(blocks.iter().all(|block| block.layers.contains_key(&layer)));
        }
    }
    eprintln!("{}", report.join("\n"));
}

// A small store: median `normalize_total` by hand.

const N_CELLS: usize = 700;
const N_GENES: usize = 200;
const ZERO_CELL: usize = 3;
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

/// The counts, one sparse row per cell: (gene, count) with increasing genes.
fn counts() -> Vec<Vec<(u32, u32)>> {
    let mut random = Lcg(11);
    (0..N_CELLS)
        .map(|cell| {
            (0..N_GENES as u32)
                .filter_map(|gene| {
                    let draw = random.next();
                    (cell != ZERO_CELL && draw.is_multiple_of(7))
                        .then_some((gene, 1 + (draw / 7 % 30) as u32))
                })
                .collect()
        })
        .collect()
}

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

/// `np.median` of the positive cell totals, and `log1p(normalize_total(counts,
/// median))` by hand as value bits (each map's output rounded to f32).
fn median_by_hand() -> (f64, Vec<Vec<(u32, u32)>>) {
    let rows = counts();
    let totals: Vec<f64> = rows
        .iter()
        .map(|row| row.iter().map(|&(_, count)| f64::from(count)).sum())
        .collect();
    let mut positive: Vec<f64> = totals.iter().copied().filter(|t| *t > 0.0).collect();
    positive.sort_by(f64::total_cmp);
    let n = positive.len();
    let median = if n % 2 == 1 {
        positive[n / 2]
    } else {
        (positive[n / 2 - 1] + positive[n / 2]) / 2.0
    };
    let expected = rows
        .iter()
        .zip(&totals)
        .map(|(row, total)| {
            let factor = if *total == 0.0 { 1.0 } else { median / total };
            row.iter()
                .map(|&(gene, count)| {
                    let normalized = (f64::from(count) * factor) as f32;
                    (gene, kyto_exec::log1p(normalized).to_bits())
                })
                .collect()
        })
        .collect();
    (median, expected)
}

/// Every row of `layer` as (gene, value bits).
fn rows_of(blocks: &[PipelineBlock], layer: &str) -> Vec<Vec<(u32, u32)>> {
    let mut rows = Vec::new();
    for block in blocks {
        let layer = &block.layers[layer];
        let CellValues::F32(values) = layer.data() else {
            panic!("not float32")
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
    rows
}

#[test]
fn a_median_normalize_total_runs_a_statistics_pass_then_its_apply_pass() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let storage = RecordingStore::new(FilesystemStore::new(&store).unwrap());
    let base = scan(&store);
    let (_, _, one_pass) = run(&layer_output(&base, "counts"), &storage, threads(2));
    let (median, expected) = median_by_hand();
    let plan = base
        .normalize_total(None, Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "log1p")
        .unwrap();
    let output = Output::members(&plan, [Member::Layer("log1p".to_owned())]).unwrap();
    // Fused (a statistics pass, then the fused apply pass) and one map at a time.
    for rules in [OptimizeOptions::ALL, OptimizeOptions::NONE] {
        let physical = physical_plan(std::slice::from_ref(&output), rules, options());
        for n_threads in [1, 3] {
            let (blocks, statistics, reads) = run(&physical, &storage, threads(n_threads));
            assert!(rows_of(&blocks, "log1p") == expected, "{rules:?}");
            assert_eq!(
                statistics.iter().map(|(_, s)| s).collect::<Vec<_>>(),
                [&NodeStatistics::MedianTotal(median)]
            );
            assert_eq!(passes(&reads, &one_pass), 2);
            assert_eq!(physical.estimate().passes(), 2);
        }
    }
    // A median alone, as the output.
    let alone = base.normalize_total(None, Some("counts"), "norm").unwrap();
    let (blocks, _, reads) = run(&layer_output(&alone, "norm"), &storage, threads(2));
    assert_eq!(passes(&reads, &one_pass), 2);
    let norm_rows = rows_of(&blocks, "norm");
    let totals: f64 = norm_rows[0]
        .iter()
        .map(|&(_, bits)| f64::from(f32::from_bits(bits)))
        .sum();
    assert!(
        (totals - median).abs() < median * 1e-5,
        "{totals} vs {median}"
    );
}
