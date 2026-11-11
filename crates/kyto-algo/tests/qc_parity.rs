//! Task 6.1: QC metrics.
//!
//! Done when: values match `scanpy.pp.calculate_qc_metrics` to 1e-6 relative. The
//! synthetic and pbmc3k fixtures (written as `.h5ad` by `tests/fixtures`, converted
//! here; skipped without anndata, exit code 77) are scanned, each QC expression is
//! resolved against the plan (`MT-` genes as `gene_column('symbol').str.starts_with('MT-')`,
//! the gene set of the stored scanpy reference, `tests/fixtures/make_reference.py`) and
//! computed on every streamed block of the counts layer. Each metric must match the
//! reference's `total_counts`, `n_genes_by_counts`, `total_counts_mt` and
//! `pct_counts_mt` to 1e-6 relative, and give the same bits at 1 and all threads and
//! at smaller blocks (R11).

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use kyto_algo::{CellQc, GeneSet, QcValues};
use kyto_core::{CellExpr, ExprNode, GeneExpr, GeneIds, QcMetric, Source, StrFunction, Tiling};
use kyto_exec::{ExecuteOptions, Kernels, Pipeline};
use kyto_plan::{Member, OptimizeOptions, Output, PhysicalOptions, Plan, physical_plan};
use kyto_store::{ConvertOptions, TableAxis, convert, open_atlas_schema, read_ids};

/// Writes `load_fixture(name)` as `.h5ad` and the scanpy QC reference's per-cell
/// arrays as little-endian `f64` files, or exits 77 without anndata.
const WRITE_FIXTURE: &str = "\
import sys
try:
    from fixtures.interop import to_anndata
except ImportError:
    sys.exit(77)
import numpy as np
from fixtures.reference import load_reference
from fixtures.registry import load_fixture
name, h5ad, out = sys.argv[1:4]
to_anndata(load_fixture(name)).write_h5ad(h5ad)
reference = load_reference(name, 'qc')
for key in ('total_counts', 'n_genes_by_counts', 'total_counts_mt', 'pct_counts_mt'):
    np.asarray(reference['cells/' + key], dtype='<f8').tofile(f'{out}/{key}.f64')
";

/// Write fixture `name` to `h5ad` and its reference to `out`; `false` without anndata.
fn write_fixture(name: &str, h5ad: &Path, out: &Path) -> bool {
    let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests");
    let status = Command::new(std::env::var("PYTHON").unwrap_or_else(|_| "python".to_owned()))
        .current_dir(tests)
        .args(["-c", WRITE_FIXTURE, name])
        .arg(h5ad)
        .arg(out)
        .status()
        .expect("run python (use `uv run cargo test`, or set PYTHON)");
    if status.code() == Some(77) {
        return false;
    }
    assert!(status.success(), "writing fixture {name} failed: {status}");
    true
}

fn read_f64(path: &Path) -> Vec<f64> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|bytes| f64::from_le_bytes(*bytes))
        .collect()
}

fn scan(store: &Path) -> Arc<Plan> {
    let schema = open_atlas_schema(store).unwrap();
    let gene_ids = GeneIds::new(read_ids(store, TableAxis::Genes, schema.n_genes).unwrap());
    Plan::scan(store.display().to_string(), schema, gene_ids)
}

fn mitochondrial() -> GeneExpr {
    GeneExpr::column("symbol")
        .str(StrFunction::StartsWith("MT-".to_owned()))
        .unwrap()
}

/// The metrics compared, with the reference array each must match.
fn metrics() -> Vec<(&'static str, QcMetric)> {
    vec![
        ("total_counts", QcMetric::TotalCounts),
        ("n_genes_by_counts", QcMetric::GenesDetected),
        (
            "total_counts_mt",
            QcMetric::TotalCountsIn {
                genes: mitochondrial(),
            },
        ),
        (
            "pct_counts_mt",
            QcMetric::PercentCountsIn {
                genes: mitochondrial(),
            },
        ),
    ]
}

/// The resolved metric of the QC expression `expr` (resolved against `plan`).
fn resolved_metric(plan: &Plan, expr: &CellExpr) -> (String, QcMetric) {
    let resolved = plan.resolve(expr).unwrap();
    let ExprNode::Source(Source::Qc {
        layer: Some(layer),
        metric,
    }) = resolved.node().as_ref()
    else {
        panic!("a resolved QC leaf names its layer: {resolved}")
    };
    (layer.clone(), metric.clone())
}

/// Every cell's value of `metric` over the stream of `layer`, as `f64`.
fn compute(
    plan: &Arc<Plan>,
    store: &Path,
    layer: &str,
    metric: &QcMetric,
    threads: usize,
    max_block_cells: Option<u64>,
    kernels: Kernels,
) -> Vec<f64> {
    let output = Output::members(plan, [Member::Layer(layer.to_owned())]).unwrap();
    let options = PhysicalOptions {
        max_block_cells: max_block_cells.and_then(NonZeroU64::new),
        ..PhysicalOptions::default()
    };
    let physical = physical_plan(std::slice::from_ref(&output), OptimizeOptions::ALL, options);
    let qc = CellQc::read(store, metric).unwrap().with_kernels(kernels);
    let stream = Pipeline::new(&physical, 0)
        .unwrap()
        .run(ExecuteOptions::new(NonZeroUsize::new(threads).unwrap()))
        .unwrap();
    let mut values = Vec::new();
    for block in stream {
        let block = block.unwrap();
        match qc.evaluate(&block.layers[layer]).unwrap() {
            QcValues::Float64(block_values) => values.extend(block_values),
            QcValues::UInt32(block_values) => {
                values.extend(block_values.into_iter().map(f64::from))
            }
        }
    }
    values
}

/// `NaN` matches `NaN`; otherwise `|kyto - scanpy| <= 1e-6 * |scanpy|`.
fn assert_close(fixture: &str, name: &str, kyto: &[f64], scanpy: &[f64]) {
    assert_eq!(kyto.len(), scanpy.len(), "{fixture} {name}: cell count");
    let mut worst = 0.0_f64;
    for (cell, (&kyto, &scanpy)) in kyto.iter().zip(scanpy).enumerate() {
        if scanpy.is_nan() {
            assert!(kyto.is_nan(), "{fixture} {name} cell {cell}: {kyto} vs NaN");
            continue;
        }
        let error = (kyto - scanpy).abs();
        assert!(
            error <= 1e-6 * scanpy.abs(),
            "{fixture} {name} cell {cell}: kyto {kyto} vs scanpy {scanpy}"
        );
        if scanpy != 0.0 {
            worst = worst.max(error / scanpy.abs());
        }
    }
    eprintln!(
        "{fixture} {name}: {} cells, worst relative error {worst:e}",
        kyto.len()
    );
}

#[test]
fn qc_metrics_match_scanpy_on_fixtures() {
    let work = tempfile::tempdir().unwrap();
    let all = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    for name in ["synthetic", "pbmc3k"] {
        let h5ad = work.path().join(format!("{name}.h5ad"));
        let reference = work.path().join(format!("{name}-reference"));
        std::fs::create_dir(&reference).unwrap();
        if !write_fixture(name, &h5ad, &reference) {
            eprintln!("skipped: anndata is not installed on this platform");
            return;
        }
        let store = work.path().join(format!("{name}.kyto"));
        let options = ConvertOptions {
            tiling: Tiling {
                cell_block_size: 256,
                gene_block_size: 4096,
            },
            ..ConvertOptions::default()
        };
        convert(&h5ad, &store, &options).unwrap();
        let plan = scan(&store);
        let mitochondrial_genes = GeneSet::read(&store, &plan.resolve(&mitochondrial()).unwrap())
            .unwrap()
            .len();
        assert!(mitochondrial_genes > 0, "{name}: the fixture has MT- genes");
        for (reference_name, metric) in metrics() {
            let expr = CellExpr::qc(Some(options.layer.clone()), metric).unwrap();
            let (layer, metric) = resolved_metric(&plan, &expr);
            let scanpy = read_f64(&reference.join(format!("{reference_name}.f64")));
            let kyto = compute(&plan, &store, &layer, &metric, 1, None, Kernels::detected());
            assert_close(name, reference_name, &kyto, &scanpy);
            // Same bits on any number of threads, block size and kernel ISA (R11).
            for (threads, max_block_cells, kernels) in [
                (all, None, Kernels::detected()),
                (all, Some(37), Kernels::detected()),
                (2, Some(100), Kernels::scalar()),
            ] {
                let again = compute(
                    &plan,
                    &store,
                    &layer,
                    &metric,
                    threads,
                    max_block_cells,
                    kernels,
                );
                let bits = |values: &[f64]| {
                    values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    bits(&again),
                    bits(&kyto),
                    "{name} {reference_name} at {threads} threads, blocks {max_block_cells:?}"
                );
            }
        }
    }
}

/// A QC column in `with_cells` gets its dtype and reads its layer and the gene-set
/// columns; the executor still refuses computed columns (5.2 Open (1)), naming 7.2.
#[test]
fn qc_columns_in_a_plan() {
    let work = tempfile::tempdir().unwrap();
    let h5ad = work.path().join("synthetic.h5ad");
    let reference = work.path().join("reference");
    std::fs::create_dir(&reference).unwrap();
    if !write_fixture("synthetic", &h5ad, &reference) {
        eprintln!("skipped: anndata is not installed on this platform");
        return;
    }
    let store = work.path().join("synthetic.kyto");
    let options = ConvertOptions::default();
    convert(&h5ad, &store, &options).unwrap();
    let layer = || Some(options.layer.clone());
    let plan = scan(&store)
        .with_cells([
            (
                "total_counts".to_owned(),
                CellExpr::qc(layer(), QcMetric::TotalCounts).unwrap(),
            ),
            (
                "n_genes_by_counts".to_owned(),
                CellExpr::qc(layer(), QcMetric::GenesDetected).unwrap(),
            ),
            (
                "pct_counts_mt".to_owned(),
                CellExpr::qc(
                    layer(),
                    QcMetric::PercentCountsIn {
                        genes: mitochondrial(),
                    },
                )
                .unwrap(),
            ),
        ])
        .unwrap();
    let schema = plan.output_schema();
    let dtype = |name: &str| schema.column(kyto_core::Axis::Cells, name).unwrap().dtype;
    assert_eq!(dtype("total_counts"), kyto_core::ColumnDtype::Float64);
    assert_eq!(dtype("n_genes_by_counts"), kyto_core::ColumnDtype::UInt32);
    assert_eq!(dtype("pct_counts_mt"), kyto_core::ColumnDtype::Float64);
    let output = Output::members(&plan, [Member::CellColumn("pct_counts_mt".to_owned())]).unwrap();
    let physical = physical_plan(
        std::slice::from_ref(&output),
        OptimizeOptions::ALL,
        PhysicalOptions::default(),
    );
    // The with_cells node reads the layer and the gene set's column.
    let reads = physical
        .optimized()
        .nodes()
        .iter()
        .flat_map(|node| node.reads())
        .collect::<Vec<_>>();
    assert!(
        reads.contains(&Member::Layer(options.layer.clone())),
        "{reads:?}"
    );
    assert!(
        reads.contains(&Member::GeneColumn("symbol".to_owned())),
        "{reads:?}"
    );
    let error = Pipeline::new(&physical, 0).unwrap_err();
    assert_eq!(error.code(), "KY-E601");
}

/// A gene set without a column (a literal from plan JSON) holds for every gene, or for
/// none; null flags are not in the set (bug-hunter, 6.1).
#[test]
fn gene_sets_from_literals_and_nulls() {
    use polars::prelude::{Column, DataFrame};
    let table = DataFrame::new(
        3,
        vec![
            Column::new("gene_id".into(), ["G1", "G2", "G3"]),
            Column::new("flag".into(), [Some(true), None, Some(false)]),
        ],
    )
    .unwrap();
    let literal = |value: bool| -> GeneExpr {
        let text = format!(r#"[{{"op": "boolean", "value": {value}}}]"#);
        serde_json::from_str::<kyto_core::ExprJson>(&text)
            .unwrap()
            .to_expr("test")
            .unwrap()
    };
    let set = |genes: &GeneExpr| GeneSet::evaluate(&table, genes, "genes.parquet").unwrap();
    assert_eq!(set(&literal(true)).mask(), [true, true, true]);
    assert_eq!(set(&literal(false)).mask(), [false, false, false]);
    assert_eq!(set(&GeneExpr::column("flag")).mask(), [true, false, false]);
}
