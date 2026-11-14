//! Task 6.13: the per-gene QC metrics and the `log1p_` QC columns match scanpy.
//!
//! On the 1.5 fixtures (synthetic and pbmc3k, written as `.h5ad` by `tests/fixtures`,
//! converted here; skipped without scanpy, exit code 77), kyto runs `scan ->
//! filter_cells(genes_detected >= g) -> with_genes(n_cells_by_counts, total_counts,
//! mean_counts, pct_dropout_by_counts, log1p_mean_counts, log1p_total_counts) ->
//! with_cells(total_counts, n_genes_by_counts and their log1p) ->
//! filter_genes(cells_detected >= 3)`, and scanpy runs `filter_cells(min_genes=g)`,
//! `calculate_qc_metrics(log1p=True)` and `filter_genes(min_cells=3)`. The cell filter
//! comes first so the per-gene metrics are over the cells present, not the store's.
//! Detected cells are equal, the kept genes are the same in the same order, and every
//! other column matches to 1e-6 relative (6.1's tolerance: scanpy computes in
//! `float32`, kyto in `float64`, deviations.md "QC metric dtypes"). The same bits at 1
//! and all threads and at blocks of 37 cells (R11).

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use kyto_core::{Axis, ColumnDtype, Tiling};
use kyto_core::{BinaryOp, CellExpr, Expr, GeneExpr, GeneIds, GeneQcMetric, Literal, QcMetric};
use kyto_exec::{ExecuteOptions, Pipeline};
use kyto_plan::{Member, OptimizeOptions, Output, PhysicalOptions, Plan, physical_plan};
use kyto_store::{ConvertOptions, TableAxis, convert, open_atlas_schema, read_ids};
use polars::prelude::{DataFrame, DataType};

/// Writes `load_fixture(name)` as `.h5ad`, then scanpy's cell filter, QC metrics
/// (`log1p=True`) and gene filter of it: the per-gene and per-cell columns (`f64`) and
/// the gene IDs `filter_genes(min_cells=3)` keeps (one per line), or exits 77 without
/// scanpy.
const WRITE_FIXTURE: &str = "\
import sys
try:
    import scanpy as sc
    from fixtures.interop import to_anndata
except ImportError:
    sys.exit(77)
import warnings
import numpy as np
from fixtures.registry import load_fixture
name, h5ad, out, min_genes = sys.argv[1:5]
adata = to_anndata(load_fixture(name))
adata.write_h5ad(h5ad)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    sc.pp.filter_cells(adata, min_genes=int(min_genes))
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None, log1p=True)
    keep, _ = sc.pp.filter_genes(adata, min_cells=3, inplace=False)
open(f'{out}/cells', 'w').write('\\n'.join(adata.obs_names))
open(f'{out}/genes', 'w').write('\\n'.join(adata.var_names[keep]))
for key in ('n_cells_by_counts', 'total_counts', 'mean_counts', 'pct_dropout_by_counts', 'log1p_mean_counts', 'log1p_total_counts'):
    np.asarray(adata.var[key], dtype='<f8').tofile(f'{out}/gene-{key}')
for key in ('total_counts', 'n_genes_by_counts', 'log1p_total_counts', 'log1p_n_genes_by_counts'):
    np.asarray(adata.obs[key], dtype='<f8').tofile(f'{out}/cell-{key}')
";

/// The cell filter per fixture: the fewest genes detected a kept cell has.
const FIXTURES: [(&str, u32); 2] = [("synthetic", 40), ("pbmc3k", 500)];

const GENE_COLUMNS: [&str; 6] = [
    "n_cells_by_counts",
    "total_counts",
    "mean_counts",
    "pct_dropout_by_counts",
    "log1p_mean_counts",
    "log1p_total_counts",
];

const CELL_COLUMNS: [&str; 4] = [
    "total_counts",
    "n_genes_by_counts",
    "log1p_total_counts",
    "log1p_n_genes_by_counts",
];

fn write_fixture(name: &str, h5ad: &Path, out: &Path, min_genes: u32) -> bool {
    let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests");
    let status = Command::new(std::env::var("PYTHON").unwrap_or_else(|_| "python".to_owned()))
        .current_dir(tests)
        .args(["-c", WRITE_FIXTURE, name])
        .arg(h5ad)
        .arg(out)
        .arg(min_genes.to_string())
        .status()
        .expect("run python (use `uv run cargo test`, or set PYTHON)");
    if status.code() == Some(77) {
        return false;
    }
    assert!(status.success(), "writing fixture {name} failed: {status}");
    true
}

fn read_floats(path: &Path) -> Vec<f64> {
    std::fs::read(path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .as_chunks::<8>()
        .0
        .iter()
        .map(|bytes| f64::from_le_bytes(*bytes))
        .collect()
}

fn read_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

fn scan(store: &Path) -> Arc<Plan> {
    let schema = open_atlas_schema(store).unwrap();
    let gene_ids = GeneIds::new(read_ids(store, TableAxis::Genes, schema.n_genes).unwrap());
    Plan::scan(store.display().to_string(), schema, gene_ids)
}

/// The plan up to the QC columns (every gene), and the same with scanpy's
/// `filter_genes(min_cells=3)` as a plan filter on top.
fn plans(store: &Path, layer: &str, min_genes: u32) -> (Arc<Plan>, Arc<Plan>) {
    let cell = |metric| CellExpr::qc(Some(layer.to_owned()), metric).unwrap();
    let gene = |metric| GeneExpr::qc(Some(layer.to_owned()), metric);
    let keep_cells = Expr::binary(
        cell(QcMetric::GenesDetected),
        BinaryOp::GtEq,
        Literal::Int(i64::from(min_genes)),
    )
    .unwrap();
    let columns = scan(store)
        .filter_cells(&keep_cells)
        .unwrap()
        .with_genes([
            (
                "n_cells_by_counts".to_owned(),
                gene(GeneQcMetric::CellsDetected),
            ),
            ("total_counts".to_owned(), gene(GeneQcMetric::TotalCounts)),
            ("mean_counts".to_owned(), gene(GeneQcMetric::MeanCounts)),
            (
                "pct_dropout_by_counts".to_owned(),
                gene(GeneQcMetric::PercentDropout),
            ),
            (
                "log1p_mean_counts".to_owned(),
                gene(GeneQcMetric::MeanCounts).log1p().unwrap(),
            ),
            (
                "log1p_total_counts".to_owned(),
                gene(GeneQcMetric::TotalCounts).log1p().unwrap(),
            ),
        ])
        .unwrap()
        .with_cells([
            ("total_counts".to_owned(), cell(QcMetric::TotalCounts)),
            (
                "n_genes_by_counts".to_owned(),
                cell(QcMetric::GenesDetected),
            ),
            (
                "log1p_total_counts".to_owned(),
                cell(QcMetric::TotalCounts).log1p().unwrap(),
            ),
            (
                "log1p_n_genes_by_counts".to_owned(),
                cell(QcMetric::GenesDetected).log1p().unwrap(),
            ),
        ])
        .unwrap();
    let keep_genes = Expr::binary(
        gene(GeneQcMetric::CellsDetected),
        BinaryOp::GtEq,
        Literal::Int(3),
    )
    .unwrap();
    let filtered = columns.filter_genes(&keep_genes).unwrap();
    (columns, filtered)
}

/// What kyto gives: cell IDs, gene IDs, and the bits of the gene and cell columns.
#[derive(Debug, PartialEq)]
struct Result {
    cells: Vec<String>,
    genes: Vec<String>,
    gene_columns: Vec<Vec<u64>>,
    cell_columns: Vec<Vec<u64>>,
}

fn bits(frame: &DataFrame, name: &str) -> Vec<u64> {
    let column = frame
        .column(name)
        .unwrap()
        .cast(&DataType::Float64)
        .unwrap();
    column
        .f64()
        .unwrap()
        .iter()
        .map(|value| value.unwrap().to_bits())
        .collect()
}

fn run(plan: &Arc<Plan>, store: &Path, threads: usize, max_block_cells: Option<u64>) -> Result {
    let members = GENE_COLUMNS
        .map(|name| Member::GeneColumn(name.to_owned()))
        .into_iter()
        .chain(CELL_COLUMNS.map(|name| Member::CellColumn(name.to_owned())));
    let output = Output::members(plan, members).unwrap();
    let options = PhysicalOptions {
        max_block_cells: max_block_cells.and_then(NonZeroU64::new),
        ..PhysicalOptions::default()
    };
    let physical = physical_plan(std::slice::from_ref(&output), OptimizeOptions::ALL, options);
    let mut stream = Pipeline::new(&physical, 0)
        .unwrap()
        .run(ExecuteOptions::new(NonZeroUsize::new(threads).unwrap()))
        .unwrap();
    let n_cells = open_atlas_schema(store).unwrap().n_cells;
    let store_cells = read_ids(store, TableAxis::Cells, n_cells).unwrap();
    let mut cells = Vec::new();
    let mut cell_columns = vec![Vec::new(); CELL_COLUMNS.len()];
    for block in stream.by_ref() {
        let block = block.unwrap();
        cells.extend(
            block
                .cells
                .iter()
                .map(|cell| store_cells[*cell as usize].clone()),
        );
        for (values, name) in cell_columns.iter_mut().zip(CELL_COLUMNS) {
            values.extend(bits(&block.columns, name));
        }
    }
    let genes = stream.genes().expect("gene columns were computed");
    let ids = genes.table().column("gene_id").unwrap().str().unwrap();
    Result {
        cells,
        genes: ids.iter().map(|id| id.unwrap().to_owned()).collect(),
        gene_columns: GENE_COLUMNS.map(|name| bits(genes.table(), name)).to_vec(),
        cell_columns,
    }
}

/// The worst relative difference of `kyto` from `scanpy` (both NaN is equal).
fn worst_difference(what: &str, kyto: &[u64], scanpy: &[f64]) -> f64 {
    assert_eq!(kyto.len(), scanpy.len(), "{what}: length");
    let mut worst = 0.0_f64;
    for (row, (kyto, scanpy)) in kyto.iter().zip(scanpy).enumerate() {
        let kyto = f64::from_bits(*kyto);
        if scanpy.is_nan() || kyto.is_nan() {
            assert!(
                kyto.is_nan() && scanpy.is_nan(),
                "{what} row {row}: {kyto} vs {scanpy}"
            );
            continue;
        }
        let difference = (kyto - scanpy).abs();
        assert!(
            difference <= 1e-6 * scanpy.abs(),
            "{what} row {row}: kyto {kyto} vs scanpy {scanpy}"
        );
        if difference > 0.0 {
            worst = worst.max(difference / scanpy.abs());
        }
    }
    worst
}

#[test]
fn gene_qc_and_log1p_columns_match_scanpy_on_fixtures() {
    let work = tempfile::tempdir().unwrap();
    let all = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    for (name, min_genes) in FIXTURES {
        let h5ad = work.path().join(format!("{name}.h5ad"));
        let reference = work.path().join(format!("{name}-reference"));
        std::fs::create_dir(&reference).unwrap();
        if !write_fixture(name, &h5ad, &reference, min_genes) {
            eprintln!("skipped: scanpy is not installed on this platform");
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
        let (columns, filtered) = plans(&store, &options.layer, min_genes);
        let schema = columns.output_schema();
        let dtype = |name: &str| schema.column(Axis::Genes, name).unwrap().dtype;
        assert_eq!(dtype("n_cells_by_counts"), ColumnDtype::UInt32);
        for name in &GENE_COLUMNS[1..] {
            assert_eq!(dtype(name), ColumnDtype::Float64, "{name}");
        }

        let kyto = run(&columns, &store, 1, None);
        let n_cells = open_atlas_schema(&store).unwrap().n_cells as usize;
        let scanpy_cells = read_lines(&reference.join("cells"));
        assert!(
            !scanpy_cells.is_empty() && scanpy_cells.len() < n_cells,
            "{name}: the cell filter keeps some cells, not all"
        );
        assert_eq!(kyto.cells, scanpy_cells, "{name}: kept cells");
        let mut report = Vec::new();
        for (values, column) in kyto.gene_columns.iter().zip(GENE_COLUMNS) {
            let scanpy = read_floats(&reference.join(format!("gene-{column}")));
            let worst = worst_difference(&format!("{name} gene {column}"), values, &scanpy);
            report.push(format!("{column} {worst:.1e}"));
        }
        for (values, column) in kyto.cell_columns.iter().zip(CELL_COLUMNS) {
            let scanpy = read_floats(&reference.join(format!("cell-{column}")));
            let worst = worst_difference(&format!("{name} cell {column}"), values, &scanpy);
            report.push(format!("cell {column} {worst:.1e}"));
        }
        // Detected cells are counts: equal, not close.
        let scanpy_detected = read_floats(&reference.join("gene-n_cells_by_counts"));
        let detected: Vec<f64> = kyto.gene_columns[0]
            .iter()
            .map(|bits| f64::from_bits(*bits))
            .collect();
        assert_eq!(detected, scanpy_detected, "{name}: n_cells_by_counts");

        // `filter(cells_detected() >= 3)` is `filter_genes(min_cells=3)`.
        let kept = run(&filtered, &store, 1, None);
        let scanpy_genes = read_lines(&reference.join("genes"));
        assert!(
            !scanpy_genes.is_empty() && scanpy_genes.len() < kyto.genes.len(),
            "{name}: the gene filter keeps some genes, not all"
        );
        assert_eq!(kept.genes, scanpy_genes, "{name}: kept genes");
        assert_eq!(
            kept.cells, kyto.cells,
            "{name}: the gene filter keeps cells"
        );
        eprintln!(
            "{name}: {} of {n_cells} cells, {} of {} genes kept; worst relative differences: {}",
            kyto.cells.len(),
            kept.genes.len(),
            kyto.genes.len(),
            report.join(", ")
        );
        // The same bits on any number of threads and block size (R11).
        for (threads, max_block_cells) in [(all, None), (all, Some(37)), (2, Some(100))] {
            assert_eq!(
                run(&columns, &store, threads, max_block_cells),
                kyto,
                "{name} at {threads} threads, blocks {max_block_cells:?}"
            );
            assert_eq!(
                run(&filtered, &store, threads, max_block_cells),
                kept,
                "{name} filtered at {threads} threads, blocks {max_block_cells:?}"
            );
        }
    }
}
