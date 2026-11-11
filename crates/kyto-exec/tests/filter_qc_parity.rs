//! Task 5.8: filter -> QC -> normalize/log1p in the executor matches scanpy.
//!
//! On the 1.5 fixtures (synthetic and pbmc3k, written as `.h5ad` by `tests/fixtures`,
//! converted here; skipped without scanpy, exit code 77), kyto runs `scan ->
//! with_cells(total_counts, n_genes_by_counts, pct_counts_mt) -> filter_cells(
//! n_genes_by_counts >= g & pct_counts_mt < m) -> filter_genes(~symbol starts with
//! 'MT-') -> normalize_total(1e4) -> log1p`, and scanpy runs
//! `calculate_qc_metrics(qc_vars=['mt'])`, the same subset, `normalize_total(1e4)`,
//! `log1p`. The kept cells and genes are the same, in the same order; the QC columns
//! match to 1e-6 relative (6.1's tolerance); the `log1p` layer has the same CSR
//! structure and every value matches to 1e-6 relative (kyto's `log1p` is correctly
//! rounded, numpy's `float32` one may be 1 ulp off, deviations.md). The same bits at 1
//! and all threads and at blocks of 37 cells (R11).

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use kyto_core::{
    BinaryOp, CellExpr, Expr, GeneExpr, GeneIds, Literal, QcMetric, StrFunction, Tiling,
};
use kyto_exec::{CellValues, ExecuteOptions, Pipeline};
use kyto_plan::{Member, OptimizeOptions, Output, PhysicalOptions, Plan, physical_plan};
use kyto_store::{ConvertOptions, TableAxis, convert, open_atlas_schema, read_ids};
use polars::prelude::DataType;

/// Writes `load_fixture(name)` as `.h5ad`, then scanpy's QC, subset, normalize and log
/// of it: the kept cell and gene IDs (one per line), the QC columns at the kept cells
/// (`f64`) and the `log1p` CSR (`f32` values, `i64` indices and indptr), or exits 77
/// without scanpy.
const WRITE_FIXTURE: &str = "\
import sys
try:
    import scanpy as sc
    from fixtures.interop import to_anndata
except ImportError:
    sys.exit(77)
import warnings
import numpy as np
from scipy.sparse import csr_matrix
from fixtures.registry import load_fixture
name, h5ad, out, min_genes, max_mt = sys.argv[1:6]
adata = to_anndata(load_fixture(name))
adata.write_h5ad(h5ad)
adata.var['mt'] = adata.var['symbol'].astype(str).str.startswith('MT-')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True, percent_top=None, log1p=False)
    keep = (adata.obs['n_genes_by_counts'] >= int(min_genes)) & (adata.obs['pct_counts_mt'] < float(max_mt))
    adata = adata[keep.to_numpy(), ~adata.var['mt'].to_numpy()].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
open(f'{out}/cells', 'w').write('\\n'.join(adata.obs_names))
open(f'{out}/genes', 'w').write('\\n'.join(adata.var_names))
for key in ('total_counts', 'n_genes_by_counts', 'pct_counts_mt'):
    np.asarray(adata.obs[key], dtype='<f8').tofile(f'{out}/{key}')
csr = csr_matrix(adata.X)
csr.sort_indices()
np.asarray(csr.data, dtype='<f4').tofile(f'{out}/data')
np.asarray(csr.indices, dtype='<i8').tofile(f'{out}/indices')
np.asarray(csr.indptr, dtype='<i8').tofile(f'{out}/indptr')
";

/// The QC thresholds per fixture: (min genes, max percent MT).
const FIXTURES: [(&str, u32, f64); 2] = [("synthetic", 40, 5.0), ("pbmc3k", 200, 5.0)];

const QC_COLUMNS: [&str; 3] = ["total_counts", "n_genes_by_counts", "pct_counts_mt"];

fn write_fixture(name: &str, h5ad: &Path, out: &Path, min_genes: u32, max_mt: f64) -> bool {
    let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests");
    let status = Command::new(std::env::var("PYTHON").unwrap_or_else(|_| "python".to_owned()))
        .current_dir(tests)
        .args(["-c", WRITE_FIXTURE, name])
        .arg(h5ad)
        .arg(out)
        .arg(min_genes.to_string())
        .arg(max_mt.to_string())
        .status()
        .expect("run python (use `uv run cargo test`, or set PYTHON)");
    if status.code() == Some(77) {
        return false;
    }
    assert!(status.success(), "writing fixture {name} failed: {status}");
    true
}

fn read_bytes<const N: usize>(path: &Path) -> Vec<[u8; N]> {
    std::fs::read(path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .as_chunks::<N>()
        .0
        .to_vec()
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

fn mitochondrial() -> GeneExpr {
    GeneExpr::column("symbol")
        .str(StrFunction::StartsWith("MT-".to_owned()))
        .unwrap()
}

/// The plan, reading `layer` (the stored counts).
fn plan(store: &Path, layer: &str, min_genes: u32, max_mt: f64) -> Arc<Plan> {
    let qc = |metric| CellExpr::qc(Some(layer.to_owned()), metric).unwrap();
    let keep = Expr::binary(
        Expr::binary(
            CellExpr::column("n_genes_by_counts"),
            BinaryOp::GtEq,
            Literal::Int(i64::from(min_genes)),
        )
        .unwrap(),
        BinaryOp::And,
        Expr::binary(
            CellExpr::column("pct_counts_mt"),
            BinaryOp::Lt,
            Literal::Float(max_mt),
        )
        .unwrap(),
    )
    .unwrap();
    scan(store)
        .with_cells([
            ("total_counts".to_owned(), qc(QcMetric::TotalCounts)),
            ("n_genes_by_counts".to_owned(), qc(QcMetric::GenesDetected)),
            (
                "pct_counts_mt".to_owned(),
                qc(QcMetric::PercentCountsIn {
                    genes: mitochondrial(),
                }),
            ),
        ])
        .unwrap()
        .filter_cells(&keep)
        .unwrap()
        .filter_genes(&mitochondrial().not().unwrap())
        .unwrap()
        .normalize_total(Some(1e4), Some(layer), "normalized")
        .unwrap()
        .log1p(Some("normalized"), "log1p")
        .unwrap()
}

/// What kyto gives: cell IDs, gene IDs, the QC columns and the `log1p` CSR.
#[derive(Debug, PartialEq)]
struct Result {
    cells: Vec<String>,
    genes: Vec<String>,
    qc: Vec<Vec<u64>>,
    indptr: Vec<u64>,
    indices: Vec<u32>,
    values: Vec<f32>,
}

fn run(plan: &Arc<Plan>, store: &Path, threads: usize, max_block_cells: Option<u64>) -> Result {
    let members = std::iter::once(Member::Layer("log1p".to_owned()))
        .chain(QC_COLUMNS.map(|name| Member::CellColumn(name.to_owned())));
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
    let mut result = Result {
        cells: Vec::new(),
        genes: Vec::new(),
        qc: vec![Vec::new(); QC_COLUMNS.len()],
        indptr: vec![0],
        indices: Vec::new(),
        values: Vec::new(),
    };
    for block in stream.by_ref() {
        let block = block.unwrap();
        result.cells.extend(
            block
                .cells
                .iter()
                .map(|cell| store_cells[*cell as usize].clone()),
        );
        for (values, name) in result.qc.iter_mut().zip(QC_COLUMNS) {
            let column = block.columns.column(name).unwrap();
            let column = column.cast(&DataType::Float64).unwrap();
            values.extend(
                column
                    .f64()
                    .unwrap()
                    .iter()
                    .map(|value| value.unwrap().to_bits()),
            );
        }
        let layer = &block.layers["log1p"];
        let offset = *result.indptr.last().unwrap();
        result
            .indptr
            .extend(layer.indptr()[1..].iter().map(|end| offset + end));
        result.indices.extend_from_slice(layer.indices());
        let CellValues::F32(values) = layer.data() else {
            panic!("log1p is float32")
        };
        result.values.extend_from_slice(values);
    }
    let genes = stream
        .genes()
        .expect("a gene filter leaves the genes present");
    let ids = genes
        .table()
        .column("gene_id")
        .unwrap()
        .str()
        .unwrap()
        .clone();
    result.genes = ids.iter().map(|id| id.unwrap().to_owned()).collect();
    result
}

fn assert_close(what: &str, kyto: f64, scanpy: f64) {
    if scanpy.is_nan() {
        assert!(kyto.is_nan(), "{what}: {kyto} vs NaN");
        return;
    }
    assert!(
        (kyto - scanpy).abs() <= 1e-6 * scanpy.abs(),
        "{what}: kyto {kyto} vs scanpy {scanpy}"
    );
}

#[test]
fn filter_qc_normalize_log1p_match_scanpy_on_fixtures() {
    let work = tempfile::tempdir().unwrap();
    let all = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    for (name, min_genes, max_mt) in FIXTURES {
        let h5ad = work.path().join(format!("{name}.h5ad"));
        let reference = work.path().join(format!("{name}-reference"));
        std::fs::create_dir(&reference).unwrap();
        if !write_fixture(name, &h5ad, &reference, min_genes, max_mt) {
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
        let plan = plan(&store, &options.layer, min_genes, max_mt);
        let kyto = run(&plan, &store, 1, None);

        let n_cells = open_atlas_schema(&store).unwrap().n_cells as usize;
        let scanpy_cells = read_lines(&reference.join("cells"));
        assert!(
            !scanpy_cells.is_empty() && scanpy_cells.len() < n_cells,
            "{name}: the filter keeps some cells, not all"
        );
        assert_eq!(kyto.cells, scanpy_cells, "{name}: kept cells");
        assert_eq!(
            kyto.genes,
            read_lines(&reference.join("genes")),
            "{name}: kept genes"
        );
        for (values, column) in kyto.qc.iter().zip(QC_COLUMNS) {
            let scanpy: Vec<f64> = read_bytes::<8>(&reference.join(column))
                .into_iter()
                .map(f64::from_le_bytes)
                .collect();
            assert_eq!(values.len(), scanpy.len(), "{name} {column}");
            for (cell, (kyto, scanpy)) in values.iter().zip(scanpy).enumerate() {
                assert_close(
                    &format!("{name} {column} cell {cell}"),
                    f64::from_bits(*kyto),
                    scanpy,
                );
            }
        }
        let indptr: Vec<u64> = read_bytes::<8>(&reference.join("indptr"))
            .into_iter()
            .map(|bytes| i64::from_le_bytes(bytes) as u64)
            .collect();
        let indices: Vec<u32> = read_bytes::<8>(&reference.join("indices"))
            .into_iter()
            .map(|bytes| i64::from_le_bytes(bytes) as u32)
            .collect();
        assert_eq!(kyto.indptr, indptr, "{name}: log1p rows");
        assert_eq!(kyto.indices, indices, "{name}: log1p genes");
        let data: Vec<f32> = read_bytes::<4>(&reference.join("data"))
            .into_iter()
            .map(f32::from_le_bytes)
            .collect();
        let mut worst = 0.0_f64;
        for (entry, (kyto, scanpy)) in kyto.values.iter().zip(&data).enumerate() {
            let (kyto, scanpy) = (f64::from(*kyto), f64::from(*scanpy));
            assert_close(&format!("{name} log1p entry {entry}"), kyto, scanpy);
            worst = worst.max((kyto - scanpy).abs() / scanpy.abs());
        }
        eprintln!(
            "{name}: {} of {n_cells} cells, {} genes, {} values, worst log1p relative error {worst:e}",
            kyto.cells.len(),
            kyto.genes.len(),
            kyto.values.len()
        );
        // The same bits on any number of threads and block size (R11).
        for (threads, max_block_cells) in [(all, None), (all, Some(37)), (2, Some(100))] {
            assert_eq!(
                run(&plan, &store, threads, max_block_cells),
                kyto,
                "{name} at {threads} threads, blocks {max_block_cells:?}"
            );
        }
    }
}
