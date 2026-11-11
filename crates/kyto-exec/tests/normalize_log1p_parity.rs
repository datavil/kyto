//! Task 6.2: normalize and log.
//!
//! Done when: matches scanpy to 1e-6; `explain()` shows one fused pass (the snapshot
//! is `kyto-plan`'s `tests/normalize_log1p_explain.rs`). The synthetic and pbmc3k
//! fixtures (written as `.h5ad` by `tests/fixtures`, converted here; skipped without
//! anndata, exit code 77) run `scan -> normalize_total(into="normalized") ->
//! log1p(into="log1p")`, with `target_sum=1e4` and with the median (scanpy's
//! `target_sum=None`), as one fused pass, against the stored scanpy reference
//! `normalize_log1p` (`tests/fixtures/make_reference.py`): per-cell and per-gene sums
//! (`f64` sums of the `float32` values) to 1e-6 relative and the same `nnz` on both
//! fixtures; on synthetic (the full CSR is stored) every value: `normalize_total`
//! bit for bit (kyto divides in scanpy's `float32` steps), `log1p` to 1e-6 relative
//! (kyto's is correctly rounded, numpy's `float32` `log1p` may be 1 ulp off,
//! deviations.md). Also the same bits at 1 and all threads and at smaller blocks (R11).

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use kyto_core::{GeneIds, Tiling};
use kyto_exec::{ExecuteOptions, Pipeline};
use kyto_plan::{Member, OptimizeOptions, Output, PhysicalOptions, Plan, physical_plan};
use kyto_store::{CellValues, ConvertOptions, TableAxis, convert, open_atlas_schema, read_ids};

/// Writes `load_fixture(name)` as `.h5ad`, the reference's target and its `log1p`
/// arrays, and scanpy's `normalize_total` computed here (the stored `normalized`
/// arrays are the `log1p` ones: `make_reference.py` keeps `adata.X`, which
/// `sc.pp.log1p` then changes in place; plan 6.2 Open), as raw little-endian files
/// (`f64` sums and target, `i64` nnz and CSR structure, `f32` CSR values), or exits 77
/// without scanpy.
const WRITE_FIXTURE: &str = "\
import sys
try:
    import scanpy as sc
    from fixtures.interop import to_anndata
except ImportError:
    sys.exit(77)
import json
import warnings
import numpy as np
from scipy.sparse import csr_matrix
from fixtures.reference import load_reference
from fixtures.registry import load_fixture
name, h5ad, out = sys.argv[1:4]
adata = to_anndata(load_fixture(name))
adata.write_h5ad(h5ad)
manifest = json.load(open('fixtures/reference/manifest.json'))
target = manifest['references'][name + '/normalize_log1p']['parameters']['target_sum']
np.asarray([target], dtype='<f8').tofile(f'{out}/target_sum')
def write(key, array):
    kind = array.dtype.kind
    dtype = '<f4' if array.dtype == np.float32 else ('<f8' if kind == 'f' else '<i8')
    np.asarray(array, dtype=dtype).tofile(f\"{out}/{key.replace('/', '.')}\")
for key, array in load_reference(name, 'normalize_log1p').items():
    if '/log1p/' in key:
        write(key, array)
for label, target_sum in (('target', target), ('median', None)):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        normalized = sc.pp.normalize_total(adata, target_sum=target_sum, inplace=False)['X']
    csr = csr_matrix(normalized)
    csr.sort_indices()
    prefix = label + '/normalized/'
    write(prefix + 'data', csr.data)
    write(prefix + 'indices', csr.indices.astype(np.int64))
    write(prefix + 'indptr', csr.indptr.astype(np.int64))
    wide = csr_matrix(csr, dtype=np.float64)
    write(prefix + 'cell_sum', np.asarray(wide.sum(axis=1)).ravel())
    write(prefix + 'gene_sum', np.asarray(wide.sum(axis=0)).ravel())
    write(prefix + 'nnz', np.asarray([csr.nnz], dtype=np.int64))
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

fn read_bytes<const N: usize>(path: &Path) -> Vec<[u8; N]> {
    std::fs::read(path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .as_chunks::<N>()
        .0
        .to_vec()
}

fn read_f64(path: &Path) -> Vec<f64> {
    read_bytes::<8>(path)
        .into_iter()
        .map(f64::from_le_bytes)
        .collect()
}

fn read_i64(path: &Path) -> Vec<i64> {
    read_bytes::<8>(path)
        .into_iter()
        .map(i64::from_le_bytes)
        .collect()
}

fn read_f32(path: &Path) -> Vec<f32> {
    read_bytes::<4>(path)
        .into_iter()
        .map(f32::from_le_bytes)
        .collect()
}

fn scan(store: &Path) -> Arc<Plan> {
    let schema = open_atlas_schema(store).unwrap();
    let gene_ids = GeneIds::new(read_ids(store, TableAxis::Genes, schema.n_genes).unwrap());
    Plan::scan(store.display().to_string(), schema, gene_ids)
}

/// One layer of the output as CSR: row pointer, gene indices, values.
#[derive(Debug, PartialEq)]
struct Csr {
    indptr: Vec<u64>,
    indices: Vec<u32>,
    values: Vec<f32>,
}

impl Csr {
    fn cell_sums(&self) -> Vec<f64> {
        self.indptr
            .windows(2)
            .map(|bounds| {
                self.values[bounds[0] as usize..bounds[1] as usize]
                    .iter()
                    .map(|&value| f64::from(value))
                    .sum()
            })
            .collect()
    }

    fn gene_sums(&self, n_genes: usize) -> Vec<f64> {
        let mut sums = vec![0.0; n_genes];
        for (&gene, &value) in self.indices.iter().zip(&self.values) {
            sums[gene as usize] += f64::from(value);
        }
        sums
    }

    fn bits(&self) -> Vec<u32> {
        self.values.iter().map(|value| value.to_bits()).collect()
    }
}

/// The `normalized` and `log1p` layers of `scan -> normalize_total(target_sum) ->
/// log1p`, run as one fused pass on `threads` threads in blocks of at most
/// `max_block_cells` cells.
fn run(
    plan: &Arc<Plan>,
    target_sum: Option<f64>,
    threads: usize,
    max_block_cells: Option<u64>,
) -> [Csr; 2] {
    let plan = plan
        .normalize_total(target_sum, Some("counts"), "normalized")
        .unwrap()
        .log1p(Some("normalized"), "log1p")
        .unwrap();
    let layers = ["normalized", "log1p"];
    let output =
        Output::members(&plan, layers.map(|layer| Member::Layer(layer.to_owned()))).unwrap();
    let options = PhysicalOptions {
        max_block_cells: max_block_cells.and_then(NonZeroU64::new),
        ..PhysicalOptions::default()
    };
    let physical = physical_plan(std::slice::from_ref(&output), OptimizeOptions::ALL, options);
    // One fused node over the scan: statistics passes only for the median.
    let fused = physical.optimized().nodes().len();
    assert_eq!(fused, 2, "scan and one fused pass:\n{physical}");
    let expected_passes = if target_sum.is_some() { 1 } else { 2 };
    assert_eq!(physical.estimate().passes(), expected_passes, "{physical}");
    let stream = Pipeline::new(&physical, 0)
        .unwrap()
        .run(ExecuteOptions::new(NonZeroUsize::new(threads).unwrap()))
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
    csr
}

/// `|kyto - scanpy| <= 1e-6 * |scanpy|` for every element; returns the worst
/// relative error.
fn assert_close(what: &str, kyto: &[f64], scanpy: &[f64]) -> f64 {
    assert_eq!(kyto.len(), scanpy.len(), "{what}: length");
    let mut worst = 0.0_f64;
    for (index, (&kyto, &scanpy)) in kyto.iter().zip(scanpy).enumerate() {
        let error = (kyto - scanpy).abs();
        assert!(
            error <= 1e-6 * scanpy.abs(),
            "{what} [{index}]: kyto {kyto} vs scanpy {scanpy}"
        );
        if scanpy != 0.0 {
            worst = worst.max(error / scanpy.abs());
        }
    }
    worst
}

#[test]
fn normalize_and_log1p_match_scanpy_on_fixtures() {
    let work = tempfile::tempdir().unwrap();
    let all = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    for name in ["synthetic", "pbmc3k"] {
        let h5ad = work.path().join(format!("{name}.h5ad"));
        let reference = work.path().join(format!("{name}-reference"));
        std::fs::create_dir(&reference).unwrap();
        if !write_fixture(name, &h5ad, &reference) {
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
        assert_eq!(options.layer, "counts");
        convert(&h5ad, &store, &options).unwrap();
        let plan = scan(&store);
        let n_genes = usize::try_from(plan.output_schema().n_genes().unwrap()).unwrap();
        let target = read_f64(&reference.join("target_sum"))[0];
        for (label, target_sum) in [("target", Some(target)), ("median", None)] {
            let kyto = run(&plan, target_sum, 1, None);
            for (step, csr) in ["normalized", "log1p"].iter().zip(&kyto) {
                let file = |key: &str| reference.join(format!("{label}.{step}.{key}"));
                let what = format!("{name} {label} {step}");
                let nnz = read_i64(&file("nnz"))[0];
                assert_eq!(csr.values.len() as i64, nnz, "{what}: nnz");
                let cells = assert_close(
                    &format!("{what} cell sums"),
                    &csr.cell_sums(),
                    &read_f64(&file("cell_sum")),
                );
                let genes = assert_close(
                    &format!("{what} gene sums"),
                    &csr.gene_sums(n_genes),
                    &read_f64(&file("gene_sum")),
                );
                let mut report = format!(
                    "{what}: worst relative error {cells:e} (cell sums), {genes:e} (gene sums)"
                );
                // Every value where the full CSR is there.
                if file("data").exists() {
                    let indptr = read_i64(&file("indptr"));
                    let indices = read_i64(&file("indices"));
                    let data = read_f32(&file("data"));
                    assert!(
                        csr.indptr.iter().map(|&end| end as i64).eq(indptr),
                        "{what}: indptr"
                    );
                    assert!(
                        csr.indices.iter().map(|&gene| i64::from(gene)).eq(indices),
                        "{what}: indices"
                    );
                    let scanpy: Vec<f64> = data.iter().map(|&value| f64::from(value)).collect();
                    let kyto_values: Vec<f64> =
                        csr.values.iter().map(|&value| f64::from(value)).collect();
                    let values = assert_close(&format!("{what} values"), &kyto_values, &scanpy);
                    let differ = csr
                        .values
                        .iter()
                        .zip(&data)
                        .filter(|(kyto, scanpy)| kyto.to_bits() != scanpy.to_bits())
                        .count();
                    if *step == "normalized" {
                        assert_eq!(differ, 0, "{what}: scanpy's float32 steps, bit for bit");
                    }
                    report += &format!(
                        ", {values:e} (values), {differ} of {nnz} values not bit-identical"
                    );
                }
                eprintln!("{report}");
            }
            // Same bits on any number of threads and block size (R11).
            for (threads, max_block_cells) in [(all, None), (all, Some(37)), (2, Some(100))] {
                let again = run(&plan, target_sum, threads, max_block_cells);
                for (step, (again, kyto)) in
                    ["normalized", "log1p"].iter().zip(again.iter().zip(&kyto))
                {
                    assert_eq!(again.indptr, kyto.indptr);
                    assert!(
                        again.bits() == kyto.bits(),
                        "{name} {label} {step} at {threads} threads, blocks {max_block_cells:?}"
                    );
                }
            }
        }
    }
}
