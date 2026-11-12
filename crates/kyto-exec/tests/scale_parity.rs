//! Task 6.4: scale.
//!
//! Done when: matches scanpy; `estimate()` reports the dense size (that half is
//! `crates/kyto-plan/tests/estimate.rs`). The synthetic and pbmc3k fixtures (written as
//! `.h5ad` by `tests/fixtures`, converted here; skipped without anndata, exit code 77)
//! run the stored scanpy reference `scale` (`tests/fixtures/make_reference.py`): genes
//! seen in at least `min_cells_per_gene` cells, `normalize_total(1e4) -> log1p`, the
//! `seurat` HVGs (`select_genes` with the reference's genes), then
//! `scale(zero_center=True, max_value=10)`; synthetic also `max_value=None` and
//! `zero_center=False`. scanpy runs live in the helper below (the reference keeps the
//! matrix on synthetic only; the helper checks the live one equals it), twice:
//! - on kyto's input (scanpy's steps with a correctly rounded `log1p`, as kyto's, task
//!   6.2): the gene means and standard deviations to 1e-12 relative and every scaled
//!   value within 1 `float32` ulp (measured: identical bits everywhere);
//! - as the reference (numpy's `float32` `log1p`, 1 ulp off on some values,
//!   deviations.md): statistics to 1e-7 relative, values to 1e-5 absolute (measured
//!   on pbmc3k: 4.6e-9, 5.9e-7; 292 of 5.4M values differ).
//!
//! Centred, scanpy's `float64` values are rounded to `float32` (kyto's layer dtype,
//! deviations.md); not centred, scanpy's values are `float32` already.
//! Also the same bits at 1 and all threads and at blocks of 37 cells (R11).

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use kyto_core::{GeneIds, Tiling};
use kyto_exec::{CellValues, ExecuteOptions, NodeStatistics, Pipeline, ScaleStatistics};
use kyto_plan::{Member, OptimizeOptions, Output, PhysicalOptions, Plan, physical_plan};
use kyto_store::{ConvertOptions, TableAxis, convert, open_atlas_schema, read_ids};

/// Writes `load_fixture(name)` as `.h5ad`; the filtered gene IDs, the HVG gene IDs
/// and the reference's `mean`/`std` (`<f8`); scanpy's scaled matrix for each variant
/// `(zero_center, max_value)` as dense `<f8` (`scaled.<zero_center>.<max_value>`);
/// prints `min_cells_per_gene target_sum max_value`. Exits 77 without anndata.
const WRITE_FIXTURE: &str = "\
import sys
try:
    from fixtures.interop import to_anndata
except ImportError:
    sys.exit(77)
import json
import numpy as np
import scanpy as sc
from fixtures.make_reference import _filtered_counts, _hvg_genes
from fixtures.reference import load_reference
from fixtures.registry import load_fixture
name, h5ad, out = sys.argv[1:4]
to_anndata(load_fixture(name)).write_h5ad(h5ad)
manifest = json.load(open('fixtures/reference/manifest.json'))['references']
parameters = manifest[name + '/scale']['parameters']
target = manifest[name + '/normalize_log1p']['parameters']['target_sum']
reference = load_reference(name, 'scale')
open(f'{out}/filtered', 'w').write('\\n'.join(_filtered_counts(name).var_names.tolist()))
open(f'{out}/hvg', 'w').write('\\n'.join(reference['genes/gene_id'].tolist()))
np.asarray(reference['genes/mean'], dtype='<f8').tofile(f'{out}/mean')
np.asarray(reference['genes/std'], dtype='<f8').tofile(f'{out}/std')
max_value = parameters['max_value']
variants = [(True, max_value)]
if name == 'synthetic':
    variants += [(True, None), (False, max_value), (False, None)]
hvg = _hvg_genes(name)
# kyto's input: numpy's float32 log1p is 1 ulp off on some values, kyto's is correctly
# rounded (deviations.md, task 6.2); scanpy's scale on that input too.
log1p = sc.pp.log1p
def rounded_log1p(adata, **kwargs):
    adata.X.data = np.log1p(adata.X.data.astype(np.float64)).astype(np.float32)
from fixtures import make_reference
sc.pp.log1p = rounded_log1p
try:
    exact = make_reference._log_normalized(name)
finally:
    sc.pp.log1p = log1p
exact = exact[:, reference['genes/gene_id']].copy()
assert list(exact.var_names) == list(hvg.var_names)
for zero_center, clip in variants:
    for suffix, data in [('', hvg), ('.exact', exact)]:
        adata = data.copy()
        sc.pp.scale(adata, zero_center=zero_center, max_value=clip)
        x = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
        if suffix == '' and zero_center and clip == max_value and 'scaled' in reference:
            assert np.array_equal(x, reference['scaled']), 'live scanpy differs from the reference'
        np.asarray(x, dtype='<f8').tofile(f'{out}/scaled.{zero_center}.{clip}{suffix}')
        if zero_center and clip == max_value:
            np.asarray(adata.var['mean'], dtype='<f8').tofile(f'{out}/mean{suffix}.live')
            np.asarray(adata.var['std'], dtype='<f8').tofile(f'{out}/std{suffix}.live')
print(parameters['min_cells_per_gene'], target, max_value)
";

/// Write fixture `name` to `h5ad` and its reference to `out`; its
/// `min_cells_per_gene`, `target_sum` and `max_value`, or `None` without anndata.
fn write_fixture(name: &str, h5ad: &Path, out: &Path) -> Option<(u64, f64, f64)> {
    let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests");
    let output = Command::new(std::env::var("PYTHON").unwrap_or_else(|_| "python".to_owned()))
        .current_dir(tests)
        .args(["-c", WRITE_FIXTURE, name])
        .arg(h5ad)
        .arg(out)
        .output()
        .expect("run python (use `uv run cargo test`, or set PYTHON)");
    if output.status.code() == Some(77) {
        return None;
    }
    assert!(
        output.status.success(),
        "writing fixture {name} failed: {output:?}"
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let mut words = text.split_whitespace();
    Some((
        words.next().unwrap().parse().unwrap(),
        words.next().unwrap().parse().unwrap(),
        words.next().unwrap().parse().unwrap(),
    ))
}

fn read_f64(path: &Path) -> Vec<f64> {
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

/// The scaled layer as dense rows (`n_cells x n_genes`, row-major, cell order) and
/// the node's statistics.
fn run(
    plan: &Arc<Plan>,
    n_cells: usize,
    threads: usize,
    block_cells: Option<u64>,
) -> (Vec<f32>, ScaleStatistics) {
    let output = Output::members(plan, [Member::Layer("scaled".to_owned())]).unwrap();
    let options = PhysicalOptions {
        max_block_cells: block_cells.and_then(NonZeroU64::new),
        ..PhysicalOptions::default()
    };
    let physical = physical_plan(std::slice::from_ref(&output), OptimizeOptions::ALL, options);
    let mut stream = Pipeline::new(&physical, 0)
        .unwrap()
        .run(ExecuteOptions::new(NonZeroUsize::new(threads).unwrap()))
        .unwrap();
    let statistics = stream
        .statistics()
        .iter()
        .find_map(|(_, statistics)| match statistics {
            NodeStatistics::Scale(scale) => Some((**scale).clone()),
            _ => None,
        })
        .expect("scale's statistics");
    let n_genes = statistics.means().len();
    let mut dense = vec![0.0_f32; n_cells * n_genes];
    let mut row = 0;
    for block in stream.by_ref() {
        let block = block.unwrap();
        let layer = &block.layers["scaled"];
        let CellValues::F32(values) = layer.data() else {
            panic!("scale writes float32")
        };
        assert_eq!(layer.n_genes() as usize, n_genes);
        for range in layer.indptr().windows(2) {
            let range = range[0] as usize..range[1] as usize;
            for (&gene, &value) in layer.indices()[range.clone()].iter().zip(&values[range]) {
                dense[row * n_genes + gene as usize] = value;
            }
            row += 1;
        }
    }
    assert_eq!(row, n_cells, "every cell streamed once");
    (dense, statistics)
}

/// The worst difference in `float32` ulps between kyto and scanpy (`f64`, rounded to
/// `float32` first), and how many values are bit-identical.
fn ulps(kyto: &[f32], scanpy: &[f64]) -> (u32, usize) {
    assert_eq!(kyto.len(), scanpy.len(), "length");
    let mut worst = 0;
    let mut identical = 0;
    for (&kyto, &scanpy) in kyto.iter().zip(scanpy) {
        let scanpy = scanpy as f32;
        if kyto.to_bits() == scanpy.to_bits() || kyto.is_nan() && scanpy.is_nan() {
            identical += 1;
            continue;
        }
        // Ordered integers: adjacent floats differ by 1, across zero too.
        let ordered = |value: f32| {
            let bits = value.to_bits() as i32;
            if bits < 0 { i32::MIN - bits } else { bits }
        };
        worst = worst.max(ordered(kyto).abs_diff(ordered(scanpy)));
    }
    (worst, identical)
}

fn relative(kyto: &[f64], scanpy: &[f64]) -> f64 {
    assert_eq!(kyto.len(), scanpy.len());
    kyto.iter()
        .zip(scanpy)
        .map(|(kyto, scanpy)| (kyto - scanpy).abs() / scanpy.abs().max(f64::MIN_POSITIVE))
        .fold(0.0, f64::max)
}

#[test]
fn scale_matches_scanpy_on_fixtures() {
    let work = tempfile::tempdir().unwrap();
    let all = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    for name in ["synthetic", "pbmc3k"] {
        let h5ad = work.path().join(format!("{name}.h5ad"));
        let reference = work.path().join(format!("{name}-reference"));
        std::fs::create_dir(&reference).unwrap();
        let Some((min_cells, target_sum, max_value)) = write_fixture(name, &h5ad, &reference)
        else {
            eprintln!("skipped: anndata is not installed on this platform");
            return;
        };
        let store = work.path().join(format!("{name}.kyto"));
        let options = ConvertOptions {
            tiling: Tiling {
                cell_block_size: 256,
                gene_block_size: 4096,
            },
            ..ConvertOptions::default()
        };
        convert(&h5ad, &store, &options).unwrap();
        let n_cells = open_atlas_schema(&store).unwrap().n_cells as usize;
        let mut base = scan(&store);
        if min_cells > 0 {
            base = base
                .select_genes(read_lines(&reference.join("filtered")))
                .unwrap();
        }
        let log1p = base
            .normalize_total(Some(target_sum), Some("counts"), "normalized")
            .unwrap()
            .log1p(Some("normalized"), "log1p")
            .unwrap()
            .select_genes(read_lines(&reference.join("hvg")))
            .unwrap();
        let mut variants = vec![(true, Some(max_value))];
        if name == "synthetic" {
            variants.extend([(true, None), (false, Some(max_value)), (false, None)]);
        }
        for (zero_center, clip) in variants {
            let label = format!(
                "{name} zero_center={zero_center} max_value={}",
                clip.map_or("None".to_owned(), |clip| clip.to_string())
            );
            let plan = log1p
                .scale(Some("log1p"), "scaled", zero_center, clip)
                .unwrap();
            let (scaled, statistics) = run(&plan, n_cells, 1, None);
            let scanpy_label = format!(
                "scaled.{}.{}",
                if zero_center { "True" } else { "False" },
                clip.map_or("None".to_owned(), |clip| format!("{clip:.1}"))
            );
            // Against scanpy on kyto's input (correctly rounded log1p): the same values.
            let means = relative(
                statistics.means(),
                &read_f64(&reference.join("mean.exact.live")),
            );
            let stds = relative(
                statistics.standard_deviations(),
                &read_f64(&reference.join("std.exact.live")),
            );
            assert!(means <= 1e-12 && stds <= 1e-12, "{label}: {means} {stds}");
            let scanpy = read_f64(&reference.join(format!("{scanpy_label}.exact")));
            let (worst, identical) = ulps(&scaled, &scanpy);
            eprintln!(
                "{label}, scanpy on kyto's log1p: means {means:.1e}, stds {stds:.1e} \
                 (relative); {identical} of {} values bit-identical, worst {worst} ulp",
                scaled.len()
            );
            assert!(worst <= 1, "{label}: {worst} ulp");
            // Against the stored reference (numpy's log1p): 6.2's log1p deviation moves
            // the statistics by up to a few 1e-9 and a value by a few ulps.
            let means = relative(statistics.means(), &read_f64(&reference.join("mean")));
            let stds = relative(
                statistics.standard_deviations(),
                &read_f64(&reference.join("std")),
            );
            let scanpy = read_f64(&reference.join(&scanpy_label));
            let (worst, identical) = ulps(&scaled, &scanpy);
            let largest = scanpy
                .iter()
                .zip(&scaled)
                .map(|(scanpy, kyto)| (f64::from(*kyto) - scanpy).abs())
                .fold(0.0, f64::max);
            eprintln!(
                "{label}, scanpy reference: means {means:.1e}, stds {stds:.1e} (relative); \
                 {identical} of {} values bit-identical, worst {worst} ulp, {largest:.1e} absolute",
                scaled.len()
            );
            assert!(means <= 1e-7 && stds <= 1e-7, "{label}: {means} {stds}");
            assert!(largest <= 1e-5, "{label}: {largest}");
            if let Some(clip) = clip {
                let clip = clip as f32;
                let lower = if zero_center { -clip } else { f32::MIN };
                assert!(
                    scaled.iter().all(|&value| (lower..=clip).contains(&value)),
                    "{label}: clipped"
                );
            }
            for (threads, block_cells) in [(all, None), (all, Some(37))] {
                let (again, again_statistics) = run(&plan, n_cells, threads, block_cells);
                assert_eq!(again_statistics, statistics, "{label}: R11 statistics");
                assert!(
                    again
                        .iter()
                        .zip(&scaled)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{label}: R11 at {threads} threads, {block_cells:?} cells"
                );
            }
        }
    }
}
