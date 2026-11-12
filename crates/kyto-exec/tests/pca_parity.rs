//! Task 6.5: PCA.
//!
//! Done when: explained variance matches scanpy to 1e-3 and subspace angles are small
//! on fixtures. The synthetic and pbmc3k fixtures (written as `.h5ad` by
//! `tests/fixtures`, converted here; skipped without anndata, exit code 77) run the
//! stored scanpy reference `pca` (`tests/fixtures/make_reference.py`: genes seen in at
//! least `min_cells_per_gene` cells, `normalize_total(1e4) -> log1p`, the `seurat`
//! HVGs, `pp.pca(n_comps, zero_center=True, svd_solver="arpack", random_state=0)` on
//! the sparse matrix, implicitly centred). kyto runs the same steps (`select_genes`
//! with the reference's genes) and `pca(n_components, seed=0)`.
//!
//! scanpy's own PCA moves by up to 1e-3 between platforms (arpack in `float32` on a
//! nearly flat spectrum, memory/fixture_references.md), so components are compared as
//! the plan says: every variance and variance ratio to 1e-3 relative, the subspaces of
//! the leading components by their largest principal angle, and each leading
//! component (loadings and embedding column, signs included: both follow
//! scikit-learn's `svd_flip` on the loadings) one by one.
//! Measured (9 power iterations, 11 passes): synthetic (100 genes, the basis spans them
//! all) variances to 1.5e-6 relative, angles 4.3e-6 rad; pbmc3k variances and ratios
//! to 1.3e-6 (the `float32` reference's own precision), the first 10 components' span
//! within 9.6e-6 rad and each within 2.2e-5 of its column, all 50 within 1.0e-3 rad.
//! Also the same bits at 1 and all threads and at blocks of 37 cells (R11), and
//! another seed changes the bits but not the result beyond the tolerances.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use faer::Mat;
use kyto_core::{GeneIds, Tiling};
use kyto_exec::{ExecuteOptions, NodeStatistics, PcaStatistics, Pipeline};
use kyto_plan::{Member, OptimizeOptions, Output, PhysicalOptions, Plan, physical_plan};
use kyto_store::{ConvertOptions, TableAxis, convert, open_atlas_schema, read_ids};

/// Writes `load_fixture(name)` as `.h5ad`; the filtered gene IDs, the HVG gene IDs and
/// the reference's arrays as `<f8` (`variance`, `variance_ratio`, `components` genes x
/// k and `embedding` cells x k, row-major); prints `min_cells_per_gene target_sum
/// n_comps`. Exits 77 without anndata.
const WRITE_FIXTURE: &str = "\
import sys
try:
    from fixtures.interop import to_anndata
except ImportError:
    sys.exit(77)
import json
import numpy as np
from fixtures.make_reference import _filtered_counts
from fixtures.reference import load_reference
from fixtures.registry import load_fixture
name, h5ad, out = sys.argv[1:4]
to_anndata(load_fixture(name)).write_h5ad(h5ad)
manifest = json.load(open('fixtures/reference/manifest.json'))['references']
parameters = manifest[name + '/pca']['parameters']
target = manifest[name + '/normalize_log1p']['parameters']['target_sum']
reference = load_reference(name, 'pca')
open(f'{out}/filtered', 'w').write('\\n'.join(_filtered_counts(name).var_names.tolist()))
open(f'{out}/hvg', 'w').write('\\n'.join(reference['genes/gene_id'].tolist()))
for key in ['variance', 'variance_ratio', 'components', 'embedding']:
    np.asarray(reference[key], dtype='<f8').tofile(f'{out}/{key}')
print(parameters['min_cells_per_gene'], target, parameters['n_comps'])
";

/// Write fixture `name` to `h5ad` and its reference to `out`; its
/// `min_cells_per_gene`, `target_sum` and `n_comps`, or `None` without anndata.
fn write_fixture(name: &str, h5ad: &Path, out: &Path) -> Option<(u64, f64, u64)> {
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

/// The PCA's statistics, and the cells the output stream gave.
fn run(plan: &Arc<Plan>, threads: usize, block_cells: Option<u64>) -> (PcaStatistics, usize) {
    let output = Output::members(plan, [Member::Embedding("X_pca".to_owned())]).unwrap();
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
            NodeStatistics::Pca(pca) => Some((**pca).clone()),
            _ => None,
        })
        .expect("pca's statistics");
    let streamed = stream
        .by_ref()
        .map(|block| block.unwrap().cells.len())
        .sum();
    (statistics, streamed)
}

fn relative(kyto: &[f64], scanpy: &[f64]) -> f64 {
    assert_eq!(kyto.len(), scanpy.len());
    kyto.iter()
        .zip(scanpy)
        .map(|(kyto, scanpy)| (kyto - scanpy).abs() / scanpy.abs())
        .fold(0.0, f64::max)
}

/// Column `column` of a row-major matrix with `width` columns.
fn column<T: Copy + Into<f64>>(values: &[T], width: usize, column: usize) -> Vec<f64> {
    values
        .iter()
        .skip(column)
        .step_by(width)
        .map(|&value| value.into())
        .collect()
}

/// The first `m` columns of a row-major `rows x width` matrix, orthonormalised.
fn leading(values: &[f64], width: usize, m: usize) -> Mat<f64> {
    let rows = values.len() / width;
    let matrix = Mat::from_fn(rows, m, |row, column| values[row * width + column]);
    matrix.qr().compute_thin_Q()
}

/// The largest principal angle (radians) between the spans of the first `m` loadings.
fn largest_angle(kyto: &[f64], scanpy: &[f64], width: usize, m: usize) -> f64 {
    let product = leading(kyto, width, m).transpose() * leading(scanpy, width, m);
    let singular = product.singular_values().unwrap();
    let smallest = singular.iter().copied().fold(f64::INFINITY, f64::min);
    smallest.min(1.0).acos()
}

/// The largest difference between two columns, relative to the larger of them.
fn column_difference(kyto: &[f64], scanpy: &[f64]) -> f64 {
    let scale = scanpy.iter().map(|value| value.abs()).fold(0.0, f64::max);
    kyto.iter()
        .zip(scanpy)
        .map(|(kyto, scanpy)| (kyto - scanpy).abs())
        .fold(0.0, f64::max)
        / scale
}

#[test]
fn pca_matches_scanpy_on_fixtures() {
    let work = tempfile::tempdir().unwrap();
    let all = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    // The leading components compared one by one and as a subspace: those with a gap
    // to the next variance (scanpy's own runs agree on them across platforms).
    for (name, compared) in [("synthetic", 3), ("pbmc3k", 10)] {
        let h5ad = work.path().join(format!("{name}.h5ad"));
        let reference = work.path().join(format!("{name}-reference"));
        std::fs::create_dir(&reference).unwrap();
        let Some((min_cells, target_sum, n_comps)) = write_fixture(name, &h5ad, &reference) else {
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
        let plan = log1p.pca(Some("log1p"), "X_pca", n_comps, 0).unwrap();
        let (pca, streamed) = run(&plan, 1, None);
        let k = n_comps as usize;
        let n_genes = read_lines(&reference.join("hvg")).len();
        assert_eq!(
            streamed, n_cells,
            "{name}: the output stream carries every cell"
        );
        assert_eq!(pca.n_components(), k);
        assert_eq!(pca.cells().len(), n_cells);
        assert!(pca.cells().windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(pca.embedding().len(), n_cells * k);
        assert_eq!(pca.loadings().len(), n_genes * k);

        let variance = relative(pca.variance(), &read_f64(&reference.join("variance")));
        let ratio = relative(
            pca.variance_ratio(),
            &read_f64(&reference.join("variance_ratio")),
        );
        let loadings: Vec<f64> = pca.loadings().iter().map(|&value| value.into()).collect();
        let components = read_f64(&reference.join("components"));
        let embedding = read_f64(&reference.join("embedding"));
        let angle = largest_angle(&loadings, &components, k, compared);
        let angle_all = largest_angle(&loadings, &components, k, k);
        let mut worst_loading: f64 = 0.0;
        let mut worst_embedding: f64 = 0.0;
        for component in 0..compared {
            worst_loading = worst_loading.max(column_difference(
                &column(&loadings, k, component),
                &column(&components, k, component),
            ));
            worst_embedding = worst_embedding.max(column_difference(
                &column(pca.embedding(), k, component),
                &column(&embedding, k, component),
            ));
        }
        eprintln!(
            "{name}: {k} components; variance {variance:.1e}, variance ratio {ratio:.1e} \
             (relative, worst); largest angle {angle:.1e} rad over the first {compared}, \
             {angle_all:.1e} over all {k}; first {compared} one by one: loadings \
             {worst_loading:.1e}, embedding {worst_embedding:.1e} (relative to the column)"
        );
        assert!(
            variance <= 1e-3 && ratio <= 1e-3,
            "{name}: {variance} {ratio}"
        );
        assert!(angle <= 1e-3, "{name}: angle {angle}");
        // The trailing components sit in a nearly flat spectrum (where scanpy moves by
        // 1e-3 between platforms): their span still agrees, more loosely.
        assert!(angle_all <= 1e-2, "{name}: angle over all {angle_all}");
        assert!(worst_loading <= 1e-3, "{name}: loadings {worst_loading}");
        assert!(
            worst_embedding <= 1e-3,
            "{name}: embedding {worst_embedding}"
        );

        for (threads, block_cells) in [(all, None), (all, Some(37))] {
            let (again, _) = run(&plan, threads, block_cells);
            assert_eq!(
                again, pca,
                "{name}: R11 at {threads} threads, {block_cells:?} cells"
            );
        }
        let (other, _) = run(
            &log1p.pca(Some("log1p"), "X_pca", n_comps, 1).unwrap(),
            1,
            None,
        );
        // synthetic's 100 genes: the basis spans them all, so the result is exact and
        // the seed changes nothing; pbmc3k's 2,000 are sketched.
        if name == "pbmc3k" {
            assert_ne!(
                other.embedding(),
                pca.embedding(),
                "{name}: the seed is used"
            );
        }
        assert!(
            relative(other.variance(), pca.variance()) <= 1e-3,
            "{name}: another seed"
        );
    }
}
