//! Task 6.9: the UMAP layout against umap-learn, by trustworthiness.
//!
//! The stored scanpy reference `umap` (`tests/fixtures/make_reference.py`: `tl.umap`
//! with scanpy's defaults on the reference `neighbors` graph, seeds 0 to 4) holds the
//! layout at seed 0 and scikit-learn's trustworthiness of each seed's layout against
//! the reference `X_pca` (the space the neighbours were searched in) at 5 and 15
//! neighbours: umap-learn's range. kyto lays out the same graph (scanpy's own
//! `distances` through 6.7's connectivities, which equal scanpy's bit for bit) and its
//! trustworthiness must fall in that range, widened by [`TRUSTWORTHINESS_MARGIN`]: the
//! Done-when check. The layouts are not compared point by point: umap-learn's own
//! layouts differ that much between seeds (another random generator, deviations.md).
//! Also: `a`, `b` equal scanpy's; the same layout at 1 and all threads (R11), spectral
//! initialisation included; random initialisation stays in range too.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::NonZeroUsize;
use std::path::Path;
use std::process::Command;

use kyto_exec::{NeighborGraph, UmapLayout, connectivities, umap_layout};
use kyto_plan::{Metric, Umap, UmapInit};

/// How far below umap-learn's lowest (or above its highest) trustworthiness over five
/// seeds kyto's may be.
const TRUSTWORTHINESS_MARGIN: f64 = 0.005;

/// scanpy stores `a`, `b` from scipy's `curve_fit` (tolerance ~1e-8 relative).
const AB_TOLERANCE: f64 = 1e-6;

/// The trustworthiness this test computes against scikit-learn's on the same layout
/// (ties among the synthetic duplicates may rank differently).
const HELPER_TOLERANCE: f64 = 1e-6;

/// Writes the reference `X_pca` (`<f4`), `distances` (`indptr`, `indices` as `<u8`,
/// `data` as `<f8`), scanpy's layout (`<f4`), its `a`, `b` and trustworthiness (`<f8`);
/// prints `n_pcs n_neighbors width n_seeds`. Exits 77 without scanpy's references'
/// readers.
const WRITE_FIXTURE: &str = "\
import sys
try:
    import scipy  # noqa: F401
except ImportError:
    sys.exit(77)
import json
import numpy as np
from fixtures.reference import load_reference
name, out = sys.argv[1:3]
manifest = json.load(open('fixtures/reference/manifest.json'))['references']
parameters = manifest[name + '/neighbors']['parameters']
pca = load_reference(name, 'pca')
neighbors = load_reference(name, 'neighbors')
umap = load_reference(name, 'umap')
np.asarray(pca['embedding'], dtype='<f4').tofile(f'{out}/space')
for key in ['indptr', 'indices']:
    np.asarray(neighbors[f'distances/{key}'], dtype='<u8').tofile(f'{out}/distances_{key}')
np.asarray(neighbors['distances/data'], dtype='<f8').tofile(f'{out}/distances_data')
np.asarray(umap['embedding'], dtype='<f4').tofile(f'{out}/layout')
np.asarray([umap['a'][0], umap['b'][0]], dtype='<f8').tofile(f'{out}/ab')
for k in (5, 15):
    np.asarray(umap[f'trustworthiness/{k}'], dtype='<f8').tofile(f'{out}/trust_{k}')
print(parameters['n_pcs'], parameters['n_neighbors'], pca['embedding'].shape[1],
      len(umap['trustworthiness/5']))
";

struct Fixture {
    out: std::path::PathBuf,
    n_neighbors: usize,
    width: usize,
}

fn write_fixture(name: &str, out: &Path) -> Option<Fixture> {
    let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests");
    let output = Command::new(std::env::var("PYTHON").unwrap_or_else(|_| "python".to_owned()))
        .current_dir(tests)
        .args(["-c", WRITE_FIXTURE, name])
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
    let words: Vec<&str> = text.split_whitespace().collect();
    Some(Fixture {
        out: out.to_owned(),
        n_neighbors: words[1].parse().unwrap(),
        width: words[2].parse().unwrap(),
    })
}

fn read<const N: usize, T>(path: &Path, from: fn([u8; N]) -> T) -> Vec<T> {
    std::fs::read(path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .as_chunks::<N>()
        .0
        .iter()
        .map(|bytes| from(*bytes))
        .collect()
}

impl Fixture {
    fn file<const N: usize, T>(&self, name: &str, from: fn([u8; N]) -> T) -> Vec<T> {
        read(&self.out.join(name), from)
    }

    /// scanpy's `distances` as a [`NeighborGraph`] (its `float64` distances as the
    /// `float32` umap computes with), as 6.7's parity test builds it.
    fn scanpy_distances(&self) -> NeighborGraph {
        let indptr = self.file("distances_indptr", u64::from_le_bytes);
        let indices = self
            .file("distances_indices", u64::from_le_bytes)
            .into_iter()
            .map(|index| u32::try_from(index).unwrap())
            .collect();
        let distances = self
            .file("distances_data", f64::from_le_bytes)
            .into_iter()
            .map(|d| d as f32)
            .collect();
        NeighborGraph::from_csr(self.n_neighbors, indptr, indices, distances).unwrap()
    }
}

/// scikit-learn's `trustworthiness(space, layout, n_neighbors=k)` from the ranks of
/// every cell's distances in the original space (ties by position).
struct Trustworthiness {
    n: usize,
    /// `ranks[i * n + j]`: 1-based rank of `j` among `i`'s distances (`i` itself last).
    ranks: Vec<u32>,
}

impl Trustworthiness {
    fn new(space: &[f32], width: usize) -> Self {
        let n = space.len() / width;
        let mut ranks = vec![0_u32; n * n];
        let row = |i: usize| &space[i * width..(i + 1) * width];
        let mut order: Vec<(f64, usize)> = Vec::with_capacity(n);
        for i in 0..n {
            order.clear();
            for j in 0..n {
                let distance = if i == j {
                    f64::INFINITY
                } else {
                    row(i)
                        .iter()
                        .zip(row(j))
                        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                        .sum::<f64>()
                };
                order.push((distance, j));
            }
            order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for (rank, &(_, j)) in order.iter().enumerate() {
                ranks[i * n + j] = rank as u32 + 1;
            }
        }
        Self { n, ranks }
    }

    fn score(&self, layout: &[f32], dim: usize, k: usize) -> f64 {
        let n = self.n;
        let row = |i: usize| &layout[i * dim..(i + 1) * dim];
        let mut total = 0_u64;
        let mut order: Vec<(f64, usize)> = Vec::with_capacity(n);
        for i in 0..n {
            order.clear();
            for j in (0..n).filter(|&j| j != i) {
                let distance: f64 = row(i)
                    .iter()
                    .zip(row(j))
                    .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                    .sum();
                order.push((distance, j));
            }
            order.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for &(_, j) in &order[..k] {
                total += u64::from(self.ranks[i * n + j].saturating_sub(k as u32));
            }
        }
        let (n, k) = (n as f64, k as f64);
        1.0 - total as f64 * (2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0)))
    }
}

fn in_pool<T: Send>(threads: usize, work: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(work)
}

fn range(values: &[f64]) -> (f64, f64) {
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (low, high)
}

#[test]
fn trustworthiness_is_within_umap_learns_range() {
    let work = tempfile::tempdir().unwrap();
    let all = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    for name in ["synthetic", "pbmc3k"] {
        let out = work.path().join(name);
        std::fs::create_dir(&out).unwrap();
        let Some(fixture) = write_fixture(name, &out) else {
            eprintln!("skipped: scipy is not installed on this platform");
            return;
        };
        let space = fixture.file("space", f32::from_le_bytes);
        let truth = Trustworthiness::new(&space, fixture.width);
        let scanpy_layout = fixture.file("layout", f32::from_le_bytes);
        let ab = fixture.file("ab", f64::from_le_bytes);
        let mut bounds = Vec::new();
        for k in [5, 15] {
            let scanpy = fixture.file(&format!("trust_{k}"), f64::from_le_bytes);
            // The helper is scikit-learn's measure.
            let helper = truth.score(&scanpy_layout, 2, k);
            assert!(
                (helper - scanpy[0]).abs() <= HELPER_TOLERANCE,
                "{name}: trustworthiness helper {helper} vs scikit-learn {}",
                scanpy[0]
            );
            let (low, high) = range(&scanpy);
            eprintln!("{name}: umap-learn trustworthiness@{k} over 5 seeds: {low:.4} to {high:.4}");
            bounds.push((
                k,
                low - TRUSTWORTHINESS_MARGIN,
                high + TRUSTWORTHINESS_MARGIN,
            ));
        }

        let graph = connectivities(&fixture.scanpy_distances());
        let data = Some((space.as_slice(), fixture.width, Metric::Euclidean));
        let check = |layout: &UmapLayout, what: &str| {
            for &(k, low, high) in &bounds {
                let score = truth.score(layout.embedding(), layout.n_components(), k);
                eprintln!("{name}: kyto {what}: trustworthiness@{k} {score:.4}");
                assert!(
                    (low..=high).contains(&score),
                    "{name} {what}: trustworthiness@{k} {score} outside [{low}, {high}]"
                );
            }
        };
        for seed in [0, 1] {
            let params = Umap {
                seed,
                ..Umap::default()
            };
            let layout = in_pool(1, || umap_layout(&graph, data, &params).unwrap());
            assert!(layout.embedding().iter().all(|value| value.is_finite()));
            assert!(
                (layout.a() - ab[0]).abs() <= AB_TOLERANCE * ab[0],
                "{name}: a"
            );
            assert!(
                (layout.b() - ab[1]).abs() <= AB_TOLERANCE * ab[1],
                "{name}: b"
            );
            check(&layout, &format!("seed {seed}"));
            if seed == 0 {
                let again = in_pool(all, || umap_layout(&graph, data, &params).unwrap());
                assert_eq!(layout, again, "{name}: 1 and {all} threads");
                let other = Umap {
                    seed: 1,
                    ..params.clone()
                };
                let moved = in_pool(1, || umap_layout(&graph, data, &other).unwrap());
                assert_ne!(
                    layout.embedding(),
                    moved.embedding(),
                    "{name}: the seed is used"
                );
            }
        }
        let random = Umap {
            init: UmapInit::Random,
            ..Umap::default()
        };
        check(&umap_layout(&graph, data, &random).unwrap(), "random init");
    }
}
