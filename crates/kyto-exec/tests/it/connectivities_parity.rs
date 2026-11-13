//! Task 6.7: connectivities against scanpy.
//!
//! The stored scanpy reference `neighbors` (`tests/fixtures/make_reference.py`:
//! `pp.neighbors(n_neighbors=15, n_pcs, method="umap", metric="euclidean")` on the
//! reference `pca` embedding) holds both graphs. The Done-when check gives kyto
//! scanpy's own `distances` (the same neighbours) and compares kyto's weights with
//! scanpy's `connectivities`: same edges, weights within [`WEIGHT_TOLERANCE`]
//! (measured: all 9,762 and 62,096 weights equal bit for bit, scanpy's self loops at
//! the synthetic duplicates included), the same at 1 and all threads (R11). Then
//! kyto's own exact search on the same stored `X_pca` (pbmc3k, where every neighbour
//! row equals scanpy's) gives the same edges and weights within
//! [`SEARCH_WEIGHT_TOLERANCE`] (measured 1.1e-5, from the distances' last bits).

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::NonZeroUsize;
use std::path::Path;
use std::process::Command;

use kyto_exec::{ConnectivityGraph, NeighborGraph, connectivities, nearest_neighbors};
use kyto_plan::{Metric, NeighborSearch, Neighbors};

/// The Done-when bound on a weight given scanpy's neighbours (plan 6.7).
const WEIGHT_TOLERANCE: f64 = 1e-5;
/// The bound on kyto's own search: its distances differ from scanpy's by up to 1.2e-7
/// relative (6.6), which `(d - rho) / sigma` amplifies (measured 1.1e-5).
const SEARCH_WEIGHT_TOLERANCE: f64 = 5e-5;

/// Writes the reference `X_pca` as `<f4` (`embedding`), the reference `distances`
/// (`distances_indptr`, `distances_indices` as `<u8`, `distances_data` as `<f8`) and
/// `connectivities` (the same, `connectivities_data` as `<f4`); prints `n_pcs
/// n_neighbors width`. Exits 77 without scanpy's references' readers.
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
np.asarray(pca['embedding'], dtype='<f4').tofile(f'{out}/embedding')
for graph, dtype in [('distances', '<f8'), ('connectivities', '<f4')]:
    for key in ['indptr', 'indices']:
        np.asarray(neighbors[f'{graph}/{key}'], dtype='<u8').tofile(f'{out}/{graph}_{key}')
    np.asarray(neighbors[f'{graph}/data'], dtype=dtype).tofile(f'{out}/{graph}_data')
print(parameters['n_pcs'], parameters['n_neighbors'], pca['embedding'].shape[1])
";

/// What `WRITE_FIXTURE` prints, and where it wrote.
struct Fixture {
    out: std::path::PathBuf,
    n_pcs: u64,
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
        n_pcs: words[0].parse().unwrap(),
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
    fn csr(&self, graph: &str) -> (Vec<u64>, Vec<u32>) {
        let indptr = read(
            &self.out.join(format!("{graph}_indptr")),
            u64::from_le_bytes,
        );
        let indices = read(
            &self.out.join(format!("{graph}_indices")),
            u64::from_le_bytes,
        )
        .into_iter()
        .map(|index| u32::try_from(index).unwrap())
        .collect();
        (indptr, indices)
    }

    /// scanpy's `distances` as a [`NeighborGraph`] (its `float64` distances as the
    /// `float32` umap computes with).
    fn scanpy_distances(&self) -> NeighborGraph {
        let (indptr, indices) = self.csr("distances");
        let data = read(&self.out.join("distances_data"), f64::from_le_bytes);
        let distances = data.into_iter().map(|d| d as f32).collect();
        NeighborGraph::from_csr(self.n_neighbors, indptr, indices, distances).unwrap()
    }

    /// scanpy's `connectivities` as (indptr, indices, weights).
    fn scanpy_connectivities(&self) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
        let (indptr, indices) = self.csr("connectivities");
        let data = read(&self.out.join("connectivities_data"), f32::from_le_bytes);
        (indptr, indices, data)
    }
}

/// Asserts the same edges as scanpy; returns the largest weight difference and the
/// number of weights equal bit for bit.
fn compare(
    name: &str,
    kyto: &ConnectivityGraph,
    scanpy: &(Vec<u64>, Vec<u32>, Vec<f32>),
) -> (f64, usize) {
    assert_eq!(kyto.indptr(), scanpy.0, "{name}: row lengths");
    assert_eq!(kyto.indices(), scanpy.1, "{name}: edges");
    let mut worst = 0.0_f64;
    let mut equal = 0;
    for (&left, &right) in kyto.weights().iter().zip(&scanpy.2) {
        worst = worst.max((f64::from(left) - f64::from(right)).abs());
        equal += usize::from(left.to_bits() == right.to_bits());
    }
    (worst, equal)
}

fn in_pool<T: Send>(threads: usize, work: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(work)
}

#[test]
fn weights_match_scanpy_given_the_same_neighbours() {
    let work = tempfile::tempdir().unwrap();
    let all = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    for name in ["synthetic", "pbmc3k"] {
        let out = work.path().join(name);
        std::fs::create_dir(&out).unwrap();
        let Some(fixture) = write_fixture(name, &out) else {
            eprintln!("skipped: scipy is not installed on this platform");
            return;
        };
        let scanpy = fixture.scanpy_connectivities();

        // The Done-when check: scanpy's own neighbours in.
        let distances = fixture.scanpy_distances();
        let weights = in_pool(1, || connectivities(&distances));
        let weights_all = in_pool(all, || connectivities(&distances));
        assert_eq!(weights, weights_all, "{name}: 1 and {all} threads");
        let (worst, equal) = compare(name, &weights, &scanpy);
        eprintln!(
            "{name}: connectivities on scanpy's distances: {} edges, {equal} equal bit for \
             bit, largest difference {worst:.1e}",
            scanpy.2.len()
        );
        assert!(worst <= WEIGHT_TOLERANCE, "{name}: {worst:.1e}");

        // kyto's own exact search on the same X_pca (every row equal on pbmc3k; the
        // synthetic duplicates differ by deviations.md's neighbour rows).
        if name == "pbmc3k" {
            let embedding = read(&out.join("embedding"), f32::from_le_bytes);
            let params = Neighbors {
                embedding: "X_pca".to_owned(),
                n_components: Some(fixture.n_pcs),
                n_neighbors: fixture.n_neighbors as u64,
                metric: Metric::Euclidean,
                search: NeighborSearch::Exact,
                seed: 0,
                key: None,
            };
            let searched = nearest_neighbors(&embedding, fixture.width, &params).unwrap();
            let (worst, equal) = compare(name, &connectivities(&searched), &scanpy);
            eprintln!(
                "{name}: connectivities on kyto's search: {equal} of {} equal bit for bit, \
                 largest difference {worst:.1e}",
                scanpy.2.len()
            );
            assert!(worst <= SEARCH_WEIGHT_TOLERANCE, "{name}: {worst:.1e}");
        }
    }
}
