//! Task 6.8: Leiden and Louvain against leidenalg.
//!
//! The stored references `leiden` and `louvain` (`tests/fixtures/make_reference.py`)
//! cluster the `neighbors` reference's `connectivities` at the resolutions 0.02, 0.5, 1
//! and 2 with seed 0: `tl.leiden(flavor="leidenalg")` and Louvain as `tl.louvain`'s
//! `vtraag` flavour runs it (no weights), through leidenalg's own Louvain loop (the
//! `louvain` package has no Python 3.13 build). kyto clusters scanpy's own graph. The
//! random streams differ, so labels are compared by the adjusted Rand index (the
//! Done-when check: printed with `--nocapture`), and the partitions' quality (modularity
//! with the resolution, the same formula on both sides, checked on scanpy's labels to
//! [`QUALITY_TOLERANCE`]) shows that kyto's is as good. Then: the same labels on every
//! run with one seed and on 1 and all threads (R11), and another seed's agreement.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;

use kyto_exec::{CellGraph, Clustering, cluster, partition_quality};
use kyto_plan::{Cluster, ClusterMethod};

/// kyto's quality formula against the reference script's on the same labels.
const QUALITY_TOLERANCE: f64 = 1e-9;
/// How far kyto's partition may fall below leidenalg's quality.
const QUALITY_SLACK: f64 = 0.01;
/// The lowest adjusted Rand index accepted for Leiden at a resolution with more than
/// one cluster.
const MIN_ADJUSTED_RAND: f64 = 0.7;
/// The same for Louvain, whose partitions vary more with the random order (two seeds
/// of kyto agree at 0.75 on pbmc3k); its quality bound is the stronger check.
const MIN_ADJUSTED_RAND_LOUVAIN: f64 = 0.5;

/// Writes the reference `connectivities` (`indptr`, `indices` as `<u8`, `data` as
/// `<f4`) and each reference's labels (`<method>_<resolution>_labels`, `<i4`); prints
/// each `method resolution quality` on a line. Exits 77 without numpy.
const WRITE_FIXTURE: &str = "\
import sys
try:
    import numpy as np
except ImportError:
    sys.exit(77)
from fixtures.reference import load_reference
name, out = sys.argv[1:3]
neighbors = load_reference(name, 'neighbors')
for key, dtype in [('indptr', '<u8'), ('indices', '<u8'), ('data', '<f4')]:
    np.asarray(neighbors[f'connectivities/{key}'], dtype=dtype).tofile(f'{out}/{key}')
for method in ['leiden', 'louvain']:
    reference = load_reference(name, method)
    for key in sorted(reference):
        if key.endswith('/labels'):
            resolution = key.split('/')[0]
            labels = np.asarray(reference[key], dtype='<i4')
            labels.tofile(f'{out}/{method}_{resolution}_labels')
            quality = float(reference[f'{resolution}/quality'][0])
            print(method, resolution, repr(quality))
";

struct Reference {
    method: ClusterMethod,
    resolution: f64,
    labels: Vec<u32>,
    quality: f64,
}

struct Fixture {
    indptr: Vec<u64>,
    indices: Vec<u32>,
    weights: Vec<f32>,
    references: Vec<Reference>,
}

impl Fixture {
    fn graph(&self) -> CellGraph<'_> {
        CellGraph::new(&self.indptr, &self.indices, &self.weights).unwrap()
    }
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
    let references = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| {
            let words: Vec<&str> = line.split_whitespace().collect();
            let method = match words[0] {
                "leiden" => ClusterMethod::Leiden,
                _ => ClusterMethod::Louvain,
            };
            let labels = read(
                &out.join(format!("{}_{}_labels", words[0], words[1])),
                i32::from_le_bytes,
            );
            Reference {
                method,
                resolution: words[1].parse().unwrap(),
                labels: labels.into_iter().map(|label| label as u32).collect(),
                quality: words[2].parse().unwrap(),
            }
        })
        .collect();
    Some(Fixture {
        indptr: read(&out.join("indptr"), u64::from_le_bytes),
        indices: read(&out.join("indices"), u64::from_le_bytes)
            .into_iter()
            .map(|index| u32::try_from(index).unwrap())
            .collect(),
        weights: read(&out.join("data"), f32::from_le_bytes),
        references,
    })
}

fn params(method: ClusterMethod, resolution: f64, seed: u64) -> Cluster {
    Cluster {
        method,
        graph: "connectivities".to_owned(),
        resolution,
        seed,
        into: method.name().to_owned(),
    }
}

/// The adjusted Rand index of two labellings (Hubert and Arabie); 1 when both put
/// every cell in one cluster.
fn adjusted_rand_index(left: &[u32], right: &[u32]) -> f64 {
    assert_eq!(left.len(), right.len());
    let pairs = |count: f64| count * (count - 1.0) / 2.0;
    let n_left = left.iter().max().map_or(0, |&label| label as usize + 1);
    let n_right = right.iter().max().map_or(0, |&label| label as usize + 1);
    let mut table = vec![0.0_f64; n_left * n_right];
    for (&a, &b) in left.iter().zip(right) {
        table[a as usize * n_right + b as usize] += 1.0;
    }
    let row_sums = (0..n_left).map(|a| (0..n_right).map(|b| table[a * n_right + b]).sum());
    let column_sums = (0..n_right).map(|b| (0..n_left).map(|a| table[a * n_right + b]).sum());
    let index: f64 = table.iter().map(|&count| pairs(count)).sum();
    let left_pairs: f64 = row_sums.map(|sum: f64| pairs(sum)).sum();
    let right_pairs: f64 = column_sums.map(|sum: f64| pairs(sum)).sum();
    let expected = left_pairs * right_pairs / pairs(left.len() as f64);
    let maximum = (left_pairs + right_pairs) / 2.0;
    if maximum == expected {
        return 1.0;
    }
    (index - expected) / (maximum - expected)
}

fn in_pool<T: Send>(threads: usize, work: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(work)
}

fn fixture_dir(work: &Path, name: &str) -> PathBuf {
    let out = work.join(name);
    std::fs::create_dir(&out).unwrap();
    out
}

#[test]
fn adjusted_rand_index_of_known_labellings() {
    assert_eq!(adjusted_rand_index(&[0, 0, 1, 1], &[1, 1, 0, 0]), 1.0);
    assert_eq!(adjusted_rand_index(&[0, 0, 0], &[0, 0, 0]), 1.0);
    // scikit-learn's adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 2]) = 0.5714...
    let value = adjusted_rand_index(&[0, 0, 1, 1], &[0, 0, 1, 2]);
    assert!((value - 4.0 / 7.0).abs() < 1e-12, "{value}");
}

#[test]
fn labels_agree_with_leidenalg() {
    let work = tempfile::tempdir().unwrap();
    for name in ["synthetic", "pbmc3k"] {
        let Some(fixture) = write_fixture(name, &fixture_dir(work.path(), name)) else {
            eprintln!("skipped: numpy is not installed on this platform");
            return;
        };
        assert_eq!(
            fixture.references.len(),
            8,
            "{name}: 2 methods x 4 resolutions"
        );
        let graph = fixture.graph();
        for reference in &fixture.references {
            let params = params(reference.method, reference.resolution, 0);
            let label = format!(
                "{name} {} {}",
                reference.method.name(),
                reference.resolution
            );
            // The same quality formula as the reference script, on its labels.
            let scanpy_quality = partition_quality(graph, &reference.labels, &params).unwrap();
            assert!(
                (scanpy_quality - reference.quality).abs() <= QUALITY_TOLERANCE,
                "{label}: quality {scanpy_quality} against the script's {}",
                reference.quality
            );
            let found = cluster(graph, &params).unwrap();
            let n_reference = reference.labels.iter().max().unwrap() + 1;
            let agreement = adjusted_rand_index(found.labels(), &reference.labels);
            eprintln!(
                "{label}: adjusted Rand index {agreement:.4}; clusters kyto {} leidenalg \
                 {n_reference}; quality kyto {:.5} leidenalg {:.5}",
                found.n_clusters(),
                found.quality(),
                reference.quality
            );
            assert!(
                found.quality() >= reference.quality - QUALITY_SLACK,
                "{label}: quality {} below leidenalg's {}",
                found.quality(),
                reference.quality
            );
            if n_reference > 1 {
                let floor = match reference.method {
                    ClusterMethod::Leiden => MIN_ADJUSTED_RAND,
                    ClusterMethod::Louvain => MIN_ADJUSTED_RAND_LOUVAIN,
                };
                assert!(agreement >= floor, "{label}: {agreement}");
            } else {
                assert_eq!(found.n_clusters(), 1, "{label}");
            }
        }
    }
}

#[test]
fn one_seed_gives_the_same_labels_on_every_run_and_thread_count() {
    let work = tempfile::tempdir().unwrap();
    let all = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let Some(fixture) = write_fixture("pbmc3k", &fixture_dir(work.path(), "pbmc3k")) else {
        eprintln!("skipped: numpy is not installed on this platform");
        return;
    };
    let graph = fixture.graph();
    for method in [ClusterMethod::Leiden, ClusterMethod::Louvain] {
        let params = params(method, 1.0, 7);
        let first: Clustering = cluster(graph, &params).unwrap();
        let again = in_pool(1, || cluster(graph, &params).unwrap());
        let wide = in_pool(all, || cluster(graph, &params).unwrap());
        assert_eq!(first, again, "{method:?}: a second run");
        assert_eq!(first, wide, "{method:?}: {all} threads");
        let other = cluster(graph, &Cluster { seed: 8, ..params }).unwrap();
        eprintln!(
            "pbmc3k {}: seeds 7 and 8 agree with adjusted Rand index {:.4}",
            method.name(),
            adjusted_rand_index(first.labels(), other.labels())
        );
    }
}
