//! Task 6.17: approximate neighbours lose no cells on clustered data.
//!
//! 10.8's 50k atlas (`benchmarks/scanpy_workflow/make_data.py`: 8 cell populations of
//! negative binomial counts) has, after the workflow's PCA (50 components), 8 clusters
//! of about 6,200 cells, each close to an isotropic Gaussian (standard deviation about
//! 0.97 in every component) with centres about 15.4 apart. The generator here draws
//! that shape from a seed: centre `j` at [`SEPARATION`] on axis `j` (15.4 apart), noise
//! [`SPREAD`] in all 50 components. On it, the earlier HNSW (`hnsw_rs`) left 1.3% of
//! 30,000 cells and 1.1% of 50,000 at recall 0, as on the real 50k data (2.0%); the
//! index of `neighbors_hnsw.rs` leaves none. The suite runs a smaller atlas of the same
//! structure; the ignored test runs 50,000 cells (`cargo test --release`).
//! Checked against kyto's exact search: the share of cells with recall below 0.5
//! (`neighbors.hnsw_lost_cells`), the mean recall (`neighbors.hnsw_recall`), and the
//! adjusted Rand index of Leiden on both graphs (`neighbors.hnsw_leiden_ari`); and the
//! approximate graph is the same on 1 and all threads (R11). Measured values are
//! printed with `--nocapture`.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::NonZeroUsize;

use kyto_exec::{CellGraph, NeighborGraph, cluster, connectivities, nearest_neighbors};
use kyto_plan::{Cluster, ClusterMethod, Metric, NeighborSearch, Neighbors};

use crate::cluster_parity::adjusted_rand_index;
use crate::tolerances::tolerance;

/// Components of the embedding (the workflow's PCA).
const WIDTH: usize = 50;
/// Cell populations.
const CLUSTERS: usize = 8;
/// A centre's offset on its own axis: centres are `SEPARATION * sqrt(2)` apart.
const SEPARATION: f64 = 10.9;
/// The standard deviation of every component around a centre.
const SPREAD: f64 = 0.97;

/// SplitMix64: the test's own seeded stream (memory/test_seed_isolation.md).
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// Uniform in `(0, 1]`.
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) + 1) as f64 / (1_u64 << 53) as f64
    }

    /// Standard normal (Box-Muller, cosine branch).
    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * self.unit().ln()).sqrt();
        radius * (std::f64::consts::TAU * self.unit()).cos()
    }
}

/// `n_cells` rows of the clustered embedding, cell `i` in cluster `i % CLUSTERS`.
fn clustered_embedding(n_cells: usize, seed: u64) -> Vec<f32> {
    let mut stream = Stream(seed);
    (0..n_cells)
        .flat_map(|cell| (0..WIDTH).map(move |component| (cell, component)))
        .map(|(cell, component)| {
            let centre = if component == cell % CLUSTERS {
                SEPARATION
            } else {
                0.0
            };
            (centre + SPREAD * stream.normal()) as f32
        })
        .collect()
}

fn params(search: NeighborSearch) -> Neighbors {
    Neighbors {
        embedding: "X_pca".to_owned(),
        n_components: None,
        n_neighbors: 15,
        metric: Metric::Euclidean,
        search,
        seed: 0,
        key: None,
    }
}

fn in_pool<T: Send>(threads: usize, work: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(work)
}

/// Each row's share of the exact neighbours the approximate row holds.
fn recalls(approximate: &NeighborGraph, exact: &NeighborGraph) -> Vec<f64> {
    let rows = |graph: &NeighborGraph| -> Vec<Vec<u32>> {
        graph
            .indptr()
            .windows(2)
            .map(|bounds| graph.indices()[bounds[0] as usize..bounds[1] as usize].to_vec())
            .collect()
    };
    rows(approximate)
        .iter()
        .zip(&rows(exact))
        .map(|(ours, exact)| {
            let found = ours.iter().filter(|row| exact.contains(row)).count();
            found as f64 / exact.len() as f64
        })
        .collect()
}

/// Leiden labels (resolution 1, seed 0) of a neighbour graph's connectivities.
fn leiden(graph: &NeighborGraph) -> Vec<u32> {
    let weights = connectivities(graph);
    let cells = CellGraph::new(weights.indptr(), weights.indices(), weights.weights()).unwrap();
    let params = Cluster {
        method: ClusterMethod::Leiden,
        graph: "connectivities".to_owned(),
        resolution: 1.0,
        seed: 0,
        into: "leiden".to_owned(),
    };
    cluster(cells, &params).unwrap().labels().to_vec()
}

fn check(n_cells: usize) {
    let embedding = clustered_embedding(n_cells, 17);
    let all = std::thread::available_parallelism().map_or(4, NonZeroUsize::get);
    let exact = in_pool(all, || {
        nearest_neighbors(&embedding, WIDTH, &params(NeighborSearch::Exact)).unwrap()
    });
    let approximate = in_pool(all, || {
        nearest_neighbors(&embedding, WIDTH, &params(NeighborSearch::Approximate)).unwrap()
    });
    let one_thread = in_pool(1, || {
        nearest_neighbors(&embedding, WIDTH, &params(NeighborSearch::Approximate)).unwrap()
    });
    assert_eq!(
        approximate, one_thread,
        "{n_cells} cells: HNSW at 1 and {all} threads"
    );

    let recalls = recalls(&approximate, &exact);
    let mean = recalls.iter().sum::<f64>() / n_cells as f64;
    let lost = recalls.iter().filter(|&&recall| recall < 0.5).count();
    let lost_share = lost as f64 / n_cells as f64;
    let exact_labels = leiden(&exact);
    let approximate_labels = leiden(&approximate);
    let agreement = adjusted_rand_index(&approximate_labels, &exact_labels);
    let n_clusters = exact_labels.iter().max().unwrap() + 1;
    println!(
        "{n_cells} cells: mean recall {mean:.5}, {lost} cells ({:.3}%) below 0.5, \
         minimum {:.3}; Leiden ARI {agreement:.6} ({n_clusters} clusters on exact)",
        100.0 * lost_share,
        recalls.iter().copied().fold(1.0, f64::min),
    );
    assert!(
        lost_share < tolerance("neighbors.hnsw_lost_cells"),
        "{n_cells} cells: {lost} with recall below 0.5"
    );
    assert!(
        mean >= tolerance("neighbors.hnsw_recall"),
        "{n_cells} cells: mean recall {mean}"
    );
    assert!(
        agreement >= tolerance("neighbors.hnsw_leiden_ari"),
        "{n_cells} cells: Leiden ARI {agreement}"
    );
}

#[test]
fn hnsw_loses_no_cells_on_clustered_data() {
    check(3_000);
}

#[test]
#[ignore = "50k cells: run in release (`cargo test --release -p kyto-exec --test it -- neighbors_recall:: --ignored --nocapture`)"]
fn hnsw_loses_no_cells_at_50k() {
    check(50_000);
}

#[test]
fn the_generator_has_the_atlas_structure() {
    let embedding = clustered_embedding(800, 17);
    let centre_distance = |left: usize, right: usize| {
        let mean = |cluster: usize| -> Vec<f64> {
            let mut sum = vec![0.0; WIDTH];
            for row in embedding
                .as_chunks::<WIDTH>()
                .0
                .iter()
                .skip(cluster)
                .step_by(CLUSTERS)
            {
                for (total, &value) in sum.iter_mut().zip(row) {
                    *total += f64::from(value);
                }
            }
            sum.iter().map(|total| total / 100.0).collect()
        };
        let (left, right) = (mean(left), mean(right));
        left.iter()
            .zip(&right)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    };
    // 15.4 apart, give or take the noise of 100 cells a centre.
    assert!((14.9..15.9).contains(&centre_distance(0, 1)));
    assert!((14.9..15.9).contains(&centre_distance(3, 7)));
    assert_eq!(clustered_embedding(800, 17), embedding);
}
