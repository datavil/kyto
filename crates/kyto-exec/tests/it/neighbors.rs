//! Task 6.6: exact nearest neighbours equal brute force.
//!
//! Done when: exact mode equals brute force. The brute force here computes every pair's
//! distance one value at a time with the kernels' documented lane order (its own lane
//! tree, as `kernels.rs`' tests do), sorts all other rows by (distance, row) and keeps
//! the first `n_neighbors - 1`; kyto's blocked search must give the same graph bit for
//! bit (indices and `float32` distances), for both metrics, on random embeddings with
//! duplicates, zero rows and ties, on sizes crossing the query blocks (64) and point
//! tiles (1024), and on 1 and 4 threads.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use kyto_exec::{NeighborGraph, SUM_LANES, combine_lanes, nearest_neighbors};
use kyto_plan::{Metric, NeighborSearch, Neighbors};
use proptest::prelude::*;

fn params(n_neighbors: u64, n_components: Option<u64>, metric: Metric) -> Neighbors {
    Neighbors {
        embedding: "X_pca".to_owned(),
        n_components,
        n_neighbors,
        metric,
        search: NeighborSearch::Exact,
        seed: 0,
        key: None,
    }
}

/// The lane sum the kernels compute, one value at a time.
fn lane_sum(terms: impl Iterator<Item = f64>) -> f64 {
    let mut lanes = [0.0; SUM_LANES];
    for (index, term) in terms.enumerate() {
        lanes[index % SUM_LANES] += term;
    }
    combine_lanes(lanes)
}

/// Brute force: (indptr, indices, distances) with columns increasing per row.
fn brute_force(
    embedding: &[f32],
    width: usize,
    used: usize,
    n_neighbors: usize,
    metric: Metric,
) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let rows: Vec<&[f32]> = embedding
        .chunks_exact(width)
        .map(|row| &row[..used])
        .collect();
    let norm =
        |row: &[f32]| lane_sum(row.iter().map(|&value| f64::from(value) * f64::from(value))).sqrt();
    let key = |left: &[f32], right: &[f32]| match metric {
        Metric::Euclidean => lane_sum(left.iter().zip(right).map(|(&a, &b)| {
            let difference = f64::from(a) - f64::from(b);
            difference * difference
        })),
        Metric::Cosine => {
            let (left_norm, right_norm) = (norm(left), norm(right));
            if left_norm == 0.0 || right_norm == 0.0 {
                1.0
            } else {
                let dot = lane_sum(
                    left.iter()
                        .zip(right)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b)),
                );
                (1.0 - dot / (left_norm * right_norm)).clamp(0.0, 2.0)
            }
        }
    };
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut distances = Vec::new();
    for (query, query_row) in rows.iter().enumerate() {
        let mut others: Vec<(f64, u32)> = rows
            .iter()
            .enumerate()
            .filter(|&(row, _)| row != query)
            .map(|(row, other)| (key(query_row, other), row as u32))
            .collect();
        others.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
        others.truncate(n_neighbors - 1);
        others.sort_by_key(|entry| entry.1);
        for (key, row) in others {
            indices.push(row);
            distances.push(match metric {
                Metric::Euclidean => key.sqrt() as f32,
                Metric::Cosine => key as f32,
            });
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, distances)
}

fn assert_brute_force(graph: &NeighborGraph, expected: &(Vec<u64>, Vec<u32>, Vec<f32>)) {
    assert_eq!(graph.indptr(), expected.0);
    assert_eq!(graph.indices(), expected.1);
    let bits = |values: &[f32]| {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(graph.distances()), bits(&expected.2));
}

fn in_pool<T: Send>(threads: usize, work: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(work)
}

/// A seeded embedding: values on a coarse grid (so ties and duplicates happen), some
/// rows copied, some rows zero.
fn embedding(n: usize, width: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    };
    let mut values: Vec<f32> = (0..n * width)
        .map(|_| (next() % 41) as f32 / 4.0 - 5.0)
        .collect();
    for row in 0..n {
        match next() % 16 {
            0 if row > 0 => {
                let from = (next() as usize) % row;
                values.copy_within(from * width..(from + 1) * width, row * width);
            }
            1 => values[row * width..(row + 1) * width].fill(0.0),
            _ => {}
        }
    }
    values
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

    #[test]
    fn exact_equals_brute_force(
        n in 2_usize..160,
        width in 1_usize..20,
        used_less in 0_usize..3,
        n_neighbors in 2_usize..20,
        cosine in any::<bool>(),
        seed in any::<u64>(),
    ) {
        let n_neighbors = n_neighbors.min(n);
        let used = width.saturating_sub(used_less).max(1);
        let metric = if cosine { Metric::Cosine } else { Metric::Euclidean };
        let values = embedding(n, width, seed);
        let graph = nearest_neighbors(
            &values,
            width,
            &params(n_neighbors as u64, Some(used as u64), metric),
        )
        .unwrap();
        assert_brute_force(&graph, &brute_force(&values, width, used, n_neighbors, metric));
    }
}

#[test]
fn exact_equals_brute_force_across_blocks_tiles_and_threads() {
    // 2100 rows: two point tiles and a partial one, 33 query blocks, the last partial.
    let (n, width) = (2100, 50);
    let values = embedding(n, width, 11);
    for metric in [Metric::Euclidean, Metric::Cosine] {
        let expected = brute_force(&values, width, 30, 15, metric);
        for threads in [1, 4] {
            let graph = in_pool(threads, || {
                nearest_neighbors(&values, width, &params(15, Some(30), metric))
            })
            .unwrap();
            assert_brute_force(&graph, &expected);
            assert_eq!(graph.n_cells(), n);
            assert_eq!(graph.n_neighbors(), 15);
        }
    }
}

#[test]
fn every_other_cell_when_n_neighbors_is_n_cells() {
    let values = embedding(5, 3, 2);
    let graph = nearest_neighbors(&values, 3, &params(5, None, Metric::Euclidean)).unwrap();
    assert_eq!(graph.indptr(), [0, 4, 8, 12, 16, 20]);
    assert_eq!(
        graph.indices(),
        [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]
    );
}

#[test]
fn approximate_on_tiny_data_finds_every_neighbour() {
    // Few rows: the index finds them all, or the row falls back to exact search.
    let values = embedding(40, 4, 5);
    let exact = nearest_neighbors(&values, 4, &params(6, None, Metric::Euclidean)).unwrap();
    for metric in [Metric::Euclidean, Metric::Cosine] {
        let mut approximate = params(6, None, metric);
        approximate.search = NeighborSearch::Approximate;
        let graph = nearest_neighbors(&values, 4, &approximate).unwrap();
        assert_eq!(graph.indptr(), exact.indptr());
        if metric == Metric::Euclidean {
            assert_eq!(graph, exact);
        }
    }
    let mut two = params(2, None, Metric::Euclidean);
    two.search = NeighborSearch::Approximate;
    let graph = nearest_neighbors(&[0.0, 1.0], 1, &two).unwrap();
    assert_eq!(graph.indices(), [1, 0]);
}

#[test]
fn errors_are_coded() {
    let values = [0.0_f32; 6];
    let error = nearest_neighbors(&values, 2, &params(4, None, Metric::Euclidean)).unwrap_err();
    assert_eq!(error.code(), "KY-E215");
    let message = error.to_string();
    assert!(
        message.contains("n_neighbors") && message.contains("3 cells reach neighbors"),
        "{message}"
    );
    let error = nearest_neighbors(
        &[1.0, f32::INFINITY],
        1,
        &params(2, None, Metric::Euclidean),
    )
    .unwrap_err();
    assert_eq!(error.code(), "KY-E215");
    assert!(error.to_string().contains("not finite"));
}
