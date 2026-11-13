//! UMAP's fuzzy graph weights on a neighbour graph (task 6.7): scanpy's
//! `connectivities` (`pp.neighbors(method="umap")`, which calls umap-learn's
//! `fuzzy_simplicial_set` with `set_op_mix_ratio=1` and `local_connectivity=1`).
//!
//! Does: for each row of a [`NeighborGraph`], as scanpy hands it to umap (the cell
//! first at distance 0, then its neighbours ranked by (distance, row position)), find
//! `rho`, the smallest distance above 0, and `sigma` by umap's 64-step bisection
//! (`smooth_knn_dist`) so that the row's memberships `exp(-max(0, d - rho) / sigma)`
//! sum to `log2(n_neighbors)` within 1e-5, with sigma at least 1e-3 of the row's mean
//! distance (of all distances when `rho` is 0); weight each entry by that membership
//! (1 when `d <= rho` or `sigma` is 0, 0 for an entry naming the row itself); and join
//! the directed weights by the fuzzy union `w + w^T - w o w^T`, dropping zeros. Every
//! step keeps the precision numba really gives umap (an `f64` bisection and means,
//! `float32` weights and union), so on the fixtures the weights equal scanpy's bit for
//! bit. Rows are weighted in parallel, each alone; the
//! union and the global mean are sequential in row order: the same graph on any
//! thread count (R11).
//! Does not: take umap's other parameters (`set_op_mix_ratio`, `local_connectivity`,
//! `bandwidth`), scanpy's `gauss` or `jaccard` methods, or write the graph (5.11).

use rayon::prelude::*;

use crate::neighbors::NeighborGraph;

/// umap's bisection steps for `sigma` (`smooth_knn_dist`'s `n_iter`).
pub const SMOOTH_K_ITERATIONS: usize = 64;
/// umap's tolerance on the membership sum (`SMOOTH_K_TOLERANCE`).
pub const SMOOTH_K_TOLERANCE: f64 = 1e-5;
/// The smallest `sigma` as a share of the mean distance (umap's `MIN_K_DIST_SCALE`).
pub const MIN_K_DIST_SCALE: f64 = 1e-3;

/// A weighted cell graph in CSR, symmetric: column indices increasing in each row,
/// `float32` weights in `(0, 1]`, no stored zeros.
#[derive(Debug, Clone, PartialEq)]
pub struct ConnectivityGraph {
    indptr: Vec<u64>,
    indices: Vec<u32>,
    weights: Vec<f32>,
}

impl ConnectivityGraph {
    /// The number of rows (and columns).
    pub fn n_cells(&self) -> usize {
        self.indptr.len() - 1
    }

    /// The row pointer: `n_cells() + 1` offsets into [`ConnectivityGraph::indices`].
    pub fn indptr(&self) -> &[u64] {
        &self.indptr
    }

    /// Each row's connected cells (row positions), increasing within a row.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// The weight of each edge, as [`ConnectivityGraph::indices`] orders them.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
}

/// scanpy's `connectivities` of `graph` (UMAP's fuzzy simplicial set), on the current
/// rayon pool.
pub fn connectivities(graph: &NeighborGraph) -> ConnectivityGraph {
    let n_cells = graph.n_cells();
    let n_neighbors = graph.n_neighbors();
    let k = n_neighbors - 1;
    // Each row's distances ranked as scanpy passes them (the cell's own 0 left out:
    // it never adds to a sum or the bisection).
    let mut ranked = vec![0.0_f32; n_cells * k];
    ranked
        .par_chunks_mut(k)
        .zip(graph.distances().par_chunks(k))
        .zip(graph.indices().par_chunks(k))
        .for_each(|((sorted, distances), indices)| {
            let mut entries: Vec<(f32, u32)> = distances
                .iter()
                .copied()
                .zip(indices.iter().copied())
                .collect();
            entries.sort_unstable_by(|left, right| {
                left.0.total_cmp(&right.0).then(left.1.cmp(&right.1))
            });
            for (slot, (distance, _)) in sorted.iter_mut().zip(entries) {
                *slot = distance;
            }
        });
    let target = (n_neighbors as f64).log2();
    let mut scales: Vec<(f32, f32)> = ranked
        .par_chunks(k)
        .map(|row| smooth_knn_distance(row, n_neighbors, target))
        .collect();
    // A row of zero distances takes its floor from the mean of every distance.
    if scales.iter().any(|&(_, rho)| rho == 0.0) {
        let floor = MIN_K_DIST_SCALE * mean(&ranked, n_cells * n_neighbors);
        for (sigma, rho) in &mut scales {
            if *rho == 0.0 && f64::from(*sigma) < floor {
                *sigma = floor as f32;
            }
        }
    }
    let mut memberships = vec![0.0_f32; n_cells * k];
    memberships
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(row, weights)| {
            let (sigma, rho) = scales[row];
            let entries = row * k..(row + 1) * k;
            let columns = &graph.indices()[entries.clone()];
            let distances = &graph.distances()[entries];
            for ((weight, &column), &distance) in weights.iter_mut().zip(columns).zip(distances) {
                *weight = membership(row, column, distance, sigma, rho);
            }
        });
    fuzzy_union(n_cells, k, graph.indices(), &memberships)
}

/// umap's `smooth_knn_dist` for one row of ranked distances (the cell's own 0 left
/// out): `(sigma, rho)`, with sigma's floor from the row's mean applied (the global
/// floor for `rho` 0 is the caller's).
fn smooth_knn_distance(ranked: &[f32], n_neighbors: usize, target: f64) -> (f32, f32) {
    // local_connectivity 1: the first distance above 0 (umap's `non_zero_dists[0]`).
    let rho = ranked.iter().copied().find(|&d| d > 0.0).unwrap_or(0.0);
    // umap declares these `float32` (numba `locals`), but inside its `prange` body numba
    // types them `f64`: only that reproduces its sigmas (every pbmc3k row).
    let ceiling = f64::from(f32::MAX);
    let (mut lo, mut hi, mut mid) = (0.0_f64, ceiling, 1.0_f64);
    for _ in 0..SMOOTH_K_ITERATIONS {
        let mut sum = 0.0_f64;
        for &distance in ranked {
            // Two `float32` array values: a `float32` difference.
            let d = f64::from(distance - rho);
            sum += if d > 0.0 { (-(d / mid)).exp() } else { 1.0 };
        }
        if (sum - target).abs() < SMOOTH_K_TOLERANCE {
            break;
        }
        if sum > target {
            hi = mid;
            mid = (lo + hi) / 2.0;
        } else {
            lo = mid;
            mid = if hi >= ceiling {
                mid * 2.0
            } else {
                (lo + hi) / 2.0
            };
        }
    }
    let mut sigma = mid as f32;
    if rho > 0.0 {
        let floor = MIN_K_DIST_SCALE * mean(ranked, n_neighbors);
        if f64::from(sigma) < floor {
            sigma = floor as f32;
        }
    }
    (sigma, rho)
}

/// umap's `np.mean` of `float32` values: a `float32` sum over `count` values (`values`
/// and zeros), divided in `f64` and not rounded back (under `parallel=True` numba
/// turns it into an `f64` reduction; only that reproduces umap's floors).
fn mean(values: &[f32], count: usize) -> f64 {
    let sum = values.iter().fold(0.0_f32, |sum, &value| sum + value);
    f64::from(sum) / count as f64
}

/// umap's `compute_membership_strengths` for one entry.
fn membership(row: usize, column: u32, distance: f32, sigma: f32, rho: f32) -> f32 {
    if column as usize == row {
        0.0
    } else if distance - rho <= 0.0 || sigma == 0.0 {
        1.0
    } else {
        (-((distance - rho) / sigma)).exp()
    }
}

/// `w + w^T - w o w^T` in `float32` as scipy computes it (`(w + w^T) - w o w^T`), zeros
/// dropped; `indices` hold `k` increasing columns a row.
fn fuzzy_union(n_cells: usize, k: usize, indices: &[u32], weights: &[f32]) -> ConnectivityGraph {
    // The transpose by counting: its rows come out with increasing columns.
    let mut transposed_indptr = vec![0_usize; n_cells + 1];
    for &column in indices {
        transposed_indptr[column as usize + 1] += 1;
    }
    for row in 0..n_cells {
        transposed_indptr[row + 1] += transposed_indptr[row];
    }
    let mut cursor = transposed_indptr.clone();
    let mut transposed = vec![(0_u32, 0.0_f32); indices.len()];
    for (entry, (&column, &weight)) in indices.iter().zip(weights).enumerate() {
        let slot = &mut cursor[column as usize];
        transposed[*slot] = ((entry / k) as u32, weight);
        *slot += 1;
    }
    let mut indptr = Vec::with_capacity(n_cells + 1);
    let mut union_indices = Vec::with_capacity(2 * indices.len());
    let mut union_weights = Vec::with_capacity(2 * indices.len());
    indptr.push(0);
    for row in 0..n_cells {
        let forward = (row * k..(row + 1) * k).map(|entry| (indices[entry], weights[entry]));
        let backward = &transposed[transposed_indptr[row]..transposed_indptr[row + 1]];
        let mut forward = forward.peekable();
        let mut backward = backward.iter().copied().peekable();
        loop {
            let (column, w, w_transposed) = match (forward.peek(), backward.peek()) {
                (None, None) => break,
                (Some(&(column, w)), None) => {
                    forward.next();
                    (column, w, 0.0)
                }
                (None, Some(&(column, w_transposed))) => {
                    backward.next();
                    (column, 0.0, w_transposed)
                }
                (Some(&(left, w)), Some(&(right, w_transposed))) => {
                    if left < right {
                        forward.next();
                        (left, w, 0.0)
                    } else if right < left {
                        backward.next();
                        (right, 0.0, w_transposed)
                    } else {
                        forward.next();
                        backward.next();
                        (left, w, w_transposed)
                    }
                }
            };
            let weight = (w + w_transposed) - w * w_transposed;
            if weight != 0.0 {
                union_indices.push(column);
                union_weights.push(weight);
            }
        }
        indptr.push(union_indices.len() as u64);
    }
    ConnectivityGraph {
        indptr,
        indices: union_indices,
        weights: union_weights,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(n_neighbors: usize, rows: &[&[(u32, f32)]]) -> NeighborGraph {
        let indptr = (0..=rows.len())
            .map(|row| (row * (n_neighbors - 1)) as u64)
            .collect();
        let indices = rows
            .iter()
            .flat_map(|row| row.iter().map(|e| e.0))
            .collect();
        let distances = rows
            .iter()
            .flat_map(|row| row.iter().map(|e| e.1))
            .collect();
        NeighborGraph::from_csr(n_neighbors, indptr, indices, distances).unwrap()
    }

    #[test]
    fn nearest_neighbour_weighs_one_and_the_union_is_symmetric() {
        // Points at 0, 1, 3 on a line, 2 neighbours each (n_neighbors 3).
        let line = graph(
            3,
            &[
                &[(1, 1.0), (2, 3.0)],
                &[(0, 1.0), (2, 2.0)],
                &[(0, 3.0), (1, 2.0)],
            ],
        );
        let result = connectivities(&line);
        assert_eq!(result.indptr(), [0, 2, 4, 6]);
        assert_eq!(result.indices(), [1, 2, 0, 2, 0, 1]);
        let weight = |row: usize, column: u32| {
            let range = result.indptr()[row] as usize..result.indptr()[row + 1] as usize;
            let position = result.indices()[range.clone()]
                .binary_search(&column)
                .unwrap();
            result.weights()[range][position]
        };
        for row in 0..3_u32 {
            for column in 0..3_u32 {
                if row != column {
                    assert_eq!(weight(row as usize, column), weight(column as usize, row));
                }
            }
        }
        // Each cell's nearest neighbour has membership 1, so the union is 1 (up to the
        // float32 rounding of `(w + 1) - w`, as scipy rounds it, where w < 1).
        assert_eq!(weight(0, 1), 1.0);
        assert!(weight(1, 2) >= 1.0 - f32::EPSILON);
        // 0 -> 2 is the second neighbour of 0 and of 2: below 1, above 0.
        assert!(weight(0, 2) > 0.0 && weight(0, 2) < 1.0);
    }

    #[test]
    fn memberships_sum_to_log2_k() {
        let distances = [0.5_f32, 0.9, 1.4, 2.0, 2.7];
        let target = 6.0_f64.log2();
        let (sigma, rho) = smooth_knn_distance(&distances, 6, target);
        assert_eq!(rho, 0.5);
        let sum: f64 = distances
            .iter()
            .map(|&d| f64::from(membership(0, 1, d, sigma, rho)))
            .sum();
        assert!((sum - target).abs() < 1e-4, "{sum} vs {target}");
    }

    #[test]
    fn sigma_floor_rounds_once_as_umap() {
        // Four duplicates, then five neighbours, n_neighbors 10: the bisected sigma is
        // below 1e-3 of the row mean, so the floor decides it. umap-learn 0.5.12 gives
        // bits 983688805 (a mean rounded to float32 first gives 983688804).
        let ranked = [
            0.0,
            0.0,
            0.0,
            0.0,
            2.400_668_1,
            2.417_103,
            2.460_597_3,
            2.479_911_6,
            2.593_708_5,
        ];
        let (sigma, rho) = smooth_knn_distance(&ranked, 10, 10.0_f64.log2());
        assert_eq!(rho, 2.400_668_1);
        assert_eq!(sigma.to_bits(), 983_688_805);
    }

    #[test]
    fn duplicates_weigh_one_and_the_cell_itself_zero() {
        // Cells 0 and 1 are duplicates; row 2 lists itself (as scanpy's graph can).
        let duplicates = graph(
            3,
            &[
                &[(1, 0.0), (2, 1.0)],
                &[(0, 0.0), (2, 1.0)],
                &[(1, 1.0), (2, 0.0)],
            ],
        );
        let result = connectivities(&duplicates);
        assert_eq!(result.indptr(), [0, 2, 4, 6]);
        assert_eq!(result.indices(), [1, 2, 0, 2, 0, 1]);
        assert_eq!(result.weights()[0], 1.0);
        assert!(result.weights().iter().all(|&w| w > 0.0 && w <= 1.0));
    }

    #[test]
    fn rows_of_zero_distances_take_the_global_floor() {
        // Three identical cells: every distance 0, so rho 0 and sigma at its floor of 0
        // (the mean of all distances); every membership is 1.
        let same = graph(
            3,
            &[
                &[(1, 0.0), (2, 0.0)],
                &[(0, 0.0), (2, 0.0)],
                &[(0, 0.0), (1, 0.0)],
            ],
        );
        let result = connectivities(&same);
        assert_eq!(result.weights(), [1.0; 6]);
    }

    #[test]
    fn underflowing_weights_are_dropped_unless_the_other_side_holds_them() {
        // Rows 0 and 3 each have two neighbours at rho (memberships 1 + 1 = log2 4)
        // and each other very far: that membership underflows to 0 on both sides, so
        // the edge 0-3 is dropped; 1-3 and 2-3 stay through rows 1 and 2.
        let far = graph(
            4,
            &[
                &[(1, 1.0), (2, 1.0), (3, 1e6)],
                &[(0, 1.0), (2, 1.0), (3, 1.0)],
                &[(0, 1.0), (1, 1.0), (3, 1.0)],
                &[(0, 1e6), (1, 1.0), (2, 1.0)],
            ],
        );
        let result = connectivities(&far);
        assert_eq!(result.indptr(), [0, 2, 5, 8, 10]);
        assert_eq!(result.indices(), [1, 2, 0, 2, 3, 0, 1, 3, 1, 2]);
    }
}
