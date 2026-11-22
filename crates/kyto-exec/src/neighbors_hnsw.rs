//! Approximate nearest neighbours through an HNSW index (tasks 6.6, 6.17).
//!
//! Does: build a hierarchical navigable small world graph (Malkov and Yashunin, "Efficient
//! and robust approximate nearest neighbor search using HNSW graphs", 2016/2018) over
//! the rows, then search layer 0 of it from every row's own node for its nearest rows,
//! and rank those candidates by their exact distance (the kernels of `neighbors.rs`), so
//! every distance written is exact and only the choice of neighbours is approximate.
//! Rows are inserted one at a time, in an order shuffled by the seed, each at a level
//! drawn from the seed (SplitMix64, integer arithmetic only, `P(level >= l) = M^-l`).
//! Insertion follows the paper: a greedy descent through the upper layers, a search of
//! [`HNSW_EF_CONSTRUCTION`] candidates in each layer the row joins, links chosen by the
//! paper's heuristic (its Algorithm 4: a candidate is kept unless it is nearer to an
//! already kept link than to the row), and the same heuristic again when a reverse
//! link overfills a node's list. Distances inside the index are `f64` squared Euclidean
//! distances from the SIMD kernels (the same bits on every platform), ordered by
//! (distance, row). Cosine search indexes the rows scaled to unit length (the same
//! order for nonzero rows). Insertion runs on one thread; queries run in parallel and
//! read the index only, so the graph is the same on any thread count (R11). A row whose
//! search reaches fewer rows than it needs is searched exactly.
//! Why it is written here (6.17): `hnsw_rs` 0.3.5 lost 2% of the cells of a 50k
//! atlas (recall 0, neighbours in another cluster). Its query moves at most one link
//! per upper layer instead of descending greedily, and it trims an overfull reverse
//! link list by dropping the farthest link, which drops exactly the links between
//! clusters that the heuristic kept; rows inserted from the wrong region were then left
//! with a few links into another cluster and none back. Every query here is a row of
//! the index, so it starts at its own node, which needs no descent at all.
//! Does not: guarantee the exact neighbours, nor the earlier-row tie rule when more
//! than `ef` rows lie at exactly the same distance (it holds among the rows reached;
//! bug hunt 6.17: 150 identical rows, `n_neighbors` 76); recall is measured by
//! `tests/it/neighbors_parity.rs` and `tests/it/neighbors_recall.rs`), or answer queries
//! for rows outside the index.

use std::borrow::Cow;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use kyto_error::KytoError;
use kyto_plan::Metric;
use rayon::prelude::*;

use crate::kernels::Kernels;
use crate::neighbors::{Nearest, QUERY_BLOCK, Space};
use crate::pca::mix;
use crate::run_control::RunControl;

/// Links per row in the index's upper layers (HNSW's `M`; layer 0 keeps twice as many).
pub const HNSW_MAX_CONNECTIONS: usize = 24;
/// The candidate list while inserting (`ef_construction`).
pub const HNSW_EF_CONSTRUCTION: usize = 200;
/// The candidate list while searching (`ef`), at least the neighbours asked for.
pub const HNSW_EF_SEARCH: usize = 128;
/// The highest level a row can be drawn at.
const MAX_LEVEL: usize = 15;
/// Rows inserted between two cancellation checks.
const INSERT_CHECK: usize = 1024;
/// Added to the seed for the level draws, so they are not the insertion order's stream.
const LEVEL_STREAM: u64 = 0x6A09_E667_F3BC_C908;

/// Each row's `k` nearest other rows among the index's candidates, ranked, `k` entries
/// a row.
pub(crate) fn approximate(
    space: &Space<'_>,
    k: usize,
    seed: u64,
    control: &RunControl,
) -> Result<Vec<(f64, u32)>, KytoError> {
    let n = space.n_rows();
    let width = space.width();
    let indexed: Cow<'_, [f32]> = match space.metric() {
        Metric::Euclidean => Cow::Borrowed(space.rows()),
        Metric::Cosine => Cow::Owned(
            (0..n)
                .flat_map(|row| {
                    let norm = space.norm(row);
                    space.row(row).iter().map(move |&value| {
                        if norm == 0.0 {
                            0.0
                        } else {
                            (f64::from(value) / norm) as f32
                        }
                    })
                })
                .collect(),
        ),
    };
    let index = Index::build(&indexed, width, seed, control)?;
    let ef = HNSW_EF_SEARCH.max(k + 1);
    let mut nearest = vec![(0.0, 0_u32); n * k];
    nearest
        .par_chunks_mut(QUERY_BLOCK * k)
        .enumerate()
        .try_for_each_init(
            || Visited::new(n),
            |visited, (block, out)| {
                control.check()?;
                let start = block * QUERY_BLOCK;
                let mut rows = Vec::with_capacity(ef);
                let mut key = [0.0];
                for (query, out) in (start..).zip(out.chunks_exact_mut(k)) {
                    let query_id = query as u32;
                    rows.clear();
                    rows.extend(
                        index
                            .search_layer(index.row(query_id), &[query_id], ef, 0, visited)
                            .into_iter()
                            .map(|candidate| candidate.row as usize)
                            .filter(|&row| row != query),
                    );
                    if rows.len() < k {
                        rows.clear();
                        rows.extend((0..n).filter(|&row| row != query));
                    }
                    let mut found = Nearest::new(k);
                    for &row in &rows {
                        space.keys(query, row, &mut key);
                        found.offer(key[0], row as u32);
                    }
                    out.copy_from_slice(found.entries());
                }
                Ok(())
            },
        )?;
    Ok(nearest)
}

/// A row of the index at its squared distance from a query, ordered by (distance, row).
#[derive(Clone, Copy, Debug)]
struct Candidate {
    distance: f64,
    row: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.row.cmp(&other.row))
    }
}

/// The rows a search has seen: a stamp a row, a new stamp a search (no clearing).
struct Visited {
    stamps: Vec<u32>,
    current: u32,
}

impl Visited {
    fn new(n: usize) -> Self {
        Self {
            stamps: vec![0; n],
            current: 0,
        }
    }

    /// Forget every row seen.
    fn clear(&mut self) {
        if self.current == u32::MAX {
            self.stamps.fill(0);
            self.current = 0;
        }
        self.current += 1;
    }

    /// Mark `row` seen; false if it was already.
    fn insert(&mut self, row: u32) -> bool {
        let stamp = &mut self.stamps[row as usize];
        if *stamp == self.current {
            return false;
        }
        *stamp = self.current;
        true
    }
}

/// The HNSW graph: each row's links on each layer it is drawn into.
struct Index<'a> {
    rows: &'a [f32],
    width: usize,
    kernels: Kernels,
    /// `links[row][layer]`: the row's neighbours on that layer.
    links: Vec<Vec<Vec<u32>>>,
    entry: u32,
    top: usize,
}

impl<'a> Index<'a> {
    /// Insert every row of `rows` (`width` values a row), in the seed's order.
    fn build(
        rows: &'a [f32],
        width: usize,
        seed: u64,
        control: &RunControl,
    ) -> Result<Self, KytoError> {
        let n = rows.len() / width;
        let mut index = Self {
            rows,
            width,
            kernels: Kernels::detected(),
            links: vec![Vec::new(); n],
            entry: 0,
            top: 0,
        };
        let mut visited = Visited::new(n);
        let mut levels = Levels::new(seed);
        for (count, row) in insertion_order(n, seed).into_iter().enumerate() {
            if count % INSERT_CHECK == 0 {
                control.check()?;
            }
            index.insert(row as u32, levels.next(), count == 0, &mut visited);
        }
        Ok(index)
    }

    fn row(&self, row: u32) -> &'a [f32] {
        let start = row as usize * self.width;
        &self.rows[start..start + self.width]
    }

    /// The squared Euclidean distance between `query` and row `row`.
    fn distance(&self, query: &[f32], row: u32) -> f64 {
        let mut out = [0.0];
        self.kernels
            .squared_distances(query, self.row(row), &mut out);
        out[0]
    }

    /// The most links a row keeps on `layer`.
    fn max_links(layer: usize) -> usize {
        if layer == 0 {
            2 * HNSW_MAX_CONNECTIONS
        } else {
            HNSW_MAX_CONNECTIONS
        }
    }

    /// The paper's Algorithm 2: the `ef` nearest rows to `query` on `layer` reached from
    /// `entries`, nearest first.
    fn search_layer(
        &self,
        query: &[f32],
        entries: &[u32],
        ef: usize,
        layer: usize,
        visited: &mut Visited,
    ) -> Vec<Candidate> {
        visited.clear();
        let mut candidates = BinaryHeap::new();
        let mut found = BinaryHeap::with_capacity(ef + 1);
        for &row in entries {
            if visited.insert(row) {
                let entry = Candidate {
                    distance: self.distance(query, row),
                    row,
                };
                candidates.push(Reverse(entry));
                found.push(entry);
                if found.len() > ef {
                    found.pop();
                }
            }
        }
        while let Some(Reverse(nearest)) = candidates.pop() {
            if found.len() >= ef && found.peek().is_some_and(|worst| nearest > *worst) {
                break;
            }
            for &row in &self.links[nearest.row as usize][layer] {
                if !visited.insert(row) {
                    continue;
                }
                let next = Candidate {
                    distance: self.distance(query, row),
                    row,
                };
                if found.len() < ef || found.peek().is_some_and(|worst| next < *worst) {
                    candidates.push(Reverse(next));
                    found.push(next);
                    if found.len() > ef {
                        found.pop();
                    }
                }
            }
        }
        found.into_sorted_vec()
    }

    /// The paper's Algorithm 4 (no extension, no pruned fill): walk `candidates` nearest
    /// first and keep one unless it is nearer to a kept row than to the query, up to
    /// `max` rows.
    fn select(&self, candidates: &[Candidate], max: usize) -> Vec<u32> {
        let mut kept: Vec<u32> = Vec::with_capacity(max);
        for candidate in candidates {
            if kept.len() == max {
                break;
            }
            let row = self.row(candidate.row);
            if kept
                .iter()
                .all(|&other| self.distance(row, other) >= candidate.distance)
            {
                kept.push(candidate.row);
            }
        }
        kept
    }

    /// The paper's Algorithm 1: link `row` at `level` into the graph.
    fn insert(&mut self, row: u32, level: usize, first: bool, visited: &mut Visited) {
        self.links[row as usize] = vec![Vec::new(); level + 1];
        if first {
            self.entry = row;
            self.top = level;
            return;
        }
        let query = self.row(row);
        let mut entries = vec![self.entry];
        for layer in (level + 1..=self.top).rev() {
            let nearest = self.search_layer(query, &entries, 1, layer, visited);
            entries = vec![nearest[0].row];
        }
        for layer in (0..=level.min(self.top)).rev() {
            let found = self.search_layer(query, &entries, HNSW_EF_CONSTRUCTION, layer, visited);
            let max = Self::max_links(layer);
            let chosen = self.select(&found, max);
            for &neighbour in &chosen {
                let mut list = std::mem::take(&mut self.links[neighbour as usize][layer]);
                list.push(row);
                if list.len() > max {
                    let centre = self.row(neighbour);
                    let mut ranked: Vec<Candidate> = list
                        .iter()
                        .map(|&other| Candidate {
                            distance: self.distance(centre, other),
                            row: other,
                        })
                        .collect();
                    ranked.sort_unstable();
                    list = self.select(&ranked, max);
                }
                self.links[neighbour as usize][layer] = list;
            }
            self.links[row as usize][layer] = chosen;
            entries = found.iter().map(|candidate| candidate.row).collect();
        }
        if level > self.top {
            self.entry = row;
            self.top = level;
        }
    }
}

/// Each inserted row's level: `l` with probability `M^-l (1 - 1/M)`, at most
/// [`MAX_LEVEL`] (the paper's `floor(-ln(U) / ln(M))`, drawn in integers).
struct Levels {
    state: u64,
}

impl Levels {
    fn new(seed: u64) -> Self {
        Self {
            state: mix(seed.wrapping_add(LEVEL_STREAM)),
        }
    }

    fn next(&mut self) -> usize {
        let threshold = u64::MAX / HNSW_MAX_CONNECTIONS as u64;
        let mut level = 0;
        loop {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            if level == MAX_LEVEL || mix(self.state) >= threshold {
                return level;
            }
            level += 1;
        }
    }
}

/// `0..n` shuffled by `seed` (Fisher-Yates on SplitMix64).
fn insertion_order(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut state = mix(seed);
    for last in (1..n).rev() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let pick = (u128::from(mix(state)) * (last as u128 + 1)) >> 64;
        order.swap(last, pick as usize);
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insertion_order_is_a_seeded_permutation() {
        let order = insertion_order(100, 3);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
        assert_eq!(order, insertion_order(100, 3));
        assert_ne!(order, insertion_order(100, 4));
        assert!(insertion_order(0, 1).is_empty());
    }

    #[test]
    fn levels_follow_one_over_m_per_layer() {
        let mut levels = Levels::new(5);
        let draws: Vec<usize> = (0..200_000).map(|_| levels.next()).collect();
        let at_least = |level: usize| draws.iter().filter(|&&drawn| drawn >= level).count();
        // Expected 200_000 / 24 = 8333 and / 576 = 347.
        assert!((7_900..8_800).contains(&at_least(1)), "{}", at_least(1));
        assert!((280..420).contains(&at_least(2)), "{}", at_least(2));
        let mut again = Levels::new(5);
        assert!(draws.iter().all(|&drawn| drawn == again.next()));
    }

    #[test]
    fn select_keeps_one_link_per_direction() {
        // Rows on a line: 0 at 0, 1 at 1, 2 at 2, 3 at -1. From a query at 0 (row 0),
        // row 2 is nearer to row 1 than to the query, so only 1 and 3 are kept.
        let rows = [0.0_f32, 1.0, 2.0, -1.0];
        let index = Index {
            rows: &rows,
            width: 1,
            kernels: Kernels::detected(),
            links: vec![Vec::new(); 4],
            entry: 0,
            top: 0,
        };
        let candidates: Vec<Candidate> = [1_u32, 3, 2]
            .iter()
            .map(|&row| Candidate {
                distance: index.distance(&rows[..1], row),
                row,
            })
            .collect();
        assert_eq!(index.select(&candidates, 4), vec![1, 3]);
        assert_eq!(index.select(&candidates, 1), vec![1]);
    }

    #[test]
    fn every_row_is_its_own_nearest_and_the_graph_is_reached() {
        // A grid of 400 points: each row's layer-0 search from its own node finds itself
        // first and its 4 grid neighbours next.
        let rows: Vec<f32> = (0..400)
            .flat_map(|cell| [(cell % 20) as f32, (cell / 20) as f32])
            .collect();
        let index = Index::build(&rows, 2, 7, &RunControl::new()).unwrap();
        let mut visited = Visited::new(400);
        for row in 0..400_u32 {
            let found = index.search_layer(index.row(row), &[row], 16, 0, &mut visited);
            assert_eq!(found[0].row, row);
            assert_eq!(found[0].distance, 0.0);
            let (x, y) = (row % 20, row / 20);
            let grid =
                usize::from(x > 0) + usize::from(x < 19) + usize::from(y > 0) + usize::from(y < 19);
            assert!(
                found[1..=grid]
                    .iter()
                    .all(|candidate| candidate.distance == 1.0)
            );
        }
    }
}
