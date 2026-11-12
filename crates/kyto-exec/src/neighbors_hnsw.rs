//! Approximate nearest neighbours through an HNSW index (`hnsw_rs`, task 6.6).
//!
//! Does: insert every row into an HNSW graph (Malkov and Yashunin) one at a time, in an
//! order shuffled by the seed (SplitMix64, integer arithmetic only), then ask the index
//! for [`CANDIDATES_PER_NEIGHBOR`] times as many rows as each row needs and rank those
//! candidates by their exact distance (the kernels of `neighbors.rs`), so every
//! distance written is exact and only the choice of neighbours is approximate. Cosine
//! search indexes the rows scaled to unit length under the Euclidean distance (the
//! same order for nonzero rows). Insertion runs on one thread: `hnsw_rs` draws each
//! row's level from its own fixed-seed generator, so a fixed insertion order gives the
//! same index on every run; queries run in parallel and read it only (R11). A row for
//! which the index finds too few candidates is searched exactly.
//! Does not: guarantee the exact neighbours (recall on pbmc3k is measured by
//! `tests/it/neighbors_parity.rs`), or promise the same index across platforms
//! (`hnsw_rs` computes its distances and levels with the platform's `f32` and `ln`).

use std::borrow::Cow;

use hnsw_rs::prelude::{DistL2, Hnsw};
use kyto_error::KytoError;
use kyto_plan::Metric;
use rayon::prelude::*;

use crate::neighbors::{Nearest, QUERY_BLOCK, Space};
use crate::pca::mix;
use crate::run_control::RunControl;

/// Links per row in the index's layers (HNSW's `M`; layer 0 keeps twice as many).
pub const HNSW_MAX_CONNECTIONS: usize = 24;
/// The candidate list while inserting (`ef_construction`).
pub const HNSW_EF_CONSTRUCTION: usize = 200;
/// The candidate list while searching (`ef`), at least the candidates asked for.
pub const HNSW_EF_SEARCH: usize = 128;
/// Candidates ranked exactly per neighbour wanted (the cell itself included).
pub const CANDIDATES_PER_NEIGHBOR: usize = 2;
/// Layers of the index (`hnsw_rs`'s maximum).
const MAX_LAYERS: usize = 16;
/// Rows inserted between two cancellation checks.
const INSERT_CHECK: usize = 1024;

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
    let index = Hnsw::<f32, DistL2>::new(
        HNSW_MAX_CONNECTIONS,
        n,
        MAX_LAYERS,
        HNSW_EF_CONSTRUCTION,
        DistL2 {},
    );
    for (count, row) in insertion_order(n, seed).into_iter().enumerate() {
        if count % INSERT_CHECK == 0 {
            control.check()?;
        }
        index.insert_slice((&indexed[row * width..(row + 1) * width], row));
    }
    let candidates = (CANDIDATES_PER_NEIGHBOR * (k + 1)).min(n);
    let ef = HNSW_EF_SEARCH.max(candidates);
    let mut nearest = vec![(0.0, 0_u32); n * k];
    nearest
        .par_chunks_mut(QUERY_BLOCK * k)
        .enumerate()
        .try_for_each(|(block, out)| {
            control.check()?;
            let start = block * QUERY_BLOCK;
            let mut rows = Vec::with_capacity(candidates);
            let mut key = [0.0];
            for (query, out) in (start..).zip(out.chunks_exact_mut(k)) {
                let query_row = &indexed[query * width..(query + 1) * width];
                rows.clear();
                rows.extend(
                    index
                        .search(query_row, candidates, ef)
                        .iter()
                        .map(|neighbour| neighbour.d_id)
                        .filter(|&row| row != query && row < n),
                );
                rows.sort_unstable();
                rows.dedup();
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
        })?;
    Ok(nearest)
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
}
