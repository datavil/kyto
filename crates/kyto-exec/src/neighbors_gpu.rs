//! Exact neighbours with a GPU (task 9.8): the GPU proposes candidates, the host ranks
//! them with the CPU's distances, so the graph equals the CPU's exact search.
//!
//! Does: for every row, kyto-gpu's tiled distances and top-k
//! ([`DeviceKernels::nearest_keys`]) give its [`candidates`] nearest other rows by an
//! `f32` key (the squared Euclidean distance, or the cosine distance from the GEMM's dot
//! products). The host computes each candidate's key as the CPU's exact search does
//! (`Space::keys`: `f64`, the SIMD kernels' lane order) and ranks them by (key, row).
//! The `f32` keys are off the exact ones by at most a stated bound
//! ([`euclidean_floor`], [`COSINE_SLACK`]); the farthest candidate's `f32` key bounds
//! every other row's exact key from below. When that floor lies above the `k`-th
//! nearest candidate's exact key, no other row can rank among the `k` nearest, and the
//! row's result is the CPU's. Otherwise (or if the farthest key is not finite) the row
//! is searched again on the CPU against every row. So the result is the CPU's exact
//! search, bit for bit, on any thread count, whatever the GPU's rounding (R11).
//! Does not: approximate search (HNSW stays on the CPU: no GPU kernel, so `auto` keeps
//! the step there), or weight the graph.

use kyto_error::KytoError;
use kyto_gpu::{DeviceKernels, PairwiseKey};
use kyto_plan::Metric;
use rayon::prelude::*;

use crate::neighbors::{Nearest, QUERY_BLOCK, Space, search_row};
use crate::run_control::RunControl;

/// The operation the GPU's errors name.
const OPERATION: &str = "neighbors";

/// Queries per call to the GPU (one cancellation check each).
const QUERIES_PER_CALL: usize = 4096;

/// `f32`'s unit roundoff, `2^-24`.
const UNIT_ROUNDOFF: f64 = 1.0 / 16_777_216.0;

/// The slack of a cosine key computed on the GPU: `1 - dot / (|q| |p|)` with an `f32`
/// dot product (error at most `width u |q| |p|`), `f32` norms and division (a few `u`,
/// more under fast math), as an absolute bound per component of the width, doubled.
pub(crate) const COSINE_SLACK: f64 = 2.0 * UNIT_ROUNDOFF;

/// The candidates kept per row: twice `k` and 16 more, at most every other row.
pub(crate) fn candidates(k: usize, n_rows: usize) -> usize {
    (2 * k + 16).min(n_rows - 1)
}

/// The least exact squared distance a row whose `f32` key on the GPU is at least
/// `key` can have: a sum of `width` squared `f32` differences has a relative error of
/// at most `(width + 2) u` (every term non-negative), doubled here, less what flushing
/// subnormals to zero (Metal) can take away.
pub(crate) fn euclidean_floor(key: f32, width: usize) -> f64 {
    let relative = 2.0 * (width as f64 + 4.0) * UNIT_ROUNDOFF;
    let flushed = (width as f64 + 1.0) * f64::from(f32::MIN_POSITIVE);
    f64::from(key) / (1.0 + relative) - flushed
}

/// Whether the cosine norms keep the GPU's keys within [`COSINE_SLACK`]: every norm 0
/// or in `[1e-15, 1e15]` (no `f32` underflow or overflow in the dot products).
fn cosine_norms_fit(space: &Space<'_>) -> bool {
    (0..space.n_rows()).all(|row| {
        let norm = space.norm(row);
        norm == 0.0 || (1e-15..=1e15).contains(&norm)
    })
}

/// Each row's `k` nearest other rows, ranked, `k` entries a row: the CPU's exact search
/// (`neighbors::exact`), with candidates from `gpu` (module doc).
///
/// # Errors
/// `KY-E503` if the run was cancelled; `KY-E505` if the GPU fails.
pub(crate) fn exact(
    space: &Space<'_>,
    k: usize,
    gpu: &DeviceKernels,
    control: &RunControl,
) -> Result<Vec<(f64, u32)>, KytoError> {
    let n = space.n_rows();
    let width = space.width();
    let m = candidates(k, n);
    let norms: Vec<f32>;
    let key = match space.metric() {
        Metric::Euclidean => PairwiseKey::SquaredEuclidean,
        Metric::Cosine => {
            if !cosine_norms_fit(space) {
                return search_rows_on_cpu(space, k, 0..n, control);
            }
            norms = (0..n).map(|row| space.norm(row) as f32).collect();
            PairwiseKey::Cosine {
                query_norms: &norms,
                point_norms: &norms,
            }
        }
    };
    let mut nearest = vec![(0.0, 0_u32); n * k];
    for first in (0..n).step_by(QUERIES_PER_CALL) {
        control.check()?;
        let last = (first + QUERIES_PER_CALL).min(n);
        let skip: Vec<u32> = (first as u32..last as u32).collect();
        let query_key = match key {
            PairwiseKey::Cosine { point_norms, .. } => PairwiseKey::Cosine {
                query_norms: &point_norms[first..last],
                point_norms,
            },
            PairwiseKey::SquaredEuclidean => PairwiseKey::SquaredEuclidean,
        };
        let (keys, points) = gpu.nearest_keys(
            &space.rows()[first * width..last * width],
            space.rows(),
            width,
            query_key,
            &skip,
            m,
            OPERATION,
        )?;
        nearest[first * k..last * k]
            .par_chunks_mut(k)
            .zip(keys.par_chunks(m).zip(points.par_chunks(m)))
            .enumerate()
            .for_each(|(offset, (out, (keys, points)))| {
                let query = first + offset;
                match rank_candidates(space, k, query, keys, points) {
                    Some(found) => out.copy_from_slice(found.entries()),
                    None => out.copy_from_slice(search_row(space, k, query).entries()),
                }
            });
    }
    Ok(nearest)
}

/// The `k` nearest of `query`'s candidates (`points`, with their GPU `keys`, smallest
/// first) by the CPU's keys, if they are shown to be its `k` nearest rows of all
/// (module doc); `None` otherwise.
fn rank_candidates(
    space: &Space<'_>,
    k: usize,
    query: usize,
    keys: &[f32],
    points: &[u32],
) -> Option<Nearest> {
    let mut found = Nearest::new(k);
    let mut key = [0.0];
    for &point in points {
        space.keys(query, point as usize, &mut key);
        found.offer(key[0], point);
    }
    if points.len() == space.n_rows() - 1 {
        // Every other row is a candidate.
        return Some(found);
    }
    let farthest = *keys.last()?;
    if !farthest.is_finite() {
        return None;
    }
    let floor = match space.metric() {
        Metric::Euclidean => euclidean_floor(farthest, space.width()),
        Metric::Cosine => f64::from(farthest) - COSINE_SLACK * (space.width() as f64 + 16.0),
    };
    let kth = found.entries().last()?.0;
    (floor > kth).then_some(found)
}

/// The CPU's exact search for the rows `queries` (all of them: `k` entries a row).
fn search_rows_on_cpu(
    space: &Space<'_>,
    k: usize,
    queries: std::ops::Range<usize>,
    control: &RunControl,
) -> Result<Vec<(f64, u32)>, KytoError> {
    let first = queries.start;
    let mut nearest = vec![(0.0, 0_u32); queries.len() * k];
    nearest
        .par_chunks_mut(QUERY_BLOCK * k)
        .enumerate()
        .try_for_each(|(block, out)| {
            control.check()?;
            for (offset, out) in out.chunks_exact_mut(k).enumerate() {
                let query = first + block * QUERY_BLOCK + offset;
                out.copy_from_slice(search_row(space, k, query).entries());
            }
            Ok(())
        })?;
    Ok(nearest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_euclidean_floor_lies_below_the_key_by_its_bound() {
        let floor = euclidean_floor(100.0, 50);
        assert!(floor < 100.0 && floor > 100.0 * (1.0 - 1e-5));
        assert!(euclidean_floor(0.0, 50) < 0.0);
    }

    #[test]
    fn candidates_are_twice_k_and_16_at_most_every_other_row() {
        assert_eq!(candidates(14, 2700), 44);
        assert_eq!(candidates(14, 20), 19);
    }
}
