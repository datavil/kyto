//! Exact neighbours on the CPU through an `f32` matrix-product filter (task 12.4).
//!
//! Does: for each block of [`QUERY_ROWS`] query rows (in parallel) and each tile of
//! [`POINT_ROWS`] rows, an `f32` matrix product ([`f32_products`]) gives an approximate
//! key per pair: `|q|^2 + |p|^2 - 2 q.p` on rows centred on the mean (Euclidean), or
//! `1 - q.p` on rows scaled to unit length (cosine, clipped to `[0, 2]`). Every
//! approximate key is within a proven slack `E_q` of the exact key the brute-force
//! search computes (`Space::keys`, `f64`): [`Prepared`] states the bound. A query keeps
//! every row whose approximate key is at most its `k`-th smallest approximate key plus
//! `2 E_q`; that set holds every row whose exact key ranks among the `k` nearest,
//! including every row tied with the `k`-th (a true neighbour's exact key is at most the
//! `k`-th exact key, which is at most the `k`-th approximate key plus `E_q`). Those
//! candidates are then ranked with their exact keys by (key, row) in a [`Nearest`], as
//! the brute-force search ranks every row. So the graph is the brute-force search's, bit
//! for bit, whatever the matrix product's rounding, summation order or thread count
//! (R11). Two guards keep that true when an assumption fails: a candidate whose exact
//! key lies farther than `E_q` from its approximate key (a library computing below
//! `f32` precision) and a query whose candidates pass [`CANDIDATE_LIMIT`] (heavily tied
//! data) are searched again against every row ([`search_row`]); embeddings whose
//! squared norms could overflow `f32` are searched by brute force throughout.
//! Does not: approximate search (`neighbors_hnsw.rs`), the GPU path (`neighbors_gpu.rs`),
//! or the graph's CSR layout (`neighbors.rs`).

use kyto_error::KytoError;
use kyto_plan::Metric;
use rayon::prelude::*;

use crate::kernels::f32_products;
use crate::neighbors::{Nearest, Space, brute_force, search_row};
use crate::run_control::RunControl;

/// Query rows a parallel task filters together (one cancellation check each).
pub(crate) const QUERY_ROWS: usize = 128;
/// Rows of one matrix product with a block of queries (`QUERY_ROWS x POINT_ROWS`
/// `f32` keys, 512 KiB, stay in cache while the queries scan them).
pub(crate) const POINT_ROWS: usize = 1024;
/// Candidates a query may hold; past it the query is searched against every row.
pub(crate) const CANDIDATE_LIMIT: usize = 4096;

/// `f32`'s unit roundoff, `2^-24`.
const UNIT_ROUNDOFF: f64 = 1.0 / 16_777_216.0;
/// The largest row norm whose squared sums stay far from `f32` overflow.
const LARGEST_NORM: f64 = 1e15;

/// Each row's `k` nearest other rows, ranked, `k` entries a row: the brute-force search's
/// result (module doc).
///
/// # Errors
/// `KY-E503` if the run was cancelled.
pub(crate) fn exact(
    space: &Space<'_>,
    k: usize,
    control: &RunControl,
) -> Result<Vec<(f64, u32)>, KytoError> {
    let Some(prepared) = Prepared::new(space) else {
        return brute_force(space, k, control);
    };
    let n = space.n_rows();
    let mut nearest = vec![(0.0, 0_u32); n * k];
    nearest
        .par_chunks_mut(QUERY_ROWS * k)
        .enumerate()
        .try_for_each_init(
            || vec![0.0_f32; QUERY_ROWS * POINT_ROWS.min(n)],
            |products, (block, out)| {
                control.check()?;
                let first = block * QUERY_ROWS;
                let queries = first..(first + QUERY_ROWS).min(n);
                search_block(space, &prepared, k, queries, products, out);
                Ok(())
            },
        )?;
    Ok(nearest)
}

/// The rows as the filter's matrix products read them, with each row's slack.
///
/// Euclidean: row `a~ = fl32(x - mean)` (the mean and difference in `f64`), offset
/// `fl32(|a~|^2)`, key `fl32(fl32(offset_q + offset_p) - 2 fl32(q.p))`. Against the exact
/// squared distance, centring moves it by at most `2u (|a| + |b|)^2`, the offsets by
/// `u (|a|^2 + |b|^2)`, the product by `w u / (1 - w u) |a| |b|` (any summation order)
/// and the two roundings by `2u (|a| + |b|)^2`: at most `(w / 2 + 6) u (|a| + |b|)^2` in
/// all. The slack `E_q = 2 (w + 8) u (|a_q| + max |a|)^2` is more than twice that, which
/// also covers the brute-force key's own `f64` rounding, plus `4 (w + 8)` times the
/// smallest normal `f32` for underflow (subnormals flushed to zero).
///
/// Cosine: row `fl32(x / |x|)` (zero rows stay zero, so their keys are exactly 1, as
/// the brute-force key is), key `clip(1 - fl32(q.p), 0, 2)`: within `(w + 5) u` of the
/// exact key; slack `2 (w + 8) u` plus the same underflow term.
struct Prepared {
    metric: Metric,
    width: usize,
    rows: Vec<f32>,
    /// Each row's `fl32(|a~|^2)` (Euclidean; empty for cosine).
    offsets: Vec<f32>,
    /// Each row's slack `E_q`.
    slack: Vec<f64>,
}

impl Prepared {
    /// The filter's rows, or `None` when a row norm is too large for `f32` keys.
    fn new(space: &Space<'_>) -> Option<Self> {
        let width = space.width();
        let n = space.n_rows();
        let points = space.rows();
        let factor = 2.0 * (width as f64 + 8.0) * UNIT_ROUNDOFF;
        let underflow = 4.0 * (width as f64 + 8.0) * f64::from(f32::MIN_POSITIVE);
        match space.metric() {
            Metric::Euclidean => {
                let mut mean = vec![0.0_f64; width];
                for row in points.chunks_exact(width) {
                    for (sum, &value) in mean.iter_mut().zip(row) {
                        *sum += f64::from(value);
                    }
                }
                for sum in &mut mean {
                    *sum /= n as f64;
                }
                let rows: Vec<f32> = points
                    .chunks_exact(width)
                    .flat_map(|row| {
                        row.iter()
                            .zip(&mean)
                            .map(|(&value, &centre)| (f64::from(value) - centre) as f32)
                    })
                    .collect();
                let squares: Vec<f64> = rows
                    .chunks_exact(width)
                    .map(|row| row.iter().map(|&value| f64::from(value).powi(2)).sum())
                    .collect();
                let largest = squares
                    .iter()
                    .fold(0.0_f64, |most, &square| most.max(square));
                let largest = largest.sqrt();
                if !largest.is_finite() || largest > LARGEST_NORM {
                    return None;
                }
                let slack = squares
                    .iter()
                    .map(|&square| factor * (square.sqrt() + largest).powi(2) + underflow)
                    .collect();
                let offsets = squares.iter().map(|&square| square as f32).collect();
                Some(Self {
                    metric: Metric::Euclidean,
                    width,
                    rows,
                    offsets,
                    slack,
                })
            }
            Metric::Cosine => {
                let rows = points
                    .chunks_exact(width)
                    .enumerate()
                    .flat_map(|(index, row)| {
                        let norm = space.norm(index);
                        row.iter().map(move |&value| {
                            if norm == 0.0 {
                                0.0
                            } else {
                                (f64::from(value) / norm) as f32
                            }
                        })
                    })
                    .collect();
                Some(Self {
                    metric: Metric::Cosine,
                    width,
                    rows,
                    offsets: Vec::new(),
                    slack: vec![factor + underflow; n],
                })
            }
        }
    }

    fn rows(&self, range: std::ops::Range<usize>) -> &[f32] {
        &self.rows[range.start * self.width..range.end * self.width]
    }
}

/// One query's filter state: its `k` smallest approximate keys, its threshold (the
/// `k`-th plus twice its slack, rounded up to `f32`) and its candidates.
struct Query {
    k: usize,
    slack: f64,
    smallest: Vec<f32>,
    threshold: f32,
    candidates: Vec<(f32, u32)>,
    /// Candidates held before the next pass that drops the ones above the threshold.
    compact_at: usize,
    /// Past [`CANDIDATE_LIMIT`]: searched against every row instead.
    overflowed: bool,
}

impl Query {
    fn new(k: usize, slack: f64) -> Self {
        Self {
            k,
            slack,
            smallest: Vec::with_capacity(k + 1),
            threshold: f32::INFINITY,
            candidates: Vec::new(),
            compact_at: 64,
            overflowed: false,
        }
    }

    /// Take row `row` at approximate key `key` (already at most the threshold).
    #[inline(never)]
    fn offer(&mut self, key: f32, row: u32) {
        if self.overflowed {
            return;
        }
        if self.smallest.len() < self.k || key < self.smallest[self.k - 1] {
            // Insertion from the end (a short list: no library call for the shift).
            if self.smallest.len() < self.k {
                self.smallest.push(key);
            }
            let mut at = self.smallest.len() - 1;
            while at > 0 && self.smallest[at - 1] > key {
                self.smallest[at] = self.smallest[at - 1];
                at -= 1;
            }
            self.smallest[at] = key;
            if self.smallest.len() == self.k {
                self.threshold = round_up(f64::from(self.smallest[self.k - 1]) + 2.0 * self.slack);
            }
        }
        if key <= self.threshold {
            self.candidates.push((key, row));
            if self.candidates.len() >= self.compact_at {
                let threshold = self.threshold;
                self.candidates.retain(|&(held, _)| held <= threshold);
                if self.candidates.len() > CANDIDATE_LIMIT {
                    self.overflowed = true;
                    self.candidates = Vec::new();
                } else {
                    self.compact_at = (2 * self.candidates.len()).max(64);
                }
            }
        }
    }
}

/// The least `f32` at or above `value`.
fn round_up(value: f64) -> f32 {
    let rounded = value as f32;
    if f64::from(rounded) < value {
        rounded.next_up()
    } else {
        rounded
    }
}

/// Filter, then rank, the queries `queries`; their `k` nearest go to `out`.
fn search_block(
    space: &Space<'_>,
    prepared: &Prepared,
    k: usize,
    queries: std::ops::Range<usize>,
    products: &mut [f32],
    out: &mut [(f64, u32)],
) {
    let n = space.n_rows();
    let mut states: Vec<Query> = queries
        .clone()
        .map(|query| Query::new(k, prepared.slack[query]))
        .collect();
    let query_rows = prepared.rows(queries.clone());
    for start in (0..n).step_by(POINT_ROWS) {
        let end = (start + POINT_ROWS).min(n);
        let columns = end - start;
        let products = &mut products[..queries.len() * columns];
        f32_products(
            query_rows,
            prepared.rows(start..end),
            prepared.width,
            products,
        );
        for ((query, state), keys) in queries
            .clone()
            .zip(&mut states)
            .zip(products.chunks_exact_mut(columns))
        {
            // The dot products become approximate keys in place, then are scanned.
            match prepared.metric {
                Metric::Euclidean => {
                    let offset = prepared.offsets[query];
                    for (key, &other) in keys.iter_mut().zip(&prepared.offsets[start..end]) {
                        *key = (offset + other) - 2.0 * *key;
                    }
                }
                Metric::Cosine => {
                    for key in keys.iter_mut() {
                        *key = (1.0 - *key).clamp(0.0, 2.0);
                    }
                }
            }
            scan(state, keys, start, query);
        }
    }
    for ((query, state), out) in queries.zip(states).zip(out.chunks_exact_mut(k)) {
        let found = rank(space, k, query, &state).unwrap_or_else(|| search_row(space, k, query));
        out.copy_from_slice(found.entries());
    }
}

/// Offer `query`'s state every row of `keys` (rows `start..`) at most its threshold,
/// except the query itself. Most chunks hold none, so each chunk is first tested whole
/// against the threshold it starts with (the threshold only falls).
#[inline]
fn scan(state: &mut Query, keys: &[f32], start: usize, query: usize) {
    const CHUNK: usize = 16;
    let (chunks, tail) = keys.as_chunks::<CHUNK>();
    let offer_from = |state: &mut Query, first: usize, block: &[f32]| {
        for (offset, &key) in block.iter().enumerate() {
            let row = first + offset;
            if key <= state.threshold && row != query {
                state.offer(key, row as u32);
            }
        }
    };
    for (index, block) in chunks.iter().enumerate() {
        let threshold = state.threshold;
        if block
            .iter()
            .fold(false, |any, &key| any | (key <= threshold))
        {
            offer_from(state, start + index * CHUNK, block);
        }
    }
    offer_from(state, start + chunks.len() * CHUNK, tail);
}

/// The `k` nearest of a query's candidates by their exact keys, or `None` if the query
/// overflowed or a candidate breaks the slack (module doc).
fn rank(space: &Space<'_>, k: usize, query: usize, state: &Query) -> Option<Nearest> {
    if state.overflowed {
        return None;
    }
    let mut found = Nearest::new(k);
    let mut exact = [0.0];
    for &(approximate, row) in &state.candidates {
        if approximate > state.threshold {
            continue;
        }
        space.keys(query, row as usize, &mut exact);
        if (exact[0] - f64::from(approximate)).abs() > state.slack {
            return None;
        }
        found.offer(exact[0], row);
    }
    (found.entries().len() == k).then_some(found)
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;

    /// A seeded xorshift stream of values in `[0, 1)`.
    struct Stream(u64);

    impl Stream {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1_u64 << 24) as f32
        }
    }

    /// Rows around `clusters` centres, `width` values a row.
    fn clustered(seed: u64, n: usize, width: usize, clusters: usize, offset: f32) -> Vec<f32> {
        let mut stream = Stream(seed);
        let centres: Vec<f32> = (0..clusters * width)
            .map(|_| 10.0 * stream.next() - 5.0)
            .collect();
        (0..n)
            .flat_map(|row| {
                let centre = row % clusters;
                (0..width)
                    .map(|component| {
                        let noise = stream.next() + stream.next() + stream.next() - 1.5;
                        centres[centre * width + component] + noise + offset
                    })
                    .collect::<Vec<f32>>()
            })
            .collect()
    }

    /// The filtered search equals the brute-force search on 1 and 4 threads.
    fn assert_equal(points: &[f32], width: usize, k: usize, metric: Metric) {
        let space = Space::new(Cow::Borrowed(points), width, metric);
        let control = RunControl::new();
        let expected = brute_force(&space, k, &control).unwrap();
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let found = pool.install(|| exact(&space, k, &control)).unwrap();
            assert!(
                found
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.0.to_bits() == b.0.to_bits() && a.1 == b.1),
                "{metric:?}, {threads} threads, width {width}, k {k}"
            );
            assert_eq!(found.len(), expected.len());
        }
    }

    #[test]
    fn equals_brute_force_on_clusters() {
        for (seed, n, width, k) in [(1, 3000, 50, 14), (2, 1500, 7, 5), (3, 2100, 1, 9)] {
            let points = clustered(seed, n, width, 8, 0.0);
            assert_equal(&points, width, k, Metric::Euclidean);
            assert_equal(&points, width, k, Metric::Cosine);
        }
    }

    #[test]
    fn equals_brute_force_far_from_the_origin() {
        let points = clustered(4, 2000, 20, 5, 1000.0);
        assert_equal(&points, 20, 14, Metric::Euclidean);
        assert_equal(&points, 20, 14, Metric::Cosine);
    }

    #[test]
    fn equals_brute_force_on_ties_and_duplicates() {
        // Small integers: 16 distinct rows, each repeated about 375 times, so most
        // queries pass the candidate limit and are searched against every row.
        let mut stream = Stream(5);
        let tied: Vec<f32> = (0..6000 * 2)
            .map(|_| (stream.next() * 4.0).floor())
            .collect();
        assert_equal(&tied, 2, 14, Metric::Euclidean);
        assert_equal(&tied, 2, 14, Metric::Cosine);
        // Fewer ties than the limit: integers in 3 dimensions.
        let tied: Vec<f32> = (0..2000 * 3)
            .map(|_| (stream.next() * 6.0).floor())
            .collect();
        assert_equal(&tied, 3, 10, Metric::Euclidean);
        // Every row the same.
        assert_equal(&[0.25; 300], 3, 7, Metric::Euclidean);
    }

    #[test]
    fn equals_brute_force_on_small_inputs() {
        assert_equal(&[0.0, 1.0], 1, 1, Metric::Euclidean);
        let points = clustered(6, 16, 4, 2, 0.0);
        assert_equal(&points, 4, 15, Metric::Euclidean);
        assert_equal(&points, 4, 15, Metric::Cosine);
        // More rows than one product tile, fewer than one query block's worth twice.
        let points = clustered(7, POINT_ROWS + QUERY_ROWS + 3, 9, 3, 0.0);
        assert_equal(&points, 9, 4, Metric::Euclidean);
    }

    #[test]
    fn cosine_with_zero_rows() {
        let mut points = clustered(8, 900, 6, 4, 0.0);
        for row in (0..900).step_by(7) {
            points[row * 6..row * 6 + 6].fill(0.0);
        }
        assert_equal(&points, 6, 14, Metric::Cosine);
    }

    #[test]
    fn huge_norms_fall_back_to_brute_force() {
        let points: Vec<f32> = clustered(9, 400, 3, 2, 0.0)
            .iter()
            .map(|&value| value * 1e20)
            .collect();
        let space = Space::new(Cow::Borrowed(&points), 3, Metric::Euclidean);
        assert!(Prepared::new(&space).is_none());
        assert_equal(&points, 3, 5, Metric::Euclidean);
    }

    #[test]
    fn round_up_never_rounds_down() {
        assert_eq!(round_up(1.0), 1.0);
        let value = 1.0 + 1e-12;
        assert!(f64::from(round_up(value)) >= value);
        assert_eq!(round_up(f64::INFINITY), f32::INFINITY);
    }

    #[test]
    fn query_keeps_rows_within_the_threshold() {
        let mut query = Query::new(2, 0.5);
        for (key, row) in [(5.0, 0), (1.0, 1), (2.0, 2), (2.9, 3), (3.1, 4), (0.5, 5)] {
            if key <= query.threshold {
                query.offer(key, row);
            }
        }
        // The 2nd smallest is 1.0, so the threshold is 2.0 and rows 1, 2 and 5 remain.
        assert_eq!(query.threshold, 2.0);
        let kept: Vec<u32> = query
            .candidates
            .iter()
            .filter(|&&(key, _)| key <= query.threshold)
            .map(|&(_, row)| row)
            .collect();
        assert_eq!(kept, [1, 2, 5]);
    }
}
