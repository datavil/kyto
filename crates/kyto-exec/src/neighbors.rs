//! The k-nearest-neighbour graph of the cells in an embedding (task 6.6).
//!
//! Does: for every row of a dense `float32` embedding (its leading `n_components`
//! columns), find the `n_neighbors - 1` nearest other rows, as scanpy's `distances`
//! graph holds them (the cell itself is excluded by its position, so a duplicate cell is
//! a neighbour at distance 0). Exact search ([`NeighborSearch::Exact`]) is blocked:
//! blocks of [`QUERY_BLOCK`] query rows run in parallel, each against tiles of
//! [`POINT_TILE`] rows through the SIMD distance kernels ([`Kernels::squared_distances`],
//! [`Kernels::dot_products`], `f64` in a fixed lane order), keeping its nearest rows in
//! a sorted list. Approximate search ([`NeighborSearch::Approximate`]) takes candidates
//! from an HNSW index (`neighbors_hnsw.rs`) and ranks them with the same exact
//! distances. Rows are ranked by (distance, row position), so ties go to the earlier
//! cell, and every row's result is computed alone: the same graph on any thread count
//! (R11). Euclidean distance is `sqrt` of the `f64` squared distance; cosine distance
//! is `1 - dot / (|a| |b|)` in `f64`, clipped to `[0, 2]`, and 1 when either row is
//! zero (scikit-learn's `cosine_distances`). Both are rounded once to `float32`.
//! The result is a CSR graph ([`NeighborGraph`]) with column indices increasing within
//! each row (format.md §7).
//! Does not: weight the graph (connectivities, 6.7), read embeddings from the store
//! (5.11), or adjust `n_neighbors` to small data as scanpy does (`KY-E215` instead).

use std::borrow::Cow;
use std::sync::Arc;

use kyto_error::KytoError;
use kyto_plan::{Metric, NeighborSearch, Neighbors};
use rayon::prelude::*;

use crate::kernels::Kernels;
use crate::neighbors_hnsw;
use crate::run_control::RunControl;

/// Query rows a parallel task searches together (one cancellation check each).
pub(crate) const QUERY_BLOCK: usize = 64;
/// Rows a block of queries compares against at once (`POINT_TILE x n_components`
/// `float32` values stay in cache while the block's queries pass over them).
pub(crate) const POINT_TILE: usize = 1024;

/// A k-nearest-neighbour graph in CSR: row `i` holds the `n_neighbors - 1` nearest
/// other rows of row `i`, column indices increasing, `float32` distances.
#[derive(Debug, Clone, PartialEq)]
pub struct NeighborGraph {
    n_neighbors: usize,
    indptr: Vec<u64>,
    indices: Vec<u32>,
    distances: Vec<f32>,
}

impl NeighborGraph {
    /// The number of rows (and columns): the cells searched.
    pub fn n_cells(&self) -> usize {
        self.indptr.len() - 1
    }

    /// Neighbours per cell, the cell itself included (each row stores one fewer).
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// The row pointer: `n_cells() + 1` offsets into [`NeighborGraph::indices`].
    pub fn indptr(&self) -> &[u64] {
        &self.indptr
    }

    /// Each row's neighbours (row positions), increasing within a row.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// The distance to each neighbour, as [`NeighborGraph::indices`] orders them (0 to a
    /// duplicate cell is a stored entry).
    pub fn distances(&self) -> &[f32] {
        &self.distances
    }
}

/// What a `neighbors` node computed (task 6.6): the distance graph over the rows of
/// the embedding it searched.
#[derive(Debug, Clone, PartialEq)]
pub struct NeighborStatistics {
    pub(crate) cells: Arc<Vec<u64>>,
    pub(crate) distances: NeighborGraph,
}

impl NeighborStatistics {
    /// The store table position of each graph row (and column): the embedding's rows,
    /// in stream order (for a `pca` embedding, its `cells()`).
    pub fn cells(&self) -> &[u64] {
        &self.cells
    }

    /// scanpy's `distances` graph.
    pub fn distances(&self) -> &NeighborGraph {
        &self.distances
    }
}

/// The neighbour graph of the rows of `embedding` (row-major, `width` values a row) as
/// `params` asks (`params.embedding` and `params.key` are not read), on the current
/// rayon pool.
///
/// # Errors
/// `KY-E215` if `width` is 0 or does not divide the embedding, `n_components` is 0 or
/// above `width`, `n_neighbors` is below 2 or above the number of rows, there are more
/// than 2^32 rows, or a value of the used components is not finite.
pub fn nearest_neighbors(
    embedding: &[f32],
    width: usize,
    params: &Neighbors,
) -> Result<NeighborGraph, KytoError> {
    search(embedding, width, params, &RunControl::new())
}

/// [`nearest_neighbors`], cancelled through `control` (`KY-E503`).
pub(crate) fn search(
    embedding: &[f32],
    width: usize,
    params: &Neighbors,
    control: &RunControl,
) -> Result<NeighborGraph, KytoError> {
    if width == 0 || !embedding.len().is_multiple_of(width) {
        return Err(invalid(
            "embedding",
            &format!("{} values", embedding.len()),
            &format!("they are not rows of {width} components"),
            "a row-major embedding with at least one component",
        ));
    }
    let n_rows = embedding.len() / width;
    let used = match params.n_components {
        None => width,
        Some(n) => match usize::try_from(n) {
            Ok(n) if (1..=width).contains(&n) => n,
            _ => {
                return Err(invalid(
                    "n_components",
                    &n.to_string(),
                    &format!("the embedding has {width} components"),
                    &format!("an int from 1 to {width}, or None for all"),
                ));
            }
        },
    };
    let n_neighbors = usize::try_from(params.n_neighbors).unwrap_or(usize::MAX);
    if n_neighbors < 2 || n_neighbors > n_rows {
        return Err(invalid(
            "n_neighbors",
            &params.n_neighbors.to_string(),
            &format!(
                "it counts the cell itself, and {n_rows} cells reach neighbors (scanpy \
                 would lower it to {}; kyto asks)",
                1 + n_rows / 2
            ),
            &format!("an int from 2 to {}", n_rows.max(2)),
        ));
    }
    if u32::try_from(n_rows - 1).is_err() {
        return Err(invalid(
            "embedding",
            &format!("{n_rows} cells"),
            "graph indices are uint32 (format.md §7)",
            "at most 2^32 cells",
        ));
    }
    let points: Cow<'_, [f32]> = if used == width {
        Cow::Borrowed(embedding)
    } else {
        Cow::Owned(
            embedding
                .chunks_exact(width)
                .flat_map(|row| &row[..used])
                .copied()
                .collect(),
        )
    };
    if let Some(position) = points.iter().position(|value| !value.is_finite()) {
        return Err(invalid(
            "embedding",
            &points[position].to_string(),
            &format!(
                "row {}, component {} is not finite",
                position / used,
                position % used
            ),
            "finite values",
        ));
    }
    let space = Space::new(points, used, params.metric);
    let k = n_neighbors - 1;
    let nearest = match params.search {
        NeighborSearch::Exact => exact(&space, k, control)?,
        NeighborSearch::Approximate => {
            neighbors_hnsw::approximate(&space, k, params.seed, control)?
        }
    };
    Ok(space.graph(nearest, n_neighbors))
}

fn invalid(parameter: &str, value: &str, problem: &str, accepted: &str) -> KytoError {
    KytoError::InvalidArgument {
        function: "neighbors".to_owned(),
        parameter: parameter.to_owned(),
        value: value.to_owned(),
        problem: problem.to_owned(),
        accepted: accepted.to_owned(),
    }
}

/// The rows searched, with what ranking them needs.
pub(crate) struct Space<'a> {
    points: Cow<'a, [f32]>,
    width: usize,
    n_rows: usize,
    metric: Metric,
    /// Each row's Euclidean norm (cosine only).
    norms: Vec<f64>,
    kernels: Kernels,
}

impl<'a> Space<'a> {
    fn new(points: Cow<'a, [f32]>, width: usize, metric: Metric) -> Self {
        let kernels = Kernels::detected();
        let n_rows = points.len() / width;
        let norms = match metric {
            Metric::Euclidean => Vec::new(),
            Metric::Cosine => {
                let mut squares = [0.0];
                points
                    .chunks_exact(width)
                    .map(|row| {
                        kernels.dot_products(row, row, &mut squares);
                        squares[0].sqrt()
                    })
                    .collect()
            }
        };
        Self {
            points,
            width,
            n_rows,
            metric,
            norms,
            kernels,
        }
    }

    pub(crate) fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn metric(&self) -> Metric {
        self.metric
    }

    /// Every row, row-major.
    pub(crate) fn rows(&self) -> &[f32] {
        &self.points
    }

    pub(crate) fn row(&self, row: usize) -> &[f32] {
        &self.points[row * self.width..(row + 1) * self.width]
    }

    pub(crate) fn norm(&self, row: usize) -> f64 {
        self.norms[row]
    }

    /// The ranking key of `query` against rows `start..start + out.len()`, into `out`:
    /// the squared Euclidean distance, or the cosine distance.
    pub(crate) fn keys(&self, query: usize, start: usize, out: &mut [f64]) {
        let points = &self.points[start * self.width..(start + out.len()) * self.width];
        match self.metric {
            Metric::Euclidean => self.kernels.squared_distances(self.row(query), points, out),
            Metric::Cosine => {
                self.kernels.dot_products(self.row(query), points, out);
                let query_norm = self.norms[query];
                for (key, &norm) in out.iter_mut().zip(&self.norms[start..]) {
                    *key = cosine_distance(*key, query_norm, norm);
                }
            }
        }
    }

    /// The distance a ranking key stands for.
    fn distance(&self, key: f64) -> f64 {
        match self.metric {
            Metric::Euclidean => key.sqrt(),
            Metric::Cosine => key,
        }
    }

    /// The CSR graph of `nearest` (`k` entries a row, each row ranked).
    fn graph(&self, nearest: Vec<(f64, u32)>, n_neighbors: usize) -> NeighborGraph {
        let k = n_neighbors - 1;
        let mut indptr = Vec::with_capacity(self.n_rows + 1);
        let mut indices = Vec::with_capacity(nearest.len());
        let mut distances = Vec::with_capacity(nearest.len());
        let mut row = Vec::with_capacity(k);
        indptr.push(0);
        for ranked in nearest.chunks_exact(k) {
            row.clear();
            row.extend_from_slice(ranked);
            row.sort_unstable_by_key(|&(_, index)| index);
            for &(key, index) in &row {
                indices.push(index);
                distances.push(self.distance(key) as f32);
            }
            indptr.push(indices.len() as u64);
        }
        NeighborGraph {
            n_neighbors,
            indptr,
            indices,
            distances,
        }
    }
}

/// `1 - dot / (|a| |b|)` clipped to `[0, 2]`; 1 when either norm is 0 (scikit-learn
/// normalises a zero row to zero, so its similarity to anything is 0).
fn cosine_distance(dot: f64, left: f64, right: f64) -> f64 {
    if left == 0.0 || right == 0.0 {
        return 1.0;
    }
    (1.0 - dot / (left * right)).clamp(0.0, 2.0)
}

/// The `k` nearest rows offered so far, ranked by (key, row).
pub(crate) struct Nearest {
    k: usize,
    entries: Vec<(f64, u32)>,
}

impl Nearest {
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            entries: Vec::with_capacity(k + 1),
        }
    }

    /// Keep row `index` at `key` if it ranks among the `k` nearest.
    #[inline]
    pub(crate) fn offer(&mut self, key: f64, index: u32) {
        let before = |entry: &(f64, u32)| entry.0 < key || (entry.0 == key && entry.1 < index);
        if self.entries.len() == self.k {
            match self.entries.last() {
                Some(worst) if before(worst) => return,
                _ => {}
            }
            self.entries.pop();
        }
        let at = self.entries.partition_point(before);
        self.entries.insert(at, (key, index));
    }

    pub(crate) fn entries(&self) -> &[(f64, u32)] {
        &self.entries
    }
}

/// Exact search: each row's `k` nearest other rows, ranked, `k` entries a row.
fn exact(space: &Space<'_>, k: usize, control: &RunControl) -> Result<Vec<(f64, u32)>, KytoError> {
    let n = space.n_rows();
    let mut nearest = vec![(0.0, 0_u32); n * k];
    nearest
        .par_chunks_mut(QUERY_BLOCK * k)
        .enumerate()
        .try_for_each(|(block, out)| {
            control.check()?;
            let queries = block * QUERY_BLOCK..(block * QUERY_BLOCK + QUERY_BLOCK).min(n);
            let mut found: Vec<Nearest> = queries.clone().map(|_| Nearest::new(k)).collect();
            let mut keys = vec![0.0; POINT_TILE.min(n)];
            for start in (0..n).step_by(POINT_TILE) {
                let keys = &mut keys[..POINT_TILE.min(n - start)];
                for (query, found) in queries.clone().zip(&mut found) {
                    space.keys(query, start, keys);
                    for (offset, &key) in keys.iter().enumerate() {
                        let row = start + offset;
                        if row != query {
                            found.offer(key, row as u32);
                        }
                    }
                }
            }
            for (found, out) in found.iter().zip(out.chunks_exact_mut(k)) {
                out.copy_from_slice(found.entries());
            }
            Ok(())
        })?;
    Ok(nearest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(n_neighbors: u64, metric: Metric, search: NeighborSearch) -> Neighbors {
        Neighbors {
            embedding: "X_pca".to_owned(),
            n_components: None,
            n_neighbors,
            metric,
            search,
            seed: 0,
            key: None,
        }
    }

    #[test]
    fn nearest_keeps_the_k_smallest_ties_to_the_earlier_row() {
        let mut nearest = Nearest::new(2);
        for (key, index) in [(3.0, 0), (1.0, 5), (1.0, 2), (0.5, 9), (1.0, 1)] {
            nearest.offer(key, index);
        }
        assert_eq!(nearest.entries(), [(0.5, 9), (1.0, 1)]);
    }

    #[test]
    fn a_line_of_points() {
        // Rows at 0, 1, 3, 7 on a line.
        let embedding = [0.0, 1.0, 3.0, 7.0];
        let graph = nearest_neighbors(
            &embedding,
            1,
            &params(3, Metric::Euclidean, NeighborSearch::Exact),
        )
        .unwrap();
        assert_eq!(graph.indptr(), [0, 2, 4, 6, 8]);
        assert_eq!(graph.indices(), [1, 2, 0, 2, 0, 1, 1, 2]);
        assert_eq!(graph.distances(), [1.0, 3.0, 1.0, 2.0, 3.0, 2.0, 6.0, 4.0]);
    }

    #[test]
    fn cosine_of_a_zero_row_is_one() {
        assert_eq!(cosine_distance(0.0, 0.0, 2.0), 1.0);
        assert_eq!(cosine_distance(4.0, 2.0, 2.0), 0.0);
        assert_eq!(cosine_distance(-4.0, 2.0, 2.0), 2.0);
    }

    #[test]
    fn refuses_bad_arguments() {
        let embedding = [0.0_f32; 6];
        let message = |result: Result<NeighborGraph, KytoError>| result.unwrap_err().to_string();
        let euclidean = params(4, Metric::Euclidean, NeighborSearch::Exact);
        assert!(message(nearest_neighbors(&embedding, 2, &euclidean)).contains("n_neighbors"));
        assert!(message(nearest_neighbors(&embedding, 4, &euclidean)).contains("rows of 4"));
        let mut wide = params(2, Metric::Euclidean, NeighborSearch::Exact);
        wide.n_components = Some(3);
        assert!(message(nearest_neighbors(&embedding, 2, &wide)).contains("n_components"));
        let not_finite = [0.0, f32::NAN, 1.0, 2.0];
        assert!(
            message(nearest_neighbors(
                &not_finite,
                2,
                &params(2, Metric::Euclidean, NeighborSearch::Exact)
            ))
            .contains("row 0, component 1")
        );
    }
}
