//! The spectral initialisation of a UMAP layout (task 6.9), as umap-learn 0.5.12's
//! `spectral_layout` lays out the graph scanpy hands it.
//!
//! Does: split the graph into connected components. One component: the eigenvectors of
//! the `dim + 1` smallest eigenvalues of the normalised Laplacian
//! `L = I - D^-1/2 A D^-1/2`, the first (the trivial one) dropped. They are found as
//! the largest of `S = (I + D^-1/2 A D^-1/2) / 2` (the same vectors, eigenvalues in
//! `[0, 1]`) by a restarted block Krylov method with Rayleigh-Ritz, to umap's tolerance
//! (scipy's ARPACK there). Several components: umap's `multi_component_layout`: a
//! meta-layout of the components (the corners `+-e_i` for at most `2 dim` components,
//! else the spectral embedding of their centroids' affinities `exp(-d^2)` in `data`),
//! each component's own spectral layout scaled into half the distance to its nearest
//! neighbour around its corner; components too small for a spectral layout get uniform
//! positions there. A graph of at most `dim + 1` cells gets uniform positions in
//! `[-10, 10)` (umap's fallback when ARPACK cannot run).
//! Every random draw comes from the seed (SplitMix64). Products with `S` run by rows on
//! the current rayon pool; dot products and the dense steps (faer, `Par::Seq`) are
//! sequential: the same bits on any thread count (R11).
//! Does not: scale or jitter the result (`umap.rs`), or match ARPACK's vectors: the
//! eigenvectors of a repeated eigenvalue are any basis of their space, and their signs
//! are fixed here by the largest absolute value being positive.

use faer::{Mat, Side};
use kyto_error::KytoError;
use kyto_plan::Metric;
use rayon::prelude::*;

use crate::connectivities::ConnectivityGraph;
use crate::pca::mix;

/// A Ritz pair is converged when `||S v - theta v||` is at most this (umap asks ARPACK
/// for 1e-4 relative accuracy on `L`'s eigenvalues; `L`'s residual is twice `S`'s).
const TOLERANCE: f64 = 5e-5;

/// Restarts of the block Krylov method before the best Ritz vectors are kept.
const MAX_RESTARTS: usize = 500;

/// Blocks (the start block included) in the Krylov basis of one restart.
const KRYLOV_BLOCKS: usize = 8;

/// Columns of the Krylov block beyond the `dim + 1` eigenvectors wanted.
const EXTRA_COLUMNS: usize = 4;

/// The rows a dot product sums in order; the partial sums are added in row order too.
const DOT_CHUNK: usize = 4096;

/// A seeded stream of SplitMix64 values (integer arithmetic only).
#[derive(Debug, Clone)]
pub(crate) struct Draws {
    state: u64,
}

impl Draws {
    /// The stream `purpose` of `seed` (different purposes give unrelated streams).
    pub(crate) fn new(seed: u64, purpose: u64) -> Self {
        Self {
            state: mix(seed ^ mix(purpose.wrapping_add(0x9E37_79B9_7F4A_7C15))),
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.state)
    }

    /// Uniform in `[0, 1)`: 53 random bits.
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * f64::powi(2.0, -53)
    }

    /// Uniform in `[low, high)`.
    pub(crate) fn uniform_in(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.uniform()
    }

    /// Standard normal (Box-Muller, the cosine branch).
    pub(crate) fn normal(&mut self) -> f64 {
        // 1 - uniform is in (0, 1], so the logarithm is finite.
        let radius = (-2.0 * (1.0 - self.uniform()).ln()).sqrt();
        radius * (std::f64::consts::TAU * self.uniform()).cos()
    }
}

/// A symmetric weighted graph over `n` vertices in CSR (`f64` weights).
struct Csr {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    weights: Vec<f64>,
}

impl Csr {
    fn n(&self) -> usize {
        self.indptr.len() - 1
    }
}

/// The spectral initialisation of `graph` (the thresholded graph the layout optimises)
/// in `dim` dimensions, row-major `n x dim` in `f64`, before umap's scaling. `data`
/// (row-major, `width` values a row, as the neighbours searched it, with `metric`)
/// places more than `2 dim` components by their centroids; without it they are placed
/// uniformly in `[0, 10)` (umap's fallback without data).
///
/// # Errors
/// `KY-E215` if an eigendecomposition does not converge (non-finite weights).
pub(crate) fn spectral_layout(
    graph: &ConnectivityGraph,
    keep: &[bool],
    dim: usize,
    data: Option<(&[f32], usize, Metric)>,
    seed: u64,
) -> Result<Vec<f64>, KytoError> {
    let n = graph.n_cells();
    let csr = kept_graph(graph, keep);
    let (labels, n_components) = components(&csr);
    if n_components <= 1 {
        return component_layout_or_random(&csr, dim, seed, 0, 10.0);
    }
    let meta = meta_layout(&labels, n_components, dim, data, seed)?;
    let mut result = vec![0.0; n * dim];
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_components];
    for (vertex, &label) in labels.iter().enumerate() {
        members[label].push(vertex);
    }
    for (label, vertices) in members.iter().enumerate() {
        let corner = &meta[label * dim..(label + 1) * dim];
        // Half the distance to the nearest other corner (umap's `data_range`).
        let nearest = (0..n_components)
            .map(|other| squared(corner, &meta[other * dim..(other + 1) * dim]).sqrt())
            .filter(|&distance| distance > 0.0)
            .fold(f64::INFINITY, f64::min);
        let data_range = if nearest.is_finite() {
            nearest / 2.0
        } else {
            1.0
        };
        let sub = subgraph(&csr, vertices);
        let size = vertices.len();
        let positions = if size < 2 * dim || size <= dim + 1 {
            let mut draws = Draws::new(seed, 0x5350_0000 + label as u64);
            (0..size * dim)
                .map(|_| draws.uniform_in(-data_range, data_range))
                .collect()
        } else {
            let mut layout = eigen_layout(&sub, dim, seed, label as u64)?;
            let largest = layout.iter().fold(0.0_f64, |most, v| most.max(v.abs()));
            let expansion = if largest > 0.0 {
                data_range / largest
            } else {
                0.0
            };
            for value in &mut layout {
                *value *= expansion;
            }
            layout
        };
        for (position, &vertex) in vertices.iter().enumerate() {
            for axis in 0..dim {
                result[vertex * dim + axis] = positions[position * dim + axis] + corner[axis];
            }
        }
    }
    Ok(result)
}

/// One connected graph's spectral layout, or uniform positions in `[-range, range)`
/// for a graph of at most `dim + 1` vertices.
fn component_layout_or_random(
    csr: &Csr,
    dim: usize,
    seed: u64,
    label: u64,
    range: f64,
) -> Result<Vec<f64>, KytoError> {
    if csr.n() <= dim + 1 {
        let mut draws = Draws::new(seed, 0x5241_0000 + label);
        return Ok((0..csr.n() * dim)
            .map(|_| draws.uniform_in(-range, range))
            .collect());
    }
    eigen_layout(csr, dim, seed, label)
}

/// The edges of `graph` whose `keep` flag is set, in `f64`.
fn kept_graph(graph: &ConnectivityGraph, keep: &[bool]) -> Csr {
    let mut indptr = Vec::with_capacity(graph.n_cells() + 1);
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    indptr.push(0);
    for row in 0..graph.n_cells() {
        let start = graph.indptr()[row] as usize;
        let end = graph.indptr()[row + 1] as usize;
        for edge in (start..end).filter(|&edge| keep[edge]) {
            indices.push(graph.indices()[edge] as usize);
            weights.push(f64::from(graph.weights()[edge]));
        }
        indptr.push(indices.len());
    }
    Csr {
        indptr,
        indices,
        weights,
    }
}

/// The connected component of each vertex, numbered by their smallest vertex (scipy's
/// `connected_components` order), and how many there are.
fn components(csr: &Csr) -> (Vec<usize>, usize) {
    let n = csr.n();
    let mut labels = vec![usize::MAX; n];
    let mut count = 0;
    let mut stack = Vec::new();
    for start in 0..n {
        if labels[start] != usize::MAX {
            continue;
        }
        labels[start] = count;
        stack.push(start);
        while let Some(vertex) = stack.pop() {
            for &next in &csr.indices[csr.indptr[vertex]..csr.indptr[vertex + 1]] {
                if labels[next] == usize::MAX {
                    labels[next] = count;
                    stack.push(next);
                }
            }
        }
        count += 1;
    }
    (labels, count)
}

/// The graph among `vertices` (increasing), renumbered by their position.
fn subgraph(csr: &Csr, vertices: &[usize]) -> Csr {
    let mut position = std::collections::HashMap::with_capacity(vertices.len());
    for (index, &vertex) in vertices.iter().enumerate() {
        position.insert(vertex, index);
    }
    let mut indptr = Vec::with_capacity(vertices.len() + 1);
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    indptr.push(0);
    for &vertex in vertices {
        for edge in csr.indptr[vertex]..csr.indptr[vertex + 1] {
            if let Some(&column) = position.get(&csr.indices[edge]) {
                indices.push(column);
                weights.push(csr.weights[edge]);
            }
        }
        indptr.push(indices.len());
    }
    Csr {
        indptr,
        indices,
        weights,
    }
}

/// umap's meta-layout of `n_components` components (`n_components x dim`).
fn meta_layout(
    labels: &[usize],
    n_components: usize,
    dim: usize,
    data: Option<(&[f32], usize, Metric)>,
    seed: u64,
) -> Result<Vec<f64>, KytoError> {
    if n_components <= 2 * dim {
        // The corners e_1 .. e_k, then -e_1 .. -e_k, k = ceil(n_components / 2).
        let k = n_components.div_ceil(2);
        let mut meta = vec![0.0; n_components * dim];
        for label in 0..n_components {
            let (axis, sign) = if label < k {
                (label, 1.0)
            } else {
                (label - k, -1.0)
            };
            meta[label * dim + axis] = sign;
        }
        return Ok(meta);
    }
    let Some((values, width, metric)) = data else {
        let mut draws = Draws::new(seed, 0x4D45_5441);
        return Ok((0..n_components * dim)
            .map(|_| draws.uniform() * 10.0)
            .collect());
    };
    // Centroids in f64, as numpy's mean of the component's rows.
    let mut centroids = vec![0.0; n_components * width];
    let mut sizes = vec![0_usize; n_components];
    for (vertex, &label) in labels.iter().enumerate() {
        sizes[label] += 1;
        for column in 0..width {
            centroids[label * width + column] += f64::from(values[vertex * width + column]);
        }
    }
    for label in 0..n_components {
        for column in 0..width {
            centroids[label * width + column] /= sizes[label] as f64;
        }
    }
    let row = |label: usize| &centroids[label * width..(label + 1) * width];
    let affinity = Mat::from_fn(n_components, n_components, |i, j| {
        if i == j {
            return 0.0;
        }
        let distance = match metric {
            Metric::Euclidean => squared(row(i), row(j)).sqrt(),
            Metric::Cosine => cosine_distance(row(i), row(j)),
        };
        (-(distance * distance)).exp()
    });
    let mut embedding = dense_spectral_embedding(&affinity, dim)?;
    let largest = embedding.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if largest.is_finite() && largest != 0.0 {
        for value in &mut embedding {
            *value /= largest;
        }
    }
    Ok(embedding)
}

/// scikit-learn's `SpectralEmbedding(affinity="precomputed")` of a small dense
/// affinity (zero diagonal): the eigenvectors of the normalised Laplacian's smallest
/// eigenvalues divided by the square root of the degrees, each signed so its largest
/// absolute value is positive, the first dropped (`n x dim`, row-major).
fn dense_spectral_embedding(affinity: &Mat<f64>, dim: usize) -> Result<Vec<f64>, KytoError> {
    let n = affinity.nrows();
    let degree: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| affinity[(i, j)]).sum::<f64>().sqrt())
        .collect();
    let laplacian = Mat::from_fn(n, n, |i, j| {
        let scaled = if degree[i] > 0.0 && degree[j] > 0.0 {
            affinity[(i, j)] / (degree[i] * degree[j])
        } else {
            0.0
        };
        if i == j { 1.0 - scaled } else { -scaled }
    });
    let eigen = laplacian
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| not_converged())?;
    let vectors = eigen.U();
    let mut embedding = vec![0.0; n * dim];
    // faer sorts eigenvalues increasing: column 0 is the trivial one.
    for axis in 0..dim.min(n.saturating_sub(1)) {
        let column: Vec<f64> = (0..n)
            .map(|i| {
                if degree[i] > 0.0 {
                    vectors[(i, axis + 1)] / degree[i]
                } else {
                    vectors[(i, axis + 1)]
                }
            })
            .collect();
        let sign = sign_of_largest(&column);
        for i in 0..n {
            embedding[i * dim + axis] = sign * column[i];
        }
    }
    Ok(embedding)
}

/// The spectral layout of a connected graph of more than `dim + 1` vertices
/// (row-major `n x dim`).
fn eigen_layout(csr: &Csr, dim: usize, seed: u64, label: u64) -> Result<Vec<f64>, KytoError> {
    let n = csr.n();
    let wanted = dim + 1;
    let sqrt_degree: Vec<f64> = (0..n)
        .map(|row| {
            csr.weights[csr.indptr[row]..csr.indptr[row + 1]]
                .iter()
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let operator = Operator {
        csr,
        inverse_sqrt_degree: sqrt_degree
            .iter()
            .map(|&value| if value > 0.0 { 1.0 / value } else { 0.0 })
            .collect(),
    };
    let block_width = (wanted + EXTRA_COLUMNS).min(n);
    let max_basis = (block_width * KRYLOV_BLOCKS).min(n);
    // The start block: the trivial eigenvector, then seeded uniform columns.
    let mut draws = Draws::new(seed, 0x4549_0000 + label);
    let mut block = Mat::from_fn(n, block_width, |_, _| 0.0);
    for row in 0..n {
        block[(row, 0)] = sqrt_degree[row];
    }
    for column in 1..block_width {
        for row in 0..n {
            block[(row, column)] = draws.uniform_in(-1.0, 1.0);
        }
    }
    let mut best = None;
    for _ in 0..MAX_RESTARTS {
        let mut basis = extend(None, &block)?;
        let mut last = basis.clone();
        while basis.ncols() < max_basis && last.ncols() > 0 {
            let product = operator.apply(&last);
            let mut new = extend(Some(&basis), &product)?;
            let room = max_basis - basis.ncols();
            if new.ncols() > room {
                new = new.subcols(0, room).to_owned();
            }
            if new.ncols() == 0 {
                break;
            }
            basis = concat(&basis, &new);
            last = new;
        }
        let products = operator.apply(&basis);
        let projected = basis.transpose() * &products;
        let symmetric = Mat::from_fn(projected.nrows(), projected.ncols(), |row, column| {
            0.5 * (projected[(row, column)] + projected[(column, row)])
        });
        let eigen = symmetric
            .self_adjoint_eigen(Side::Lower)
            .map_err(|_| not_converged())?;
        let m = symmetric.nrows();
        let taken = block_width.min(m);
        let values = eigen.S().column_vector();
        // Largest first (faer sorts increasing).
        let order: Vec<usize> = (0..taken).map(|index| m - 1 - index).collect();
        let chosen = Mat::from_fn(m, taken, |row, column| eigen.U()[(row, order[column])]);
        let ritz = &basis * &chosen;
        let ritz_products = &products * &chosen;
        let converged = (0..wanted.min(taken)).all(|column| {
            let theta = values[order[column]];
            let residual: f64 = (0..n)
                .map(|row| {
                    let difference = ritz_products[(row, column)] - theta * ritz[(row, column)];
                    difference * difference
                })
                .sum();
            residual.sqrt() <= TOLERANCE
        });
        let spans_everything = basis.ncols() >= n;
        best = Some(ritz.clone());
        if converged || spans_everything || taken < wanted {
            break;
        }
        block = ritz;
    }
    let Some(vectors) = best else {
        return Err(not_converged());
    };
    let mut layout = vec![0.0; n * dim];
    for axis in 0..dim.min(vectors.ncols().saturating_sub(1)) {
        let column: Vec<f64> = (0..n).map(|row| vectors[(row, axis + 1)]).collect();
        let sign = sign_of_largest(&column);
        for row in 0..n {
            layout[row * dim + axis] = sign * column[row];
        }
    }
    Ok(layout)
}

/// `S = (I + D^-1/2 A D^-1/2) / 2` applied to blocks of columns.
struct Operator<'a> {
    csr: &'a Csr,
    inverse_sqrt_degree: Vec<f64>,
}

impl Operator<'_> {
    /// `S x` for every column of `x`, rows in parallel (each row's sum in edge order).
    fn apply(&self, x: &Mat<f64>) -> Mat<f64> {
        let n = x.nrows();
        let columns = x.ncols();
        let scaled = Mat::from_fn(n, columns, |row, column| {
            x[(row, column)] * self.inverse_sqrt_degree[row]
        });
        let rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|row| {
                let mut sums = vec![0.0; columns];
                for edge in self.csr.indptr[row]..self.csr.indptr[row + 1] {
                    let weight = self.csr.weights[edge];
                    let other = self.csr.indices[edge];
                    for (column, sum) in sums.iter_mut().enumerate() {
                        *sum += weight * scaled[(other, column)];
                    }
                }
                sums
            })
            .collect();
        Mat::from_fn(n, columns, |row, column| {
            0.5 * (x[(row, column)] + self.inverse_sqrt_degree[row] * rows[row][column])
        })
    }
}

/// The orthonormal directions `block` adds to `basis`: the basis projected off twice
/// (sequential dot products in [`DOT_CHUNK`] row chunks), then normalised column by
/// column (modified Gram-Schmidt, twice); columns whose rest is below 1e-10 of their
/// norm are dropped.
fn extend(basis: Option<&Mat<f64>>, block: &Mat<f64>) -> Result<Mat<f64>, KytoError> {
    let n = block.nrows();
    let mut columns: Vec<Vec<f64>> = (0..block.ncols())
        .map(|column| (0..n).map(|row| block[(row, column)]).collect())
        .collect();
    let basis_columns: Vec<Vec<f64>> = basis.map_or_else(Vec::new, |basis| {
        (0..basis.ncols())
            .map(|column| (0..n).map(|row| basis[(row, column)]).collect())
            .collect()
    });
    let mut kept: Vec<Vec<f64>> = Vec::new();
    for mut column in columns.drain(..) {
        let norm = dot(&column, &column).sqrt();
        if !norm.is_finite() {
            return Err(not_converged());
        }
        if norm == 0.0 {
            continue;
        }
        for _ in 0..2 {
            for other in basis_columns.iter().chain(kept.iter()) {
                let coefficient = dot(other, &column);
                for (value, &o) in column.iter_mut().zip(other) {
                    *value -= coefficient * o;
                }
            }
        }
        let rest = dot(&column, &column).sqrt();
        if rest <= 1e-10 * norm {
            continue;
        }
        for value in &mut column {
            *value /= rest;
        }
        kept.push(column);
    }
    Ok(Mat::from_fn(n, kept.len(), |row, column| kept[column][row]))
}

/// `[left | right]`.
fn concat(left: &Mat<f64>, right: &Mat<f64>) -> Mat<f64> {
    let split = left.ncols();
    Mat::from_fn(left.nrows(), split + right.ncols(), |row, column| {
        if column < split {
            left[(row, column)]
        } else {
            right[(row, column - split)]
        }
    })
}

/// `x . y` summed in [`DOT_CHUNK`] row chunks, the chunks in row order.
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.chunks(DOT_CHUNK)
        .zip(y.chunks(DOT_CHUNK))
        .map(|(a, b)| a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>())
        .sum()
}

/// `1` if the largest absolute value (the first on ties) is positive, else `-1`.
fn sign_of_largest(values: &[f64]) -> f64 {
    let mut largest = 0.0_f64;
    let mut sign = 1.0;
    for &value in values {
        if value.abs() > largest {
            largest = value.abs();
            sign = value.signum();
        }
    }
    sign
}

fn squared(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum()
}

fn cosine_distance(x: &[f64], y: &[f64]) -> f64 {
    let norms =
        (x.iter().map(|v| v * v).sum::<f64>() * y.iter().map(|v| v * v).sum::<f64>()).sqrt();
    if norms == 0.0 {
        return 1.0;
    }
    (1.0 - x.iter().zip(y).map(|(a, b)| a * b).sum::<f64>() / norms).clamp(0.0, 2.0)
}

fn not_converged() -> KytoError {
    KytoError::InvalidArgument {
        function: "umap".to_owned(),
        parameter: "graph".to_owned(),
        value: "the graph's spectral layout".to_owned(),
        problem: "the eigendecomposition did not converge".to_owned(),
        accepted: "a graph of finite weights, or init='random'".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A circulant graph: each vertex joined to the ones `offsets` away, weights in
    /// `(0.5, 1]` from the edge's ends (symmetric).
    fn circulant(n: usize, offsets: &[usize]) -> Csr {
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for vertex in 0..n {
            for &offset in offsets {
                let other = (vertex + offset) % n;
                let low = vertex.min(other);
                let weight =
                    0.5 + 0.5 * ((low * 7919 + offset * 104_729) % 1000 + 1) as f64 / 1000.0;
                rows[vertex].push((other, weight));
                rows[other].push((vertex, weight));
            }
        }
        let mut indptr = vec![0];
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        for mut row in rows {
            row.sort_by_key(|&(column, _)| column);
            for (column, weight) in row {
                indices.push(column);
                weights.push(weight);
            }
            indptr.push(indices.len());
        }
        Csr {
            indptr,
            indices,
            weights,
        }
    }

    fn path_graph(n: usize) -> Csr {
        let mut indptr = vec![0];
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        for row in 0..n {
            if row > 0 {
                indices.push(row - 1);
                weights.push(1.0);
            }
            if row + 1 < n {
                indices.push(row + 1);
                weights.push(1.0);
            }
            indptr.push(indices.len());
        }
        Csr {
            indptr,
            indices,
            weights,
        }
    }

    /// The normalised Laplacian's eigenvectors by faer's dense solver.
    fn dense_reference(csr: &Csr, dim: usize) -> Vec<Vec<f64>> {
        let n = csr.n();
        let degree: Vec<f64> = (0..n)
            .map(|row| {
                csr.weights[csr.indptr[row]..csr.indptr[row + 1]]
                    .iter()
                    .sum()
            })
            .collect();
        let mut laplacian = Mat::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 });
        for row in 0..n {
            for edge in csr.indptr[row]..csr.indptr[row + 1] {
                let column = csr.indices[edge];
                laplacian[(row, column)] -=
                    csr.weights[edge] / (degree[row] * degree[column]).sqrt();
            }
        }
        let eigen = laplacian.self_adjoint_eigen(Side::Lower).unwrap();
        (1..=dim)
            .map(|axis| (0..n).map(|row| eigen.U()[(row, axis)]).collect())
            .collect()
    }

    #[test]
    fn krylov_finds_the_laplacian_eigenvectors() {
        for (n, csr) in [
            (5, path_graph(5)),
            (40, path_graph(40)),
            (300, circulant(300, &[1, 7, 31])),
            (500, circulant(500, &[1, 2, 3, 50])),
        ] {
            let layout = eigen_layout(&csr, 2, 3, 0).unwrap();
            let reference = dense_reference(&csr, 2);
            for (axis, expected) in reference.iter().enumerate() {
                let got: Vec<f64> = (0..n).map(|row| layout[row * 2 + axis]).collect();
                let alignment = dot(&got, expected).abs();
                assert!(
                    (alignment - 1.0).abs() < 1e-6,
                    "n {n} axis {axis}: {alignment}"
                );
            }
        }
    }

    #[test]
    fn the_same_layout_on_any_thread_count() {
        let csr = circulant(3000, &[1, 7, 31, 113]);
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| eigen_layout(&csr, 2, 9, 0).unwrap())
        };
        let one = run(1);
        assert_eq!(one, run(4));
    }

    #[test]
    fn components_are_numbered_by_their_first_vertex() {
        let csr = Csr {
            indptr: vec![0, 1, 1, 1, 2],
            indices: vec![3, 0],
            weights: vec![1.0; 2],
        };
        assert_eq!(components(&csr), (vec![0, 1, 2, 0], 3));
    }

    #[test]
    fn few_components_sit_on_corners() {
        let meta = meta_layout(&[0, 1, 2], 3, 2, None, 0).unwrap();
        assert_eq!(meta, vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0]);
    }

    #[test]
    fn draws_are_in_range() {
        let mut draws = Draws::new(1, 2);
        let values: Vec<f64> = (0..10_000).map(|_| draws.uniform()).collect();
        assert!(values.iter().all(|v| (0.0..1.0).contains(v)));
        let normals: Vec<f64> = (0..10_000).map(|_| draws.normal()).collect();
        let mean = normals.iter().sum::<f64>() / 1e4;
        let variance = normals.iter().map(|v| v * v).sum::<f64>() / 1e4;
        assert!(mean.abs() < 0.05 && (variance - 1.0).abs() < 0.05);
    }
}
