//! The small dense steps of the streamed PCA (task 6.5), with faer on one thread.
//!
//! Does: hold row-major `f64` matrices ([`Dense`]: genes x columns, as the passes fold
//! them); grow the PCA's Krylov basis by one block (project the block off the basis
//! twice, orthonormalise it by a thin SVD and drop the directions it no longer adds:
//! [`extend_basis`]); and find the leading eigenpairs of the centred covariance on the
//! basis (Rayleigh-Ritz, [`rayleigh_ritz`]), completed by orthonormal directions of
//! zero variance when the data has fewer than `n_components` directions, with
//! scikit-learn's sign rule (`svd_flip(u_based_decision=False)`: the largest absolute
//! loading of each component is positive). faer runs without its `rayon` feature, so
//! every step is sequential and gives the same bits on any thread count (R11).
//! Does not: read the data (see `pca.rs`).

use faer::{Mat, Side};
use kyto_error::KytoError;

/// A singular value of a new block at most this times the block's norm is a direction
/// the basis already holds (rounding), and is dropped.
const DROP_TOLERANCE: f64 = 1e-10;

/// A row-major `rows x columns` matrix of `f64`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Dense {
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) values: Vec<f64>,
}

impl Dense {
    /// A `rows x columns` matrix of zeros.
    pub(crate) fn zeros(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            values: vec![0.0; rows * columns],
        }
    }

    /// Row `row`.
    pub(crate) fn row(&self, row: usize) -> &[f64] {
        &self.values[row * self.columns..(row + 1) * self.columns]
    }

    /// Row `row`, to change.
    pub(crate) fn row_mut(&mut self, row: usize) -> &mut [f64] {
        &mut self.values[row * self.columns..(row + 1) * self.columns]
    }

    /// `self` beside `right` (the same rows): `[self | right]`.
    pub(crate) fn beside(&self, right: &Self) -> Self {
        debug_assert_eq!(self.rows, right.rows);
        let columns = self.columns + right.columns;
        let mut values = Vec::with_capacity(self.rows * columns);
        for row in 0..self.rows {
            values.extend_from_slice(self.row(row));
            values.extend_from_slice(right.row(row));
        }
        Self {
            rows: self.rows,
            columns,
            values,
        }
    }

    fn to_faer(&self) -> Mat<f64> {
        Mat::from_fn(self.rows, self.columns, |row, column| {
            self.values[row * self.columns + column]
        })
    }

    fn from_faer(matrix: faer::MatRef<'_, f64>) -> Self {
        let mut values = Vec::with_capacity(matrix.nrows() * matrix.ncols());
        for row in 0..matrix.nrows() {
            for column in 0..matrix.ncols() {
                values.push(matrix[(row, column)]);
            }
        }
        Self {
            rows: matrix.nrows(),
            columns: matrix.ncols(),
            values,
        }
    }
}

/// The orthonormal directions `block` adds to `basis` (orthonormal columns, the same
/// rows): `block` with the basis projected off twice (classical Gram-Schmidt run twice
/// is as stable as modified), then the left singular vectors of the rest whose
/// singular value is above [`DROP_TOLERANCE`] times `block`'s norm. No columns when the
/// basis already spans `block`.
///
/// # Errors
/// `KY-E215` if the SVD does not converge (a non-finite block).
pub(crate) fn extend_basis(basis: &Dense, block: &Dense) -> Result<Dense, KytoError> {
    let mut rest = block.to_faer();
    let scale = rest.norm_l2();
    if scale == 0.0 || block.columns == 0 {
        return Ok(Dense::zeros(block.rows, 0));
    }
    if basis.columns > 0 {
        let basis = basis.to_faer();
        for _ in 0..2 {
            let projection = basis.transpose() * &rest;
            rest -= &basis * projection;
        }
    }
    let svd = rest
        .thin_svd()
        .map_err(|_| not_converged("orthonormalising a Krylov block"))?;
    let singular = svd.S().column_vector();
    let keep = (0..singular.nrows())
        .take_while(|&index| singular[index] > DROP_TOLERANCE * scale)
        .count();
    Ok(Dense::from_faer(svd.U().subcols(0, keep)))
}

/// The `n_components` leading eigenpairs of the symmetric matrix `C` known on the
/// orthonormal `basis` through `products` (`C x basis`): the eigenvalues, largest
/// first (negative rounding clamped to 0), and the loadings (`rows x n_components`,
/// orthonormal columns), each flipped so its largest absolute value is positive.
/// Directions beyond the basis complete the loadings with eigenvalue 0 (the basis
/// spans every direction of non-zero variance when it is narrower than
/// `n_components`).
///
/// # Errors
/// `KY-E215` if the eigendecomposition does not converge (a non-finite matrix).
pub(crate) fn rayleigh_ritz(
    basis: &Dense,
    products: &Dense,
    n_components: usize,
) -> Result<(Vec<f64>, Dense), KytoError> {
    let mut loadings = Dense::zeros(basis.rows, 0);
    let mut eigenvalues = Vec::with_capacity(n_components);
    if basis.columns > 0 {
        let basis_matrix = basis.to_faer();
        let projected = basis_matrix.transpose() * products.to_faer();
        let symmetric = Mat::from_fn(projected.nrows(), projected.ncols(), |row, column| {
            0.5 * (projected[(row, column)] + projected[(column, row)])
        });
        let eigen = symmetric
            .self_adjoint_eigen(Side::Lower)
            .map_err(|_| not_converged("the Rayleigh-Ritz eigendecomposition"))?;
        let values = eigen.S().column_vector();
        let n = values.nrows();
        let taken = n_components.min(n);
        // faer sorts eigenvalues increasing: the largest are last.
        let order: Vec<usize> = (0..taken).map(|index| n - 1 - index).collect();
        eigenvalues.extend(order.iter().map(|&index| values[index].max(0.0)));
        let vectors = eigen.U();
        let chosen = Mat::from_fn(n, taken, |row, column| vectors[(row, order[column])]);
        loadings = Dense::from_faer((&basis_matrix * chosen).as_ref());
    }
    let missing = n_components - eigenvalues.len();
    if missing > 0 {
        loadings = loadings.beside(&complement(&loadings, missing));
        eigenvalues.resize(n_components, 0.0);
    }
    flip_signs(&mut loadings);
    Ok((eigenvalues, loadings))
}

/// `count` orthonormal columns orthogonal to `columns` (orthonormal): each time the
/// unit vector of the gene least covered so far (the first such gene on ties), with the
/// columns projected off twice.
fn complement(columns: &Dense, count: usize) -> Dense {
    let mut all = columns.clone();
    let mut added = Dense::zeros(columns.rows, 0);
    for _ in 0..count {
        let covered = |row: usize| all.row(row).iter().map(|value| value * value).sum::<f64>();
        let gene = (0..all.rows)
            .min_by(|&left, &right| covered(left).total_cmp(&covered(right)))
            .unwrap_or(0);
        let mut vector = vec![0.0; all.rows];
        vector[gene] = 1.0;
        for _ in 0..2 {
            let coefficients: Vec<f64> = (0..all.columns)
                .map(|column| {
                    (0..all.rows)
                        .map(|row| all.values[row * all.columns + column] * vector[row])
                        .sum()
                })
                .collect();
            for (row, value) in vector.iter_mut().enumerate() {
                *value -= all
                    .row(row)
                    .iter()
                    .zip(&coefficients)
                    .map(|(left, right)| left * right)
                    .sum::<f64>();
            }
        }
        let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
        let column = Dense {
            rows: all.rows,
            columns: 1,
            values: vector.iter().map(|value| value / norm).collect(),
        };
        all = all.beside(&column);
        added = added.beside(&column);
    }
    added
}

/// scikit-learn's `svd_flip(u_based_decision=False)` on the loadings: each column's
/// largest absolute value (the first on ties) made positive.
fn flip_signs(loadings: &mut Dense) {
    for column in 0..loadings.columns {
        let mut largest = 0.0_f64;
        let mut sign = 1.0;
        for row in 0..loadings.rows {
            let value = loadings.values[row * loadings.columns + column];
            if value.abs() > largest {
                largest = value.abs();
                sign = value.signum();
            }
        }
        if sign < 0.0 {
            for row in 0..loadings.rows {
                loadings.values[row * loadings.columns + column] *= -1.0;
            }
        }
    }
}

fn not_converged(step: &str) -> KytoError {
    KytoError::InvalidArgument {
        function: "pca".to_owned(),
        parameter: "layer".to_owned(),
        value: "the PCA's input".to_owned(),
        problem: format!("{step} did not converge"),
        accepted: "a layer of finite values".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(rows: usize, columns: usize, values: &[f64]) -> Dense {
        Dense {
            rows,
            columns,
            values: values.to_vec(),
        }
    }

    fn gram(matrix: &Dense) -> Vec<f64> {
        let mut out = vec![0.0; matrix.columns * matrix.columns];
        for left in 0..matrix.columns {
            for right in 0..matrix.columns {
                out[left * matrix.columns + right] = (0..matrix.rows)
                    .map(|row| matrix.row(row)[left] * matrix.row(row)[right])
                    .sum();
            }
        }
        out
    }

    fn assert_orthonormal(matrix: &Dense) {
        let gram = gram(matrix);
        for left in 0..matrix.columns {
            for right in 0..matrix.columns {
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!((gram[left * matrix.columns + right] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn a_block_the_basis_spans_adds_nothing() {
        let basis = dense(3, 1, &[1.0, 0.0, 0.0]);
        let block = dense(3, 2, &[2.0, 4.0, 0.0, 1.0, 0.0, 0.0]);
        let added = extend_basis(&basis, &block).unwrap();
        // Only the second gene's direction is new.
        assert_eq!(added.columns, 1);
        assert!((added.row(1)[0].abs() - 1.0).abs() < 1e-12);
        let empty = extend_basis(&basis.beside(&added), &block).unwrap();
        assert_eq!(empty.columns, 0);
        assert_eq!(
            extend_basis(&basis, &Dense::zeros(3, 2)).unwrap().columns,
            0
        );
    }

    #[test]
    fn rayleigh_ritz_finds_the_eigenpairs_and_flips_signs() {
        // C = diag(1, 3, 2) on the whole space.
        let basis = dense(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let products = dense(3, 3, &[1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0]);
        let (values, loadings) = rayleigh_ritz(&basis, &products, 2).unwrap();
        assert!((values[0] - 3.0).abs() < 1e-12 && (values[1] - 2.0).abs() < 1e-12);
        assert!((loadings.row(1)[0] - 1.0).abs() < 1e-12);
        assert!((loadings.row(2)[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn missing_directions_have_zero_variance() {
        let basis = dense(3, 1, &[0.0, -1.0, 0.0]);
        let products = dense(3, 1, &[0.0, -5.0, 0.0]);
        let (values, loadings) = rayleigh_ritz(&basis, &products, 3).unwrap();
        assert_eq!(values, vec![5.0, 0.0, 0.0]);
        assert_orthonormal(&loadings);
        // Flipped: the largest value of each column is positive.
        assert!((loadings.row(1)[0] - 1.0).abs() < 1e-12);
    }
}
