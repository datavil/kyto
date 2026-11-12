//! The loess fit of `seurat_v3` highly variable genes (task 6.3): scanpy fits
//! `log10(variance)` against `log10(mean)` with `skmisc.loess.loess(x, y, span=0.3,
//! degree=2)`, whose defaults are netlib `dloess`'s (Cleveland, Grosse and Shyu): one
//! predictor, local quadratic fits, tricube weights, Gaussian family (no robustness
//! iterations) and `surface="interpolate"`, `cell=0.2`.
//!
//! Does: the `interpolate` surface in one dimension, as `dloess` builds it. A k-d tree
//! splits the sorted predictor at its median (`m = (l + u) / 2` in 1-based positions,
//! moved to the nearest position whose value differs from the next one, the 2006
//! tie fix) until a cell holds at most `floor(n * span * cell)` points; its vertices
//! are the bounding box (widened by 0.5 % of the range) and every split value. At each
//! vertex a local quadratic is fitted by weighted least squares over the
//! `floor(n * span)` nearest points (tricube weights of the distance over the
//! farthest one's), giving the surface's value and slope there; a point's fitted value
//! is the cubic Hermite interpolation between the vertices around it. Checked against
//! `skmisc` (`tests/loess_parity.rs`): the same values to about 1e-14.
//! Does not: other degrees, spans above 1 (no `seurat_v3` passes one), robustness
//! iterations, `surface="direct"`, standard errors, or more than one predictor.

/// `dloess`'s default `cell`: a k-d tree cell holds at most `n * span * cell` points.
const CELL: f64 = 0.2;

/// The fitted values of a loess of `y` on `x` (one point each, same length): degree 2,
/// `span` the fraction of points in each local fit (`0 < span <= 1`), as `skmisc`'s
/// `loess(x, y, span=span, degree=2).fit()` then `outputs.fitted_values`.
///
/// Returns NaN for every point when a local fit has no point with a positive weight
/// (fewer distinct values than the fit needs, where `dloess` stops with "span too
/// small"); an empty `x` gives an empty result.
///
/// # Panics
/// If `x` and `y` differ in length (a kyto bug: the caller pairs them).
pub fn loess_fitted(x: &[f64], y: &[f64], span: f64) -> Vec<f64> {
    assert_eq!(x.len(), y.len(), "loess: one y per x");
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mut sorted = x.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let vertices = vertices(&sorted, span);
    // Fortran's `ifloor(n * f)`; f * cell is one product first (`v(2)` in lowesd).
    let n_local = ((n as f64 * span).floor() as usize).clamp(1, n);
    let fits: Vec<(f64, f64)> = vertices
        .iter()
        .map(|&vertex| local_fit(x, y, vertex, n_local, span))
        .collect();
    x.iter()
        .map(|&z| interpolate(&vertices, &fits, z))
        .collect()
}

/// The k-d tree's vertices over `sorted` (the predictor, ascending), ascending.
fn vertices(sorted: &[f64], span: f64) -> Vec<f64> {
    let n = sorted.len();
    let (low, high) = (sorted[0], sorted[n - 1]);
    // ehg126: widen the bounding box a little.
    let margin = 0.005 * (high - low).max(1e-10 * low.abs().max(high.abs()) + 1e-30);
    let (low, high) = (low - margin, high + margin);
    let cell_points = (n as f64 * (span * CELL)).floor() as usize;
    let mut vertices = vec![low, high];
    // Cells as 0-based inclusive position ranges of `sorted`, with their bounds.
    let mut cells = vec![(0_usize, n - 1, low, high)];
    while let Some((first, last, lower, upper)) = cells.pop() {
        // An empty cell (a split at its parent's last point) or a small one is a leaf.
        if first > last || last - first < cell_points {
            continue;
        }
        // ehg129: m = (l + u) / 2 with 1-based l and u, here 0-based.
        let mut split = (first + last + 2) / 2 - 1;
        // The tie fix (btyner 2006): the nearest position, trying +0, +1, -1, +2, -2,
        // ..., whose value differs from the next one.
        let mut offset: isize = 0;
        loop {
            let position = split as isize + offset;
            if position >= last as isize || position < first as isize {
                break;
            }
            let position = position as usize;
            if sorted[position] == sorted[position + 1] {
                offset = -offset;
                if offset >= 0 {
                    offset += 1;
                }
            } else {
                split = position;
                break;
            }
        }
        let value = sorted[split];
        if value == lower || value == upper {
            continue;
        }
        vertices.push(value);
        cells.push((first, split, lower, value));
        cells.push((split + 1, last, value, upper));
    }
    vertices.sort_unstable_by(f64::total_cmp);
    vertices.dedup();
    vertices
}

/// The local quadratic at `vertex`: its value and slope there (NaN without a point of
/// positive weight).
fn local_fit(x: &[f64], y: &[f64], vertex: f64, n_local: usize, span: f64) -> (f64, f64) {
    let distances: Vec<f64> = x.iter().map(|&value| (value - vertex).powi(2)).collect();
    let mut order = distances.clone();
    let (_, nth, _) = order.select_nth_unstable_by(n_local - 1, f64::total_cmp);
    // ehg127: the squared distance of the q-th nearest point, enlarged for spans > 1.
    let radius = *nth * span.max(1.0);
    if radius.is_nan() || radius <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    // Rows of the weighted design [1, d, d^2] (d = x - vertex) and the weighted y,
    // over the points strictly inside the radius (the others weigh 0).
    let mut rows: Vec<[f64; 4]> = Vec::with_capacity(n_local);
    for ((&value, &response), &distance) in x.iter().zip(y).zip(&distances) {
        if distance < radius {
            let scaled = (distance / radius).sqrt();
            let weight = (1.0 - scaled.powi(3)).powi(3).sqrt();
            let offset = value - vertex;
            rows.push([
                weight,
                weight * offset,
                weight * offset * offset,
                weight * response,
            ]);
        }
    }
    if rows.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let coefficients = least_squares(&mut rows);
    (coefficients[0], coefficients[1])
}

/// The minimum-norm least-squares solution of `rows` (each `[a0, a1, a2, b]`):
/// Householder QR, then the pseudo-inverse of `R` through its singular values, those
/// at most `100 * eps * largest` treated as zero (as `dloess`'s `ehg127`).
fn least_squares(rows: &mut [[f64; 4]]) -> [f64; 3] {
    let m = rows.len();
    let mut r = [[0.0_f64; 3]; 3];
    let mut qtb = [0.0_f64; 3];
    for column in 0..3 {
        if column >= m {
            break;
        }
        let norm = rows[column..]
            .iter()
            .map(|row| row[column] * row[column])
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            let alpha = if rows[column][column] > 0.0 {
                -norm
            } else {
                norm
            };
            // v = x - alpha e1, stored in place; H = I - 2 v v^T / (v^T v).
            rows[column][column] -= alpha;
            let v_norm2: f64 = rows[column..]
                .iter()
                .map(|row| row[column] * row[column])
                .sum();
            if v_norm2 > 0.0 {
                for target in column + 1..4 {
                    let dot: f64 = rows[column..]
                        .iter()
                        .map(|row| row[column] * row[target])
                        .sum();
                    let factor = 2.0 * dot / v_norm2;
                    for row in rows[column..].iter_mut() {
                        row[target] -= factor * row[column];
                    }
                }
            }
            rows[column][column] = alpha;
        }
        for target in column..3 {
            r[column][target] = rows[column][target];
        }
        qtb[column] = rows[column][3];
    }
    let (u, sigma, v) = svd3(r);
    let tolerance = 100.0 * f64::EPSILON * sigma.iter().fold(0.0_f64, |a, &b| a.max(b));
    let mut solution = [0.0_f64; 3];
    for k in 0..3 {
        if sigma[k] > tolerance {
            let projection: f64 = (0..3).map(|i| u[i][k] * qtb[i]).sum::<f64>() / sigma[k];
            for (i, value) in solution.iter_mut().enumerate() {
                *value += v[i][k] * projection;
            }
        }
    }
    solution
}

/// The singular value decomposition `a = u diag(sigma) v^T` of a 3 x 3 matrix by
/// one-sided Jacobi rotations (columns of `u` and `v`, `sigma` not sorted).
fn svd3(a: [[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3], [[f64; 3]; 3]) {
    let mut w = a;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..60 {
        let mut rotated = false;
        for p in 0..2 {
            for q in p + 1..3 {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for row in &w {
                    alpha += row[p] * row[p];
                    beta += row[q] * row[q];
                    gamma += row[p] * row[q];
                }
                if gamma == 0.0 || gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let t = if zeta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for row in w.iter_mut().chain(v.iter_mut()) {
                    let (left, right) = (row[p], row[q]);
                    row[p] = c * left - s * right;
                    row[q] = s * left + c * right;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let mut sigma = [0.0; 3];
    let mut u = [[0.0; 3]; 3];
    for k in 0..3 {
        sigma[k] = w.iter().map(|row| row[k] * row[k]).sum::<f64>().sqrt();
        if sigma[k] > 0.0 {
            for i in 0..3 {
                u[i][k] = w[i][k] / sigma[k];
            }
        }
    }
    (u, sigma, v)
}

/// The surface at `z`: the cubic Hermite interpolation of the vertex fits around it.
fn interpolate(vertices: &[f64], fits: &[(f64, f64)], z: f64) -> f64 {
    // The first vertex at or above z; z lies in the bounding box.
    let upper = vertices
        .partition_point(|&vertex| vertex < z)
        .clamp(1, vertices.len() - 1);
    let (low, high) = (vertices[upper - 1], vertices[upper]);
    let ((value_low, slope_low), (value_high, slope_high)) = (fits[upper - 1], fits[upper]);
    let width = high - low;
    let t = (z - low) / width;
    let phi_low = (1.0 - t) * (1.0 - t) * (1.0 + 2.0 * t);
    let phi_high = t * t * (3.0 - 2.0 * t);
    let psi_low = t * (1.0 - t) * (1.0 - t);
    let psi_high = t * t * (t - 1.0);
    phi_low * value_low
        + phi_high * value_high
        + (psi_low * slope_low + psi_high * slope_high) * width
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quadratic_is_fitted_exactly() {
        let x: Vec<f64> = (0..200).map(|i| f64::from(i) / 10.0 - 7.0).collect();
        let y: Vec<f64> = x.iter().map(|x| 0.5 * x * x - 2.0 * x + 1.0).collect();
        for (fitted, expected) in loess_fitted(&x, &y, 0.3).iter().zip(&y) {
            assert!((fitted - expected).abs() < 1e-9, "{fitted} vs {expected}");
        }
    }

    #[test]
    fn vertices_split_at_medians_and_skip_ties() {
        // 10 points, span 0.5: a cell of at most floor(10 * 0.1) = 1 point stops.
        let sorted: Vec<f64> = (0..10).map(f64::from).collect();
        let vertices = vertices(&sorted, 0.5);
        assert_eq!(vertices.first(), Some(&-0.045));
        assert_eq!(vertices.last(), Some(&9.045));
        assert!(vertices.contains(&4.0), "the first split is the 5th value");
        // All equal but one: no split value at the tied run's ends.
        let tied = [1.0, 1.0, 1.0, 1.0, 2.0];
        let vertices = super::vertices(&tied, 1.0);
        assert!(vertices.iter().all(|&v| v != 2.0));
    }

    #[test]
    fn too_few_distinct_points_give_nan() {
        let fitted = loess_fitted(&[1.0, 1.0, 1.0], &[2.0, 3.0, 4.0], 0.3);
        assert!(fitted.iter().all(|value| value.is_nan()));
        assert!(loess_fitted(&[], &[], 0.3).is_empty());
    }

    #[test]
    fn svd3_reconstructs() {
        let a = [[4.0, 1.0, 2.0], [0.0, 3.0, -1.0], [0.0, 0.0, 1e-3]];
        let (u, sigma, v) = svd3(a);
        for i in 0..3 {
            for j in 0..3 {
                let value: f64 = (0..3).map(|k| u[i][k] * sigma[k] * v[j][k]).sum();
                assert!((value - a[i][j]).abs() < 1e-12);
            }
        }
    }
}
