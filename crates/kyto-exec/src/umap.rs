//! The UMAP layout of a connectivity graph (task 6.9), as scanpy 1.12.5's `tl.umap`
//! runs umap-learn 0.5.12's `simplicial_set_embedding`.
//!
//! Does, in umap's order and precision:
//! - fit `a`, `b` of the curve `1 / (1 + a d^(2b))` to umap's offset exponential
//!   (`find_ab_params`: least squares on 300 points, Levenberg-Marquardt here, scipy's
//!   `curve_fit` there);
//! - drop the edges whose `float32` weight is below the largest over the epochs
//!   (`float32` arithmetic, as numpy computes it), and give each kept edge its epochs
//!   per sample (`n_epochs / (n_epochs * w / w_max)`, the product in `float32`);
//! - start from the spectral layout (`umap_spectral.rs`) scaled so its largest
//!   absolute value is 10, plus normal noise of scale 1e-4, or from uniform positions
//!   in `[-10, 10)`; then scale each dimension to `[0, 10]` (`float32`);
//! - run umap's SGD (`optimize_layout_euclidean` with `move_other`, as scanpy calls it
//!   with `parallel=False`): epoch by epoch, edge by edge in CSR order, the attraction
//!   moving both ends and the negative samples moving the head, gradients in `f64`
//!   clipped to 4, positions in `float32`, the learning rate falling linearly.
//!
//! The SGD is sequential (as umap-learn's with a seed), so the layout is the same on
//! any thread count (R11). Negative samples draw from umap's Tausworthe generator
//! (`tau_rand_int`), one state per vertex seeded from kyto's seed (SplitMix64), not
//! from numpy's `RandomState`: kyto's layout is not umap's bit for bit (deviations.md).
//! Does not: densMAP, other output metrics, `init_pos` from an embedding or PAGA.

use std::sync::Arc;

use kyto_error::KytoError;
use kyto_plan::{Metric, UMAP_MAX_COMPONENTS, Umap, UmapInit};

use crate::connectivities::ConnectivityGraph;
use crate::run_control::RunControl;
use crate::umap_spectral::{Draws, spectral_layout};

/// scanpy's epochs up to this many cells (500), and 200 above.
pub const UMAP_SMALL_GRAPH_CELLS: usize = 10_000;

/// A UMAP layout.
#[derive(Debug, Clone, PartialEq)]
pub struct UmapLayout {
    n_components: usize,
    embedding: Vec<f32>,
    a: f64,
    b: f64,
}

impl UmapLayout {
    /// The layout's dimensions.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// The positions, row-major cells x [`UmapLayout::n_components`] (scanpy's
    /// `X_umap`, in `float32`).
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    /// The curve's `a` (scanpy's `uns['umap']['params']['a']`).
    pub fn a(&self) -> f64 {
        self.a
    }

    /// The curve's `b` (scanpy's `uns['umap']['params']['b']`).
    pub fn b(&self) -> f64 {
        self.b
    }
}

/// What a `umap` node computed: the layout of its graph's rows.
#[derive(Debug, Clone, PartialEq)]
pub struct UmapStatistics {
    pub(crate) cells: Arc<Vec<u64>>,
    pub(crate) layout: UmapLayout,
}

impl UmapStatistics {
    /// The store table position of each row: the graph's rows (the `neighbors` node's
    /// `cells()`).
    pub fn cells(&self) -> &[u64] {
        &self.cells
    }

    /// The layout.
    pub fn layout(&self) -> &UmapLayout {
        &self.layout
    }
}

/// The UMAP layout of `graph` as `params` asks (`params.graph` and `params.into` are
/// not read). `data` (row-major, `width` values a row, searched with `metric`) is what
/// the neighbours were found in: umap places more than `2 n_components` connected
/// components by their centroids in it; without it they are placed at random. Runs
/// the spectral initialisation's products on the current rayon pool.
///
/// # Errors
/// `KY-E215` for `n_components` of 0 or above [`UMAP_MAX_COMPONENTS`], a `spread` or
/// `alpha` that is not positive and finite, a `min_dist` or `gamma` that is negative or not finite, `min_dist` above
/// `spread`, `data` whose length is not `width` times the graph's rows, or a spectral
/// layout that does not converge.
pub fn umap_layout(
    graph: &ConnectivityGraph,
    data: Option<(&[f32], usize, Metric)>,
    params: &Umap,
) -> Result<UmapLayout, KytoError> {
    layout(graph, data, params, &RunControl::new())
}

/// [`umap_layout`], cancelled through `control` (`KY-E503`, checked every epoch).
pub(crate) fn layout(
    graph: &ConnectivityGraph,
    data: Option<(&[f32], usize, Metric)>,
    params: &Umap,
    control: &RunControl,
) -> Result<UmapLayout, KytoError> {
    check(params)?;
    let n = graph.n_cells();
    if let Some((values, width, _)) = data
        && values.len() != n.saturating_mul(width)
    {
        return Err(invalid(
            "data",
            &format!("{} values", values.len()),
            &format!(
                "the graph has {n} rows, so {width} values a row make {}",
                n * width
            ),
            "the embedding the graph's neighbours were searched in",
        ));
    }
    let dim = usize::try_from(params.n_components).unwrap_or(usize::MAX);
    let (a, b) = find_ab_params(params.spread, params.min_dist);
    let default_epochs = if n <= UMAP_SMALL_GRAPH_CELLS {
        500
    } else {
        200
    };
    let n_epochs = params.n_epochs.map_or(default_epochs, |value| {
        usize::try_from(value).unwrap_or(usize::MAX)
    });
    let keep = kept_edges(graph.weights(), n_epochs, default_epochs);
    let mut embedding = initial_layout(graph, &keep, dim, data, params)?;
    normalize(&mut embedding, n, dim);
    let edges = Edges::new(graph, &keep, n_epochs);
    optimize(
        &mut embedding,
        &edges,
        &Sgd {
            n,
            dim,
            n_epochs,
            a,
            b,
            gamma: params.gamma,
            initial_alpha: params.alpha,
            negative_sample_rate: params.negative_sample_rate,
            seed: params.seed,
        },
        control,
    )?;
    Ok(UmapLayout {
        n_components: dim,
        embedding,
        a,
        b,
    })
}

/// The plan's checks again, for callers of [`umap_layout`].
fn check(params: &Umap) -> Result<(), KytoError> {
    if params.n_components == 0 || params.n_components > UMAP_MAX_COMPONENTS {
        return Err(invalid(
            "n_components",
            &params.n_components.to_string(),
            &format!("a layout has from 1 to {UMAP_MAX_COMPONENTS} dimensions"),
            &format!("an int from 1 to {UMAP_MAX_COMPONENTS}"),
        ));
    }
    for (name, value) in [("spread", params.spread), ("alpha", params.alpha)] {
        if !(value.is_finite() && value > 0.0) {
            return Err(invalid(
                name,
                &value.to_string(),
                "it is not a positive finite number",
                "a positive finite float",
            ));
        }
    }
    for (name, value) in [("min_dist", params.min_dist), ("gamma", params.gamma)] {
        if !(value.is_finite() && value >= 0.0) {
            return Err(invalid(
                name,
                &value.to_string(),
                "it is not a non-negative finite number",
                "a finite float of at least 0",
            ));
        }
    }
    if params.min_dist > params.spread {
        return Err(invalid(
            "min_dist",
            &params.min_dist.to_string(),
            &format!("it is above spread={}", params.spread),
            "a float from 0 to spread",
        ));
    }
    Ok(())
}

fn invalid(parameter: &str, value: &str, problem: &str, accepted: &str) -> KytoError {
    KytoError::InvalidArgument {
        function: "umap".to_owned(),
        parameter: parameter.to_owned(),
        value: value.to_owned(),
        problem: problem.to_owned(),
        accepted: accepted.to_owned(),
    }
}

/// umap's `find_ab_params(spread, min_dist)`: the least-squares `a`, `b` of
/// `1 / (1 + a x^(2b))` against 1 below `min_dist` and `exp(-(x - min_dist) / spread)`
/// above, on 300 points from 0 to `3 spread` (Levenberg-Marquardt from `(1, 1)`,
/// scipy's start).
pub fn find_ab_params(spread: f64, min_dist: f64) -> (f64, f64) {
    const POINTS: usize = 300;
    let stop = spread * 3.0;
    let step = stop / (POINTS - 1) as f64;
    let x: Vec<f64> = (0..POINTS)
        .map(|index| {
            if index + 1 == POINTS {
                stop
            } else {
                index as f64 * step
            }
        })
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&value| {
            if value < min_dist {
                1.0
            } else {
                (-(value - min_dist) / spread).exp()
            }
        })
        .collect();
    // Residuals and the Jacobian's columns at (a, b).
    let evaluate = |a: f64, b: f64| {
        let mut cost = 0.0;
        let mut jtj = [0.0; 3];
        let mut jtr = [0.0; 2];
        for (&xi, &yi) in x.iter().zip(&y) {
            let power = if xi > 0.0 { xi.powf(2.0 * b) } else { 0.0 };
            let denominator = 1.0 + a * power;
            let residual = 1.0 / denominator - yi;
            let squared = denominator * denominator;
            let da = -power / squared;
            let db = if xi > 0.0 {
                -a * power * 2.0 * xi.ln() / squared
            } else {
                0.0
            };
            cost += residual * residual;
            jtj[0] += da * da;
            jtj[1] += da * db;
            jtj[2] += db * db;
            jtr[0] += da * residual;
            jtr[1] += db * residual;
        }
        (cost, jtj, jtr)
    };
    let (mut a, mut b) = (1.0_f64, 1.0_f64);
    let (mut cost, mut jtj, mut jtr) = evaluate(a, b);
    let mut damping = 1e-3;
    for _ in 0..1000 {
        let m00 = jtj[0] * (1.0 + damping);
        let m11 = jtj[2] * (1.0 + damping);
        let m01 = jtj[1];
        let determinant = m00 * m11 - m01 * m01;
        if determinant == 0.0 || !determinant.is_finite() {
            break;
        }
        let step_a = -(m11 * jtr[0] - m01 * jtr[1]) / determinant;
        let step_b = -(m00 * jtr[1] - m01 * jtr[0]) / determinant;
        let (trial_a, trial_b) = (a + step_a, b + step_b);
        let (trial_cost, trial_jtj, trial_jtr) = evaluate(trial_a, trial_b);
        if trial_cost.is_finite() && trial_cost <= cost {
            let small = step_a.abs() <= 1e-15 * a.abs().max(1e-300)
                && step_b.abs() <= 1e-15 * b.abs().max(1e-300);
            (a, b, cost, jtj, jtr) = (trial_a, trial_b, trial_cost, trial_jtj, trial_jtr);
            damping = (damping / 10.0).max(1e-15);
            if small {
                break;
            }
        } else {
            damping *= 10.0;
            if damping > 1e15 {
                break;
            }
        }
    }
    (a, b)
}

/// Which edges umap keeps: `weight >= max / epochs` in `float32` (numpy divides the
/// `float32` maximum by a Python float and keeps `float32`), with the default epochs
/// when 10 or fewer are asked.
fn kept_edges(weights: &[f32], n_epochs: usize, default_epochs: usize) -> Vec<bool> {
    let largest = weights.iter().copied().fold(0.0_f32, f32::max);
    let epochs = if n_epochs > 10 {
        n_epochs
    } else {
        default_epochs
    };
    let threshold = largest / epochs as f32;
    weights.iter().map(|&weight| weight >= threshold).collect()
}

/// The start of the layout: spectral (scaled to 10 with noise) or uniform.
fn initial_layout(
    graph: &ConnectivityGraph,
    keep: &[bool],
    dim: usize,
    data: Option<(&[f32], usize, Metric)>,
    params: &Umap,
) -> Result<Vec<f32>, KytoError> {
    let n = graph.n_cells();
    match params.init {
        UmapInit::Random => {
            let mut draws = Draws::new(params.seed, 0x5241_4E44);
            Ok((0..n * dim)
                .map(|_| draws.uniform_in(-10.0, 10.0) as f32)
                .collect())
        }
        UmapInit::Spectral => {
            let coordinates = spectral_layout(graph, keep, dim, data, params.seed)?;
            // umap's `noisy_scale_coords(max_coord=10, noise=0.0001)`.
            let largest = coordinates
                .iter()
                .fold(0.0_f64, |most, v| most.max(v.abs()));
            let expansion = if largest > 0.0 { 10.0 / largest } else { 0.0 };
            let mut draws = Draws::new(params.seed, 0x4E4F_4953);
            Ok(coordinates
                .iter()
                .map(|&value| (value * expansion) as f32 + (draws.normal() * 1e-4) as f32)
                .collect())
        }
    }
}

/// Each dimension scaled to `[0, 10]`: `10 (x - min) / (max - min)` in `float32` (a
/// dimension of one value becomes 0; umap's would be NaN).
fn normalize(embedding: &mut [f32], n: usize, dim: usize) {
    for axis in 0..dim {
        let column = (0..n).map(|row| embedding[row * dim + axis]);
        let low = column.clone().fold(f32::INFINITY, f32::min);
        let high = column.fold(f32::NEG_INFINITY, f32::max);
        let range = high - low;
        for row in 0..n {
            let value = &mut embedding[row * dim + axis];
            *value = if range > 0.0 {
                10.0 * (*value - low) / range
            } else {
                0.0
            };
        }
    }
}

/// The kept edges in CSR order and their sampling schedule.
struct Edges {
    head: Vec<usize>,
    tail: Vec<usize>,
    epochs_per_sample: Vec<f64>,
}

impl Edges {
    /// umap's `make_epochs_per_sample(weights, n_epochs)`: `n_epochs / n_samples` with
    /// `n_samples = n_epochs * (w / w_max)` in `float32`, -1 where it is 0.
    fn new(graph: &ConnectivityGraph, keep: &[bool], n_epochs: usize) -> Self {
        let mut head = Vec::new();
        let mut tail = Vec::new();
        let mut weights = Vec::new();
        for row in 0..graph.n_cells() {
            let start = graph.indptr()[row] as usize;
            let end = graph.indptr()[row + 1] as usize;
            for edge in (start..end).filter(|&edge| keep[edge]) {
                head.push(row);
                tail.push(graph.indices()[edge] as usize);
                weights.push(graph.weights()[edge]);
            }
        }
        let largest = weights.iter().copied().fold(0.0_f32, f32::max);
        let epochs_per_sample = weights
            .iter()
            .map(|&weight| {
                let samples = n_epochs as f32 * (weight / largest);
                if samples > 0.0 {
                    n_epochs as f64 / f64::from(samples)
                } else {
                    -1.0
                }
            })
            .collect();
        Self {
            head,
            tail,
            epochs_per_sample,
        }
    }
}

/// The SGD's constants.
struct Sgd {
    n: usize,
    dim: usize,
    n_epochs: usize,
    a: f64,
    b: f64,
    gamma: f64,
    initial_alpha: f64,
    negative_sample_rate: u64,
    seed: u64,
}

/// umap's `clip`: the gradient bounded to `[-4, 4]`.
fn clip(value: f64) -> f64 {
    value.clamp(-4.0, 4.0)
}

/// umap's `rdist`: the squared distance in `float32`.
fn rdist(embedding: &[f32], j: usize, k: usize, dim: usize) -> f32 {
    let mut result = 0.0_f32;
    for axis in 0..dim {
        let difference = embedding[j * dim + axis] - embedding[k * dim + axis];
        result += difference * difference;
    }
    result
}

/// umap's `tau_rand_int` on a vertex's three states (Tausworthe, 32-bit states held in
/// `i64` as umap holds them; the result truncated to `int32`, umap's signature).
fn tau_rand_int(state: &mut [i64; 3]) -> i64 {
    state[0] = (((state[0] & 4_294_967_294) << 12) & 0xFFFF_FFFF)
        ^ ((((state[0] << 13) & 0xFFFF_FFFF) ^ state[0]) >> 19);
    state[1] = (((state[1] & 4_294_967_288) << 4) & 0xFFFF_FFFF)
        ^ ((((state[1] << 2) & 0xFFFF_FFFF) ^ state[1]) >> 25);
    state[2] = (((state[2] & 4_294_967_280) << 17) & 0xFFFF_FFFF)
        ^ ((((state[2] << 3) & 0xFFFF_FFFF) ^ state[2]) >> 11);
    i64::from((state[0] ^ state[1] ^ state[2]) as i32)
}

/// The generator states of every vertex: 32-bit values from the seed, each above the
/// Tausworthe minimum of its component (2, 8, 16).
fn vertex_states(n: usize, seed: u64) -> Vec<[i64; 3]> {
    let mut draws = Draws::new(seed, 0x5441_5553);
    (0..n)
        .map(|_| {
            let mut state = [0_i64; 3];
            for (value, minimum) in state.iter_mut().zip([2_i64, 8, 16]) {
                *value = i64::from((draws.next_u64() >> 32) as u32) | minimum;
            }
            state
        })
        .collect()
}

/// umap's `optimize_layout_euclidean(move_other=True, parallel=False)`.
fn optimize(
    embedding: &mut [f32],
    edges: &Edges,
    sgd: &Sgd,
    control: &RunControl,
) -> Result<(), KytoError> {
    let Sgd {
        n,
        dim,
        n_epochs,
        a,
        b,
        gamma,
        initial_alpha,
        negative_sample_rate,
        seed,
    } = *sgd;
    let mut states = vertex_states(n, seed);
    let epochs_per_negative_sample: Vec<f64> = edges
        .epochs_per_sample
        .iter()
        .map(|&value| value / negative_sample_rate as f64)
        .collect();
    let mut next_negative = epochs_per_negative_sample.clone();
    let mut next_sample = edges.epochs_per_sample.clone();
    let n_vertices = n as i64;
    let mut alpha = initial_alpha;
    for epoch in 0..n_epochs {
        control.check()?;
        let current_epoch = epoch as f64;
        for edge in 0..edges.head.len() {
            if next_sample[edge] > current_epoch {
                continue;
            }
            let j = edges.head[edge];
            let k = edges.tail[edge];
            let dist_squared = f64::from(rdist(embedding, j, k, dim));
            let grad_coeff = if dist_squared > 0.0 {
                -2.0 * a * b * dist_squared.powf(b - 1.0) / (a * dist_squared.powf(b) + 1.0)
            } else {
                0.0
            };
            for axis in 0..dim {
                let difference = embedding[j * dim + axis] - embedding[k * dim + axis];
                let grad = clip(grad_coeff * f64::from(difference));
                embedding[j * dim + axis] =
                    (f64::from(embedding[j * dim + axis]) + grad * alpha) as f32;
                embedding[k * dim + axis] =
                    (f64::from(embedding[k * dim + axis]) - grad * alpha) as f32;
            }
            next_sample[edge] += edges.epochs_per_sample[edge];
            if negative_sample_rate == 0 {
                continue;
            }
            // umap truncates toward zero; a negative count samples nothing.
            let count =
                ((current_epoch - next_negative[edge]) / epochs_per_negative_sample[edge]) as i64;
            for _ in 0..count.max(0) {
                // numba's `%` on a signed int is Python's: never negative.
                let k = tau_rand_int(&mut states[j]).rem_euclid(n_vertices) as usize;
                let dist_squared = f64::from(rdist(embedding, j, k, dim));
                let grad_coeff = if dist_squared > 0.0 {
                    2.0 * gamma * b / ((0.001 + dist_squared) * (a * dist_squared.powf(b) + 1.0))
                } else if j == k {
                    continue;
                } else {
                    0.0
                };
                if grad_coeff > 0.0 {
                    for axis in 0..dim {
                        let difference = embedding[j * dim + axis] - embedding[k * dim + axis];
                        let grad = clip(grad_coeff * f64::from(difference));
                        embedding[j * dim + axis] =
                            (f64::from(embedding[j * dim + axis]) + grad * alpha) as f32;
                    }
                }
            }
            next_negative[edge] += count as f64 * epochs_per_negative_sample[edge];
        }
        alpha = initial_alpha * (1.0 - current_epoch / n_epochs as f64);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ab_params_match_umap() {
        // umap-learn 0.5.12's find_ab_params (scipy curve_fit, tolerance ~1e-8).
        for (spread, min_dist, a, b) in [
            (1.0, 0.5, 0.583_030_020_757_172_3, 1.334_166_992_130_317_2),
            (1.0, 0.1, 1.576_943_460_575_499_3, 0.895_060_878_168_034_7),
            (1.0, 0.0, 1.932_808_397_545_408, 0.790_494_973_590_513_9),
            (2.0, 1.0, 0.091_715_509_992_948_63, 1.334_166_559_632_488),
            (0.5, 0.5, 1.667_716_866_662_288_6, 1.929_240_435_347_153),
            (3.0, 0.01, 0.335_618_816_842_932_7, 0.793_864_075_593_629_1),
        ] {
            let (fit_a, fit_b) = find_ab_params(spread, min_dist);
            assert!(
                (fit_a - a).abs() <= 1e-5 * a,
                "{spread} {min_dist}: a {fit_a} vs {a}"
            );
            assert!(
                (fit_b - b).abs() <= 1e-5 * b,
                "{spread} {min_dist}: b {fit_b} vs {b}"
            );
        }
    }

    #[test]
    fn tau_rand_int_matches_umap() {
        // umap.utils.tau_rand_int on np.array([12345, 67890, 13579]), five draws.
        let mut state = [12_345_i64, 67_890, 13_579];
        let draws: Vec<i64> = (0..5).map(|_| tau_rand_int(&mut state)).collect();
        assert_eq!(draws, TAU_REFERENCE);
    }

    const TAU_REFERENCE: [i64; 5] = [
        1_762_857_971,
        962_756_195,
        1_349_868_690,
        -1_122_795_377,
        -1_413_367_045,
    ];

    #[test]
    fn kept_edges_compare_in_float32() {
        let keep = kept_edges(&[1.0, 0.002, 0.001_999, 0.5], 500, 500);
        assert_eq!(keep, [true, true, false, true]);
        // Ten epochs or fewer use the default for the threshold.
        let keep = kept_edges(&[1.0, 0.01, 0.002], 5, 500);
        assert_eq!(keep, [true, true, true]);
    }

    #[test]
    fn too_many_dimensions_are_refused_before_allocating() {
        // Bug-hunter (6.9): 2^40 dimensions aborted the process on allocation.
        let graph = crate::connectivities::connectivities(
            &crate::NeighborGraph::from_csr(2, vec![0, 1, 2], vec![1, 0], vec![1.0, 1.0]).unwrap(),
        );
        for n_components in [0, 101, 1 << 40, u64::MAX] {
            let params = Umap {
                n_components,
                ..Umap::default()
            };
            let error = umap_layout(&graph, None, &params).unwrap_err();
            assert_eq!(error.code(), "KY-E215", "{n_components}");
        }
        let params = Umap {
            n_components: 100,
            n_epochs: Some(2),
            ..Umap::default()
        };
        assert_eq!(
            umap_layout(&graph, None, &params)
                .unwrap()
                .embedding()
                .len(),
            200
        );
    }

    #[test]
    fn normalize_scales_each_dimension_to_ten() {
        let mut embedding = vec![1.0, 5.0, 3.0, 5.0, 2.0, 5.0];
        normalize(&mut embedding, 3, 2);
        assert_eq!(embedding, vec![0.0, 0.0, 10.0, 0.0, 5.0, 0.0]);
    }
}
