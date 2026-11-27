//! e-means metacells (task 6.16, opt-in): many small groups of cells in an embedding.
//!
//! Does: run the mini-batch k-means of Bessa et al., "e-means: outperforming k-means
//! (in one-tenth the epochs)" (arXiv 2611.09501), Algorithm 1, on the leading
//! components of a cell embedding. `k` random cells start the centroids; each epoch
//! visits the cells in a fresh shuffled order in batches; each cell joins its nearest
//! centroid, and every centroid is a running sum over a running count, both updated by
//! an EMA whose step follows a cosine schedule from the first step down to a tenth of
//! it; every five batches a nearly empty cluster takes half of a populous one (Faiss's
//! split). The batch size and first step are the paper's Eq. 7 and 6, unchanged. After
//! the last epoch the centroids are the means of the cells each took in it; then every
//! cell joins its nearest such centroid once more and the centroids are the means again.
//! Departures from the paper (measured on pbmc3k and a 50k data set, deviations.md
//! "metacells"): revival below the lower of 10% of the mean count and two cells' worth
//! (the paper's 10% alone revives real small populations away at k = n/200), each
//! epoch's last batch takes the remainder, and the final assignment pass.
//! Deterministic (R7, R11): the initial cells, the shuffles and the revival donors come
//! from SplitMix64 streams of the seed (integer arithmetic only); distances add the
//! components in order in `f64`, one cell at a time (cells run in parallel, each on its
//! own), and sums add the cells in batch order (the final means in cell order), so
//! the bits are the same on any thread count. `tests/parity/emeans.py` replays it.
//! Does not: choose `k` (the caller does), or read the embedding (the pipeline hands
//! the rows over).

use std::f64::consts::PI;
use std::sync::Arc;

use kyto_error::KytoError;
use kyto_plan::{METACELLS_MAX_EPOCHS, Metacells};
use rayon::prelude::*;

use crate::run_control::RunControl;
use crate::umap_spectral::Draws;

/// The cosine schedule's floor, a fraction of the first step (the paper's epsilon).
const SCHEDULE_FLOOR: f64 = 0.1;
/// Clusters below this fraction of the mean count are revived (the paper's tau)...
const REVIVAL_FRACTION: f64 = 0.1;
/// ...or below this many cells' worth of count, if that is lower (task 6.16).
const REVIVAL_CELLS: f64 = 2.0;
/// Revival is checked after every this many batches.
const REVIVAL_EVERY: u64 = 5;
/// The starting count of every cluster.
const INITIAL_COUNT: f64 = 1e-10;
/// The relative perturbation of a split centroid (Faiss's `EPS`).
const SPLIT_PERTURBATION: f64 = 1.0 / 1024.0;
/// The draw streams of the seed: initial cells, shuffles, revival donors.
const PURPOSE_INIT: u64 = 1;
const PURPOSE_SHUFFLE: u64 = 2;
const PURPOSE_REVIVAL: u64 = 3;
/// Cells a parallel assignment task takes.
const ASSIGN_CHUNK: usize = 256;

/// What e-means found: each cell's metacell and the metacells' centroids.
#[derive(Debug, Clone, PartialEq)]
pub struct MetacellGrouping {
    labels: Vec<u32>,
    centroids: Vec<f64>,
    n_metacells: usize,
    width: usize,
    batch_size: usize,
    first_step: f64,
}

impl MetacellGrouping {
    /// Each cell's metacell, `0..n_metacells`, in row order.
    pub fn labels(&self) -> &[u32] {
        &self.labels
    }

    /// The centroids, row-major, `width` values a metacell; a metacell no cell joined
    /// keeps its running centroid.
    pub fn centroids(&self) -> &[f64] {
        &self.centroids
    }

    /// The number of metacells (`k`).
    pub fn n_metacells(&self) -> usize {
        self.n_metacells
    }

    /// The components used.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The batch size the paper's Eq. 7 gave.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// The first step the paper's Eq. 6 gave.
    pub fn first_step(&self) -> f64 {
        self.first_step
    }

    /// The mean squared distance of a cell to its metacell's centroid (training MSE).
    pub fn mse(&self, values: &[f64], row_width: usize) -> f64 {
        let n = self.labels.len();
        let mut total = 0.0;
        for (row, &label) in self.labels.iter().enumerate() {
            let centroid = &self.centroids[label as usize * self.width..][..self.width];
            total += squared_distance(&values[row * row_width..][..self.width], centroid);
        }
        total / n as f64
    }
}

/// What a `metacells` node computed: the grouping of the cells reaching it.
#[derive(Debug, Clone, PartialEq)]
pub struct MetacellStatistics {
    pub(crate) cells: Arc<Vec<u64>>,
    pub(crate) grouping: MetacellGrouping,
}

impl MetacellStatistics {
    /// The store table position of each row of the grouping.
    pub fn cells(&self) -> &[u64] {
        &self.cells
    }

    /// The grouping.
    pub fn grouping(&self) -> &MetacellGrouping {
        &self.grouping
    }

    /// The centroid embedding: each cell's row is its metacell's centroid, `float32`.
    pub(crate) fn centroid_rows(&self) -> Vec<f32> {
        let width = self.grouping.width;
        let mut rows = Vec::with_capacity(self.grouping.labels.len() * width);
        for &label in &self.grouping.labels {
            // Rounded to the embedding's float32, as `pca` stores its rows.
            #[allow(clippy::cast_possible_truncation)]
            rows.extend(
                self.grouping.centroids[label as usize * width..][..width]
                    .iter()
                    .map(|&value| value as f32),
            );
        }
        rows
    }
}

/// The paper's Eq. 7 (knee at 0.2%, divided by 8 past two million cells), rounded to a
/// power of two and capped at `n`.
pub(crate) fn batch_size(n: usize, k: usize, epochs: u64) -> usize {
    let mut raw =
        0.035 * (k as f64).powf(0.13) * (n as f64).powf(0.88) * (epochs as f64).powf(0.81);
    if n > 2_000_000 {
        raw /= 8.0;
    }
    let exponent = (raw.log2() + 0.5).floor().clamp(0.0, 62.0);
    // `exponent` is a whole number from 0 to 62.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let power = 1_usize << (exponent as u32);
    power.min(n)
}

/// The paper's Eq. 6, at most 1.
pub(crate) fn first_step(n: usize, k: usize, epochs: u64, batch_size: usize) -> f64 {
    (0.824
        * (k as f64).powf(-0.46)
        * (batch_size as f64).powf(0.90)
        * (n as f64).powf(-0.45)
        * (epochs as f64).powf(-0.11))
    .min(1.0)
}

/// The step (`1 - gamma`) of update `t` of `total`: the cosine schedule.
fn step(t: u64, total: u64, first_step: f64) -> f64 {
    first_step
        * (SCHEDULE_FLOOR
            + (1.0 - SCHEDULE_FLOOR) * 0.5 * (1.0 + (PI * t as f64 / total as f64).cos()))
}

/// Squared Euclidean distance, components added in order.
fn squared_distance(left: &[f64], right: &[f64]) -> f64 {
    let mut total = 0.0;
    for (a, b) in left.iter().zip(right) {
        let difference = a - b;
        total += difference * difference;
    }
    total
}

/// The nearest centroid (the first of equals).
fn nearest(point: &[f64], centroids: &[f64], width: usize) -> u32 {
    let mut best = 0;
    let mut best_distance = f64::INFINITY;
    for (cluster, centroid) in centroids.chunks_exact(width).enumerate() {
        let distance = squared_distance(point, centroid);
        if distance < best_distance {
            best_distance = distance;
            best = cluster;
        }
    }
    // `cluster` < k, and k fits in u32 (checked by `group`).
    #[allow(clippy::cast_possible_truncation)]
    let best = best as u32;
    best
}

impl Draws {
    /// Uniform in `0..n` (multiply-shift).
    fn below(&mut self, n: usize) -> usize {
        // The product's high half is below `n`.
        #[allow(clippy::cast_possible_truncation)]
        let pick = ((u128::from(self.next_u64()) * n as u128) >> 64) as usize;
        pick
    }
}

/// `k` distinct rows of `0..n`: the first `k` slots of a partial Fisher-Yates shuffle.
fn sample(draws: &mut Draws, n: usize, k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    for slot in 0..k {
        let pick = slot + draws.below(n - slot);
        order.swap(slot, pick);
    }
    order.truncate(k);
    order
}

/// `0..n` shuffled (Fisher-Yates from the last slot).
fn shuffled(draws: &mut Draws, n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    for last in (1..n).rev() {
        let pick = draws.below(last + 1);
        order.swap(last, pick);
    }
    order
}

/// The running statistics of the clusters.
struct Clusters {
    sums: Vec<f64>,
    counts: Vec<f64>,
    /// The last centroids read: a cluster whose count is 0 keeps its centroid (a
    /// first step clamped to 1 gives `gamma = 0`, so a cluster no cell of the first
    /// batch joined would otherwise be `0 / 0`).
    previous: Vec<f64>,
    width: usize,
}

impl Clusters {
    /// The current centroids, `sum / count`, or the previous one where the count is 0.
    fn centroids(&mut self) -> Vec<f64> {
        let width = self.width;
        for (cluster, &count) in self.counts.iter().enumerate() {
            if count > 0.0 {
                for component in 0..width {
                    self.previous[cluster * width + component] =
                        self.sums[cluster * width + component] / count;
                }
            }
        }
        self.previous.clone()
    }

    /// Split a populous cluster into each one whose count is below `fraction` of the
    /// mean, the donor drawn with probability proportional to its count.
    fn revive(&mut self, draws: &mut Draws, fraction: f64) {
        let k = self.counts.len();
        let width = self.width;
        let mean = self.counts.iter().sum::<f64>() / k as f64;
        for dead in 0..k {
            if self.counts[dead] >= fraction * mean {
                continue;
            }
            let total: f64 = self.counts.iter().sum();
            let target = draws.uniform() * total;
            let mut donor = k - 1;
            let mut running = 0.0;
            for (cluster, &count) in self.counts.iter().enumerate() {
                running += count;
                if target < running {
                    donor = cluster;
                    break;
                }
            }
            if donor == dead || self.counts[donor] <= 0.0 {
                continue;
            }
            let count = self.counts[donor];
            let half = count * 0.5;
            self.counts[dead] = half;
            self.counts[donor] = half;
            for component in 0..width {
                let centroid = self.sums[donor * width + component] / count;
                let sign = if component % 2 == 0 { 1.0 } else { -1.0 };
                self.sums[dead * width + component] =
                    centroid * (1.0 + sign * SPLIT_PERTURBATION) * half;
                self.sums[donor * width + component] =
                    centroid * (1.0 - sign * SPLIT_PERTURBATION) * half;
            }
        }
    }
}

/// Each cluster's mean of the rows labelled with it (rows added in order), `fallback`
/// where no row is.
fn means(points: &[f64], labels: &[u32], fallback: &[f64], width: usize) -> Vec<f64> {
    let k = fallback.len() / width;
    let mut sums = vec![0.0; k * width];
    let mut counts = vec![0_u64; k];
    for (point, &label) in points.chunks_exact(width).zip(labels) {
        let label = label as usize;
        counts[label] += 1;
        for (sum, value) in sums[label * width..][..width].iter_mut().zip(point) {
            *sum += value;
        }
    }
    let mut centroids = fallback.to_vec();
    for (cluster, &count) in counts.iter().enumerate() {
        if count > 0 {
            for component in 0..width {
                centroids[cluster * width + component] =
                    sums[cluster * width + component] / count as f64;
            }
        }
    }
    centroids
}

/// Each of `rows` (indices into `points`) assigned to its nearest centroid, in order,
/// on the current rayon pool.
fn assign(points: &[f64], rows: &[usize], centroids: &[f64], width: usize) -> Vec<u32> {
    let mut labels = vec![0_u32; rows.len()];
    labels
        .par_chunks_mut(ASSIGN_CHUNK)
        .zip(rows.par_chunks(ASSIGN_CHUNK))
        .for_each(|(labels, rows)| {
            for (label, &row) in labels.iter_mut().zip(rows) {
                *label = nearest(&points[row * width..][..width], centroids, width);
            }
        });
    labels
}

fn invalid(parameter: &str, value: &str, problem: &str, accepted: &str) -> KytoError {
    KytoError::InvalidArgument {
        function: "metacells".to_owned(),
        parameter: parameter.to_owned(),
        value: value.to_owned(),
        problem: problem.to_owned(),
        accepted: accepted.to_owned(),
    }
}

/// e-means on the rows of `values` (row-major, `width` values a row), using the first
/// `params.n_components` components (all for `None`); `params.embedding` and
/// `params.into` are not read. Runs on the current rayon pool.
///
/// # Errors
/// `KY-E215` for `n_metacells` of 0 or above the rows, `n_components` of 0 or above
/// `width`, `epochs` of 0 or above [`METACELLS_MAX_EPOCHS`], `values` whose length is
/// not a multiple of `width`, or a value that is not finite.
pub fn metacells(
    values: &[f64],
    width: usize,
    params: &Metacells,
) -> Result<MetacellGrouping, KytoError> {
    group(values, width, params, &RunControl::new())
}

/// [`metacells`], cancelled through `control` (`KY-E503`, checked every batch).
#[allow(clippy::too_many_lines)]
pub(crate) fn group(
    values: &[f64],
    width: usize,
    params: &Metacells,
    control: &RunControl,
) -> Result<MetacellGrouping, KytoError> {
    if width == 0 || !values.len().is_multiple_of(width) {
        return Err(invalid(
            "values",
            &format!("{} values", values.len()),
            &format!("it is not a whole number of rows of {width} components"),
            "one row of the embedding per cell",
        ));
    }
    let n = values.len() / width;
    let used = match params.n_components {
        Some(used) => usize::try_from(used).unwrap_or(usize::MAX),
        None => width,
    };
    if used == 0 || used > width {
        return Err(invalid(
            "n_components",
            &used.to_string(),
            &format!("the embedding has {width} components"),
            &format!("an int from 1 to {width}, or None for all"),
        ));
    }
    let k = usize::try_from(params.n_metacells).unwrap_or(usize::MAX);
    if n == 0 {
        return Err(invalid(
            "n_metacells",
            &params.n_metacells.to_string(),
            "no cell reaches the step (an earlier filter removed them all)",
            "1 or more, after filters that keep at least one cell,",
        ));
    }
    if k == 0 || k > n || u32::try_from(k).is_err() {
        return Err(invalid(
            "n_metacells",
            &params.n_metacells.to_string(),
            &format!("{n} cells reach the step"),
            &format!("an int from 1 to {n} (n/50 to n/500 is typical)"),
        ));
    }
    if params.epochs == 0 || params.epochs > METACELLS_MAX_EPOCHS {
        return Err(invalid(
            "epochs",
            &params.epochs.to_string(),
            &format!("e-means runs from 1 to {METACELLS_MAX_EPOCHS} passes"),
            &format!("an int from 1 to {METACELLS_MAX_EPOCHS}"),
        ));
    }
    // The leading components, in f64.
    let points: Vec<f64> = if used == width {
        values.to_vec()
    } else {
        values
            .chunks_exact(width)
            .flat_map(|row| row[..used].iter().copied())
            .collect()
    };
    // Within float32's range: the centroids are stored as float32, and squared
    // distances of such values cannot overflow f64.
    let limit = f64::from(f32::MAX);
    if let Some(position) = points
        .iter()
        .position(|value| !(value.is_finite() && value.abs() <= limit))
    {
        return Err(invalid(
            "embedding",
            &format!("{} at cell {}", points[position], position / used),
            "e-means needs finite coordinates within float32's range (the centroids \
             are stored as float32)",
            "an embedding of finite values of magnitude at most 3.4e38",
        ));
    }
    let width = used;
    let batch = batch_size(n, k, params.epochs);
    let first = first_step(n, k, params.epochs, batch);
    let fraction = REVIVAL_FRACTION.min(REVIVAL_CELLS * k as f64 / n as f64);

    let mut clusters = Clusters {
        sums: Vec::with_capacity(k * width),
        counts: vec![INITIAL_COUNT; k],
        previous: Vec::with_capacity(k * width),
        width,
    };
    for row in sample(&mut Draws::new(params.seed, PURPOSE_INIT), n, k) {
        let point = &points[row * width..][..width];
        clusters
            .sums
            .extend(point.iter().map(|value| value * INITIAL_COUNT));
        clusters.previous.extend_from_slice(point);
    }
    let mut shuffles = Draws::new(params.seed, PURPOSE_SHUFFLE);
    let mut revivals = Draws::new(params.seed, PURPOSE_REVIVAL);
    let per_epoch = n.div_ceil(batch) as u64;
    let total = params.epochs * per_epoch;
    let mut labels = vec![0_u32; n];
    let mut batch_sums = vec![0.0; k * width];
    let mut batch_counts = vec![0.0; k];
    let mut t = 0_u64;
    for _ in 0..params.epochs {
        let order = shuffled(&mut shuffles, n);
        for rows in order.chunks(batch) {
            control.check()?;
            let centroids = clusters.centroids();
            let assigned = assign(&points, rows, &centroids, width);
            batch_sums.fill(0.0);
            batch_counts.fill(0.0);
            for (&row, &label) in rows.iter().zip(&assigned) {
                labels[row] = label;
                let label = label as usize;
                batch_counts[label] += 1.0;
                for (sum, value) in batch_sums[label * width..][..width]
                    .iter_mut()
                    .zip(&points[row * width..][..width])
                {
                    *sum += value;
                }
            }
            let alpha = step(t, total, first);
            let gamma = 1.0 - alpha;
            for (count, &found) in clusters.counts.iter_mut().zip(&batch_counts) {
                *count = gamma * *count + alpha * found;
            }
            for (sum, &found) in clusters.sums.iter_mut().zip(&batch_sums) {
                *sum = gamma * *sum + alpha * found;
            }
            t += 1;
            if t.is_multiple_of(REVIVAL_EVERY) {
                clusters.revive(&mut revivals, fraction);
            }
        }
    }
    control.check()?;
    let centroids = means(&points, &labels, &clusters.centroids(), width);
    // One more assignment, to the nearest final centroid, and the means again.
    let all: Vec<usize> = (0..n).collect();
    let labels = assign(&points, &all, &centroids, width);
    let centroids = means(&points, &labels, &centroids, width);
    Ok(MetacellGrouping {
        labels,
        centroids,
        n_metacells: k,
        width,
        batch_size: batch,
        first_step: first,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(k: u64) -> Metacells {
        Metacells {
            embedding: "pca".to_owned(),
            n_components: None,
            n_metacells: k,
            epochs: 3,
            seed: 0,
            into: "metacells".to_owned(),
        }
    }

    #[test]
    fn the_paper_laws_round_to_powers_of_two_and_cap() {
        assert_eq!(batch_size(2638, 52, 3), 128);
        assert_eq!(batch_size(49_888, 997, 3), 2048);
        assert_eq!(batch_size(10, 2, 25), 4);
        assert_eq!(batch_size(3, 1, 25), 1);
        assert_eq!(batch_size(50_000_000, 100_000, 25), 1 << 21);
        assert!(first_step(10, 10, 1, 10) <= 1.0);
        let step_size = first_step(49_888, 997, 3, 2048);
        assert!((step_size - 0.224).abs() < 1e-3, "{step_size}");
    }

    #[test]
    fn the_schedule_falls_from_the_first_step_to_a_tenth() {
        assert_eq!(step(0, 100, 0.5), 0.5);
        assert!((step(100, 100, 0.5) - 0.05).abs() < 1e-15);
        assert!(step(50, 100, 0.5) < 0.5 && step(50, 100, 0.5) > 0.05);
    }

    #[test]
    fn separated_groups_become_one_metacell_each() {
        // Three tight groups of 40 points, far apart.
        let mut values = Vec::new();
        for group in 0..3 {
            for point in 0..40 {
                values.push(f64::from(group) * 100.0 + f64::from(point % 5) * 0.01);
                values.push(f64::from(point % 3) * 0.01);
            }
        }
        let found = metacells(&values, 2, &params(3)).unwrap();
        for group in 0..3 {
            let labels = &found.labels()[group * 40..][..40];
            assert!(labels.iter().all(|&label| label == labels[0]));
        }
        let mut distinct = found.labels().to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 3);
        assert!(found.mse(&values, 2) < 1e-3);
    }

    #[test]
    fn centroids_are_the_means_of_their_cells() {
        let values: Vec<f64> = (0..200).map(|value| f64::from(value * 37 % 101)).collect();
        let found = metacells(&values, 2, &params(7)).unwrap();
        assert!(found.labels().iter().all(|&label| label < 7));
        for cluster in 0..7_u32 {
            let members: Vec<&[f64; 2]> = values
                .as_chunks::<2>()
                .0
                .iter()
                .zip(found.labels())
                .filter(|&(_, &label)| label == cluster)
                .map(|(row, _)| row)
                .collect();
            if members.is_empty() {
                continue;
            }
            for component in 0..2 {
                let mean =
                    members.iter().map(|row| row[component]).sum::<f64>() / members.len() as f64;
                let centroid = found.centroids()[cluster as usize * 2 + component];
                assert!((mean - centroid).abs() <= 1e-12 * mean.abs().max(1.0));
            }
        }
    }

    #[test]
    fn arguments_out_of_range_are_coded() {
        let values = vec![0.0; 20];
        for (bad, parameter) in [
            (
                Metacells {
                    n_metacells: 0,
                    ..params(1)
                },
                "n_metacells",
            ),
            (
                Metacells {
                    n_metacells: 11,
                    ..params(1)
                },
                "n_metacells",
            ),
            (
                Metacells {
                    epochs: 0,
                    ..params(1)
                },
                "epochs",
            ),
            (
                Metacells {
                    epochs: 26,
                    ..params(1)
                },
                "epochs",
            ),
            (
                Metacells {
                    n_components: Some(3),
                    ..params(1)
                },
                "n_components",
            ),
        ] {
            let error = metacells(&values, 2, &bad).unwrap_err();
            assert_eq!(error.code(), "KY-E215", "{error}");
            assert!(error.to_string().contains(parameter), "{error}");
        }
        let error = metacells(&[0.0, f64::NAN], 2, &params(1)).unwrap_err();
        assert_eq!(error.code(), "KY-E215");
        let error = metacells(&[0.0; 3], 2, &params(1)).unwrap_err();
        assert_eq!(error.code(), "KY-E215");
        let error = metacells(&[0.0, 1e200], 2, &params(1)).unwrap_err();
        assert!(error.to_string().contains("float32"), "{error}");
        let error = metacells(&[], 2, &params(1)).unwrap_err();
        assert!(error.to_string().contains("no cell reaches"), "{error}");
    }

    #[test]
    fn a_first_step_of_one_leaves_no_centroid_undefined() {
        // n = 2700, k = 100, 25 epochs: Eq. 6 gives a first step above 1, clamped.
        let values: Vec<f64> = (0..2700_u32)
            .map(|cell| f64::from(cell.wrapping_mul(2_654_435_761) % 1000) / 7.0)
            .collect();
        let found = metacells(
            &values,
            1,
            &Metacells {
                epochs: 25,
                ..params(100)
            },
        )
        .unwrap();
        assert!(first_step(2700, 100, 25, batch_size(2700, 100, 25)) >= 1.0);
        assert!(found.centroids().iter().all(|value| value.is_finite()));
    }
}
