//! Scrublet's automatic threshold (task 6.12): scikit-image 0.26's
//! `threshold_minimum` on the simulated doublets' scores, step for step.
//!
//! Does: numpy's 256-bin histogram over the scores' range (`np.histogram`'s uniform
//! bins: `linspace` edges, the index from `(x - first) / (last - first) * 256` then
//! corrected against the edges, the last bin closed), scipy's `uniform_filter1d` of
//! width 3 (`reflect` ends, a running sum in `f64`, each pass rounded to `float32`)
//! until at most two local maxima remain (scikit-image's plateau-aware search, at most
//! 10,000 passes), and the centre of the lowest bin between the two maxima (the first
//! on ties). `None` where scikit-image raises (no scores, not exactly two maxima, or
//! the last pass reached): Scrublet then calls no doublet.
//! Does not: score anything (see `scrublet.rs`).

/// scikit-image's `nbins`.
const N_BINS: usize = 256;
/// scikit-image's `max_num_iter`.
const MAX_PASSES: usize = 10_000;

/// The threshold scikit-image's `threshold_minimum(values)` returns, or `None` where it
/// raises. `values` must be finite.
pub(crate) fn threshold_minimum(values: &[f64]) -> Option<f64> {
    let (counts, centers) = histogram(values)?;
    let mut smooth: Vec<f32> = counts.iter().map(|&count| count as f32).collect();
    let mut maxima = Vec::new();
    let mut last_pass = 0;
    for pass in 0..MAX_PASSES {
        last_pass = pass;
        smooth = uniform_filter3(&smooth);
        maxima = local_maxima(&smooth);
        if maxima.len() < 3 {
            break;
        }
    }
    if maxima.len() != 2 || last_pass == MAX_PASSES - 1 {
        return None;
    }
    let (first, second) = (maxima[0], maxima[1]);
    let mut lowest = first;
    for index in first..=second {
        if smooth[index] < smooth[lowest] {
            lowest = index;
        }
    }
    Some(centers[lowest])
}

/// `np.histogram(values, bins=256)`'s counts and the bins' centres
/// (`(edges[:-1] + edges[1:]) / 2`), or `None` for no values.
fn histogram(values: &[f64]) -> Option<(Vec<u64>, Vec<f64>)> {
    let mut first = values.iter().copied().reduce(f64::min)?;
    let mut last = values.iter().copied().reduce(f64::max)?;
    if first == last {
        first -= 0.5;
        last += 0.5;
    }
    let edges = linspace(first, last, N_BINS + 1);
    let width = last - first;
    let mut counts = vec![0_u64; N_BINS];
    for &value in values {
        // numpy: `f_indices = (x - first) / (last - first) * n` truncated, then moved by
        // one where the division disagrees with the edges.
        let mut index = ((value - first) / width * N_BINS as f64) as usize;
        if index == N_BINS {
            index -= 1;
        }
        if value < edges[index] {
            index -= 1;
        }
        if value >= edges[index + 1] && index != N_BINS - 1 {
            index += 1;
        }
        counts[index] += 1;
    }
    let centers = edges
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) / 2.0)
        .collect();
    Some((counts, centers))
}

/// `np.linspace(start, stop, num)` (`num >= 2`): `i * step + start`, the last value
/// `stop` exactly.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let divisions = (num - 1) as f64;
    let step = (stop - start) / divisions;
    let mut values: Vec<f64> = (0..num)
        .map(|index| {
            if step == 0.0 {
                index as f64 / divisions * (stop - start) + start
            } else {
                index as f64 * step + start
            }
        })
        .collect();
    values[num - 1] = stop;
    values
}

/// scipy's `ndimage.uniform_filter1d(values, 3)` on `float32` values: the ends
/// reflected (`a | a b c | c`), a running sum in `f64` (the first window summed, then
/// one value in and one out, each change divided by 3), each output rounded to
/// `float32`.
fn uniform_filter3(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    let at = |index: isize| -> f64 {
        let clamped = index.clamp(0, n as isize - 1) as usize;
        f64::from(values[clamped])
    };
    let mut out = Vec::with_capacity(n);
    let mut sum = 0.0;
    for index in -1..=1 {
        sum += at(index);
    }
    sum /= 3.0;
    out.push(sum as f32);
    for index in 1..n as isize {
        sum += (at(index + 1) - at(index - 2)) / 3.0;
        out.push(sum as f32);
    }
    out
}

/// scikit-image's `find_local_maxima_idx`: where a rise turns into a fall, a plateau
/// counted once at its last index.
fn local_maxima(values: &[f32]) -> Vec<usize> {
    let mut maxima = Vec::new();
    let mut rising = true;
    for index in 0..values.len().saturating_sub(1) {
        if rising {
            if values[index + 1] < values[index] {
                rising = false;
                maxima.push(index);
            }
        } else if values[index + 1] > values[index] {
            rising = true;
        }
    }
    maxima
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_filter_follows_scipy() {
        // scipy.ndimage.uniform_filter1d(np.float32([0, 3, 1, 7, 2]), 3)
        let out = uniform_filter3(&[0.0, 3.0, 1.0, 7.0, 2.0]);
        assert_eq!(
            out,
            [1.0, 1.333_333_4, 3.666_666_7, 3.333_333_3, 3.666_666_7]
        );
        assert_eq!(uniform_filter3(&[5.0]), [5.0]);
    }

    #[test]
    fn plateaus_are_one_maximum() {
        assert_eq!(local_maxima(&[0.0, 2.0, 2.0, 1.0, 3.0, 3.0]), [2]);
        assert_eq!(local_maxima(&[3.0, 1.0, 2.0, 0.0]), [0, 2]);
    }

    #[test]
    fn the_histogram_closes_its_last_bin() {
        let values: Vec<f64> = (0..=256).map(|value| value as f64).collect();
        let (counts, centers) = histogram(&values).unwrap();
        assert_eq!(counts.iter().sum::<u64>(), 257);
        assert_eq!(counts[255], 2);
        assert_eq!(centers[0], 0.5);
        // One value: numpy widens the range by 0.5 on each side.
        let (counts, centers) = histogram(&[2.0, 2.0]).unwrap();
        assert_eq!(counts[128], 2);
        assert!((centers[128] - 2.001_953_125).abs() < 1e-12);
    }

    #[test]
    fn two_modes_give_the_valley_between() {
        let mut values = Vec::new();
        for (center, count) in [(0.1, 400), (0.7, 120)] {
            for index in 0..count {
                let spread = (index as f64 / count as f64 - 0.5) * 0.1;
                values.push(center + spread);
            }
        }
        // scikit-image gives 0.09233235677083333: the flat modes are plateaus, and the
        // lowest bin between the two maxima is the first mode's edge.
        assert_eq!(threshold_minimum(&values), Some(0.092_332_356_770_833_33));
        assert_eq!(threshold_minimum(&[]), None);
        // scikit-image's `threshold_minimum(np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.3]))`.
        assert_eq!(
            threshold_minimum(&[0.1, 0.1, 0.2, 0.2, 0.2, 0.3]),
            Some(0.101_953_125)
        );
    }
}
