//! Per-group gene means and variances for differential expression, and Welch's
//! t-test on them (task 6.10), as scanpy 1.12.5's `rank_genes_groups` computes them.
//!
//! Does: fold each cell's stored non-zero values into its group's moments, in cell
//! order (one sequential fold, so the bits do not depend on blocks or threads, R11):
//! for the t-test Welford's running mean and squared deviations (scanpy's
//! `aggregate(func=["mean", "var"])`, `mean_var_csr`), for Wilcoxon plain `f64` sums
//! (`aggregate(func="mean")`, `agg_sum_csr`); merge the group's zeros in; derive each
//! group's "rest" (every other cell): means as total minus group, variances by Chan's
//! leave-one-out combine (`_vars_rest`); and Welch's t statistic and p-value as
//! `scipy.stats.ttest_ind_from_stats(equal_var=False)`.
//! Does not: read blocks or labels (`rank_genes.rs`), rank (`rank_genes_wilcoxon.rs`).

use crate::distributions::student_two_sided;

/// What a group's moments hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MomentKind {
    /// Welford's mean and squared deviations of the stored values (t-test).
    Welford,
    /// Sums of the stored values (Wilcoxon's means).
    Sums,
}

/// The moments of one group's cells, per gene, over the stored non-zero values.
#[derive(Debug, Clone)]
pub(crate) struct GroupMoments {
    n_cells: u64,
    nonzero: Vec<u64>,
    /// Welford's running mean, or the sum.
    first: Vec<f64>,
    /// Welford's squared deviations (empty for sums).
    second: Vec<f64>,
}

impl GroupMoments {
    pub(crate) fn new(kind: MomentKind, n_genes: usize) -> Self {
        Self {
            n_cells: 0,
            nonzero: vec![0; n_genes],
            first: vec![0.0; n_genes],
            second: match kind {
                MomentKind::Welford => vec![0.0; n_genes],
                MomentKind::Sums => Vec::new(),
            },
        }
    }

    /// The number of cells folded.
    pub(crate) fn n_cells(&self) -> u64 {
        self.n_cells
    }

    /// The number of non-zero values of each gene.
    pub(crate) fn nonzero(&self) -> &[u64] {
        &self.nonzero
    }

    /// Fold one cell: its genes and values (zeros skipped, as scanpy's
    /// `eliminate_zeros`).
    pub(crate) fn add_cell(&mut self, entries: impl Iterator<Item = (usize, f64)>) {
        self.n_cells += 1;
        let welford = !self.second.is_empty();
        for (gene, value) in entries {
            if value == 0.0 {
                continue;
            }
            let n = self.nonzero[gene] + 1;
            self.nonzero[gene] = n;
            if welford {
                let mean = self.first[gene];
                let delta = value - mean;
                let mean = mean + delta / n as f64;
                self.first[gene] = mean;
                self.second[gene] += delta * (value - mean);
            } else {
                self.first[gene] += value;
            }
        }
    }

    /// Each gene's mean over the group's cells, zeros included, and (Welford) its
    /// variance with `ddof=1` (`NaN` for one cell); a group without cells has mean and
    /// variance 0 (scanpy fills a missing group with zeros).
    pub(crate) fn mean_var(&self) -> (Vec<f64>, Option<Vec<f64>>) {
        let n_genes = self.nonzero.len();
        if self.n_cells == 0 {
            let variances = (!self.second.is_empty()).then(|| vec![0.0; n_genes]);
            return (vec![0.0; n_genes], variances);
        }
        let n = self.n_cells as f64;
        if self.second.is_empty() {
            return (self.first.iter().map(|sum| sum / n).collect(), None);
        }
        let mut means = vec![0.0; n_genes];
        let mut variances = vec![0.0; n_genes];
        let denominator = if self.n_cells > 1 { n - 1.0 } else { f64::NAN };
        for gene in 0..n_genes {
            let stored = self.nonzero[gene];
            let zeros = self.n_cells - stored;
            let mut mean = self.first[gene];
            let mut squared = self.second[gene];
            if zeros > 0 && stored > 0 {
                let mean_stored = mean;
                mean = mean_stored * stored as f64 / n;
                squared += mean_stored * mean_stored * stored as f64 * zeros as f64 / n;
            }
            means[gene] = mean;
            variances[gene] = squared / n * (n / denominator);
        }
        (means, Some(variances))
    }
}

/// The groups compared, in scanpy's order, and one more group for every other cell
/// (scanpy's "remainder": cells of groups not selected, and cells without a group).
#[derive(Debug, Clone)]
pub(crate) struct GroupStatistics {
    /// Cells per group, the remainder last.
    pub(crate) counts: Vec<u64>,
    /// Gene means per group, the remainder last.
    pub(crate) means: Vec<Vec<f64>>,
    /// Gene variances (`ddof=1`) per group, the remainder last (t-test only).
    pub(crate) variances: Option<Vec<Vec<f64>>>,
}

impl GroupStatistics {
    /// The statistics of `groups` (selected, in order, then the remainder).
    pub(crate) fn new(groups: &[&GroupMoments]) -> Self {
        let mut counts = Vec::with_capacity(groups.len());
        let mut means = Vec::with_capacity(groups.len());
        let mut variances = Vec::with_capacity(groups.len());
        for group in groups {
            counts.push(group.n_cells());
            let (mean, variance) = group.mean_var();
            means.push(mean);
            if let Some(variance) = variance {
                variances.push(variance);
            }
        }
        let variances = (variances.len() == groups.len()).then_some(variances);
        Self {
            counts,
            means,
            variances,
        }
    }

    /// The number of selected groups (without the remainder).
    fn selected(&self) -> usize {
        self.counts.len() - 1
    }

    /// The mean of every cell outside selected group `group`, per gene
    /// (`_derive_rest_stats`): the total of `count * mean` over every group (the
    /// remainder included), minus the group's, over the other cells.
    pub(crate) fn rest_means(&self, group: usize, n_cells: u64) -> Vec<f64> {
        let n_genes = self.means[group].len();
        let n_rest = (n_cells - self.counts[group]) as f64;
        (0..n_genes)
            .map(|gene| {
                let total = self
                    .counts
                    .iter()
                    .zip(&self.means)
                    .fold(0.0, |total, (count, means)| {
                        total + *count as f64 * means[gene]
                    });
                (total - self.counts[group] as f64 * self.means[group][gene]) / n_rest
            })
            .collect()
    }

    /// The variance (`ddof=1`) of every cell outside each selected group, per gene, by
    /// Chan's leave-one-out combine of the other groups (`_vars_rest`): `[group][gene]`.
    pub(crate) fn rest_variances(&self) -> Vec<Vec<f64>> {
        let Some(variances) = &self.variances else {
            unreachable!("rest variances are the t-test's, which has variances")
        };
        let n_groups = self.counts.len();
        let selected = self.selected();
        let n_genes = self.means[0].len();
        let counts: Vec<f64> = self.counts.iter().map(|count| *count as f64).collect();
        // M2 = var * (count - 1), 0 for groups of at most one cell.
        let squared: Vec<Vec<f64>> = (0..n_groups)
            .map(|group| {
                variances[group]
                    .iter()
                    .map(|variance| {
                        if self.counts[group] <= 1 {
                            0.0
                        } else {
                            variance * (counts[group] - 1.0)
                        }
                    })
                    .collect()
            })
            .collect();
        let mut rest = vec![vec![0.0; n_genes]; selected];
        let mut forward = vec![(0.0, 0.0, 0.0); n_groups];
        let mut backward = vec![(0.0, 0.0, 0.0); n_groups];
        for gene in 0..n_genes {
            let stats =
                |group: usize| (counts[group], self.means[group][gene], squared[group][gene]);
            let mut running = (0.0, 0.0, 0.0);
            for (group, combined) in forward.iter_mut().enumerate() {
                running = chan_combine(running, stats(group));
                *combined = running;
            }
            let mut running = (0.0, 0.0, 0.0);
            for (group, combined) in backward.iter_mut().enumerate().rev() {
                running = chan_combine(running, stats(group));
                *combined = running;
            }
            for (group, rest) in rest.iter_mut().enumerate() {
                let after = backward[group + 1];
                let (n, _, m2) = if group >= 1 {
                    chan_combine(forward[group - 1], after)
                } else {
                    after
                };
                rest[gene] = m2 / (n - 1.0);
            }
        }
        rest
    }
}

/// Combine two `(count, mean, M2)` groups (scanpy's `_chan_combine`).
fn chan_combine(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    let (n_a, mean_a, m2_a) = a;
    let (n_b, mean_b, m2_b) = b;
    if n_a == 0.0 {
        return b;
    }
    if n_b == 0.0 {
        return a;
    }
    let n = n_a + n_b;
    let delta = mean_b - mean_a;
    (
        n,
        (n_a * mean_a + n_b * mean_b) / n,
        m2_a + m2_b + delta * delta * n_a * n_b / n,
    )
}

/// Welch's t statistic and two-sided p-value of one gene, as
/// `ttest_ind_from_stats(mean1, sqrt(var1), n1, mean2, sqrt(var2), n2,
/// equal_var=False)`, with scanpy's `NaN` statistic 0 and `NaN` p-value 1.
pub(crate) fn welch(
    (mean_1, variance_1, n_1): (f64, f64, f64),
    (mean_2, variance_2, n_2): (f64, f64, f64),
) -> (f64, f64) {
    let squared = |variance: f64| {
        let std = variance.sqrt();
        std * std
    };
    let vn_1 = squared(variance_1) / n_1;
    let vn_2 = squared(variance_2) / n_2;
    let sum = vn_1 + vn_2;
    let mut df = sum * sum / (vn_1 * vn_1 / (n_1 - 1.0) + vn_2 * vn_2 / (n_2 - 1.0));
    if df.is_nan() {
        df = 1.0;
    }
    let t = (mean_1 - mean_2) / sum.sqrt();
    let p = student_two_sided(t, df);
    (
        if t.is_nan() { 0.0 } else { t },
        if p.is_nan() { 1.0 } else { p },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_with_zeros_matches_two_pass() {
        let mut moments = GroupMoments::new(MomentKind::Welford, 2);
        for row in [[1.0, 0.0], [2.0, 3.0], [0.0, 1.0], [4.0, 0.0]] {
            moments.add_cell(row.into_iter().enumerate());
        }
        let (means, variances) = moments.mean_var();
        let variances = variances.unwrap();
        assert!((means[0] - 1.75).abs() < 1e-15 && (means[1] - 1.0).abs() < 1e-15);
        // numpy var(ddof=1): [2.916666..., 2.0]
        assert!((variances[0] - 2.916_666_666_666_667).abs() < 1e-14);
        assert!((variances[1] - 2.0).abs() < 1e-14);
        let mut sums = GroupMoments::new(MomentKind::Sums, 2);
        sums.add_cell([(0, 1.0), (1, 0.0)].into_iter());
        sums.add_cell([(0, 2.0)].into_iter());
        assert_eq!(sums.mean_var(), (vec![1.5, 0.0], None));
        assert_eq!(sums.nonzero(), &[2, 0]);
    }

    #[test]
    fn rest_variance_is_the_variance_of_the_other_cells() {
        // Groups: a = [1, 2], b = [4, 6, 8], remainder = [10]; rest of a = [4, 6, 8, 10]
        // (variance 6.666...), rest of b = [1, 2, 10] (variance 24.333...).
        let mut groups = Vec::new();
        for values in [&[1.0, 2.0][..], &[4.0, 6.0, 8.0], &[10.0]] {
            let mut moments = GroupMoments::new(MomentKind::Welford, 1);
            for value in values {
                moments.add_cell([(0, *value)].into_iter());
            }
            groups.push(moments);
        }
        let statistics = GroupStatistics::new(&groups.iter().collect::<Vec<_>>());
        let rest = statistics.rest_variances();
        assert!((rest[0][0] - 20.0 / 3.0).abs() < 1e-12, "{rest:?}");
        assert!((rest[1][0] - 73.0 / 3.0).abs() < 1e-12, "{rest:?}");
        let means = statistics.rest_means(0, 6);
        assert!((means[0] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn welch_matches_scipy() {
        // scipy.stats.ttest_ind_from_stats(1.0, 1.0, 10, 0.5, 2.0, 20, equal_var=False)
        let (t, p) = welch((1.0, 1.0, 10.0), (0.5, 4.0, 20.0));
        assert!((t - 0.912_870_929_175_276_8).abs() < 1e-12, "{t}");
        assert!((p - 0.369_109_117_862_548_6).abs() < 1e-9, "{p}");
        // Both variances 0, equal means: NaN statistic -> 0, p -> 1.
        assert_eq!(welch((1.0, 0.0, 3.0), (1.0, 0.0, 3.0)), (0.0, 1.0));
    }
}
