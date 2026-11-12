//! Highly variable genes from the statistics passes' moments (task 6.3): scanpy
//! 1.12.5's `highly_variable_genes`, step by step.
//!
//! Does: `seurat` and `cell_ranger` (dispersion = variance / mean per gene; `seurat`
//! on `expm1` of the data with log dispersions and `log1p` means; genes binned by mean,
//! 20 equal-width bins (`pd.cut`) or the 10th to 100th percentiles (`np.percentile`);
//! dispersions normalised within their bin by mean and standard deviation, or median
//! and MAD; flagged as the `n_top_genes` highest normalised dispersions, or by scanpy's
//! default cutoffs) and `seurat_v3` (a loess of `log10` variance on `log10` mean,
//! [`seurat_v3_fit`], before the second pass; normalised variance of the clipped values;
//! the `n_top_genes` best ranked). With batches each batch is ranked on its own cells
//! (genes it never sees left out, for the dispersion flavours) and the batches are
//! combined as scanpy does: means of the statistics, the number of batches flagging a
//! gene, then a sort. Pandas and numpy reductions are followed where they decide bits
//! or ties: Kahan means and Welford variances of groupby, `np.median`, statsmodels'
//! `mad`, numpy's percentile interpolation, stable sorts.
//! Does not: accumulate anything (see `cross_cell.rs`), scanpy's other parameters
//! (cutoffs, `n_bins`, `span`, `subset`, `seurat_v3_paper`), or the `mean_bin` column.

use std::cmp::Ordering;

use kyto_error::KytoError;
use kyto_plan::HvgFlavor;

use crate::loess::loess_fitted;
use crate::statistics::{GeneMoments, HighlyVariableGeneStatistics, HvgValues};

/// scanpy's default cutoffs without `n_top_genes`.
const MIN_DISPERSION: f64 = 0.5;
const MAX_DISPERSION: f64 = f64::INFINITY;
const MIN_MEAN: f64 = 0.0125;
const MAX_MEAN: f64 = 3.0;
/// scanpy's `n_bins`.
const N_BINS: usize = 20;
/// scanpy's `span` for `seurat_v3`.
const SPAN: f64 = 0.3;
/// scanpy's `n_top_genes` for `seurat_v3` when none is given.
pub(crate) const SEURAT_V3_N_TOP_GENES: u64 = 2000;
/// `scipy.stats.norm.ppf(0.75)`: statsmodels' `mad` divides by it.
const MAD_SCALE: f64 = 0.674_489_750_196_081_7;

/// `seurat_v3`'s fit for one group of cells after the first pass: each gene's clip
/// value and regularized standard deviation, and the mean the second pass centres on.
#[derive(Debug, Clone)]
pub(crate) struct SeuratV3Fit {
    pub(crate) clip: Vec<f64>,
    pub(crate) regularized_std: Vec<f64>,
    pub(crate) means: Vec<f64>,
}

/// Fit `seurat_v3`'s loess for one group: without batches on the moments' means and
/// variances (`mean_var`), with batches on Welford's (scanpy's `aggregate`, which the
/// moments must carry). `reg_std = sqrt(10^fit)` for genes of positive variance (1
/// for the others), `clip = reg_std * sqrt(n) + mean`.
///
/// # Errors
/// `KY-E215` if the loess fit fails (skmisc stops with "span too small"): too few genes
/// of positive variance, or too few distinct means among them, in `batch` (or all
/// cells).
pub(crate) fn seurat_v3_fit(
    moments: &GeneMoments,
    batch: Option<&str>,
) -> Result<SeuratV3Fit, KytoError> {
    let (means, variances) = match &moments.welford {
        Some(_) => aggregate_mean_var(moments),
        None => (moments.means(), moments.variances()),
    };
    let positive: Vec<usize> = (0..variances.len())
        .filter(|&gene| variances[gene] > 0.0)
        .collect();
    let mut fitted = vec![0.0; variances.len()];
    if !positive.is_empty() {
        let x: Vec<f64> = positive.iter().map(|&gene| means[gene].log10()).collect();
        let y: Vec<f64> = positive
            .iter()
            .map(|&gene| variances[gene].log10())
            .collect();
        for (&gene, value) in positive.iter().zip(loess_fitted(&x, &y, SPAN)) {
            if !value.is_finite() {
                let place = batch.map_or_else(String::new, |label| format!(" in batch '{label}'"));
                return Err(KytoError::InvalidArgument {
                    function: "highly_variable_genes".to_owned(),
                    parameter: "flavor".to_owned(),
                    value: "'seurat_v3'".to_owned(),
                    problem: format!(
                        "seurat_v3's loess fit of log variance on log mean failed{place}: \
                         {} genes of positive variance, too few or too few distinct means \
                         for span {SPAN} (skmisc stops with 'span too small')",
                        positive.len()
                    ),
                    accepted: "more genes of positive variance with distinct means, or \
                               flavor 'seurat'"
                        .to_owned(),
                });
            }
            fitted[gene] = value;
        }
    }
    let sqrt_n = (moments.n_cells as f64).sqrt();
    let regularized_std: Vec<f64> = fitted
        .iter()
        .map(|value| 10_f64.powf(*value).sqrt())
        .collect();
    let clip = regularized_std
        .iter()
        .zip(&means)
        .map(|(std, mean)| std * sqrt_n + mean)
        .collect();
    Ok(SeuratV3Fit {
        clip,
        regularized_std,
        means,
    })
}

/// Means and variances (`ddof=1`) from Welford's moments of the stored values merged
/// with the group's zeros, as scanpy's `mean_var_csr` and `Aggregate.mean_var`.
fn aggregate_mean_var(moments: &GeneMoments) -> (Vec<f64>, Vec<f64>) {
    let Some(welford) = &moments.welford else {
        unreachable!("called with Welford's moments")
    };
    let n_cells = moments.n_cells;
    let n = n_cells as f64;
    let n_genes = welford.counts.len();
    let mut means = vec![0.0; n_genes];
    let mut variances = vec![0.0; n_genes];
    if n_cells == 0 {
        return (means, variances);
    }
    for gene in 0..n_genes {
        let stored = welford.counts[gene];
        let zeros = n_cells - stored;
        let mut mean = welford.means[gene];
        let mut squared = welford.squared_deviations[gene];
        if zeros > 0 && stored > 0 {
            let mean_stored = mean;
            mean = mean_stored * stored as f64 / n;
            squared += mean_stored * mean_stored * stored as f64 * zeros as f64 / n;
        }
        means[gene] = mean;
        let denominator = if n_cells > 1 { n - 1.0 } else { f64::NAN };
        variances[gene] = squared / n * (n / denominator);
    }
    (means, variances)
}

/// The gene columns of `statistics` (its moments and fits computed), in the plan's
/// schema order; `gene_ids` breaks ties between batched dispersion genes as scanpy's
/// groupby (sorted by gene ID).
///
/// # Errors
/// `KY-E215` for `cell_ranger` when the percentile edges of the gene means repeat
/// (pandas' "Bin edges must be unique").
pub(crate) fn columns(
    statistics: &HighlyVariableGeneStatistics,
    n_top_genes: Option<u64>,
    gene_ids: &[String],
) -> Result<Vec<(String, HvgValues)>, KytoError> {
    match statistics.flavor {
        HvgFlavor::Seurat | HvgFlavor::CellRanger => {
            dispersion_columns(statistics, n_top_genes, gene_ids)
        }
        HvgFlavor::SeuratV3 => Ok(seurat_v3_columns(
            statistics,
            n_top_genes.unwrap_or(SEURAT_V3_N_TOP_GENES),
        )),
    }
}

/// One group's dispersion statistics (`_highly_variable_genes_single_batch`).
struct Dispersions {
    means: Vec<f64>,
    dispersions: Vec<f64>,
    normalized: Vec<f64>,
    highly_variable: Vec<bool>,
}

fn dispersion_columns(
    statistics: &HighlyVariableGeneStatistics,
    n_top_genes: Option<u64>,
    gene_ids: &[String],
) -> Result<Vec<(String, HvgValues)>, KytoError> {
    let flavor = statistics.flavor;
    if statistics.batches.is_empty() {
        let single = dispersions(&statistics.moments, flavor, false, n_top_genes)?;
        return Ok(vec![
            (
                "highly_variable".to_owned(),
                HvgValues::Boolean(single.highly_variable),
            ),
            ("means".to_owned(), HvgValues::Float64(single.means)),
            (
                "dispersions".to_owned(),
                HvgValues::Float64(single.dispersions),
            ),
            (
                "dispersions_norm".to_owned(),
                HvgValues::Float64(single.normalized),
            ),
        ]);
    }
    let batches = statistics
        .batches
        .iter()
        .map(|batch| dispersions(&batch.moments, flavor, true, n_top_genes))
        .collect::<Result<Vec<_>, _>>()?;
    let n_genes = statistics.moments.sums.len();
    // groupby(gene).agg(mean): Kahan means over the batches, NaN skipped.
    let mean_over = |pick: fn(&Dispersions) -> &Vec<f64>| -> Vec<f64> {
        (0..n_genes)
            .map(|gene| kahan_mean(batches.iter().map(|batch| pick(batch)[gene])))
            .collect()
    };
    let means = mean_over(|batch| &batch.means);
    let dispersions = mean_over(|batch| &batch.dispersions);
    let mut normalized = mean_over(|batch| &batch.normalized);
    let n_batches: Vec<i64> = (0..n_genes)
        .map(|gene| {
            batches
                .iter()
                .filter(|batch| batch.highly_variable[gene])
                .count() as i64
        })
        .collect();
    let intersection: Vec<bool> = n_batches
        .iter()
        .map(|&count| count == batches.len() as i64)
        .collect();
    let highly_variable = match n_top_genes {
        Some(n_top_genes) => {
            // Sorted by batches flagging the gene, then normalised dispersion (both
            // descending, NaN last), ties in groupby's gene order (sorted IDs).
            let mut order: Vec<usize> = (0..n_genes).collect();
            order.sort_by(|&left, &right| {
                n_batches[right]
                    .cmp(&n_batches[left])
                    .then_with(|| descending_nan_last(normalized[left], normalized[right]))
                    .then_with(|| gene_ids[left].cmp(&gene_ids[right]))
            });
            let mut flagged = vec![false; n_genes];
            for &gene in order
                .iter()
                .take(usize::try_from(n_top_genes).unwrap_or(usize::MAX))
            {
                flagged[gene] = true;
            }
            flagged
        }
        None => {
            for value in &mut normalized {
                if value.is_nan() {
                    *value = 0.0;
                }
            }
            (0..n_genes)
                .map(|gene| in_bounds(means[gene], normalized[gene]))
                .collect()
        }
    };
    Ok(vec![
        (
            "highly_variable".to_owned(),
            HvgValues::Boolean(highly_variable),
        ),
        ("means".to_owned(), HvgValues::Float64(means)),
        ("dispersions".to_owned(), HvgValues::Float64(dispersions)),
        (
            "dispersions_norm".to_owned(),
            HvgValues::Float64(normalized),
        ),
        (
            "highly_variable_nbatches".to_owned(),
            HvgValues::Int64(n_batches),
        ),
        (
            "highly_variable_intersection".to_owned(),
            HvgValues::Boolean(intersection),
        ),
    ])
}

/// scanpy's `_highly_variable_genes_single_batch` on one group's moments; with
/// `filter_unexpressed`, genes without a positive value take no part and get zeros.
fn dispersions(
    moments: &GeneMoments,
    flavor: HvgFlavor,
    filter_unexpressed: bool,
    n_top_genes: Option<u64>,
) -> Result<Dispersions, KytoError> {
    let n_genes = moments.sums.len();
    let kept: Vec<usize> = (0..n_genes)
        .filter(|&gene| !filter_unexpressed || moments.positives[gene] > 0)
        .collect();
    let all_means = moments.means();
    let all_variances = moments.variances();
    let mut means = Vec::with_capacity(kept.len());
    let mut dispersions = Vec::with_capacity(kept.len());
    for &gene in &kept {
        let mut mean = all_means[gene];
        if mean == 0.0 {
            mean = 1e-12;
        }
        let mut dispersion = all_variances[gene] / mean;
        if flavor == HvgFlavor::Seurat {
            if dispersion == 0.0 {
                dispersion = f64::NAN;
            }
            dispersion = dispersion.ln();
            mean = mean.ln_1p();
        }
        means.push(mean);
        dispersions.push(dispersion);
    }
    let bins = match flavor {
        HvgFlavor::Seurat => equal_width_bins(&means),
        _ => percentile_bins(&means)?,
    };
    let normalized = normalize_by_bin(&dispersions, &bins, flavor);
    let flagged: Vec<bool> = match n_top_genes {
        Some(n_top_genes) => {
            let mut finite: Vec<f64> = normalized
                .iter()
                .copied()
                .filter(|value| !value.is_nan())
                .collect();
            let n = usize::try_from(n_top_genes)
                .unwrap_or(usize::MAX)
                .min(kept.len())
                .min(finite.len());
            if n == 0 {
                vec![false; kept.len()]
            } else {
                finite.sort_unstable_by(|left, right| right.total_cmp(left));
                let cutoff = finite[n - 1];
                normalized
                    .iter()
                    .map(|&value| nan_to_num(value, f64::NEG_INFINITY) >= cutoff)
                    .collect()
            }
        }
        None => means
            .iter()
            .zip(&normalized)
            .map(|(&mean, &value)| in_bounds(mean, nan_to_num(value, 0.0)))
            .collect(),
    };
    let mut result = Dispersions {
        means: vec![0.0; n_genes],
        dispersions: vec![0.0; n_genes],
        normalized: vec![0.0; n_genes],
        highly_variable: vec![false; n_genes],
    };
    for (index, &gene) in kept.iter().enumerate() {
        result.means[gene] = means[index];
        result.dispersions[gene] = dispersions[index];
        result.normalized[gene] = normalized[index];
        result.highly_variable[gene] = flagged[index];
    }
    Ok(result)
}

/// scanpy's default cutoffs (`_Cutoffs.in_bounds`).
fn in_bounds(mean: f64, normalized: f64) -> bool {
    mean > MIN_MEAN && mean < MAX_MEAN && normalized > MIN_DISPERSION && normalized < MAX_DISPERSION
}

/// `np.nan_to_num(value, nan=nan)`: infinities become the largest finite values.
fn nan_to_num(value: f64, nan: f64) -> f64 {
    if value.is_nan() {
        nan
    } else if value == f64::INFINITY {
        f64::MAX
    } else if value == f64::NEG_INFINITY {
        f64::MIN
    } else {
        value
    }
}

/// `left` before `right` when larger; NaN after every number.
fn descending_nan_last(left: f64, right: f64) -> Ordering {
    match (left.is_nan(), right.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => right.partial_cmp(&left).unwrap_or(Ordering::Equal),
    }
}

/// Each value's bin (`None` for NaN) among 20 equal-width bins over its range: pandas'
/// `pd.cut(values, bins=20)` (`np.linspace` edges, the first lowered by 0.1 % of the
/// range; a constant range widened by 0.1 % first).
fn equal_width_bins(values: &[f64]) -> Vec<Option<usize>> {
    let finite = values.iter().copied().filter(|value| !value.is_nan());
    let Some(low) = finite.clone().reduce(f64::min) else {
        return vec![None; values.len()];
    };
    let high = finite.reduce(f64::max).unwrap_or(low);
    let mut edges;
    if low == high {
        let low = low - if low == 0.0 { 0.001 } else { 0.001 * low.abs() };
        let high = high
            + if high == 0.0 {
                0.001
            } else {
                0.001 * high.abs()
            };
        edges = linspace(low, high, N_BINS + 1);
    } else {
        edges = linspace(low, high, N_BINS + 1);
        edges[0] -= (high - low) * 0.001;
    }
    values
        .iter()
        .map(|&value| {
            if value.is_nan() {
                return None;
            }
            // include_lowest: the first edge belongs to the first bin.
            let index = edges.partition_point(|&edge| edge < value).max(1);
            (index < edges.len()).then(|| index - 1)
        })
        .collect()
}

/// `np.linspace(start, stop, num)`: `i * step + start`, the last one `stop`.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let step = (stop - start) / (num - 1) as f64;
    let mut values: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    values[num - 1] = stop;
    values
}

/// Each value's bin among `cell_ranger`'s: edges `-inf`, the 10th, 15th, ..., 100th
/// percentiles (`np.percentile`, linear) and `inf`, bins right-closed (`pd.cut`).
///
/// # Errors
/// `KY-E215` if two edges are equal (pandas raises "Bin edges must be unique").
fn percentile_bins(values: &[f64]) -> Result<Vec<Option<usize>>, KytoError> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let mut edges = vec![f64::NEG_INFINITY];
    edges.extend((10..=100).step_by(5).map(|p| percentile(&sorted, p)));
    edges.push(f64::INFINITY);
    if edges.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(KytoError::InvalidArgument {
            function: "highly_variable_genes".to_owned(),
            parameter: "flavor".to_owned(),
            value: "'cell_ranger'".to_owned(),
            problem: "cell_ranger bins genes by the 10th to 100th percentiles of their \
                      means, and two of them are equal (too many genes share a mean, for \
                      example genes seen in no cell)"
                .to_owned(),
            accepted: "genes with distinct mean percentiles: filter out genes seen in few \
                       cells first (scanpy raises 'Bin edges must be unique' here), or \
                       flavor 'seurat'"
                .to_owned(),
        });
    }
    Ok(values
        .iter()
        .map(|&value| {
            if value.is_nan() {
                return None;
            }
            let index = edges.partition_point(|&edge| edge < value);
            (index > 0 && index < edges.len()).then(|| index - 1)
        })
        .collect())
}

/// `np.percentile(values, p)` of ascending `sorted` (linear method): NaN if any value
/// is NaN or there is none.
fn percentile(sorted: &[f64], p: usize) -> f64 {
    let n = sorted.len();
    if n == 0 || sorted[n - 1].is_nan() {
        return f64::NAN;
    }
    let quantile = p as f64 / 100.0;
    let virtual_index = (n - 1) as f64 * quantile;
    if virtual_index >= (n - 1) as f64 {
        return sorted[n - 1];
    }
    let previous = virtual_index.floor();
    let gamma = virtual_index - previous;
    let previous = previous as usize;
    let (a, b) = (sorted[previous], sorted[previous + 1]);
    // numpy's `_lerp`.
    let difference = b - a;
    if gamma >= 0.5 {
        b - difference * (1.0 - gamma)
    } else {
        a + difference * gamma
    }
}

/// Each dispersion normalised by its bin: `(dispersion - average) / deviation`, the
/// bin's mean and standard deviation (`seurat`; a bin without a deviation gets
/// average 0 and the mean as deviation, scanpy's single-gene bins) or median and MAD
/// (`cell_ranger`). NaN outside every bin.
fn normalize_by_bin(dispersions: &[f64], bins: &[Option<usize>], flavor: HvgFlavor) -> Vec<f64> {
    let mut members: Vec<Vec<f64>> = vec![Vec::new(); N_BINS];
    for (&dispersion, bin) in dispersions.iter().zip(bins) {
        if let Some(bin) = bin {
            members[*bin].push(dispersion);
        }
    }
    let statistics: Vec<(f64, f64)> = members
        .iter()
        .map(|values| match flavor {
            HvgFlavor::Seurat => {
                let average = kahan_mean(values.iter().copied());
                let deviation = welford_std(values);
                if deviation.is_nan() {
                    (0.0, average)
                } else {
                    (average, deviation)
                }
            }
            _ => (median_skip_nan(values), mad(values)),
        })
        .collect();
    dispersions
        .iter()
        .zip(bins)
        .map(|(&dispersion, bin)| match bin {
            Some(bin) => {
                let (average, deviation) = statistics[*bin];
                (dispersion - average) / deviation
            }
            None => f64::NAN,
        })
        .collect()
}

/// pandas' groupby mean: Kahan summation of the non-NaN values over their count (NaN
/// for none).
fn kahan_mean(values: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut compensation, mut count) = (0.0_f64, 0.0_f64, 0_u64);
    for value in values {
        if value.is_nan() {
            continue;
        }
        count += 1;
        let y = value - compensation;
        let t = sum + y;
        compensation = t - sum - y;
        if compensation.is_nan() {
            compensation = 0.0;
        }
        sum = t;
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// pandas' groupby std (`ddof=1`): Welford over the non-NaN values, NaN for fewer than
/// two.
fn welford_std(values: &[f64]) -> f64 {
    let (mut count, mut mean, mut squared) = (0_u64, 0.0_f64, 0.0_f64);
    for &value in values {
        if value.is_nan() {
            continue;
        }
        count += 1;
        let previous = mean;
        mean += (value - previous) / count as f64;
        squared += (value - mean) * (value - previous);
    }
    if count <= 1 {
        f64::NAN
    } else {
        (squared / (count - 1) as f64).sqrt()
    }
}

/// `np.median` of `values` (NaN if any is NaN or there is none).
fn median(values: &[f64]) -> f64 {
    if values.is_empty() || values.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// pandas' groupby median: the median of the non-NaN values (NaN for none).
fn median_skip_nan(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    median(&finite)
}

/// statsmodels' `mad(values)`: `median(|values - median(values)| / 0.6745)` (NaN if any
/// value is NaN or there is none).
fn mad(values: &[f64]) -> f64 {
    let center = median(values);
    let deviations: Vec<f64> = values
        .iter()
        .map(|value| (value - center).abs() / MAD_SCALE)
        .collect();
    median(&deviations)
}

/// `seurat_v3`'s columns: each group's normalised variance of the clipped values, its
/// rank among the group's genes, then scanpy's combination over groups.
fn seurat_v3_columns(
    statistics: &HighlyVariableGeneStatistics,
    n_top_genes: u64,
) -> Vec<(String, HvgValues)> {
    let groups: Vec<_> = if statistics.batches.is_empty() {
        vec![statistics.clipped.as_ref()]
    } else {
        statistics
            .batches
            .iter()
            .map(|batch| batch.clipped.as_ref())
            .collect()
    };
    let n_genes = statistics.moments.sums.len();
    let n_top = n_top_genes as f64;
    let mut normalized_variances: Vec<Vec<f64>> = Vec::with_capacity(groups.len());
    let mut ranks: Vec<Vec<f64>> = Vec::with_capacity(groups.len());
    for clipped in groups {
        let Some(clipped) = clipped else {
            unreachable!("seurat_v3's second pass sets every group's clipped moments")
        };
        let moments = &clipped.moments;
        let n = moments.n_cells as f64;
        let normalized: Vec<f64> = (0..n_genes)
            .map(|gene| {
                let std = clipped.regularized_std[gene];
                let mean = clipped.means[gene];
                (1.0 / ((n - 1.0) * (std * std)))
                    * ((n * (mean * mean)) + moments.sums_of_squares[gene]
                        - 2.0 * moments.sums[gene] * mean)
            })
            .collect();
        // argsort(argsort(-normalized)): stable, NaN last, -0 equal to 0.
        let mut order: Vec<usize> = (0..n_genes).collect();
        order.sort_by(|&left, &right| descending_nan_last(normalized[left], normalized[right]));
        let mut rank = vec![0.0; n_genes];
        for (position, &gene) in order.iter().enumerate() {
            rank[gene] = f64::from(position as f32);
        }
        normalized_variances.push(normalized);
        ranks.push(rank);
    }
    let n_batches: Vec<i64> = (0..n_genes)
        .map(|gene| ranks.iter().filter(|rank| rank[gene] < n_top).count() as i64)
        .collect();
    let median_rank: Vec<f64> = (0..n_genes)
        .map(|gene| {
            let kept: Vec<f64> = ranks
                .iter()
                .map(|rank| rank[gene])
                .filter(|&rank| rank < n_top)
                .collect();
            median(&kept)
        })
        .collect();
    let variances_norm: Vec<f64> = (0..n_genes)
        .map(|gene| {
            let sum = normalized_variances
                .iter()
                .fold(0.0, |sum, normalized| sum + normalized[gene]);
            sum / normalized_variances.len() as f64
        })
        .collect();
    // Sorted by median rank (NaN last), then batches flagging the gene (descending),
    // ties in gene order; the first n_top_genes are flagged.
    let mut order: Vec<usize> = (0..n_genes).collect();
    order.sort_by(|&left, &right| {
        descending_nan_last(-median_rank[left], -median_rank[right])
            .then_with(|| n_batches[right].cmp(&n_batches[left]))
    });
    let mut highly_variable = vec![false; n_genes];
    for &gene in order
        .iter()
        .take(usize::try_from(n_top_genes).unwrap_or(usize::MAX))
    {
        highly_variable[gene] = true;
    }
    let mut columns = vec![
        (
            "highly_variable".to_owned(),
            HvgValues::Boolean(highly_variable),
        ),
        (
            "highly_variable_rank".to_owned(),
            HvgValues::Float64(median_rank),
        ),
        (
            "means".to_owned(),
            HvgValues::Float64(statistics.moments.means()),
        ),
        (
            "variances".to_owned(),
            HvgValues::Float64(statistics.moments.variances()),
        ),
        (
            "variances_norm".to_owned(),
            HvgValues::Float64(variances_norm),
        ),
    ];
    if !statistics.batches.is_empty() {
        columns.push((
            "highly_variable_nbatches".to_owned(),
            HvgValues::Int64(n_batches),
        ));
    }
    columns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_follow_numpy() {
        let sorted = [1.0, 2.0, 4.0, 8.0];
        // virtual index 3 * 0.1 = 0.3: 1 + 1 * 0.3
        assert_eq!(percentile(&sorted, 10), 1.3);
        // 3 * 0.5 = 1.5, gamma >= 0.5: 4 - 2 * 0.5
        assert_eq!(percentile(&sorted, 50), 3.0);
        assert_eq!(percentile(&sorted, 100), 8.0);
        assert!(percentile(&[], 50).is_nan());
    }

    #[test]
    fn equal_width_bins_follow_pandas_cut() {
        let values = [0.0, 1.0, 2.0, 0.05, 0.1];
        let bins = equal_width_bins(&values);
        // Edges 0 (lowered), 0.1, 0.2, ...: 0 in bin 0, 0.05 in bin 0, 0.1 in bin 0
        // (right-closed), 1 in bin 9, 2 in the last.
        assert_eq!(bins, [Some(0), Some(9), Some(19), Some(0), Some(0)]);
        assert_eq!(equal_width_bins(&[3.0, 3.0]), [Some(9), Some(9)]);
        assert_eq!(equal_width_bins(&[f64::NAN]), [None]);
    }

    #[test]
    fn group_statistics_follow_pandas() {
        assert_eq!(kahan_mean([1.0, f64::NAN, 2.0].into_iter()), 1.5);
        assert!(kahan_mean(std::iter::empty()).is_nan());
        assert!((welford_std(&[1.0, 2.0, 3.0, 4.0]) - 1.290_994_448_735_805_6).abs() < 1e-15);
        assert!(welford_std(&[1.0]).is_nan());
        assert_eq!(median(&[3.0, 1.0, 2.0, 10.0]), 2.5);
        // |x - 2.5| = [0.5, 1.5, 0.5, 7.5]: median 1.0, over 0.6745.
        assert!((mad(&[3.0, 1.0, 2.0, 10.0]) - 1.0 / MAD_SCALE).abs() < 1e-15);
    }

    #[test]
    fn repeated_percentile_edges_are_refused() {
        let error = percentile_bins(&[0.0; 30]).unwrap_err();
        assert_eq!(error.code(), "KY-E215");
    }

    #[test]
    fn welford_moments_give_mean_var() {
        // Three leaves of one gene's values (zeros not stored), combined in order.
        let leaves: [&[f64]; 3] = [&[1.0, 4.0, 0.0], &[0.0, 0.0], &[7.0, 2.0, 2.0, 0.0]];
        let mut total = GeneMoments::new(1, true);
        for leaf in leaves {
            let mut moments = GeneMoments::new(1, true);
            for &value in leaf {
                moments.n_cells += 1;
                moments.sums[0] += value;
                moments.sums_of_squares[0] += value * value;
                if value != 0.0 {
                    moments.welford.as_mut().unwrap().add(0, value);
                }
            }
            total.combine(&moments);
        }
        let (means, variances) = aggregate_mean_var(&total);
        assert!((means[0] - total.means()[0]).abs() < 1e-14, "{means:?}");
        assert!(
            (variances[0] - total.variances()[0]).abs() < 1e-14,
            "{variances:?} {:?}",
            total.variances()
        );
    }

    #[test]
    fn a_failed_loess_fit_is_refused() {
        // Two genes of positive variance (the other four constant): too few points
        // for span 0.3, skmisc stops; kyto must not rank on a NaN fit (bug-hunter).
        let mut moments = GeneMoments::new(6, false);
        for cell in 0..10_u32 {
            moments.n_cells += 1;
            for gene in 0..6 {
                let value = match gene {
                    0 => f64::from(cell % 3),
                    1 => f64::from(cell % 5) * 2.0,
                    _ => 1.0,
                };
                moments.sums[gene] += value;
                moments.sums_of_squares[gene] += value * value;
            }
        }
        let Err(error) = seurat_v3_fit(&moments, Some("b1")) else {
            panic!("a loess fit over two points ran");
        };
        assert_eq!(error.code(), "KY-E215");
        assert!(error.to_string().contains("batch 'b1'"), "{error}");
    }

    #[test]
    fn nan_to_num_maps_infinities_to_finite() {
        assert_eq!(nan_to_num(f64::NAN, -1.0), -1.0);
        assert_eq!(nan_to_num(f64::INFINITY, 0.0), f64::MAX);
        assert_eq!(nan_to_num(f64::NEG_INFINITY, 0.0), f64::MIN);
    }
}
