//! Doublet detection (task 6.12): scanpy 1.12.5's `pp.scrublet`, step for step, on
//! counts held in memory.
//!
//! Does, for each batch (every cell without a `batch_key`), as scanpy's
//! `_run_scrublet`: keep the genes with a positive count in at least 3 of the batch's
//! cells, then the cells with a positive count in at least 3 of those genes (the cells
//! scored; the others get no score); `normalize_total` to the median total and
//! `log1p` (kyto's steps of task 6.2); the `seurat` highly variable genes at scanpy's
//! default cutoffs (task 6.3's code); simulate doublets as the sum of the counts of two
//! cells (`floor(n_cells x sim_doublet_ratio)` distinct ordered pairs, a cell possibly
//! paired with itself, as scanpy's `sample_comb`; drawn from the seed with SplitMix64,
//! or given, [`Doublets::Given`]); `normalize_total` the cells' normalised values and the
//! doublets' counts on those genes to 1e6 (`float32` steps); z-score both with the
//! cells' means and `ddof=1` standard deviations (`f64`, `fast_array_utils`' formula);
//! a PCA of the cells (`n_prin_comps` components of the exact covariance: the genes'
//! Gram matrix of the centred cells and its eigenvectors with faer on one thread,
//! signed as scikit-learn's `svd_flip(u_based_decision=False)`), the cells and the
//! doublets projected on it; the exact `k_adj - 1` nearest other rows of every cell and
//! doublet (Euclidean, in `f64`: `doublet_neighbors`), where
//! `k = n_neighbors` or `round(0.5 sqrt(n_cells))` and `k_adj = round(k (1 + n_sim /
//! n_cells))` (Python's rounding, half to even; lowered as scanpy's `Neighbors` when
//! above the rows), a doublet counting itself too unless two rows are equal (scanpy's
//! `_has_self_column` quirk, see `has_duplicate_rows`); Scrublet's Bayesian score of
//! the doublets among them
//! (`q = (d + 1) / (k_adj + 2)`, `q rho / r / (1 - rho - q (1 - rho - rho / r))`);
//! the threshold (given, or `scrublet_threshold.rs` on the doublets' scores) and the
//! calls `score > threshold` (none when no threshold is found, as scanpy).
//! Does not: gather the counts (the pipeline's pass, `scrublet_pass.rs`), or scanpy's
//! approximate search above 8,192 rows (pynndescent): kyto always searches exactly.

use std::collections::{BTreeMap, HashSet};

use faer::{Mat, Side};
use kyto_error::KytoError;
use kyto_plan::{HvgFlavor, Scrublet};
use kyto_store::{CellBlock, CellValues};

use crate::cross_cell::median;
use crate::elementwise::{cell_totals, normalize_divisor};
use crate::hvg;
use crate::kernels::{KernelValue, Kernels};
use crate::run_control::RunControl;
use crate::statistics::{GeneMoments, HighlyVariableGeneStatistics, HvgValues};
use crate::umap_spectral::Draws;

/// scanpy's `filter_genes(min_cells=3)` and `filter_cells(min_genes=3)` in
/// `pp.scrublet`.
const MIN_CELLS_PER_GENE: u64 = 3;
const MIN_GENES_PER_CELL: u64 = 3;
/// The total `pp.scrublet` normalises cells and doublets to before the z-score.
const TARGET_SUM: f64 = 1e6;
/// The SplitMix64 stream the doublets' pairs are drawn from.
const PAIRS_STREAM: u64 = 0x5C2B;

/// Where the simulated doublets' parents come from.
#[derive(Debug, Clone, PartialEq)]
pub enum Doublets {
    /// Drawn from `Scrublet::seed` (R7).
    Seeded,
    /// Given per batch label (`""` without a `batch_key`): pairs of positions among
    /// the batch's scored cells (scanpy's `uns['scrublet']['doublet_parents']`).
    Given(BTreeMap<String, Vec<[u32; 2]>>),
}

/// Scrublet's results for one batch.
#[derive(Debug, Clone, PartialEq)]
pub struct ScrubletBatch {
    /// The batch's label (`""` without a `batch_key`).
    pub label: String,
    /// The rows of the input scored (the batch's cells that passed the filters), in
    /// row order; scores and parents refer to cells by their position here.
    pub cells: Vec<usize>,
    /// The scores of those cells.
    pub scores: Vec<f64>,
    /// The simulated doublets' scores (scanpy's `doublet_scores_sim`).
    pub simulated_scores: Vec<f64>,
    /// The simulated doublets' parents (scanpy's `doublet_parents`).
    pub parents: Vec<[u32; 2]>,
    /// The threshold; `None` when it was automatic and not found (no calls).
    pub threshold: Option<f64>,
    /// The highly variable genes used, as positions in the input's genes.
    pub genes: Vec<u32>,
    /// The neighbours the score counts, the row itself included (`k_adj`).
    pub n_neighbors: u64,
}

/// Scrublet's results: per input row the score and call (`None` for a row not scored),
/// and each batch's details.
#[derive(Debug, Clone, PartialEq)]
pub struct ScrubletResult {
    /// Each row's doublet score (scanpy's `obs['doublet_score']`).
    pub scores: Vec<Option<f64>>,
    /// Each row's call (scanpy's `obs['predicted_doublet']`).
    pub predicted: Vec<Option<bool>>,
    /// Each batch, in label order.
    pub batches: Vec<ScrubletBatch>,
}

/// Scrublet on the rows of `counts` (cells x genes), each batch of `labels` (one label
/// per row) on its own, as `params` asks (`params.score_into`, `predicted_into` are not
/// read), with the doublets' parents from `doublets`.
///
/// # Errors
/// `KY-E215` for a batch with fewer than 2 scored cells or no highly variable gene, a
/// `sim_doublet_ratio` that simulates no doublet or more pairs than exist,
/// `n_prin_comps` not below both the scored cells and the highly variable genes
/// (scikit-learn's `arpack` refuses it), a non-finite value, given parents that are not
/// `floor(n_cells x sim_doublet_ratio)` pairs of scored cells, or labels that are not
/// one per row.
pub fn scrublet(
    counts: &CellBlock,
    labels: Option<&[String]>,
    params: &Scrublet,
    doublets: &Doublets,
) -> Result<ScrubletResult, KytoError> {
    run(counts, labels, params, doublets, &RunControl::new())
}

/// [`scrublet`], cancelled through `control` (`KY-E503`), searching on the current
/// rayon pool.
pub(crate) fn run(
    counts: &CellBlock,
    labels: Option<&[String]>,
    params: &Scrublet,
    doublets: &Doublets,
    control: &RunControl,
) -> Result<ScrubletResult, KytoError> {
    let n_rows = counts.indptr().len().saturating_sub(1);
    if let Some(labels) = labels
        && labels.len() != n_rows
    {
        return Err(invalid(
            "batch_key",
            &format!("{} labels", labels.len()),
            &format!("there are {n_rows} cells"),
            "one label per cell",
        ));
    }
    let matrix = Csr::from_block(counts);
    let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for row in 0..n_rows {
        let label = labels.map_or("", |labels| labels[row].as_str());
        groups.entry(label).or_default().push(row);
    }
    let mut result = ScrubletResult {
        scores: vec![None; n_rows],
        predicted: vec![None; n_rows],
        batches: Vec::with_capacity(groups.len()),
    };
    for (label, rows) in groups {
        control.check()?;
        let given = match doublets {
            Doublets::Seeded => None,
            Doublets::Given(given) => Some(given.get(label).ok_or_else(|| {
                invalid(
                    "doublets",
                    &format!("no pairs for batch {label:?}"),
                    "every batch needs its simulated doublets",
                    "pairs for each batch label",
                )
            })?),
        };
        let batch = score_batch(
            &matrix.rows(&rows),
            labels.is_some().then_some(label),
            params,
            given,
            control,
        )?;
        for (position, &cell) in batch.cells.iter().enumerate() {
            let row = rows[cell];
            let score = batch.scores[position];
            result.scores[row] = Some(score);
            result.predicted[row] = Some(batch.threshold.is_some_and(|t| score > t));
        }
        let cells = batch.cells.iter().map(|&cell| rows[cell]).collect();
        result.batches.push(ScrubletBatch { cells, ..batch });
    }
    Ok(result)
}

/// A CSR matrix of `float32` values (scanpy's data is `float32`: counts above 2^24
/// round as there).
#[derive(Debug, Clone, Default)]
struct Csr {
    indptr: Vec<u64>,
    indices: Vec<u32>,
    values: Vec<f32>,
    n_genes: usize,
}

impl Csr {
    fn from_block(block: &CellBlock) -> Self {
        let values = match block.data() {
            CellValues::F32(values) => values.to_vec(),
            CellValues::U32(values) => values.iter().map(|value| value.to_f32()).collect(),
        };
        Self {
            indptr: block.indptr().to_vec(),
            indices: block.indices().to_vec(),
            values,
            n_genes: usize::try_from(block.n_genes()).unwrap_or(usize::MAX),
        }
    }

    fn n_rows(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    fn row(&self, row: usize) -> (&[u32], &[f32]) {
        let range = self.indptr[row] as usize..self.indptr[row + 1] as usize;
        (&self.indices[range.clone()], &self.values[range])
    }

    /// The rows `rows`, in that order.
    fn rows(&self, rows: &[usize]) -> Self {
        let mut out = Self {
            indptr: vec![0],
            n_genes: self.n_genes,
            ..Self::default()
        };
        for &row in rows {
            let (indices, values) = self.row(row);
            out.push(indices.iter().copied().zip(values.iter().copied()));
        }
        out
    }

    fn push(&mut self, entries: impl Iterator<Item = (u32, f32)>) {
        for (gene, value) in entries {
            self.indices.push(gene);
            self.values.push(value);
        }
        self.indptr.push(self.indices.len() as u64);
    }

    /// The rows `rows` restricted to the genes `remap` maps (old gene -> new gene),
    /// over `n_genes` new genes.
    fn select(&self, rows: &[usize], remap: &[Option<u32>], n_genes: usize) -> Self {
        let mut out = Self {
            indptr: vec![0],
            n_genes,
            ..Self::default()
        };
        for &row in rows {
            let (indices, values) = self.row(row);
            out.push(
                indices
                    .iter()
                    .zip(values)
                    .filter_map(|(&gene, &value)| remap[gene as usize].map(|new| (new, value))),
            );
        }
        out
    }

    /// The same rows with the values `values`.
    fn with_values(&self, values: Vec<f32>) -> Self {
        Self {
            values,
            ..self.clone()
        }
    }

    /// scanpy's `normalize_total(target_sum=target_sum)` (`None`: the median of the
    /// positive totals) in its `float32` steps (task 6.2).
    fn normalize_total(&self, kernels: Kernels, target_sum: Option<f64>) -> Self {
        let mut totals = Vec::new();
        cell_totals(kernels, &self.indptr, &self.values, &mut totals);
        let target_sum = target_sum.or_else(|| {
            let mut positive: Vec<f64> = totals
                .iter()
                .map(|&total| f64::from(total as f32))
                .filter(|&total| total > 0.0)
                .collect();
            Some(median(&mut positive))
        });
        let divisors: Vec<f32> = totals
            .iter()
            .map(|&total| normalize_divisor(total, target_sum))
            .collect();
        let mut out = vec![0.0; self.values.len()];
        kernels.divide_rows(&self.indptr, &self.values, &divisors, &mut out);
        self.with_values(out)
    }

    /// Per column, the rows with a positive value.
    fn positives_per_gene(&self) -> Vec<u64> {
        let mut counts = vec![0; self.n_genes];
        for (&gene, &value) in self.indices.iter().zip(&self.values) {
            if value > 0.0 {
                counts[gene as usize] += 1;
            }
        }
        counts
    }
}

/// One batch: filters, normalisation, genes, doublets, embedding, neighbours, scores.
/// A batch too small or flat for Scrublet: `batch_key`'s value, or the layer for all
/// cells.
fn unscorable(batch: Option<&str>, problem: &str, accepted: &str) -> KytoError {
    match batch {
        Some(label) => invalid("batch_key", &format!("{label:?}"), problem, accepted),
        None => invalid("layer", "all cells", problem, accepted),
    }
}

/// One batch scored (`batch_label` its label, `None` for all cells).
fn score_batch(
    batch: &Csr,
    batch_label: Option<&str>,
    params: &Scrublet,
    given: Option<&Vec<[u32; 2]>>,
    control: &RunControl,
) -> Result<ScrubletBatch, KytoError> {
    let kernels = Kernels::detected();
    // filter_genes(min_cells=3), then filter_cells(min_genes=3) on the kept genes.
    let gene_kept: Vec<bool> = batch
        .positives_per_gene()
        .iter()
        .map(|&count| count >= MIN_CELLS_PER_GENE)
        .collect();
    let mut remap = vec![None; batch.n_genes];
    let mut kept_genes = Vec::new();
    for (gene, _) in gene_kept.iter().enumerate().filter(|(_, kept)| **kept) {
        remap[gene] = Some(kept_genes.len() as u32);
        kept_genes.push(gene as u32);
    }
    let cells: Vec<usize> = (0..batch.n_rows())
        .filter(|&row| {
            let (indices, values) = batch.row(row);
            let detected = indices
                .iter()
                .zip(values)
                .filter(|&(&gene, &value)| gene_kept[gene as usize] && value > 0.0)
                .count() as u64;
            detected >= MIN_GENES_PER_CELL
        })
        .collect();
    let n_obs = cells.len();
    if n_obs < 2 {
        return Err(unscorable(
            batch_label,
            &format!(
                "the batch has {n_obs} cells with at least {MIN_GENES_PER_CELL} genes seen in \
                 {MIN_CELLS_PER_GENE} of its cells; Scrublet needs at least 2"
            ),
            "batches of cells with counts",
        ));
    }
    let raw = batch.select(&cells, &remap, kept_genes.len());

    // normalize_total (median), log1p, seurat HVGs.
    let normalized = raw.normalize_total(kernels, None);
    let mut logged = vec![0.0; normalized.values.len()];
    kernels.log1p(&normalized.values, &mut logged);
    let highly_variable = seurat_genes(&normalized.with_values(logged), n_obs)?;
    let mut hvg_remap = vec![None; kept_genes.len()];
    let mut genes = Vec::new();
    for (gene, _) in highly_variable.iter().enumerate().filter(|(_, hv)| **hv) {
        hvg_remap[gene] = Some(genes.len() as u32);
        genes.push(kept_genes[gene]);
    }
    let n_hvg = genes.len();
    if n_hvg == 0 {
        return Err(unscorable(
            batch_label,
            "no gene of the batch passes scanpy's seurat cutoffs for highly variable genes",
            "batches with variable genes",
        ));
    }
    let all: Vec<usize> = (0..n_obs).collect();
    let observed = normalized.select(&all, &hvg_remap, n_hvg);
    let raw = raw.select(&all, &hvg_remap, n_hvg);

    // Simulated doublets.
    let n_sim = (n_obs as f64 * params.sim_doublet_ratio) as usize;
    let parents = match given {
        Some(pairs) => check_pairs(pairs, n_obs, n_sim)?,
        None => draw_pairs(n_obs, n_sim, params.seed)?,
    };
    let mut simulated = Csr {
        indptr: vec![0],
        n_genes: n_hvg,
        ..Csr::default()
    };
    for [first, second] in &parents {
        simulated.push(add_rows(raw.row(*first as usize), raw.row(*second as usize)).into_iter());
    }
    let observed = observed.normalize_total(kernels, Some(TARGET_SUM));
    let simulated = simulated.normalize_total(kernels, Some(TARGET_SUM));
    let counts_itself = !has_duplicate_rows(&[&observed, &simulated]);
    control.check()?;

    // z-score, PCA, projection.
    let (means, inverse_stds) = zscore_statistics(&observed);
    let observed = dense_zscore(&observed, &means, &inverse_stds);
    let simulated = dense_zscore(&simulated, &means, &inverse_stds);
    let n_components = usize::try_from(params.n_prin_comps).unwrap_or(usize::MAX);
    if n_components >= n_obs.min(n_hvg) {
        return Err(invalid(
            "n_prin_comps",
            &params.n_prin_comps.to_string(),
            &format!(
                "{} has {n_obs} scored cells and {n_hvg} highly variable genes; \
                 the PCA (scikit-learn's arpack) needs fewer components than both",
                batch_label.map_or_else(
                    || "the layer".to_owned(),
                    |label| format!("batch {label:?}")
                )
            ),
            &format!("an int below {}", n_obs.min(n_hvg)),
        ));
    }
    let embedding = embed(&observed, &simulated, n_components)?;
    control.check()?;

    // Neighbours and scores.
    let n_total = n_obs + n_sim;
    let k = match params.n_neighbors {
        Some(n_neighbors) => n_neighbors as f64,
        None => round_half_even(0.5 * (n_obs as f64).sqrt()),
    };
    let k_adj = round_half_even(k * (1.0 + n_sim as f64 / n_obs as f64)) as usize;
    // scanpy's `Neighbors` lowers a count above the rows to `1 + n / 2`, and keeps at
    // most every row.
    let searched = if k_adj > n_total {
        1 + n_total / 2
    } else {
        k_adj
    }
    .clamp(1, n_total);
    let mut simulated_neighbors =
        doublet_neighbors(&embedding, n_components, n_obs, searched - 1, control)?;
    if counts_itself {
        // scanpy puts each row back as its own first neighbour, so a doublet counts
        // itself (see `has_duplicate_rows`).
        for count in &mut simulated_neighbors[n_obs..] {
            *count += 1;
        }
    }
    let scores = doublet_scores(
        &simulated_neighbors,
        n_obs,
        n_sim,
        k_adj as f64,
        params.expected_doublet_rate,
    );
    let simulated_scores = scores[n_obs..].to_vec();
    let threshold = match params.threshold {
        Some(threshold) => Some(threshold),
        None => crate::scrublet_threshold::threshold_minimum(&simulated_scores),
    };
    Ok(ScrubletBatch {
        label: batch_label.unwrap_or_default().to_owned(),
        cells,
        scores: scores[..n_obs].to_vec(),
        simulated_scores,
        parents,
        threshold,
        genes,
        n_neighbors: k_adj as u64,
    })
}

/// scanpy's `seurat` highly variable genes (default cutoffs, no `n_top_genes`) of the
/// `log1p` values `logged` over `n_cells` cells: task 6.3's moments of `expm1` in
/// `float32` and its flags.
fn seurat_genes(logged: &Csr, n_cells: usize) -> Result<Vec<bool>, KytoError> {
    let mut moments = GeneMoments::new(logged.n_genes, false);
    moments.n_cells = n_cells as u64;
    for (&gene, &value) in logged.indices.iter().zip(&logged.values) {
        let gene = gene as usize;
        let value = f64::from(value);
        if value > 0.0 {
            moments.positives[gene] += 1;
        }
        let transformed = value.exp_m1() as f32;
        moments.sums[gene] += f64::from(transformed);
        moments.sums_of_squares[gene] += f64::from(transformed * transformed);
    }
    let statistics = HighlyVariableGeneStatistics {
        flavor: HvgFlavor::Seurat,
        moments,
        clipped: None,
        batches: Vec::new(),
        columns: Vec::new(),
    };
    let columns = hvg::columns(&statistics, None, &[])?;
    match columns.into_iter().next() {
        Some((_, HvgValues::Boolean(flags))) => Ok(flags),
        _ => unreachable!("highly_variable is the first column of the seurat flavour"),
    }
}

/// Whether two rows of `matrices` (taken together) are equal, value for value.
///
/// scanpy 1.12.5's classifier asks its `Neighbors` for `k_adj` neighbours, removes
/// each row's first one (itself, it assumes), then reads them back with
/// `_get_indices_distances_from_sparse_matrix`, which puts the row itself back first
/// only when no row's first stored neighbour is that row (`_has_self_column`). That
/// happens when a row ranked an exact duplicate before itself: then no row counts
/// itself (`k_adj - 1` neighbours, a doublet not counting itself); otherwise every
/// row does. Measured on every fixture batch (memory/scrublet_reference.md): scanpy's
/// manifold had duplicate rows exactly when the self column was missing. Equal
/// normalised rows give equal z-scores and projections, so they are checked here.
fn has_duplicate_rows(matrices: &[&Csr]) -> bool {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut seen: std::collections::HashMap<u64, Vec<(usize, usize)>> =
        std::collections::HashMap::new();
    for (which, matrix) in matrices.iter().enumerate() {
        for row in 0..matrix.n_rows() {
            let (indices, values) = matrix.row(row);
            let mut hasher = DefaultHasher::new();
            indices.hash(&mut hasher);
            for value in values {
                value.to_bits().hash(&mut hasher);
            }
            let bucket = seen.entry(hasher.finish()).or_default();
            let equal = bucket.iter().any(|&(other_which, other_row)| {
                let (other_indices, other_values) = matrices[other_which].row(other_row);
                other_indices == indices
                    && other_values
                        .iter()
                        .zip(values)
                        .all(|(left, right)| left.to_bits() == right.to_bits())
            });
            if equal {
                return true;
            }
            bucket.push((which, row));
        }
    }
    false
}

/// The sum of two sparse rows (scipy's `e1 + e2` on `float32`), by increasing gene.
fn add_rows(first: (&[u32], &[f32]), second: (&[u32], &[f32])) -> Vec<(u32, f32)> {
    let (mut left, mut right) = (0, 0);
    let mut out = Vec::with_capacity(first.0.len() + second.0.len());
    while left < first.0.len() || right < second.0.len() {
        let left_gene = first.0.get(left).copied().unwrap_or(u32::MAX);
        let right_gene = second.0.get(right).copied().unwrap_or(u32::MAX);
        if left_gene == right_gene {
            out.push((left_gene, first.1[left] + second.1[right]));
            left += 1;
            right += 1;
        } else if left_gene < right_gene {
            out.push((left_gene, first.1[left]));
            left += 1;
        } else {
            out.push((right_gene, second.1[right]));
            right += 1;
        }
    }
    out
}

/// `n_sim` distinct ordered pairs of `0..n_obs` (as scanpy's `sample_comb` over the
/// `n_obs x n_obs` grid: a flat index `first x n_obs + second` drawn without
/// repeating), from SplitMix64 on `seed`.
fn draw_pairs(n_obs: usize, n_sim: usize, seed: u64) -> Result<Vec<[u32; 2]>, KytoError> {
    let grid = (n_obs as u128) * (n_obs as u128);
    check_count(n_obs, n_sim)?;
    let mut draws = Draws::new(seed, PAIRS_STREAM);
    let mut seen = HashSet::with_capacity(n_sim);
    let mut pairs = Vec::with_capacity(n_sim);
    while pairs.len() < n_sim {
        // Lemire's multiply-shift: uniform enough for 2^64 >> the grid, integers only.
        let flat = ((u128::from(draws.next_u64()) * grid) >> 64) as u64;
        if seen.insert(flat) {
            let n = n_obs as u64;
            pairs.push([(flat / n) as u32, (flat % n) as u32]);
        }
    }
    Ok(pairs)
}

/// Given pairs, checked: `n_sim` of them, each of scored cells.
fn check_pairs(pairs: &[[u32; 2]], n_obs: usize, n_sim: usize) -> Result<Vec<[u32; 2]>, KytoError> {
    check_count(n_obs, n_sim)?;
    if pairs.len() != n_sim
        || pairs
            .iter()
            .flatten()
            .any(|&parent| parent as usize >= n_obs)
    {
        return Err(invalid(
            "doublets",
            &format!("{} pairs", pairs.len()),
            &format!("the batch simulates {n_sim} doublets of {n_obs} scored cells"),
            &format!("{n_sim} pairs of positions below {n_obs}"),
        ));
    }
    Ok(pairs.to_vec())
}

fn check_count(n_obs: usize, n_sim: usize) -> Result<(), KytoError> {
    let grid = (n_obs as u128) * (n_obs as u128);
    if n_sim == 0 || n_sim as u128 > grid || u32::try_from(n_obs).is_err() {
        return Err(invalid(
            "sim_doublet_ratio",
            &format!("{n_sim} doublets of {n_obs} cells"),
            "Scrublet simulates at least one doublet and at most one per ordered pair of cells",
            "a ratio giving from 1 to n_cells^2 doublets",
        ));
    }
    Ok(())
}

/// Each gene's mean and `1 / std` (`ddof=1`) over the rows, as `fast_array_utils`'
/// `mean_var(correction=1)` on `float32` data: `f64` sums of the values and their
/// squares in row order, `var = (squares / n - mean^2) * (n / (n - 1))`.
fn zscore_statistics(matrix: &Csr) -> (Vec<f64>, Vec<f64>) {
    let n_rows = matrix.n_rows();
    let mut sums = vec![0.0_f64; matrix.n_genes];
    let mut squares = vec![0.0_f64; matrix.n_genes];
    for (&gene, &value) in matrix.indices.iter().zip(&matrix.values) {
        let value = f64::from(value);
        sums[gene as usize] += value;
        squares[gene as usize] += value * value;
    }
    let n = n_rows as f64;
    let means: Vec<f64> = sums.iter().map(|sum| sum / n).collect();
    let inverse_stds = squares
        .iter()
        .zip(&means)
        .map(|(squares, mean)| {
            let mut variance = squares / n - mean * mean;
            if n_rows != 1 {
                variance *= n / (n - 1.0);
            }
            1.0 / variance.sqrt()
        })
        .collect();
    (means, inverse_stds)
}

/// The rows as a dense row-major `f64` matrix of z-scores `(x - mean) * inverse_std`.
fn dense_zscore(matrix: &Csr, means: &[f64], inverse_stds: &[f64]) -> Mat<f64> {
    let mut dense = Mat::from_fn(matrix.n_rows(), matrix.n_genes, |_, gene| {
        (0.0 - means[gene]) * inverse_stds[gene]
    });
    for row in 0..matrix.n_rows() {
        let (indices, values) = matrix.row(row);
        for (&gene, &value) in indices.iter().zip(values) {
            let gene = gene as usize;
            dense[(row, gene)] = (f64::from(value) - means[gene]) * inverse_stds[gene];
        }
    }
    dense
}

/// The `f64` rows of the cells, then the doublets, in the first `n_components`
/// principal axes of the cells (exact: the eigenvectors of the centred cells' Gram
/// matrix), centred by the cells' means (scikit-learn's `PCA.transform`).
fn embed(
    observed: &Mat<f64>,
    simulated: &Mat<f64>,
    n_components: usize,
) -> Result<Vec<f64>, KytoError> {
    let (n_obs, n_genes) = (observed.nrows(), observed.ncols());
    if let Some(value) = observed
        .col_iter()
        .chain(simulated.col_iter())
        .flat_map(|column| column.iter().copied())
        .find(|value| !value.is_finite())
    {
        return Err(invalid(
            "layer",
            &value.to_string(),
            "a z-scored value is not finite (a highly variable gene with no variance, or a \
             count that is not finite)",
            "finite counts",
        ));
    }
    let mut means = vec![0.0; n_genes];
    for row in 0..n_obs {
        for (gene, mean) in means.iter_mut().enumerate() {
            *mean += observed[(row, gene)];
        }
    }
    for mean in &mut means {
        *mean /= n_obs as f64;
    }
    let centred = Mat::from_fn(n_obs, n_genes, |row, gene| {
        observed[(row, gene)] - means[gene]
    });
    let gram = centred.transpose() * &centred;
    let eigen = gram.self_adjoint_eigen(Side::Lower).map_err(|_| {
        invalid(
            "layer",
            "the z-scored highly variable genes",
            "their covariance eigendecomposition did not converge",
            "finite counts",
        )
    })?;
    // faer sorts eigenvalues increasing: the largest are last.
    let vectors = eigen.U();
    let mut axes = Mat::from_fn(n_genes, n_components, |gene, component| {
        vectors[(gene, n_genes - 1 - component)]
    });
    for component in 0..n_components {
        // svd_flip(u_based_decision=False): the largest absolute loading positive (the
        // first on ties).
        let mut largest = 0;
        for gene in 1..n_genes {
            if axes[(gene, component)].abs() > axes[(largest, component)].abs() {
                largest = gene;
            }
        }
        if axes[(largest, component)] < 0.0 {
            for gene in 0..n_genes {
                axes[(gene, component)] = -axes[(gene, component)];
            }
        }
    }
    let offsets: Vec<f64> = (0..n_components)
        .map(|component| {
            (0..n_genes)
                .map(|gene| means[gene] * axes[(gene, component)])
                .sum()
        })
        .collect();
    let mut embedding = Vec::with_capacity((n_obs + simulated.nrows()) * n_components);
    for matrix in [observed, simulated] {
        let projected = matrix * &axes;
        for row in 0..projected.nrows() {
            for (component, offset) in offsets.iter().enumerate() {
                embedding.push(projected[(row, component)] - offset);
            }
        }
    }
    Ok(embedding)
}

/// Per row of `embedding` (`width` values a row; the cells, then from `n_obs` on the
/// doublets), how many doublets are among its `others` nearest other rows: exact `f64`
/// Euclidean distances (each difference and square in `f64`, summed in component
/// order), ties by the earlier row. Rows run in parallel on the current rayon pool,
/// each on its own, so the counts do not depend on the threads (R11). `f64`, not task
/// 6.6's `float32` search: the classifier's neighbours on synthetic data have
/// near-ties (1e-8 relative) that `float32` rounding flips.
fn doublet_neighbors(
    embedding: &[f64],
    width: usize,
    n_obs: usize,
    others: usize,
    control: &RunControl,
) -> Result<Vec<usize>, KytoError> {
    use rayon::prelude::*;
    /// Rows per cancellation check.
    const CHUNK: usize = 64;
    let n_rows = embedding.len() / width.max(1);
    if others == 0 || width == 0 {
        return Ok(vec![0; n_rows]);
    }
    let counts: Vec<Vec<usize>> = (0..n_rows)
        .collect::<Vec<_>>()
        .par_chunks(CHUNK)
        .map(|rows| {
            control.check()?;
            let mut keys: Vec<(f64, usize)> = Vec::with_capacity(n_rows);
            Ok(rows
                .iter()
                .map(|&row| {
                    let query = &embedding[row * width..(row + 1) * width];
                    keys.clear();
                    keys.extend(
                        embedding
                            .chunks_exact(width)
                            .enumerate()
                            .filter(|&(other, _)| other != row)
                            .map(|(other, point)| {
                                let distance: f64 = query
                                    .iter()
                                    .zip(point)
                                    .map(|(left, right)| (left - right) * (left - right))
                                    .sum();
                                (distance, other)
                            }),
                    );
                    let order = |left: &(f64, usize), right: &(f64, usize)| {
                        left.0.total_cmp(&right.0).then(left.1.cmp(&right.1))
                    };
                    let taken = others.min(keys.len());
                    if taken < keys.len() {
                        keys.select_nth_unstable_by(taken - 1, order);
                    }
                    keys[..taken]
                        .iter()
                        .filter(|&&(_, other)| other >= n_obs)
                        .count()
                })
                .collect())
        })
        .collect::<Result<_, KytoError>>()?;
    Ok(counts.into_iter().flatten().collect())
}

/// Scrublet's Bayesian doublet score of each row from the doublets among its
/// neighbours, with scanpy's operations in its order.
fn doublet_scores(
    simulated_neighbors: &[usize],
    n_obs: usize,
    n_sim: usize,
    k_adj: f64,
    expected_doublet_rate: f64,
) -> Vec<f64> {
    let rho = expected_doublet_rate;
    let r = n_sim as f64 / n_obs as f64;
    simulated_neighbors
        .iter()
        .map(|&count| {
            let q = (count as f64 + 1.0) / (k_adj + 2.0);
            q * rho / r / (1.0 - rho - q * (1.0 - rho - rho / r))
        })
        .collect()
}

/// Python's `round` of a float: to the nearest integer, halves to even.
fn round_half_even(value: f64) -> f64 {
    let rounded = value.round();
    if (value - value.trunc()).abs() == 0.5 && rounded % 2.0 != 0.0 {
        rounded - value.signum()
    } else {
        rounded
    }
}

fn invalid(parameter: &str, value: &str, problem: &str, accepted: &str) -> KytoError {
    KytoError::InvalidArgument {
        function: "scrublet".to_owned(),
        parameter: parameter.to_owned(),
        value: value.to_owned(),
        problem: problem.to_owned(),
        accepted: accepted.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_is_pythons() {
        for (value, expected) in [
            (0.5, 0.0),
            (1.5, 2.0),
            (2.5, 2.0),
            (25.98, 26.0),
            (77.0, 77.0),
        ] {
            assert_eq!(round_half_even(value), expected, "{value}");
        }
    }

    #[test]
    fn rows_add_by_gene() {
        let sum = add_rows((&[0, 2, 5], &[1.0, 2.0, 3.0]), (&[2, 3], &[4.0, 5.0]));
        assert_eq!(sum, [(0, 1.0), (2, 6.0), (3, 5.0), (5, 3.0)]);
    }

    #[test]
    fn pairs_are_distinct_and_seeded() {
        let pairs = draw_pairs(5, 25, 7).unwrap();
        let distinct: HashSet<[u32; 2]> = pairs.iter().copied().collect();
        assert_eq!(distinct.len(), 25);
        assert_eq!(draw_pairs(5, 25, 7).unwrap(), pairs);
        assert_ne!(
            draw_pairs(50, 10, 8).unwrap(),
            draw_pairs(50, 10, 7).unwrap()
        );
        assert!(draw_pairs(5, 26, 7).is_err());
        assert!(draw_pairs(5, 0, 7).is_err());
    }

    #[test]
    fn scores_follow_scrublet() {
        // n_obs 2, n_sim 4: r = 2; 78 neighbours, 10 of them doublets.
        let scores = doublet_scores(&[10], 2, 4, 78.0, 0.05);
        let q: f64 = 11.0 / 80.0;
        let expected = q * 0.05 / 2.0 / (1.0 - 0.05 - q * (1.0 - 0.05 - 0.05 / 2.0));
        assert_eq!(scores, [expected]);
    }
}
