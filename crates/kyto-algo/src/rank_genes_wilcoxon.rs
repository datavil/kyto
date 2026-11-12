//! Wilcoxon rank-sum statistics for differential expression (task 6.10), as scanpy
//! 1.12.5's `rank_genes_groups(method="wilcoxon")` computes them.
//!
//! Does: collect each gene's stored non-zero values with the group slot of their cell
//! ([`RankEntries`], a range of genes at a time, at most a given number of values),
//! rank them among the compared cells with the zeros as one tied block (scipy's
//! `rankdata` average ranks: a block of ties at positions `s+1..=e` gets
//! `(s + e + 1) / 2`), sum the ranks per group, and turn the sums into scanpy's
//! normal-approximation z-scores (with `tie_correct`, scipy's `tiecorrect`).
//! Ranks are halves of integers, so the sums are exact in `f64` whatever the order.
//! Does not: compute p-values or fold changes (`rank_genes.rs`).

/// The slot of cells outside every compared group (scanpy's remainder, or no group).
pub(crate) const REMAINDER: u32 = u32::MAX;

/// Stored non-zero values of the genes `start..end`, each with its cell's slot.
#[derive(Debug)]
pub(crate) struct RankEntries {
    start: usize,
    end: usize,
    by_gene: Vec<Vec<(f64, u32)>>,
    held: usize,
    capacity: usize,
    overflowed: bool,
}

impl RankEntries {
    /// Collect genes `start..end`, at most `capacity` values (beyond that the
    /// collection stops and [`RankEntries::overflowed`] says so).
    pub(crate) fn new(start: usize, end: usize, capacity: usize) -> Self {
        Self {
            start,
            end,
            by_gene: vec![Vec::new(); end - start],
            held: 0,
            capacity,
            overflowed: false,
        }
    }

    /// Whether more values came than the capacity (then nothing is held).
    pub(crate) fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// Add one cell of slot `slot`: its genes and values (zeros skipped).
    pub(crate) fn add_cell(&mut self, slot: u32, entries: impl Iterator<Item = (usize, f64)>) {
        if self.overflowed {
            return;
        }
        for (gene, value) in entries {
            if value == 0.0 || gene < self.start || gene >= self.end {
                continue;
            }
            if self.held == self.capacity {
                self.overflowed = true;
                self.by_gene = Vec::new();
                return;
            }
            self.by_gene[gene - self.start].push((value, slot));
            self.held += 1;
        }
    }

    /// The genes collected and, for each, its values (gene `start + i` at `i`).
    pub(crate) fn into_genes(self) -> (usize, Vec<Vec<(f64, u32)>>) {
        (self.start, self.by_gene)
    }
}

/// The rank sum of each summed slot and the tie coefficient of one gene.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RankSums {
    /// By position (see [`rank_sums`]).
    pub(crate) sums: Vec<f64>,
    /// scipy's `tiecorrect`: `1 - sum(t^3 - t) / (n^3 - n)` over blocks of ties.
    pub(crate) tie: f64,
}

/// Rank the compared cells of one gene: `values` are their stored non-zero values with
/// slots (sorted here, `NaN` last as numpy sorts), `n_cells` how many cells are
/// compared (the others are zeros). The ranks of a cell of slot `s` are summed into
/// `sums[positions[s]]` (not summed when `positions[s]` is `None` or `s` is beyond
/// it); `counts[i]` is the number of cells summed into `sums[i]`.
pub(crate) fn rank_sums(
    values: &mut [(f64, u32)],
    n_cells: u64,
    positions: &[Option<usize>],
    counts: &[u64],
) -> RankSums {
    values.sort_by(|left, right| {
        (left.0.is_nan(), left.0)
            .partial_cmp(&(right.0.is_nan(), right.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n_stored = values.len() as u64;
    let zeros = n_cells - n_stored;
    let negatives = values.partition_point(|(value, _)| *value < 0.0) as u64;
    let mut sums = vec![0.0; counts.len()];
    let mut stored_in = vec![0_u64; counts.len()];
    let mut ties = 0.0;
    let cube = |count: u64| {
        let count = count as f64;
        count.powi(3) - count
    };
    let mut start = 0;
    while start < values.len() {
        let value = values[start].0;
        let mut end = start + 1;
        // `!=` as scipy's `arr[1:] != arr[:-1]`: each NaN is a block of its own.
        while end < values.len() && values[end].0 == value {
            end += 1;
        }
        let offset = if value < 0.0 { 0 } else { zeros };
        let rank = 0.5 * ((start as u64 + offset) as f64 + (end as u64 + offset) as f64 + 1.0);
        for (_, slot) in &values[start..end] {
            let position = usize::try_from(*slot)
                .ok()
                .and_then(|slot| positions.get(slot).copied().flatten());
            if let Some(position) = position {
                sums[position] += rank;
                stored_in[position] += 1;
            }
        }
        ties += cube((end - start) as u64);
        start = end;
    }
    let zero_rank = 0.5 * (negatives as f64 + (negatives + zeros) as f64 + 1.0);
    for position in 0..counts.len() {
        let zeros_in = counts[position] - stored_in[position];
        sums[position] += zeros_in as f64 * zero_rank;
    }
    ties += cube(zeros);
    let size = n_cells as f64;
    let tie = if n_cells >= 2 {
        1.0 - ties / (size.powi(3) - size)
    } else {
        1.0
    };
    RankSums { sums, tie }
}

/// scanpy's z-score of a group's rank sum against the rest of the cells: group of
/// `n` cells among `total` (`NaN` becomes 0). `tie` is `Some` with `tie_correct`.
pub(crate) fn z_versus_rest(rank_sum: f64, n: u64, total: u64, tie: Option<f64>) -> f64 {
    let expected = u128::from(n) * u128::from(total + 1);
    let expected = expected as f64 / 2.0;
    let variance = match tie {
        Some(tie) => tie * n as f64 * (total - n) as f64 * (total + 1) as f64 / 12.0,
        None => (u128::from(n) * u128::from(total - n) * u128::from(total + 1)) as f64 / 12.0,
    };
    finite_or_zero((rank_sum - expected) / variance.sqrt())
}

/// scanpy's z-score of a group of `n` cells against a reference group of `m` cells.
pub(crate) fn z_versus_reference(rank_sum: f64, n: u64, m: u64, tie: Option<f64>) -> f64 {
    let expected = n as f64 * ((n + m + 1) as f64 / 2.0);
    let variance = match tie {
        Some(tie) => tie * n as f64 * m as f64 * (n + m + 1) as f64 / 12.0,
        None => (u128::from(n) * u128::from(m) * u128::from(n + m + 1)) as f64 / 12.0,
    };
    finite_or_zero((rank_sum - expected) / variance.sqrt())
}

/// scanpy sets `NaN` scores to 0 (an infinite one stays).
fn finite_or_zero(score: f64) -> f64 {
    if score.is_nan() { 0.0 } else { score }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// scipy's `rankdata` on dense values: average ranks.
    fn dense_ranks(values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .map(|value| {
                let below = values.iter().filter(|other| *other < value).count() as f64;
                let equal = values.iter().filter(|other| *other == value).count() as f64;
                below + (equal + 1.0) / 2.0
            })
            .collect()
    }

    #[test]
    fn sparse_ranks_equal_dense_ranks() {
        // Slots 0 and 1, and a cell outside both (REMAINDER); negative values, ties,
        // and implicit zeros in every slot.
        let cells: [(f64, u32); 10] = [
            (0.0, 0),
            (2.0, 0),
            (-1.0, 1),
            (2.0, 1),
            (0.0, 1),
            (5.0, REMAINDER),
            (0.0, REMAINDER),
            (-1.0, 0),
            (3.0, 0),
            (0.0, 0),
        ];
        let dense: Vec<f64> = cells.iter().map(|cell| cell.0).collect();
        let ranks = dense_ranks(&dense);
        let mut expected = [0.0; 2];
        for (cell, rank) in cells.iter().zip(&ranks) {
            if cell.1 != REMAINDER {
                expected[cell.1 as usize] += rank;
            }
        }
        let mut stored: Vec<(f64, u32)> =
            cells.iter().copied().filter(|cell| cell.0 != 0.0).collect();
        let result = rank_sums(&mut stored, 10, &[Some(0), Some(1)], &[5, 3]);
        assert_eq!(result.sums, expected);
        // Ties: zeros 4, -1 twice, 2 twice: 1 - (60 + 6 + 6) / 990.
        assert!((result.tie - (1.0 - 72.0 / 990.0)).abs() < 1e-15);
    }

    #[test]
    fn z_scores_follow_scanpy() {
        // Group of 3 among 10 with rank sum 20: (20 - 16.5) / sqrt(3 * 7 * 11 / 12).
        let z = z_versus_rest(20.0, 3, 10, None);
        assert!((z - 3.5 / (231.0_f64 / 12.0).sqrt()).abs() < 1e-15);
        assert_eq!(z_versus_rest(1.0, 1, 1, None), 0.0, "NaN becomes 0");
        let z = z_versus_reference(20.0, 3, 4, Some(0.5));
        assert!((z - (20.0 - 12.0) / (0.5_f64 * 3.0 * 4.0 * 8.0 / 12.0).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn collection_stops_at_capacity() {
        let mut entries = RankEntries::new(1, 3, 2);
        entries.add_cell(0, [(0, 1.0), (1, 2.0), (2, 0.0)].into_iter());
        entries.add_cell(1, [(1, 1.0)].into_iter());
        assert!(!entries.overflowed());
        entries.add_cell(1, [(2, 1.0)].into_iter());
        assert!(entries.overflowed());
    }
}
