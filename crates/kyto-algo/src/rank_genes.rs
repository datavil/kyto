//! Differential expression per group of cells (task 6.10): scanpy 1.12.5's
//! `rank_genes_groups` with `method="t-test"` (Welch) or `"wilcoxon"`, streamed,
//! returning one Polars frame.
//!
//! Does: run a plan's output (the tested layer and the `groupby` cell column) through
//! the executor and fold it in cell order on the calling thread: per group gene
//! moments ([`GroupMoments`]) in a first pass, which is all the t-test needs; Wilcoxon
//! also ranks each gene's values among the compared cells ([`RankEntries`]): in that
//! same pass when all stored non-zero values fit in `max_rank_values`, otherwise in
//! one more pass per range of genes that fits. Then each compared group's statistic,
//! log2 fold change (from the group and rest means, `expm1`, `+1e-9`), p-value,
//! adjusted p-value (over all genes), genes ordered by statistic (descending), as
//! `sc.get.rank_genes_groups_df`.
//! Groups: the `groupby` values as text, in scanpy's category order (an `Enum`'s
//! categories, numbers by value, text sorted) or as listed in
//! `groups` (with the reference added); a cell with a null value is in no group but,
//! against `"rest"`, still in every group's rest (scanpy's remainder).
//! Does not: build a plan node (the result is a frame, not a member of the atlas),
//! estimate its memory, or provide scanpy's `logreg`, `t-test_overestim_var`, `pts`,
//! `mask_var` or `use_raw` (deviations.md).

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use kyto_error::KytoError;
use kyto_exec::{CellValues, ExecuteOptions, Pipeline, PipelineBlock};
use kyto_plan::{Member, OptimizeOptions, Output, PhysicalOptions, Plan, physical_plan};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};

use crate::distributions::{benjamini_hochberg, bonferroni, normal_two_sided};
use crate::group_labels::{GroupColumn, LabelCodes};
use crate::rank_genes_moments::{GroupMoments, GroupStatistics, MomentKind, welch};
use crate::rank_genes_wilcoxon::{
    REMAINDER, RankEntries, rank_sums, z_versus_reference, z_versus_rest,
};

/// The default of [`RankGenesGroups::max_rank_values`]: 2^25 values (512 MiB of
/// value and slot pairs) collected at once for Wilcoxon's ranks.
pub const DEFAULT_MAX_RANK_VALUES: NonZeroUsize = NonZeroUsize::new(1 << 25).unwrap();

/// The columns of [`rank_genes_groups`]' frame, in order.
pub const RANK_GENES_COLUMNS: [&str; 6] = [
    "group",
    "gene_id",
    "statistic",
    "log_fold_change",
    "p_value",
    "adjusted_p_value",
];

/// The test (scanpy's `method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentialTest {
    /// Welch's t-test (`"t-test"`).
    TTest,
    /// Wilcoxon's rank-sum test, normal approximation (`"wilcoxon"`).
    Wilcoxon,
}

impl DifferentialTest {
    /// The name scanpy uses.
    pub fn name(self) -> &'static str {
        match self {
            Self::TTest => "t-test",
            Self::Wilcoxon => "wilcoxon",
        }
    }
}

/// How p-values are adjusted for testing every gene (scanpy's `corr_method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PValueCorrection {
    /// Benjamini-Hochberg false discovery rate (`"benjamini-hochberg"`).
    BenjaminiHochberg,
    /// Bonferroni (`"bonferroni"`).
    Bonferroni,
}

/// The arguments of [`rank_genes_groups`] (scanpy's names, but `max_rank_values`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankGenesGroups {
    /// The cell column whose values are the groups.
    pub groupby: String,
    /// The test.
    pub method: DifferentialTest,
    /// The layer tested; `None` takes the plan's default layer.
    pub layer: Option<String>,
    /// The groups tested, in this order; `None` tests every group (sorted). Against
    /// `"rest"`, the rest is still every other cell.
    pub groups: Option<Vec<String>>,
    /// The group every other is compared with; `None` compares with the rest.
    pub reference: Option<String>,
    /// Keep the first this many genes of each group; `None` keeps every gene.
    pub n_genes: Option<NonZeroUsize>,
    /// Order genes by the absolute statistic.
    pub rankby_abs: bool,
    /// Wilcoxon only: correct the variance for ties.
    pub tie_correct: bool,
    /// The p-value adjustment.
    pub correction: PValueCorrection,
    /// Wilcoxon only: at most this many stored values are held at once for ranking;
    /// more take one more pass over the plan per range of genes that fits.
    pub max_rank_values: NonZeroUsize,
}

impl RankGenesGroups {
    /// Test every group of `groupby` against the rest with `method`, scanpy's
    /// defaults otherwise.
    pub fn new(groupby: impl Into<String>, method: DifferentialTest) -> Self {
        Self {
            groupby: groupby.into(),
            method,
            layer: None,
            groups: None,
            reference: None,
            n_genes: None,
            rankby_abs: false,
            tie_correct: false,
            correction: PValueCorrection::BenjaminiHochberg,
            max_rank_values: DEFAULT_MAX_RANK_VALUES,
        }
    }
}

const FUNCTION: &str = "rank_genes_groups";

fn invalid(parameter: &str, value: &str, problem: &str, accepted: &str) -> KytoError {
    KytoError::InvalidArgument {
        function: FUNCTION.to_owned(),
        parameter: parameter.to_owned(),
        value: value.to_owned(),
        problem: problem.to_owned(),
        accepted: accepted.to_owned(),
    }
}

/// Rank the genes of each group of `parameters.groupby` in `plan`'s output by a
/// differential expression test against the rest (or a reference group), as scanpy's
/// `rank_genes_groups` then `rank_genes_groups_df`: one row per (group, gene), groups
/// in order, each group's genes by statistic, descending (ties in gene order); columns
/// [`RANK_GENES_COLUMNS`] (`String`, `String`, then `Float64`). The plan runs with
/// `physical` and `execute` once, or more for Wilcoxon over many values.
///
/// # Errors
/// Before reading data: `KY-E211` without a layer and a default layer; `KY-E109` for a
/// layer or `groupby` column the plan does not output; `KY-E215` for an empty or
/// repeated `groups` list. While running: any error of the run; `KY-E214` for a group
/// in `groups` or a `reference` no cell has; `KY-E215` when no cell has a group or a
/// tested group has fewer than two cells (scanpy's "only contain one sample").
pub fn rank_genes_groups(
    plan: &Arc<Plan>,
    parameters: &RankGenesGroups,
    physical: PhysicalOptions,
    execute: &ExecuteOptions,
) -> Result<DataFrame, KytoError> {
    let layer =
        match &parameters.layer {
            Some(layer) => layer.clone(),
            None => plan.default_layer().map(str::to_owned).ok_or_else(|| {
                KytoError::NoDefaultLayer {
                    expression: format!("{FUNCTION}(layer=None)"),
                    layers: plan
                        .schema()
                        .layers
                        .iter()
                        .map(|layer| layer.name.clone())
                        .collect(),
                }
            })?,
        };
    let selection = Selection::new(parameters)?;
    let output = Output::members(
        plan,
        [
            Member::Layer(layer.clone()),
            Member::CellColumn(parameters.groupby.clone()),
        ],
    )?;
    let physical = physical_plan(
        std::slice::from_ref(&output),
        OptimizeOptions::ALL,
        physical,
    );
    let store = plan.root().source().to_owned();
    let mut run = Run {
        pipeline: Pipeline::new(&physical, 0)?,
        plan_gene_ids: plan.genes().to_vec(),
        execute,
        layer,
        store: store.clone(),
        column: GroupColumn::new(&parameters.groupby, &store),
        selection,
        codes: LabelCodes::default(),
        buffer: Vec::new(),
    };
    let mut first = run.first_pass(parameters)?;
    let groups = run.groups(&first, parameters)?;
    let n_genes = first.gene_ids.len();
    let n_cells = first.n_cells;
    let compared: Vec<usize> = (0..groups.order.len())
        .filter(|group| Some(*group) != groups.reference)
        .collect();
    let entries = first.entries.take();
    let statistics = {
        let mut moments: Vec<&GroupMoments> = groups
            .order
            .iter()
            .map(|slot| &first.slots[*slot])
            .collect();
        moments.push(&first.remainder);
        GroupStatistics::new(&moments)
    };
    let tests = match parameters.method {
        DifferentialTest::TTest => t_test(&statistics, &compared, groups.reference, n_cells),
        DifferentialTest::Wilcoxon => {
            let table = run.rank_sums(entries, &first, &groups, parameters)?;
            wilcoxon(&table, &statistics, &compared, &groups, n_cells, parameters)
        }
    };
    let mut frame = FrameColumns::default();
    for (group, (statistic, p_values)) in compared.iter().zip(tests) {
        let rest_means = match groups.reference {
            Some(reference) => statistics.means[reference].clone(),
            None => statistics.rest_means(*group, n_cells),
        };
        let fold_changes: Vec<f64> = statistics.means[*group]
            .iter()
            .zip(&rest_means)
            .map(|(group, rest)| ((group.exp_m1() + 1e-9) / (rest.exp_m1() + 1e-9)).log2())
            .collect();
        let adjusted = match parameters.correction {
            PValueCorrection::BenjaminiHochberg => benjamini_hochberg(&p_values),
            PValueCorrection::Bonferroni => bonferroni(&p_values, n_genes),
        };
        let key = |gene: usize| {
            if parameters.rankby_abs {
                statistic[gene].abs()
            } else {
                statistic[gene]
            }
        };
        let mut genes: Vec<usize> = (0..n_genes).collect();
        genes.sort_by(|left, right| key(*right).total_cmp(&key(*left)));
        let kept = parameters.n_genes.map_or(n_genes, |n| n.get().min(n_genes));
        let label = &groups.labels[*group];
        for gene in genes.into_iter().take(kept) {
            frame.group.push(label.clone());
            frame.gene_id.push(first.gene_ids[gene].clone());
            frame.statistic.push(statistic[gene]);
            frame.log_fold_change.push(fold_changes[gene]);
            frame.p_value.push(p_values[gene]);
            frame.adjusted_p_value.push(adjusted[gene]);
        }
    }
    frame.finish()
}

/// Which groups are tested, known before the data: the listed groups (with the
/// reference added), or every group.
#[derive(Debug)]
struct Selection {
    /// `groups` with the reference added, in order; `None` for every group.
    listed: Option<Vec<String>>,
}

impl Selection {
    fn new(parameters: &RankGenesGroups) -> Result<Self, KytoError> {
        let Some(groups) = &parameters.groups else {
            return Ok(Self { listed: None });
        };
        let value = format!("{groups:?}");
        if groups.is_empty() {
            return Err(invalid(
                "groups",
                &value,
                "the list is empty",
                "None (every group) or a list of group labels",
            ));
        }
        let mut seen = HashSet::new();
        if let Some(repeated) = groups.iter().find(|group| !seen.insert(*group)) {
            return Err(invalid(
                "groups",
                &value,
                &format!("'{repeated}' is listed twice"),
                "a list of distinct group labels",
            ));
        }
        let mut listed = groups.clone();
        if let Some(reference) = &parameters.reference
            && !listed.contains(reference)
        {
            listed.push(reference.clone());
        }
        Ok(Self {
            listed: Some(listed),
        })
    }
}

/// What the first pass found.
struct FirstPass {
    gene_ids: Vec<String>,
    n_cells: u64,
    /// Moments by slot (a listed group's position, or a label's code).
    slots: Vec<GroupMoments>,
    /// Moments of the cells in no slot.
    remainder: GroupMoments,
    /// Wilcoxon's values, when they fitted.
    entries: Option<RankEntries>,
}

impl FirstPass {
    /// Stored non-zero values per gene over every cell.
    fn nonzero(&self) -> Vec<u64> {
        let mut nonzero = self.remainder.nonzero().to_vec();
        for slot in &self.slots {
            for (total, count) in nonzero.iter_mut().zip(slot.nonzero()) {
                *total += count;
            }
        }
        nonzero
    }
}

/// The tested groups in scanpy's order.
struct Groups {
    /// The slot of each group.
    order: Vec<usize>,
    /// The label of each group.
    labels: Vec<String>,
    /// The reference's position in `order`.
    reference: Option<usize>,
}

/// One call's run: the pipeline, and how labels become slots.
struct Run<'a> {
    pipeline: Pipeline,
    plan_gene_ids: Vec<String>,
    execute: &'a ExecuteOptions,
    layer: String,
    store: String,
    column: GroupColumn,
    selection: Selection,
    codes: LabelCodes,
    buffer: Vec<(usize, f64)>,
}

impl Run<'_> {
    /// The slot of a cell with `label`, or [`REMAINDER`].
    fn slot(&mut self, label: Option<&str>) -> u32 {
        let Some(label) = label else {
            return REMAINDER;
        };
        match &self.selection.listed {
            Some(listed) => listed
                .iter()
                .position(|group| group == label)
                .and_then(|position| u32::try_from(position).ok())
                .unwrap_or(REMAINDER),
            None => self.codes.code(label),
        }
    }

    fn corrupt(&self, problem: String) -> KytoError {
        KytoError::CorruptTable {
            path: self.store.clone(),
            problem,
        }
    }

    /// Stream the plan once: the gene IDs, then every cell's slot and stored values
    /// handed to `fold` (in cell order), and the number of cells.
    fn pass(
        &mut self,
        mut fold: impl FnMut(&[String], u32, &[(usize, f64)]),
    ) -> Result<(Vec<String>, u64), KytoError> {
        let mut stream = self.pipeline.run(self.execute.clone())?;
        let gene_ids: Vec<String> = match stream.genes() {
            Some(genes) => genes
                .table()
                .column("gene_id")
                .and_then(|column| column.str().cloned())
                .map(|ids| {
                    ids.iter()
                        .map(|id| id.unwrap_or_default().to_owned())
                        .collect()
                })
                .map_err(|error| self.corrupt(format!("genes.parquet: {error}")))?,
            // No gene steps: the plan's genes are exactly the stream's.
            None => self.plan_gene_ids.clone(),
        };
        let mut n_cells = 0;
        while let Some(block) = stream.next() {
            let block = block?;
            n_cells += self.block(&block, &gene_ids, &mut fold)?;
            stream.recycle(block);
        }
        Ok((gene_ids, n_cells))
    }

    fn block(
        &mut self,
        block: &PipelineBlock,
        gene_ids: &[String],
        fold: &mut impl FnMut(&[String], u32, &[(usize, f64)]),
    ) -> Result<u64, KytoError> {
        let labels = self.column.labels(block)?;
        let Some(layer) = block.layers.get(&self.layer) else {
            return Err(self.corrupt(format!("the stream does not carry layer '{}'", self.layer)));
        };
        if layer.n_genes() != gene_ids.len() as u64 {
            return Err(self.corrupt(format!(
                "a block of layer '{}' has {} genes, the stream's gene table {}",
                self.layer,
                layer.n_genes(),
                gene_ids.len()
            )));
        }
        let indptr = layer.indptr();
        let indices = layer.indices();
        let mut buffer = std::mem::take(&mut self.buffer);
        for (row, label) in labels.iter().enumerate() {
            let slot = self.slot(label.as_deref());
            let range = indptr[row] as usize..indptr[row + 1] as usize;
            buffer.clear();
            let genes = indices[range.clone()].iter().map(|gene| *gene as usize);
            match layer.data() {
                CellValues::F32(values) => {
                    buffer.extend(genes.zip(values[range].iter().map(|value| f64::from(*value))));
                }
                CellValues::U32(values) => {
                    buffer.extend(genes.zip(values[range].iter().map(|value| f64::from(*value))));
                }
            }
            fold(gene_ids, slot, &buffer);
        }
        self.buffer = buffer;
        Ok(labels.len() as u64)
    }
}
impl Run<'_> {
    /// The first pass: every slot's moments and, for Wilcoxon, the values to rank if
    /// they fit.
    fn first_pass(&mut self, parameters: &RankGenesGroups) -> Result<FirstPass, KytoError> {
        let kind = match parameters.method {
            DifferentialTest::TTest => MomentKind::Welford,
            DifferentialTest::Wilcoxon => MomentKind::Sums,
        };
        let wilcoxon = parameters.method == DifferentialTest::Wilcoxon;
        let capacity = parameters.max_rank_values.get();
        let mut slots: Vec<GroupMoments> = Vec::new();
        let mut remainder: Option<GroupMoments> = None;
        let mut entries: Option<RankEntries> = None;
        let (gene_ids, n_cells) = self.pass(|gene_ids, slot, values| {
            let n_genes = gene_ids.len();
            let remainder = remainder.get_or_insert_with(|| {
                if wilcoxon {
                    entries = Some(RankEntries::new(0, n_genes, capacity));
                }
                GroupMoments::new(kind, n_genes)
            });
            let moments = if slot == REMAINDER {
                remainder
            } else {
                let slot = slot as usize;
                while slots.len() <= slot {
                    slots.push(GroupMoments::new(kind, n_genes));
                }
                &mut slots[slot]
            };
            moments.add_cell(values.iter().copied());
            if let Some(entries) = &mut entries {
                entries.add_cell(slot, values.iter().copied());
            }
        })?;
        let n_genes = gene_ids.len();
        let listed = self.selection.listed.as_ref().map_or(0, Vec::len);
        while slots.len() < listed {
            slots.push(GroupMoments::new(kind, n_genes));
        }
        if wilcoxon && entries.is_none() {
            entries = Some(RankEntries::new(0, n_genes, capacity));
        }
        Ok(FirstPass {
            gene_ids,
            n_cells,
            slots,
            remainder: remainder.unwrap_or_else(|| GroupMoments::new(kind, n_genes)),
            entries: entries.filter(|entries| !entries.overflowed()),
        })
    }

    /// The tested groups, checked against what the cells hold.
    fn groups(&self, first: &FirstPass, parameters: &RankGenesGroups) -> Result<Groups, KytoError> {
        let seen: Vec<String> = match &self.selection.listed {
            None => self.codes.labels().to_vec(),
            Some(listed) => listed
                .iter()
                .zip(&first.slots)
                .filter(|(_, moments)| moments.n_cells() > 0)
                .map(|(label, _)| label.clone())
                .collect(),
        };
        let all_labels = || {
            let mut labels = match &self.selection.listed {
                None => self.codes.labels().to_vec(),
                Some(_) => seen.clone(),
            };
            labels.sort_by(|left, right| self.column.compare(left, right));
            labels
        };
        let (order, labels): (Vec<usize>, Vec<String>) = match &self.selection.listed {
            None => {
                if seen.is_empty() {
                    return Err(invalid(
                        "groupby",
                        &format!("'{}'", parameters.groupby),
                        "no cell has a group (every value is null, or there are no cells)",
                        "a cell column with a value for at least two cells of a group",
                    ));
                }
                let mut order: Vec<usize> = (0..seen.len()).collect();
                order.sort_by(|left, right| self.column.compare(&seen[*left], &seen[*right]));
                let labels = order.iter().map(|slot| seen[*slot].clone()).collect();
                (order, labels)
            }
            Some(listed) => {
                if let Some((position, missing)) = listed
                    .iter()
                    .enumerate()
                    .find(|(position, _)| first.slots[*position].n_cells() == 0)
                {
                    let parameter = if Some(missing) == parameters.reference.as_ref()
                        && parameters
                            .groups
                            .as_ref()
                            .is_some_and(|groups| position >= groups.len())
                    {
                        "reference"
                    } else {
                        "groups"
                    };
                    return Err(KytoError::InvalidChoice {
                        function: FUNCTION.to_owned(),
                        parameter: parameter.to_owned(),
                        value: format!("'{missing}'"),
                        accepted: all_labels(),
                    });
                }
                ((0..listed.len()).collect(), listed.clone())
            }
        };
        let reference = match &parameters.reference {
            None => None,
            Some(reference) => match labels.iter().position(|label| label == reference) {
                Some(position) => Some(position),
                None => {
                    return Err(KytoError::InvalidChoice {
                        function: FUNCTION.to_owned(),
                        parameter: "reference".to_owned(),
                        value: format!("'{reference}'"),
                        accepted: all_labels(),
                    });
                }
            },
        };
        let small: Vec<&str> = order
            .iter()
            .zip(&labels)
            .filter(|(slot, _)| first.slots[**slot].n_cells() < 2)
            .map(|(_, label)| label.as_str())
            .collect();
        if !small.is_empty() {
            return Err(invalid(
                "groupby",
                &format!("'{}'", parameters.groupby),
                &format!(
                    "group(s) {} have only one cell, too few for a test",
                    small
                        .iter()
                        .map(|label| format!("'{label}'"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                "groups of at least two cells; leave the others out with groups=[...] \
                 or filter their cells first",
            ));
        }
        Ok(Groups {
            order,
            labels,
            reference,
        })
    }

    /// Wilcoxon's rank sums `[group][gene]` of every compared group (`compared` order
    /// as in [`rank_genes_groups`]: groups but the reference), and each gene's tie
    /// coefficient per compared group: `(sums, ties)`.
    fn rank_sums(
        &mut self,
        entries: Option<RankEntries>,
        first: &FirstPass,
        groups: &Groups,
        parameters: &RankGenesGroups,
    ) -> Result<RankSumTable, KytoError> {
        let n_genes = first.gene_ids.len();
        let compared: Vec<usize> = (0..groups.order.len())
            .filter(|group| Some(*group) != groups.reference)
            .collect();
        let mut table = RankSumTable {
            sums: vec![vec![0.0; n_genes]; compared.len()],
            ties: vec![vec![1.0; n_genes]; compared.len()],
        };
        let counts: Vec<u64> = groups
            .order
            .iter()
            .map(|slot| first.slots[*slot].n_cells())
            .collect();
        // The position in `compared` of each slot's group (against the rest).
        let mut positions = vec![None; first.slots.len()];
        for (position, group) in compared.iter().enumerate() {
            positions[groups.order[*group]] = Some(position);
        }
        let compared_counts: Vec<u64> = compared.iter().map(|group| counts[*group]).collect();
        // Against a reference: only the compared group's ranks are summed.
        let alone: Vec<Vec<Option<usize>>> = compared
            .iter()
            .map(|group| {
                let mut alone = vec![None; first.slots.len()];
                alone[groups.order[*group]] = Some(0);
                alone
            })
            .collect();
        let mut rank = |start: usize, genes: Vec<Vec<(f64, u32)>>| {
            let mut scratch = Vec::new();
            for (offset, mut values) in genes.into_iter().enumerate() {
                let gene = start + offset;
                match groups.reference {
                    None => {
                        let result =
                            rank_sums(&mut values, first.n_cells, &positions, &compared_counts);
                        for (position, sum) in result.sums.iter().enumerate() {
                            table.sums[position][gene] = *sum;
                            table.ties[position][gene] = result.tie;
                        }
                    }
                    Some(reference) => {
                        let reference_slot = groups.order[reference] as u32;
                        for (position, group) in compared.iter().enumerate() {
                            let slot = groups.order[*group] as u32;
                            scratch.clear();
                            scratch.extend(
                                values
                                    .iter()
                                    .copied()
                                    .filter(|(_, cell)| *cell == slot || *cell == reference_slot),
                            );
                            let result = rank_sums(
                                &mut scratch,
                                counts[*group] + counts[reference],
                                &alone[position],
                                &[counts[*group]],
                            );
                            table.sums[position][gene] = result.sums[0];
                            table.ties[position][gene] = result.tie;
                        }
                    }
                }
            }
        };
        if let Some(entries) = entries {
            let (start, genes) = entries.into_genes();
            rank(start, genes);
            return Ok(table);
        }
        // Too many values at once: one pass per range of genes that fits.
        let capacity = parameters.max_rank_values.get();
        let nonzero = first.nonzero();
        let mut start = 0;
        while start < n_genes {
            let mut end = start;
            let mut held = 0_u64;
            while end < n_genes && (end == start || held + nonzero[end] <= capacity as u64) {
                held += nonzero[end];
                end += 1;
            }
            let mut entries = RankEntries::new(
                start,
                end,
                capacity.max(usize::try_from(held).unwrap_or(usize::MAX)),
            );
            self.pass(|_, slot, values| entries.add_cell(slot, values.iter().copied()))?;
            if entries.overflowed() {
                return Err(
                    self.corrupt("a later pass streamed more values than the first".to_owned())
                );
            }
            let (start_gene, genes) = entries.into_genes();
            rank(start_gene, genes);
            start = end;
        }
        Ok(table)
    }
}

/// Wilcoxon's rank sums and tie coefficients, `[compared group][gene]`.
struct RankSumTable {
    sums: Vec<Vec<f64>>,
    ties: Vec<Vec<f64>>,
}

/// Welch's t statistics and p-values of each compared group, `(statistic, p)` per
/// gene, as scanpy's `_RankGenes.t_test`.
fn t_test(
    statistics: &GroupStatistics,
    compared: &[usize],
    reference: Option<usize>,
    n_cells: u64,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let Some(variances) = &statistics.variances else {
        unreachable!("the t-test's moments have variances")
    };
    let rest_variances = reference.is_none().then(|| statistics.rest_variances());
    compared
        .iter()
        .map(|group| {
            let n_group = statistics.counts[*group] as f64;
            let (rest_means, rest_variances, n_rest) = match reference {
                Some(reference) => (
                    statistics.means[reference].clone(),
                    variances[reference].clone(),
                    statistics.counts[reference] as f64,
                ),
                None => (
                    statistics.rest_means(*group, n_cells),
                    rest_variances
                        .as_ref()
                        .map_or_else(Vec::new, |rest| rest[*group].clone()),
                    (n_cells - statistics.counts[*group]) as f64,
                ),
            };
            statistics.means[*group]
                .iter()
                .zip(&variances[*group])
                .zip(rest_means.iter().zip(&rest_variances))
                .map(|((mean, variance), (rest_mean, rest_variance))| {
                    welch(
                        (*mean, *variance, n_group),
                        (*rest_mean, *rest_variance, n_rest),
                    )
                })
                .unzip()
        })
        .collect()
}

/// Wilcoxon's z-scores and p-values of each compared group, as scanpy's
/// `_RankGenes.wilcoxon`.
fn wilcoxon(
    table: &RankSumTable,
    statistics: &GroupStatistics,
    compared: &[usize],
    groups: &Groups,
    n_cells: u64,
    parameters: &RankGenesGroups,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    compared
        .iter()
        .enumerate()
        .map(|(position, group)| {
            let n = statistics.counts[*group];
            table.sums[position]
                .iter()
                .zip(&table.ties[position])
                .map(|(sum, tie)| {
                    let tie = parameters.tie_correct.then_some(*tie);
                    let z = match groups.reference {
                        None => z_versus_rest(*sum, n, n_cells, tie),
                        Some(reference) => {
                            z_versus_reference(*sum, n, statistics.counts[reference], tie)
                        }
                    };
                    (z, normal_two_sided(z))
                })
                .unzip()
        })
        .collect()
}

/// The frame's columns as they fill.
#[derive(Debug, Default)]
struct FrameColumns {
    group: Vec<String>,
    gene_id: Vec<String>,
    statistic: Vec<f64>,
    log_fold_change: Vec<f64>,
    p_value: Vec<f64>,
    adjusted_p_value: Vec<f64>,
}

impl FrameColumns {
    fn finish(self) -> Result<DataFrame, KytoError> {
        let height = self.group.len();
        let [
            group,
            gene_id,
            statistic,
            log_fold_change,
            p_value,
            adjusted_p_value,
        ] = RANK_GENES_COLUMNS;
        DataFrame::new(
            height,
            vec![
                Series::new(group.into(), self.group).into_column(),
                Series::new(gene_id.into(), self.gene_id).into_column(),
                Series::new(statistic.into(), self.statistic).into_column(),
                Series::new(log_fold_change.into(), self.log_fold_change).into_column(),
                Series::new(p_value.into(), self.p_value).into_column(),
                Series::new(adjusted_p_value.into(), self.adjusted_p_value).into_column(),
            ],
        )
        .map_err(|error| KytoError::CorruptTable {
            path: FUNCTION.to_owned(),
            problem: error.to_string(),
        })
    }
}
