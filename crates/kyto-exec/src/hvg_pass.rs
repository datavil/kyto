//! The statistics passes of `highly_variable_genes` (tasks 5.4 and 6.3): what a leaf
//! of a pass sums, and what a pass leaves for the next.
//!
//! Does: sum, per group of cells (every cell, or each batch: the `batch_key` cell
//! column's value as text, read from the block's computed columns or the stored cell
//! table at its rows) and per gene, the values (of `expm1(value)` for `seurat`) and
//! their squares as scanpy's numba kernel does on `float32` data (the value and its
//! square in `float32`, summed in `f64`), the cells with a positive value, and for
//! `seurat_v3` with batches Welford's moments of the stored values (scanpy's
//! `aggregate`). `seurat_v3` checks its values are counts (non-negative integers), fits
//! each group's loess after the first pass (`hvg.rs`) and sums `min(value, clip)` in
//! the second. Leaves combine in a fixed tree in block order (`reduction.rs`, R11).
//! Does not: rank or flag genes (see `hvg.rs`).

use std::collections::BTreeMap;
use std::sync::Arc;

use kyto_error::KytoError;
use kyto_plan::HvgFlavor;
use kyto_store::{CellBlock, CellValues};
use polars::prelude::{DataFrame, DataType, IdxCa, IdxSize};

use crate::hvg::{SeuratV3Fit, seurat_v3_fit};
use crate::kernels::KernelValue;
use crate::pipeline_block::PipelineBlock;
use crate::statistics::{
    BatchStatistics, ClippedMoments, GeneMoments, HighlyVariableGeneStatistics,
};

/// The group of every cell when there is no `batch_key`.
const ALL_CELLS: &str = "";

/// The `batch_key` column and the stored cell table to read it from.
#[derive(Debug, Clone)]
pub(crate) struct BatchColumn {
    pub(crate) name: String,
    pub(crate) table: Arc<DataFrame>,
    pub(crate) store: String,
}

/// One leaf of a pass (or several combined): moments per group, by label.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GroupPartial {
    groups: BTreeMap<String, GeneMoments>,
    /// The first value that is not a count, for `seurat_v3`'s check.
    non_count: Option<f64>,
}

impl GroupPartial {
    /// `self` (the cells before) and `right` (the cells after) as one partial.
    pub(crate) fn combine(mut self, right: Self) -> Self {
        for (label, moments) in right.groups {
            match self.groups.get_mut(&label) {
                Some(left) => left.combine(&moments),
                None => {
                    self.groups.insert(label, moments);
                }
            }
        }
        self.non_count = self.non_count.or(right.non_count);
        self
    }
}

/// What `highly_variable_genes`' passes have found so far.
#[derive(Debug, Clone)]
pub(crate) struct HvgAccumulator {
    layer: String,
    flavor: HvgFlavor,
    n_genes: usize,
    batch: Option<BatchColumn>,
    /// The first pass's moments per group.
    first: BTreeMap<String, GeneMoments>,
    /// `seurat_v3`'s fit per group, after the first pass.
    fits: BTreeMap<String, SeuratV3Fit>,
    /// `seurat_v3`'s second pass's moments per group.
    second: BTreeMap<String, GeneMoments>,
}

impl HvgAccumulator {
    pub(crate) fn new(
        layer: &str,
        flavor: HvgFlavor,
        n_genes: usize,
        batch: Option<BatchColumn>,
    ) -> Self {
        Self {
            layer: layer.to_owned(),
            flavor,
            n_genes,
            batch,
            first: BTreeMap::new(),
            fits: BTreeMap::new(),
            second: BTreeMap::new(),
        }
    }

    /// An empty leaf.
    pub(crate) fn leaf(&self) -> GroupPartial {
        GroupPartial {
            groups: BTreeMap::new(),
            non_count: None,
        }
    }

    /// Whether pass `pass` keeps Welford's moments: `seurat_v3`'s first pass with
    /// batches. They depend on the order of the cells, so the pass is folded as one
    /// leaf in cell order, as scanpy's sequential loop (bit for bit: the loess below is
    /// sensitive to a last-bit difference between tied means).
    pub(crate) fn sequential(&self, pass: u32) -> bool {
        pass == 0 && self.flavor == HvgFlavor::SeuratV3 && self.batch.is_some()
    }

    fn empty_moments(&self, pass: u32) -> GeneMoments {
        GeneMoments::new(self.n_genes, self.sequential(pass))
    }

    /// Add the rows of `layer` (the input layer of `block`) to `leaf`, a leaf of pass
    /// `pass`.
    ///
    /// # Errors
    /// `KY-E117` if the block's genes are not the accumulator's; `KY-E215` for a null
    /// batch; `KY-E113` if the batch column cannot be read as text.
    pub(crate) fn fold(
        &self,
        pass: u32,
        leaf: &mut GroupPartial,
        layer: &CellBlock,
        block: &PipelineBlock,
    ) -> Result<(), KytoError> {
        if layer.n_genes() as usize != self.n_genes {
            return Err(KytoError::InvalidCellBlock {
                context: "highly_variable_genes statistics pass".to_owned(),
                problem: format!(
                    "a statistics pass over {} genes got a block of {} genes",
                    self.n_genes,
                    layer.n_genes()
                ),
            });
        }
        let rows = layer.indptr().len().saturating_sub(1);
        let labels = match &self.batch {
            Some(batch) => row_labels("highly_variable_genes", batch, block, rows)?,
            None => vec![ALL_CELLS.to_owned(); rows],
        };
        match layer.data() {
            CellValues::F32(values) => {
                self.fold_rows(pass, leaf, layer.indptr(), layer.indices(), values, &labels)
            }
            CellValues::U32(values) => {
                self.fold_rows(pass, leaf, layer.indptr(), layer.indices(), values, &labels)
            }
        }
        Ok(())
    }

    fn fold_rows<T: KernelValue>(
        &self,
        pass: u32,
        leaf: &mut GroupPartial,
        indptr: &[u64],
        indices: &[u32],
        values: &[T],
        labels: &[String],
    ) {
        let check_counts = self.flavor == HvgFlavor::SeuratV3 && pass == 0;
        let expm1 = self.flavor == HvgFlavor::Seurat;
        let mut row = 0;
        while row < labels.len() {
            // A run of rows of one group.
            let label = &labels[row];
            let end = labels[row..]
                .iter()
                .position(|other| other != label)
                .map_or(labels.len(), |length| row + length);
            if !leaf.groups.contains_key(label) {
                leaf.groups.insert(label.clone(), self.empty_moments(pass));
            }
            let Some(moments) = leaf.groups.get_mut(label) else {
                unreachable!("inserted above")
            };
            moments.n_cells += (end - row) as u64;
            let entries = indptr[row] as usize..indptr[end] as usize;
            let clip = (pass == 1).then(|| self.fits.get(label).map(|fit| fit.clip.as_slice()));
            for (&gene, raw) in indices[entries.clone()].iter().zip(&values[entries]) {
                let gene = gene as usize;
                // scanpy's data is `float32`: counts above 2^24 round as there.
                let value32 = raw.to_f64() as f32;
                let value = f64::from(value32);
                match clip {
                    None => {
                        if check_counts
                            && leaf.non_count.is_none()
                            && !(value >= 0.0 && value.fract() == 0.0)
                        {
                            leaf.non_count = Some(value);
                        }
                        if value > 0.0 {
                            moments.positives[gene] += 1;
                        }
                        let transformed = if expm1 {
                            value.exp_m1() as f32
                        } else {
                            value32
                        };
                        moments.sums[gene] += f64::from(transformed);
                        moments.sums_of_squares[gene] += f64::from(transformed * transformed);
                        if let Some(welford) = &mut moments.welford {
                            welford.add(gene, f64::from(transformed));
                        }
                    }
                    Some(clip) => {
                        let element = clip.map_or(f64::NAN, |clip| value.min(clip[gene]));
                        moments.sums[gene] += element;
                        moments.sums_of_squares[gene] += element * element;
                    }
                }
            }
            row = end;
        }
    }

    /// End pass `pass` with its reduction (`None` for a stream without cells).
    ///
    /// # Errors
    /// `KY-E215` if `seurat_v3`'s first pass met a value that is not a count.
    pub(crate) fn end_pass(
        &mut self,
        pass: u32,
        reduced: Option<GroupPartial>,
    ) -> Result<(), KytoError> {
        let reduced = reduced.unwrap_or_else(|| self.leaf());
        if pass == 0 {
            if let Some(value) = reduced.non_count {
                return Err(KytoError::InvalidArgument {
                    function: "highly_variable_genes".to_owned(),
                    parameter: "layer".to_owned(),
                    value: format!("'{}'", self.layer),
                    problem: format!(
                        "flavor 'seurat_v3' models raw counts (non-negative integers), but the \
                         layer holds {value}"
                    ),
                    accepted: "a layer of raw counts (the 'counts' layer before \
                               normalize_total and log1p); scanpy only warns here"
                        .to_owned(),
                });
            }
            self.first = reduced.groups;
            // No cell reached the step (every cell filtered out): nothing to rank, and
            // without a batch the batched columns would have no rows to come from.
            if self.first.values().all(|moments| moments.n_cells == 0) {
                return Err(KytoError::InvalidArgument {
                    function: "highly_variable_genes".to_owned(),
                    parameter: "layer".to_owned(),
                    value: format!("'{}'", self.layer),
                    problem: "no cell reaches highly_variable_genes (the filters before it \
                              keep none)"
                        .to_owned(),
                    accepted: "a plan keeping at least one cell before highly_variable_genes"
                        .to_owned(),
                });
            }
            if self.flavor == HvgFlavor::SeuratV3 {
                self.fits = self
                    .first
                    .iter()
                    .map(|(label, moments)| {
                        let batch = self.batch.as_ref().map(|_| label.as_str());
                        Ok((label.clone(), seurat_v3_fit(moments, batch)?))
                    })
                    .collect::<Result<_, KytoError>>()?;
            }
        } else {
            self.second = reduced.groups;
        }
        Ok(())
    }

    /// The statistics of every pass (the gene columns not computed yet).
    pub(crate) fn finish(mut self) -> HighlyVariableGeneStatistics {
        let mut clipped_of = |label: &str, fits: &mut BTreeMap<String, SeuratV3Fit>| {
            let fit = fits.remove(label)?;
            let moments = self
                .second
                .remove(label)
                .unwrap_or_else(|| GeneMoments::new(self.n_genes, false));
            Some(ClippedMoments {
                clip: fit.clip,
                regularized_std: fit.regularized_std,
                means: fit.means,
                moments,
            })
        };
        let mut fits = std::mem::take(&mut self.fits);
        if self.batch.is_none() {
            let moments = self
                .first
                .remove(ALL_CELLS)
                .unwrap_or_else(|| GeneMoments::new(self.n_genes, false));
            let clipped = clipped_of(ALL_CELLS, &mut fits);
            return HighlyVariableGeneStatistics {
                flavor: self.flavor,
                moments,
                clipped,
                batches: Vec::new(),
                columns: Vec::new(),
            };
        }
        let mut overall = GeneMoments::new(self.n_genes, false);
        let first = std::mem::take(&mut self.first);
        let batches = first
            .into_iter()
            .map(|(label, moments)| {
                overall.combine(&moments);
                let clipped = clipped_of(&label, &mut fits);
                BatchStatistics {
                    label,
                    moments,
                    clipped,
                }
            })
            .collect();
        HighlyVariableGeneStatistics {
            flavor: self.flavor,
            moments: overall,
            clipped: None,
            batches,
            columns: Vec::new(),
        }
    }
}

/// The batch of each row of `block`: its `batch_key` value as text, from the block's
/// computed columns or else the stored cell table.
///
/// # Errors
/// `KY-E215` for a null value (naming the cell); `KY-E113` if the column cannot be
/// read or cast to text.
/// The `batch_key` value of each row of `block` as text (`function` names the method in
/// the error for a null batch).
pub(crate) fn row_labels(
    function: &str,
    batch: &BatchColumn,
    block: &PipelineBlock,
    rows: usize,
) -> Result<Vec<String>, KytoError> {
    let corrupt = |problem: String| KytoError::CorruptTable {
        path: format!("{}/cells.parquet", batch.store),
        problem,
    };
    let positions = || {
        block
            .cells
            .iter()
            .map(|cell| IdxSize::try_from(*cell))
            .collect::<Result<Vec<_>, _>>()
            .map(|positions| IdxCa::from_vec("cells".into(), positions))
            .map_err(|_| corrupt("a cell position is beyond Polars' index".to_owned()))
    };
    let series = match block.columns.column(&batch.name) {
        Ok(column) => column.as_materialized_series().clone(),
        Err(_) => batch
            .table
            .column(&batch.name)
            .map_err(|error| corrupt(error.to_string()))?
            .as_materialized_series()
            .take(&positions()?)
            .map_err(|error| corrupt(error.to_string()))?,
    };
    let text = series
        .cast(&DataType::String)
        .map_err(|error| corrupt(error.to_string()))?;
    let text = text.str().map_err(|error| corrupt(error.to_string()))?;
    let mut labels = Vec::with_capacity(rows);
    for (row, value) in text.iter().enumerate() {
        match value {
            Some(value) => labels.push(value.to_owned()),
            None => {
                let cell = batch
                    .table
                    .column("cell_id")
                    .ok()
                    .and_then(|ids| ids.str().ok().cloned())
                    .and_then(|ids| {
                        let position = usize::try_from(*block.cells.get(row)?).ok()?;
                        ids.get(position).map(str::to_owned)
                    })
                    .unwrap_or_else(|| format!("at row {row}"));
                return Err(KytoError::InvalidArgument {
                    function: function.to_owned(),
                    parameter: "batch_key".to_owned(),
                    value: format!("'{}'", batch.name),
                    problem: format!("cell '{cell}' has no batch (null)"),
                    accepted: format!(
                        "a cell column without nulls; drop those cells first with \
                         filter_cells(cell_column('{}').is_not_null())",
                        batch.name
                    ),
                });
            }
        }
    }
    if labels.len() != rows {
        return Err(corrupt(format!(
            "the batch column has {} values for a block of {rows} cells",
            labels.len()
        )));
    }
    Ok(labels)
}
