//! Scrublet in the stream (task 6.12): the pass that gathers its input, the run, and
//! the two cell columns it adds to every later block.
//!
//! Does: gather the rows of the node's input layer reaching it (as `float32`, scanpy's
//! dtype) with each cell's store table position and, with a `batch_key`, its batch as
//! text (the block's computed column or the stored one: `KY-E215` for a null batch), in
//! cell order (leaves of one pass concatenated in block order, `reduction.rs`); run
//! [`crate::scrublet`] on them once the pass ended; then add the score (`Float64`) and
//! call (`Boolean`) columns to each block, null for a cell Scrublet did not score.
//! Does not: detect doublets (see `scrublet.rs`). Holds the whole layer in memory, as
//! scanpy does (the estimate counts it).

use std::sync::Arc;

use kyto_error::KytoError;
use kyto_plan::Scrublet;
use kyto_store::{CellBlock, CellValues};
use polars::prelude::{DataFrame, NamedFrom, Series};

use crate::hvg_pass::{BatchColumn, row_labels};
use crate::kernels::KernelValue;
use crate::pipeline_block::PipelineBlock;
use crate::run_control::RunControl;
use crate::scrublet::{self, Doublets, ScrubletResult};

/// The gathered rows of some cells (a leaf, or leaves combined), in cell order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GatheredRows {
    cells: Vec<u64>,
    labels: Vec<String>,
    indptr: Vec<u64>,
    indices: Vec<u32>,
    values: Vec<f32>,
}

impl Default for GatheredRows {
    /// No rows (an `indptr` of one 0).
    fn default() -> Self {
        Self {
            cells: Vec::new(),
            labels: Vec::new(),
            indptr: vec![0],
            indices: Vec::new(),
            values: Vec::new(),
        }
    }
}

impl GatheredRows {
    /// `self` (the cells before) and `right` (the cells after) as one.
    pub(crate) fn combine(mut self, right: Self) -> Self {
        let offset = self.indices.len() as u64;
        self.cells.extend(right.cells);
        self.labels.extend(right.labels);
        self.indptr
            .extend(right.indptr.iter().skip(1).map(|end| end + offset));
        self.indices.extend(right.indices);
        self.values.extend(right.values);
        self
    }
}

/// What Scrublet's pass gathers, and its parameters.
#[derive(Debug, Clone)]
pub(crate) struct ScrubletAccumulator {
    params: Scrublet,
    n_genes: usize,
    batch: Option<BatchColumn>,
    gathered: GatheredRows,
}

impl ScrubletAccumulator {
    pub(crate) fn new(params: &Scrublet, n_genes: usize, batch: Option<BatchColumn>) -> Self {
        Self {
            params: params.clone(),
            n_genes,
            batch,
            gathered: GatheredRows::default(),
        }
    }

    pub(crate) fn leaf(&self) -> GatheredRows {
        GatheredRows::default()
    }

    /// Add the rows of `layer` (the input layer of `block`) to `leaf`.
    ///
    /// # Errors
    /// `KY-E117` if the block's genes are not the accumulator's; `KY-E215` for a null
    /// batch; `KY-E113` if the batch column cannot be read as text.
    pub(crate) fn fold(
        &self,
        leaf: &mut GatheredRows,
        layer: &CellBlock,
        block: &PipelineBlock,
    ) -> Result<(), KytoError> {
        if layer.n_genes() as usize != self.n_genes {
            return Err(KytoError::InvalidCellBlock {
                context: "scrublet statistics pass".to_owned(),
                problem: format!(
                    "a statistics pass over {} genes got a block of {} genes",
                    self.n_genes,
                    layer.n_genes()
                ),
            });
        }
        let rows = layer.indptr().len().saturating_sub(1);
        if let Some(batch) = &self.batch {
            leaf.labels
                .extend(row_labels("scrublet", batch, block, rows)?);
        }
        leaf.cells.extend_from_slice(&block.cells);
        let offset = leaf.indices.len() as u64;
        leaf.indptr
            .extend(layer.indptr().iter().skip(1).map(|end| end + offset));
        leaf.indices.extend_from_slice(layer.indices());
        match layer.data() {
            CellValues::F32(values) => leaf.values.extend_from_slice(values),
            CellValues::U32(values) => leaf
                .values
                .extend(values.iter().map(|value| value.to_f32())),
        }
        Ok(())
    }

    /// Keep the pass's reduction (`None` for a stream without cells).
    pub(crate) fn end_pass(&mut self, reduced: Option<GatheredRows>) {
        self.gathered = reduced.unwrap_or_default();
    }

    /// Run Scrublet on the gathered rows: its statistics and the columns' operator.
    ///
    /// # Errors
    /// Those of [`crate::scrublet::scrublet`]; `KY-E503` when cancelled.
    pub(crate) fn finish(
        self,
        control: &RunControl,
    ) -> Result<(ScrubletStatistics, DoubletColumns), KytoError> {
        let GatheredRows {
            cells,
            labels,
            indptr,
            indices,
            values,
        } = self.gathered;
        let counts = CellBlock::from_f32(indptr, indices, values, self.n_genes as u64)?;
        let labels = self.batch.is_some().then_some(labels.as_slice());
        let result = scrublet::run(&counts, labels, &self.params, &Doublets::Seeded, control)?;
        let columns = DoubletColumns::new(&self.params, &cells, &result);
        Ok((
            ScrubletStatistics {
                cells: Arc::new(cells),
                result,
            },
            columns,
        ))
    }
}

/// What a `scrublet` node computed: its cells and their scores, calls and batches.
#[derive(Debug, Clone, PartialEq)]
pub struct ScrubletStatistics {
    pub(crate) cells: Arc<Vec<u64>>,
    pub(crate) result: ScrubletResult,
}

impl ScrubletStatistics {
    /// The store table position of each cell reaching the node, in stream order (the
    /// rows of [`ScrubletStatistics::result`]).
    pub fn cells(&self) -> &[u64] {
        &self.cells
    }

    /// The scores, calls and per-batch details, one row per cell of `cells()`.
    pub fn result(&self) -> &ScrubletResult {
        &self.result
    }
}

/// One cell's row of the doublet columns: (store table position, score, call).
type DoubletRow = (u64, Option<f64>, Option<bool>);

/// The two cell columns of a Scrublet run, added to each block.
#[derive(Debug, Clone)]
pub(crate) struct DoubletColumns {
    score_name: String,
    predicted_name: String,
    /// (store table position, score, call), by position.
    rows: Arc<Vec<DoubletRow>>,
}

impl DoubletColumns {
    fn new(params: &Scrublet, cells: &[u64], result: &ScrubletResult) -> Self {
        let mut rows: Vec<DoubletRow> = cells
            .iter()
            .zip(&result.scores)
            .zip(&result.predicted)
            .map(|((&cell, &score), &predicted)| (cell, score, predicted))
            .collect();
        rows.sort_unstable_by_key(|&(cell, _, _)| cell);
        Self {
            score_name: params.score_into.clone(),
            predicted_name: params.predicted_into.clone(),
            rows: Arc::new(rows),
        }
    }

    /// `block` with its cells' scores and calls as two computed columns.
    ///
    /// # Errors
    /// `KY-E215` if a cell of the block did not reach the node (the plan order makes
    /// that impossible, so this is a bug).
    pub(crate) fn apply(&self, mut block: PipelineBlock) -> Result<PipelineBlock, KytoError> {
        let mut scores = Vec::with_capacity(block.cells.len());
        let mut predicted = Vec::with_capacity(block.cells.len());
        for cell in &block.cells {
            let index = self
                .rows
                .binary_search_by_key(cell, |&(position, _, _)| position)
                .map_err(|_| {
                    internal(&format!(
                        "the cell at table position {cell} did not reach scrublet"
                    ))
                })?;
            let (_, score, call) = self.rows[index];
            scores.push(score);
            predicted.push(call);
        }
        let rows = block.cells.len();
        let mut computed =
            std::mem::replace(&mut block.columns, DataFrame::empty_with_height(rows));
        for column in [
            Series::new(self.score_name.as_str().into(), scores),
            Series::new(self.predicted_name.as_str().into(), predicted),
        ] {
            computed
                .with_column(column.into())
                .map_err(|error| internal(&error.to_string()))?;
        }
        block.columns = computed;
        Ok(block)
    }
}

fn internal(problem: &str) -> KytoError {
    KytoError::InvalidArgument {
        function: "Pipeline::run".to_owned(),
        parameter: "plan".to_owned(),
        value: "scrublet".to_owned(),
        problem: format!("the doublet columns: {problem}"),
        accepted: "a plan built by kyto".to_owned(),
    }
}
