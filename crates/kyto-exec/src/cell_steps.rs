//! Cell steps of the stream (task 5.8): `filter_cells`, `take_cells` and `with_cells`,
//! run on every block as it streams by.
//!
//! Does: evaluate a resolved cell expression over a block's rows ([`CellStep::apply`]):
//! the cell table columns it reads (the store's table at the block's rows, with the
//! columns earlier `with_cells` steps computed for them in the block), a gene's values
//! read from a layer of the block, and per-cell QC metrics (task 6.1) computed from a
//! layer of the block over the genes present at that step. A filter keeps the rows
//! where the predicate is true (null is false, R9); `take_cells` keeps the cells whose
//! IDs it lists, in the stream's (store) order, and reports at the stream's end an ID
//! an earlier step removed; `with_cells` adds or replaces the block's computed columns,
//! cast to the dtypes the plan declares. Every block is computed alone, so the result
//! does not depend on threads or block sizes (R11).
//! Does not: reorder cells (`take_cells` keeps store order, plan 5.8), read embeddings
//! or recompute stored columns (`KY-E601` when the pipeline is built, plan 5.9), or
//! resolve names (the plan did).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use kyto_core::{ColumnDtype, Source};
use kyto_error::KytoError;
use kyto_store::{CellBlock, CellData, CellValues, DenseValues};
use polars::prelude::{
    BooleanChunked, DataFrame, IdxCa, IdxSize, NamedFrom, NewChunkedArray, Series,
};

use crate::cell_block_pool::CellBlockPool;
use crate::expression::{Prepared, cast, mask};
use crate::pipeline_block::PipelineBlock;
use crate::qc::{CellQc, QcValues};

/// One cell step, ready to run: its expressions' leaves resolved against the genes
/// present at its place in the stream.
#[derive(Debug, Clone)]
pub(crate) struct CellStep {
    kind: CellStepKind,
    /// The store's cell table (the rows blocks point into).
    table: Arc<DataFrame>,
    /// The cell table's path, for errors.
    table_path: String,
    store: String,
}

#[derive(Debug, Clone)]
enum CellStepKind {
    Filter(Evaluation),
    Take(Take),
    Columns(Vec<(String, Evaluation, ColumnDtype)>),
}

/// `take_cells`: the store positions of its cells, increasing, and how many of them the
/// current stream has met.
#[derive(Debug, Clone)]
pub(crate) struct Take {
    positions: Arc<[u64]>,
    ids: Arc<[String]>,
    found: Arc<AtomicU64>,
}

/// An expression with each non-column leaf resolved to how a block computes it.
#[derive(Debug, Clone)]
pub(crate) struct Evaluation {
    prepared: Prepared,
    leaves: Vec<CellLeaf>,
}

/// How a block computes one leaf.
#[derive(Debug, Clone)]
pub(crate) enum CellLeaf {
    /// A gene's values in a layer, at its position among the block's genes.
    Gene { layer: String, position: u32 },
    /// A QC metric of a layer.
    Qc { layer: String, qc: Box<CellQc> },
}

impl Evaluation {
    /// `prepared` with `leaves`, one per [`Prepared::leaves`].
    pub(crate) fn new(prepared: Prepared, leaves: Vec<CellLeaf>) -> Self {
        Self { prepared, leaves }
    }

    /// The layers its leaves read.
    fn reads_layers(&self) -> bool {
        !self.leaves.is_empty()
    }
}

impl Take {
    /// The cells at store `positions` (any order), listed by `ids` in the plan.
    pub(crate) fn new(mut positions: Vec<u64>, ids: &[String]) -> Self {
        positions.sort_unstable();
        Self {
            positions: positions.into(),
            ids: ids.into(),
            found: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl CellStep {
    /// A filter keeping the rows where `predicate` is true.
    pub(crate) fn filter(predicate: Evaluation, table: Arc<DataFrame>, store: &str) -> Self {
        Self::new(CellStepKind::Filter(predicate), table, store)
    }

    /// `take_cells`.
    pub(crate) fn take(take: Take, table: Arc<DataFrame>, store: &str) -> Self {
        Self::new(CellStepKind::Take(take), table, store)
    }

    /// `with_cells`: each column's name, expression and declared dtype.
    pub(crate) fn columns(
        columns: Vec<(String, Evaluation, ColumnDtype)>,
        table: Arc<DataFrame>,
        store: &str,
    ) -> Self {
        Self::new(CellStepKind::Columns(columns), table, store)
    }

    fn new(kind: CellStepKind, table: Arc<DataFrame>, store: &str) -> Self {
        Self {
            kind,
            table,
            table_path: format!("{store}/cells.parquet"),
            store: store.to_owned(),
        }
    }

    /// Whether it reads a layer of the blocks.
    pub(crate) fn reads_layers(&self) -> bool {
        match &self.kind {
            CellStepKind::Filter(predicate) => predicate.reads_layers(),
            CellStepKind::Take(_) => false,
            CellStepKind::Columns(columns) => columns
                .iter()
                .any(|(_, evaluation, _)| evaluation.reads_layers()),
        }
    }

    /// The same step counting from zero (a new stream over it).
    pub(crate) fn fresh(&self) -> Self {
        let mut step = self.clone();
        if let CellStepKind::Take(take) = &mut step.kind {
            take.found = Arc::new(AtomicU64::new(0));
        }
        step
    }

    /// At the end of a stream: `KY-E210` for a `take_cells` whose cells were not all
    /// met (an earlier step removed some).
    pub(crate) fn check_end(&self) -> Result<(), KytoError> {
        let CellStepKind::Take(take) = &self.kind else {
            return Ok(());
        };
        if take.found.load(Ordering::Relaxed) == take.positions.len() as u64 {
            return Ok(());
        }
        Err(KytoError::UnknownName {
            kind: "cell".to_owned(),
            name: take.ids.first().cloned().unwrap_or_default(),
            expression: format!("take_cells([{} IDs])", take.ids.len()),
            problem: format!(
                "only {} of its {} cells reach it: an earlier step (a filter or another \
                 take_cells) removed the others",
                take.found.load(Ordering::Relaxed),
                take.positions.len()
            ),
        })
    }

    /// Run the step on `block`.
    ///
    /// # Errors
    /// - `KY-E109` if a leaf's layer is not in the block (a plan trimmed wrongly).
    /// - `KY-E113` if an expression cannot be evaluated or cast over the cell table.
    /// - `KY-E117` if a QC gene set does not match the block's genes.
    pub(crate) fn apply(
        &self,
        mut block: PipelineBlock,
        pool: &CellBlockPool,
    ) -> Result<PipelineBlock, KytoError> {
        match &self.kind {
            CellStepKind::Filter(predicate) => {
                let values = self.evaluate(predicate, &block)?;
                let keep = mask(&values, &self.table_path)?;
                keep_rows(block, &keep, pool)
            }
            CellStepKind::Take(take) => {
                let keep: Vec<bool> = block
                    .cells
                    .iter()
                    .map(|cell| take.positions.binary_search(cell).is_ok())
                    .collect();
                let found = keep.iter().filter(|kept| **kept).count() as u64;
                take.found.fetch_add(found, Ordering::Relaxed);
                keep_rows(block, &keep, pool)
            }
            CellStepKind::Columns(columns) => {
                // Every column is computed from the block as it came in (Polars'
                // `with_columns`), then they are added in order.
                let values = columns
                    .iter()
                    .map(|(name, evaluation, dtype)| {
                        let values = self.evaluate(evaluation, &block)?;
                        cast(values, name, dtype, &self.table_path)
                    })
                    .collect::<Result<Vec<_>, KytoError>>()?;
                let rows = block.cells.len();
                let mut computed =
                    std::mem::replace(&mut block.columns, DataFrame::empty_with_height(rows));
                for values in values {
                    computed
                        .with_column(values.into())
                        .map_err(|error| self.corrupt(error.to_string()))?;
                }
                block.columns = computed;
                Ok(block)
            }
        }
    }

    fn corrupt(&self, problem: String) -> KytoError {
        KytoError::CorruptTable {
            path: self.table_path.clone(),
            problem,
        }
    }

    /// `evaluation` over the rows of `block`.
    fn evaluate(
        &self,
        evaluation: &Evaluation,
        block: &PipelineBlock,
    ) -> Result<Series, KytoError> {
        let rows = block.cells.len();
        let prepared = &evaluation.prepared;
        // The stored columns it reads at the block's rows, then the block's computed
        // columns over them (a computed column replaces a stored one of its name).
        let stored: Vec<&str> = prepared
            .columns
            .iter()
            .map(String::as_str)
            .filter(|name| block.columns.column(name).is_err())
            .collect();
        let mut frame = if stored.is_empty() {
            DataFrame::empty_with_height(rows)
        } else {
            let positions = block
                .cells
                .iter()
                .map(|cell| IdxSize::try_from(*cell))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| self.corrupt("a cell position is beyond Polars' index".to_owned()))?;
            self.table
                .select(stored)
                .and_then(|table| table.take(&IdxCa::from_vec("cells".into(), positions)))
                .map_err(|error| self.corrupt(error.to_string()))?
        };
        for name in &prepared.columns {
            if let Ok(column) = block.columns.column(name) {
                frame
                    .with_column(column.clone())
                    .map_err(|error| self.corrupt(error.to_string()))?;
            }
        }
        let leaves = evaluation
            .leaves
            .iter()
            .map(|leaf| self.leaf_values(leaf, block))
            .collect::<Result<Vec<_>, KytoError>>()?;
        prepared.evaluate(frame, leaves, &self.table_path)
    }

    /// The values of `leaf` over the rows of `block`.
    fn leaf_values(&self, leaf: &CellLeaf, block: &PipelineBlock) -> Result<Series, KytoError> {
        let layer = |name: &str| {
            block
                .layers
                .get(name)
                .ok_or_else(|| KytoError::MissingMember {
                    kind: "layer".to_owned(),
                    name: name.to_owned(),
                    store: self.store.clone(),
                })
        };
        Ok(match leaf {
            CellLeaf::Gene {
                layer: name,
                position,
            } => match layer(name)?.gene_values(*position)? {
                DenseValues::F32(values) => Series::new("".into(), values),
                // `uint32` leaves are `Float64` for arithmetic, so `counts - 10` does
                // not wrap (user decision, Nov 12, 2026; deviations.md, task 5.8). Exact:
                // every `u32` is an `f64`. A plain read is cast back to `UInt32`.
                DenseValues::U32(values) => Series::new("".into(), widen(&values)),
            },
            CellLeaf::Qc { layer: name, qc } => match qc.evaluate(layer(name)?)? {
                QcValues::Float64(values) => Series::new("".into(), values),
                QcValues::UInt32(values) => Series::new("".into(), widen(&values)),
            },
        })
    }
}

/// The sources of a cell expression's leaves that a block computes (everything but
/// table columns): [`Source::Expression`] and [`Source::Qc`]. Others are refused when
/// the pipeline is built.
pub(crate) fn is_block_leaf(source: &Source) -> bool {
    matches!(source, Source::Expression { .. } | Source::Qc { .. })
}

/// `block` with only the rows where `keep` is set (all of it when every row is kept).
pub(crate) fn keep_rows(
    mut block: PipelineBlock,
    keep: &[bool],
    pool: &CellBlockPool,
) -> Result<PipelineBlock, KytoError> {
    if keep.iter().all(|kept| *kept) {
        return Ok(block);
    }
    let mut kept = keep.iter();
    block
        .cells
        .retain(|_| kept.next().copied().unwrap_or(false));
    if block.columns.width() > 0 {
        let flags = BooleanChunked::from_slice("keep".into(), keep);
        block.columns =
            block
                .columns
                .filter(&flags)
                .map_err(|error| KytoError::InvalidCellBlock {
                    context: "keep_rows".to_owned(),
                    problem: error.to_string(),
                })?;
    } else {
        block.columns = DataFrame::empty_with_height(block.cells.len());
    }
    let layers = std::mem::take(&mut block.layers);
    for (name, layer) in layers {
        let subset = rows_of(&layer, keep, pool)?;
        pool.recycle(layer);
        block.layers.insert(name, subset);
    }
    Ok(block)
}

/// The rows of `layer` where `keep` is set, as a new block from `pool`.
fn rows_of(layer: &CellBlock, keep: &[bool], pool: &CellBlockPool) -> Result<CellBlock, KytoError> {
    let mut parts = pool.take(layer.dtype());
    let indptr = layer.indptr();
    let indices = layer.indices();
    parts.indptr.push(0);
    let mut nnz = 0_u64;
    for (row, kept) in keep.iter().enumerate() {
        if !*kept {
            continue;
        }
        let range = indptr[row] as usize..indptr[row + 1] as usize;
        parts.indices.extend_from_slice(&indices[range.clone()]);
        match (&mut parts.data, layer.data()) {
            (CellData::F32(out), CellValues::F32(values)) => {
                out.extend_from_slice(&values[range.clone()])
            }
            (CellData::U32(out), CellValues::U32(values)) => {
                out.extend_from_slice(&values[range.clone()])
            }
            _ => unreachable!("the pool's parts take the layer's dtype"),
        }
        nnz += range.len() as u64;
        parts.indptr.push(nnz);
    }
    CellBlock::from_parts(parts, layer.n_genes())
}

/// `uint32` leaf values as `f64` (exact), the dtype arithmetic on them uses.
pub(crate) fn widen(values: &[u32]) -> Vec<f64> {
    values.iter().map(|&value| f64::from(value)).collect()
}
