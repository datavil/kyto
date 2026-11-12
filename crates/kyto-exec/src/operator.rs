//! The operators a stream runs on every block, in plan order (tasks 5.2, 5.8).
//!
//! Does: dispatch one block to per-cell maps (`normalize_total`, `log1p`, fused or
//! not), a cell step (`filter_cells`, `take_cells`, `with_cells`) or a gene selection
//! (the remap a `filter_genes`/`select_genes` step leaves for later blocks) or
//! `scale`'s apply pass (task 6.4); say
//! whether an operator reads layers; give each new stream counters of its own and
//! check them at the stream's end.
//! Does not: compute statistics passes or gene steps (the pipeline does, before the
//! output's stream).

use kyto_error::KytoError;

use crate::cell_block_pool::CellBlockPool;
use crate::cell_steps::CellStep;
use crate::elementwise::ElementwiseMaps;
use crate::gene_steps::GeneSelection;
use crate::pipeline_block::PipelineBlock;
use crate::scale::ScaleMap;

/// One streamed operator.
#[derive(Debug, Clone)]
pub(crate) enum Operator {
    /// Per-cell maps.
    Maps(ElementwiseMaps),
    /// A cell filter, `take_cells` or `with_cells`.
    Cells(CellStep),
    /// The genes kept by a gene filter or selection.
    Genes(GeneSelection),
    /// `scale`'s apply pass (task 6.4).
    Scale(ScaleMap),
}

impl Operator {
    /// Run the operator on `block`.
    ///
    /// # Errors
    /// Those of the operator ([`ElementwiseMaps::apply`], [`CellStep::apply`],
    /// [`GeneSelection::apply`]).
    pub(crate) fn apply(
        &self,
        block: PipelineBlock,
        pool: &CellBlockPool,
        store: &str,
    ) -> Result<PipelineBlock, KytoError> {
        match self {
            Self::Maps(maps) => maps.apply(block, pool, store),
            Self::Cells(step) => step.apply(block, pool),
            Self::Genes(selection) => selection.apply(block, pool),
            Self::Scale(scale) => scale.apply(block, pool, store),
        }
    }

    /// Whether it needs the blocks' layers (a gene selection only remaps what is there).
    pub(crate) fn reads_layers(&self) -> bool {
        match self {
            Self::Maps(_) | Self::Scale(_) => true,
            Self::Cells(step) => step.reads_layers(),
            Self::Genes(_) => false,
        }
    }

    /// The same operator for a new stream (its counters at zero).
    pub(crate) fn fresh(&self) -> Self {
        match self {
            Self::Cells(step) => Self::Cells(step.fresh()),
            other => other.clone(),
        }
    }

    /// At the end of a stream: the error a `take_cells` reports if some of its cells
    /// never reached it.
    pub(crate) fn check_end(&self) -> Result<(), KytoError> {
        match self {
            Self::Cells(step) => step.check_end(),
            _ => Ok(()),
        }
    }
}
