//! The operators a stream runs on every block, in plan order (tasks 5.2, 5.8).
//!
//! Does: dispatch one block to per-cell maps (`normalize_total`, `log1p`, fused or
//! not), a cell step (`filter_cells`, `take_cells`, `with_cells`) or a gene selection
//! (the remap a `filter_genes`/`select_genes` step leaves for later blocks),
//! `scale`'s apply pass (task 6.4), a clustering's labels (task 6.8) or Scrublet's
//! columns (task 6.12); say
//! whether an operator reads layers and which a block keeps after each operator
//! ([`kept_layers`], task 5.16); give each new stream counters of its own and
//! check them at the stream's end.
//! Does not: compute statistics passes or gene steps (the pipeline does, before the
//! output's stream).

use std::collections::BTreeSet;

use kyto_error::KytoError;

use crate::cell_block_pool::CellBlockPool;
use crate::cell_steps::CellStep;
use crate::cluster_labels::ClusterLabels;
use crate::elementwise::ElementwiseMaps;
use crate::gene_steps::GeneSelection;
use crate::pipeline_block::PipelineBlock;
use crate::scale::ScaleMap;
use crate::scrublet_pass::DoubletColumns;

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
    /// A clustering's labels as a computed cell column (task 6.8).
    Labels(ClusterLabels),
    /// Scrublet's scores and calls as computed cell columns (task 6.12).
    Doublets(DoubletColumns),
}

impl Operator {
    /// Run the operator on `block`; `kept` are the layers the block keeps after it
    /// ([`kept_layers`]), so per-cell maps can free a layer between their maps.
    ///
    /// # Errors
    /// Those of the operator ([`ElementwiseMaps::apply`], [`CellStep::apply`],
    /// [`GeneSelection::apply`]).
    pub(crate) fn apply(
        &self,
        block: PipelineBlock,
        pool: &CellBlockPool,
        store: &str,
        kept: &BTreeSet<String>,
    ) -> Result<PipelineBlock, KytoError> {
        match self {
            Self::Maps(maps) => maps.apply(block, pool, store, kept),
            Self::Cells(step) => step.apply(block, pool),
            Self::Genes(selection) => selection.apply(block, pool),
            Self::Scale(scale) => scale.apply(block, pool, store),
            Self::Labels(labels) => labels.apply(block),
            Self::Doublets(columns) => columns.apply(block),
        }
    }

    /// Whether it needs the blocks' layers (a gene selection only remaps what is there).
    pub(crate) fn reads_layers(&self) -> bool {
        match self {
            Self::Maps(_) | Self::Scale(_) => true,
            Self::Cells(step) => step.reads_layers(),
            Self::Genes(_) | Self::Labels(_) | Self::Doublets(_) => false,
        }
    }

    /// The layers it reads from a block (task 5.16). A gene selection remaps whatever
    /// layers a block carries and reads none of its own.
    pub(crate) fn read_layers(&self) -> Vec<&str> {
        match self {
            Self::Maps(maps) => maps.input_layers().collect(),
            Self::Cells(step) => step.layers(),
            Self::Scale(scale) => vec![scale.input_layer()],
            Self::Genes(_) | Self::Labels(_) | Self::Doublets(_) => Vec::new(),
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

/// The layers a block keeps before the first of `operators` and after each of them: those
/// a later operator reads or the stream outputs (`output`). One set more than
/// operators; the stream drops (to its pool) every other layer as soon as it may, so a
/// layer no later step reads is neither carried nor copied by a gene selection (task
/// 5.16). The last set is `output`.
pub(crate) fn kept_layers(
    operators: &[Operator],
    output: &BTreeSet<String>,
) -> Vec<BTreeSet<String>> {
    let mut kept = vec![output.clone()];
    for operator in operators.iter().rev() {
        let mut before = kept.last().cloned().unwrap_or_default();
        before.extend(operator.read_layers().into_iter().map(str::to_owned));
        kept.push(before);
    }
    kept.reverse();
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use kyto_plan::MapElementwise;

    /// `normalize_total(10)` of `counts` into `norm`, then `log1p` of `norm` into `log1p`.
    fn normalize_then_log1p() -> Vec<MapElementwise> {
        serde_json::from_str(
            r#"[{"op": {"op": "normalize_total", "target_sum": 10.0}, "layer": "counts", "into": "norm"},
                {"op": {"op": "log1p"}, "layer": "norm", "into": "log1p"}]"#,
        )
        .unwrap()
    }

    #[test]
    fn a_layer_is_kept_until_its_last_reader() {
        // normalize_total (counts -> norm), then log1p (norm -> log1p), output log1p.
        let operators: Vec<Operator> = normalize_then_log1p()
            .into_iter()
            .map(|map| Operator::Maps(ElementwiseMaps::new(vec![map], None)))
            .collect();
        let set = |names: &[&str]| names.iter().map(|name| (*name).to_owned()).collect();
        let output: BTreeSet<String> = set(&["log1p"]);
        assert_eq!(
            kept_layers(&operators, &output),
            [
                set(&["counts", "log1p", "norm"]),
                set(&["log1p", "norm"]),
                output.clone()
            ]
        );
        assert_eq!(kept_layers(&[], &output), [output]);
    }
}
