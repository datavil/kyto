//! The cell column a clustering writes (task 6.8), added to every block.
//!
//! Does: hold each clustered cell's label (by store table position) and add the
//! column `into` to each block's computed columns (replacing one of that name), as a
//! Polars `Enum` with the categories `"0"`, `"1"`, ... in number order: scanpy's
//! categorical labels (`natsorted`), every cluster a category even where a later
//! filter leaves it no cell. The plan declares the column `String` (the categories are
//! known only once the clustering ran); a sink writes it as this `Enum`.
//! Does not: cluster (see `leiden.rs`).

use std::sync::Arc;

use kyto_error::KytoError;
use polars::prelude::{DataFrame, DataType, FrozenCategories, NamedFrom, Series};

use crate::pipeline_block::PipelineBlock;

/// The labels of one clustering, added to each block.
#[derive(Debug, Clone)]
pub(crate) struct ClusterLabels {
    name: String,
    /// (store table position, label), by position.
    labels: Arc<Vec<(u64, u32)>>,
    categories: Arc<Vec<String>>,
    dtype: DataType,
}

impl ClusterLabels {
    /// The column `name` of `labels[i]` for the cell at table position `cells[i]`,
    /// with `n_clusters` categories.
    pub(crate) fn new(
        name: &str,
        cells: &[u64],
        labels: &[u32],
        n_clusters: u32,
    ) -> Result<Self, KytoError> {
        let mut by_cell: Vec<(u64, u32)> =
            cells.iter().copied().zip(labels.iter().copied()).collect();
        by_cell.sort_unstable_by_key(|&(cell, _)| cell);
        let categories: Vec<String> = (0..n_clusters).map(|label| label.to_string()).collect();
        let frozen = FrozenCategories::new(categories.iter().map(String::as_str))
            .map_err(|error| internal(name, &error.to_string()))?;
        Ok(Self {
            name: name.to_owned(),
            labels: Arc::new(by_cell),
            categories: Arc::new(categories),
            dtype: DataType::from_frozen_categories(frozen),
        })
    }

    /// `block` with the labels of its cells as column `name`.
    ///
    /// # Errors
    /// `KY-E215` if a cell of the block was not clustered (the pipeline refuses plans
    /// where that could happen, so this is a bug).
    pub(crate) fn apply(&self, mut block: PipelineBlock) -> Result<PipelineBlock, KytoError> {
        let values = block
            .cells
            .iter()
            .map(|cell| {
                self.labels
                    .binary_search_by_key(cell, |&(position, _)| position)
                    .map(|index| self.categories[self.labels[index].1 as usize].as_str())
                    .map_err(|_| {
                        internal(
                            &self.name,
                            &format!("the cell at table position {cell} was not clustered"),
                        )
                    })
            })
            .collect::<Result<Vec<&str>, KytoError>>()?;
        let column = Series::new(self.name.as_str().into(), values)
            .strict_cast(&self.dtype)
            .map_err(|error| internal(&self.name, &error.to_string()))?;
        let rows = block.cells.len();
        let mut computed =
            std::mem::replace(&mut block.columns, DataFrame::empty_with_height(rows));
        computed
            .with_column(column.into())
            .map_err(|error| internal(&self.name, &error.to_string()))?;
        block.columns = computed;
        Ok(block)
    }
}

fn internal(name: &str, problem: &str) -> KytoError {
    KytoError::InvalidArgument {
        function: "Pipeline::run".to_owned(),
        parameter: "plan".to_owned(),
        value: name.to_owned(),
        problem: format!("the cluster labels '{name}': {problem}"),
        accepted: "a plan built by kyto".to_owned(),
    }
}
