//! The group label of each cell of a stream's blocks (task 6.10's `groupby`).
//!
//! Does: give each row of a [`PipelineBlock`] its `groupby` value as text, from the
//! block's computed columns (`with_cells`) or else the stored cell table (read once,
//! at the rows' table positions); a null value is `None` (no group). Labels get small
//! codes in the order first seen ([`LabelCodes`]). Labels sort as scanpy orders its
//! categories ([`GroupColumn::compare`]): an `Enum` column in its category order, a
//! numeric one by value, any other as text.
//! Does not: pick groups (`rank_genes.rs`).

use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::Path;

use kyto_error::KytoError;
use kyto_exec::PipelineBlock;
use kyto_store::{TableAxis, read_table};
use polars::prelude::{DataFrame, DataType, IdxCa, IdxSize};

/// The `groupby` cell column of one run and, once needed, the stored cell table.
#[derive(Debug)]
pub(crate) struct GroupColumn {
    name: String,
    store: String,
    stored: Option<DataFrame>,
    order: LabelOrder,
}

/// How the labels of a column sort (scanpy's category order).
#[derive(Debug, Default)]
enum LabelOrder {
    /// As text (strings: pandas sorts new categories).
    #[default]
    Text,
    /// By numeric value (integers and floats become sorted numeric categories).
    Numeric,
    /// By position in the `Enum`'s categories (a categorical keeps its order).
    Categories(HashMap<String, usize>),
}

impl GroupColumn {
    /// The column `name` of the cells of the store at `store`.
    pub(crate) fn new(name: &str, store: &str) -> Self {
        Self {
            name: name.to_owned(),
            store: store.to_owned(),
            stored: None,
            order: LabelOrder::Text,
        }
    }

    /// The order of two labels of this column (as seen in the last block read).
    pub(crate) fn compare(&self, left: &str, right: &str) -> Ordering {
        match &self.order {
            LabelOrder::Text => left.cmp(right),
            LabelOrder::Numeric => {
                let value = |label: &str| label.parse::<f64>().unwrap_or(f64::NAN);
                value(left)
                    .total_cmp(&value(right))
                    .then_with(|| left.cmp(right))
            }
            LabelOrder::Categories(positions) => {
                let position = |label: &str| positions.get(label).copied().unwrap_or(usize::MAX);
                position(left)
                    .cmp(&position(right))
                    .then_with(|| left.cmp(right))
            }
        }
    }

    fn corrupt(&self, problem: String) -> KytoError {
        KytoError::CorruptTable {
            path: format!("{}/cells.parquet", self.store),
            problem,
        }
    }

    /// The label of each row of `block` (`None` for a null value).
    ///
    /// # Errors
    /// As [`read_table`]; `KY-E113` if the column cannot be read or cast to text.
    pub(crate) fn labels(
        &mut self,
        block: &PipelineBlock,
    ) -> Result<Vec<Option<String>>, KytoError> {
        let series = if let Ok(column) = block.columns.column(&self.name) {
            column.as_materialized_series().clone()
        } else {
            if self.stored.is_none() {
                self.stored = Some(read_table(Path::new(&self.store), TableAxis::Cells)?);
            }
            let positions = block
                .cells
                .iter()
                .map(|cell| IdxSize::try_from(*cell))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| self.corrupt("a cell position is beyond Polars' index".to_owned()))?;
            let positions = IdxCa::from_vec("cells".into(), positions);
            let table = self.stored.as_ref().map_or_else(
                || Err(self.corrupt("the cell table was not read".to_owned())),
                Ok,
            )?;
            table
                .column(&self.name)
                .map_err(|error| self.corrupt(error.to_string()))?
                .as_materialized_series()
                .take(&positions)
                .map_err(|error| self.corrupt(error.to_string()))?
        };
        self.order = match series.dtype() {
            DataType::Enum(categories, _) => LabelOrder::Categories(
                categories
                    .categories()
                    .values_iter()
                    .enumerate()
                    .map(|(position, label)| (label.to_owned(), position))
                    .collect(),
            ),
            dtype if dtype.is_primitive_numeric() => LabelOrder::Numeric,
            _ => LabelOrder::Text,
        };
        let text = series
            .cast(&DataType::String)
            .map_err(|error| self.corrupt(error.to_string()))?;
        let text = text
            .str()
            .map_err(|error| self.corrupt(error.to_string()))?;
        let labels: Vec<Option<String>> =
            text.iter().map(|value| value.map(str::to_owned)).collect();
        if labels.len() != block.cells.len() {
            return Err(self.corrupt(format!(
                "column '{}' has {} values for a block of {} cells",
                self.name,
                labels.len(),
                block.cells.len()
            )));
        }
        Ok(labels)
    }
}

/// Codes for labels in the order first seen; `None` (no group) has no code.
#[derive(Debug, Default)]
pub(crate) struct LabelCodes {
    codes: HashMap<String, u32>,
    labels: Vec<String>,
}

impl LabelCodes {
    /// The code of `label`, new if not seen before.
    pub(crate) fn code(&mut self, label: &str) -> u32 {
        if let Some(code) = self.codes.get(label) {
            return *code;
        }
        let code = u32::try_from(self.labels.len()).unwrap_or(u32::MAX);
        self.codes.insert(label.to_owned(), code);
        self.labels.push(label.to_owned());
        code
    }

    /// Every label seen, by code.
    pub(crate) fn labels(&self) -> &[String] {
        &self.labels
    }
}
