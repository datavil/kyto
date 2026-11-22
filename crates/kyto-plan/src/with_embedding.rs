//! The node that adds a cell embedding computed outside kyto: `with_embedding`.
//!
//! Does: hold the embedding's name and its values (one row per cell reaching the node,
//! in stream order, `width` values a row, `float32` or `float64` as given), shared
//! behind an `Arc` so plans built on it copy no values. Its output schema adds the
//! cell embedding `name`, replacing one of that name. Changes no axis. Cross-cell:
//! row `r` belongs to the `r`-th cell reaching the node, so no cell filter may move
//! across it (predicate pushdown moves cell filters across per-cell nodes only).
//! Does not: compute anything (task 6.11's interop hooks: scvi-tools and Harmony run in
//! Python), or check the row count against cells known only when the plan runs (the
//! executor does, `KY-E215`).

use std::fmt;
use std::sync::Arc;

use kyto_core::{Axis, EmbeddingSchema, FloatDtype};
use kyto_error::KytoError;

use crate::names::MemberGroup;
use crate::node::{CellScope, LogicalNode, Transform};
use crate::plan::Plan;
use crate::plan_schema::PlanSchema;

/// The values of a given embedding, row-major, in the dtype they were given in.
#[derive(Debug, Clone)]
pub enum GivenValues {
    /// `float32` values.
    Float32(Vec<f32>),
    /// `float64` values.
    Float64(Vec<f64>),
}

impl GivenValues {
    /// The number of values.
    pub fn len(&self) -> usize {
        match self {
            Self::Float32(values) => values.len(),
            Self::Float64(values) => values.len(),
        }
    }

    /// Whether there are no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Their dtype.
    pub fn dtype(&self) -> FloatDtype {
        match self {
            Self::Float32(_) => FloatDtype::Float32,
            Self::Float64(_) => FloatDtype::Float64,
        }
    }

    /// The position of the first value that is NaN or infinite.
    fn first_not_finite(&self) -> Option<usize> {
        match self {
            Self::Float32(values) => values.iter().position(|value| !value.is_finite()),
            Self::Float64(values) => values.iter().position(|value| !value.is_finite()),
        }
    }
}

/// Equal when the dtypes and the bits of every value are equal (so `-0.0` and `0.0`
/// differ: sub-plan reuse must not merge two plans whose outputs differ in bits).
impl PartialEq for GivenValues {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Float32(left), Self::Float32(right)) => {
                left.len() == right.len()
                    && left
                        .iter()
                        .zip(right)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
            }
            (Self::Float64(left), Self::Float64(right)) => {
                left.len() == right.len()
                    && left
                        .iter()
                        .zip(right)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
            }
            _ => false,
        }
    }
}

/// The parameters of `with_embedding`: a cell embedding given by value.
#[derive(Debug, Clone)]
pub struct WithEmbedding {
    name: String,
    width: u64,
    values: Arc<GivenValues>,
}

impl WithEmbedding {
    /// The embedding `name` with `width` values a row, row-major. Checked by
    /// [`Plan::with_embedding`].
    pub fn new(name: impl Into<String>, width: u64, values: GivenValues) -> Self {
        Self {
            name: name.into(),
            width,
            values: Arc::new(values),
        }
    }

    /// The embedding's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The values a row (the embedding's components).
    pub fn width(&self) -> u64 {
        self.width
    }

    /// The values, row-major, in their dtype.
    pub fn values(&self) -> &Arc<GivenValues> {
        &self.values
    }

    /// The number of rows (cells).
    pub fn n_rows(&self) -> u64 {
        let len = self.values.len() as u64;
        len.checked_div(self.width).unwrap_or(0)
    }
}

/// Equal when the names, widths and values (by bits) are equal.
impl PartialEq for WithEmbedding {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.width == other.width
            && (Arc::ptr_eq(&self.values, &other.values) || self.values == other.values)
    }
}

impl LogicalNode for WithEmbedding {
    fn name(&self) -> &'static str {
        "WITH_EMBEDDING"
    }

    fn method(&self) -> &'static str {
        "with_embedding"
    }

    fn changes_axis(&self) -> Option<Axis> {
        None
    }

    fn cell_scope(&self) -> CellScope {
        CellScope::CrossCell
    }

    fn describe(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dtype = match self.values.dtype() {
            FloatDtype::Float32 => "float32",
            FloatDtype::Float64 => "float64",
        };
        write!(
            formatter,
            "{:?} ({} cells x {} {dtype})",
            self.name,
            self.n_rows(),
            self.width
        )
    }
}

impl Transform for WithEmbedding {
    fn output_schema(&self, input: &PlanSchema) -> PlanSchema {
        input.clone().with_embedding(EmbeddingSchema {
            name: self.name.clone(),
            axis: Axis::Cells,
            width: self.width,
            dtype: self.values.dtype(),
        })
    }
}

impl Plan {
    /// This plan with the cell embedding `params.name()` holding `params.values()`:
    /// row `r` for the `r`-th cell of this plan (its `cell_ids()` order).
    ///
    /// # Errors
    /// `KY-E215` for a width of 0, values that do not fill whole rows, a value that is
    /// not finite, a row count other than this plan's cells when that is known without
    /// running it, a `name` of a gene embedding, or a `name` that breaks format.md §2.1
    /// or differs from an embedding of the plan only in case; `KY-E216` if this plan ends in a sink.
    pub fn with_embedding(self: &Arc<Self>, params: WithEmbedding) -> Result<Arc<Self>, KytoError> {
        const METHOD: &str = "with_embedding";
        self.check_open(METHOD)?;
        self.check_member_name(
            METHOD,
            "name",
            &params.name,
            MemberGroup::Embeddings,
            &params.name,
        )?;
        if let Some(existing) = self.output_schema().embedding(&params.name)
            && existing.axis != Axis::Cells
        {
            return Err(Self::invalid_argument(
                METHOD,
                "name",
                &format!("'{}'", params.name),
                "it names a gene embedding (such as pca's loadings); a cell embedding \
                 cannot replace it",
                "a new name, or the name of a cell embedding",
            ));
        }
        let len = params.values.len() as u64;
        if params.width == 0 || !len.is_multiple_of(params.width) {
            return Err(Self::invalid_argument(
                METHOD,
                "values",
                &format!("{len} values, {} a row", params.width),
                "they do not form rows of at least one value",
                "a 2-D array of shape (n_cells, n_components) with n_components of at least 1",
            ));
        }
        if let Some(position) = params.values.first_not_finite() {
            let width = params.width;
            return Err(Self::invalid_argument(
                METHOD,
                "values",
                &format!(
                    "row {}, column {}",
                    position as u64 / width,
                    position as u64 % width
                ),
                "it is NaN or infinite",
                "finite values only (replace or drop the cells whose values are not finite)",
            ));
        }
        if let Some(n_cells) = self.output_schema().n_cells()
            && params.n_rows() != n_cells
        {
            return Err(Self::invalid_argument(
                METHOD,
                "values",
                &format!("{} rows", params.n_rows()),
                &format!("the plan has {n_cells} cells"),
                "one row per cell of the plan, in its cell_ids() order",
            ));
        }
        Ok(self.extend(params, Self::WithEmbedding))
    }
}
