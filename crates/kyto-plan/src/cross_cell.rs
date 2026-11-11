//! The node for steps that need statistics over all cells first: `scale`,
//! `highly_variable_genes` and `pca`.
//!
//! Does: hold the operation and its parameters, checked when built, with the input
//! layer written in full (R6). Each one's output schema follows from its parameters:
//! - `scale` writes layer `into` (`float32`); zero-centred it is dense, so its `nnz`
//!   has the bound `n_cells x n_genes` (a value equal to its gene's mean becomes 0,
//!   so never exact), otherwise it keeps the input's `nnz`. Writing a
//!   layer that exists makes stored columns computed from it stale (`KY-E301`).
//! - `highly_variable_genes` adds scanpy's gene columns for its flavour (replacing
//!   columns of the same name): `highly_variable`, `means`, `dispersions` and
//!   `dispersions_norm` (`seurat`, `cell_ranger`), or `highly_variable`,
//!   `highly_variable_rank`, `means`, `variances` and `variances_norm` (`seurat_v3`);
//!   with a `batch_key` also `highly_variable_nbatches` and (not `seurat_v3`)
//!   `highly_variable_intersection`.
//! - `pca` adds the cell embedding `into` (`n_components` wide) and the gene embedding
//!   `<into>_loadings` (scanpy's `varm['PCs']`), both `float32`.
//!
//! None changes an axis; all are cross-cell (statistics pass, then output).
//! Does not: compute anything (6.3, 6.4, 6.5), or check what only data shows (that
//! `seurat_v3` reads counts).

use std::fmt;
use std::sync::Arc;

use kyto_core::{
    Axis, CellExpr, ColumnDtype, ColumnSchema, EmbeddingSchema, ExprNode, FloatDtype, LayerDtype,
    LayerSchema, Source,
};
use kyto_error::KytoError;
use serde::{Deserialize, Serialize};

use crate::node::{CellScope, LogicalNode, Transform};
use crate::plan::Plan;
use crate::plan_schema::{PlanSchema, SparseMember};

/// The flavour of `highly_variable_genes` (scanpy's `flavor`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HvgFlavor {
    /// Seurat's dispersion-based method on log data.
    Seurat,
    /// Cell Ranger's dispersion-based method.
    CellRanger,
    /// Seurat v3's variance-stabilising method on counts.
    SeuratV3,
}

impl HvgFlavor {
    /// The name scanpy and the Python API use.
    pub fn name(self) -> &'static str {
        match self {
            Self::Seurat => "seurat",
            Self::CellRanger => "cell_ranger",
            Self::SeuratV3 => "seurat_v3",
        }
    }
}

/// A step that needs statistics over all cells.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "op")]
pub enum CrossCellOp {
    /// Z-score each gene of `layer` into layer `into`.
    Scale {
        /// The input layer.
        layer: String,
        /// The output layer.
        into: String,
        /// Whether to subtract each gene's mean (densifies).
        zero_center: bool,
        /// Clip scaled values to at most this, if set.
        max_value: Option<f64>,
    },
    /// Flag the highly variable genes of `layer`.
    HighlyVariableGenes {
        /// The input layer.
        layer: String,
        /// The method.
        flavor: HvgFlavor,
        /// How many genes to flag; `None` uses the flavour's cutoffs.
        n_top_genes: Option<u64>,
        /// A cell column: genes are ranked per batch, then combined.
        batch_key: Option<String>,
    },
    /// Principal components of `layer`.
    Pca {
        /// The input layer.
        layer: String,
        /// The cell embedding written (scanpy's `X_pca`).
        into: String,
        /// The number of components.
        n_components: u64,
        /// The random seed (R7: required, never global).
        seed: u64,
    },
}

/// A step that needs statistics over all cells before it outputs anything.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossCell {
    op: CrossCellOp,
}

impl CrossCell {
    /// The operation and its parameters.
    pub fn op(&self) -> &CrossCellOp {
        &self.op
    }
}

/// The name of the gene embedding a PCA into `into` writes its loadings to.
pub fn loadings_name(into: &str) -> String {
    format!("{into}_loadings")
}

/// The gene columns `highly_variable_genes` writes, in order, with their dtypes.
pub fn hvg_columns(flavor: HvgFlavor, batched: bool) -> Vec<ColumnSchema> {
    let float = |name: &str| ColumnSchema::new(name, ColumnDtype::Float64);
    let mut columns = vec![ColumnSchema::new("highly_variable", ColumnDtype::Boolean)];
    match flavor {
        HvgFlavor::Seurat | HvgFlavor::CellRanger => {
            columns.extend([
                float("means"),
                float("dispersions"),
                float("dispersions_norm"),
            ]);
        }
        HvgFlavor::SeuratV3 => {
            columns.extend([
                float("highly_variable_rank"),
                float("means"),
                float("variances"),
                float("variances_norm"),
            ]);
        }
    }
    if batched {
        columns.push(ColumnSchema::new(
            "highly_variable_nbatches",
            ColumnDtype::Int64,
        ));
        if flavor != HvgFlavor::SeuratV3 {
            columns.push(ColumnSchema::new(
                "highly_variable_intersection",
                ColumnDtype::Boolean,
            ));
        }
    }
    columns
}

impl LogicalNode for CrossCell {
    fn name(&self) -> &'static str {
        match self.op {
            CrossCellOp::Scale { .. } => "SCALE",
            CrossCellOp::HighlyVariableGenes { .. } => "HIGHLY_VARIABLE_GENES",
            CrossCellOp::Pca { .. } => "PCA",
        }
    }

    fn method(&self) -> &'static str {
        match self.op {
            CrossCellOp::Scale { .. } => "scale",
            CrossCellOp::HighlyVariableGenes { .. } => "highly_variable_genes",
            CrossCellOp::Pca { .. } => "pca",
        }
    }

    fn changes_axis(&self) -> Option<Axis> {
        None
    }

    fn cell_scope(&self) -> CellScope {
        CellScope::CrossCell
    }

    fn describe(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.op {
            CrossCellOp::Scale {
                layer,
                into,
                zero_center,
                max_value,
            } => {
                write!(formatter, "{layer:?} -> {into:?} zero_center={zero_center}")?;
                if let Some(max_value) = max_value {
                    write!(formatter, " max_value={max_value}")?;
                }
                Ok(())
            }
            CrossCellOp::HighlyVariableGenes {
                layer,
                flavor,
                n_top_genes,
                batch_key,
            } => {
                write!(formatter, "{layer:?} flavor={}", flavor.name())?;
                if let Some(n_top_genes) = n_top_genes {
                    write!(formatter, " n_top_genes={n_top_genes}")?;
                }
                if let Some(batch_key) = batch_key {
                    write!(formatter, " batch_key={batch_key:?}")?;
                }
                Ok(())
            }
            CrossCellOp::Pca {
                layer,
                into,
                n_components,
                seed,
            } => write!(
                formatter,
                "{layer:?} -> {into:?} n_components={n_components} seed={seed}"
            ),
        }
    }
}

impl Transform for CrossCell {
    fn output_schema(&self, input: &PlanSchema) -> PlanSchema {
        match &self.op {
            CrossCellOp::Scale {
                layer,
                into,
                zero_center,
                ..
            } => {
                let (nnz, exact) = if *zero_center {
                    let bound = input
                        .axis_len_bound(Axis::Cells)
                        .saturating_mul(input.axis_len_bound(Axis::Genes));
                    // A bound: a value equal to its gene's mean scales to 0.
                    (bound, false)
                } else {
                    let source = SparseMember::Layer(layer.clone());
                    let nnz = input.layer(layer).map_or(0, |layer| layer.nnz);
                    (nnz, input.nnz_is_exact(&source))
                };
                let layer = LayerSchema {
                    name: into.clone(),
                    dtype: LayerDtype::Float32,
                    nnz,
                };
                input.clone().with_layer(layer, exact)
            }
            CrossCellOp::HighlyVariableGenes {
                flavor, batch_key, ..
            } => hvg_columns(*flavor, batch_key.is_some())
                .into_iter()
                .fold(input.clone(), |schema, column| {
                    schema.with_column(Axis::Genes, column)
                }),
            CrossCellOp::Pca {
                into, n_components, ..
            } => input
                .clone()
                .with_embedding(EmbeddingSchema {
                    name: into.clone(),
                    axis: Axis::Cells,
                    width: *n_components,
                    dtype: FloatDtype::Float32,
                })
                .with_embedding(EmbeddingSchema {
                    name: loadings_name(into),
                    axis: Axis::Genes,
                    width: *n_components,
                    dtype: FloatDtype::Float32,
                }),
        }
    }

    fn writes_layer(&self) -> Option<&str> {
        match &self.op {
            CrossCellOp::Scale { into, .. } => Some(into),
            CrossCellOp::HighlyVariableGenes { .. } | CrossCellOp::Pca { .. } => None,
        }
    }
}

impl Plan {
    /// This plan with each gene of `layer` (or the default layer) z-scored into layer
    /// `into`: centred when `zero_center` (dense), then clipped to `max_value` if set.
    ///
    /// # Errors
    /// `KY-E109` for a layer not in the plan; `KY-E211` for no layer and no default
    /// layer; `KY-E215` for a `max_value` that is not a positive finite number or an
    /// empty `into`; `KY-E216` if this plan ends in a sink.
    pub fn scale(
        self: &Arc<Self>,
        layer: Option<&str>,
        into: &str,
        zero_center: bool,
        max_value: Option<f64>,
    ) -> Result<Arc<Self>, KytoError> {
        self.check_open("scale")?;
        let layer = self.input_layer("scale", layer)?;
        Self::check_name("scale", "into", into)?;
        if let Some(max_value) = max_value {
            Self::check_positive("scale", "max_value", max_value)?;
        }
        self.cross_cell(CrossCellOp::Scale {
            layer,
            into: into.to_owned(),
            zero_center,
            max_value,
        })
    }

    /// This plan with the highly variable genes of `layer` (or the default layer)
    /// flagged in new gene columns (see the module doc for which).
    ///
    /// # Errors
    /// `KY-E109` and `KY-E211` for the layer as [`Plan::scale`]; `KY-E215` for
    /// `n_top_genes = 0`; any error of [`Plan::resolve`] for
    /// `cell_column(batch_key)`; `KY-E216` if this plan ends in a sink.
    pub fn highly_variable_genes(
        self: &Arc<Self>,
        layer: Option<&str>,
        flavor: HvgFlavor,
        n_top_genes: Option<u64>,
        batch_key: Option<&str>,
    ) -> Result<Arc<Self>, KytoError> {
        let method = "highly_variable_genes";
        self.check_open(method)?;
        let layer = self.input_layer(method, layer)?;
        if n_top_genes == Some(0) {
            return Err(Self::invalid_argument(
                method,
                "n_top_genes",
                "0",
                "no gene would be flagged",
                "a positive int, or None for the flavour's cutoffs",
            ));
        }
        let batch_key = match batch_key {
            Some(name) => {
                let resolved = self.resolve(&CellExpr::column(name))?;
                match resolved.node().as_ref() {
                    ExprNode::Source(Source::Column { name }) => Some(name.clone()),
                    _ => Some(name.to_owned()),
                }
            }
            None => None,
        };
        self.cross_cell(CrossCellOp::HighlyVariableGenes {
            layer,
            flavor,
            n_top_genes,
            batch_key,
        })
    }

    /// This plan with `n_components` principal components of `layer` (or the default
    /// layer) as the cell embedding `into`, and their loadings as the gene embedding
    /// `<into>_loadings`. `seed` is required (R7).
    ///
    /// # Errors
    /// `KY-E109` and `KY-E211` for the layer as [`Plan::scale`]; `KY-E215` for
    /// `n_components` of 0 or above the plan's bound of `min(n_cells, n_genes)`, or an
    /// empty `into`; `KY-E216` if this plan ends in a sink.
    pub fn pca(
        self: &Arc<Self>,
        layer: Option<&str>,
        into: &str,
        n_components: u64,
        seed: u64,
    ) -> Result<Arc<Self>, KytoError> {
        self.check_open("pca")?;
        let layer = self.input_layer("pca", layer)?;
        Self::check_name("pca", "into", into)?;
        let schema = self.output_schema();
        let limit = schema
            .axis_len_bound(Axis::Cells)
            .min(schema.axis_len_bound(Axis::Genes));
        if n_components == 0 || n_components > limit {
            let problem = if n_components == 0 {
                "a PCA needs at least one component".to_owned()
            } else {
                format!(
                    "the plan has at most {limit} cells or genes, the most components a PCA has"
                )
            };
            return Err(Self::invalid_argument(
                "pca",
                "n_components",
                &n_components.to_string(),
                &problem,
                &format!("an int from 1 to {limit}"),
            ));
        }
        self.cross_cell(CrossCellOp::Pca {
            layer,
            into: into.to_owned(),
            n_components,
            seed,
        })
    }

    fn cross_cell(self: &Arc<Self>, op: CrossCellOp) -> Result<Arc<Self>, KytoError> {
        Ok(self.extend(CrossCell { op }, Self::CrossCell))
    }
}
