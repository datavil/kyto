//! The read-only handles on one part of a plan (task 3.6): [`Table`], [`Matrix`],
//! [`Embedding`], [`Graph`] and [`Modality`], and the [`Estimate`] they give.
//!
//! Does: hold the plan a handle was taken from (shared, never changed: R2) and the
//! member it names, checked against the plan's schema when the handle is built
//! (`KY-E109`); answer `names()`, `schema()` and `estimate()` from the schema and the
//! gene IDs the scan read, without reading any data.
//! Does not: run terminals (`to_*` need the executor, section 5, and the data hand-off,
//! 7.5), estimate cost (bytes read, peak memory, passes: task 4.6), or change anything:
//! a handle has no method that mutates (R2, checked by `tests/test_handles.py`).
//!
//! Handles live here, not in kyto-core as the repo map lists, for the reason `Plan`
//! does (task 3.2): a handle holds an `Arc<Plan>`, and kyto-plan depends on kyto-core.

use std::sync::Arc;

use kyto_core::{
    Axis, EmbeddingSchema, FloatDtype, GraphSchema, LayerDtype, LayerSchema, ModalitySchema,
    TableSchema,
};
use kyto_error::KytoError;

use crate::plan::Plan;

/// The column names of a graph's edge list (`Graph.names()`, `to_edges`).
pub const EDGE_COLUMNS: [&str; 3] = ["source", "target", "weight"];

/// What a handle's data looks like, from metadata alone (task 3.6).
///
/// `None` means the value is not known without running the plan (for example the
/// number of cells after a filter, task 4.x) or has no meaning for the handle. The
/// cost estimate (bytes read, peak memory, passes, device) is task 4.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Estimate {
    /// Rows: cells, genes, or a graph's cells.
    pub n_rows: Option<u64>,
    /// Columns: table columns, genes, embedding components, a graph's cells, or a
    /// modality's features.
    pub n_columns: Option<u64>,
    /// Stored entries of a sparse matrix or graph.
    pub nnz: Option<u64>,
    /// Bytes of a dense `n_rows x n_columns` array of the stored value dtype: what
    /// `to_dense` (a matrix) or `to_numpy` (an embedding) would allocate. A `u128`
    /// that saturates at `u128::MAX` (only for axes near `u64::MAX`), never wraps.
    pub dense_bytes: Option<u128>,
}

/// Bytes of a dense `rows x columns` array of `item_bytes`-byte values, saturating.
/// `rows * columns` fits a `u128` for any two `u64`.
fn dense_bytes(rows: u64, columns: u64, item_bytes: u64) -> u128 {
    (u128::from(rows) * u128::from(columns)).saturating_mul(u128::from(item_bytes))
}

/// Bytes of one value of a float dtype.
fn float_bytes(dtype: FloatDtype) -> u64 {
    match dtype {
        FloatDtype::Float32 => 4,
        FloatDtype::Float64 => 8,
    }
}

/// The length of `axis` in `plan`.
fn axis_length(plan: &Plan, axis: Axis) -> u64 {
    match axis {
        Axis::Cells => plan.schema().n_cells,
        Axis::Genes => plan.schema().n_genes,
    }
}

/// `KY-E109` for a member of kind `kind` named `name` that `plan` does not have.
fn missing(plan: &Plan, kind: &str, name: &str) -> KytoError {
    KytoError::MissingMember {
        kind: kind.to_owned(),
        name: name.to_owned(),
        store: plan.root().source().to_owned(),
    }
}

/// The cell or the gene table of a plan.
#[derive(Clone)]
pub struct Table {
    plan: Arc<Plan>,
    axis: Axis,
}

impl Table {
    /// The table of `axis` (cells or genes) of `plan`.
    pub fn new(plan: &Arc<Plan>, axis: Axis) -> Self {
        Self {
            plan: Arc::clone(plan),
            axis,
        }
    }

    /// The plan the handle was taken from.
    pub fn plan(&self) -> &Arc<Plan> {
        &self.plan
    }

    /// Which table: cells or genes.
    pub fn axis(&self) -> Axis {
        self.axis
    }

    /// The table's columns, as `schema.json` lists them.
    pub fn schema(&self) -> &TableSchema {
        match self.axis {
            Axis::Cells => &self.plan.schema().cells,
            Axis::Genes => &self.plan.schema().genes,
        }
    }

    /// The column names, ID column first, in Parquet order.
    pub fn names(&self) -> Vec<String> {
        self.schema()
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect()
    }

    /// One row per cell or gene, one column per table column. No `dense_bytes`: string
    /// and `Enum` column sizes are not in the schema.
    pub fn estimate(&self) -> Estimate {
        Estimate {
            n_rows: Some(axis_length(&self.plan, self.axis)),
            n_columns: Some(self.schema().columns.len() as u64),
            nnz: None,
            dense_bytes: None,
        }
    }
}

/// A layer of a plan: a sparse `n_cells x n_genes` matrix.
#[derive(Clone)]
pub struct Matrix {
    plan: Arc<Plan>,
    layer: LayerSchema,
}

impl Matrix {
    /// The layer `name` of `plan`.
    ///
    /// # Errors
    /// `KY-E109` if the plan's schema has no layer named `name` (case-sensitive).
    pub fn new(plan: &Arc<Plan>, name: &str) -> Result<Self, KytoError> {
        let layer = plan
            .schema()
            .layers
            .iter()
            .find(|layer| layer.name == name)
            .ok_or_else(|| missing(plan, "layer", name))?
            .clone();
        Ok(Self {
            plan: Arc::clone(plan),
            layer,
        })
    }

    /// The plan the handle was taken from.
    pub fn plan(&self) -> &Arc<Plan> {
        &self.plan
    }

    /// The layer's entry in `schema.json`.
    pub fn schema(&self) -> &LayerSchema {
        &self.layer
    }

    /// The column names: the gene IDs, in gene table order.
    pub fn names(&self) -> &[String] {
        self.plan.root().gene_ids().ordered()
    }

    /// `n_cells x n_genes`, the stored entries, and the bytes of the dense array.
    pub fn estimate(&self) -> Estimate {
        let schema = self.plan.schema();
        let item_bytes = match self.layer.dtype {
            LayerDtype::Float32 | LayerDtype::Uint32 => 4,
        };
        Estimate {
            n_rows: Some(schema.n_cells),
            n_columns: Some(schema.n_genes),
            nnz: Some(self.layer.nnz),
            dense_bytes: Some(dense_bytes(schema.n_cells, schema.n_genes, item_bytes)),
        }
    }
}

/// An embedding of a plan: a dense array, one row per cell or gene.
#[derive(Clone)]
pub struct Embedding {
    plan: Arc<Plan>,
    embedding: EmbeddingSchema,
}

impl Embedding {
    /// The embedding `name` of `plan`.
    ///
    /// # Errors
    /// `KY-E109` if the plan's schema has no embedding named `name` (case-sensitive).
    pub fn new(plan: &Arc<Plan>, name: &str) -> Result<Self, KytoError> {
        let embedding = plan
            .schema()
            .embeddings
            .iter()
            .find(|embedding| embedding.name == name)
            .ok_or_else(|| missing(plan, "embedding", name))?
            .clone();
        Ok(Self {
            plan: Arc::clone(plan),
            embedding,
        })
    }

    /// The plan the handle was taken from.
    pub fn plan(&self) -> &Arc<Plan> {
        &self.plan
    }

    /// The embedding's entry in `schema.json`.
    pub fn schema(&self) -> &EmbeddingSchema {
        &self.embedding
    }

    /// The column names: `<embedding>:<component>` for each component, from 0
    /// (the full-name form of docs/feature_names.md).
    pub fn names(&self) -> Vec<String> {
        (0..self.embedding.width)
            .map(|component| format!("{}:{component}", self.embedding.name))
            .collect()
    }

    /// One row per cell or gene, one column per component, and the dense bytes.
    pub fn estimate(&self) -> Estimate {
        let rows = axis_length(&self.plan, self.embedding.axis);
        let width = self.embedding.width;
        Estimate {
            n_rows: Some(rows),
            n_columns: Some(width),
            nnz: None,
            dense_bytes: Some(dense_bytes(rows, width, float_bytes(self.embedding.dtype))),
        }
    }
}

/// A graph of a plan: a sparse `n_cells x n_cells` matrix of edge weights.
#[derive(Clone)]
pub struct Graph {
    plan: Arc<Plan>,
    graph: GraphSchema,
}

impl Graph {
    /// The graph `name` of `plan`.
    ///
    /// # Errors
    /// `KY-E109` if the plan's schema has no graph named `name` (case-sensitive).
    pub fn new(plan: &Arc<Plan>, name: &str) -> Result<Self, KytoError> {
        let graph = plan
            .schema()
            .graphs
            .iter()
            .find(|graph| graph.name == name)
            .ok_or_else(|| missing(plan, "graph", name))?
            .clone();
        Ok(Self {
            plan: Arc::clone(plan),
            graph,
        })
    }

    /// The plan the handle was taken from.
    pub fn plan(&self) -> &Arc<Plan> {
        &self.plan
    }

    /// The graph's entry in `schema.json`.
    pub fn schema(&self) -> &GraphSchema {
        &self.graph
    }

    /// The column names of its edge list: [`EDGE_COLUMNS`].
    pub fn names(&self) -> Vec<String> {
        EDGE_COLUMNS.iter().map(|&name| name.to_owned()).collect()
    }

    /// `n x n` over its axis, the stored edges, and the bytes of the dense array.
    pub fn estimate(&self) -> Estimate {
        let n = axis_length(&self.plan, self.graph.axis);
        Estimate {
            n_rows: Some(n),
            n_columns: Some(n),
            nnz: Some(self.graph.nnz),
            dense_bytes: Some(dense_bytes(n, n, float_bytes(self.graph.dtype))),
        }
    }
}

/// A modality of a plan: a second feature space on the same cells.
///
/// Minimal: format v0 stores no modalities and `ModalitySchema` is provisional (3.1),
/// so a v0 plan has none and [`Modality::new`] always fails on one.
#[derive(Clone)]
pub struct Modality {
    plan: Arc<Plan>,
    modality: ModalitySchema,
}

impl Modality {
    /// The modality `name` of `plan`.
    ///
    /// # Errors
    /// `KY-E109` if the plan's schema has no modality named `name` (case-sensitive).
    pub fn new(plan: &Arc<Plan>, name: &str) -> Result<Self, KytoError> {
        let modality = plan
            .schema()
            .modalities
            .iter()
            .find(|modality| modality.name == name)
            .ok_or_else(|| missing(plan, "modality", name))?
            .clone();
        Ok(Self {
            plan: Arc::clone(plan),
            modality,
        })
    }

    /// The plan the handle was taken from.
    pub fn plan(&self) -> &Arc<Plan> {
        &self.plan
    }

    /// The modality's schema.
    pub fn schema(&self) -> &ModalitySchema {
        &self.modality
    }

    /// The names of its layers.
    pub fn names(&self) -> Vec<String> {
        self.modality
            .layers
            .iter()
            .map(|layer| layer.name.clone())
            .collect()
    }

    /// One row per cell, one column per feature.
    pub fn estimate(&self) -> Estimate {
        Estimate {
            n_rows: Some(self.plan.schema().n_cells),
            n_columns: Some(self.modality.n_features),
            nnz: None,
            dense_bytes: None,
        }
    }
}
