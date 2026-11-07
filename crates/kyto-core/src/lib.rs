//! kyto-core: the data model checked at plan build time.
//!
//! Does: schema types, typed `CellExpr` / `GeneExpr` expressions, the feature
//! name resolver and the read-only handles.
//! Does not: read matrix data or execute plans.
//!
//! So far: [`AtlasSchema`], the typed contents of a store's `schema.json` (task 3.1).
#![forbid(unsafe_code)]

mod schema;
mod schema_column;

pub use schema::{
    AtlasSchema, Axis, DEFAULT_CELL_BLOCK_SIZE, DEFAULT_GENE_BLOCK_SIZE, EmbeddingSchema,
    FloatDtype, GraphSchema, LayerDtype, LayerSchema, ModalitySchema, TableSchema, Tiling,
};
pub use schema_column::{ColumnDtype, ColumnSchema, Lineage};
