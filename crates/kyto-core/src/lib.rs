//! kyto-core: the data model checked at plan build time.
//!
//! Does: schema types, typed `CellExpr` / `GeneExpr` expressions, the feature
//! name resolver and the read-only handles.
//! Does not: read matrix data or execute plans.
//!
//! So far: [`AtlasSchema`], the typed contents of a store's `schema.json` (task 3.1);
//! the typed expressions [`CellExpr`] and [`GeneExpr`] with their [`ExprDtype`] and
//! [`Source`] tag, lowered to Polars when they read metadata only (task 3.3).
#![forbid(unsafe_code)]

mod expr;
mod expr_dtype;
mod expr_polars;
mod expr_source;
mod schema;
mod schema_column;

pub use expr::{
    BinaryOp, CellAxis, CellExpr, Expr, ExprAxis, ExprNode, GeneAxis, GeneExpr, Literal,
    MAX_EXPR_DEPTH, Operand, StrFunction, mixed_axes,
};
pub use expr_dtype::ExprDtype;
pub use expr_source::Source;
pub use schema::{
    AtlasSchema, Axis, DEFAULT_CELL_BLOCK_SIZE, DEFAULT_GENE_BLOCK_SIZE, EmbeddingSchema,
    FloatDtype, GraphSchema, LayerDtype, LayerSchema, ModalitySchema, TableSchema, Tiling,
};
pub use schema_column::{ColumnDtype, ColumnSchema, Lineage};
