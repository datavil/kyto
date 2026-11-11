//! kyto-core: the data model checked at plan build time.
//!
//! Does: schema types, typed `CellExpr` / `GeneExpr` expressions and the feature
//! name resolver. The read-only handles (task 3.6) hold a plan, so they live in
//! kyto-plan.
//! Does not: read matrix data or execute plans.
//!
//! So far: [`AtlasSchema`], the typed contents of a store's `schema.json` (task 3.1);
//! the typed expressions [`CellExpr`] and [`GeneExpr`] with their [`ExprDtype`] and
//! [`Source`] tag, lowered to Polars when they read metadata only (task 3.3); the
//! parsed [`FeatureName`] of `feature(...)` (task 3.4); the [`Resolver`] that checks every
//! name against a plan's schema and [`GeneIds`] and stores it in full (task 3.5); the
//! missing-value rules of R9 (task 3.7): null tests on expressions ([`NullTest`]) and
//! the [`RemovedGenes`] a resolver refuses to read (`KY-E203`). The format's name rules
//! ([`member_name_problem`], [`column_name_problem`]), shared by store and plan (task 4.2).
#![forbid(unsafe_code)]

mod expr;
mod expr_dtype;
mod expr_polars;
mod expr_source;
mod feature_name;
mod gene_ids;
mod member_name;
mod removed_genes;
mod resolve;
mod schema;
mod schema_column;

pub use expr::{
    BinaryOp, CellAxis, CellExpr, Expr, ExprAxis, ExprNode, GeneAxis, GeneExpr, Literal,
    MAX_EXPR_DEPTH, NullTest, Operand, StrFunction, mixed_axes,
};
pub use expr_dtype::ExprDtype;
pub use expr_source::Source;
pub use feature_name::FeatureName;
pub use gene_ids::GeneIds;
pub use member_name::{MAX_MEMBER_NAME_BYTES, column_name_problem, member_name_problem};
pub use removed_genes::RemovedGenes;
pub use resolve::{META_PREFIX, Resolver, SYMBOL_COLUMN};
pub use schema::{
    AtlasSchema, Axis, DEFAULT_CELL_BLOCK_SIZE, DEFAULT_GENE_BLOCK_SIZE, EmbeddingSchema,
    FloatDtype, GraphSchema, LayerDtype, LayerSchema, ModalitySchema, TableSchema, Tiling,
};
pub use schema_column::{ColumnDtype, ColumnSchema, Lineage};
