//! kyto-core: the data model checked at plan build time.
//!
//! Does: schema types, typed `CellExpr` / `GeneExpr` expressions, the feature
//! name resolver and the read-only handles.
//! Does not: read matrix data or execute plans.
#![forbid(unsafe_code)]
