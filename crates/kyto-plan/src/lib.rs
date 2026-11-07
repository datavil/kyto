//! kyto-plan: the lazy logical plan.
//!
//! Does: logical nodes, build-time validation, plan JSON, optimizer rules,
//! the physical plan, `explain` and `estimate`.
//! Does not: execute plans or read matrix data.
//!
//! So far (task 3.2): [`Plan`], a persistent chain of nodes shared through `Arc`, with
//! a [`Scan`] root carrying the store's `AtlasSchema` and one metadata node,
//! [`WithDefaultLayer`], and [`FilterGenes`] (4.1's `Filter` on genes, taken early for
//! task 3.8's stale-column check). The full node set arrives with task 4.1. `Plan::resolve`
//! resolves an expression's names against the plan (task 3.5, R6). The read-only
//! handles [`Table`], [`Matrix`], [`Embedding`], [`Graph`] and [`Modality`] (task 3.6).
#![forbid(unsafe_code)]

mod filter_genes;
mod handles;
mod plan;
mod scan;
mod with_default_layer;

pub use filter_genes::FilterGenes;
pub use handles::{EDGE_COLUMNS, Embedding, Estimate, Graph, Matrix, Modality, Table};
pub use plan::{Plan, PlanNodes};
pub use scan::Scan;
pub use with_default_layer::WithDefaultLayer;
