//! kyto-plan: the lazy logical plan.
//!
//! Does: logical nodes, build-time validation, plan JSON, optimizer rules,
//! the physical plan, `explain` and `estimate`.
//! Does not: execute plans or read matrix data.
//!
//! So far: [`Plan`] (task 3.2), a persistent chain of nodes shared through `Arc`, with
//! the logical nodes of task 4.1, one file each: [`Scan`] (the root), [`WithDefaultLayer`],
//! [`Filter`] (cells or genes), [`SelectGenes`], [`TakeCells`], [`MapElementwise`]
//! (`normalize_total`, `log1p`), [`CrossCell`] (`scale`, `highly_variable_genes`, `pca`),
//! [`WithColumns`], [`Neighbors`], [`Cluster`] and [`Sink`]. Each declares the axis it
//! changes and whether it is per-cell or cross-cell ([`LogicalNode`]) and computes its
//! output [`PlanSchema`] from its input's ([`Transform`]). `Plan::resolve` resolves an
//! expression's names against the plan (task 3.5, R6). The read-only handles [`Table`],
//! [`Matrix`], [`Embedding`], [`Graph`] and [`Modality`] (task 3.6).
#![forbid(unsafe_code)]

mod cluster;
mod cross_cell;
mod filter;
mod handles;
mod map_elementwise;
mod neighbors;
mod node;
mod plan;
mod plan_schema;
mod scan;
mod select_genes;
mod sink;
mod take_cells;
mod with_columns;
mod with_default_layer;

pub use cluster::{Cluster, ClusterMethod};
pub use cross_cell::{CrossCell, CrossCellOp, HvgFlavor, hvg_columns, loadings_name};
pub use filter::{Filter, Predicate};
pub use handles::{EDGE_COLUMNS, Embedding, Estimate, Graph, Matrix, Modality, Table};
pub use map_elementwise::{ElementwiseOp, MapElementwise};
pub use neighbors::{Metric, NeighborSearch, Neighbors};
pub use node::{CellScope, LogicalNode, Node, NodeId, Transform};
pub use plan::{Plan, PlanNodes};
pub use plan_schema::{PlanSchema, SparseMember};
pub use scan::Scan;
pub use select_genes::SelectGenes;
pub use sink::Sink;
pub use take_cells::TakeCells;
pub use with_columns::{Columns, WithColumns, column_dtype};
pub use with_default_layer::WithDefaultLayer;
