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
//! [`Matrix`], [`Embedding`], [`Graph`] and [`Modality`] (task 3.6). The plan JSON
//! (task 4.3): [`Plan::to_json`] and [`Plan::from_json`]. The optimizer (task 4.4):
//! [`optimize`] (and [`Plan::optimize`]) turns the plans of one or more [`Output`]s into
//! an [`OptimizedPlan`] by common sub-plan reuse, projection pushdown, elementwise
//! fusion and predicate pushdown, each its own `optimize_*.rs` file. The physical plan
//! (task 4.5): [`PhysicalPlan`] (and [`Plan::physical`], [`Plan::explain`]) says the
//! device, passes and blocks of each optimized node. The cost estimate (task 4.6):
//! [`CostEstimate`] ([`PhysicalPlan::estimate`], [`Plan::estimate`]) gives the passes
//! over the store, bytes read, peak memory and device without reading anything.
//! Explain (task 4.7): [`Explain`] ([`Plan::explanation`], [`Plan::explain`], [`Plan::explain_json`]) shows
//! the logical and optimized trees with resolved names, as text or JSON.
#![forbid(unsafe_code)]

mod cluster;
mod cross_cell;
mod estimate;
mod explain;
mod filter;
mod handles;
mod map_elementwise;
mod members;
mod names;
mod neighbors;
mod node;
mod optimize;
mod optimize_fusion;
mod optimize_predicate;
mod optimize_projection;
mod optimize_reuse;
mod optimized_plan;
mod physical_plan;
mod plan;
mod plan_json;
mod plan_schema;
mod scan;
mod select_genes;
mod sink;
mod take_cells;
mod with_columns;
mod with_default_layer;

pub use cluster::{Cluster, ClusterMethod};
pub use cross_cell::{CrossCell, CrossCellOp, HvgFlavor, hvg_columns, loadings_name};
pub use estimate::{ASSUMED_NNZ_CHUNK_SIZE, ASSUMED_NNZ_SHARD_SIZE, CostEstimate, PCA_OVERSAMPLES};
pub use explain::{Explain, LogicalEntry, OptimizedEntry, OutputEntry};
pub use filter::{Filter, Predicate};
pub use handles::{EDGE_COLUMNS, Embedding, Estimate, Graph, Matrix, Modality, Table};
pub use map_elementwise::{ElementwiseOp, MapElementwise};
pub use members::{Member, Members, all_members};
pub use neighbors::{Metric, NeighborSearch, Neighbors};
pub use node::{CellScope, LogicalNode, Node, NodeId, Transform};
pub use optimize::{OptimizeOptions, Output, optimize};
pub use optimized_plan::{Operation, OptimizedNode, OptimizedOutput, OptimizedPlan};
pub use physical_plan::{
    Blocks, DEFAULT_BLOCKS_IN_FLIGHT, Device, Execution, PCA_POWER_ITERATIONS, PhysicalOptions,
    PhysicalPlan, physical_plan,
};
pub use plan::{Plan, PlanNodes};
pub use plan_json::{PLAN_JSON_FORMAT, PLAN_JSON_VERSION};
pub use plan_schema::{PlanSchema, SparseMember};
pub use scan::Scan;
pub use select_genes::SelectGenes;
pub use sink::Sink;
pub use take_cells::TakeCells;
pub use with_columns::{Columns, WithColumns, column_dtype};
pub use with_default_layer::WithDefaultLayer;
