//! kyto-exec: the streaming CPU executor.
//!
//! Does: run physical plans over a stream of `CellBlock`s within a memory limit,
//! deterministically, with SIMD kernels. Blocks take their 64-byte aligned buffers from
//! a [`CellBlockPool`] (task 5.1). [`Pipeline`] (task 5.2) lowers one output of a
//! physical plan to a scan and a chain of streamed operators and runs it as a pull-based
//! [`BlockStream`]: rayon over blocks on a pool of the threads asked for
//! ([`ExecuteOptions`]), a bounded channel for backpressure, the same blocks on any
//! thread count (R11). [`Kernels`] (task 5.3) are the SIMD kernels the operators run on
//! (runtime dispatch through `pulp`, the same bits on every instruction set).
//! Cross-cell nodes (task 5.4) run their statistics passes before the output's stream
//! (exactly the passes the physical plan gives them) and hand their results over as
//! [`Statistics`]; each pass folds its blocks in parallel and combines them in a tree
//! of fixed shape in block order (task 5.6), the same bits on any thread count and any
//! block size. A memory limit (task 5.5, [`fit_memory_limit`]) bounds the
//! process's resident memory: the plan's blocks shrink until its estimate, with the
//! allocator's overhead, fits beside what the process already holds, or `KY-E401`.
//! A run is cancelled and followed through its [`RunControl`] (task 5.7): it stops at
//! its next block with `KY-E503`, and counts the cell rows it has computed.
//! Filters, `select_genes`, `take_cells` and computed columns (task 5.8) run in the
//! stream: cell steps on every block (with per-cell QC metrics, [`CellQc`], task 6.1),
//! gene steps once before the output's stream, leaving the genes present ([`Genes`]).
//! A plan ending in a sink (task 5.9) is written by [`Pipeline::sink`]: its stream into
//! kyto-store's `StoreWriter` (an `.h5ad` through a temporary store and `sink_h5ad`).
//! Highly variable genes (task 6.3) take their moments from those passes (with
//! batches, `seurat_v3`'s loess fit, [`loess_fitted`]) and write their gene columns
//! into the gene table later steps read. `scale` (task 6.4) takes per-gene moments
//! from one pass and z-scores the layer in its apply pass ([`ScaleStatistics`]).
//! `neighbors` (task 6.6) searches the embedding of the `pca` below it, exactly or
//! through an HNSW index, and leaves its distance graph and its UMAP connectivities
//! (task 6.7) as [`NeighborStatistics`] ([`nearest_neighbors`] runs the same search
//! on any embedding, [`connectivities`] weights any neighbour graph). `leiden` and
//! `louvain` (task 6.8) cluster the graph of the `neighbors` node below them
//! ([`cluster`] on any [`CellGraph`]) and add their labels as a cell column
//! ([`ClusterStatistics`]). `umap` (task 6.9) lays out a `neighbors` node's
//! connectivities ([`UmapStatistics`]; [`umap_layout`] lays out any connectivity
//! graph).
//! Does not: build or optimize plans, or implement algorithms (see kyto-algo).
//!
//! `unsafe` is allowed here, only in kernels, each block with a `// SAFETY:` note.

mod block_stream;
mod cell_block_pool;
mod cell_steps;
mod cluster_graph;
mod cluster_labels;
mod connectivities;
mod cross_cell;
mod elementwise;
mod expression;
mod gene_steps;
mod hvg;
mod hvg_pass;
mod kernels;
mod leiden;
mod loess;
mod memory_limit;
mod neighbors;
mod neighbors_hnsw;
mod operator;
mod pca;
mod pca_dense;
mod pipeline;
mod pipeline_block;
mod qc;
mod recipe;
mod reduction;
mod run_control;
mod run_members;
mod scale;
mod scrublet;
mod scrublet_pass;
mod scrublet_threshold;
mod sink;
mod statistics;
mod umap;
mod umap_spectral;

pub use block_stream::BlockStream;
pub use cell_block_pool::CellBlockPool;
pub use cluster_graph::CellGraph;
pub use connectivities::{
    ConnectivityGraph, MIN_K_DIST_SCALE, SMOOTH_K_ITERATIONS, SMOOTH_K_TOLERANCE, connectivities,
};
pub use gene_steps::Genes;
pub use kernels::{KernelValue, Kernels, SUM_LANES, combine_lanes, log1p};
pub use leiden::{ClusterStatistics, Clustering, cluster, partition_quality};
pub use loess::loess_fitted;
pub use memory_limit::{
    ALLOCATOR_OVERHEAD_PERCENT, fit_memory_limit, fit_memory_limit_with, process_resident_bytes,
};
pub use neighbors::{NeighborGraph, NeighborStatistics, nearest_neighbors};
pub use neighbors_hnsw::{
    CANDIDATES_PER_NEIGHBOR, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, HNSW_MAX_CONNECTIONS,
};
pub use pca::PcaStatistics;
pub use pipeline::{ExecuteOptions, Pipeline};
pub use pipeline_block::PipelineBlock;
pub use qc::{CellQc, GeneSet, QcValues};
pub use run_control::{Progress, RunControl};
pub use scrublet::{Doublets, ScrubletBatch, ScrubletResult, scrublet};
pub use scrublet_pass::ScrubletStatistics;
pub use statistics::{
    BatchStatistics, ClippedMoments, GeneMoments, HighlyVariableGeneStatistics, HvgValues,
    NodeStatistics, ScaleStatistics, Statistics,
};
pub use umap::{UMAP_SMALL_GRAPH_CELLS, UmapLayout, UmapStatistics, find_ab_params, umap_layout};
// The block type lives in kyto-store (the tile reader builds it, the writer consumes
// it); the executor names it from here.
pub use kyto_store::{
    AlignedBuffer, BUFFER_ALIGNMENT, CellBlock, CellBlockParts, CellData, CellValues, ValueDtype,
};
