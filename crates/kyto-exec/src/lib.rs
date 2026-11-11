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
//! Does not: build or optimize plans, or implement algorithms (see kyto-algo).
//!
//! `unsafe` is allowed here, only in kernels, each block with a `// SAFETY:` note.

mod block_stream;
mod cell_block_pool;
mod cell_steps;
mod cross_cell;
mod elementwise;
mod expression;
mod gene_steps;
mod kernels;
mod memory_limit;
mod operator;
mod pipeline;
mod pipeline_block;
mod qc;
mod reduction;
mod run_control;
mod sink;
mod statistics;

pub use block_stream::BlockStream;
pub use cell_block_pool::CellBlockPool;
pub use gene_steps::Genes;
pub use kernels::{KernelValue, Kernels, SUM_LANES, combine_lanes, log1p};
pub use memory_limit::{
    ALLOCATOR_OVERHEAD_PERCENT, fit_memory_limit, fit_memory_limit_with, process_resident_bytes,
};
pub use pipeline::{ExecuteOptions, Pipeline};
pub use pipeline_block::PipelineBlock;
pub use qc::{CellQc, GeneSet, QcValues};
pub use run_control::{Progress, RunControl};
pub use statistics::{
    ClippedMoments, GeneMoments, HighlyVariableGeneStatistics, NodeStatistics, Statistics,
};
// The block type lives in kyto-store (the tile reader builds it, the writer consumes
// it); the executor names it from here.
pub use kyto_store::{
    AlignedBuffer, BUFFER_ALIGNMENT, CellBlock, CellBlockParts, CellData, CellValues, ValueDtype,
};
