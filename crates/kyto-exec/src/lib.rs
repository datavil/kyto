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
//! Does not: build or optimize plans, or implement algorithms (see kyto-algo).
//!
//! `unsafe` is allowed here, only in kernels, each block with a `// SAFETY:` note.

mod block_stream;
mod cell_block_pool;
mod elementwise;
mod kernels;
mod pipeline;
mod pipeline_block;

pub use block_stream::BlockStream;
pub use cell_block_pool::CellBlockPool;
pub use kernels::{KernelValue, Kernels, SUM_LANES, combine_lanes, log1p};
pub use pipeline::{ExecuteOptions, Pipeline};
pub use pipeline_block::PipelineBlock;
// The block type lives in kyto-store (the tile reader builds it, the writer consumes
// it); the executor names it from here.
pub use kyto_store::{
    AlignedBuffer, BUFFER_ALIGNMENT, CellBlock, CellBlockParts, CellData, CellValues, ValueDtype,
};
