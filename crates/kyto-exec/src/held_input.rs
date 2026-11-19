//! Held inputs (task 5.12): a node with several statistics passes reads its input from
//! the store once, holds it in memory as CSR blocks, and makes every pass over them.
//!
//! Does: stream a cross-cell node's input once through the stages below it
//! ([`HeldInput::read`]: the store's tiles decoded, genes selected, maps applied once),
//! keep each block's rows of the node's input layer in a compact copy (exactly the
//! entries, no growth slack: the size 4.6's estimate counts), and run a statistics pass
//! over the held blocks ([`HeldInput::reduce_pass`]) with the leaves and the tree a
//! streamed pass uses (`reduction.rs`, task 5.6): one leaf per store cell block, its
//! blocks folded in cell order (the parts of a split cell block continue it), the
//! leaves in a [`BlockOrderTree`] in block order, folded in parallel over cell blocks.
//! A leaf's blocks go to the accumulator as one group (`Accumulator::fold_group`), so
//! `pca` splits a whole cell block's rows and genes over the threads (task 5.15), not
//! the small parts 5.14 cuts on many threads one after another.
//! The blocks are the ones a streamed pass would fold, in the same order, so every pass
//! gives the same bits as streaming it (R11), on any thread count.
//! Does not: choose which nodes hold their input (the physical plan does, by the memory
//! limit: `kyto_plan::Execution::holds_input`), hold anything but the node's input
//! layer, or feed the output's stream (an apply pass still streams from the store).

use kyto_error::KytoError;
use rayon::prelude::*;
use zarrs::storage::ReadableStorageTraits;

use crate::block_stream::{StreamSource, Windows};
use crate::cell_block_pool::CellBlockPool;
use crate::cross_cell::{Accumulator, Partial};
use crate::pipeline_block::PipelineBlock;
use crate::reduction::BlockOrderTree;
use crate::run_control::RunControl;

/// A node's input held in memory: the stream's blocks in cell order, each with only the
/// input layer.
pub(crate) struct HeldInput {
    blocks: Vec<PipelineBlock>,
    layer: String,
}

impl HeldInput {
    /// Stream `source` once on `thread_pool` and keep each block's `layer` (a compact
    /// copy; the stream's buffers go back to its pool).
    ///
    /// # Errors
    /// The stream's errors (opening, reading, the operators); `KY-E109` if a block lacks
    /// `layer` (a kyto bug, reported).
    pub(crate) fn read<S>(
        source: &StreamSource<S>,
        thread_pool: &rayon::ThreadPool,
        layer: &str,
    ) -> Result<Self, KytoError>
    where
        S: ?Sized + ReadableStorageTraits + Send + Sync + 'static,
    {
        thread_pool.install(|| {
            // As reduce_blocks: the window's blocks and one.
            let layers_per_block = source.layers.len() + source.operators.len();
            let pool = CellBlockPool::new(
                (source.window.get().saturating_add(1)).saturating_mul(layers_per_block),
            );
            let mut windows = Windows::open(source, &pool)?;
            let mut blocks = Vec::new();
            while let Some(items) = windows.next_window() {
                for item in items {
                    let mut block = item?;
                    let held = block.layers.get(layer).cloned().ok_or_else(|| {
                        KytoError::MissingMember {
                            kind: "layer".to_owned(),
                            name: layer.to_owned(),
                            store: source.store.clone(),
                            suggestions: Box::default(),
                        }
                    })?;
                    for (_, buffers) in std::mem::take(&mut block.layers) {
                        pool.recycle(buffers);
                    }
                    block.layers.insert(layer.to_owned(), held);
                    blocks.push(block);
                }
            }
            Ok(Self {
                blocks,
                layer: layer.to_owned(),
            })
        })
    }

    /// Run statistics pass `pass` of `accumulator` over the held blocks on
    /// `thread_pool`: the same leaves, fold order and tree as `reduction::reduce_pass`
    /// streaming them, so the same bits. `None` without blocks. Stops at a cancel in
    /// `control` (the caller counts the pass's progress). Cell blocks are folded in
    /// parallel, as many at once as the pool has threads, so at most that many leaves
    /// and the tree's subtrees are held at once (as a streamed pass holds a window's).
    ///
    /// # Errors
    /// `KY-E503` if the run was cancelled; the fold's `KY-E117`.
    pub(crate) fn reduce_pass(
        &self,
        thread_pool: &rayon::ThreadPool,
        accumulator: &Accumulator,
        pass: u32,
        control: &RunControl,
    ) -> Result<Option<Partial>, KytoError> {
        let sequential = accumulator.sequential(pass);
        // Consecutive blocks of one store cell block are one leaf (all of them, for a
        // sequential pass), as reduce_blocks groups them.
        let mut groups: Vec<&[PipelineBlock]> = Vec::new();
        let mut start = 0;
        for end in 1..=self.blocks.len() {
            let ends = end == self.blocks.len()
                || !sequential && self.blocks[end].cell_block != self.blocks[start].cell_block;
            if ends {
                groups.push(&self.blocks[start..end]);
                start = end;
            }
        }
        let mut tree = BlockOrderTree::new(Partial::combine);
        for chunk in groups.chunks(thread_pool.current_num_threads().max(1)) {
            let leaves = thread_pool.install(|| {
                chunk
                    .par_iter()
                    .map(|group| {
                        control.check()?;
                        let mut leaf = accumulator.leaf(pass);
                        // Present: `read` kept it in every block.
                        let blocks: Vec<_> = group
                            .iter()
                            .map(|block| (&block.layers[&self.layer], block))
                            .collect();
                        accumulator.fold_group(pass, &mut leaf, &blocks)?;
                        Ok(leaf)
                    })
                    .collect::<Result<Vec<_>, KytoError>>()
            })?;
            for leaf in leaves {
                tree.push(leaf);
            }
        }
        Ok(tree.finish())
    }
}
