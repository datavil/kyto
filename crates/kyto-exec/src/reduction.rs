//! Deterministic reductions (task 5.6): a statistics pass folds its blocks in parallel
//! and combines them in a tree of fixed shape, in block order, never in the order
//! threads finish (R11).
//!
//! Does: run one statistics pass of a cross-cell node ([`reduce_pass`]) over the
//! stream's windows (read and computed in parallel, as the output's stream) and fold
//! each window's blocks into leaves in parallel. A leaf is one **store cell block**:
//! its rows folded one after another in cell order, starting from an empty leaf, so
//! the parts a memory limit splits a cell block into (task 5.5) continue the same leaf
//! (sequentially, even across windows) and give the same bits as the whole block.
//! Leaves combine in a [`BlockOrderTree`]: a binary-counter tree whose shape depends
//! only on the number of leaves (the store's cell blocks), not on threads, windows or
//! block sizes. So a pass gives the same bits on any thread count, any
//! `blocks_in_flight` and any memory limit; another cell block size (another tiling
//! of the store, which is the input) changes the leaves and the last bits of the sums.
//! A pass whose fold depends on the order of every cell before it (Welford's
//! moments of `seurat_v3` with batches, as scanpy's `aggregate`, task 6.3) asks for
//! one leaf ([`Accumulator::sequential`]): its blocks are folded one after another in
//! cell order (reading and the maps stay parallel).
//! Does not: choose what a leaf sums (see `cross_cell.rs`), or reduce within a block
//! (the kernels' fixed lanes, task 5.3, do that for row sums).

use kyto_error::KytoError;
use rayon::prelude::*;
use zarrs::storage::ReadableStorageTraits;

use crate::block_stream::{StreamSource, Windows};
use crate::cell_block_pool::CellBlockPool;
use crate::cross_cell::{Accumulator, Partial};
use crate::pipeline_block::PipelineBlock;

/// A reduction over leaves pushed in block order, in a tree of fixed shape: leaves
/// combine like a binary counter (two subtrees of the same size combine as soon as
/// the right one is complete), and [`BlockOrderTree::finish`] combines what is left
/// from the right. For `n` leaves the tree is: one leaf as it is, else the tree of the
/// first `p` leaves combined with the tree of the rest, `p` the largest power of two
/// below `n`. It holds at most `log2(n) + 1` subtrees at once.
pub(crate) struct BlockOrderTree<T> {
    /// Complete subtrees, left to right, each with its level (`2^level` leaves); the
    /// levels strictly decrease.
    stack: Vec<(u32, T)>,
    combine: fn(T, T) -> T,
}

impl<T> BlockOrderTree<T> {
    /// An empty tree combining `(left, right)` with `combine`.
    pub(crate) fn new(combine: fn(T, T) -> T) -> Self {
        Self {
            stack: Vec::new(),
            combine,
        }
    }

    /// Add the next leaf in block order.
    pub(crate) fn push(&mut self, leaf: T) {
        let (mut level, mut right) = (0, leaf);
        while let Some((top, _)) = self.stack.last()
            && *top == level
        {
            let Some((_, left)) = self.stack.pop() else {
                unreachable!("the stack has a top")
            };
            right = (self.combine)(left, right);
            level += 1;
        }
        self.stack.push((level, right));
    }

    /// The reduction of every leaf; `None` without leaves.
    pub(crate) fn finish(mut self) -> Option<T> {
        let (_, mut right) = self.stack.pop()?;
        while let Some((_, left)) = self.stack.pop() {
            right = (self.combine)(left, right);
        }
        Some(right)
    }
}

/// Run statistics pass `pass` of `accumulator` over `source`'s stream on
/// `thread_pool`: windows read and computed in parallel, their blocks of
/// `input_layer` folded into one leaf per store cell block (in parallel over cell
/// blocks), the leaves combined in a [`BlockOrderTree`]. `None` for a stream without
/// blocks. The same bits on any thread count and block size (module doc).
///
/// # Errors
/// The stream's errors (opening, reading, the operators); `KY-E109` if a block lacks
/// `input_layer` (a kyto bug, reported); the fold's `KY-E117`.
pub(crate) fn reduce_pass<S>(
    source: &StreamSource<S>,
    thread_pool: &rayon::ThreadPool,
    input_layer: &str,
    accumulator: &Accumulator,
    pass: u32,
) -> Result<Option<Partial>, KytoError>
where
    S: ?Sized + ReadableStorageTraits + Send + Sync + 'static,
{
    thread_pool.install(|| {
        // As the output's stream (BlockStream::start): the window's blocks and one.
        let layers_per_block = source.layers.len() + source.operators.len();
        let pool = CellBlockPool::new(
            (source.window.get().saturating_add(1)).saturating_mul(layers_per_block),
        );
        let mut windows = Windows::open(source, &pool)?;
        let sequential = accumulator.sequential(pass);
        let mut tree = BlockOrderTree::new(Partial::combine);
        // The last leaf of the window before, with its cell block: it goes on if the
        // next window starts with the same cell block (a split cell block).
        let mut open_leaf: Option<(u64, Partial)> = None;
        while let Some(items) = windows.next_window() {
            let blocks = items.into_iter().collect::<Result<Vec<_>, _>>()?;
            let mut groups: Vec<(u64, Vec<PipelineBlock>)> = Vec::new();
            for block in blocks {
                // An order-dependent fold (Welford's moments, task 6.3) is one leaf: every
                // block continues it, in cell order, as the parts of one cell block.
                let leaf_key = if sequential { 0 } else { block.cell_block };
                match groups.last_mut() {
                    Some((key, group)) if *key == leaf_key => {
                        group.push(block);
                    }
                    _ => groups.push((leaf_key, vec![block])),
                }
            }
            let mut starts = Vec::with_capacity(groups.len());
            if let Some((cell_block, leaf)) = open_leaf.take() {
                if groups
                    .first()
                    .is_some_and(|(first, _)| *first == cell_block)
                {
                    starts.push(leaf);
                } else {
                    tree.push(leaf);
                }
            }
            while starts.len() < groups.len() {
                starts.push(accumulator.leaf());
            }
            let leaves = groups
                .into_par_iter()
                .zip(starts)
                .map(|((cell_block, blocks), mut leaf)| {
                    for block in blocks {
                        let layer = block.layers.get(input_layer).ok_or_else(|| {
                            KytoError::MissingMember {
                                kind: "layer".to_owned(),
                                name: input_layer.to_owned(),
                                store: source.store.clone(),
                                suggestions: Box::default(),
                            }
                        })?;
                        accumulator.fold(pass, &mut leaf, layer, &block)?;
                        for (_, layer) in block.layers {
                            pool.recycle(layer);
                        }
                    }
                    Ok((cell_block, leaf))
                })
                .collect::<Result<Vec<_>, KytoError>>()?;
            // Every leaf but the last ended in this window; the last may go on.
            let mut leaves = leaves.into_iter();
            open_leaf = leaves.next_back();
            for (_, leaf) in leaves {
                tree.push(leaf);
            }
        }
        if let Some((_, leaf)) = open_leaf {
            tree.push(leaf);
        }
        Ok(tree.finish())
    })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::BlockOrderTree;

    /// The tree's shape written as text: `(left right)`.
    fn shape(left: String, right: String) -> String {
        format!("({left} {right})")
    }

    /// The documented shape, recursively.
    fn reference(leaves: &[String]) -> String {
        if leaves.len() == 1 {
            return leaves[0].clone();
        }
        let mut split = 1;
        while split * 2 < leaves.len() {
            split *= 2;
        }
        shape(reference(&leaves[..split]), reference(&leaves[split..]))
    }

    fn reduce(leaves: &[String]) -> Option<String> {
        let mut tree = BlockOrderTree::new(shape);
        for leaf in leaves {
            tree.push(leaf.clone());
        }
        tree.finish()
    }

    #[test]
    fn the_shape_is_a_binary_counter() {
        let leaves: Vec<String> = (0..6).map(|leaf| leaf.to_string()).collect();
        assert_eq!(reduce(&leaves[..0]), None);
        assert_eq!(reduce(&leaves[..1]).unwrap(), "0");
        assert_eq!(reduce(&leaves[..3]).unwrap(), "((0 1) 2)");
        assert_eq!(reduce(&leaves[..4]).unwrap(), "((0 1) (2 3))");
        assert_eq!(reduce(&leaves[..6]).unwrap(), "(((0 1) (2 3)) (4 5))");
    }

    proptest! {
        #[test]
        fn the_shape_depends_only_on_the_number_of_leaves(n in 1_usize..300) {
            let leaves: Vec<String> = (0..n).map(|leaf| leaf.to_string()).collect();
            prop_assert_eq!(reduce(&leaves).unwrap(), reference(&leaves));
        }
    }
}
