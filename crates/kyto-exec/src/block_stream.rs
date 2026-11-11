//! The running stream of a [`crate::Pipeline`]: a producer thread that reads and
//! computes blocks, a bounded channel, and the caller pulling blocks from it.
//!
//! Does: on a thread of its own, inside a rayon pool of exactly the threads asked
//! for (R7), read `blocks_in_flight` cell blocks of every scanned layer at once (their
//! tiles in parallel, task 2.4), run the operators over those blocks in parallel (rayon
//! over blocks), and hand them over in cell order through a rendezvous channel
//! (`sync_channel(0)`): the producer waits until the caller takes each block, so a slow
//! consumer stops the reading (backpressure) and at most `blocks_in_flight + 1` blocks
//! are held, the window being computed and the one the caller holds. The order and the
//! values do not depend on the thread count (R11): each block is computed alone, and an
//! indexed parallel collect keeps cell order. After an error the stream ends; blocks
//! before it in cell order come first. Blocks the caller is done with go back to the
//! stream's [`CellBlockPool`] through [`BlockStream::recycle`]. Every pass of one run
//! (the statistics passes of task 5.4, then the output's stream) is a stream of its
//! own over the same opened readers and thread pool ([`StreamSource`]); the output's
//! stream carries the run's [`Statistics`].
//! Does not: cancel a window being computed when dropped (it finishes, then stops;
//! Ctrl-C is 5.7), or limit bytes (5.5).

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;

use kyto_error::KytoError;
use kyto_store::{AxisSelection, CellBlocks, LayerReader, TileGrid};
use rayon::prelude::*;
use zarrs::storage::ReadableStorageTraits;

use crate::cell_block_pool::CellBlockPool;
use crate::elementwise::ElementwiseMaps;
use crate::pipeline_block::PipelineBlock;
use crate::statistics::Statistics;

type Item = Result<PipelineBlock, KytoError>;

/// The blocks of a running pipeline, in cell order: an iterator the caller pulls.
///
/// Each item is a `Result`; after the first error the stream ends. Dropping the
/// stream stops the producer once its current window is computed.
#[derive(Debug)]
pub struct BlockStream {
    receiver: Option<Receiver<Item>>,
    producer: Option<JoinHandle<()>>,
    pool: Arc<CellBlockPool>,
    statistics: Statistics,
}

/// What one stream reads and runs: the scanned layers (shared by every pass of a
/// run), the per-cell operators over them and the layers its blocks keep.
pub(crate) struct StreamSource<S: ?Sized> {
    /// One opened reader per scanned layer; none streams the cells' rows only.
    pub(crate) readers: Arc<Vec<LayerReader<S>>>,
    /// The name of each reader's layer.
    pub(crate) layers: Vec<String>,
    pub(crate) grid: TileGrid,
    pub(crate) operators: Vec<ElementwiseMaps>,
    /// The layers each block keeps at the stream's end.
    pub(crate) output_layers: BTreeSet<String>,
    pub(crate) window: NonZeroUsize,
    pub(crate) store: String,
}

/// What the producer thread owns.
struct Producer<S: ?Sized> {
    source: StreamSource<S>,
    pool: Arc<CellBlockPool>,
    sender: SyncSender<Item>,
}

impl BlockStream {
    /// Start the producer thread of `source` on `thread_pool`.
    ///
    /// # Errors
    /// `KY-E502` if the producer thread cannot be started.
    pub(crate) fn start<S>(
        source: StreamSource<S>,
        thread_pool: Arc<rayon::ThreadPool>,
        statistics: Statistics,
    ) -> Result<Self, KytoError>
    where
        S: ?Sized + ReadableStorageTraits + Send + Sync + 'static,
    {
        // Each block in flight holds at most the scanned layers and one output per
        // operator; keep that many blocks' buffers for the window and the caller's block.
        let window = source.window;
        let layers_per_block = source.layers.len() + source.operators.len();
        let pool = Arc::new(CellBlockPool::new(
            (window.get().saturating_add(1)).saturating_mul(layers_per_block),
        ));
        let (sender, receiver) = sync_channel(0);
        let producer = Producer {
            source,
            pool: Arc::clone(&pool),
            sender,
        };
        let threads = thread_pool.current_num_threads();
        let handle = std::thread::Builder::new()
            .name("kyto-exec-producer".to_owned())
            .spawn(move || thread_pool.install(|| producer.run()))
            .map_err(|error| KytoError::ThreadStart {
                threads: threads as u64,
                problem: error.to_string(),
            })?;
        Ok(Self {
            receiver: Some(receiver),
            producer: Some(handle),
            pool,
            statistics,
        })
    }

    /// The statistics the cross-cell nodes below the output computed in their
    /// statistics passes, before this stream started (task 5.4).
    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }

    /// Give a block the caller is done with back, so its buffers serve later blocks.
    pub fn recycle(&self, block: PipelineBlock) {
        for (_, layer) in block.layers {
            self.pool.recycle(layer);
        }
    }

    /// The pool that later blocks take their buffers from.
    pub fn pool(&self) -> &CellBlockPool {
        &self.pool
    }

    /// Wait for the producer; re-raise its panic (a bug), if any.
    fn join_producer(&mut self) {
        if let Some(handle) = self.producer.take()
            && let Err(payload) = handle.join()
        {
            std::panic::resume_unwind(payload);
        }
    }
}

impl Iterator for BlockStream {
    type Item = Item;

    fn next(&mut self) -> Option<Item> {
        let receiver = self.receiver.as_ref()?;
        if let Ok(item) = receiver.recv() {
            return Some(item);
        }
        // The producer hung up: it is done, or it panicked.
        self.receiver = None;
        self.join_producer();
        None
    }
}

impl Drop for BlockStream {
    fn drop(&mut self) {
        // Hang up first, so a producer waiting to send stops.
        self.receiver = None;
        if let Some(handle) = self.producer.take() {
            // Never panic in drop; a producer panic was a bug already reported by
            // the panic hook.
            let _ = handle.join();
        }
    }
}

impl<S: ?Sized + ReadableStorageTraits + 'static> Producer<S> {
    /// Read, compute and send windows of blocks until the end, an error, or a caller
    /// that hung up.
    fn run(self) {
        let source = &self.source;
        let streams = source
            .readers
            .iter()
            .map(|reader| {
                reader.cell_blocks(&AxisSelection::All, &AxisSelection::All, source.window)
            })
            .collect::<Result<Vec<_>, _>>();
        let mut streams = match streams {
            Ok(streams) => streams,
            Err(error) => {
                let _ = self.sender.send(Err(error));
                return;
            }
        };
        let mut next_cell_block = 0;
        loop {
            let mut inputs = Vec::with_capacity(source.window.get());
            let mut failure = None;
            while inputs.len() < source.window.get() {
                match self.read_next(&mut streams, &mut next_cell_block) {
                    None => break,
                    Some(Ok(block)) => inputs.push(block),
                    Some(Err(error)) => {
                        failure = Some(error);
                        break;
                    }
                }
            }
            if inputs.is_empty() && failure.is_none() {
                return;
            }
            // Rayon over blocks; an indexed collect keeps cell order (R11).
            let outputs: Vec<Item> = inputs
                .into_par_iter()
                .map(|block| {
                    let mut block =
                        source.operators.iter().try_fold(block, |block, operator| {
                            operator.apply(block, &self.pool, &source.store)
                        })?;
                    // Keep only what the output reads; the rest goes back to the pool.
                    let unread: Vec<String> = block
                        .layers
                        .keys()
                        .filter(|layer| !source.output_layers.contains(*layer))
                        .cloned()
                        .collect();
                    for layer in unread {
                        if let Some(dropped) = block.layers.remove(&layer) {
                            self.pool.recycle(dropped);
                        }
                    }
                    Ok(block)
                })
                .collect();
            for item in outputs.into_iter().chain(failure.map(Err)) {
                let failed = item.is_err();
                if self.sender.send(item).is_err() || failed {
                    return;
                }
            }
        }
    }

    /// The next cell block of every scanned layer, joined; without layers, the next
    /// cell block's rows alone.
    fn read_next(
        &self,
        streams: &mut [CellBlocks<'_, S>],
        next_cell_block: &mut u64,
    ) -> Option<Item> {
        if streams.is_empty() {
            let block = *next_cell_block;
            let grid = self.source.grid;
            if block >= grid.n_cell_blocks() {
                return None;
            }
            *next_cell_block += 1;
            let first = block * grid.cell_block_size();
            return Some(Ok(PipelineBlock {
                cell_block: block,
                cells: (first..first + grid.rows(block)).collect(),
                layers: BTreeMap::new(),
            }));
        }
        let mut layers = BTreeMap::new();
        let mut rows = None;
        // Every layer's stream has the same cell blocks (same selection, same grid).
        for (stream, layer) in streams.iter_mut().zip(&self.source.layers) {
            let selected = match stream.next()? {
                Ok(selected) => selected,
                Err(error) => return Some(Err(error)),
            };
            rows.get_or_insert((selected.cell_block, selected.cells));
            layers.insert(layer.clone(), selected.block);
        }
        let (cell_block, cells) = rows?;
        Some(Ok(PipelineBlock {
            cell_block,
            cells,
            layers,
        }))
    }
}
