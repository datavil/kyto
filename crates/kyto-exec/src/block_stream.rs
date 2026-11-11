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
//! stream carries the run's [`Statistics`]. The statistics passes read and compute
//! through the same [`Windows`] without the producer thread (`reduction.rs`).
//! Blocks hold at most `block_cells` rows of one store cell block: the store's cell
//! blocks, or parts of them when the physical plan splits them for a memory limit
//! (task 5.5).
//! A cancelled run (task 5.7, the source's [`RunControl`]) stops at its next block:
//! no block is read or computed after the cancel is seen, the window ends with
//! `KY-E503` and the stream after it. Each computed block adds its cells to the run's
//! progress.
//! Does not: cancel a window being computed when only dropped (it finishes, then
//! stops), interrupt a tile read or kernel already started, or choose block sizes (the
//! physical plan does, 5.5).

use std::collections::{BTreeMap, BTreeSet};
use std::num::{NonZeroU64, NonZeroUsize};
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
use crate::run_control::RunControl;
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
    /// Cells per block: the store's `C`, or fewer to split its cell blocks (5.5).
    pub(crate) block_cells: NonZeroU64,
    pub(crate) store: String,
    /// The run's cancel flag and progress counters.
    pub(crate) control: RunControl,
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
        let mut windows = match Windows::open(&self.source, &self.pool) {
            Ok(windows) => windows,
            Err(error) => {
                let _ = self.sender.send(Err(error));
                return;
            }
        };
        while let Some(outputs) = windows.next_window() {
            for item in outputs {
                let failed = item.is_err();
                if self.sender.send(item).is_err() || failed {
                    return;
                }
            }
        }
    }
}

/// The windows of a stream: `window` blocks read, then computed in parallel (rayon
/// over blocks), in cell order. Shared by the producer thread and the statistics
/// passes (task 5.6, `reduction.rs`); runs on the rayon pool it is called from.
pub(crate) struct Windows<'a, S: ?Sized> {
    source: &'a StreamSource<S>,
    pool: &'a CellBlockPool,
    streams: Vec<CellBlocks<'a, S>>,
    /// The next block of rows without layers: (cell block, its first row).
    next_cell_block: (u64, u64),
    done: bool,
}

impl<'a, S: ?Sized + ReadableStorageTraits + 'static> Windows<'a, S> {
    /// Open every scanned layer's stream of cell blocks.
    ///
    /// # Errors
    /// As [`LayerReader::cell_blocks_of_rows`].
    pub(crate) fn open(
        source: &'a StreamSource<S>,
        pool: &'a CellBlockPool,
    ) -> Result<Self, KytoError> {
        let streams = source
            .readers
            .iter()
            .map(|reader| {
                reader.cell_blocks_of_rows(
                    &AxisSelection::All,
                    &AxisSelection::All,
                    source.block_cells,
                    source.window,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            source,
            pool,
            streams,
            next_cell_block: (0, 0),
            done: false,
        })
    }

    /// The next window's blocks in cell order, each through every operator; an error
    /// ends the window and the stream (blocks before it in cell order come first).
    /// `None` at the end.
    pub(crate) fn next_window(&mut self) -> Option<Vec<Item>> {
        if self.done {
            return None;
        }
        let source = self.source;
        let mut inputs = Vec::with_capacity(source.window.get());
        let mut failure = None;
        while inputs.len() < source.window.get() {
            if let Err(cancelled) = source.control.check() {
                failure = Some(cancelled);
                break;
            }
            match self.read_next() {
                None => break,
                Some(Ok(block)) => inputs.push(block),
                Some(Err(error)) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        if inputs.is_empty() && failure.is_none() {
            self.done = true;
            return None;
        }
        self.done = failure.is_some();
        let pool = self.pool;
        // Rayon over blocks; an indexed collect keeps cell order (R11).
        let mut outputs: Vec<Item> = inputs
            .into_par_iter()
            .map(|block| {
                source.control.check()?;
                let mut block = source.operators.iter().try_fold(block, |block, operator| {
                    operator.apply(block, pool, &source.store)
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
                        pool.recycle(dropped);
                    }
                }
                source.control.add_done(block.cells.len() as u64);
                Ok(block)
            })
            .collect();
        outputs.extend(failure.map(Err));
        Some(outputs)
    }

    /// The next cell block of every scanned layer, joined; without layers, the next
    /// cell block's rows alone.
    fn read_next(&mut self) -> Option<Item> {
        let streams = &mut self.streams;
        let next_cell_block = &mut self.next_cell_block;
        if streams.is_empty() {
            // The same blocks the readers would give: each cell block's rows, at most
            // block_cells at a time.
            let (block, row) = *next_cell_block;
            let grid = self.source.grid;
            if block >= grid.n_cell_blocks() {
                return None;
            }
            let rows = grid.rows(block);
            let end = rows.min(row.saturating_add(self.source.block_cells.get()));
            *next_cell_block = if end >= rows {
                (block + 1, 0)
            } else {
                (block, end)
            };
            let first = block * grid.cell_block_size();
            return Some(Ok(PipelineBlock {
                cell_block: block,
                cells: (first + row..first + end).collect(),
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
