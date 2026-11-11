//! Read selected tiles of a layer in parallel and stream them as `CellBlock`s
//! (format.md §4.7, §4.8, §3 and §11 on open).
//!
//! Does: open a layer after checking the store's `format_version`, then for a cell and a
//! gene selection read exactly the tiles in the product of the selected cell blocks and
//! gene blocks (projection and predicate), check each tile, filter unselected rows and
//! genes in memory, and yield one [`SelectedBlock`] per selected cell block, in cell
//! order, whatever the thread count (R11).
//! [`LayerReader::cell_blocks_of_rows`] splits each cell block into blocks of at most a
//! given number of selected rows, reading only those rows' entries of each tile (task
//! 5.5's smaller blocks).
//! Does not: resolve names, evaluate predicates, plan or choose a block size for a
//! memory limit (tasks 3.x, 4.x, 5.5); it takes plain, canonical position selections.
//! Never writes.

use std::collections::VecDeque;
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use kyto_error::KytoError;
use rayon::prelude::*;
use zarrs::array::Array;
use zarrs::array_subset::ArraySubset;
use zarrs::storage::ReadableStorageTraits;

use crate::aligned_buffer::AlignedBuffer;
use crate::cell_block::{CellBlock, CellData};
use crate::layer_open::{OpenedLayer, ValueDtype, open_layer};
use crate::selection::{AxisSelection, BlockPick, block_picks};
use crate::store_version::check_format_version;
use crate::table_axis::TableAxis;
use crate::tile_decode::{check_tile_indptr, check_tile_rows};
use crate::tiling::TileGrid;
use crate::value::LayerValue;
use crate::zarr_arrays::store_io;

/// A reader of one layer that reads only the tiles a selection touches.
///
/// Generic over the Zarr storage, so a test can count reads through a wrapping store
/// and remote stores (task 2.8) reuse it. Reading never modifies the store.
pub struct LayerReader<S: ?Sized> {
    store: String,
    grid: TileGrid,
    layer: OpenedLayer<S>,
}

/// One cell block's selected rows (or, from [`LayerReader::cell_blocks_of_rows`], a
/// run of them), as yielded by [`LayerReader::cell_blocks`].
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedBlock {
    /// The cell block `b` the rows come from (format.md §4.1).
    pub cell_block: u64,
    /// Table position of the cell of each row, increasing.
    pub cells: Vec<u64>,
    /// The rows over the selected genes.
    pub block: CellBlock,
}

/// One read tile: its `indptr` row and the entries of the rows read (format.md §4.3).
struct Tile<T> {
    indptr: Vec<u32>,
    /// `indptr` of the first row read: `indices[0]` is the tile's entry `first_entry`.
    first_entry: u32,
    indices: Vec<u32>,
    data: Vec<T>,
}

impl<S: ?Sized + ReadableStorageTraits + 'static> LayerReader<S> {
    /// Open layer `layer` of the store in `storage`.
    ///
    /// Checks the store's `format_version` (format.md §3, §11), opens the layer's arrays
    /// against `grid`, and reads and checks `tile_offsets` (§4.7 step 1, §1.2). `grid`
    /// holds the store's `n_cells`, `n_genes` and tiling from `schema.json` (§8). `store`
    /// names the store in error messages.
    ///
    /// # Errors
    /// - `KY-E116` if `storage` is not a kyto store, `KY-E115` if its version is not
    ///   supported.
    /// - `KY-E102` if `layer` breaks format.md §2.1, `KY-E109` if there is no such layer.
    /// - `KY-E107` if the layer's arrays break format.md §4.
    /// - `KY-E110` if the storage fails.
    pub fn open(
        storage: Arc<S>,
        store: &str,
        layer: &str,
        grid: TileGrid,
    ) -> Result<Self, KytoError> {
        check_format_version(&*storage, store)?;
        let layer = open_layer(&storage, store, layer, &grid)?;
        Ok(Self {
            store: store.to_owned(),
            grid,
            layer,
        })
    }

    /// The layer's tile grid.
    pub fn grid(&self) -> &TileGrid {
        &self.grid
    }

    /// The store name given to [`LayerReader::open`].
    pub fn store(&self) -> &str {
        &self.store
    }

    /// Stream the `cells` x `genes` part of the layer, one [`SelectedBlock`] per cell
    /// block that holds a selected cell, in cell order.
    ///
    /// Reads exactly the tiles `(b, g)` with `b` a selected cell block and `g` a selected
    /// gene block, skipping empty ones (format.md §4.7); 3 genes in one gene block read
    /// one tile per selected cell block. CellBlock `indices` are positions in `genes`
    /// (§4.8). Up to `blocks_in_flight` cell blocks are read at once, their tiles in
    /// parallel on the current rayon pool; the output is the same on any thread count
    /// (R11). An empty cell selection yields nothing; an empty gene selection yields
    /// rows without entries and reads no tiles.
    ///
    /// # Errors
    /// `KY-E114` before reading anything if a selection is out of range or not strictly
    /// increasing. While streaming, an item is `KY-E107` for a tile that breaks
    /// format.md §4.3 or `KY-E110` if the storage fails; the stream ends after it.
    pub fn cell_blocks(
        &self,
        cells: &AxisSelection,
        genes: &AxisSelection,
        blocks_in_flight: NonZeroUsize,
    ) -> Result<CellBlocks<'_, S>, KytoError> {
        // A cell block never holds more than C selected rows: nothing is split.
        let max_rows = NonZeroU64::new(self.grid.cell_block_size()).unwrap_or(NonZeroU64::MIN);
        self.cell_blocks_of_rows(cells, genes, max_rows, blocks_in_flight)
    }

    /// As [`LayerReader::cell_blocks`], but each cell block's selected rows come as
    /// blocks of at most `max_rows` rows, in cell order (task 5.5: smaller blocks for a
    /// memory limit); a block's [`SelectedBlock::cell_block`] is the cell block its
    /// rows come from. A cell block with more selected rows than `max_rows` is split,
    /// and each part reads, per tile, the tile's `indptr` (checked whole) and only the
    /// entries of the local rows from its first to its last selected row (checked).
    /// A cell block that is not split is read as by [`LayerReader::cell_blocks`]: its
    /// whole tiles, every row checked.
    ///
    /// # Errors
    /// As [`LayerReader::cell_blocks`].
    pub fn cell_blocks_of_rows(
        &self,
        cells: &AxisSelection,
        genes: &AxisSelection,
        max_rows: NonZeroU64,
        blocks_in_flight: NonZeroUsize,
    ) -> Result<CellBlocks<'_, S>, KytoError> {
        let grid = &self.grid;
        let cell_picks = block_picks(
            cells,
            TableAxis::Cells,
            grid.n_cells(),
            grid.cell_block_size(),
        )?
        .iter()
        .flat_map(|pick| {
            let parts = pick.split(max_rows.get());
            let split = parts.len() > 1;
            parts.into_iter().map(move |part| (part, split))
        })
        .collect();
        let gene_picks = block_picks(
            genes,
            TableAxis::Genes,
            grid.n_genes(),
            grid.gene_block_size(),
        )?;
        let n_genes = gene_picks.iter().map(BlockPick::len).sum();
        Ok(CellBlocks {
            reader: self,
            cell_picks,
            gene_picks,
            n_genes,
            next: 0,
            blocks_in_flight: blocks_in_flight.get(),
            ready: VecDeque::new(),
        })
    }

    /// Read the selected rows of one cell block over every selected gene block.
    fn read_block(
        &self,
        (cell_pick, split): &(BlockPick, bool),
        gene_picks: &[BlockPick],
        n_genes: u64,
    ) -> Result<SelectedBlock, KytoError> {
        let block = match self.layer.dtype {
            ValueDtype::F32 => {
                let (indptr, indices, data) =
                    self.read_rows::<f32>(cell_pick, *split, gene_picks)?;
                CellBlock::new(indptr, indices, CellData::F32(data), n_genes)
            }
            ValueDtype::U32 => {
                let (indptr, indices, data) =
                    self.read_rows::<u32>(cell_pick, *split, gene_picks)?;
                CellBlock::new(indptr, indices, CellData::U32(data), n_genes)
            }
        };
        let first_cell = cell_pick.block * self.grid.cell_block_size();
        Ok(SelectedBlock {
            cell_block: cell_pick.block,
            cells: cell_pick
                .locals()
                .into_iter()
                .map(|row| first_cell + u64::from(row))
                .collect(),
            block,
        })
    }

    /// Read the block's tiles in parallel and join the selected rows (format.md §4.8):
    /// tiles in increasing gene block, entries of unselected genes dropped.
    #[allow(clippy::type_complexity)]
    fn read_rows<T: LayerValue>(
        &self,
        cell_pick: &BlockPick,
        split: bool,
        gene_picks: &[BlockPick],
    ) -> Result<(AlignedBuffer<u64>, AlignedBuffer<u32>, AlignedBuffer<T>), KytoError> {
        // Collect every result in gene block order, then take the first error, so the
        // error reported does not depend on the thread count (R11).
        let tiles = gene_picks
            .par_iter()
            .map(|gene_pick| self.read_tile::<T>(cell_pick, split, gene_pick.block))
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let rows = cell_pick.locals();
        // Joined arrays grow by push, as 4.6's estimate models (its Open (2)).
        let mut indptr = AlignedBuffer::with_capacity(rows.len() + 1);
        let mut indices = AlignedBuffer::new();
        let mut data = AlignedBuffer::new();
        indptr.push(0);
        for &row in &rows {
            let row = row as usize;
            for (gene_pick, tile) in gene_picks.iter().zip(&tiles) {
                let Some(tile) = tile else { continue };
                // Rows of this pick lie in the span read: indptr[row] >= first_entry.
                let from = (tile.indptr[row] - tile.first_entry) as usize;
                let to = (tile.indptr[row + 1] - tile.first_entry) as usize;
                for (&local, &value) in tile.indices[from..to].iter().zip(&tile.data[from..to]) {
                    if let Some(position) = gene_pick.position(local) {
                        // position < selection length <= n_genes <= 2^32 - 1.
                        indices.push(position as u32);
                        data.push(value);
                    }
                }
            }
            indptr.push(indices.len() as u64);
        }
        Ok((indptr, indices, data))
    }

    /// Read and check tile `(block, gene_block)` (format.md §4.7), with `block` the
    /// cell pick's: its `indptr` and its entries, only those of the local rows the pick
    /// spans if it is `split` from its cell block; `None` if the tile is empty.
    fn read_tile<T: LayerValue>(
        &self,
        cell_pick: &BlockPick,
        split: bool,
        gene_block: u64,
    ) -> Result<Option<Tile<T>>, KytoError> {
        let block = cell_pick.block;
        let grid = &self.grid;
        let layer = &self.layer;
        let tile = block * grid.n_gene_blocks() + gene_block;
        let start = layer.tile_offsets[tile as usize];
        let end = layer.tile_offsets[tile as usize + 1];
        if start == end {
            return Ok(None);
        }
        let rows = grid.rows(block);
        let width = grid.width(gene_block);
        // tile_offsets passed check_tile_offsets, so end >= start. Bound the tile before
        // allocating for it (format.md §4.5: tile nnz <= rows(b) * width(g)).
        if end - start > rows * width {
            return Err(KytoError::CorruptLayer {
                layer: layer.name.clone(),
                problem: format!(
                    "tile ({block}, {gene_block}): tile_offsets gives {} entries, more than its {rows} x {width} cells",
                    end - start
                ),
            });
        }
        let indptr: Vec<u32> = read_subset(
            &layer.indptr,
            &[
                block..block + 1,
                gene_block..gene_block + 1,
                0..grid.indptr_len(),
            ],
        )?;
        let tile_id = (block, gene_block);
        check_tile_indptr(&layer.name, tile_id, &indptr, rows as usize, end - start)?;
        // The pick's local rows lie below rows(b) (block_picks' bound), so the checked
        // indptr gives first_entry <= last_entry <= the tile's nnz.
        let (first_row, last_row) = if split {
            cell_pick.span()
        } else {
            (0, rows as u32)
        };
        let (first_row, last_row) = (first_row as usize, last_row as usize);
        let first_entry = indptr[first_row];
        let last_entry = indptr[last_row];
        let (indices, data) = if first_entry == last_entry {
            (Vec::new(), Vec::new())
        } else {
            let entry_range = start + u64::from(first_entry)..start + u64::from(last_entry);
            let entries = std::slice::from_ref(&entry_range);
            let indices: Vec<u32> = read_subset(&layer.indices, entries)?;
            let data: Vec<T> = read_subset(&layer.data, entries)?;
            (indices, data)
        };
        check_tile_rows(
            &layer.name,
            tile_id,
            &indptr,
            first_row..last_row,
            width,
            &indices,
        )?;
        Ok(Some(Tile {
            indptr,
            first_entry,
            indices,
            data,
        }))
    }
}

fn read_subset<S, T>(array: &Array<S>, ranges: &[std::ops::Range<u64>]) -> Result<Vec<T>, KytoError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
    T: zarrs::array::ElementOwned,
{
    array
        .retrieve_array_subset_elements(&ArraySubset::new_with_ranges(ranges))
        .map_err(|error| store_io("read", array.path().as_str(), error))
}

/// The stream of [`SelectedBlock`]s from [`LayerReader::cell_blocks`], in cell order.
///
/// Each item is a `Result`; after the first error the stream ends.
pub struct CellBlocks<'a, S: ?Sized> {
    reader: &'a LayerReader<S>,
    /// Each cell pick, and whether it is a part split from its cell block.
    cell_picks: Vec<(BlockPick, bool)>,
    gene_picks: Vec<BlockPick>,
    n_genes: u64,
    next: usize,
    blocks_in_flight: usize,
    ready: VecDeque<Result<SelectedBlock, KytoError>>,
}

impl<S: ?Sized + ReadableStorageTraits + 'static> Iterator for CellBlocks<'_, S> {
    type Item = Result<SelectedBlock, KytoError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ready.is_empty() && self.next < self.cell_picks.len() {
            let end = (self.next + self.blocks_in_flight).min(self.cell_picks.len());
            let (reader, gene_picks, n_genes) = (self.reader, &self.gene_picks, self.n_genes);
            // An indexed parallel collect keeps cell order on any thread count (R11).
            let window: Vec<_> = self.cell_picks[self.next..end]
                .par_iter()
                .map(|cell_pick| reader.read_block(cell_pick, gene_picks, n_genes))
                .collect();
            self.ready.extend(window);
            self.next = end;
        }
        let item = self.ready.pop_front()?;
        if item.is_err() {
            // End the stream after the first error, in cell order.
            self.ready.clear();
            self.next = self.cell_picks.len();
        }
        Some(item)
    }
}
