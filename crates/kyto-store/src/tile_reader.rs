//! Read selected tiles of a layer in parallel and stream them as `CellBlock`s
//! (format.md §4.7, §4.8, §3 and §11 on open).
//!
//! Does: open a layer after checking the store's `format_version`, then for a cell and a
//! gene selection read exactly the tiles in the product of the selected cell blocks and
//! gene blocks (projection and predicate), check each tile, filter unselected rows and
//! genes in memory, and yield one [`SelectedBlock`] per selected cell block, in cell
//! order, whatever the thread count (R11).
//! Does not: resolve names, evaluate predicates, plan or shrink blocks to a memory limit
//! (tasks 3.x, 4.x, 5.5); it takes plain, canonical position selections. Never writes.

use std::collections::VecDeque;
use std::num::NonZeroUsize;
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
use crate::tile_decode::check_tile;
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

/// One cell block's selected rows, as yielded by [`LayerReader::cell_blocks`].
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedBlock {
    /// The cell block `b` the rows come from (format.md §4.1).
    pub cell_block: u64,
    /// Table position of the cell of each row, increasing.
    pub cells: Vec<u64>,
    /// The rows over the selected genes.
    pub block: CellBlock,
}

/// One read tile: its `indptr` row and its entries (format.md §4.3).
struct Tile<T> {
    indptr: Vec<u32>,
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
        let grid = &self.grid;
        let cell_picks = block_picks(
            cells,
            TableAxis::Cells,
            grid.n_cells(),
            grid.cell_block_size(),
        )?;
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
        cell_pick: &BlockPick,
        gene_picks: &[BlockPick],
        n_genes: u64,
    ) -> Result<SelectedBlock, KytoError> {
        let block = match self.layer.dtype {
            ValueDtype::F32 => {
                let (indptr, indices, data) = self.read_rows::<f32>(cell_pick, gene_picks)?;
                CellBlock::new(indptr, indices, CellData::F32(data), n_genes)
            }
            ValueDtype::U32 => {
                let (indptr, indices, data) = self.read_rows::<u32>(cell_pick, gene_picks)?;
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
        gene_picks: &[BlockPick],
    ) -> Result<(AlignedBuffer<u64>, AlignedBuffer<u32>, AlignedBuffer<T>), KytoError> {
        // Collect every result in gene block order, then take the first error, so the
        // error reported does not depend on the thread count (R11).
        let tiles = gene_picks
            .par_iter()
            .map(|gene_pick| self.read_tile::<T>(cell_pick.block, gene_pick.block))
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
                let from = tile.indptr[row] as usize;
                let to = tile.indptr[row + 1] as usize;
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

    /// Read and check tile `(block, gene_block)` (format.md §4.7); `None` if empty.
    fn read_tile<T: LayerValue>(
        &self,
        block: u64,
        gene_block: u64,
    ) -> Result<Option<Tile<T>>, KytoError> {
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
        let entry_range = start..end;
        let entries = std::slice::from_ref(&entry_range);
        let indices: Vec<u32> = read_subset(&layer.indices, entries)?;
        let data: Vec<T> = read_subset(&layer.data, entries)?;
        check_tile(
            &layer.name,
            (block, gene_block),
            &indptr,
            rows as usize,
            end - start,
            width,
            &indices,
        )?;
        Ok(Some(Tile {
            indptr,
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
    cell_picks: Vec<BlockPick>,
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
