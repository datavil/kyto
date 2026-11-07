//! Stream rows of one layer into its tiled arrays (format.md §4, §10.2).
//!
//! Does: take rows in batches of any size, re-block them into cell blocks of exactly `C`
//! rows, and per cell block write its `indptr` shard and append its tiles (gene blocks
//! in increasing order, zeros dropped) to `indices` and `data`, writing each shard of `S`
//! entries as soon as it fills. [`LayerStream::finish`] encodes the last, shorter cell
//! block, writes the last shard (full shape, fill padded) and then, as format.md §10.2
//! orders, each array's `zarr.json` and `tile_offsets`.
//! Does not: check the rows (the store writer does, with `CellBlock`'s checks), write the
//! layer group's `zarr.json`, or know about other layers.
//!
//! Memory: between calls a stream holds fewer than `C` buffered rows and fewer than `S`
//! unwritten entries, plus `tile_offsets` (8 bytes per tile). While a batch is pushed it
//! also holds a copy of that batch and one `indptr` shard (`n_gene_blocks * (C + 1) * 4`
//! bytes). It never holds the whole layer.
//!
//! The bytes are those of [`crate::write_layer`] for the same matrix, however the rows
//! were batched: chunk contents depend only on the tiles, and chunks are encoded on one
//! thread (memory/zarrs_shard_determinism.md).

use std::sync::Arc;

use kyto_error::KytoError;
use zarrs::filesystem::FilesystemStore;

use crate::csr::CsrMatrix;
use crate::tile_encode::{TiledEntries, encode_cell_block};
use crate::tiling::{TileGrid, Tiling};
use crate::value::LayerValue;
use crate::write_options::LayerWriteOptions;
use crate::zarr_arrays::{
    LayerArrays, StoreArray, TILE_OFFSETS_CHUNK, store_1d, store_chunk, store_metadata,
};

/// Largest number of cells (format.md §4.5); sizes the arrays chunks are written through.
const MAX_CELLS: u64 = u32::MAX as u64;
/// Largest layer nnz (format.md §4.5); the same.
const MAX_NNZ: u64 = i64::MAX as u64;

/// One layer being written, cell block by cell block.
pub(crate) struct LayerStream<T> {
    storage: Arc<FilesystemStore>,
    group_path: String,
    tiling: Tiling,
    n_genes: u64,
    options: LayerWriteOptions,
    /// Arrays shaped for the largest layer, used only to write chunks: a chunk's bytes do
    /// not depend on the array shape, which is known only at the end.
    indptr: StoreArray,
    indices: StoreArray,
    data: StoreArray,
    /// Rows not yet in a full cell block, global gene indices; fewer than `C` between calls.
    pending: CsrMatrix<T>,
    /// All `tile_offsets` so far, and the entries not yet written in a shard.
    entries: TiledEntries<T>,
    cell_blocks_written: u64,
    shards_written: u64,
}

impl<T: LayerValue> LayerStream<T> {
    /// Start layer `name` (already checked) of a store with `n_genes` genes. The tiling
    /// and options must be valid (`TileGrid::new`, `LayerWriteOptions::validate`).
    ///
    /// # Errors
    /// `KY-E104`, `KY-E105` or `KY-E110` if the arrays cannot be built.
    pub(crate) fn new(
        storage: Arc<FilesystemStore>,
        name: &str,
        n_genes: u64,
        tiling: Tiling,
        options: LayerWriteOptions,
    ) -> Result<Self, KytoError> {
        let group_path = format!("/layers/{name}");
        let grid = TileGrid::new(MAX_CELLS, n_genes, tiling)?;
        let arrays = LayerArrays {
            storage: &storage,
            group_path: &group_path,
            grid: &grid,
            nnz: MAX_NNZ,
            options: &options,
        };
        let indptr = arrays.indptr()?;
        let indices = arrays.entries::<u32>("indices")?;
        let data = arrays.entries::<T>("data")?;
        Ok(Self {
            storage: storage.clone(),
            group_path,
            tiling,
            n_genes,
            options,
            indptr,
            indices,
            data,
            pending: empty_rows(n_genes),
            entries: TiledEntries::new(),
            cell_blocks_written: 0,
            shards_written: 0,
        })
    }

    /// Append rows given as valid CSR over all genes (`indptr` from 0 to `indices.len()`),
    /// writing every cell block they complete.
    ///
    /// # Errors
    /// `KY-E110` if a chunk cannot be written.
    pub(crate) fn push(
        &mut self,
        indptr: &[u64],
        indices: &[u32],
        data: &[T],
    ) -> Result<(), KytoError> {
        let base = self.pending.nnz();
        self.pending
            .indptr
            .extend(indptr.iter().skip(1).map(|&end| base + end));
        self.pending.indices.extend_from_slice(indices);
        self.pending.data.extend_from_slice(data);
        self.pending.n_cells += indptr.len().saturating_sub(1) as u64;
        let full_blocks = self.pending.n_cells / self.tiling.cell_block_size;
        if full_blocks == 0 {
            return Ok(());
        }
        let grid = TileGrid::new(self.pending.n_cells, self.n_genes, self.tiling)?;
        for block in 0..full_blocks {
            self.encode(&grid, block)?;
        }
        // Keep only the rows after the last full block.
        let rows = (full_blocks * self.tiling.cell_block_size) as usize;
        let consumed = self.pending.indptr[rows];
        self.pending.indices.drain(..consumed as usize);
        self.pending.data.drain(..consumed as usize);
        self.pending.indptr.drain(..rows);
        for start in &mut self.pending.indptr {
            *start -= consumed;
        }
        self.pending.n_cells -= rows as u64;
        Ok(())
    }

    /// Encode cell block `block` of the pending rows (laid out on `grid`) and write it.
    fn encode(&mut self, grid: &TileGrid, block: u64) -> Result<(), KytoError> {
        let shard = encode_cell_block(&self.pending, grid, block, &mut self.entries);
        if grid.n_gene_blocks() > 0 {
            store_chunk(&self.indptr, &[self.cell_blocks_written, 0, 0], &shard)?;
        }
        self.cell_blocks_written += 1;
        let shard_len = self.options.nnz_shard_size as usize;
        while self.entries.indices.len() >= shard_len {
            self.write_shard(shard_len)?;
        }
        Ok(())
    }

    /// Write the first `len` unwritten entries as the next shard, padded with the fill
    /// value to the full shard shape (format.md §4.4).
    fn write_shard(&mut self, len: usize) -> Result<(), KytoError> {
        let shard_len = self.options.nnz_shard_size as usize;
        let mut indices: Vec<u32> = self.entries.indices.drain(..len).collect();
        let mut data: Vec<T> = self.entries.data.drain(..len).collect();
        indices.resize(shard_len, 0);
        data.resize(shard_len, T::zero());
        store_chunk(&self.indices, &[self.shards_written], &indices)?;
        store_chunk(&self.data, &[self.shards_written], &data)?;
        self.shards_written += 1;
        Ok(())
    }

    /// Write the rest of the layer once all `n_cells` rows were pushed: the last cell
    /// block, the last shard, then each array's `zarr.json` and `tile_offsets`
    /// (format.md §10.2). Returns the layer's nnz.
    ///
    /// # Errors
    /// `KY-E105` if `n_cells` is above the format limit, `KY-E110` if a write fails.
    pub(crate) fn finish(mut self, n_cells: u64) -> Result<u64, KytoError> {
        if self.pending.n_cells > 0 {
            let grid = TileGrid::new(self.pending.n_cells, self.n_genes, self.tiling)?;
            self.encode(&grid, 0)?;
        }
        if !self.entries.indices.is_empty() {
            self.write_shard(self.entries.indices.len())?;
        }
        let nnz = self.entries.tile_offsets.last().copied().unwrap_or(0);
        let grid = TileGrid::new(n_cells, self.n_genes, self.tiling)?;
        let arrays = LayerArrays {
            storage: &self.storage,
            group_path: &self.group_path,
            grid: &grid,
            nnz,
            options: &self.options,
        };
        let indptr = arrays.indptr()?;
        let tile_offsets = arrays.tile_offsets()?;
        let indices = arrays.entries::<u32>("indices")?;
        let data = arrays.entries::<T>("data")?;
        for array in [&indptr, &tile_offsets, &indices, &data] {
            store_metadata(array)?;
        }
        store_1d(
            &tile_offsets,
            &self.entries.tile_offsets,
            TILE_OFFSETS_CHUNK,
            0,
        )?;
        Ok(nnz)
    }

    /// Buffered rows and unwritten entries, for the memory bound test.
    #[cfg(test)]
    fn buffered(&self) -> (u64, usize) {
        (self.pending.n_cells, self.entries.indices.len())
    }
}

fn empty_rows<T>(n_genes: u64) -> CsrMatrix<T> {
    CsrMatrix {
        n_cells: 0,
        n_genes,
        indptr: vec![0],
        indices: Vec::new(),
        data: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use zarrs::filesystem::FilesystemStore;

    use super::LayerStream;
    use crate::tiling::Tiling;
    use crate::write_options::LayerWriteOptions;

    #[test]
    fn buffers_stay_below_one_cell_block_and_one_shard() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Arc::new(FilesystemStore::new(dir.path()).unwrap());
        let tiling = Tiling {
            cell_block_size: 4,
            gene_block_size: 3,
        };
        let options = LayerWriteOptions {
            nnz_chunk_size: 2,
            nnz_shard_size: 6,
            zstd_level: 3,
        };
        let n_genes = 7_u64;
        let mut stream = LayerStream::<u32>::new(storage, "x", n_genes, tiling, options).unwrap();
        // Dense rows in batches of 1, 5, 2, 9, 3 rows.
        let mut n_cells = 0;
        for rows in [1_u64, 5, 2, 9, 3] {
            let indptr: Vec<u64> = (0..=rows).map(|row| row * n_genes).collect();
            let indices: Vec<u32> = (0..rows).flat_map(|_| 0..n_genes as u32).collect();
            let data = vec![1_u32; indices.len()];
            stream.push(&indptr, &indices, &data).unwrap();
            n_cells += rows;
            let (buffered_rows, unwritten) = stream.buffered();
            assert!(buffered_rows < tiling.cell_block_size);
            assert!(unwritten < options.nnz_shard_size as usize);
        }
        assert_eq!(stream.finish(n_cells).unwrap(), n_cells * n_genes);
    }
}
