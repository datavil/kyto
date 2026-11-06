//! Cut a CSR matrix into the tiles of one cell block (format.md §4.3, §4.6).
//!
//! Does: for cell block `b`, build its `indptr` shard (all gene blocks, padded rows)
//! and append its tiles' entries, gene blocks in increasing order, with local gene
//! indices and zeros dropped. Pure, in-memory.
//! Does not: validate the input (the caller does, with `CsrMatrix::validate` and
//! `TileGrid::new`) or touch storage.

use crate::csr::CsrMatrix;
use crate::tiling::TileGrid;
use crate::value::LayerValue;

/// The tiled arrays of a layer being built, one cell block at a time.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TiledEntries<T> {
    /// `tile_offsets` so far: starts as `[0]`, one more value per tile.
    pub tile_offsets: Vec<u64>,
    /// Local gene index of each entry.
    pub indices: Vec<u32>,
    /// Value of each entry.
    pub data: Vec<T>,
}

impl<T> TiledEntries<T> {
    pub(crate) fn new() -> Self {
        Self {
            tile_offsets: vec![0],
            indices: Vec::new(),
            data: Vec::new(),
        }
    }
}

/// Encode cell block `block`: append its tiles to `entries` and return its `indptr`
/// shard, `n_gene_blocks * (C + 1)` values (tile `g` at `g * (C + 1)`).
///
/// The matrix must be valid ([`CsrMatrix::validate`]) and have the grid's shape.
pub(crate) fn encode_cell_block<T: LayerValue>(
    matrix: &CsrMatrix<T>,
    grid: &TileGrid,
    block: u64,
    entries: &mut TiledEntries<T>,
) -> Vec<u32> {
    let first_row = (block * grid.cell_block_size()) as usize;
    let rows = grid.rows(block) as usize;
    let row_len = grid.indptr_len() as usize;
    let mut indptr = vec![0_u32; grid.n_gene_blocks() as usize * row_len];
    // Next unread entry of each row; rows are sorted, so gene blocks are consumed in order.
    let mut cursor: Vec<usize> = matrix.indptr[first_row..first_row + rows]
        .iter()
        .map(|&start| start as usize)
        .collect();
    for (gene_block, tile_indptr) in indptr.chunks_exact_mut(row_len).enumerate() {
        let gene_start = gene_block as u64 * grid.gene_block_size();
        let gene_end = gene_start + grid.width(gene_block as u64);
        // tile nnz <= rows * width <= C * G <= 2^32 - 1 (TileGrid::new), so u32 holds it.
        let mut tile_nnz: u32 = 0;
        for (row, next) in cursor.iter_mut().enumerate() {
            let row_end = matrix.indptr[first_row + row + 1] as usize;
            while *next < row_end && u64::from(matrix.indices[*next]) < gene_end {
                let value = matrix.data[*next];
                if !value.is_zero() {
                    // gene_start <= index < 2^32, so the difference fits u32.
                    entries
                        .indices
                        .push((u64::from(matrix.indices[*next]) - gene_start) as u32);
                    entries.data.push(value);
                    tile_nnz += 1;
                }
                *next += 1;
            }
            tile_indptr[row + 1] = tile_nnz;
        }
        // Padding rows of the last cell block repeat the tile's nnz (format.md §4.3).
        tile_indptr[rows + 1..].fill(tile_nnz);
        let start = entries.tile_offsets.last().copied().unwrap_or(0);
        entries.tile_offsets.push(start + u64::from(tile_nnz));
    }
    indptr
}

#[cfg(test)]
mod tests {
    use super::{TiledEntries, encode_cell_block};
    use crate::csr::CsrMatrix;
    use crate::tiling::{TileGrid, Tiling};

    #[test]
    fn three_by_five_in_two_by_two_tiles() {
        // Rows: [1 0 2 0 3], [0 0 0 0 0], [0 4 0 -0.0 5]; -0.0 and 0.0 are dropped.
        let matrix = CsrMatrix {
            n_cells: 3,
            n_genes: 5,
            indptr: vec![0, 4, 4, 8],
            indices: vec![0, 1, 2, 4, 1, 2, 3, 4],
            data: vec![1.0_f32, 0.0, 2.0, 3.0, 4.0, 0.0, -0.0, 5.0],
        };
        let tiling = Tiling {
            cell_block_size: 2,
            gene_block_size: 2,
        };
        let grid = TileGrid::new(3, 5, tiling).unwrap();
        let mut entries = TiledEntries::new();
        let block0 = encode_cell_block(&matrix, &grid, 0, &mut entries);
        // Tiles (0,0): row0 {0:1}; (0,1): row0 {0:2}; (0,2): row0 {0:3}.
        assert_eq!(block0, vec![0, 1, 1, 0, 1, 1, 0, 1, 1]);
        let block1 = encode_cell_block(&matrix, &grid, 1, &mut entries);
        // Tiles (1,0): row0 {1:4}; (1,1): empty; (1,2): row0 {0:5}; padding row repeats.
        assert_eq!(block1, vec![0, 1, 1, 0, 0, 0, 0, 1, 1]);
        assert_eq!(entries.tile_offsets, vec![0, 1, 2, 3, 4, 4, 5]);
        assert_eq!(entries.indices, vec![0, 0, 0, 1, 0]);
        assert_eq!(entries.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}
