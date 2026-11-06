//! Tile sizes and the tile grid of a layer (format.md §4.1, §4.5).
//!
//! Does: hold the store-wide tile sizes `C` and `G`, derive block counts, block row
//! counts and widths, and check the size limits of format.md §4.5.
//! Does not: read or write anything.

use kyto_error::KytoError;

/// Provisional default `cell_block_size` (`C`). Task 10.4 sets the real default.
pub const DEFAULT_CELL_BLOCK_SIZE: u64 = 4096;
/// Provisional default `gene_block_size` (`G`). Task 10.4 sets the real default.
pub const DEFAULT_GENE_BLOCK_SIZE: u64 = 1024;

/// Largest value of an axis length and of `C * G` (format.md §4.5).
const U32_LIMIT: u64 = u32::MAX as u64;
/// Largest size in bytes of one buffered `indptr` shard (format.md §4.5).
const SHARD_BYTES_LIMIT: u64 = 1 << 31;

/// The tile sizes of a store: `C` cells by `G` genes per tile (format.md §4.1).
///
/// The same for every layer of a store; recorded in `schema.json` `tiling` (§8).
/// [`Tiling::default`] gives provisional sizes until task 10.4 benchmarks them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tiling {
    /// `C`: cells per cell block.
    pub cell_block_size: u64,
    /// `G`: genes per gene block.
    pub gene_block_size: u64,
}

impl Default for Tiling {
    fn default() -> Self {
        Self {
            cell_block_size: DEFAULT_CELL_BLOCK_SIZE,
            gene_block_size: DEFAULT_GENE_BLOCK_SIZE,
        }
    }
}

/// The tile grid of one `n_cells x n_genes` layer under a [`Tiling`].
///
/// Built only by [`TileGrid::new`], which checks every limit of format.md §4.5 that
/// depends on the shape and tile sizes, so the arithmetic below cannot overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileGrid {
    /// Number of cells.
    n_cells: u64,
    /// Number of genes.
    n_genes: u64,
    /// `C`.
    cell_block_size: u64,
    /// `G`.
    gene_block_size: u64,
    /// `ceil(n_cells / C)`.
    n_cell_blocks: u64,
    /// `ceil(n_genes / G)`.
    n_gene_blocks: u64,
}

impl TileGrid {
    /// Build the grid, checking the tile sizes and format limits.
    ///
    /// # Errors
    /// `KY-E104` if a tile size is 0; `KY-E105` if `n_cells`, `n_genes`, `C * G` or
    /// one `indptr` shard is above its limit (format.md §4.5).
    pub fn new(n_cells: u64, n_genes: u64, tiling: Tiling) -> Result<Self, KytoError> {
        let Tiling {
            cell_block_size,
            gene_block_size,
        } = tiling;
        for (parameter, value) in [
            ("cell_block_size", cell_block_size),
            ("gene_block_size", gene_block_size),
        ] {
            if value == 0 {
                return Err(KytoError::InvalidParameter {
                    parameter: parameter.to_owned(),
                    value: "0".to_owned(),
                    accepted: "a positive integer".to_owned(),
                });
            }
        }
        check_limit(
            "n_cells",
            u128::from(n_cells),
            U32_LIMIT,
            "split the cells into several stores",
        )?;
        check_limit(
            "n_genes",
            u128::from(n_genes),
            U32_LIMIT,
            "split the genes into several stores",
        )?;
        check_limit(
            "cell_block_size * gene_block_size",
            u128::from(cell_block_size) * u128::from(gene_block_size),
            U32_LIMIT,
            "use a smaller cell_block_size or gene_block_size",
        )?;
        let n_cell_blocks = n_cells.div_ceil(cell_block_size);
        let n_gene_blocks = n_genes.div_ceil(gene_block_size);
        // C * G <= 2^32 - 1 with both >= 1, so C + 1 cannot overflow.
        check_limit(
            "bytes of one indptr shard, max(1, n_gene_blocks) * (cell_block_size + 1) * 4",
            u128::from(n_gene_blocks.max(1)) * u128::from(cell_block_size + 1) * 4,
            SHARD_BYTES_LIMIT,
            "use a larger gene_block_size or a smaller cell_block_size",
        )?;
        Ok(Self {
            n_cells,
            n_genes,
            cell_block_size,
            gene_block_size,
            n_cell_blocks,
            n_gene_blocks,
        })
    }

    /// Number of cells.
    pub fn n_cells(&self) -> u64 {
        self.n_cells
    }

    /// Number of genes.
    pub fn n_genes(&self) -> u64 {
        self.n_genes
    }

    /// `C`, cells per cell block.
    pub fn cell_block_size(&self) -> u64 {
        self.cell_block_size
    }

    /// `G`, genes per gene block.
    pub fn gene_block_size(&self) -> u64 {
        self.gene_block_size
    }

    /// `ceil(n_cells / C)`.
    pub fn n_cell_blocks(&self) -> u64 {
        self.n_cell_blocks
    }

    /// `ceil(n_genes / G)`.
    pub fn n_gene_blocks(&self) -> u64 {
        self.n_gene_blocks
    }

    /// Number of tiles, `n_cell_blocks * n_gene_blocks` (fits `u64`: both are
    /// at most `2^32 - 1`).
    pub fn n_tiles(&self) -> u64 {
        self.n_cell_blocks * self.n_gene_blocks
    }

    /// Row count of cell block `block`: `C`, except possibly for the last block.
    pub fn rows(&self, block: u64) -> u64 {
        let start = block * self.cell_block_size;
        self.cell_block_size.min(self.n_cells - start)
    }

    /// Width of gene block `block`: `G`, except possibly for the last block.
    pub fn width(&self, block: u64) -> u64 {
        let start = block * self.gene_block_size;
        self.gene_block_size.min(self.n_genes - start)
    }

    /// Length of one tile's `indptr` row, `C + 1`.
    pub fn indptr_len(&self) -> u64 {
        self.cell_block_size + 1
    }
}

fn check_limit(quantity: &str, value: u128, limit: u64, advice: &str) -> Result<(), KytoError> {
    if value > u128::from(limit) {
        return Err(KytoError::LimitExceeded {
            quantity: quantity.to_owned(),
            value: u64::try_from(value).unwrap_or(u64::MAX),
            limit,
            advice: advice.to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{TileGrid, Tiling};

    fn tiling(cell_block_size: u64, gene_block_size: u64) -> Tiling {
        Tiling {
            cell_block_size,
            gene_block_size,
        }
    }

    #[test]
    fn block_counts_rows_and_widths() {
        let grid = TileGrid::new(10, 7, tiling(4, 3)).unwrap();
        assert_eq!(
            (grid.n_cell_blocks, grid.n_gene_blocks, grid.n_tiles()),
            (3, 3, 9)
        );
        assert_eq!([grid.rows(0), grid.rows(1), grid.rows(2)], [4, 4, 2]);
        assert_eq!([grid.width(0), grid.width(1), grid.width(2)], [3, 3, 1]);
        assert_eq!(grid.indptr_len(), 5);
        let exact = TileGrid::new(8, 6, tiling(4, 3)).unwrap();
        assert_eq!([exact.rows(1), exact.width(1)], [4, 3]);
        let empty = TileGrid::new(0, 0, tiling(4, 3)).unwrap();
        assert_eq!(empty.n_tiles(), 0);
    }

    #[test]
    fn zero_tile_size_is_rejected() {
        let error = TileGrid::new(1, 1, tiling(0, 3)).unwrap_err();
        assert_eq!(error.code(), "KY-E104");
        assert!(error.to_string().contains("cell_block_size = 0"));
        let error = TileGrid::new(1, 1, tiling(3, 0)).unwrap_err();
        assert!(error.to_string().contains("gene_block_size = 0"));
    }

    #[test]
    fn limits_of_format_4_5() {
        let error = TileGrid::new(1, 1, tiling(1 << 16, 1 << 16)).unwrap_err();
        assert_eq!(error.code(), "KY-E105");
        assert!(
            error
                .to_string()
                .contains("cell_block_size * gene_block_size")
        );
        TileGrid::new(1, 1, tiling(1 << 16, (1 << 16) - 1)).unwrap();
        let error = TileGrid::new(1 << 32, 1, tiling(4, 4)).unwrap_err();
        assert!(error.to_string().contains("n_cells is 4294967296"));
        let error = TileGrid::new(1, 1 << 32, tiling(4, 4)).unwrap_err();
        assert!(error.to_string().contains("n_genes"));
        // 2^20 gene blocks of width 1 with C = 1023: 2^20 * 1024 * 4 = 2^32 bytes.
        let error = TileGrid::new(1, 1 << 20, tiling(1023, 1)).unwrap_err();
        assert!(error.to_string().contains("indptr shard"));
        let fits = TileGrid::new(1, 1 << 19, tiling(1023, 1)).unwrap();
        assert_eq!(fits.n_gene_blocks, 1 << 19);
    }

    #[test]
    fn default_is_provisional_4096_by_1024() {
        assert_eq!(Tiling::default(), tiling(4096, 1024));
    }
}
