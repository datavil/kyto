//! Join the tiles of a layer back into CSR rows, checking them (format.md §1.2, §4.3).
//!
//! Does: check `tile_offsets` and each tile's `indptr` and local indices, then append
//! a cell block's rows, gene blocks in increasing order, with global gene indices.
//! Pure, in-memory.
//! Does not: read storage, select genes or cells (task 2.4), or check for stored zeros
//! (readers MAY assume none, format.md §4.6).

use kyto_error::KytoError;

use crate::csr::CsrMatrix;
use crate::tile_encode::TiledEntries;
use crate::tiling::TileGrid;

fn corrupt(layer: &str, problem: String) -> KytoError {
    KytoError::CorruptLayer {
        layer: layer.to_owned(),
        problem,
    }
}

/// Check `tile_offsets`: `n_tiles + 1` values, starting at 0, non-decreasing, ending
/// at `nnz` (format.md §4.2, §4.3).
///
/// # Errors
/// `KY-E107` naming the first problem.
pub(crate) fn check_tile_offsets(
    layer: &str,
    grid: &TileGrid,
    tile_offsets: &[u64],
    nnz: u64,
) -> Result<(), KytoError> {
    let expected = u128::from(grid.n_tiles()) + 1;
    if tile_offsets.len() as u128 != expected {
        return Err(corrupt(
            layer,
            format!(
                "tile_offsets has {} values but n_tiles + 1 is {expected}",
                tile_offsets.len()
            ),
        ));
    }
    if tile_offsets.first() != Some(&0) {
        return Err(corrupt(layer, "tile_offsets[0] is not 0".to_owned()));
    }
    if let Some(tile) = tile_offsets.windows(2).position(|pair| pair[1] < pair[0]) {
        return Err(corrupt(
            layer,
            format!("tile_offsets decreases after tile {tile}"),
        ));
    }
    if tile_offsets.last() != Some(&nnz) {
        return Err(corrupt(
            layer,
            format!(
                "tile_offsets ends at {:?} but indices and data hold {nnz} entries",
                tile_offsets.last()
            ),
        ));
    }
    Ok(())
}

/// Check every tile of cell block `block` and append its rows to `out`.
///
/// `indptr` is the block's shard (`n_gene_blocks * (C + 1)` values). `entries` holds
/// the whole layer: its `tile_offsets` passed [`check_tile_offsets`] and its `indices`
/// and `data` have equal length. `out.indptr` must end at `out.indices.len()`.
///
/// # Errors
/// `KY-E107` naming the tile and the invariant that fails.
pub(crate) fn decode_cell_block<T: Copy>(
    layer: &str,
    grid: &TileGrid,
    block: u64,
    indptr: &[u32],
    entries: &TiledEntries<T>,
    out: &mut CsrMatrix<T>,
) -> Result<(), KytoError> {
    let TiledEntries {
        tile_offsets,
        indices,
        data,
    } = entries;
    let rows = grid.rows(block) as usize;
    let row_len = grid.indptr_len() as usize;
    let n_gene_blocks = grid.n_gene_blocks() as usize;
    let first_tile = (block * grid.n_gene_blocks()) as usize;
    for (gene_block, tile_indptr) in indptr.chunks_exact(row_len).enumerate() {
        let tile = first_tile + gene_block;
        let start = tile_offsets[tile];
        let tile_nnz = tile_offsets[tile + 1] - start;
        check_tile(
            layer,
            (block, gene_block as u64),
            tile_indptr,
            rows,
            tile_nnz,
            grid.width(gene_block as u64),
            &indices[start as usize..(start + tile_nnz) as usize],
        )?;
    }
    for row in 0..rows {
        for gene_block in 0..n_gene_blocks {
            let tile_indptr = &indptr[gene_block * row_len..(gene_block + 1) * row_len];
            let start = tile_offsets[first_tile + gene_block] as usize;
            let from = start + tile_indptr[row] as usize;
            let to = start + tile_indptr[row + 1] as usize;
            let gene_start = gene_block as u64 * grid.gene_block_size();
            // Checked above: local < width(g), so gene_start + local < n_genes <= 2^32 - 1.
            out.indices.extend(
                indices[from..to]
                    .iter()
                    .map(|&local| (gene_start + u64::from(local)) as u32),
            );
            out.data.extend_from_slice(&data[from..to]);
        }
        out.indptr.push(out.indices.len() as u64);
    }
    Ok(())
}

/// Check one tile: `indptr` starts at 0, never decreases, is flat over padding rows
/// and ends at the tile's nnz; each row's local indices strictly increase below the
/// gene block's width (format.md §1.2, §4.3).
fn check_tile(
    layer: &str,
    (block, gene_block): (u64, u64),
    tile_indptr: &[u32],
    rows: usize,
    tile_nnz: u64,
    width: u64,
    tile_indices: &[u32],
) -> Result<(), KytoError> {
    let fail = |what: String| {
        Err(corrupt(
            layer,
            format!("tile ({block}, {gene_block}): {what}"),
        ))
    };
    if tile_indptr.first() != Some(&0) {
        return fail("indptr does not start at 0".to_owned());
    }
    let end = tile_indptr.last().copied().map(u64::from);
    if end != Some(tile_nnz) {
        return fail(format!(
            "indptr ends at {end:?} but tile_offsets gives {tile_nnz} entries"
        ));
    }
    if let Some(row) = tile_indptr.windows(2).position(|pair| pair[1] < pair[0]) {
        return fail(format!("indptr decreases after row {row}"));
    }
    if tile_indptr[rows + 1..]
        .iter()
        .any(|&value| value != tile_indptr[rows])
    {
        return fail(format!("indptr is not flat after the last row {rows}"));
    }
    for row in 0..rows {
        let row_indices = &tile_indices[tile_indptr[row] as usize..tile_indptr[row + 1] as usize];
        if let Some(&local) = row_indices.iter().find(|&&local| u64::from(local) >= width) {
            return fail(format!(
                "row {row}: local gene index {local} is not below the block width {width}"
            ));
        }
        if row_indices.windows(2).any(|pair| pair[1] <= pair[0]) {
            return fail(format!(
                "row {row}: gene indices are not strictly increasing"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{check_tile_offsets, decode_cell_block};
    use crate::csr::CsrMatrix;
    use crate::tile_encode::TiledEntries;
    use crate::tiling::{TileGrid, Tiling};

    fn entries(tile_offsets: &[u64], indices: &[u32], data: &[f32]) -> TiledEntries<f32> {
        TiledEntries {
            tile_offsets: tile_offsets.to_vec(),
            indices: indices.to_vec(),
            data: data.to_vec(),
        }
    }

    fn grid() -> TileGrid {
        let tiling = Tiling {
            cell_block_size: 2,
            gene_block_size: 2,
        };
        TileGrid::new(3, 5, tiling).unwrap()
    }

    fn empty_out() -> CsrMatrix<f32> {
        CsrMatrix {
            n_cells: 3,
            n_genes: 5,
            indptr: vec![0],
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    fn problem(result: Result<(), kyto_error::KytoError>) -> String {
        let error = result.unwrap_err();
        assert_eq!(error.code(), "KY-E107");
        error.to_string()
    }

    // The tiles of tile_encode's test matrix.
    const OFFSETS: [u64; 7] = [0, 1, 2, 3, 4, 4, 5];
    const INDICES: [u32; 5] = [0, 0, 0, 1, 0];
    const DATA: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];

    #[test]
    fn decodes_rows_with_global_indices() {
        let grid = grid();
        check_tile_offsets("x", &grid, &OFFSETS, 5).unwrap();
        let stored = entries(&OFFSETS, &INDICES, &DATA);
        let mut out = empty_out();
        let block0 = [0, 1, 1, 0, 1, 1, 0, 1, 1];
        let block1 = [0, 1, 1, 0, 0, 0, 0, 1, 1];
        decode_cell_block("x", &grid, 0, &block0, &stored, &mut out).unwrap();
        decode_cell_block("x", &grid, 1, &block1, &stored, &mut out).unwrap();
        assert_eq!(out.indptr, vec![0, 3, 3, 5]);
        assert_eq!(out.indices, vec![0, 2, 4, 1, 4]);
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn bad_tile_offsets_are_reported() {
        let grid = grid();
        assert!(
            problem(check_tile_offsets("x", &grid, &OFFSETS[..6], 5)).contains("n_tiles + 1 is 7")
        );
        assert!(
            problem(check_tile_offsets("x", &grid, &[1, 1, 2, 3, 4, 4, 5], 5))
                .contains("[0] is not 0")
        );
        assert!(
            problem(check_tile_offsets("x", &grid, &[0, 2, 1, 3, 4, 4, 5], 5))
                .contains("decreases after tile 1")
        );
        assert!(problem(check_tile_offsets("x", &grid, &OFFSETS, 6)).contains("hold 6 entries"));
    }

    #[test]
    fn bad_tiles_are_reported() {
        let grid = grid();
        let decode = |block0: [u32; 9], indices: [u32; 5]| {
            let mut out = empty_out();
            decode_cell_block(
                "x",
                &grid,
                0,
                &block0,
                &entries(&OFFSETS, &indices, &DATA),
                &mut out,
            )
        };
        let good = [0, 1, 1, 0, 1, 1, 0, 1, 1];
        decode(good, INDICES).unwrap();
        assert!(
            problem(decode([1, 1, 1, 0, 1, 1, 0, 1, 1], INDICES))
                .contains("tile (0, 0): indptr does not start at 0")
        );
        assert!(
            problem(decode([0, 1, 2, 0, 1, 1, 0, 1, 1], INDICES)).contains("tile_offsets gives 1")
        );
        assert!(
            problem(decode([0, 1, 1, 0, 1, 1, 0, 2, 1], INDICES))
                .contains("tile (0, 2): indptr decreases")
        );
        assert!(
            problem(decode(good, [2, 0, 0, 1, 0]))
                .contains("local gene index 2 is not below the block width 2")
        );
        assert!(
            problem(decode(good, [0, 0, 1, 1, 0]))
                .contains("tile (0, 2): row 0: local gene index 1 is not below the block width 1")
        );
        // Block 1 has one row, so row 2 of its tiles is padding and must repeat row 1.
        let mut out = empty_out();
        let block1 = [0, 1, 1, 0, 0, 0, 0, 0, 1];
        let found = decode_cell_block(
            "x",
            &grid,
            1,
            &block1,
            &entries(&OFFSETS, &INDICES, &DATA),
            &mut out,
        );
        assert!(problem(found).contains("tile (1, 2): indptr is not flat after the last row 1"));
    }

    #[test]
    fn unsorted_row_is_reported() {
        let tiling = Tiling {
            cell_block_size: 1,
            gene_block_size: 4,
        };
        let grid = TileGrid::new(1, 4, tiling).unwrap();
        let mut out = empty_out();
        let found = decode_cell_block(
            "x",
            &grid,
            0,
            &[0, 2],
            &entries(&[0, 2], &[3, 1], &[1.0, 2.0]),
            &mut out,
        );
        assert!(problem(found).contains("row 0: gene indices are not strictly increasing"));
    }
}
