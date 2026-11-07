//! Task 3.7, the first missing-value rule of R9: an absent sparse value densifies to 0,
//! never null or `NaN`. Layers are written to a store and read back through the tile
//! reader, then densified with `CellBlock::to_dense` and `CellBlock::gene_values`, over
//! partial tiles, unstored (all-zero) tiles and gene selections; stored `NaN` and
//! `inf` stay values (format.md §4.6). This is what `scanpy`/`anndata` give for
//! `X.toarray()` and `sc.get.obs_df(adata, keys=[gene])`.
#![allow(clippy::unwrap_used)]

use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use kyto_store::{
    AxisSelection, CellBlock, CsrMatrix, DenseValues, LayerReader, LayerValue, LayerWriteOptions,
    TileGrid, Tiling, write_layer,
};
use proptest::prelude::*;
use zarrs::filesystem::FilesystemStore;

const LAYER: &str = "x";
const TILING: Tiling = Tiling {
    cell_block_size: 2,
    gene_block_size: 3,
};

/// The dense matrix as CSR, dropping zeros as a store does (format.md §4.6).
fn csr<T: LayerValue + PartialEq>(rows: &[Vec<T>], zero: T) -> CsrMatrix<T> {
    let mut matrix = CsrMatrix {
        n_cells: rows.len() as u64,
        n_genes: rows.first().map_or(0, Vec::len) as u64,
        indptr: vec![0],
        indices: Vec::new(),
        data: Vec::new(),
    };
    for row in rows {
        for (gene, &value) in row.iter().enumerate() {
            if value != zero {
                matrix.indices.push(gene as u32);
                matrix.data.push(value);
            }
        }
        matrix.indptr.push(matrix.indices.len() as u64);
    }
    matrix
}

/// Write `matrix` as the only layer of a store in `dir` and open its tile reader.
fn reader<T: LayerValue>(dir: &Path, matrix: &CsrMatrix<T>) -> LayerReader<FilesystemStore> {
    write_layer(dir, LAYER, matrix, TILING, &LayerWriteOptions::default()).unwrap();
    std::fs::write(
        dir.join("zarr.json"),
        r#"{"zarr_format":3,"node_type":"group","attributes":{"kyto":{"format_version":"0.1"}}}"#,
    )
    .unwrap();
    let grid = TileGrid::new(matrix.n_cells, matrix.n_genes, TILING).unwrap();
    let storage = Arc::new(FilesystemStore::new(dir).unwrap());
    LayerReader::open(storage, "test.kyto", LAYER, grid).unwrap()
}

/// Every block of `genes` (all cells), densified and stacked row by row.
fn densified(reader: &LayerReader<FilesystemStore>, genes: &AxisSelection) -> Vec<f32> {
    let mut dense = Vec::new();
    for selected in reader
        .cell_blocks(&AxisSelection::All, genes, NonZeroUsize::new(2).unwrap())
        .unwrap()
    {
        match selected.unwrap().block.to_dense().unwrap() {
            DenseValues::F32(values) => dense.extend(values),
            DenseValues::U32(values) => dense.extend(values.iter().map(|&value| value as f32)),
        }
    }
    dense
}

/// Gene position `position` of `genes` for every cell, block by block.
fn gene_column(
    reader: &LayerReader<FilesystemStore>,
    genes: &AxisSelection,
    position: u32,
) -> Vec<f32> {
    let mut column = Vec::new();
    for selected in reader
        .cell_blocks(&AxisSelection::All, genes, NonZeroUsize::new(1).unwrap())
        .unwrap()
    {
        match selected.unwrap().block.gene_values(position).unwrap() {
            DenseValues::F32(values) => column.extend(values),
            DenseValues::U32(values) => column.extend(values.iter().map(|&value| value as f32)),
        }
    }
    column
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// 5 cells x 7 genes over 2 x 3 tiles: partial last tiles on both axes, the tile of
/// cells 2-3 and genes 3-5 all zero (not stored), a stored NaN and inf, and an empty row.
fn matrix() -> Vec<Vec<f32>> {
    vec![
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        vec![0.0, f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0; 7],
        vec![f32::INFINITY, 0.0, 0.0, 4.5, 0.0, -1.0, 0.0],
    ]
}

#[test]
fn absent_values_densify_to_zero_through_the_tile_reader() {
    let dir = tempfile::tempdir().unwrap();
    let rows = matrix();
    let reader = reader(dir.path(), &csr(&rows, 0.0));
    let expected: Vec<f32> = rows.concat();
    assert_eq!(
        bits(&densified(&reader, &AxisSelection::All)),
        bits(&expected)
    );
    assert!(
        !densified(&reader, &AxisSelection::All)
            .iter()
            .zip(&expected)
            .any(|(got, want)| *want == 0.0 && got.is_nan())
    );

    // Over a gene selection: positions are selection positions.
    let picked = AxisSelection::Indices(vec![1, 4, 6]);
    let expected: Vec<f32> = rows
        .iter()
        .flat_map(|row| [row[1], row[4], row[6]])
        .collect();
    assert_eq!(bits(&densified(&reader, &picked)), bits(&expected));

    // One gene at a time, as an expression reads it: gene 4 is never stored.
    for (position, gene) in [(0, 1), (1, 4), (2, 6)] {
        let want: Vec<f32> = rows.iter().map(|row| row[gene]).collect();
        assert_eq!(bits(&gene_column(&reader, &picked, position)), bits(&want));
    }
}

#[test]
fn uint32_layers_densify_to_zero() {
    let dir = tempfile::tempdir().unwrap();
    let rows = vec![vec![0_u32, 7, 0], vec![0, 0, 0], vec![u32::MAX, 0, 1]];
    let reader = reader(dir.path(), &csr(&rows, 0));
    let mut dense = Vec::new();
    for selected in reader
        .cell_blocks(
            &AxisSelection::All,
            &AxisSelection::All,
            NonZeroUsize::new(1).unwrap(),
        )
        .unwrap()
    {
        match selected.unwrap().block.to_dense().unwrap() {
            DenseValues::U32(values) => dense.extend(values),
            DenseValues::F32(_) => panic!("a uint32 layer densified to float32"),
        }
    }
    assert_eq!(dense, rows.concat());
}

#[test]
fn edge_shapes_densify() {
    // No rows, no genes, and rows without entries.
    let none = CellBlock::from_f32(vec![0], Vec::new(), Vec::new(), 4).unwrap();
    assert_eq!(none.to_dense().unwrap(), DenseValues::F32(Vec::new()));
    let no_genes = CellBlock::from_u32(vec![0, 0, 0], Vec::new(), Vec::new(), 0).unwrap();
    assert_eq!(no_genes.to_dense().unwrap(), DenseValues::U32(Vec::new()));
    let empty_rows = CellBlock::from_u32(vec![0, 0, 0], Vec::new(), Vec::new(), 2).unwrap();
    assert_eq!(empty_rows.to_dense().unwrap(), DenseValues::U32(vec![0; 4]));
    assert_eq!(
        empty_rows.gene_values(1).unwrap(),
        DenseValues::U32(vec![0, 0])
    );
    assert!(empty_rows.to_dense().unwrap().len() == 4);
}

#[test]
fn bad_requests_are_coded_errors() {
    let block = CellBlock::from_f32(vec![0, 1], vec![2], vec![1.0], 3).unwrap();
    let error = block.gene_values(3).unwrap_err();
    assert_eq!(error.code(), "KY-E117");
    assert!(
        error.to_string().contains("not below n_genes = 3"),
        "{error}"
    );
    // More values than the platform addresses: refused, not a panic or abort.
    let wide = CellBlock::from_f32(vec![0, 0, 0], Vec::new(), Vec::new(), u64::MAX).unwrap();
    let error = wide.to_dense().unwrap_err();
    assert_eq!(error.code(), "KY-E117");
    assert!(error.to_string().contains("CellBlock::to_dense"), "{error}");
    // The value count fits in usize but the bytes do not, or the bytes pass isize::MAX:
    // both refused before allocating (2^62 and 2^61 floats).
    for n_genes in [1_u64 << 62, 1 << 61] {
        let block = CellBlock::from_f32(vec![0, 1], vec![5], vec![f32::NAN], n_genes).unwrap();
        let error = block.to_dense().unwrap_err();
        assert_eq!(error.code(), "KY-E117");
        assert!(error.to_string().contains("can address"), "{error}");
    }
}

proptest! {
    /// Any block densifies to its entries at their positions and 0 elsewhere, and each
    /// gene's values equal that column of the dense block.
    #[test]
    fn densify_matches_the_entries(
        rows in prop::collection::vec(
            prop::collection::btree_map(0_u32..9, 1_u32..100, 0..9), 0..6),
    ) {
        let n_genes = 9_u64;
        let mut indptr = vec![0_u64];
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        for row in &rows {
            indices.extend(row.keys().copied());
            data.extend(row.values().copied());
            indptr.push(indices.len() as u64);
        }
        let block = CellBlock::from_u32(indptr, indices, data, n_genes).unwrap();
        let DenseValues::U32(dense) = block.to_dense().unwrap() else {
            panic!("uint32 block");
        };
        prop_assert_eq!(dense.len(), rows.len() * 9);
        for (row, entries) in rows.iter().enumerate() {
            for gene in 0..9_u32 {
                let want = entries.get(&gene).copied().unwrap_or(0);
                prop_assert_eq!(dense[row * 9 + gene as usize], want);
            }
        }
        for gene in 0..9_u32 {
            let DenseValues::U32(column) = block.gene_values(gene).unwrap() else {
                panic!("uint32 block");
            };
            let want: Vec<u32> = (0..rows.len()).map(|row| dense[row * 9 + gene as usize]).collect();
            prop_assert_eq!(column, want);
        }
    }
}
