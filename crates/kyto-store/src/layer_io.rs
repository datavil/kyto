//! Write and read one whole tiled sparse layer, `layers/<name>` (format.md §4).
//!
//! Does: [`write_layer`] checks a CSR matrix, tiles it and writes the layer group and
//! its four arrays; [`read_layer`] reads them back, checks them and returns the CSR
//! matrix.
//! Does not: write the rest of a store (`schema.json`, tables, the root and `layers`
//! group `zarr.json`), write atomically or stream (task 2.5), or read only some tiles
//! (see `tile_reader`). A failed write can leave a partial layer behind.

use std::path::Path;
use std::sync::Arc;

use kyto_error::KytoError;
use zarrs::array::DataType;
use zarrs::array_subset::ArraySubset;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;

use crate::csr::CsrMatrix;
use crate::layer_open::{LAYER_KIND, open_layer, read_all};
use crate::names::{find_case_insensitive, validate_member_name};
use crate::tile_decode::decode_cell_block;
use crate::tile_encode::{TiledEntries, encode_cell_block};
use crate::tiling::{TileGrid, Tiling};
use crate::value::LayerValue;
use crate::write_options::LayerWriteOptions;
use crate::zarr_arrays::{
    LayerArrays, TILE_OFFSETS_CHUNK, store_1d, store_chunk, store_io, store_metadata,
};

fn open_storage(store: &Path) -> Result<Arc<FilesystemStore>, KytoError> {
    FilesystemStore::new(store)
        .map(Arc::new)
        .map_err(|error| store_io("open store", &store.display().to_string(), error))
}

/// Write `matrix` as layer `name` of the store at `store` (format.md §4).
///
/// Zeros in `matrix.data` (and `-0.0`) are dropped, so the stored layer is canonical
/// CSR (format.md §4.6). `tiling` must be the store's tiling (format.md §4.1).
///
/// # Errors
/// - `KY-E102` if `name` breaks format.md §2.1.
/// - `KY-E103` if a layer of that name, ignoring case, already exists.
/// - `KY-E104` if a tile size or an option is out of range.
/// - `KY-E105` if the shape or tiling is above a limit of format.md §4.5.
/// - `KY-E106` if `matrix` is not valid CSR of its shape.
/// - `KY-E110` if the file system or zarrs fails.
pub fn write_layer<T: LayerValue>(
    store: &Path,
    name: &str,
    matrix: &CsrMatrix<T>,
    tiling: Tiling,
    options: &LayerWriteOptions,
) -> Result<(), KytoError> {
    validate_member_name("layer", name)?;
    options.validate()?;
    let grid = TileGrid::new(matrix.n_cells, matrix.n_genes, tiling)?;
    matrix.validate(name)?;
    if let Some(existing) = find_case_insensitive(&store.join("layers"), name)? {
        return Err(KytoError::MemberExists {
            kind: "layer".to_owned(),
            name: name.to_owned(),
            existing,
        });
    }
    let nnz = matrix.data.iter().filter(|value| !value.is_zero()).count() as u64;

    let storage = open_storage(store)?;
    let group_path = format!("/layers/{name}");
    let mut attributes = serde_json::Map::new();
    attributes.insert("kyto".to_owned(), serde_json::json!({ "kind": LAYER_KIND }));
    let group = GroupBuilder::new()
        .attributes(attributes)
        .build(storage.clone(), &group_path)
        .map_err(|error| store_io("create group", &group_path, error))?;
    group
        .store_metadata()
        .map_err(|error| store_io("write metadata of", &group_path, error))?;

    let arrays = LayerArrays {
        storage: &storage,
        group_path: &group_path,
        grid: &grid,
        nnz,
        options,
    };
    let indptr = arrays.indptr()?;
    let mut entries = TiledEntries::new();
    for block in 0..grid.n_cell_blocks() {
        let shard = encode_cell_block(matrix, &grid, block, &mut entries);
        if grid.n_gene_blocks() > 0 {
            store_chunk(&indptr, &[block, 0, 0], &shard)?;
        }
    }
    let tile_offsets = arrays.tile_offsets()?;
    let indices = arrays.entries::<u32>("indices")?;
    let data = arrays.entries::<T>("data")?;
    let shard = options.nnz_shard_size;
    store_1d(&tile_offsets, &entries.tile_offsets, TILE_OFFSETS_CHUNK, 0)?;
    store_1d(&indices, &entries.indices, shard, 0)?;
    store_1d(&data, &entries.data, shard, T::zero())?;
    for array in [&indptr, &tile_offsets, &indices, &data] {
        store_metadata(array)?;
    }
    Ok(())
}

/// Read layer `name` of the store at `store` whole, as CSR (format.md §4).
///
/// `n_cells`, `n_genes` and `tiling` come from the store's `schema.json` (§8); the
/// layer's arrays must agree with them. Every tile is checked (format.md §1.2, §4.3).
///
/// # Errors
/// - `KY-E102` if `name` breaks format.md §2.1.
/// - `KY-E104` or `KY-E105` if `tiling` or the shape is invalid.
/// - `KY-E107` if the layer breaks format.md §4.
/// - `KY-E108` if the layer stores another value dtype than `T`.
/// - `KY-E109` if the store has no layer `name`.
/// - `KY-E110` if the file system or zarrs fails.
pub fn read_layer<T: LayerValue>(
    store: &Path,
    name: &str,
    n_cells: u64,
    n_genes: u64,
    tiling: Tiling,
) -> Result<CsrMatrix<T>, KytoError> {
    validate_member_name("layer", name)?;
    let grid = TileGrid::new(n_cells, n_genes, tiling)?;
    let storage = open_storage(store)?;
    let layer = open_layer(&storage, &store.display().to_string(), name, &grid)?;
    check_value_dtype::<T>(name, layer.data.data_type())?;
    let nnz = layer.indices.shape().first().copied().unwrap_or(0);
    let entries = TiledEntries {
        tile_offsets: layer.tile_offsets,
        indices: read_all(&layer.indices)?,
        data: read_all(&layer.data)?,
    };
    let indptr = layer.indptr;

    let mut out = CsrMatrix {
        n_cells,
        n_genes,
        indptr: Vec::with_capacity(n_cells as usize + 1),
        indices: Vec::with_capacity(nnz as usize),
        data: Vec::with_capacity(nnz as usize),
    };
    out.indptr.push(0);
    for block in 0..grid.n_cell_blocks() {
        let shard: Vec<u32> = if grid.n_gene_blocks() == 0 {
            Vec::new()
        } else {
            let subset = ArraySubset::new_with_ranges(&[
                block..block + 1,
                0..grid.n_gene_blocks(),
                0..grid.indptr_len(),
            ]);
            indptr
                .retrieve_array_subset_elements(&subset)
                .map_err(|error| store_io("read", indptr.path().as_str(), error))?
        };
        decode_cell_block(name, &grid, block, &shard, &entries, &mut out)?;
    }
    Ok(out)
}

/// `KY-E108` if the stored value dtype is the other layer dtype, `KY-E107` if it is
/// not a layer dtype at all.
fn check_value_dtype<T: LayerValue>(layer: &str, stored: &DataType) -> Result<(), KytoError> {
    let stored = stored.to_string();
    if stored == T::ZARR_DTYPE {
        return Ok(());
    }
    if stored == f32::ZARR_DTYPE || stored == u32::ZARR_DTYPE {
        return Err(KytoError::LayerDtype {
            layer: layer.to_owned(),
            stored,
            requested: T::ZARR_DTYPE.to_owned(),
        });
    }
    Err(KytoError::CorruptLayer {
        layer: layer.to_owned(),
        problem: format!("array data has dtype {stored}; a layer stores float32 or uint32"),
    })
}
