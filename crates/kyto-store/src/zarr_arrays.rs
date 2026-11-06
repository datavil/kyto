//! The four Zarr arrays of a layer: shapes, chunks and codecs (format.md §4.2, §4.4).
//!
//! Does: build each array's metadata exactly as format.md §4.4 specifies (sharding
//! with zstd and a crc32c index, little endian), write 1-D arrays chunk by chunk, and
//! turn zarrs errors into `KY-E110`.
//! Does not: decide what goes into the arrays (see `tile_encode`) or check what comes
//! out (see `tile_decode`).

use std::num::NonZeroU64;
use std::sync::Arc;

use kyto_error::KytoError;
use zarrs::array::codec::{
    ArrayToBytesCodecTraits, BytesCodec, CodecOptions, ShardingCodecBuilder, ShardingIndexLocation,
    ZstdCodec,
};
use zarrs::array::{Array, ArrayBuilder, ArrayMetadataOptions, DataType, Element};
use zarrs::filesystem::FilesystemStore;

use crate::tiling::TileGrid;
use crate::value::LayerValue;
use crate::write_options::LayerWriteOptions;

/// Chunk length of `tile_offsets` (format.md §4.4).
pub(crate) const TILE_OFFSETS_CHUNK: u64 = 65536;

/// A Zarr array of a layer in a filesystem store.
pub(crate) type StoreArray = Array<FilesystemStore>;

/// Build the `KY-E110` error for a failed store operation.
pub(crate) fn store_io(operation: &str, path: &str, error: impl std::fmt::Display) -> KytoError {
    KytoError::StoreIo {
        operation: operation.to_owned(),
        path: path.to_owned(),
        detail: error.to_string(),
    }
}

fn non_zero(values: &[u64]) -> Result<Vec<NonZeroU64>, KytoError> {
    values
        .iter()
        .map(|&value| {
            NonZeroU64::new(value).ok_or_else(|| KytoError::InvalidParameter {
                parameter: "chunk shape".to_owned(),
                value: format!("{values:?}"),
                accepted: "positive sizes only".to_owned(),
            })
        })
        .collect()
}

/// `sharding_indexed` with inner `[bytes(little), zstd(level)]` and index
/// `[bytes(little), crc32c]` at the end (format.md §4.4).
fn sharding(
    inner_chunk_shape: &[u64],
    zstd_level: i32,
) -> Result<Arc<dyn ArrayToBytesCodecTraits>, KytoError> {
    let codec = ShardingCodecBuilder::new(non_zero(inner_chunk_shape)?.into())
        .array_to_bytes_codec(Arc::new(BytesCodec::little()))
        .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(zstd_level, false))])
        .index_array_to_bytes_codec(Arc::new(BytesCodec::little()))
        .index_location(ShardingIndexLocation::End)
        .build_arc();
    Ok(codec)
}

/// The arrays of one layer to create, given its grid, entry count and options.
pub(crate) struct LayerArrays<'a> {
    /// The filesystem store holding the layer.
    pub storage: &'a Arc<FilesystemStore>,
    /// Zarr path of the layer group, `/layers/<name>`.
    pub group_path: &'a str,
    /// The layer's tile grid.
    pub grid: &'a TileGrid,
    /// Total stored entries.
    pub nnz: u64,
    /// Chunk sizes and zstd level.
    pub options: &'a LayerWriteOptions,
}

impl LayerArrays<'_> {
    fn build(
        &self,
        name: &str,
        builder: &mut ArrayBuilder,
        dimension_names: &[&str],
    ) -> Result<StoreArray, KytoError> {
        let path = format!("{}/{name}", self.group_path);
        builder
            .dimension_names(Some(dimension_names.to_vec()))
            .build(self.storage.clone(), &path)
            .map_err(|error| store_io("create array", &path, error))
    }

    /// `indptr`: `uint32 [n_cell_blocks, n_gene_blocks, C + 1]`, one shard per cell
    /// block, one inner chunk per tile.
    pub(crate) fn indptr(&self) -> Result<StoreArray, KytoError> {
        let grid = self.grid;
        let row_len = grid.indptr_len();
        let shape = vec![grid.n_cell_blocks(), grid.n_gene_blocks(), row_len];
        let shard = vec![1, grid.n_gene_blocks().max(1), row_len];
        let mut builder = ArrayBuilder::new(shape, shard, DataType::UInt32, 0_u32);
        builder.array_to_bytes_codec(sharding(&[1, 1, row_len], self.options.zstd_level)?);
        self.build("indptr", &mut builder, &["cell_block", "gene_block", "row"])
    }

    /// `tile_offsets`: `uint64 [n_tiles + 1]`, chunks of 65536, not sharded.
    pub(crate) fn tile_offsets(&self) -> Result<StoreArray, KytoError> {
        let shape = vec![self.grid.n_tiles() + 1];
        let mut builder =
            ArrayBuilder::new(shape, vec![TILE_OFFSETS_CHUNK], DataType::UInt64, 0_u64);
        builder
            .array_to_bytes_codec(Arc::new(BytesCodec::little()))
            .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(
                self.options.zstd_level,
                false,
            ))]);
        self.build("tile_offsets", &mut builder, &["tile"])
    }

    /// `indices` (`uint32`) or `data` (`float32`/`uint32`): `[nnz]`, shards of `S`
    /// entries holding inner chunks of `K`.
    pub(crate) fn entries<T: LayerValue>(&self, name: &str) -> Result<StoreArray, KytoError> {
        let options = self.options;
        let mut builder = ArrayBuilder::new(
            vec![self.nnz],
            vec![options.nnz_shard_size],
            T::data_type(),
            T::fill_value(),
        );
        builder.array_to_bytes_codec(sharding(&[options.nnz_chunk_size], options.zstd_level)?);
        self.build(name, &mut builder, &["entry"])
    }
}

/// Write an array's `zarr.json`, without zarrs' own version attributes (so bytes
/// depend only on the format, format.md §10.3).
pub(crate) fn store_metadata(array: &StoreArray) -> Result<(), KytoError> {
    let options = ArrayMetadataOptions::default().with_include_zarrs_metadata(false);
    array
        .store_metadata_opt(&options)
        .map_err(|error| store_io("write metadata of", array.path().as_str(), error))
}

/// Write a 1-D array whole, one chunk (or shard) at a time. The last chunk is padded
/// with the fill value to the full chunk shape (format.md §4.4); chunks that are all
/// fill value are not stored (zarrs skips them).
pub(crate) fn store_1d<T: Element + Copy>(
    array: &StoreArray,
    values: &[T],
    chunk_len: u64,
    fill: T,
) -> Result<(), KytoError> {
    for (chunk, slice) in values.chunks(chunk_len as usize).enumerate() {
        let mut padded;
        let elements = if slice.len() as u64 == chunk_len {
            slice
        } else {
            padded = slice.to_vec();
            padded.resize(chunk_len as usize, fill);
            &padded[..]
        };
        store_chunk(array, &[chunk as u64], elements)?;
    }
    Ok(())
}

/// Write one chunk (or shard) of an array, encoding it on one thread.
///
/// zarrs' sharding encoder places inner chunks in the shard in the order parallel
/// workers finish, so shard bytes vary between runs. One thread places them in inner
/// chunk order, as byte-identical stores require (format.md §10.3, R11).
pub(crate) fn store_chunk<T: Element>(
    array: &StoreArray,
    chunk_indices: &[u64],
    elements: &[T],
) -> Result<(), KytoError> {
    let serial = CodecOptions::builder().concurrent_target(1).build();
    array
        .store_chunk_elements_opt(chunk_indices, elements, &serial)
        .map_err(|error| store_io("write chunk of", array.path().as_str(), error))
}
