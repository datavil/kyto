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
use rayon::prelude::*;
use zarrs::array::codec::{
    ArrayToBytesCodecTraits, BytesCodec, CodecChain, CodecOptions, Crc32cCodec,
    ShardingCodecBuilder, ShardingIndexLocation, ZstdCodec,
};
use zarrs::array::{
    Array, ArrayBuilder, ArrayBytes, ArrayMetadataOptions, ChunkRepresentation, DataType, Element,
};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;
use zarrs::storage::WritableStorageTraits;

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
pub(crate) fn sharding(
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

/// Write the `zarr.json` of the group at Zarr path `path`, with `{"kyto": kyto}` as its
/// attributes, or none when `kyto` is `None`.
pub(crate) fn store_group(
    storage: &Arc<FilesystemStore>,
    path: &str,
    kyto: Option<serde_json::Value>,
) -> Result<(), KytoError> {
    let mut attributes = serde_json::Map::new();
    if let Some(kyto) = kyto {
        attributes.insert("kyto".to_owned(), kyto);
    }
    GroupBuilder::new()
        .attributes(attributes)
        .build(storage.clone(), path)
        .map_err(|error| store_io("create group", path, error))?
        .store_metadata()
        .map_err(|error| store_io("write metadata of", path, error))
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

/// Write one shard of a sharded array (format.md §4.4), compressing its inner chunks in
/// parallel on the current rayon pool.
///
/// The bytes are those [`store_chunk`] writes (zarrs' sharding encoder on one thread):
/// the encoded inner chunks that are not all fill value, in inner chunk order, then the
/// index (offset and length of each inner chunk, `u64::MAX` twice for a skipped one) in
/// `[bytes(little), crc32c]`. A shard that is all fill value is not stored. Each inner
/// chunk is encoded on its own, with zstd on one thread, so nothing depends on the
/// thread count (R11, memory/zarrs_shard_determinism.md).
///
/// The array's codec must be [`sharding`] with `zstd_level`, and its inner chunks must
/// be runs of `inner_len` consecutive elements of the shard: true for every kyto layer
/// array (1-D `indices` and `data`; `indptr`'s inner chunks span its last dimension).
pub(crate) fn store_shard<T: Element>(
    storage: &FilesystemStore,
    array: &StoreArray,
    chunk_indices: &[u64],
    elements: &[T],
    inner_len: usize,
    zstd_level: i32,
) -> Result<(), KytoError> {
    let path = array.path().as_str();
    let fail = |error: &dyn std::fmt::Display| store_io("write chunk of", path, error);
    let bytes = T::into_array_bytes(array.data_type(), elements)
        .and_then(|bytes| bytes.into_fixed().map_err(Into::into))
        .map_err(|error| fail(&error))?;
    let fill = array.fill_value();
    if ArrayBytes::new_flen(&bytes[..]).is_fill_value(fill) {
        return array
            .erase_chunk(chunk_indices)
            .map_err(|error| fail(&error));
    }
    let element_size = array.data_type().fixed_size().unwrap_or(1);
    let inner_shape = non_zero(&[inner_len as u64])?;
    // zarrs takes only whole shards; a short last inner chunk would encode other bytes.
    if !elements.len().is_multiple_of(inner_len) {
        return Err(fail(&format!(
            "{} elements are not whole inner chunks of {inner_len}",
            elements.len()
        )));
    }
    let inner = ChunkRepresentation::new(inner_shape, array.data_type().clone(), fill.clone())
        .map_err(|error| fail(&error))?;
    let chain = CodecChain::new(
        vec![],
        Arc::new(BytesCodec::little()),
        vec![Arc::new(ZstdCodec::new(zstd_level, false))],
    );
    let serial = CodecOptions::builder().concurrent_target(1).build();
    // An indexed collect keeps inner chunk order on any thread count.
    let encoded: Vec<Result<Option<Vec<u8>>, KytoError>> = bytes
        .par_chunks(inner_len * element_size)
        .map(|chunk| {
            let chunk = ArrayBytes::new_flen(chunk);
            if chunk.is_fill_value(fill) {
                return Ok(None);
            }
            chain
                .encode(chunk, &inner, &serial)
                .map(|encoded| Some(encoded.into_owned()))
                .map_err(|error| fail(&error))
        })
        .collect();
    let mut index = vec![u64::MAX; 2 * encoded.len()];
    let mut shard: Vec<u8> = Vec::new();
    for (position, chunk) in encoded.into_iter().enumerate() {
        if let Some(chunk) = chunk? {
            index[2 * position] = shard.len() as u64;
            index[2 * position + 1] = chunk.len() as u64;
            shard.extend_from_slice(&chunk);
        }
    }
    let index_shape = non_zero(&[(index.len() / 2) as u64, 2])?;
    let index_representation = ChunkRepresentation::new(index_shape, DataType::UInt64, u64::MAX)
        .map_err(|error| fail(&error))?;
    let index_chain = CodecChain::new(
        vec![],
        Arc::new(BytesCodec::little()),
        vec![Arc::new(Crc32cCodec::new())],
    );
    let index_bytes = index_chain
        .encode(
            ArrayBytes::new_flen(bytemuck::cast_slice::<u64, u8>(&index)),
            &index_representation,
            &serial,
        )
        .map_err(|error| fail(&error))?;
    shard.extend_from_slice(&index_bytes);
    storage
        .set(&array.chunk_key(chunk_indices), shard.into())
        .map_err(|error| fail(&error))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use zarrs::array::{ArrayBuilder, DataType, Element};
    use zarrs::filesystem::FilesystemStore;

    use super::{StoreArray, sharding, store_chunk, store_shard};

    /// An array of `shard` shards with inner chunks `inner`, in a fresh store.
    fn array(
        storage: &Arc<FilesystemStore>,
        data_type: DataType,
        fill: impl Into<zarrs::array::FillValue>,
        shape: Vec<u64>,
        shard: Vec<u64>,
        inner: &[u64],
    ) -> StoreArray {
        let mut builder = ArrayBuilder::new(shape, shard, data_type, fill.into());
        builder.array_to_bytes_codec(sharding(inner, 3).unwrap());
        builder.build(storage.clone(), "/a").unwrap()
    }

    /// The files of a store, by relative path.
    fn files(root: &std::path::Path) -> Vec<(String, Vec<u8>)> {
        let mut found = Vec::new();
        let mut pending = vec![root.to_path_buf()];
        while let Some(dir) = pending.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    pending.push(path);
                } else {
                    let name = path.strip_prefix(root).unwrap().display().to_string();
                    found.push((name, std::fs::read(&path).unwrap()));
                }
            }
        }
        found.sort();
        found
    }

    /// Write every shard with `store_chunk` in one store and `store_shard` (on pools of
    /// 1, 3 and 8 threads) in others; the files must be the same bytes.
    fn same_bytes<T: Element + Copy + Sync>(
        data_type: DataType,
        fill: T,
        shard_shape: &[u64],
        inner: &[u64],
        shards: &[Vec<T>],
    ) where
        zarrs::array::FillValue: From<T>,
    {
        let shard_len: u64 = shard_shape.iter().product();
        let inner_len: u64 = inner.iter().product();
        let mut shape = shard_shape.to_vec();
        shape[0] *= shards.len() as u64;
        let write = |parallel: Option<usize>| {
            let dir = tempfile::tempdir().unwrap();
            let storage = Arc::new(FilesystemStore::new(dir.path()).unwrap());
            let target = array(
                &storage,
                data_type.clone(),
                fill,
                shape.clone(),
                shard_shape.to_vec(),
                inner,
            );
            for (position, elements) in shards.iter().enumerate() {
                assert_eq!(elements.len() as u64, shard_len);
                let mut indices = vec![0; shard_shape.len()];
                indices[0] = position as u64;
                match parallel {
                    None => store_chunk(&target, &indices, elements).unwrap(),
                    Some(threads) => rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap()
                        .install(|| {
                            store_shard(
                                &storage,
                                &target,
                                &indices,
                                elements,
                                inner_len as usize,
                                3,
                            )
                        })
                        .unwrap(),
                }
            }
            files(dir.path())
        };
        let expected = write(None);
        assert!(!expected.is_empty());
        for threads in [1, 3, 8] {
            assert_eq!(write(Some(threads)), expected, "{threads} threads");
        }
    }

    /// Values from a fixed linear congruential sequence, zero where `zero(i)`.
    fn values(len: usize, seed: u64, zero: impl Fn(usize) -> bool) -> Vec<u32> {
        let mut state = seed;
        (0..len)
            .map(|position| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                if zero(position) {
                    0
                } else {
                    ((state >> 40) % 1000) as u32
                }
            })
            .collect()
    }

    #[test]
    fn entries_shards_match_the_serial_encoder() {
        let inner = 64_usize;
        let len = 16 * inner;
        // Mixed, inner chunks all fill (1, 7, 15), an all-fill shard, a padded last shard.
        let shards = vec![
            values(len, 1, |position| {
                position / inner == 1 || position % 5 == 0
            }),
            values(len, 2, |position| [7, 15].contains(&(position / inner))),
            vec![0; len],
            values(len, 3, |position| position >= 3 * inner + 5),
        ];
        same_bytes(
            DataType::UInt32,
            0_u32,
            &[len as u64],
            &[inner as u64],
            &shards,
        );
        // float32 with -0.0 (not the fill value's bytes) and NaN.
        let floats: Vec<Vec<f32>> = shards
            .iter()
            .map(|shard| {
                shard
                    .iter()
                    .enumerate()
                    .map(|(position, &value)| match position % 97 {
                        3 => -0.0,
                        11 => f32::NAN,
                        _ => value as f32 * 0.37,
                    })
                    .collect()
            })
            .collect();
        same_bytes(
            DataType::Float32,
            0.0_f32,
            &[len as u64],
            &[inner as u64],
            &floats,
        );
    }

    #[test]
    fn indptr_shards_match_the_serial_encoder() {
        // `indptr`-like: shards [1, gene blocks, C + 1], inner chunks [1, 1, C + 1].
        let row = 33_usize;
        let gene_blocks = 5_usize;
        let shards = vec![
            values(gene_blocks * row, 4, |position| position / row == 2),
            vec![0; gene_blocks * row],
            values(gene_blocks * row, 5, |_| false),
        ];
        same_bytes(
            DataType::UInt32,
            0_u32,
            &[1, gene_blocks as u64, row as u64],
            &[1, 1, row as u64],
            &shards,
        );
    }
}
