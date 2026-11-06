//! Writer parameters of one layer's entry arrays (format.md §4.4).
//!
//! Does: hold `nnz_chunk_size` (`K`), `nnz_shard_size` (`S`) and the zstd level, with
//! provisional defaults, and check them against format.md §4.4 and §4.5.
//! Does not: hold the tile sizes (see `tiling`), which are per store, not per layer.

use kyto_error::KytoError;

/// Provisional default `nnz_chunk_size` (`K`): 64 Ki entries, 256 KiB per inner chunk.
/// Task 10.4 sets the real default relative to typical tile nnz (format.md §4.4).
pub const DEFAULT_NNZ_CHUNK_SIZE: u64 = 1 << 16;
/// Provisional default `nnz_shard_size` (`S`): 16 inner chunks, 4 MiB per shard.
pub const DEFAULT_NNZ_SHARD_SIZE: u64 = 1 << 20;
/// Provisional default zstd level.
pub const DEFAULT_ZSTD_LEVEL: i32 = 3;

/// Largest `K` or `S` in entries: one buffered chunk or shard of 4-byte values is at
/// most 2^31 bytes (format.md §4.5).
const MAX_ENTRIES: u64 = (1 << 31) / 4;

/// How a layer's `indices` and `data` arrays are chunked and compressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerWriteOptions {
    /// `K`: entries per inner chunk of `indices` and `data`.
    pub nnz_chunk_size: u64,
    /// `S`: entries per shard; a positive multiple of `K`.
    pub nnz_shard_size: u64,
    /// zstd compression level, 1 to 22, for every array of the layer.
    pub zstd_level: i32,
}

impl Default for LayerWriteOptions {
    fn default() -> Self {
        Self {
            nnz_chunk_size: DEFAULT_NNZ_CHUNK_SIZE,
            nnz_shard_size: DEFAULT_NNZ_SHARD_SIZE,
            zstd_level: DEFAULT_ZSTD_LEVEL,
        }
    }
}

impl LayerWriteOptions {
    /// Check the options against format.md §4.4 and §4.5.
    ///
    /// # Errors
    /// `KY-E104` naming the first parameter out of range.
    pub fn validate(&self) -> Result<(), KytoError> {
        let invalid = |parameter: &str, value: String, accepted: String| {
            Err(KytoError::InvalidParameter {
                parameter: parameter.to_owned(),
                value,
                accepted,
            })
        };
        let Self {
            nnz_chunk_size,
            nnz_shard_size,
            zstd_level,
        } = *self;
        if !(1..=MAX_ENTRIES).contains(&nnz_chunk_size) {
            return invalid(
                "nnz_chunk_size",
                nnz_chunk_size.to_string(),
                format!("between 1 and {MAX_ENTRIES} entries"),
            );
        }
        if !(1..=MAX_ENTRIES).contains(&nnz_shard_size) || nnz_shard_size % nnz_chunk_size != 0 {
            return invalid(
                "nnz_shard_size",
                nnz_shard_size.to_string(),
                format!(
                    "a positive multiple of nnz_chunk_size ({nnz_chunk_size}) of at most {MAX_ENTRIES} entries"
                ),
            );
        }
        if !(1..=22).contains(&zstd_level) {
            return invalid(
                "zstd_level",
                zstd_level.to_string(),
                "between 1 and 22".to_owned(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{LayerWriteOptions, MAX_ENTRIES};

    fn options(nnz_chunk_size: u64, nnz_shard_size: u64, zstd_level: i32) -> LayerWriteOptions {
        LayerWriteOptions {
            nnz_chunk_size,
            nnz_shard_size,
            zstd_level,
        }
    }

    fn rejected(options: &LayerWriteOptions) -> String {
        let error = options.validate().unwrap_err();
        assert_eq!(error.code(), "KY-E104");
        error.to_string()
    }

    #[test]
    fn defaults_and_edges_are_valid() {
        LayerWriteOptions::default().validate().unwrap();
        options(1, 1, 1).validate().unwrap();
        options(3, 9, 22).validate().unwrap();
        options(MAX_ENTRIES, MAX_ENTRIES, 3).validate().unwrap();
    }

    #[test]
    fn out_of_range_is_rejected() {
        assert!(rejected(&options(0, 4, 3)).contains("nnz_chunk_size = 0"));
        assert!(rejected(&options(MAX_ENTRIES + 1, MAX_ENTRIES + 1, 3)).contains("nnz_chunk_size"));
        assert!(rejected(&options(4, 6, 3)).contains("nnz_shard_size = 6"));
        assert!(rejected(&options(4, 0, 3)).contains("nnz_shard_size = 0"));
        assert!(rejected(&options(1, MAX_ENTRIES + 1, 3)).contains("nnz_shard_size"));
        assert!(rejected(&options(4, 8, 0)).contains("zstd_level = 0"));
        assert!(rejected(&options(4, 8, 23)).contains("zstd_level = 23"));
    }
}
