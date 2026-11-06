//! kyto-store: the on-disk format of a kyto atlas.
//!
//! Does: the Zarr v3 layout, tiled sparse matrix I/O, metadata tables and
//! importers (`.h5ad`, 10x `.h5`, `.mtx`).
//! Does not: plan, optimize or compute anything; it only reads and writes tiles.
//!
//! So far (task 2.2): one whole tiled sparse layer, written by [`write_layer`] and read
//! by [`read_layer`] (`docs/format.md` §4).
#![forbid(unsafe_code)]

mod csr;
mod layer_io;
mod names;
mod tile_decode;
mod tile_encode;
mod tiling;
mod value;
mod write_options;
mod zarr_arrays;

pub use csr::CsrMatrix;
pub use layer_io::{read_layer, write_layer};
pub use names::validate_member_name;
pub use tiling::{DEFAULT_CELL_BLOCK_SIZE, DEFAULT_GENE_BLOCK_SIZE, TileGrid, Tiling};
pub use value::LayerValue;
pub use write_options::{
    DEFAULT_NNZ_CHUNK_SIZE, DEFAULT_NNZ_SHARD_SIZE, DEFAULT_ZSTD_LEVEL, LayerWriteOptions,
};
