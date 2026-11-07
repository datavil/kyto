//! kyto-store: the on-disk format of a kyto atlas.
//!
//! Does: the Zarr v3 layout, tiled sparse matrix I/O, metadata tables and
//! importers (`.h5ad`, 10x `.h5`, `.mtx`).
//! Does not: plan, optimize or compute anything; it only reads and writes tiles.
//!
//! So far: one whole tiled sparse layer, written by [`write_layer`] and read by
//! [`read_layer`] (task 2.2, `docs/format.md` §4), and the cell and gene tables, written
//! by [`write_table`] and read by [`read_table`] (task 2.3, §5), and selected tiles of a
//! layer, streamed as [`CellBlock`]s by [`LayerReader`] (task 2.4, §4.7, §4.8).
#![forbid(unsafe_code)]

mod cell_block;
mod csr;
mod layer_io;
mod layer_open;
mod names;
mod selection;
mod store_version;
mod table_axis;
mod table_check;
mod table_columns;
mod table_io;
mod tile_decode;
mod tile_encode;
mod tile_reader;
mod tiling;
mod value;
mod write_options;
mod zarr_arrays;

pub use cell_block::{CellBlock, CellValues};
pub use csr::CsrMatrix;
pub use layer_io::{read_layer, write_layer};
pub use names::validate_member_name;
pub use selection::AxisSelection;
pub use store_version::{FORMAT_VERSION, SUPPORTED_FORMAT_VERSIONS, check_format_version};
pub use table_axis::TableAxis;
pub use table_columns::{ColumnDtype, TableColumn, table_columns};
pub use table_io::{read_table, write_table};
pub use tile_reader::{CellBlocks, LayerReader, SelectedBlock};
pub use tiling::{DEFAULT_CELL_BLOCK_SIZE, DEFAULT_GENE_BLOCK_SIZE, TileGrid, Tiling};
pub use value::LayerValue;
pub use write_options::{
    DEFAULT_NNZ_CHUNK_SIZE, DEFAULT_NNZ_SHARD_SIZE, DEFAULT_ZSTD_LEVEL, LayerWriteOptions,
};
