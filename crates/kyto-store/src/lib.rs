//! kyto-store: the on-disk format of a kyto atlas.
//!
//! Does: the Zarr v3 layout, tiled sparse matrix I/O, metadata tables and
//! importers (`.h5ad`, 10x `.h5`, `.mtx`), the `.h5ad` export, remote (object store) reads.
//! Does not: plan, optimize or compute anything; it only reads and writes tiles.
//!
//! So far: one whole tiled sparse layer, written by [`write_layer`] and read by
//! [`read_layer`] (task 2.2, `docs/format.md` §4), and the cell and gene tables, written
//! by [`write_table`] and read by [`read_table`] (task 2.3, §5), and selected tiles of a
//! layer, streamed as [`CellBlock`]s by [`LayerReader`] (task 2.4, §4.7, §4.8), and a
//! whole store written atomically from a stream of CellBlocks by [`StoreWriter`] (task
//! 2.5, §10), and inputs imported into a new store by [`convert`] (`.h5ad`, 10x `.h5`,
//! Matrix Market; task 2.6), and a store exported as `.h5ad` by [`sink_h5ad`] (task 2.7), and stores on S3, GCS
//! or HTTP read through [`RemoteStorage`] by the same [`LayerReader`] (task 2.8), and `schema.json` read as a
//! [`kyto_core::AtlasSchema`] by [`read_atlas_schema`] (task 3.1), the one typed model the
//! writer also writes it from.
#![forbid(unsafe_code)]

mod atomic_dir;
mod atomic_file;
mod cell_block;
mod csr;
mod export_h5;
mod export_h5ad;
mod export_h5ad_table;
mod import_10x;
mod import_convert;
mod import_h5;
mod import_h5_matrix;
mod import_h5ad;
mod import_ids;
mod import_mtx;
mod import_rows;
mod import_transpose;
mod layer_io;
mod layer_open;
mod layer_stream;
mod names;
mod remote_store;
mod selection;
mod store_json;
mod store_schema;
mod store_version;
mod store_writer;
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

pub use cell_block::{CellBlock, CellValues, DenseValues};
pub use csr::CsrMatrix;
pub use export_h5ad::{DEFAULT_EXPORT_BLOCKS_IN_FLIGHT, SinkH5adOptions, sink_h5ad};
pub use import_convert::{ConvertOptions, DEFAULT_IMPORT_BUDGET_BYTES, convert};
pub use import_ids::OnDuplicateIds;
pub use layer_io::{read_layer, write_layer};
pub use layer_open::ValueDtype;
pub use names::validate_member_name;
pub use remote_store::RemoteStorage;
pub use selection::AxisSelection;
pub use store_json::ImportSource;
pub use store_schema::{open_atlas_schema, read_atlas_schema};
pub use store_version::{FORMAT_VERSION, SUPPORTED_FORMAT_VERSIONS, check_format_version};
pub use store_writer::{LayerSpec, StoreWriter};
pub use table_axis::TableAxis;
pub use table_columns::{ColumnDtype, ColumnSchema, table_columns};
pub use table_io::{read_ids, read_symbols, read_table, write_table};
pub use tile_reader::{CellBlocks, LayerReader, SelectedBlock};
pub use tiling::{DEFAULT_CELL_BLOCK_SIZE, DEFAULT_GENE_BLOCK_SIZE, TileGrid, Tiling};
pub use value::LayerValue;
pub use write_options::{
    DEFAULT_NNZ_CHUNK_SIZE, DEFAULT_NNZ_SHARD_SIZE, DEFAULT_ZSTD_LEVEL, LayerWriteOptions,
};
