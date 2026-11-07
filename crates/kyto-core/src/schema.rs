//! `AtlasSchema`: the typed model of what a store holds, read without the matrix
//! (format.md §8).
//!
//! Does: describe the axis lengths, tiling, the cell and gene tables (columns), layers
//! (name, dtype, nnz), embeddings (name, axis, width, dtype), graphs and modalities, and
//! (de)serialize them as `schema.json` with serde: keys in spec order, unknown keys
//! rejected. kyto-store writes and reads `schema.json` through these types only.
//! Does not: read files, or check that the schema agrees with the members it describes
//! (each kyto-store reader checks what it opens).

use serde::{Deserialize, Serialize};

use crate::schema_column::ColumnSchema;

/// Provisional default `cell_block_size` (`C`). Task 10.4 sets the real default.
pub const DEFAULT_CELL_BLOCK_SIZE: u64 = 4096;
/// Provisional default `gene_block_size` (`G`). Task 10.4 sets the real default.
pub const DEFAULT_GENE_BLOCK_SIZE: u64 = 1024;

/// The tile sizes of a store: `C` cells by `G` genes per tile (format.md §4.1).
///
/// The same for every layer of a store; recorded in `schema.json` `tiling` (§8).
/// [`Tiling::default`] gives provisional sizes until task 10.4 benchmarks them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

/// The two named axes (R4): rows are cells, columns are genes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Axis {
    /// One entry per cell.
    Cells,
    /// One entry per gene.
    Genes,
}

/// The value dtype of a layer, by its Zarr name (format.md §4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LayerDtype {
    /// `float32`.
    Float32,
    /// `uint32`.
    Uint32,
}

/// The value dtype of an embedding or a graph, by its Zarr name (format.md §6, §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FloatDtype {
    /// `float32`.
    Float32,
    /// `float64`.
    Float64,
}

/// The columns of the cell or the gene table, ID column first (format.md §5, §8).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableSchema {
    /// Every Parquet column, in Parquet order.
    pub columns: Vec<ColumnSchema>,
}

/// A tiled sparse layer (format.md §4).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerSchema {
    /// Layer name (format.md §2.1).
    pub name: String,
    /// Stored value dtype.
    pub dtype: LayerDtype,
    /// Stored entries.
    pub nnz: u64,
}

/// A dense embedding of the cells or genes (format.md §6).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingSchema {
    /// Embedding name (format.md §2.1).
    pub name: String,
    /// The axis its rows follow.
    pub axis: Axis,
    /// Number of components (columns), at least 1.
    pub width: u64,
    /// Stored value dtype.
    pub dtype: FloatDtype,
}

/// A sparse graph over one axis (format.md §7; v0 has cell graphs only).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphSchema {
    /// Graph name (format.md §2.1).
    pub name: String,
    /// The axis of both its rows and columns.
    pub axis: Axis,
    /// Stored edge weight dtype.
    pub dtype: FloatDtype,
    /// Stored edges.
    pub nnz: u64,
}

/// A second feature space measured on the same cells (for example surface proteins),
/// with its own feature table and layers.
///
/// Provisional: format v0 reserves `modalities` and stores none (format.md §2, §13.2),
/// so a v0 schema always has an empty list and `schema.json` omits the key. The shape
/// is fixed by the format version that adds modalities.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalitySchema {
    /// Modality name (format.md §2.1).
    pub name: String,
    /// Number of features.
    pub n_features: u64,
    /// The feature table.
    pub features: TableSchema,
    /// Its layers, `n_cells x n_features`.
    pub layers: Vec<LayerSchema>,
}

/// Everything a store holds, read without the matrix: the contents of `schema.json`
/// (format.md §8).
///
/// Serializes to `schema.json`'s keys in spec order; deserializing rejects unknown keys,
/// `null` for optional keys and wrong types, so any accepted text writes back exactly
/// (same formatter). The one exception: an empty `modalities` list is accepted and not
/// written back, as v0 writers never write the key. Member lists keep their order (writers sort them by name, §8).
///
/// # Examples
/// ```
/// use kyto_core::{AtlasSchema, ColumnDtype, ColumnSchema, TableSchema, Tiling};
///
/// let schema = AtlasSchema {
///     n_cells: 2,
///     n_genes: 3,
///     tiling: Tiling::default(),
///     cells: TableSchema { columns: vec![ColumnSchema::new("cell_id", ColumnDtype::String)] },
///     genes: TableSchema { columns: vec![ColumnSchema::new("gene_id", ColumnDtype::String)] },
///     layers: vec![],
///     embeddings: vec![],
///     graphs: vec![],
///     modalities: vec![],
/// };
/// let text = serde_json::to_string(&schema).unwrap();
/// assert!(text.starts_with(r#"{"n_cells":2,"n_genes":3,"tiling":"#));
/// assert_eq!(serde_json::from_str::<AtlasSchema>(&text).unwrap(), schema);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AtlasSchema {
    /// Number of cells (matrix rows).
    pub n_cells: u64,
    /// Number of genes (matrix columns).
    pub n_genes: u64,
    /// Tile sizes, the same for every layer.
    pub tiling: Tiling,
    /// The cell table, one row per cell.
    pub cells: TableSchema,
    /// The gene table, one row per gene.
    pub genes: TableSchema,
    /// The tiled sparse layers.
    pub layers: Vec<LayerSchema>,
    /// The dense embeddings.
    pub embeddings: Vec<EmbeddingSchema>,
    /// The sparse graphs.
    pub graphs: Vec<GraphSchema>,
    /// Further modalities; always empty in format v0, and then not written.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modalities: Vec<ModalitySchema>,
}
