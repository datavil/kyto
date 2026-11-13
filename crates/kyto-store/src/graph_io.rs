//! Graphs in a store (format.md §7): an `n_cells x n_cells` sparse matrix in plain CSR.
//!
//! Does: [`Graph`] holds one cell graph in memory and checks it ([`Graph::check`]:
//! format.md §7's invariants); [`read_graph`] reads one whole from a local store and
//! checks it against `schema.json`; `write_graph` (the [`crate::StoreWriter`]'s)
//! writes the group `graphs/<name>/` with `indptr` (`uint64`, chunks of 65536, not
//! sharded) and `indices`/`data` sharded as layer entries (§4.4), with the given `K`,
//! `S` and zstd level.
//! Does not: read part of a graph (graphs are read whole, §7), or tile it.

use std::path::Path;
use std::sync::Arc;

use kyto_core::{Axis, FloatDtype, GraphSchema};
use kyto_error::KytoError;
use zarrs::array::codec::{BytesCodec, ZstdCodec};
use zarrs::array::{Array, ArrayBuilder, DataType, FillValue};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;

use crate::embedding_io::FloatValues;
use crate::layer_open::read_all;
use crate::store_schema::read_atlas_schema;
use crate::write_options::LayerWriteOptions;
use crate::zarr_arrays::{
    StoreArray, TILE_OFFSETS_CHUNK, sharding, store_1d, store_group, store_io, store_metadata,
};

/// The `kind` attribute of a graph group (format.md §7).
pub(crate) const GRAPH_KIND: &str = "csr";

/// One cell graph in CSR: row `i` holds the edges `indptr[i]..indptr[i + 1]` of cell `i`
/// of its store, to the cells `indices` (strictly increasing in a row), weighted by
/// `weights`. A stored edge may weigh 0; absence means no edge (§7).
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    /// The graph name (format.md §2.1).
    pub name: String,
    /// `n_cells + 1` offsets.
    pub indptr: Vec<u64>,
    /// Column (cell) of each edge.
    pub indices: Vec<u32>,
    /// Weight of each edge.
    pub weights: FloatValues,
}

impl Graph {
    /// The number of cells (rows and columns).
    pub fn n_cells(&self) -> u64 {
        self.indptr.len().saturating_sub(1) as u64
    }

    /// The number of edges.
    pub fn nnz(&self) -> u64 {
        self.indices.len() as u64
    }

    /// Its entry of `schema.json` (format.md §8).
    pub fn schema(&self) -> GraphSchema {
        GraphSchema {
            name: self.name.clone(),
            axis: Axis::Cells,
            dtype: self.weights.dtype(),
            nnz: self.nnz(),
        }
    }

    /// What breaks format.md §7 for a graph of `n_cells` cells, or `None`: `indptr` of
    /// `n_cells + 1` non-decreasing offsets from 0 to the number of edges, one weight
    /// an edge, indices strictly increasing in each row and below `n_cells`.
    pub fn check(&self, n_cells: u64) -> Option<String> {
        if self.indptr.len() as u64 != n_cells + 1 {
            return Some(format!(
                "indptr has {} offsets for {n_cells} cells; it needs n_cells + 1",
                self.indptr.len()
            ));
        }
        if self.indptr.first() != Some(&0) || self.indptr.last() != Some(&self.nnz()) {
            return Some(format!(
                "indptr must run from 0 to the {} edges",
                self.indices.len()
            ));
        }
        if self.weights.len() != self.indices.len() {
            return Some(format!(
                "{} weights for {} edges",
                self.weights.len(),
                self.indices.len()
            ));
        }
        for (row, pair) in self.indptr.windows(2).enumerate() {
            if pair[0] > pair[1] {
                return Some(format!("row {row}: indptr decreases"));
            }
            let entries = &self.indices[pair[0] as usize..pair[1] as usize];
            if let Some(&index) = entries.iter().find(|&&index| u64::from(index) >= n_cells) {
                return Some(format!(
                    "row {row}: index {index} is not below the {n_cells} cells"
                ));
            }
            if let Some(pair) = entries.windows(2).find(|pair| pair[0] >= pair[1]) {
                return Some(format!(
                    "row {row}: index {} follows {}; indices must be strictly increasing",
                    pair[1], pair[0]
                ));
            }
        }
        None
    }
}

/// Read graph `name` of the local store at `store` whole, checked against format.md §7
/// and its `schema.json` entry (dtype, `nnz`, `n_cells`).
///
/// # Errors
/// - `KY-E109` if the store's schema lists no graph `name`.
/// - `KY-E126` if the group breaks format.md §7 or disagrees with `schema.json`.
/// - `KY-E110` or `KY-E122` if the store or its schema cannot be read.
pub fn read_graph(store: &Path, name: &str) -> Result<Graph, KytoError> {
    let store_text = store.display().to_string();
    let schema = read_atlas_schema(store)?;
    let Some(entry) = schema.graphs.iter().find(|graph| graph.name == name) else {
        return Err(KytoError::MissingMember {
            kind: "graph".to_owned(),
            name: name.to_owned(),
            store: store_text,
            suggestions: kyto_core::suggest(
                name,
                schema.graphs.iter().map(|graph| graph.name.as_str()),
            )
            .into(),
        });
    };
    let corrupt = |problem: String| KytoError::CorruptMember {
        kind: "graph".to_owned(),
        name: name.to_owned(),
        problem,
    };
    let storage = Arc::new(
        FilesystemStore::new(store).map_err(|error| store_io("open store", &store_text, error))?,
    );
    let path = format!("/graphs/{name}");
    let group = Group::open(storage.clone(), &path)
        .map_err(|error| corrupt(format!("cannot open the group: {error}")))?;
    let kind = group
        .attributes()
        .get("kyto")
        .and_then(|kyto| kyto.get("kind"))
        .and_then(|kind| kind.as_str());
    if kind != Some(GRAPH_KIND) {
        return Err(corrupt(format!(
            "attributes.kyto.kind is {kind:?}, not \"{GRAPH_KIND}\""
        )));
    }
    let open = |array: &str, shape: u64, data_type: &DataType| {
        let opened = Array::open(storage.clone(), &format!("{path}/{array}"))
            .map_err(|error| corrupt(format!("cannot open array {array}: {error}")))?;
        if opened.shape() != [shape] || opened.data_type().to_string() != data_type.to_string() {
            return Err(corrupt(format!(
                "array {array} has shape {:?} and dtype {}, but schema.json gives [{shape}] of {data_type}",
                opened.shape(),
                opened.data_type()
            )));
        }
        Ok(opened)
    };
    let indptr = open("indptr", schema.n_cells + 1, &DataType::UInt64)?;
    let indices = open("indices", entry.nnz, &DataType::UInt32)?;
    let weights = match entry.dtype {
        FloatDtype::Float32 => {
            FloatValues::F32(read_all(&open("data", entry.nnz, &DataType::Float32)?)?)
        }
        FloatDtype::Float64 => {
            FloatValues::F64(read_all(&open("data", entry.nnz, &DataType::Float64)?)?)
        }
    };
    let graph = Graph {
        name: name.to_owned(),
        indptr: read_all(&indptr)?,
        indices: read_all(&indices)?,
        weights,
    };
    if let Some(problem) = graph.check(schema.n_cells) {
        return Err(corrupt(problem));
    }
    Ok(graph)
}

/// Write `graph` to `/graphs/<name>/` of `storage` (format.md §7). The caller has
/// checked it ([`Graph::check`]) and its name.
pub(crate) fn write_graph(
    storage: &Arc<FilesystemStore>,
    graph: &Graph,
    options: &LayerWriteOptions,
) -> Result<(), KytoError> {
    let group = format!("/graphs/{}", graph.name);
    let indptr_path = format!("{group}/indptr");
    let indptr: StoreArray = ArrayBuilder::new(
        vec![graph.indptr.len() as u64],
        vec![TILE_OFFSETS_CHUNK],
        DataType::UInt64,
        0_u64,
    )
    .array_to_bytes_codec(Arc::new(BytesCodec::little()))
    .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(options.zstd_level, false))])
    .dimension_names(Some(vec!["row"]))
    .build(storage.clone(), &indptr_path)
    .map_err(|error| store_io("create array", &indptr_path, error))?;
    store_1d(&indptr, &graph.indptr, TILE_OFFSETS_CHUNK, 0)?;
    store_metadata(&indptr)?;
    let entries = |name: &str, data_type: DataType, fill: FillValue| {
        let path = format!("{group}/{name}");
        ArrayBuilder::new(
            vec![graph.nnz()],
            vec![options.nnz_shard_size],
            data_type,
            fill,
        )
        .array_to_bytes_codec(sharding(&[options.nnz_chunk_size], options.zstd_level)?)
        .dimension_names(Some(vec!["entry"]))
        .build(storage.clone(), &path)
        .map_err(|error| store_io("create array", &path, error))
    };
    let shard = options.nnz_shard_size;
    let indices = entries("indices", DataType::UInt32, 0_u32.into())?;
    store_1d(&indices, &graph.indices, shard, 0)?;
    store_metadata(&indices)?;
    let data = match &graph.weights {
        FloatValues::F32(weights) => {
            let data = entries("data", DataType::Float32, 0.0_f32.into())?;
            store_1d(&data, weights, shard, 0.0)?;
            data
        }
        FloatValues::F64(weights) => {
            let data = entries("data", DataType::Float64, 0.0_f64.into())?;
            store_1d(&data, weights, shard, 0.0)?;
            data
        }
    };
    store_metadata(&data)?;
    let kyto = serde_json::json!({ "kind": GRAPH_KIND, "axis": "cells" });
    store_group(storage, &group, Some(kyto))
}
