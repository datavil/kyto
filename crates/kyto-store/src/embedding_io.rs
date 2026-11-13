//! Embeddings in a store (format.md §6): dense `float32`/`float64` arrays of one row per
//! cell or per gene.
//!
//! Does: [`Embedding`] holds one embedding in memory, row-major; [`read_embedding`]
//! reads one whole from a local store and checks it against `schema.json`;
//! `write_embedding` (the [`crate::StoreWriter`]'s) writes one as format.md §6 says:
//! chunks of `C` (cells) or `G` (genes) rows and every component, `[bytes, zstd]`, not
//! sharded, all-zero chunks not stored.
//! Does not: read part of an embedding (they are small next to the matrix: `n x width`
//! values), or check what the values mean (`NaN` is a value, §6).

use std::path::Path;
use std::sync::Arc;

use kyto_core::{Axis, EmbeddingSchema, FloatDtype};
use kyto_error::KytoError;
use zarrs::array::codec::{BytesCodec, ZstdCodec};
use zarrs::array::{Array, ArrayBuilder, DataType, FillValue};
use zarrs::filesystem::FilesystemStore;

use crate::layer_open::read_all;
use crate::store_schema::read_atlas_schema;
use crate::zarr_arrays::{StoreArray, store_chunk, store_io, store_metadata};

/// Values of an embedding or of a graph's edges, in their stored dtype.
#[derive(Debug, Clone, PartialEq)]
pub enum FloatValues {
    /// `float32` values.
    F32(Vec<f32>),
    /// `float64` values.
    F64(Vec<f64>),
}

impl FloatValues {
    /// The number of values.
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    /// Whether there are no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The stored dtype.
    pub fn dtype(&self) -> FloatDtype {
        match self {
            Self::F32(_) => FloatDtype::Float32,
            Self::F64(_) => FloatDtype::Float64,
        }
    }

    /// The values at `positions`, in that order, in the same dtype.
    pub fn take(&self, positions: impl Iterator<Item = usize>) -> Self {
        match self {
            Self::F32(values) => Self::F32(positions.map(|at| values[at]).collect()),
            Self::F64(values) => Self::F64(positions.map(|at| values[at]).collect()),
        }
    }

    /// The values as `float32` (`float64` rounded to nearest).
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Self::F32(values) => values.clone(),
            #[allow(clippy::cast_possible_truncation)]
            Self::F64(values) => values.iter().map(|&value| value as f32).collect(),
        }
    }
}

/// One embedding: row `i` is cell (or gene) `i` of its store, `width` components a row.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    /// The embedding name (format.md §2.1).
    pub name: String,
    /// The axis its rows follow.
    pub axis: Axis,
    /// Components per row, at least 1.
    pub width: u64,
    /// Row-major values, `rows x width`.
    pub values: FloatValues,
}

impl Embedding {
    /// The number of rows.
    pub fn n_rows(&self) -> u64 {
        (self.values.len() as u64)
            .checked_div(self.width)
            .unwrap_or(0)
    }

    /// Its entry of `schema.json` (format.md §8).
    pub fn schema(&self) -> EmbeddingSchema {
        EmbeddingSchema {
            name: self.name.clone(),
            axis: self.axis,
            width: self.width,
            dtype: self.values.dtype(),
        }
    }
}

/// Read embedding `name` of the local store at `store` whole, checked against its
/// `schema.json` entry (axis, width, dtype, one row per cell or gene).
///
/// # Errors
/// - `KY-E109` if the store's schema lists no embedding `name`.
/// - `KY-E126` if the array breaks format.md §6 or disagrees with `schema.json`.
/// - `KY-E110` or `KY-E122` if the store or its schema cannot be read.
pub fn read_embedding(store: &Path, name: &str) -> Result<Embedding, KytoError> {
    let store_text = store.display().to_string();
    let schema = read_atlas_schema(store)?;
    let Some(entry) = schema
        .embeddings
        .iter()
        .find(|embedding| embedding.name == name)
    else {
        return Err(KytoError::MissingMember {
            kind: "embedding".to_owned(),
            name: name.to_owned(),
            store: store_text,
            suggestions: kyto_core::suggest(
                name,
                schema
                    .embeddings
                    .iter()
                    .map(|embedding| embedding.name.as_str()),
            )
            .into(),
        });
    };
    let corrupt = |problem: String| KytoError::CorruptMember {
        kind: "embedding".to_owned(),
        name: name.to_owned(),
        problem,
    };
    let storage = Arc::new(
        FilesystemStore::new(store).map_err(|error| store_io("open store", &store_text, error))?,
    );
    let path = format!("/embeddings/{name}");
    let array = Array::open(storage, &path)
        .map_err(|error| corrupt(format!("cannot open the array: {error}")))?;
    let axis = array
        .attributes()
        .get("kyto")
        .and_then(|kyto| kyto.get("axis"))
        .and_then(|axis| axis.as_str());
    if axis != Some(axis_name(entry.axis)) {
        return Err(corrupt(format!(
            "attributes.kyto.axis is {axis:?}, but schema.json gives \"{}\"",
            axis_name(entry.axis)
        )));
    }
    let rows = match entry.axis {
        Axis::Cells => schema.n_cells,
        Axis::Genes => schema.n_genes,
    };
    if array.shape() != [rows, entry.width] {
        return Err(corrupt(format!(
            "the array has shape {:?}, but schema.json gives [{rows}, {}]",
            array.shape(),
            entry.width
        )));
    }
    let stored = array.data_type().to_string();
    let values = match entry.dtype {
        FloatDtype::Float32 if stored == DataType::Float32.to_string() => {
            FloatValues::F32(read_all(&array)?)
        }
        FloatDtype::Float64 if stored == DataType::Float64.to_string() => {
            FloatValues::F64(read_all(&array)?)
        }
        _ => {
            return Err(corrupt(format!(
                "the array has dtype {stored}, but schema.json gives {}",
                float_name(entry.dtype)
            )));
        }
    };
    Ok(Embedding {
        name: name.to_owned(),
        axis: entry.axis,
        width: entry.width,
        values,
    })
}

/// Write `embedding` to `/embeddings/<name>` of `storage`, `block_rows` rows a chunk
/// (the store's `C` or `G`), compressed at `zstd_level` (format.md §6). The caller has
/// checked its name and row count.
pub(crate) fn write_embedding(
    storage: &Arc<FilesystemStore>,
    embedding: &Embedding,
    block_rows: u64,
    zstd_level: i32,
) -> Result<(), KytoError> {
    let path = format!("/embeddings/{}", embedding.name);
    let rows = embedding.n_rows();
    let width = embedding.width;
    let (data_type, fill) = match embedding.values {
        FloatValues::F32(_) => (DataType::Float32, FillValue::from(0.0_f32)),
        FloatValues::F64(_) => (DataType::Float64, FillValue::from(0.0_f64)),
    };
    let mut attributes = serde_json::Map::new();
    attributes.insert(
        "kyto".to_owned(),
        serde_json::json!({ "axis": axis_name(embedding.axis) }),
    );
    let row_name = match embedding.axis {
        Axis::Cells => "cell",
        Axis::Genes => "gene",
    };
    let array: StoreArray =
        ArrayBuilder::new(vec![rows, width], vec![block_rows, width], data_type, fill)
            .array_to_bytes_codec(Arc::new(BytesCodec::little()))
            .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(zstd_level, false))])
            .dimension_names(Some(vec![row_name, "component"]))
            .attributes(attributes)
            .build(storage.clone(), &path)
            .map_err(|error| store_io("create array", &path, error))?;
    let chunk_len = (block_rows * width) as usize;
    match &embedding.values {
        FloatValues::F32(values) => store_rows(&array, values, chunk_len, 0.0)?,
        FloatValues::F64(values) => store_rows(&array, values, chunk_len, 0.0)?,
    }
    store_metadata(&array)
}

/// Write row-major `values` chunk by chunk (`chunk_len` values each), the last padded
/// with `fill` to the full chunk shape (format.md §4.4).
fn store_rows<T: zarrs::array::Element + Copy>(
    array: &StoreArray,
    values: &[T],
    chunk_len: usize,
    fill: T,
) -> Result<(), KytoError> {
    for (chunk, slice) in values.chunks(chunk_len.max(1)).enumerate() {
        let mut padded;
        let elements = if slice.len() == chunk_len {
            slice
        } else {
            padded = slice.to_vec();
            padded.resize(chunk_len, fill);
            &padded[..]
        };
        store_chunk(array, &[chunk as u64, 0], elements)?;
    }
    Ok(())
}

/// The axis as format.md names it.
pub(crate) fn axis_name(axis: Axis) -> &'static str {
    match axis {
        Axis::Cells => "cells",
        Axis::Genes => "genes",
    }
}

/// The dtype as Zarr names it.
pub(crate) fn float_name(dtype: FloatDtype) -> &'static str {
    match dtype {
        FloatDtype::Float32 => "float32",
        FloatDtype::Float64 => "float64",
    }
}
